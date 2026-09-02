#!/usr/bin/env python3
"""Realtime ROS 2 wrapper for DepthART metric monocular depth."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


def sensor_qos_depth_one() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class DepthARTNode(Node):
    def __init__(self) -> None:
        super().__init__("depthart")

        self.declare_parameter("image_topic", "/camera/image_rect")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("depth_topic", "/depthart/depth")
        self.declare_parameter("depthart_root", "/opt/DepthART")
        self.declare_parameter(
            "checkpoint",
            "checkpoints/metric/depthart_metric_indoor_s_448.pth",
        )
        self.declare_parameter("encoder", "S")
        self.declare_parameter("domain", "indoor")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("optimized_scan", True)
        self.declare_parameter("model_width", 640)
        self.declare_parameter("model_height", 480)
        self.declare_parameter("intrinsics_source", "p")
        self.declare_parameter("stats_period_sec", 5.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.depthart_root = Path(
            str(self.get_parameter("depthart_root").value)
        ).expanduser().resolve()

        checkpoint = Path(str(self.get_parameter("checkpoint").value)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = self.depthart_root / checkpoint
        self.checkpoint = checkpoint.resolve()

        self.encoder = str(self.get_parameter("encoder").value)
        self.domain = str(self.get_parameter("domain").value)
        self.device_name = str(self.get_parameter("device").value).lower()
        self.optimized_scan_requested = bool(
            self.get_parameter("optimized_scan").value
        )
        self.model_width = int(self.get_parameter("model_width").value)
        self.model_height = int(self.get_parameter("model_height").value)
        self.intrinsics_source = str(
            self.get_parameter("intrinsics_source").value
        ).lower()
        self.stats_period_sec = float(self.get_parameter("stats_period_sec").value)

        if self.encoder not in {"S", "B", "L"}:
            raise ValueError("encoder must be S, B, or L")
        if self.domain not in {"indoor", "outdoor"}:
            raise ValueError("domain must be indoor or outdoor")
        if self.intrinsics_source not in {"p", "k"}:
            raise ValueError("intrinsics_source must be 'p' or 'k'")

        self.bridge = CvBridge()

        self._camera_lock = threading.Lock()
        self._camera_info: Optional[CameraInfo] = None

        self._frame_lock = threading.Lock()
        self._latest_image: Optional[Image] = None
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()

        self._received = 0
        self._processed = 0
        self._replaced = 0
        self._ema_pre_ms: Optional[float] = None
        self._ema_infer_ms: Optional[float] = None
        self._ema_total_ms: Optional[float] = None
        self._last_stats_time = time.perf_counter()

        self.get_logger().info(f"DepthART root: {self.depthart_root}")
        self.get_logger().info(f"Checkpoint: {self.checkpoint}")

        self._preprocess, self._make_K, self._load_model, tvimblock = (
            self._import_depthart()
        )

        self.device = self._select_device()
        self.get_logger().info(f"Inference device: {self.device}")
        self.get_logger().info("Loading DepthART model...")

        self.model = self._load_model(
            str(self.checkpoint), self.encoder, self.domain, self.device
        )
        self.model.eval()

        optimized = False
        if self.optimized_scan_requested and self.device == "cuda":
            optimized = self._enable_optimized_scan(tvimblock)
        elif self.optimized_scan_requested:
            self.get_logger().info("Optimized selective scan disabled on CPU")
        self.get_logger().info(
            "Selective scan: "
            + ("optimized CUDA extension" if optimized else "reference fallback")
        )

        qos = sensor_qos_depth_one()
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback, qos
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_callback, qos
        )
        self.depth_pub = self.create_publisher(Image, self.depth_topic, qos)

        self._worker = threading.Thread(
            target=self._inference_worker,
            name="depthart-inference",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            f"Ready: {self.image_topic} + {self.camera_info_topic} "
            f"-> {self.depth_topic} (32FC1 meters)"
        )

    def _import_depthart(self):
        metric_root = self.depthart_root / "metric"
        if not metric_root.is_dir():
            raise RuntimeError(
                f"DepthART metric directory not found: {metric_root}. "
                "Set parameter 'depthart_root' to the DepthART repository."
            )
        if not self.checkpoint.is_file():
            raise RuntimeError(
                f"DepthART checkpoint not found: {self.checkpoint}. "
                "Set parameter 'checkpoint' to the metric model checkpoint."
            )

        sys.path.insert(0, str(metric_root))
        try:
            from common import make_K, preprocess
            from model import load_model
            from network import tvimblock
        except Exception as exc:
            raise RuntimeError(
                "Could not import DepthART metric modules. Ensure DepthART and its "
                f"Python dependencies are installed. Original error: {exc}"
            ) from exc

        return preprocess, make_K, load_model, tvimblock

    def _select_device(self) -> str:
        if self.device_name == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but CUDA is unavailable")
            return "cuda"
        if self.device_name == "cpu":
            return "cpu"
        raise ValueError("device must be auto, cuda, or cpu")

    def _enable_optimized_scan(self, tvimblock) -> bool:
        extension_root = self.depthart_root / "deploy/shared/selective_scan"
        if not extension_root.is_dir():
            self.get_logger().warning(
                f"Selective-scan directory not found: {extension_root}"
            )
            return False

        sys.path.insert(0, str(extension_root))
        try:
            import depthart_selective_scan_cuda  # noqa: F401
            from depthart_selective_scan import install_depthart
        except ImportError as exc:
            self.get_logger().warning(
                f"Optimized selective scan unavailable; using fallback ({exc})"
            )
            return False

        install_depthart(tvimblock)
        return True

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        with self._camera_lock:
            self._camera_info = msg

    def _image_callback(self, msg: Image) -> None:
        with self._frame_lock:
            if self._latest_image is not None:
                self._replaced += 1
            self._latest_image = msg
            self._received += 1
        self._frame_ready.set()

    def _take_latest_image(self) -> Optional[Image]:
        with self._frame_lock:
            msg = self._latest_image
            self._latest_image = None
            return msg

    def _get_camera_info(self) -> Optional[CameraInfo]:
        with self._camera_lock:
            return self._camera_info

    def _intrinsics_for_image(
        self, info: CameraInfo, image_width: int, image_height: int
    ) -> np.ndarray:
        if self.intrinsics_source == "p":
            fx, fy = float(info.p[0]), float(info.p[5])
            cx, cy = float(info.p[2]), float(info.p[6])
        else:
            fx, fy = float(info.k[0]), float(info.k[4])
            cx, cy = float(info.k[2]), float(info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(
                f"CameraInfo.{self.intrinsics_source.upper()} is uncalibrated "
                f"(fx={fx}, fy={fy})"
            )
        if info.width <= 0 or info.height <= 0:
            raise RuntimeError("CameraInfo width/height must be positive")

        sx = image_width / float(info.width)
        sy = image_height / float(info.height)
        fx, cx = fx * sx, cx * sx
        fy, cy = fy * sy, cy * sy
        return self._make_K(fx, fy, cx, cy)

    @staticmethod
    def _smooth(old: Optional[float], new: float, alpha: float = 0.1) -> float:
        return new if old is None else (1.0 - alpha) * old + alpha * new

    def _inference_worker(self) -> None:
        try:
            with torch.inference_mode():
                while not self._stop_event.is_set():
                    self._frame_ready.wait(timeout=0.1)
                    self._frame_ready.clear()
                    if self._stop_event.is_set():
                        break

                    image_msg = self._take_latest_image()
                    if image_msg is None:
                        continue

                    info = self._get_camera_info()
                    if info is None:
                        self.get_logger().warning(
                            "Waiting for CameraInfo; dropping image.",
                            throttle_duration_sec=2.0,
                        )
                        continue

                    try:
                        self._process_image(image_msg, info)
                    except Exception as exc:
                        self.get_logger().error(
                            f"DepthART inference failed: {type(exc).__name__}: {exc}"
                        )
                        time.sleep(0.05)
        except Exception as exc:
            self.get_logger().fatal(
                f"DepthART worker terminated: {type(exc).__name__}: {exc}"
            )

    def _process_image(self, image_msg: Image, info: CameraInfo) -> None:
        total_t0 = time.perf_counter()

        # infer_stream.py received OpenCV BGR frames; preserve that tested convention.
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        K_stream = self._intrinsics_for_image(info, width, height)

        t0 = time.perf_counter()
        tensor, K_model = self._preprocess(
            frame, K_stream, self.model_width, self.model_height
        )
        pre_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred = self.model(tensor.to(self.device), K_model.to(self.device))[:, None]
        pred = F.interpolate(
            pred, (height, width), mode="bilinear", align_corners=True
        )[0, 0]
        depth = pred.float().cpu().numpy().astype(np.float32, copy=False)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header = image_msg.header
        self.depth_pub.publish(depth_msg)

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        self._processed += 1
        self._ema_pre_ms = self._smooth(self._ema_pre_ms, pre_ms)
        self._ema_infer_ms = self._smooth(self._ema_infer_ms, infer_ms)
        self._ema_total_ms = self._smooth(self._ema_total_ms, total_ms)

        now = time.perf_counter()
        if self.stats_period_sec > 0.0 and now - self._last_stats_time >= self.stats_period_sec:
            fps = 1000.0 / max(self._ema_total_ms or 1.0, 1e-6)
            self.get_logger().info(
                "DepthART "
                f"fps={fps:.1f} "
                f"pre={self._ema_pre_ms:.1f}ms "
                f"infer={self._ema_infer_ms:.1f}ms "
                f"total={self._ema_total_ms:.1f}ms "
                f"received={self._received} "
                f"processed={self._processed} "
                f"replaced={self._replaced}"
            )
            self._last_stats_time = now

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._frame_ready.set()
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DepthARTNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
