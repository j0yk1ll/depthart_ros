# depthart_ros

ROS 2 integration for [DepthART](https://github.com/xuefeng-cvr/DepthART) metric monocular depth inference.

This repository contains the ROS package only. The upstream DepthART source code,
checkpoints, PyTorch/CUDA runtime, and optional selective-scan extension are **not**
vendored or installed by this package.

## ROS interface

Default inputs:

- `/camera/image_rect` (`sensor_msgs/msg/Image`)
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`)

Default output:

- `/depthart/depth` (`sensor_msgs/msg/Image`)
- encoding: `32FC1`
- units: meters
- input image header copied exactly to the depth image

For rectified input, the default configuration reads intrinsics from
`CameraInfo.P`.

```text
/camera/image_rect + /camera/camera_info
                 |
                 v
             depthart
                 |
                 v
          /depthart/depth
             32FC1 m
```

## Latest-frame behavior

The subscription callback keeps only the newest pending image. DepthART inference
runs in a worker thread. If inference is slower than the camera stream, stale
pending frames are replaced rather than queued. This is intended for real-time
robotics pipelines where low latency matters more than processing every frame.

## DepthART dependency

By default the node looks for an existing DepthART checkout at:

```text
/opt/DepthART
```

and the default checkpoint at:

```text
/opt/DepthART/checkpoints/metric/depthart_metric_indoor_s_448.pth
```

Both are configurable in `config/depthart.yaml` or through ROS parameters.
`/opt/DepthART` is only a convenient container convention; it is not required.

The runtime environment must already provide DepthART's Python dependencies,
including PyTorch. The optional optimized selective-scan extension is loaded if
available and enabled.

## Build from source

Clone this repository into a ROS 2 workspace:

```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone <repository-url> depthart_ros

cd ~/ros_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select depthart_ros
source install/setup.bash
```

## Run

```bash
ros2 launch depthart_ros depthart.launch.py
```

Or run the executable directly:

```bash
ros2 run depthart_ros depthart_node
```

Verify the output:

```bash
ros2 topic info /depthart/depth --verbose
ros2 topic echo /depthart/depth --field encoding --once
```

The expected encoding is `32FC1`.

## Configuration

See `config/depthart.yaml`. Important parameters are:

- `image_topic`
- `camera_info_topic`
- `depth_topic`
- `depthart_root`
- `checkpoint`
- `encoder`
- `domain`
- `device`
- `optimized_scan`
- `model_width`
- `model_height`
- `intrinsics_source`
- `stats_period_sec`

Default topics are deliberately generic and can be overridden by downstream
robot projects.

## Package naming

The ROS package and repository are named `depthart_ros` to make the relationship
to the upstream DepthART project explicit. This follows a common ROS wrapper
pattern used by packages such as `rtabmap_ros` and `apriltag_ros`. If released
through the ROS build farm for Jazzy, the Debian package would be named
`ros-jazzy-depthart-ros`.

## Licensing

The ROS integration code in this repository is licensed under Apache-2.0.
DepthART itself is a separate upstream project and remains subject to its own
license and terms. This repository does not redistribute DepthART source code or
model checkpoints.

## Before a public release

Replace the placeholder maintainer email in `package.xml` and `setup.py` with
your real release contact before submitting the package to rosdistro/bloom.
