# Repository Guidelines

## Project Structure & Module Organization

This repository is a ROS 1 description package for the Handone robot. `urdf/Handone.urdf` is the primary robot model; `urdf/Handone.xml` is its MuJoCo/MJCF counterpart, and `urdf/convert.py` regenerates that file. Keep referenced geometry in `meshes/`, controller joint names in `config/`, and ROS entry points in `launch/`. `viewer.py` provides a minimal MuJoCo viewer. Package metadata and catkin installation rules live in `package.xml` and `CMakeLists.txt`. The empty `textures/` directory is reserved for future model assets.

## Build, Test, and Development Commands

Run catkin commands from the enclosing ROS workspace, with this repository placed under `src/`:

```bash
catkin_make
source devel/setup.bash
roslaunch Handone display.launch
roslaunch Handone gazebo.launch
```

`catkin_make` configures and installs the description package. The launch commands inspect the robot in RViz or spawn it in Gazebo. From this package directory, use `check_urdf urdf/Handone.urdf` for structural validation and `python viewer.py` for a MuJoCo smoke test. To regenerate MJCF, run `python convert.py` from `urdf/` so its relative input path resolves correctly.

## Coding Style & Naming Conventions

Use four spaces in Python and two spaces in XML/YAML; preserve the existing tab indentation in CMake loops. Keep Python imports grouped at the top and scripts small and single-purpose. Name ROS links and joints with lowercase snake case, such as `bigfinger1_joint`. Preserve the exact case of existing package and asset names (`Handone`, `base_link.STL`): URDF mesh paths can be case-sensitive. Update URDF references whenever an asset is renamed.

## Testing Guidelines

There is no automated test suite or coverage threshold. Every model change should pass `check_urdf`, load without errors in RViz, and receive a MuJoCo or Gazebo smoke test when it affects geometry, joints, inertial data, or collision data. Verify that all mesh paths resolve and move each edited joint through its configured limits.

## Commit & Pull Request Guidelines

Git history is not included in this checkout, so use short imperative commit subjects, for example `Fix thumb joint limits`. Keep generated XML and its source URDF in the same commit when both change. Pull requests should explain the model behavior changed, list validation performed, link relevant issues, and include RViz/Gazebo screenshots for visible geometry or pose changes. Avoid committing local logs, caches, or simulator-generated files.
