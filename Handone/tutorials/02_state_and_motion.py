"""Observe MuJoCo state and run a small, repeatable motion experiment.

Run from any directory:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/02_state_and_motion.py

This is intentionally headless. Numerical experiments should be repeatable and
fast; the interactive viewer is used only after the numbers are understood.
"""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
JOINT_NAMES = ("bigfinger1_joint", "bigfinger3_joint")
ACTUATOR_NAMES = ("finger1_motor", "finger3_motor")
LOOP_PAIRS = (
    ("bigfinger4_A", "bigfinger1_A"),
    ("bigfinger4_B", "bigfinger1_B"),
)


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MJCF object not found: {name}")
    return object_id


def loop_error_mm(model, data):
    errors = []
    for first, second in LOOP_PAIRS:
        first_id = named_id(model, mujoco.mjtObj.mjOBJ_SITE, first)
        second_id = named_id(model, mujoco.mjtObj.mjOBJ_SITE, second)
        errors.append(1000.0 * np.linalg.norm(data.site_xpos[first_id] - data.site_xpos[second_id]))
    return np.asarray(errors)


def joint_state(model, data):
    angles = []
    velocities = []
    for name in JOINT_NAMES:
        joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        angles.append(data.qpos[model.jnt_qposadr[joint_id]])
        velocities.append(data.qvel[model.jnt_dofadr[joint_id]])
    return np.asarray(angles), np.asarray(velocities)


def simulate_segment(model, data, target, duration):
    actuator_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ACTUATOR_NAMES
    ]
    data.ctrl[actuator_ids] = target
    start_time = data.time
    max_loop_error = 0.0
    max_speed = 0.0

    while data.time - start_time < duration:
        mujoco.mj_step(model, data)
        _, velocity = joint_state(model, data)
        max_speed = max(max_speed, float(np.max(np.abs(velocity))))
        max_loop_error = max(max_loop_error, float(np.max(loop_error_mm(model, data))))

    angle, velocity = joint_state(model, data)
    actuator_force = data.actuator_force[actuator_ids].copy()
    return angle, velocity, actuator_force, max_speed, max_loop_error


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("Experiment: target angles -> simulated state")
    print(f"timestep = {model.opt.timestep:g} s")
    print("Angles below are printed in degrees; MuJoCo stores them in radians.")

    experiments = (
        (np.deg2rad((0.0, 0.0)), 0.5),
        (np.deg2rad((10.0, 15.0)), 1.0),
        (np.deg2rad((25.0, 30.0)), 1.0),
        (np.deg2rad((5.0, 5.0)), 1.0),
    )

    for index, (target, duration) in enumerate(experiments, start=1):
        result = simulate_segment(model, data, target, duration)
        angle, velocity, force, max_speed, max_loop_error = result
        print(f"\nSegment {index}: {duration:.1f} s")
        print(f"  target [deg]       = {np.rad2deg(target).round(3)}")
        print(f"  actual [deg]       = {np.rad2deg(angle).round(3)}")
        print(f"  velocity [rad/s]   = {velocity.round(6)}")
        print(f"  actuator force [Nm]= {force.round(6)}")
        print(f"  peak speed [rad/s] = {max_speed:.6f}")
        print(f"  peak loop error [mm] = {max_loop_error:.6f}")

    finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    print(f"\nFinite final state: {finite}")
    if not finite:
        raise RuntimeError("Simulation became numerically unstable")


if __name__ == "__main__":
    main()
