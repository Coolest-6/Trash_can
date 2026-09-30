"""Inspect the Handone MJCF without changing its state.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/01_model_audit.py
"""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
LOOP_SITE_PAIRS = (
    ("bigfinger4_A", "bigfinger1_A"),
    ("bigfinger4_B", "bigfinger1_B"),
)


def object_name(model, object_type, object_id):
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"unnamed_{object_id}"


def site_distance_mm(model, data, first_name, second_name):
    first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, first_name)
    second_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, second_name)
    delta = data.site_xpos[first_id] - data.site_xpos[second_id]
    return 1000.0 * np.linalg.norm(delta)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Model: {MODEL_PATH}")
    print(f"MuJoCo: {mujoco.__version__}")
    print(
        "Size: "
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"njnt={model.njnt}, neq={model.neq}, ngeom={model.ngeom}"
    )
    print(f"Solver: timestep={model.opt.timestep:g} s, iterations={model.opt.iterations}")

    print("\nJoints")
    for joint_id in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        dof_id = model.jnt_dofadr[joint_id]
        limited = bool(model.jnt_limited[joint_id])
        limits = model.jnt_range[joint_id] if limited else (np.nan, np.nan)
        print(
            f"  {name}: range={limits[0]:.6g}..{limits[1]:.6g} rad, "
            f"damping={model.dof_damping[dof_id]:.6g}"
        )

    print("\nActuators")
    for actuator_id in range(model.nu):
        name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        control_range = model.actuator_ctrlrange[actuator_id]
        force_range = model.actuator_forcerange[actuator_id]
        print(
            f"  {name} -> {joint_name}: "
            f"ctrl={control_range[0]:.6g}..{control_range[1]:.6g} rad, "
            f"force={force_range[0]:.6g}..{force_range[1]:.6g} N*m"
        )

    print("\nClosed-loop error at qpos=0")
    for first_name, second_name in LOOP_SITE_PAIRS:
        distance = site_distance_mm(model, data, first_name, second_name)
        print(f"  {first_name} <-> {second_name}: {distance:.6f} mm")

    print(f"\nEquality constraints active: {[bool(value) for value in data.eq_active]}")
    print(f"Initial contacts: {data.ncon}")
    print(f"Initial qpos: {np.array2string(data.qpos, precision=6)}")
    print(f"Initial qvel: {np.array2string(data.qvel, precision=6)}")


if __name__ == "__main__":
    main()
