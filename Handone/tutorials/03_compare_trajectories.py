"""Compare an angle step with a smooth quintic trajectory.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/03_compare_trajectories.py

The experiment writes a CSV to tutorials/results/03_trajectory_comparison.csv.
"""

import csv
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
OUTPUT_PATH = ROOT / "tutorials" / "results" / "03_trajectory_comparison.csv"
JOINT_NAMES = ("bigfinger1_joint", "bigfinger3_joint")
ACTUATOR_NAMES = ("finger1_motor", "finger3_motor")
LOOP_PAIRS = (
    ("bigfinger4_A", "bigfinger1_A"),
    ("bigfinger4_B", "bigfinger1_B"),
)
TARGET = np.deg2rad((25.0, 30.0))
TRAJECTORY_DURATION = 0.5
EXPERIMENT_DURATION = 1.5


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MJCF object not found: {name}")
    return object_id


def quintic_progress(time_value, duration):
    """Return 0..1 progress with zero velocity/acceleration at both ends."""
    normalized_time = np.clip(time_value / duration, 0.0, 1.0)
    return (
        10.0 * normalized_time**3
        - 15.0 * normalized_time**4
        + 6.0 * normalized_time**5
    )


def run_experiment(model, mode):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    actuator_ids = np.asarray(
        [
            named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
    )
    joint_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ]
    qpos_addresses = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    )
    dof_addresses = np.asarray(
        [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
    )
    site_pairs = [
        (
            named_id(model, mujoco.mjtObj.mjOBJ_SITE, first),
            named_id(model, mujoco.mjtObj.mjOBJ_SITE, second),
        )
        for first, second in LOOP_PAIRS
    ]

    rows = []
    previous_velocity = data.qvel[dof_addresses].copy()
    step_count = round(EXPERIMENT_DURATION / model.opt.timestep)

    for _ in range(step_count + 1):
        if mode == "step":
            progress = 1.0
        elif mode == "quintic":
            progress = quintic_progress(data.time, TRAJECTORY_DURATION)
        else:
            raise ValueError(f"Unknown trajectory mode: {mode}")

        command = progress * TARGET
        data.ctrl[actuator_ids] = command
        mujoco.mj_step(model, data)

        angle = data.qpos[qpos_addresses].copy()
        velocity = data.qvel[dof_addresses].copy()
        acceleration = (velocity - previous_velocity) / model.opt.timestep
        previous_velocity = velocity
        loop_error = np.asarray(
            [
                1000.0
                * np.linalg.norm(
                    data.site_xpos[first] - data.site_xpos[second]
                )
                for first, second in site_pairs
            ]
        )

        rows.append(
            {
                "mode": mode,
                "time_s": data.time,
                "target_finger1_rad": command[0],
                "target_finger3_rad": command[1],
                "q_finger1_rad": angle[0],
                "q_finger3_rad": angle[1],
                "qvel_finger1_rad_s": velocity[0],
                "qvel_finger3_rad_s": velocity[1],
                "qacc_finger1_rad_s2": acceleration[0],
                "qacc_finger3_rad_s2": acceleration[1],
                "force_finger1_nm": data.actuator_force[actuator_ids[0]],
                "force_finger3_nm": data.actuator_force[actuator_ids[1]],
                "loop_A_error_mm": loop_error[0],
                "loop_B_error_mm": loop_error[1],
                "contact_count": data.ncon,
            }
        )

    return rows


def summarize(rows):
    target = np.asarray(
        [
            [row["target_finger1_rad"], row["target_finger3_rad"]]
            for row in rows
        ]
    )
    angle = np.asarray(
        [[row["q_finger1_rad"], row["q_finger3_rad"]] for row in rows]
    )
    velocity = np.asarray(
        [
            [row["qvel_finger1_rad_s"], row["qvel_finger3_rad_s"]]
            for row in rows
        ]
    )
    acceleration = np.asarray(
        [
            [row["qacc_finger1_rad_s2"], row["qacc_finger3_rad_s2"]]
            for row in rows
        ]
    )
    force = np.asarray(
        [[row["force_finger1_nm"], row["force_finger3_nm"]] for row in rows]
    )
    loop_error = np.asarray(
        [[row["loop_A_error_mm"], row["loop_B_error_mm"]] for row in rows]
    )

    return {
        "peak_speed": float(np.max(np.abs(velocity))),
        "peak_acceleration": float(np.max(np.abs(acceleration))),
        "peak_force": float(np.max(np.abs(force))),
        "peak_loop_error": float(np.max(loop_error)),
        "final_error_deg": np.rad2deg(TARGET - angle[-1]),
        "maximum_tracking_error_deg": float(
            np.max(np.abs(np.rad2deg(target - angle)))
        ),
        "maximum_contacts": max(row["contact_count"] for row in rows),
        "finite": bool(
            np.isfinite(angle).all()
            and np.isfinite(velocity).all()
            and np.isfinite(force).all()
        ),
    }


def print_summary(name, summary):
    print(f"\n{name}")
    print(f"  peak speed          = {summary['peak_speed']:.6f} rad/s")
    print(
        f"  peak acceleration   = "
        f"{summary['peak_acceleration']:.6f} rad/s^2"
    )
    print(f"  peak actuator force = {summary['peak_force']:.6f} N*m")
    print(f"  peak loop error     = {summary['peak_loop_error']:.6f} mm")
    print(f"  final error         = {summary['final_error_deg'].round(6)} deg")
    print(
        "  max tracking error = "
        f"{summary['maximum_tracking_error_deg']:.6f} deg"
    )
    print(f"  maximum contacts    = {summary['maximum_contacts']}")
    print(f"  finite state        = {summary['finite']}")


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    step_rows = run_experiment(model, "step")
    quintic_rows = run_experiment(model, "quintic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(step_rows[0])
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(step_rows)
        writer.writerows(quintic_rows)

    step_summary = summarize(step_rows)
    quintic_summary = summarize(quintic_rows)

    print("Trajectory comparison")
    print(f"  target              = {np.rad2deg(TARGET)} deg")
    print(f"  quintic move time   = {TRAJECTORY_DURATION:.3f} s")
    print(f"  total test time     = {EXPERIMENT_DURATION:.3f} s")
    print_summary("Step command", step_summary)
    print_summary("Quintic command", quintic_summary)

    print("\nQuintic / step ratios (smaller is better)")
    for key, label in (
        ("peak_speed", "peak speed"),
        ("peak_acceleration", "peak acceleration"),
        ("peak_force", "peak actuator force"),
        ("peak_loop_error", "peak loop error"),
    ):
        ratio = quintic_summary[key] / step_summary[key]
        print(f"  {label:20s} = {ratio:.4f}")

    print(f"\nCSV written to: {OUTPUT_PATH}")
    if not step_summary["finite"] or not quintic_summary["finite"]:
        raise RuntimeError("A trajectory produced NaN or Inf")


if __name__ == "__main__":
    main()
