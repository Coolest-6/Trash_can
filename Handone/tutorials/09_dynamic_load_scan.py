"""Compare dynamic loads for the same motion at different durations.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/09_dynamic_load_scan.py

Output:
    tutorials/results/09_dynamic_load_scan.csv
"""

import csv
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
OUTPUT_PATH = ROOT / "tutorials" / "results" / "09_dynamic_load_scan.csv"

ACTUATOR_NAMES = ("finger1_motor", "finger3_motor")
JOINT_NAMES = (
    "bigfinger1_joint",
    "bigfinger2_joint",
    "bigfinger3_joint",
    "bigfinger4_joint",
)
LOOP_PAIRS = (
    ("bigfinger4_A", "bigfinger1_A"),
    ("bigfinger4_B", "bigfinger1_B"),
)

TARGET_COMMAND = np.asarray((0.70, 0.70))
MOVE_DURATIONS = (0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20)
SETTLE_DURATION = 0.40
PRE_SETTLE_DURATION = 0.25
VELOCITY_LIMITS = np.asarray((10.0, 20.0, 20.0, 20.0))


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MJCF object not found: {name}")
    return object_id


def quintic_progress(time_value, duration):
    ratio = np.clip(time_value / duration, 0.0, 1.0)
    return 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5


def run_motion(model, move_duration):
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

    data.ctrl[actuator_ids] = 0.0
    while data.time < PRE_SETTLE_DURATION:
        mujoco.mj_step(model, data)

    peak_velocity = np.zeros(4)
    peak_acceleration = np.zeros(4)
    peak_force = np.zeros(2)
    peak_power = np.zeros(2)
    positive_work = np.zeros(2)
    absolute_work = np.zeros(2)
    saturation_steps = np.zeros(2, dtype=int)
    peak_loop_error = 0.0
    maximum_contacts = 0
    previous_velocity = data.qvel[dof_addresses].copy()
    total_steps = 0

    experiment_start = data.time
    end_time = move_duration + SETTLE_DURATION
    while data.time - experiment_start < end_time:
        elapsed = data.time - experiment_start
        if elapsed < move_duration:
            progress = quintic_progress(elapsed, move_duration)
            data.ctrl[actuator_ids] = progress * TARGET_COMMAND
        else:
            data.ctrl[actuator_ids] = TARGET_COMMAND

        mujoco.mj_step(model, data)
        velocity = data.qvel[dof_addresses].copy()
        acceleration = (
            velocity - previous_velocity
        ) / model.opt.timestep
        previous_velocity = velocity
        force = data.actuator_force[actuator_ids].copy()
        actuator_velocity = data.actuator_velocity[
            actuator_ids
        ].copy()
        power = force * actuator_velocity

        peak_velocity = np.maximum(peak_velocity, np.abs(velocity))
        peak_acceleration = np.maximum(
            peak_acceleration, np.abs(acceleration)
        )
        peak_force = np.maximum(peak_force, np.abs(force))
        peak_power = np.maximum(peak_power, np.abs(power))
        positive_work += np.maximum(power, 0.0) * model.opt.timestep
        absolute_work += np.abs(power) * model.opt.timestep

        force_ranges = model.actuator_forcerange[actuator_ids]
        force_limits = np.maximum(
            np.abs(force_ranges[:, 0]), np.abs(force_ranges[:, 1])
        )
        saturation_steps += np.abs(force) >= 0.999 * force_limits

        for first, second in site_pairs:
            peak_loop_error = max(
                peak_loop_error,
                1000.0
                * float(
                    np.linalg.norm(
                        data.site_xpos[first] - data.site_xpos[second]
                    )
                ),
            )
        maximum_contacts = max(maximum_contacts, data.ncon)
        total_steps += 1

    active_qpos = data.qpos[
        [
            qpos_addresses[0],
            qpos_addresses[2],
        ]
    ]
    final_error = TARGET_COMMAND - active_qpos
    finite = bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
    )
    velocity_ratio = peak_velocity / VELOCITY_LIMITS
    force_ranges = model.actuator_forcerange[actuator_ids]
    force_limits = np.maximum(
        np.abs(force_ranges[:, 0]), np.abs(force_ranges[:, 1])
    )
    force_ratio = peak_force / force_limits
    acceptable = bool(
        finite
        and maximum_contacts == 0
        and peak_loop_error <= 0.10
        and np.max(np.abs(np.rad2deg(final_error))) <= 0.10
        and np.max(velocity_ratio) <= 1.0
        and np.max(force_ratio) <= 0.90
    )

    return {
        "move_duration_s": move_duration,
        "theoretical_command_peak_velocity_rad_s": 1.875
        * float(np.max(TARGET_COMMAND))
        / move_duration,
        "theoretical_command_peak_acceleration_rad_s2": (
            10.0
            * np.sqrt(3.0)
            / 3.0
            * float(np.max(TARGET_COMMAND))
            / move_duration**2
        ),
        "peak_qvel_q1_rad_s": peak_velocity[0],
        "peak_qvel_q2_rad_s": peak_velocity[1],
        "peak_qvel_q3_rad_s": peak_velocity[2],
        "peak_qvel_q4_rad_s": peak_velocity[3],
        "peak_qacc_q1_rad_s2": peak_acceleration[0],
        "peak_qacc_q2_rad_s2": peak_acceleration[1],
        "peak_qacc_q3_rad_s2": peak_acceleration[2],
        "peak_qacc_q4_rad_s2": peak_acceleration[3],
        "peak_force_finger1_nm": peak_force[0],
        "peak_force_finger3_nm": peak_force[1],
        "peak_power_finger1_w": peak_power[0],
        "peak_power_finger3_w": peak_power[1],
        "positive_work_finger1_j": positive_work[0],
        "positive_work_finger3_j": positive_work[1],
        "absolute_work_finger1_j": absolute_work[0],
        "absolute_work_finger3_j": absolute_work[1],
        "saturation_fraction_finger1": saturation_steps[0]
        / total_steps,
        "saturation_fraction_finger3": saturation_steps[1]
        / total_steps,
        "peak_loop_error_mm": peak_loop_error,
        "maximum_contacts": maximum_contacts,
        "final_error_finger1_deg": np.rad2deg(final_error[0]),
        "final_error_finger3_deg": np.rad2deg(final_error[1]),
        "maximum_velocity_limit_ratio": float(
            np.max(velocity_ratio)
        ),
        "maximum_force_limit_ratio": float(np.max(force_ratio)),
        "finite": finite,
        "acceptable": acceptable,
    }


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    results = [
        run_motion(model, duration) for duration in MOVE_DURATIONS
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("Dynamic load scan")
    print(
        f"  target motors = {TARGET_COMMAND} rad "
        f"({np.rad2deg(TARGET_COMMAND).round(3)} deg)"
    )
    print(
        "  acceptance: force < 90% limit, velocity <= URDF limit, "
        "loop error <= 0.1 mm, final error <= 0.1 deg"
    )
    print(
        "\n  time[s]  max_vel  max_acc    F1[Nm]  F3[Nm]  "
        "Pmax[W]  loop[mm]  sat1[%]  pass"
    )
    for result in results:
        maximum_velocity = max(
            result[f"peak_qvel_q{index}_rad_s"]
            for index in range(1, 5)
        )
        maximum_acceleration = max(
            result[f"peak_qacc_q{index}_rad_s2"]
            for index in range(1, 5)
        )
        maximum_power = max(
            result["peak_power_finger1_w"],
            result["peak_power_finger3_w"],
        )
        print(
            f"  {result['move_duration_s']:7.2f}  "
            f"{maximum_velocity:7.3f}  "
            f"{maximum_acceleration:8.1f}  "
            f"{result['peak_force_finger1_nm']:7.4f}  "
            f"{result['peak_force_finger3_nm']:7.4f}  "
            f"{maximum_power:7.4f}  "
            f"{result['peak_loop_error_mm']:8.4f}  "
            f"{100.0 * result['saturation_fraction_finger1']:7.3f}  "
            f"{result['acceptable']}"
        )

    acceptable = [
        result for result in results if result["acceptable"]
    ]
    if acceptable:
        fastest = min(
            acceptable,
            key=lambda result: result["move_duration_s"],
        )
        print(
            f"\n  fastest tested acceptable move = "
            f"{fastest['move_duration_s']:.2f} s"
        )
    else:
        print("\n  no tested duration met all acceptance criteria")
    print(f"  CSV written to: {OUTPUT_PATH}")

    if not all(result["finite"] for result in results):
        raise RuntimeError("A dynamic test produced NaN or Inf")


if __name__ == "__main__":
    main()
