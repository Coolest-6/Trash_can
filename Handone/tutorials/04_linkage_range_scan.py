"""Scan the closed linkage through its software-controlled range.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/04_linkage_range_scan.py

The scan moves finger3_motor slowly from minimum to maximum and back while
finger1_motor remains at zero. It records passive-joint motion, closed-loop
error, actuator force, contacts, and the bigfinger4 body-origin trajectory.
"""

import csv
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
OUTPUT_PATH = ROOT / "tutorials" / "results" / "04_linkage_range_scan.csv"

SAMPLE_COUNT = 41
MOVE_DURATION = 0.20
SETTLE_DURATION = 0.15
FINGER1_TARGET = 0.0

JOINT_NAMES = (
    "bigfinger1_joint",
    "bigfinger2_joint",
    "bigfinger3_joint",
    "bigfinger4_joint",
)
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


def quintic_progress(time_value, duration):
    ratio = np.clip(time_value / duration, 0.0, 1.0)
    return 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5


def contact_body_pair(model, contact):
    first_body_id = int(model.geom_bodyid[contact.geom1])
    second_body_id = int(model.geom_bodyid[contact.geom2])
    first_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, first_body_id
    )
    second_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, second_body_id
    )
    return first_name or "world", second_name or "world"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    actuator_ids = {
        name: named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ACTUATOR_NAMES
    }
    joint_ids = {
        name: named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    }
    qpos_addresses = {
        name: int(model.jnt_qposadr[joint_id])
        for name, joint_id in joint_ids.items()
    }
    dof_addresses = {
        name: int(model.jnt_dofadr[joint_id])
        for name, joint_id in joint_ids.items()
    }
    loop_site_ids = [
        (
            named_id(model, mujoco.mjtObj.mjOBJ_SITE, first),
            named_id(model, mujoco.mjtObj.mjOBJ_SITE, second),
        )
        for first, second in LOOP_PAIRS
    ]
    link4_body_id = named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "bigfinger4_Link"
    )

    finger3_id = actuator_ids["finger3_motor"]
    lower, upper = model.actuator_ctrlrange[finger3_id]
    upward_targets = np.linspace(lower, upper, SAMPLE_COUNT)
    scan_targets = [
        ("up", index, value)
        for index, value in enumerate(upward_targets)
    ] + [
        ("down", index, upward_targets[index])
        for index in range(SAMPLE_COUNT - 2, -1, -1)
    ]

    rows = []
    previous_target = lower

    for direction, sample_index, target in scan_targets:
        segment_start = data.time
        segment_end = segment_start + MOVE_DURATION
        peak_force = 0.0
        peak_loop_error = 0.0
        maximum_contacts = 0
        minimum_contact_distance = np.nan
        contact_pairs = set()

        while data.time < segment_end:
            progress = quintic_progress(
                data.time - segment_start, MOVE_DURATION
            )
            command = previous_target + progress * (
                target - previous_target
            )
            data.ctrl[actuator_ids["finger1_motor"]] = FINGER1_TARGET
            data.ctrl[finger3_id] = command
            mujoco.mj_step(model, data)

            peak_force = max(
                peak_force,
                float(np.max(np.abs(data.actuator_force))),
            )
            peak_loop_error = max(
                peak_loop_error,
                *(
                    1000.0
                    * np.linalg.norm(
                        data.site_xpos[first] - data.site_xpos[second]
                    )
                    for first, second in loop_site_ids
                ),
            )
            maximum_contacts = max(maximum_contacts, data.ncon)
            for contact in data.contact[: data.ncon]:
                contact_pairs.add(contact_body_pair(model, contact))
                distance_mm = 1000.0 * float(contact.dist)
                if np.isnan(minimum_contact_distance):
                    minimum_contact_distance = distance_mm
                else:
                    minimum_contact_distance = min(
                        minimum_contact_distance, distance_mm
                    )

        data.ctrl[actuator_ids["finger1_motor"]] = FINGER1_TARGET
        data.ctrl[finger3_id] = target
        settle_end = data.time + SETTLE_DURATION
        while data.time < settle_end:
            mujoco.mj_step(model, data)
            peak_force = max(
                peak_force,
                float(np.max(np.abs(data.actuator_force))),
            )
            peak_loop_error = max(
                peak_loop_error,
                *(
                    1000.0
                    * np.linalg.norm(
                        data.site_xpos[first] - data.site_xpos[second]
                    )
                    for first, second in loop_site_ids
                ),
            )
            maximum_contacts = max(maximum_contacts, data.ncon)
            for contact in data.contact[: data.ncon]:
                contact_pairs.add(contact_body_pair(model, contact))
                distance_mm = 1000.0 * float(contact.dist)
                if np.isnan(minimum_contact_distance):
                    minimum_contact_distance = distance_mm
                else:
                    minimum_contact_distance = min(
                        minimum_contact_distance, distance_mm
                    )

        qpos = {
            name: float(data.qpos[address])
            for name, address in qpos_addresses.items()
        }
        qvel = {
            name: float(data.qvel[dof_addresses[name]])
            for name in JOINT_NAMES
        }
        loop_error = [
            1000.0
            * np.linalg.norm(
                data.site_xpos[first] - data.site_xpos[second]
            )
            for first, second in loop_site_ids
        ]
        pair_text = ";".join(
            f"{first}<->{second}"
            for first, second in sorted(contact_pairs)
        )

        rows.append(
            {
                "direction": direction,
                "sample_index": sample_index,
                "command_finger3_rad": target,
                "command_finger3_deg": np.rad2deg(target),
                "q1_rad": qpos["bigfinger1_joint"],
                "q2_rad": qpos["bigfinger2_joint"],
                "q3_rad": qpos["bigfinger3_joint"],
                "q4_rad": qpos["bigfinger4_joint"],
                "q1_deg": np.rad2deg(qpos["bigfinger1_joint"]),
                "q2_deg": np.rad2deg(qpos["bigfinger2_joint"]),
                "q3_deg": np.rad2deg(qpos["bigfinger3_joint"]),
                "q4_deg": np.rad2deg(qpos["bigfinger4_joint"]),
                "maximum_abs_qvel_rad_s": max(
                    abs(value) for value in qvel.values()
                ),
                "finger3_force_nm": data.actuator_force[finger3_id],
                "segment_peak_actuator_force_nm": peak_force,
                "loop_A_error_mm": loop_error[0],
                "loop_B_error_mm": loop_error[1],
                "segment_peak_loop_error_mm": peak_loop_error,
                "maximum_contacts": maximum_contacts,
                "minimum_contact_distance_mm": minimum_contact_distance,
                "contact_body_pairs": pair_text,
                "link4_origin_x_m": data.xpos[link4_body_id, 0],
                "link4_origin_y_m": data.xpos[link4_body_id, 1],
                "link4_origin_z_m": data.xpos[link4_body_id, 2],
            }
        )
        previous_target = target

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    upward = [row for row in rows if row["direction"] == "up"]
    downward_by_index = {
        row["sample_index"]: row
        for row in rows
        if row["direction"] == "down"
    }
    hysteresis = []
    for row in upward[:-1]:
        reverse = downward_by_index[row["sample_index"]]
        hysteresis.append(
            max(
                abs(row["q2_rad"] - reverse["q2_rad"]),
                abs(row["q3_rad"] - reverse["q3_rad"]),
                abs(row["q4_rad"] - reverse["q4_rad"]),
            )
        )

    command = np.asarray(
        [row["command_finger3_rad"] for row in upward]
    )
    q2 = np.asarray([row["q2_rad"] for row in upward])
    q4 = np.asarray([row["q4_rad"] for row in upward])
    dq2_dcommand = np.gradient(q2, command)
    dq4_dcommand = np.gradient(q4, command)
    q2_peak_index = int(np.argmax(np.abs(dq2_dcommand)))
    q4_peak_index = int(np.argmax(np.abs(dq4_dcommand)))

    finite = all(
        np.isfinite(
            [
                row["q1_rad"],
                row["q2_rad"],
                row["q3_rad"],
                row["q4_rad"],
                row["loop_A_error_mm"],
                row["loop_B_error_mm"],
            ]
        ).all()
        for row in rows
    )
    contact_rows = [row for row in rows if row["maximum_contacts"] > 0]
    max_loop_error = max(
        row["segment_peak_loop_error_mm"] for row in rows
    )
    max_force = max(
        row["segment_peak_actuator_force_nm"] for row in rows
    )

    print("Closed-linkage range scan")
    print(
        f"  finger3 command range = {lower:.6f}..{upper:.6f} rad "
        f"({np.rad2deg(lower):.3f}..{np.rad2deg(upper):.3f} deg)"
    )
    print(f"  samples per direction = {SAMPLE_COUNT}")
    print(
        "  passive q2 range      = "
        f"{np.rad2deg(q2.min()):.3f}..{np.rad2deg(q2.max()):.3f} deg"
    )
    print(
        "  passive q4 range      = "
        f"{np.rad2deg(q4.min()):.3f}..{np.rad2deg(q4.max()):.3f} deg"
    )
    print(
        f"  maximum loop error    = {max_loop_error:.6f} mm"
    )
    print(f"  maximum actuator force= {max_force:.6f} N*m")
    print(
        "  maximum hysteresis    = "
        f"{np.rad2deg(max(hysteresis, default=0.0)):.6f} deg"
    )
    print(
        "  largest |dq2/du|      = "
        f"{dq2_dcommand[q2_peak_index]:.6f} at "
        f"{np.rad2deg(command[q2_peak_index]):.3f} deg"
    )
    print(
        "  largest |dq4/du|      = "
        f"{dq4_dcommand[q4_peak_index]:.6f} at "
        f"{np.rad2deg(command[q4_peak_index]):.3f} deg"
    )
    print(f"  samples with contact  = {len(contact_rows)}")
    if contact_rows:
        first_contact = contact_rows[0]
        print(
            "  first contact         = "
            f"{first_contact['command_finger3_deg']:.3f} deg, "
            f"{first_contact['contact_body_pairs']}, "
            f"minimum distance "
            f"{first_contact['minimum_contact_distance_mm']:.6f} mm"
        )
    print(f"  finite state          = {finite}")
    print(f"  CSV written to        = {OUTPUT_PATH}")

    if not finite:
        raise RuntimeError("The range scan produced NaN or Inf")


if __name__ == "__main__":
    main()
