"""Scan the two-motor thumb workspace and evaluate each configuration.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/06_thumb_workspace_scan.py

Outputs:
    tutorials/results/06_thumb_workspace.csv
    tutorials/results/06_thumb_workspace_projections.svg
"""

import csv
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
RESULTS_DIR = ROOT / "tutorials" / "results"
CSV_PATH = RESULTS_DIR / "06_thumb_workspace.csv"
SVG_PATH = RESULTS_DIR / "06_thumb_workspace_projections.svg"

GRID_SIZE = 21
MOVE_DURATION = 0.15
SETTLE_DURATION = 0.20
TRACKING_TOLERANCE = np.deg2rad(0.5)
VELOCITY_TOLERANCE = 0.02
LOOP_ERROR_TOLERANCE_MM = 0.10

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


def loop_errors_mm(data, site_pairs):
    return np.asarray(
        [
            1000.0
            * np.linalg.norm(
                data.site_xpos[first] - data.site_xpos[second]
            )
            for first, second in site_pairs
        ]
    )


def advance_to_target(
    model,
    data,
    actuator_ids,
    site_pairs,
    start_command,
    target_command,
):
    move_start = data.time
    move_end = move_start + MOVE_DURATION
    peak_force = np.zeros(2)
    peak_loop_error = 0.0
    maximum_contacts = 0

    while data.time < move_end:
        progress = quintic_progress(data.time - move_start, MOVE_DURATION)
        data.ctrl[actuator_ids] = start_command + progress * (
            target_command - start_command
        )
        mujoco.mj_step(model, data)
        peak_force = np.maximum(
            peak_force, np.abs(data.actuator_force[actuator_ids])
        )
        peak_loop_error = max(
            peak_loop_error,
            float(np.max(loop_errors_mm(data, site_pairs))),
        )
        maximum_contacts = max(maximum_contacts, data.ncon)

    data.ctrl[actuator_ids] = target_command
    settle_end = data.time + SETTLE_DURATION
    while data.time < settle_end:
        mujoco.mj_step(model, data)
        peak_force = np.maximum(
            peak_force, np.abs(data.actuator_force[actuator_ids])
        )
        peak_loop_error = max(
            peak_loop_error,
            float(np.max(loop_errors_mm(data, site_pairs))),
        )
        maximum_contacts = max(maximum_contacts, data.ncon)

    return peak_force, peak_loop_error, maximum_contacts


def add_jacobian_metrics(rows, points, q1_targets, q3_targets):
    dp_dq1 = np.gradient(points, q1_targets, axis=0, edge_order=2)
    dp_dq3 = np.gradient(points, q3_targets, axis=1, edge_order=2)

    for row in rows:
        i = row["q1_index"]
        j = row["q3_index"]
        jacobian = np.column_stack((dp_dq1[i, j], dp_dq3[i, j]))
        singular_values = np.linalg.svd(
            jacobian, compute_uv=False
        )
        sigma_max, sigma_min = singular_values
        condition_number = (
            sigma_max / sigma_min if sigma_min > 1e-9 else math.inf
        )
        row["jacobian_sigma_max_m_rad"] = sigma_max
        row["jacobian_sigma_min_m_rad"] = sigma_min
        row["jacobian_condition"] = condition_number
        row["manipulability_m2_rad2"] = sigma_max * sigma_min


def svg_color(normalized_value):
    normalized_value = float(np.clip(normalized_value, 0.0, 1.0))
    red = round(20 + 210 * normalized_value)
    green = round(115 + 35 * (1.0 - normalized_value))
    blue = round(190 - 140 * normalized_value)
    return f"rgb({red},{green},{blue})"


def write_projection_svg(rows):
    projections = (
        ("XY", "tip_x_mm", "tip_y_mm", "X [mm]", "Y [mm]"),
        ("XZ", "tip_x_mm", "tip_z_mm", "X [mm]", "Z [mm]"),
        ("YZ", "tip_y_mm", "tip_z_mm", "Y [mm]", "Z [mm]"),
    )
    width = 1200
    height = 430
    panel_width = 380
    margin_left = 54
    plot_top = 54
    plot_width = 285
    plot_height = 290
    target_min = min(row["target_q3_rad"] for row in rows)
    target_max = max(row["target_q3_rad"] for row in rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="18" '
        'fill="#222">Thumb-tip workspace projections</text>',
    ]

    for panel_index, (title, x_key, y_key, x_label, y_label) in enumerate(
        projections
    ):
        panel_x = panel_index * panel_width + 20
        x_values = np.asarray([row[x_key] for row in rows])
        y_values = np.asarray([row[y_key] for row in rows])
        x_min, x_max = float(x_values.min()), float(x_values.max())
        y_min, y_max = float(y_values.min()), float(y_values.max())
        x_padding = max(0.05 * (x_max - x_min), 0.5)
        y_padding = max(0.05 * (y_max - y_min), 0.5)
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def map_x(value):
            return (
                panel_x
                + margin_left
                + (value - x_min) / (x_max - x_min) * plot_width
            )

        def map_y(value):
            return (
                plot_top
                + plot_height
                - (value - y_min) / (y_max - y_min) * plot_height
            )

        parts.extend(
            [
                f'<text x="{panel_x + margin_left}" y="48" '
                'font-family="sans-serif" font-size="15" fill="#222">'
                f"{title}</text>",
                f'<rect x="{panel_x + margin_left}" y="{plot_top}" '
                f'width="{plot_width}" height="{plot_height}" '
                'fill="none" stroke="#777" stroke-width="1"/>',
                f'<text x="{panel_x + margin_left + plot_width / 2}" '
                f'y="{plot_top + plot_height + 42}" text-anchor="middle" '
                'font-family="sans-serif" font-size="13" fill="#333">'
                f"{x_label}</text>",
                f'<text x="{panel_x + 14}" '
                f'y="{plot_top + plot_height / 2}" '
                f'transform="rotate(-90 {panel_x + 14} '
                f'{plot_top + plot_height / 2})" text-anchor="middle" '
                'font-family="sans-serif" font-size="13" fill="#333">'
                f"{y_label}</text>",
                f'<text x="{panel_x + margin_left}" '
                f'y="{plot_top + plot_height + 17}" '
                'font-family="sans-serif" font-size="11" fill="#555">'
                f"{x_min:.1f}</text>",
                f'<text x="{panel_x + margin_left + plot_width}" '
                f'y="{plot_top + plot_height + 17}" text-anchor="end" '
                'font-family="sans-serif" font-size="11" fill="#555">'
                f"{x_max:.1f}</text>",
                f'<text x="{panel_x + margin_left - 7}" '
                f'y="{plot_top + 4}" text-anchor="end" '
                'font-family="sans-serif" font-size="11" fill="#555">'
                f"{y_max:.1f}</text>",
                f'<text x="{panel_x + margin_left - 7}" '
                f'y="{plot_top + plot_height}" text-anchor="end" '
                'font-family="sans-serif" font-size="11" fill="#555">'
                f"{y_min:.1f}</text>",
            ]
        )

        for row in rows:
            normalized = (
                (row["target_q3_rad"] - target_min)
                / (target_max - target_min)
                if target_max > target_min
                else 0.0
            )
            color = svg_color(normalized)
            opacity = "0.78" if row["reliable"] else "0.22"
            parts.append(
                f'<circle cx="{map_x(row[x_key]):.3f}" '
                f'cy="{map_y(row[y_key]):.3f}" r="2.0" '
                f'fill="{color}" fill-opacity="{opacity}">'
                f"<title>q1={row['target_q1_deg']:.2f} deg, "
                f"q3={row['target_q3_deg']:.2f} deg, "
                f"condition={row['jacobian_condition']:.2f}</title>"
                "</circle>"
            )

    parts.extend(
        [
            '<text x="24" y="410" font-family="sans-serif" '
            'font-size="12" fill="#555">'
            "Point color: blue = low finger3 angle, red = high finger3 angle. "
            "Faint points fail reliability thresholds.</text>",
            "</svg>",
        ]
    )
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    actuator_ids = np.asarray(
        [
            named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
    )
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
    site_pairs = [
        (
            named_id(model, mujoco.mjtObj.mjOBJ_SITE, first),
            named_id(model, mujoco.mjtObj.mjOBJ_SITE, second),
        )
        for first, second in LOOP_PAIRS
    ]
    tip_site_id = named_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "thumb_tip"
    )

    q1_lower, q1_upper = model.actuator_ctrlrange[actuator_ids[0]]
    q3_lower, q3_upper = model.actuator_ctrlrange[actuator_ids[1]]
    q1_targets = np.linspace(q1_lower, q1_upper, GRID_SIZE)
    q3_targets = np.linspace(q3_lower, q3_upper, GRID_SIZE)
    points = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    rows = []
    previous_command = np.asarray((q1_lower, q3_lower))

    for q1_index, q1_target in enumerate(q1_targets):
        q3_indices = (
            range(GRID_SIZE)
            if q1_index % 2 == 0
            else range(GRID_SIZE - 1, -1, -1)
        )
        for q3_index in q3_indices:
            q3_target = q3_targets[q3_index]
            target_command = np.asarray((q1_target, q3_target))
            peak_force, peak_loop_error, maximum_contacts = (
                advance_to_target(
                    model,
                    data,
                    actuator_ids,
                    site_pairs,
                    previous_command,
                    target_command,
                )
            )
            previous_command = target_command

            qpos = {
                name: float(data.qpos[address])
                for name, address in qpos_addresses.items()
            }
            maximum_velocity = max(
                abs(float(data.qvel[address]))
                for address in dof_addresses.values()
            )
            loop_error = loop_errors_mm(data, site_pairs)
            tip_position = data.site_xpos[tip_site_id].copy()
            points[q1_index, q3_index] = tip_position
            tracking_error = max(
                abs(qpos["bigfinger1_joint"] - q1_target),
                abs(qpos["bigfinger3_joint"] - q3_target),
            )
            finite = bool(
                np.isfinite(tip_position).all()
                and np.isfinite(list(qpos.values())).all()
            )
            reliable = bool(
                finite
                and maximum_contacts == 0
                and tracking_error <= TRACKING_TOLERANCE
                and maximum_velocity <= VELOCITY_TOLERANCE
                and float(np.max(loop_error))
                <= LOOP_ERROR_TOLERANCE_MM
            )

            rows.append(
                {
                    "q1_index": q1_index,
                    "q3_index": q3_index,
                    "target_q1_rad": q1_target,
                    "target_q3_rad": q3_target,
                    "target_q1_deg": np.rad2deg(q1_target),
                    "target_q3_deg": np.rad2deg(q3_target),
                    "q1_rad": qpos["bigfinger1_joint"],
                    "q2_rad": qpos["bigfinger2_joint"],
                    "q3_rad": qpos["bigfinger3_joint"],
                    "q4_rad": qpos["bigfinger4_joint"],
                    "tip_x_m": tip_position[0],
                    "tip_y_m": tip_position[1],
                    "tip_z_m": tip_position[2],
                    "tip_x_mm": 1000.0 * tip_position[0],
                    "tip_y_mm": 1000.0 * tip_position[1],
                    "tip_z_mm": 1000.0 * tip_position[2],
                    "tracking_error_deg": np.rad2deg(tracking_error),
                    "maximum_abs_qvel_rad_s": maximum_velocity,
                    "steady_force_finger1_nm": data.actuator_force[
                        actuator_ids[0]
                    ],
                    "steady_force_finger3_nm": data.actuator_force[
                        actuator_ids[1]
                    ],
                    "segment_peak_force_finger1_nm": peak_force[0],
                    "segment_peak_force_finger3_nm": peak_force[1],
                    "loop_A_error_mm": loop_error[0],
                    "loop_B_error_mm": loop_error[1],
                    "segment_peak_loop_error_mm": peak_loop_error,
                    "maximum_contacts": maximum_contacts,
                    "finite": finite,
                    "reliable": reliable,
                }
            )

    add_jacobian_metrics(rows, points, q1_targets, q3_targets)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_projection_svg(rows)

    reliable_rows = [row for row in rows if row["reliable"]]
    point_cloud_mm = 1000.0 * points.reshape(-1, 3)
    minimum = point_cloud_mm.min(axis=0)
    maximum = point_cloud_mm.max(axis=0)
    span = maximum - minimum
    worst_condition = max(
        rows,
        key=lambda row: (
            row["jacobian_condition"]
            if np.isfinite(row["jacobian_condition"])
            else math.inf
        ),
    )
    minimum_sigma = min(
        rows, key=lambda row: row["jacobian_sigma_min_m_rad"]
    )

    print("Thumb-tip workspace scan")
    print(f"  grid                    = {GRID_SIZE} x {GRID_SIZE}")
    print(f"  configurations          = {len(rows)}")
    print(
        "  X range [mm]            = "
        f"{minimum[0]:.3f} .. {maximum[0]:.3f}"
    )
    print(
        "  Y range [mm]            = "
        f"{minimum[1]:.3f} .. {maximum[1]:.3f}"
    )
    print(
        "  Z range [mm]            = "
        f"{minimum[2]:.3f} .. {maximum[2]:.3f}"
    )
    print(
        "  axis spans [mm]          = "
        f"X {span[0]:.3f}, Y {span[1]:.3f}, Z {span[2]:.3f}"
    )
    print(
        "  reliable configurations = "
        f"{len(reliable_rows)} / {len(rows)}"
    )
    print(
        "  maximum tracking error  = "
        f"{max(row['tracking_error_deg'] for row in rows):.6f} deg"
    )
    print(
        "  maximum loop error      = "
        f"{max(max(row['loop_A_error_mm'], row['loop_B_error_mm']) for row in rows):.6f} mm"
    )
    print(
        "  maximum contacts        = "
        f"{max(row['maximum_contacts'] for row in rows)}"
    )
    print(
        "  minimum Jacobian sigma  = "
        f"{minimum_sigma['jacobian_sigma_min_m_rad']:.8f} m/rad "
        f"at q1={minimum_sigma['target_q1_deg']:.3f} deg, "
        f"q3={minimum_sigma['target_q3_deg']:.3f} deg"
    )
    print(
        "  worst condition number  = "
        f"{worst_condition['jacobian_condition']:.6f} "
        f"at q1={worst_condition['target_q1_deg']:.3f} deg, "
        f"q3={worst_condition['target_q3_deg']:.3f} deg"
    )
    print(f"  CSV                     = {CSV_PATH}")
    print(f"  projections             = {SVG_PATH}")

    if not all(row["finite"] for row in rows):
        raise RuntimeError("The workspace scan produced NaN or Inf")


if __name__ == "__main__":
    main()
