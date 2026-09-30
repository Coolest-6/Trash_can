"""Audit URDF mass properties against the compiled MuJoCo model.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/08_mass_properties_audit.py

Output:
    tutorials/results/08_mass_properties_audit.csv
"""

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "urdf" / "Handone.urdf"
MJCF_PATH = ROOT / "urdf" / "Handone.xml"
OUTPUT_PATH = ROOT / "tutorials" / "results" / "08_mass_properties_audit.csv"

MASS_RELATIVE_TOLERANCE = 1e-8
COM_TOLERANCE_M = 1e-9
INERTIA_RELATIVE_TOLERANCE = 1e-5


def vector(text):
    return np.asarray([float(value) for value in text.split()])


def urdf_inertia_matrix(attributes):
    return np.asarray(
        [
            [
                float(attributes["ixx"]),
                float(attributes["ixy"]),
                float(attributes["ixz"]),
            ],
            [
                float(attributes["ixy"]),
                float(attributes["iyy"]),
                float(attributes["iyz"]),
            ],
            [
                float(attributes["ixz"]),
                float(attributes["iyz"]),
                float(attributes["izz"]),
            ],
        ]
    )


def mujoco_inertia_matrix(model, body_id):
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, model.body_iquat[body_id])
    rotation = rotation_flat.reshape(3, 3)
    return (
        rotation
        @ np.diag(model.body_inertia[body_id])
        @ rotation.T
    )


def relative_error(actual, expected):
    denominator = max(float(np.linalg.norm(expected)), 1e-16)
    return float(np.linalg.norm(actual - expected) / denominator)


def parse_urdf_links(root):
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        links[link.get("name")] = {
            "mass": float(inertial.find("mass").get("value")),
            "com": vector(inertial.find("origin").get("xyz")),
            "inertia": urdf_inertia_matrix(
                inertial.find("inertia").attrib
            ),
        }
    return links


def parse_urdf_joints(root):
    joints = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        joints[joint.get("name")] = {
            "axis": vector(joint.find("axis").get("xyz")),
            "range": np.asarray(
                [
                    float(limit.get("lower")),
                    float(limit.get("upper")),
                ]
            ),
        }
    return joints


def main():
    urdf_root = ET.parse(URDF_PATH).getroot()
    urdf_links = parse_urdf_links(urdf_root)
    urdf_joints = parse_urdf_joints(urdf_root)
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    rows = []

    print("Mass-property audit")
    print(
        "  The MJCF inertia matrix is reconstructed in each body-local frame "
        "before comparison."
    )

    for name, urdf_data in urdf_links.items():
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise RuntimeError(f"MJCF body not found: {name}")

        mjcf_mass = float(model.body_mass[body_id])
        mjcf_com = model.body_ipos[body_id].copy()
        mjcf_inertia = mujoco_inertia_matrix(model, body_id)
        principal = np.linalg.eigvalsh(urdf_data["inertia"])
        positive_definite = bool(np.all(principal > 0.0))
        triangle_valid = bool(
            principal[0] + principal[1] >= principal[2] - 1e-14
        )
        mass_error = abs(mjcf_mass - urdf_data["mass"]) / max(
            urdf_data["mass"], 1e-16
        )
        com_error = float(
            np.linalg.norm(mjcf_com - urdf_data["com"])
        )
        inertia_error = relative_error(
            mjcf_inertia, urdf_data["inertia"]
        )
        matches = bool(
            mass_error <= MASS_RELATIVE_TOLERANCE
            and com_error <= COM_TOLERANCE_M
            and inertia_error <= INERTIA_RELATIVE_TOLERANCE
            and positive_definite
            and triangle_valid
        )
        radius_of_gyration = np.sqrt(
            principal / urdf_data["mass"]
        )

        rows.append(
            {
                "body": name,
                "urdf_mass_kg": urdf_data["mass"],
                "mjcf_mass_kg": mjcf_mass,
                "mass_relative_error": mass_error,
                "com_error_mm": 1000.0 * com_error,
                "inertia_relative_error": inertia_error,
                "principal_I_min_kg_m2": principal[0],
                "principal_I_mid_kg_m2": principal[1],
                "principal_I_max_kg_m2": principal[2],
                "radius_gyration_min_mm": 1000.0
                * radius_of_gyration[0],
                "radius_gyration_mid_mm": 1000.0
                * radius_of_gyration[1],
                "radius_gyration_max_mm": 1000.0
                * radius_of_gyration[2],
                "positive_definite": positive_definite,
                "triangle_valid": triangle_valid,
                "matches": matches,
                "fixed_body": bool(model.body_dofnum[body_id] == 0),
            }
        )

        print(
            f"  {name:19s} mass={mjcf_mass:.8f} kg, "
            f"COM error={1000.0 * com_error:.6g} mm, "
            f"inertia error={inertia_error:.3e}, "
            f"physical={positive_definite and triangle_valid}, "
            f"match={matches}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    urdf_total_mass = sum(
        data["mass"] for data in urdf_links.values()
    )
    mjcf_total_mass = sum(
        float(model.body_mass[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
        ])
        for name in urdf_links
    )
    all_match = all(row["matches"] for row in rows)

    print(f"\n  URDF total mass = {urdf_total_mass:.8f} kg")
    print(f"  MJCF total mass = {mjcf_total_mass:.8f} kg")
    print(f"  all mass properties match = {all_match}")

    print("\nJoint definitions that differ from the source URDF")
    difference_count = 0
    for name, urdf_data in urdf_joints.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            print(f"  {name}: missing from MJCF")
            difference_count += 1
            continue
        mjcf_axis = model.jnt_axis[joint_id]
        mjcf_range = model.jnt_range[joint_id]
        axis_differs = not np.allclose(
            mjcf_axis, urdf_data["axis"], atol=1e-10
        )
        range_differs = not np.allclose(
            mjcf_range, urdf_data["range"], atol=1e-10
        )
        if axis_differs or range_differs:
            difference_count += 1
            print(
                f"  {name}: "
                f"axis URDF={urdf_data['axis']} MJCF={mjcf_axis}; "
                f"range URDF={urdf_data['range']} MJCF={mjcf_range}"
            )
    if difference_count == 0:
        print("  none")

    print(f"\nCSV written to: {OUTPUT_PATH}")
    if not all_match:
        raise RuntimeError("One or more MJCF mass properties do not match the URDF")


if __name__ == "__main__":
    main()
