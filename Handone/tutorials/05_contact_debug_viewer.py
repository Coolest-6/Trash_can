"""Visualize the raw mesh contact that is excluded in the production model.

Run from the repository root:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/05_contact_debug_viewer.py

Red markers/arrows show contacts reported between the convex collision hulls.
Pass --production to view the normal model with its contact exclusions.
"""

import argparse
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
PERIOD = 12.0
EXCLUSION = (
    '    <exclude body1="base_link" body2="bigfinger2_Link"/>\n'
)


def load_model(show_raw_contact):
    if not show_raw_contact:
        return mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    xml = MODEL_PATH.read_text(encoding="utf-8")
    xml = xml.replace(EXCLUSION, "")
    mesh_directory = (ROOT / "meshes").as_posix() + "/"
    xml = xml.replace("../meshes/", mesh_directory)
    return mujoco.MjModel.from_xml_string(xml)


def body_name(model, geom_id):
    body_id = int(model.geom_bodyid[geom_id])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name or "world"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production contact exclusions instead of raw mesh contact.",
    )
    args = parser.parse_args()

    model = load_model(show_raw_contact=not args.production)
    data = mujoco.MjData(model)
    finger1_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger1_motor"
    )
    finger3_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger3_motor"
    )
    lower, upper = model.actuator_ctrlrange[finger3_id]

    model.vis.rgba.contactpoint[:] = (1.0, 0.0, 0.0, 1.0)
    model.vis.rgba.contactforce[:] = (1.0, 0.0, 0.0, 1.0)
    model.vis.scale.contactwidth = 0.004
    model.vis.scale.contactheight = 0.008
    model.vis.scale.forcewidth = 0.003

    mode = "production exclusions" if args.production else "raw convex-hull contact"
    print(f"Viewer mode: {mode}")
    print("finger3_motor moves smoothly through 0..44.69 degrees and back.")
    print("Red contact markers appear only when MuJoCo reports contact.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        wall_start = time.time()
        simulation_start = data.time
        was_contacting = False

        while viewer.is_running():
            while data.time - simulation_start < time.time() - wall_start:
                phase = 2.0 * math.pi * data.time / PERIOD
                data.ctrl[finger1_id] = 0.0
                data.ctrl[finger3_id] = lower + 0.5 * (
                    upper - lower
                ) * (1.0 - math.cos(phase))
                mujoco.mj_step(model, data)

                matching_contacts = []
                for contact in data.contact[: data.ncon]:
                    first = body_name(model, contact.geom1)
                    second = body_name(model, contact.geom2)
                    if {first, second} == {
                        "base_link",
                        "bigfinger2_Link",
                    }:
                        matching_contacts.append(contact)

                is_contacting = bool(matching_contacts)
                if is_contacting and not was_contacting:
                    joint_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        "bigfinger3_joint",
                    )
                    angle = data.qpos[model.jnt_qposadr[joint_id]]
                    depth = min(
                        -1000.0 * float(contact.dist)
                        for contact in matching_contacts
                    )
                    print(
                        f"Contact begins: q3={math.degrees(angle):.3f} deg, "
                        f"penetration={depth:.6f} mm"
                    )
                elif was_contacting and not is_contacting:
                    print("Contact ends")
                was_contacting = is_contacting

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
