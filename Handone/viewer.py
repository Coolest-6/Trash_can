import mujoco
import mujoco.viewer
import math
import time

model = mujoco.MjModel.from_xml_path("urdf/Handone.xml")
data = mujoco.MjData(model)

MOTOR_PERIODS = {
    "finger1_motor": 4.0,
    "finger3_motor": 3.0,
}

motor_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    for name in MOTOR_PERIODS
}
if any(motor_id == -1 for motor_id in motor_ids.values()):
    raise RuntimeError("The MJCF must define finger1_motor and finger3_motor")


def update_motor_targets(sim_time):
    """Move each motor smoothly between its configured control limits."""
    for name, period in MOTOR_PERIODS.items():
        motor_id = motor_ids[name]
        lower, upper = model.actuator_ctrlrange[motor_id]
        phase = 2.0 * math.pi * sim_time / period
        data.ctrl[motor_id] = lower + 0.5 * (upper - lower) * (1.0 - math.cos(phase))


with mujoco.viewer.launch_passive(model, data) as viewer:
    wall_start = time.time()
    sim_start = data.time

    while viewer.is_running():
        # Keep simulation time synchronized with wall time. A fixed sleep after
        # every 1 ms step makes the viewer run slower than real time.
        while data.time - sim_start < time.time() - wall_start:
            update_motor_targets(data.time)
            mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.001)
