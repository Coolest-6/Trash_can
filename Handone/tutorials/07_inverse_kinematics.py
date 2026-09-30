"""Solve thumb inverse kinematics from the sampled workspace.

Examples:
    D:\miniconda\envs\mujoco_env\python.exe tutorials/07_inverse_kinematics.py
    D:\miniconda\envs\mujoco_env\python.exe tutorials/07_inverse_kinematics.py --target-mm 75 -55 85
    D:\miniconda\envs\mujoco_env\python.exe tutorials/07_inverse_kinematics.py --viewer

The solver uses the workspace table for fast numerical optimization, then
validates every solution in the full MuJoCo closed-loop dynamics model.
"""

import argparse
import csv
import math
from pathlib import Path
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "urdf" / "Handone.xml"
WORKSPACE_PATH = ROOT / "tutorials" / "results" / "06_thumb_workspace.csv"
OUTPUT_PATH = ROOT / "tutorials" / "results" / "07_inverse_kinematics_results.csv"

ACTUATOR_NAMES = ("finger1_motor", "finger3_motor")
TIP_SITE_NAME = "thumb_tip"
DEMO_COMMANDS_DEG = ((10.0, 15.0), (25.0, 30.0), (40.0, 40.0))
POSITION_TOLERANCE_MM = 0.25
DAMPING = 0.002
MAX_ITERATIONS = 60
MAX_STEP_RAD = 0.08


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MJCF object not found: {name}")
    return object_id


def quintic_progress(time_value, duration):
    ratio = np.clip(time_value / duration, 0.0, 1.0)
    return 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5


class WorkspaceMap:
    """Bilinear forward-kinematics map built from the workspace CSV."""

    def __init__(self, csv_path):
        with csv_path.open(encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        if not rows:
            raise RuntimeError(f"Workspace CSV is empty: {csv_path}")

        self.q1 = np.asarray(
            sorted({float(row["target_q1_rad"]) for row in rows})
        )
        self.q3 = np.asarray(
            sorted({float(row["target_q3_rad"]) for row in rows})
        )
        self.points = np.zeros((len(self.q1), len(self.q3), 3))
        q1_index = {value: index for index, value in enumerate(self.q1)}
        q3_index = {value: index for index, value in enumerate(self.q3)}

        for row in rows:
            i = q1_index[float(row["target_q1_rad"])]
            j = q3_index[float(row["target_q3_rad"])]
            self.points[i, j] = (
                float(row["tip_x_m"]),
                float(row["tip_y_m"]),
                float(row["tip_z_m"]),
            )

        self.lower = np.asarray((self.q1[0], self.q3[0]))
        self.upper = np.asarray((self.q1[-1], self.q3[-1]))
        self.sample_commands = np.asarray(
            [(q1_value, q3_value) for q1_value in self.q1 for q3_value in self.q3]
        )
        self.sample_points = self.points.reshape(-1, 3)

    @staticmethod
    def _interval(values, query):
        upper_index = int(np.searchsorted(values, query, side="right"))
        upper_index = min(max(upper_index, 1), len(values) - 1)
        lower_index = upper_index - 1
        lower_value = values[lower_index]
        upper_value = values[upper_index]
        fraction = (
            (query - lower_value) / (upper_value - lower_value)
            if upper_value > lower_value
            else 0.0
        )
        return lower_index, upper_index, float(fraction)

    def forward(self, command):
        command = np.clip(command, self.lower, self.upper)
        i0, i1, q1_fraction = self._interval(self.q1, command[0])
        j0, j1, q3_fraction = self._interval(self.q3, command[1])
        p00 = self.points[i0, j0]
        p10 = self.points[i1, j0]
        p01 = self.points[i0, j1]
        p11 = self.points[i1, j1]
        return (
            (1.0 - q1_fraction) * (1.0 - q3_fraction) * p00
            + q1_fraction * (1.0 - q3_fraction) * p10
            + (1.0 - q1_fraction) * q3_fraction * p01
            + q1_fraction * q3_fraction * p11
        )

    def jacobian(self, command):
        epsilon = 1e-4
        columns = []
        for axis in range(2):
            lower_command = command.copy()
            upper_command = command.copy()
            lower_command[axis] = max(
                self.lower[axis], command[axis] - epsilon
            )
            upper_command[axis] = min(
                self.upper[axis], command[axis] + epsilon
            )
            difference = upper_command[axis] - lower_command[axis]
            if difference <= 0.0:
                columns.append(np.zeros(3))
            else:
                columns.append(
                    (
                        self.forward(upper_command)
                        - self.forward(lower_command)
                    )
                    / difference
                )
        return np.column_stack(columns)

    def nearest_seeds(self, target, count=8):
        distance = np.linalg.norm(self.sample_points - target, axis=1)
        indices = np.argsort(distance)[:count]
        return self.sample_commands[indices]


def solve_from_seed(workspace, target, seed):
    command = np.clip(np.asarray(seed, dtype=float), workspace.lower, workspace.upper)
    best_command = command.copy()
    best_error = math.inf

    for iteration in range(1, MAX_ITERATIONS + 1):
        position = workspace.forward(command)
        error_vector = target - position
        error = float(np.linalg.norm(error_vector))
        if error < best_error:
            best_error = error
            best_command = command.copy()
        if 1000.0 * error <= POSITION_TOLERANCE_MM:
            break

        jacobian = workspace.jacobian(command)
        normal_matrix = jacobian.T @ jacobian + DAMPING**2 * np.eye(2)
        delta = np.linalg.solve(normal_matrix, jacobian.T @ error_vector)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > MAX_STEP_RAD:
            delta *= MAX_STEP_RAD / delta_norm

        accepted = False
        scale = 1.0
        for _ in range(8):
            candidate = np.clip(
                command + scale * delta, workspace.lower, workspace.upper
            )
            candidate_error = np.linalg.norm(
                target - workspace.forward(candidate)
            )
            if candidate_error < error:
                command = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

    return {
        "command": best_command,
        "predicted_position": workspace.forward(best_command),
        "predicted_error_m": best_error,
        "iterations": iteration,
    }


def solve_ik(workspace, target):
    candidates = [
        solve_from_seed(workspace, target, seed)
        for seed in workspace.nearest_seeds(target)
    ]
    return min(candidates, key=lambda result: result["predicted_error_m"])


def validate_in_mujoco(model, target_command, move_duration=0.8, settle_duration=0.4):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    actuator_ids = np.asarray(
        [
            named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
    )
    tip_site_id = named_id(
        model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE_NAME
    )

    while data.time < move_duration:
        progress = quintic_progress(data.time, move_duration)
        data.ctrl[actuator_ids] = progress * target_command
        mujoco.mj_step(model, data)
    data.ctrl[actuator_ids] = target_command
    end_time = data.time + settle_duration
    while data.time < end_time:
        mujoco.mj_step(model, data)

    return {
        "position": data.site_xpos[tip_site_id].copy(),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "actuator_force": data.actuator_force[actuator_ids].copy(),
        "contacts": data.ncon,
        "finite": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
        ),
    }


def animate_solutions(model, solutions):
    import mujoco.viewer

    data = mujoco.MjData(model)
    actuator_ids = np.asarray(
        [
            named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
    )
    commands = [np.zeros(2)] + [
        solution["command"] for solution in solutions
    ]
    segment_duration = 1.5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        simulation_start = data.time
        while viewer.is_running():
            while data.time - simulation_start < time.time() - wall_start:
                sequence_time = data.time % (
                    segment_duration * len(commands)
                )
                segment_index = int(sequence_time / segment_duration)
                next_index = (segment_index + 1) % len(commands)
                local_time = sequence_time - segment_index * segment_duration
                progress = quintic_progress(
                    local_time, segment_duration
                )
                data.ctrl[actuator_ids] = (
                    commands[segment_index]
                    + progress
                    * (commands[next_index] - commands[segment_index])
                )
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Solve one target position expressed in world coordinates [mm].",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Animate the solved targets after numerical validation.",
    )
    args = parser.parse_args()

    if not WORKSPACE_PATH.exists():
        raise RuntimeError(
            "Workspace CSV not found. Run 06_thumb_workspace_scan.py first."
        )

    workspace = WorkspaceMap(WORKSPACE_PATH)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    if args.target_mm is not None:
        targets = [np.asarray(args.target_mm, dtype=float) / 1000.0]
        target_labels = ["custom target"]
    else:
        demo_commands = [
            np.deg2rad(command) for command in DEMO_COMMANDS_DEG
        ]
        targets = [
            workspace.forward(command) for command in demo_commands
        ]
        target_labels = [
            f"demo from q={command_deg} deg"
            for command_deg in DEMO_COMMANDS_DEG
        ]

    results = []
    for label, target in zip(target_labels, targets):
        solution = solve_ik(workspace, target)
        validation = validate_in_mujoco(model, solution["command"])
        actual_error = float(
            np.linalg.norm(target - validation["position"])
        )
        reachable = bool(
            1000.0 * actual_error <= POSITION_TOLERANCE_MM
            and validation["finite"]
            and validation["contacts"] == 0
        )
        result = {
            **solution,
            "label": label,
            "target": target,
            "validation": validation,
            "actual_error_m": actual_error,
            "reachable": reachable,
        }
        results.append(result)

        print(f"\n{label}")
        print(f"  target [mm]          = {(1000.0 * target).round(4)}")
        print(
            "  solved motors [deg]  = "
            f"{np.rad2deg(solution['command']).round(5)}"
        )
        print(
            "  predicted tip [mm]   = "
            f"{(1000.0 * solution['predicted_position']).round(4)}"
        )
        print(
            f"  predicted error      = "
            f"{1000.0 * solution['predicted_error_m']:.6f} mm"
        )
        print(
            "  MuJoCo tip [mm]      = "
            f"{(1000.0 * validation['position']).round(4)}"
        )
        print(
            f"  MuJoCo error         = {1000.0 * actual_error:.6f} mm"
        )
        print(f"  iterations           = {solution['iterations']}")
        print(f"  reachable            = {reachable}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = (
            "label",
            "target_x_mm",
            "target_y_mm",
            "target_z_mm",
            "motor1_deg",
            "motor3_deg",
            "predicted_error_mm",
            "mujoco_error_mm",
            "iterations",
            "reachable",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "label": result["label"],
                    "target_x_mm": 1000.0 * result["target"][0],
                    "target_y_mm": 1000.0 * result["target"][1],
                    "target_z_mm": 1000.0 * result["target"][2],
                    "motor1_deg": np.rad2deg(result["command"][0]),
                    "motor3_deg": np.rad2deg(result["command"][1]),
                    "predicted_error_mm": 1000.0
                    * result["predicted_error_m"],
                    "mujoco_error_mm": 1000.0
                    * result["actual_error_m"],
                    "iterations": result["iterations"],
                    "reachable": result["reachable"],
                }
            )
    print(f"\nResults written to: {OUTPUT_PATH}")

    if args.viewer:
        animate_solutions(model, results)


if __name__ == "__main__":
    main()
