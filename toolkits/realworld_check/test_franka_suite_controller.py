# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive smoke test for :class:`FrankaSuiteController` (FR3 via the
franka_suite HTTP server). Type ``help`` at the prompt for the command list.

Prereqs on the robot host (see the franka_suite repo):
  1. ROS2 impedance controller running (``ros2 launch ... impedance.launch.py``)
  2. Flask robot server running (``./scripts/run_robot_server.sh``)
  3. ``FRANKA_SERVER_URL`` exported (e.g. ``http://127.0.0.1:5000``)

This validates the RLinf Ray-actor -> HTTP -> franka_suite path in isolation,
before wiring the controller into a full FrankaEnv / RL run.
"""

import os
import time

import ray  # noqa: E402

if not ray.is_initialized():
    try:
        ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
    except Exception:
        ray.init(log_to_driver=False, logging_level="ERROR")

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from rlinf.envs.realworld.franka.franka_suite_controller import (  # noqa: E402
    FrankaSuiteController,
)


def _print_help() -> None:
    print(
        "commands: q | getstate | getpos | getpos_euler | getjoint | "
        "nudge <axis xyz 0-2> <d_m> | jointreset | "
        "open | close | grip <0-255> | compliance <key> <val>"
    )


def main() -> None:
    server_url = os.environ.get("FRANKA_SERVER_URL", "http://127.0.0.1:5000")
    robot_ip = os.environ.get("FRANKA_ROBOT_IP")

    controller = FrankaSuiteController.launch_controller(
        robot_ip=robot_ip,
        server_url=server_url,
    )

    start_time = time.time()
    while not controller.is_robot_up().wait()[0]:
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(f"Waited {time.time() - start_time:.1f}s for FR3 to be ready")
            break

    print(f"Connected to franka_suite server at {server_url}")
    _print_help()

    while True:
        try:
            cmd_str = input("cmd> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd_str:
            continue
        parts = cmd_str.split()
        cmd = parts[0].lower()

        try:
            if cmd == "q":
                break
            elif cmd == "help":
                _print_help()
            elif cmd == "getstate":
                st = controller.get_state().wait()[0]
                print(
                    f"pose={np.round(st.tcp_pose, 4)}\n"
                    f"q={np.round(st.arm_joint_position, 4)}\n"
                    f"force={np.round(st.tcp_force, 3)} "
                    f"gripper_pos={st.gripper_position:.3f} open={st.gripper_open}"
                )
            elif cmd == "getpos":
                print(controller.get_state().wait()[0].tcp_pose)
            elif cmd == "getpos_euler":
                pose = controller.get_state().wait()[0].tcp_pose
                euler = R.from_quat(pose[3:].copy()).as_euler("xyz")
                print(np.concatenate([pose[:3], euler]))
            elif cmd == "getjoint":
                print(controller.get_state().wait()[0].arm_joint_position)
            elif cmd == "nudge":
                if len(parts) != 3:
                    print("usage: nudge <axis 0=x,1=y,2=z> <delta_m>")
                    continue
                axis = int(parts[1])
                delta = float(parts[2])
                assert 0 <= axis < 3, "axis must be 0..2"
                assert abs(delta) <= 0.05, "refusing nudge > 0.05 m"
                pose = controller.get_state().wait()[0].tcp_pose.copy()
                pose[axis] += delta
                print(f"move_arm: {np.round(pose, 4)}")
                controller.move_arm(pose).wait()
            elif cmd == "jointreset":
                print("joint reset (server-side reset config)...")
                controller.reset_joint().wait()
                print("done")
            elif cmd == "open":
                controller.open_gripper().wait()
                print("gripper opened")
            elif cmd == "close":
                controller.close_gripper().wait()
                print("gripper closed")
            elif cmd == "grip":
                if len(parts) != 2:
                    print("usage: grip <0-255>")
                    continue
                controller.move_gripper(int(parts[1])).wait()
                print(f"gripper moved to {parts[1]}")
            elif cmd == "compliance":
                if len(parts) != 3:
                    print("usage: compliance <param_name> <value>")
                    continue
                controller.reconfigure_compliance_params(
                    {parts[1]: float(parts[2])}
                ).wait()
                print(f"compliance updated: {parts[1]}={parts[2]}")
            else:
                print(f"unknown cmd: {cmd_str}")
                _print_help()
        except Exception as e:
            print(f"command failed: {e}")

        time.sleep(0.05)

    print("shutting down...")
    try:
        controller.cleanup().wait()
    except Exception:
        pass


if __name__ == "__main__":
    main()
