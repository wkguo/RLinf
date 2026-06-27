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

"""Franka controller backed by the ``franka_suite`` HTTP robot server.

This is the ROS-free integration seam for a real **Franka Research 3 (FR3)**.
The arm is driven by the ``franka_suite`` stack on the robot host:

    RLinf EnvWorker / FrankaEnv
        -> FrankaSuiteController (this Ray actor, requests-only)
        -> HTTP  http://<robot-host>:5000   (franka_suite Flask server)
        -> ROS2  serl_franka_controllers_ros2 (1 kHz Cartesian impedance)
        -> libfranka -> FR3

The controller exposes exactly the method surface that :class:`FrankaEnv`
calls on its controller (``get_state``/``move_arm``/``reset_joint``/
``clear_errors``/``reconfigure_compliance_params``/gripper ops), so the
existing ``FrankaEnv`` / ``RealWorldEnv`` / async-RL pipeline is reused
unchanged.

The thin HTTP client is vendored here (only depends on ``requests``) so the
RLinf side is self-contained: the robot host needs the full ``franka_suite``
ROS2 stack to run the *server*, but RLinf only needs ``requests`` to talk to
it. The client mirrors ``franka_suite/integration/franka_client.py``; keep the
two in sync if the HTTP API changes.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .franka_robot_state import FrankaRobotState

# Default normalized-width threshold above which the gripper counts as "open".
# franka_suite reports gripper width in [0, 1] (~1 open, ~0 closed); RLinf's
# binary gripper gating needs a boolean ``gripper_open``.
_DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.5
_DEFAULT_SERVER_URL = "http://127.0.0.1:5000"


class _HttpFrankaClient:
    """Minimal, dependency-light client for one franka_suite robot server.

    One instance == one server == one FR3. Methods map 1:1 to the HTTP routes
    documented in ``franka_suite/docs/02_HTTP_API_REFERENCE.md``. Vendored so
    RLinf does not have to import the ``franka_suite`` package.
    """

    def __init__(self, base_url: str, timeout: float = 5.0):
        import requests  # lazy: keep module importable on nodes without requests

        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self._session = requests.Session()

    def _post(self, route: str, json: Any = None):
        return self._session.post(self.base_url + route, json=json, timeout=self.timeout)

    def get_state(self) -> Dict[str, Any]:
        return self._post("getstate").json()

    def move_pose(self, pose7: Sequence[float]) -> None:
        self._post("pose", json={"arr": [float(x) for x in pose7]})

    def clear_error(self) -> None:
        self._post("clearerr")

    def update_compliance(self, params: Dict[str, float]) -> None:
        self._post("update_param", json={k: float(v) for k, v in params.items()})

    def start_impedance(self) -> Dict[str, Any]:
        return self._post("startimp").json()

    def stop_impedance(self) -> Dict[str, Any]:
        return self._post("stopimp").json()

    def joint_reset(self) -> Dict[str, Any]:
        r = self._post("jointreset")
        try:
            return r.json()
        except ValueError:
            return {"ok": r.ok, "message": r.text}

    def open_gripper(self) -> bool:
        return self._post("open_gripper").status_code != 503

    def close_gripper(self, slow: bool = False) -> bool:
        return self._post("close_gripper_slow" if slow else "close_gripper").status_code != 503

    def move_gripper(self, position: int) -> None:
        self._post("move_gripper", json={"gripper_pos": int(position)})

    def is_alive(self) -> bool:
        import requests

        try:
            return self._post("getstate").ok
        except requests.RequestException:
            return False


class FrankaSuiteController(Worker):
    """One FR3 arm driven through the franka_suite HTTP server.

    Drop-in alternative to :class:`FrankaController` (ROS) and
    :class:`FrankyController` (libfranka). Spawned per-arm as a Ray actor by
    :meth:`launch_controller`, normally pinned to the robot host so the HTTP
    round-trip to ``franka_suite`` stays on localhost.
    """

    @staticmethod
    def launch_controller(
        robot_ip: Optional[str] = None,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        server_url: Optional[str] = None,
        gripper_open_threshold: float = _DEFAULT_GRIPPER_OPEN_THRESHOLD,
        **_ignored,
    ):
        """Launch a FrankaSuiteController on the robot host's node.

        Args:
            robot_ip: FR3 FCI IP. Informational only here (control goes over
                HTTP); the franka_suite server owns the FCI connection.
            env_idx / worker_rank: used only to name the Ray actor uniquely.
            node_rank: cluster node rank to place the actor on (the robot host).
            server_url: franka_suite Flask URL. Defaults to localhost:5000,
                i.e. the actor co-located with the server on the robot host.
            gripper_open_threshold: normalized width above which the gripper
                is considered open (for binary gripper gating).
        """
        return FrankaSuiteController.create_group(
            server_url or _DEFAULT_SERVER_URL,
            robot_ip,
            gripper_open_threshold,
        ).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
            name=f"FrankaSuiteController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        server_url: str,
        robot_ip: Optional[str] = None,
        gripper_open_threshold: float = _DEFAULT_GRIPPER_OPEN_THRESHOLD,
    ):
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._open_threshold = float(gripper_open_threshold)
        self._state = FrankaRobotState()

        self._client = _HttpFrankaClient(server_url)
        self._logger.info(
            f"FrankaSuiteController connecting to franka_suite server at {server_url}"
            + (f" (FR3 IP {robot_ip})" if robot_ip else "")
        )
        # Ensure the impedance controller is active. The ROS2 launch normally
        # already activated it; /startimp is load+activate and idempotent-ish.
        try:
            self._client.start_impedance()
        except Exception as exc:  # pragma: no cover - server may already be up
            self._logger.warning(f"start_impedance on connect failed (continuing): {exc}")

    # ------------------------------------------------------------------ state
    def is_robot_up(self) -> bool:
        return self._client.is_alive()

    def get_state(self) -> FrankaRobotState:
        s = self._client.get_state()
        st = self._state
        st.tcp_pose = np.asarray(s["pose"], dtype=np.float64)
        st.tcp_vel = np.asarray(s["vel"], dtype=np.float64)
        st.arm_joint_position = np.asarray(s["q"], dtype=np.float64)
        st.arm_joint_velocity = np.asarray(s["dq"], dtype=np.float64)
        st.tcp_force = np.asarray(s["force"], dtype=np.float64)
        st.tcp_torque = np.asarray(s["torque"], dtype=np.float64)
        st.arm_jacobian = np.asarray(s["jacobian"], dtype=np.float64).reshape(6, 7)
        gripper_pos = float(s["gripper_pos"])
        st.gripper_position = gripper_pos
        st.gripper_open = gripper_pos > self._open_threshold
        st.hand_position = None  # franka_suite has no dexterous-hand backend
        return st

    # ----------------------------------------------------------------- motion
    def move_arm(self, position: np.ndarray) -> None:
        """Set the Cartesian equilibrium target ``[x, y, z, qx, qy, qz, qw]``."""
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        self._client.move_pose(position)

    def clear_errors(self) -> None:
        self._client.clear_error()

    def reconfigure_compliance_params(self, params: Dict[str, float]) -> None:
        if params:
            self._client.update_compliance(params)

    def reset_joint(self, reset_pos=None) -> None:
        """Drive to the reset joint configuration.

        NOTE: franka_suite's ``/jointreset`` uses a fixed reset configuration
        held server-side; the ``reset_pos`` argument is accepted for interface
        parity with :class:`FrankaController` but is ignored. Configure the
        target joints on the franka_suite server side.
        """
        if reset_pos is not None:
            self._logger.debug(
                "reset_joint(reset_pos=...) ignored; franka_suite holds the "
                "reset configuration server-side."
            )
        self._client.joint_reset()

    def start_impedance(self) -> Dict[str, Any]:
        return self._client.start_impedance()

    def stop_impedance(self) -> Dict[str, Any]:
        return self._client.stop_impedance()

    # ---------------------------------------------------------------- gripper
    def open_gripper(self) -> None:
        self._client.open_gripper()
        self._state.gripper_open = True

    def close_gripper(self) -> None:
        self._client.close_gripper()
        self._state.gripper_open = False

    def move_gripper(self, position: int, speed: float = 0.3) -> None:
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        self._client.move_gripper(position)

    def command_end_effector(self, action: np.ndarray) -> bool:
        """Binary gripper command (parity with FrankaController).

        franka_suite exposes a 2-finger gripper only; dexterous hands are not
        supported by this backend.
        """
        value = float(np.asarray(action).reshape(-1)[0])
        if value <= -0.5 and self._state.gripper_open:
            self.close_gripper()
            return True
        if value >= 0.5 and not self._state.gripper_open:
            self.open_gripper()
            return True
        return False

    def reset_end_effector(self, target_state=None) -> None:
        if target_state is not None:
            self.command_end_effector(np.asarray(target_state))

    # --------------------------------------------------------------- lifecycle
    def cleanup(self) -> None:
        try:
            self._client.stop_impedance()
        except Exception:  # pragma: no cover
            pass
