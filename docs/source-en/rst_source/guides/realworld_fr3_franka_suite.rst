FR3 via franka_suite (ROS-free HTTP backend)
============================================

This guide explains how to drive a real **Franka Research 3 (FR3)** from RLinf
through the ``franka_suite`` HTTP robot server, instead of RLinf's native
ROS ``FrankaController``. It is the recommended path for FR3 and for scaling to
**multi-robot / multi-machine real-robot RL** (treat single-arm as the ``n = 1``
case of an ``n``-robot fleet).

For cluster / ``node_groups`` / ``component_placement`` basics see
:doc:`realworld_robot`, :doc:`multi_node`, and :doc:`hetero`.


Why a separate backend
-----------------------

``franka_suite`` (a port of HIL-SERL's serl impedance stack to ROS 2 / FR3)
exposes the arm behind a **Flask HTTP API** on port ``5000``. Everything above
that API is ROS-free. RLinf consumes the robot through that single seam, so the
RLinf learner / env worker needs only ``requests`` — **no ROS in RLinf**.

::

   RLinf EnvWorker / FrankaEnv
       -> FrankaSuiteController (Ray actor, requests-only)
       -> HTTP  http://<robot-host>:5000   (franka_suite Flask server)
       -> ROS2  serl_franka_controllers_ros2 (1 kHz Cartesian impedance)
       -> libfranka -> FR3

The adapter ``rlinf/envs/realworld/franka/franka_suite_controller.py`` is a
drop-in alternative to ``FrankaController`` exposing the identical method surface
(``get_state`` / ``move_arm`` / ``reset_joint`` / ``clear_errors`` /
``reconfigure_compliance_params`` / gripper ops), so the existing ``FrankaEnv``,
``RealWorldEnv``, env worker, async SAC/RLPD pipeline, and configs are reused
unchanged. Selection is via one config field, ``controller_backend:
franka_suite``.


What runs where
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Node
     - What it runs
   * - **GPU host** (rank 0, head)
     - ``actor`` / ``rollout`` / ``reward`` (and optionally ``env``). **No ROS**;
       needs only ``requests`` on the RLinf side.
   * - **Robot host** (rank 1 … N)
     - The full ``franka_suite`` stack: ROS 2 + ``libfranka`` + the impedance
       controller + the Flask server, plus the RLinf ``FrankaSuiteController``
       Ray actor (which talks to ``localhost:5000``).

Bring-up order on the robot host — **all before** ``ray start`` (Ray freezes the
environment at start):

#. Unlock the FR3 and activate FCI in Desk; bring up the FCI network.
#. ``ros2 launch serl_franka_controllers_ros2 impedance.launch.py robot_type:=fr3 robot_ip:=<FCI_IP> load_gripper:=false``
#. ``ROBOT_IP=<FCI_IP> FLASK_URL=0.0.0.0 ./scripts/run_robot_server.sh``  (in the franka_suite repo)
#. Put ``franka_suite`` and RLinf on ``PYTHONPATH``, ``export RLINF_NODE_RANK=1``, then ``ray start --address=<head>:6379``.


Configuration
-------------

Select the backend in the env override and/or the Franka hardware config. See
``examples/embodiment/config/realworld_fr3_peginsertion_franka_suite.yaml``:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:   {node_group: "4090", placement: 0}
       rollout: {node_group: "4090", placement: 0}
       reward:  {node_group: "4090", placement: 0}
       env:     {node_group: franka, placement: 0}
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: <FCI_IP>        # informational; control is over HTTP
               node_rank: 1
               controller_backend: franka_suite
               franka_server_url: "http://127.0.0.1:5000"
               disable_validate: true     # franka_suite owns ping/camera validation

   env:
     train:
       override_cfg:
         controller_backend: franka_suite
         franka_server_url: "http://127.0.0.1:5000"


Bring-up checklist
------------------

#. **HTTP**: ``curl -X POST http://<host>:5000/getstate``.
#. **Adapter (Ray actor -> HTTP)**: ``FRANKA_SERVER_URL=http://<host>:5000 python toolkits/realworld_check/test_franka_suite_controller.py``.
#. **Cluster/placement**: validate with a dummy config before touching the arm.
#. **Single-arm RL**: ``bash examples/embodiment/run_realworld_async.sh realworld_fr3_peginsertion_franka_suite``.
#. **Cameras + reward**, then start learning.


Scaling to a fleet (multi-robot / multi-machine)
------------------------------------------------

Run **one franka_suite server per arm** (one robot host per arm is cleanest), add
one ``hardware.configs`` entry and one env worker per arm:

.. code-block:: yaml

   component_placement:
     env: {node_group: franka, placement: 0-1}   # one env worker per robot
   node_groups:
     - label: franka
       node_ranks: 1-2
       hardware:
         type: Franka
         configs:
           - {robot_ip: <FCI_IP_1>, node_rank: 1, controller_backend: franka_suite}
           - {robot_ip: <FCI_IP_2>, node_rank: 2, controller_backend: franka_suite}

Each robot becomes an independent physical agent with its own
``FrankaSuiteController``; episodes run independently and feed a shared replay
buffer. This is the path to MARL — the single-arm setup above is the ``n = 1``
special case.


Caveats specific to this backend
--------------------------------

- **Cartesian only.** The HTTP API offers Cartesian equilibrium control plus a
  one-shot ``/jointreset``; there is no streaming joint-space control, so the
  dual-arm joint-space env (``DualFrankaJointEnv``) does not apply here. FR3
  single-arm Cartesian (delta-pose + binary gripper) is the supported mode.
- **Joint reset config is server-side.** ``reset_joint`` ignores any joint target
  passed from RLinf; configure the reset configuration in ``franka_suite``.
- **Cameras are not in the HTTP API.** Run camera capture on the robot host
  (place ``env`` there, or reuse RLinf's RealSense driver) and keep
  ``disable_validate: true`` so RLinf does not contend with franka_suite over the
  camera / ping checks.
- **No auth on the HTTP seam.** Keep robot hosts on an isolated experiment LAN;
  co-locating the actor with the server (``localhost``) keeps the seam off the
  network entirely.
