"""MoveIt 2 backend (move_group, OMPL RRTConnect + KDL IK) behind the one planner interface (lib/planning.py).

move_group runs as its own process in an isolated ROS domain (so it never meets Isaac's topics), with the configuration built
by tools/make_moveit_ur10.py (UR10 URDF + 2F-85 and D455 collision boxes). This class talks to it through its ROS services.
Connect (start-up of move_group) time is not counted in planning time, as for the other backends; planning time includes the
local service round trip.
  free motion, pose goal   -> /plan_kinematic_path (OMPL RRTConnect; pose constraint 1 mm / 1 mrad on tool0)
  free motion, joint goal  -> /plan_kinematic_path (joint constraints)
  linear motion            -> the same Cartesian steps as every backend (1 cm / 2 deg), MoveIt's own collision-aware IK
                              (/compute_ik, KDL) at each, seeded from the previous, then polished on our DH
Collision: the robot (arm, 2F-85 open, D455 as boxes) against itself and the cell obstacles (lib/scene.py, same list for every backend).
MoveIt's base_link is our DH base turned 180 deg about z: poses are flipped on the way in; joint values need no conversion.
"""
import math, os, signal, subprocess, time
import numpy as np, cv2
import cell, kinematics as kin
from planning import Planner, PlanFailure, polish_ik

DOMAIN = 71                       # scanned clean on this LAN (the zone stack uses others)
LAUNCH = os.path.join(cell.WS, "models", "moveit_ur10", "move_group.launch.py")
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
FLIP = np.diag([-1, -1, 1, 1.0])


class MoveItPlanner(Planner):
    name = "moveit"

    def __init__(self, cfg):
        super().__init__(cfg)
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from moveit_msgs.srv import GetMotionPlan, GetPositionIK
        self.dh = cfg["robot"]["dh"]
        if not os.path.exists(LAUNCH):
            raise RuntimeError(f"MoveIt not set up: {LAUNCH} is missing (run tools/make_moveit_ur10.py)")
        env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN))
        env.pop("ROS_LOCALHOST_ONLY", None)
        self.proc = subprocess.Popen(["nice", "-n", "10", "ros2", "launch", LAUNCH], env=env, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        self.ctx = rclpy.Context(); rclpy.init(context=self.ctx, domain_id=DOMAIN)     # own context: Isaac's domain stays untouched
        self.node = rclpy.create_node("moveit_backend", context=self.ctx)
        self.ex = SingleThreadedExecutor(context=self.ctx); self.ex.add_node(self.node)
        self.plan_cli = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.ik_cli = self.node.create_client(GetPositionIK, "/compute_ik")
        t0 = time.time()
        while not (self.plan_cli.service_is_ready() and self.ik_cli.service_is_ready()):
            self.ex.spin_once(timeout_sec=0.2)
            if time.time() - t0 > 90 or self.proc.poll() is not None:
                self.close(); raise RuntimeError("move_group did not come up (run `ros2 launch models/moveit_ur10/move_group.launch.py`)")
        self.add_scene(cfg)

    def add_scene(self, cfg):
        """The cell's obstacles (lib/scene.py) as collision objects in move_group's planning scene (frame base_link = ours turned
        180 deg about z: x, y negated). Applied through /apply_planning_scene; the planning scene monitor then uses them for planning and IK."""
        import scene
        from moveit_msgs.srv import ApplyPlanningScene
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive, Mesh as MeshMsg, MeshTriangle
        from geometry_msgs.msg import Pose, Point
        sc = scene.load(cfg); objs = []
        ident = Pose(); ident.orientation.w = 1.0
        for c in sc["cuboids"]:
            o = CollisionObject(); o.id = c["name"]; o.header.frame_id = "base_link"; o.operation = CollisionObject.ADD
            sp = SolidPrimitive(); sp.type = SolidPrimitive.BOX; sp.dimensions = [float(v) for v in c["dims"]]
            p = Pose(); p.position.x, p.position.y, p.position.z = -c["centre"][0], -c["centre"][1], c["centre"][2]; p.orientation.w = 1.0
            o.primitives = [sp]; o.primitive_poses = [p]; objs.append(o)
        for m in sc["meshes"]:
            o = CollisionObject(); o.id = m["name"]; o.header.frame_id = "base_link"; o.operation = CollisionObject.ADD
            mm = MeshMsg(); mm.vertices = [Point(x=-float(v[0]), y=-float(v[1]), z=float(v[2])) for v in m["vertices"]]
            mm.triangles = [MeshTriangle(vertex_indices=[int(i) for i in f]) for f in m["faces"]]
            o.meshes = [mm]; o.mesh_poses = [ident]; objs.append(o)
        cli = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        t0 = time.time()
        while not cli.service_is_ready():
            self.ex.spin_once(timeout_sec=0.2)
            if time.time() - t0 > 30:
                raise RuntimeError("move_group has no /apply_planning_scene service")
        rq = ApplyPlanningScene.Request(); rq.scene.is_diff = True; rq.scene.world.collision_objects = objs
        if not self._call(cli, rq, 10).success:
            raise RuntimeError("move_group refused the cell obstacles")
        self.n_obstacles = len(objs)

    def close(self):
        import rclpy
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT); self.proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            self.node.destroy_node(); rclpy.shutdown(context=self.ctx)
        except Exception:
            pass

    def __del__(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.close()

    def _call(self, cli, req, timeout):
        fut = cli.call_async(req); self.ex.spin_until_future_complete(fut, timeout_sec=timeout)
        if not fut.done():
            raise PlanFailure("timeout", "move_group did not answer in time")
        return fut.result()

    def _state(self, q):
        from moveit_msgs.msg import RobotState
        s = RobotState(); s.joint_state.name = ARM; s.joint_state.position = [float(v) for v in q]
        return s

    @staticmethod
    def _pose(T):
        from geometry_msgs.msg import Pose
        from scipy.spatial.transform import Rotation as R
        T = FLIP @ np.asarray(T, float); q = R.from_matrix(T[:3, :3]).as_quat()
        p = Pose(); p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
        return p

    def _valid(self, q):
        from moveit_msgs.srv import GetStateValidity
        if not hasattr(self, "valid_cli"):
            self.valid_cli = self.node.create_client(GetStateValidity, "/check_state_validity")
            while not self.valid_cli.service_is_ready():
                self.ex.spin_once(timeout_sec=0.1)
        rq = GetStateValidity.Request(); rq.robot_state = self._state(q); rq.group_name = "manipulator"
        return self._call(self.valid_cli, rq, 5).valid

    def _ik(self, T, seed):
        """Goal joints for a flange pose: the IK solution nearest the start (Newton from the start with its base swung towards the goal, on the UR10 kinematics, the
        same model as MoveIt's URDF), accepted if MoveIt's own collision check passes; otherwise MoveIt's KDL IK seeded from the
        start. KDL alone returned far, wrist-flipped solutions (2026-11-24: elbow pushed past its +-pi limit, OMPL error 99999)."""
        from moveit_msgs.srv import GetPositionIK
        seed = np.asarray(seed, float); T0 = kin.fk(seed, self.dh)
        q = seed.copy()                                              # seed: the start with its base swung towards the goal
        q[0] += math.atan2(T[1, 3], T[0, 3]) - math.atan2(T0[1, 3], T0[0, 3])
        q, ep, er = polish_ik(q, T, self.dh, iters=40)
        q = seed + (q - seed + np.pi) % (2 * np.pi) - np.pi; q[2] = (q[2] + np.pi) % (2 * np.pi) - np.pi
        if ep < 5e-5 and er < math.radians(0.01) and abs(q[2]) <= math.pi and self._valid(q):
            return q
        rq = GetPositionIK.Request(); ik = rq.ik_request
        ik.group_name = "manipulator"; ik.avoid_collisions = True; ik.robot_state = self._state(seed)
        ik.pose_stamped.header.frame_id = "world"; ik.pose_stamped.pose = self._pose(T); ik.ik_link_name = "tool0"
        ik.timeout.sec = 0; ik.timeout.nanosec = 200_000_000
        r = self._call(self.ik_cli, rq, 5)
        if r.error_code.val != 1:
            raise PlanFailure("collision" if r.error_code.val in (-12, -13, -14, -10) else "no_ik", f"MoveIt IK for the goal pose failed (error code {r.error_code.val})")
        names = list(r.solution.joint_state.name); pos = list(r.solution.joint_state.position)
        q, ep, er = polish_ik(np.array([pos[names.index(j)] for j in ARM]), T, self.dh)
        if ep > 5e-5 or er > math.radians(0.01):
            raise PlanFailure("no_ik", f"goal IK would not polish ({ep * 1000:.2f} mm)")
        w = seed + (q - seed + np.pi) % (2 * np.pi) - np.pi             # nearest turn of each joint, except the elbow (+-pi)
        w[2] = q[2]
        return w

    CATEGORY = {-31: "no_ik", -10: "collision", -12: "collision", -13: "collision", -14: "collision", -6: "timeout", -2: "unreachable",
                -1: "unreachable", -3: "unreachable", -4: "unreachable", -5: "unreachable"}

    def plan_path(self, req):
        from moveit_msgs.srv import GetMotionPlan, GetPositionIK
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint, BoundingVolume
        from shape_msgs.msg import SolidPrimitive
        start = np.asarray(req.start, float)
        notes = dict(planner="MoveIt 2 / OMPL RRTConnect + KDL IK", collision="self + cell obstacles (cell.yaml scene)")
        if req.motion == "free" and "pose" in req.goal:
            # pose goal -> joint goal with MoveIt's OWN IK seeded from the start (as RoboDK seeds its IK and cuRobo optimises
            # from the start). Without this, MoveIt's goal sampler took any IK solution: 2026-11-24 the move over the basket
            # ended at base 4.37 / shoulder 5.04 rad, a 56.7 rad, 74 s path that drove the arm into the scene.
            q_goal = self._ik(np.asarray(req.goal["pose"], float), start)
            notes["goal_ik"] = "MoveIt /compute_ik (KDL) seeded from the start, polished on our DH"
            req = type(req)(**{**req.__dict__, "goal": {"joints": q_goal.tolist()}})
        if req.motion == "free":
            m = GetMotionPlan.Request().motion_plan_request
            m.group_name = "manipulator"; m.pipeline_id = "ompl"; m.planner_id = "RRTConnectkConfigDefault"
            m.num_planning_attempts = 1; m.allowed_planning_time = float(req.timeout_s)
            m.max_velocity_scaling_factor = 1.0; m.max_acceleration_scaling_factor = 1.0
            m.start_state = self._state(start)
            c = Constraints()
            if "joints" in req.goal:
                for n, v in zip(ARM, req.goal["joints"]):
                    j = JointConstraint(); j.joint_name = n; j.position = float(v)
                    j.tolerance_above = j.tolerance_below = 1e-4; j.weight = 1.0; c.joint_constraints.append(j)
            else:
                pose = self._pose(req.goal["pose"])
                pc = PositionConstraint(); pc.header.frame_id = "base_link"; pc.link_name = "tool0"; pc.weight = 1.0
                sp = SolidPrimitive(); sp.type = SolidPrimitive.SPHERE; sp.dimensions = [0.001]
                pc.constraint_region = BoundingVolume(primitives=[sp], primitive_poses=[pose])
                oc = OrientationConstraint(); oc.header.frame_id = "base_link"; oc.link_name = "tool0"; oc.orientation = pose.orientation
                oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 1e-3; oc.weight = 1.0
                c.position_constraints.append(pc); c.orientation_constraints.append(oc)
            m.goal_constraints = [c]
            rq = GetMotionPlan.Request(); rq.motion_plan_request = m
            res = self._call(self.plan_cli, rq, req.timeout_s + 5).motion_plan_response
            if res.error_code.val != 1:
                raise PlanFailure(self.CATEGORY.get(res.error_code.val, "backend_error"), f"MoveIt error code {res.error_code.val}")
            pts = res.trajectory.joint_trajectory.points
            path = [np.asarray(p.positions, float) for p in pts]
            notes.update(moveit_planning_time_s=round(res.planning_time, 5), waypoints=len(path))
            return [start] + path[1:] if path and np.allclose(path[0], start, atol=1e-3) else [start] + path, notes
        if req.motion == "linear":
            T0 = kin.fk(start, self.dh)
            T1 = np.asarray(req.goal["pose"], float) if "pose" in req.goal else kin.fk(np.asarray(req.goal["joints"], float), self.dh)
            rvec, _ = cv2.Rodrigues(T0[:3, :3].T @ T1[:3, :3]); ang = float(np.linalg.norm(rvec))
            n = max(2, int(math.ceil(max(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / 0.01, ang / math.radians(2)))))
            out, q = [start], start
            for k, s in enumerate(np.linspace(0, 1, n + 1)[1:]):
                T = np.eye(4); T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]; T[:3, :3] = T0[:3, :3] @ cv2.Rodrigues(rvec * s)[0]
                rq = GetPositionIK.Request(); ik = rq.ik_request
                ik.group_name = "manipulator"; ik.avoid_collisions = True; ik.robot_state = self._state(q)
                ik.pose_stamped.header.frame_id = "world"; ik.pose_stamped.pose = self._pose(T); ik.ik_link_name = "tool0"   # IK wants the planning frame; world == base_link here
                ik.timeout.sec = 0; ik.timeout.nanosec = 100_000_000
                r = self._call(self.ik_cli, rq, 5)
                if r.error_code.val != 1:
                    raise PlanFailure("collision" if r.error_code.val in (-12, -13, -14, -10) else "no_ik",
                                      f"MoveIt IK failed at waypoint {k + 1} of {n} (error code {r.error_code.val})")
                names = list(r.solution.joint_state.name); pos = list(r.solution.joint_state.position)
                qk = np.array([pos[names.index(j)] for j in ARM])
                qk, ep, er = polish_ik(qk, T, self.dh)
                if ep > 5e-5 or er > math.radians(0.01):
                    raise PlanFailure("no_ik", f"IK at waypoint {k + 1} would not polish to the line (error {ep * 1000:.2f} mm)")
                if np.max(np.abs(qk - q)) > 0.2:
                    raise PlanFailure("unreachable", f"joint jump of {math.degrees(np.max(np.abs(qk - q))):.0f} deg at waypoint {k + 1} "
                                                     "(configuration flip or singularity)")
                out.append(qk); q = qk
            notes["line_waypoints"] = n
            return out, notes
        raise PlanFailure("unsupported", f"motion '{req.motion}'")
