"""RoboDK backend for the planner interface (planning.py). RoboDK is reached through its own Python API
(~/RoboDK/Python, socket on cell.yaml planner.robodk.port), NOT through the Isaac bridge: nothing comes from the
Isaac scene. The RoboDK station must hold a UR10 whose base IS the UR base frame (our robot base frame).

Kinematics: OUR UR10 DH (kinematics.fk, cell.yaml robot.dh) is the truth (it matches the simulated robot to 0.00 mm, and
on a real cell it is that robot's calibration). RoboDK's UR10 model differs slightly (measured 2026-10-23: d1 128.0,
a2 612.7, a3 571.6, d4 163.9 mm vs our 127.3 / 612.0 / 572.3 / 163.941: up to 2.1 mm at the flange). So every RoboDK
IK solution is POLISHED with our kinematics (Newton steps to < 0.05 mm, < 0.01 deg): RoboDK chooses the arm
configuration and the path, our kinematics fix where the flange ends up. On connect the model difference is measured
and recorded; above 5 mm the station's robot or base frame is wrong and nothing is planned.

  free   : goal joints (given, or RoboDK SolveIK for a pose goal, seeded with the start) -> joint-space move;
           RoboDK MoveJ_Test reports collisions when the station has collision checking on
  linear : straight tool line: poses interpolated start -> goal (position linear, rotation along the shortest
           arc), RoboDK SolveIK at each step seeded with the previous joints; a joint jump > 0.2 rad between steps
           (configuration flip / singularity) = unreachable. RoboDK MoveL_Test reports the same check natively.
Units: RoboDK speaks mm and degrees; this file converts at the boundary, everything outside is m and rad.
RoboDK is not a collision-aware sampling planner by default: report it as partly a tool comparison.
Obstacles: the cell's shared list (lib/scene.py) is added to the station as `obs_*` shapes when collision_check is on.
"""
import math, os, sys
import numpy as np
import cv2
import kinematics as kin
from planning import Planner, PlanFailure

ROBODK_PY = os.path.expanduser("~/RoboDK/Python")
CHECK_JOINTS = [[0, -90, 90, -90, -90, 0], [90, -112.5, 122.9, -100.3, -90, 0], [30, -60, 80, -110, -70, 45], [-45, -100, 60, -50, 90, -30]]


class RoboDKPlanner(Planner):
    name = "robodk"

    def __init__(self, cfg):
        super().__init__(cfg)
        if ROBODK_PY not in sys.path:
            sys.path.insert(0, ROBODK_PY)
        from robodk import robolink, robomath
        self.rl, self.rm = robolink, robomath
        c = (cfg.get("planner") or {}).get("robodk") or {}
        self.rdk, self.port = self.connect(c)
        self.robot = self.rdk.Item(c.get("robot") or "", robolink.ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise RuntimeError(f"no robot '{c.get('robot') or '(first)'}' in the open RoboDK station")
        lims = self.robot.JointLimits()                       # RoboDK 6.0.6 returns (lower, upper, joint type)
        self.lo, self.hi = np.radians(np.array(lims[0].list()))[:6], np.radians(np.array(lims[1].list()))[:6]
        self.collision_check = bool(c.get("collision_check", False))
        self.fk_check = self.self_check()
        self.n_obstacles = self.add_scene(cfg) if self.collision_check else 0

    def connect(self, c):
        """The running RoboDK that has our station open. RoboDK takes the next free API port when 20500 is busy at start
        (seen 2026-10-26: 20501), and Robolink() STARTS a new RoboDK when nothing answers, so ports are probed with a
        plain socket first and nothing is ever launched from here."""
        import socket
        want = os.path.splitext(os.path.basename(os.path.expanduser(c.get("station") or "")))[0]
        first = int(c.get("port", 20500)); seen = []
        for port in [first] + [p for p in range(20500, 20511) if p != first]:
            with socket.socket() as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    continue
            rdk = self.rl.Robolink(port=port)
            st = rdk.ActiveStation(); name = st.Name() if st.Valid() else None
            seen.append(f"{port}: {name}")
            if not want or name == want:
                return rdk, port
        raise RuntimeError(f"no RoboDK with station '{want}' open on ports 20500-20510 (found: {seen or 'none'}); "
                           f"start RoboDK and open {c.get('station')}")

    def add_scene(self, cfg):
        """The cell's obstacles (lib/scene.py, ROBOT BASE frame; the station origin IS the UR base frame) as shapes in the open
        station, items named `obs_<name>` (replaced on every connect, so the station never collects duplicates), mm at the boundary.
        RoboDK's MoveJ_Test / MoveL_Test then count robot-vs-obstacle collisions with the other checks."""
        import scene
        sc = scene.load(cfg)
        for it in self.rdk.ItemList(self.rl.ITEM_TYPE_OBJECT):
            if it.Name().startswith("obs_"):
                it.Delete()
        n = 0
        for c in sc["cuboids"]:
            tri = scene.box_triangles(c["centre"], c["dims"]) * 1000.0
            n += self._shape("obs_" + c["name"], tri)
        for m in sc["meshes"]:
            n += self._shape("obs_" + m["name"], m["vertices"][m["faces"].reshape(-1)] * 1000.0)
        return n

    def _shape(self, name, tri_mm):
        item = self.rdk.AddShape(self.rm.Mat(np.asarray(tri_mm, float).T.tolist()))
        item.setName(name); item.setColor([0.5, 0.55, 0.65, 0.35])
        return 1

    # ---------- unit / frame conversion at the boundary ----------
    def to_mat(self, T):
        M = self.rm.Mat(np.asarray(T, float).tolist())
        M.setPos([v * 1000.0 for v in np.asarray(T)[:3, 3]])
        return M

    def from_mat(self, M):
        T = np.array(M.rows, float)
        T[:3, 3] /= 1000.0
        return T

    def self_check(self):
        worst = 0.0
        for jd in CHECK_JOINTS:
            ours = kin.fk(np.radians(jd), self.dh); theirs = self.from_mat(self.robot.SolveFK(jd))
            worst = max(worst, float(np.linalg.norm(ours[:3, 3] - theirs[:3, 3])))
        if worst > 0.005:
            raise RuntimeError(f"RoboDK flange FK differs from our UR10 kinematics by up to {worst * 1000:.1f} mm: the station's "
                               "robot base must be the UR base frame and the robot a UR10")
        return round(worst * 1000, 3)

    @property
    def dh(self):
        return (self.cfg.get("robot") or {}).get("dh") or kin.UR10_DH

    @staticmethod
    def pose_error(T, T_goal):
        """6-vector: position error (m) and rotation error (axis * angle, rad), both in the base frame."""
        w, _ = cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)
        return np.concatenate([T_goal[:3, 3] - T[:3, 3], w.ravel()])

    def polish(self, q, T_goal, iters=10):
        """Newton steps with OUR kinematics so the flange lands exactly on T_goal (RoboDK's model is ~2 mm off)."""
        q = np.array(q, float)
        for _ in range(iters):
            e = self.pose_error(kin.fk(q, self.dh), T_goal)
            if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-6:
                break
            J = np.zeros((6, 6)); h = 1e-6
            for i in range(6):
                dq = q.copy(); dq[i] += h
                J[:, i] = (self.pose_error(kin.fk(q, self.dh), T_goal) - self.pose_error(kin.fk(dq, self.dh), T_goal)) / h
            q = q + np.linalg.lstsq(J, e, rcond=None)[0]
        e = self.pose_error(kin.fk(q, self.dh), T_goal)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def ik(self, T_tool0, seed):
        q = self.robot.SolveIK(self.to_mat(T_tool0), np.degrees(seed).tolist())
        q = np.radians(np.array(q.list() if hasattr(q, "list") else q, float))
        if len(q) < 6:
            return None
        q, ep, er = self.polish(q[:6], T_tool0)
        if ep > 5e-5 or er > np.radians(0.01):
            return None
        if np.any(q < self.lo - 1e-6) or np.any(q > self.hi + 1e-6):
            return None
        return q

    # ---------- planning ----------
    def plan_path(self, req):
        # RoboDK's MoveJ_Test / MoveL_Test leave its robot at the end of the tested move (seen 2026-10-26): put it back,
        # so the station stays as the operator left it
        j0 = self.robot.Joints()
        try:
            return self._plan_path(req)
        finally:
            self.robot.setJoints(j0)

    def _plan_path(self, req):
        start = np.asarray(req.start, float)
        notes = dict(robodk_model_offset_mm=self.fk_check, ik_polished_with="our UR10 DH", collision_checked=self.collision_check)
        if req.motion == "free":
            if "joints" in req.goal:
                goal = np.asarray(req.goal["joints"], float)
            else:
                goal = self.ik(np.asarray(req.goal["pose"], float), start)
                if goal is None:
                    raise PlanFailure("no_ik", "RoboDK SolveIK found no solution within the joint limits")
            if self.collision_check:
                hits = self.robot.MoveJ_Test(np.degrees(start).tolist(), np.degrees(goal).tolist())
                notes["movej_test"] = int(hits)
                if hits > 0:
                    raise PlanFailure("collision", f"RoboDK MoveJ_Test: {hits} collision(s) on the joint move")
            return [start, goal], notes
        if req.motion == "linear":
            T0 = kin.fk(start, self.dh)
            T1 = np.asarray(req.goal["pose"], float) if "pose" in req.goal else kin.fk(np.asarray(req.goal["joints"], float), self.dh)
            if self.collision_check:   # > 0 = colliding pairs (fail); < 0 = RoboDK's own model can't reach (note only:
                hits = int(self.robot.MoveL_Test(np.degrees(start).tolist(), self.to_mat(T1)))   # our IK below decides)
                notes["movel_test"] = hits
                if hits > 0:
                    raise PlanFailure("collision", f"RoboDK MoveL_Test: {hits} collision(s) on the straight line")
            rvec, _ = cv2.Rodrigues(T0[:3, :3].T @ T1[:3, :3]); ang = float(np.linalg.norm(rvec))
            n = max(2, int(math.ceil(max(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / 0.01, ang / math.radians(2)))) + 1)
            path, q = [start], start
            for s in np.linspace(0, 1, n)[1:]:
                T = np.eye(4); T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
                T[:3, :3] = T0[:3, :3] @ cv2.Rodrigues(rvec * s)[0]
                qn = self.ik(T, q)
                if qn is None:
                    raise PlanFailure("no_ik", f"no IK at {s * 100:.0f} % of the straight line")
                if np.max(np.abs(qn - q)) > 0.2:
                    raise PlanFailure("unreachable", f"joint jump of {np.degrees(np.max(np.abs(qn - q))):.0f} deg at {s * 100:.0f} % "
                                                     "of the line (configuration flip or singularity)")
                path.append(qn); q = qn
            notes["line_steps"] = n - 1
            return path, notes
        raise PlanFailure("unsupported", f"motion '{req.motion}'")
