"""The ONE planner interface (script 5's core). Every planner under test sits behind plan(); everything around it is
shared, so the benchmark compares planners and not plumbing (problem statement Figure 1: perception and execution
shared, only the planning layer swapped).

  request  = PlanRequest(start, goal, motion, scene, limits, timeout_s)
             start  : 6 joint angles (rad), where the arm is
             goal   : {"joints": [6 rad]}  or  {"pose": 4x4 base_T_tool0 (m)}   (robot BASE frame, UR convention)
             motion : "free"   = any collision-free path (joint-space move)
                      "linear" = the tool moves on a straight line (approach / lift)
  result   = PlanResult(success, failure, planning_time_s, path, trajectory, backend, notes)
             failure: None | "no_ik" | "unreachable" | "collision" | "timeout" | "backend_error" | "unsupported"

Shared for every backend:
  retime(path, limits)   one time-parameterisation (minimum-jerk along the joint path, scaled to the joint speed
                         limit), so durations and smoothness are comparable; the backend's own timing is kept in notes
  metrics(trajectory)    duration, joint path length, per-joint displacement, tool path length (UR10 DH), peak joint
                         speed / acceleration / jerk
  execute(trajectory)    streams /joint_command to Isaac at the trajectory's times, then checks arrival (lazy ROS import)

Backends register by name: get_planner("interim" | "robodk" | ...). Nothing here reads the Isaac scene.
"""
import time
from dataclasses import dataclass, field
import numpy as np
import kinematics as kin

FAILURES = ("no_ik", "unreachable", "collision", "timeout", "backend_error", "unsupported")


@dataclass
class PlanRequest:
    start: list
    goal: dict
    motion: str = "free"
    scene: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    timeout_s: float = 5.0


@dataclass
class PlanResult:
    success: bool
    backend: str
    failure: str = None
    planning_time_s: float = None
    path: list = None              # joint waypoints (rad) as the backend produced them
    trajectory: dict = None        # {"t": [s], "q": [[rad] x 6]} after the shared retiming
    notes: dict = field(default_factory=dict)


class Planner:
    """Base class: a backend implements plan_path(req) -> (path, notes) or raises PlanFailure."""
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    def plan_path(self, req):
        raise NotImplementedError

    def plan(self, req):
        t0 = time.perf_counter()
        try:
            path, notes = self.plan_path(req)
            dt = time.perf_counter() - t0
            if dt > req.timeout_s:
                return PlanResult(False, self.name, "timeout", dt, notes={"limit_s": req.timeout_s})
            traj = retime(path, req.limits or self.cfg.get("planner", {}).get("limits", {}))
            return PlanResult(True, self.name, None, dt, [list(map(float, q)) for q in path], traj, notes)
        except PlanFailure as e:
            return PlanResult(False, self.name, e.category, time.perf_counter() - t0, notes={"reason": str(e)})
        except Exception as e:
            return PlanResult(False, self.name, "backend_error", time.perf_counter() - t0, notes={"reason": f"{type(e).__name__}: {e}"})


class PlanFailure(Exception):
    def __init__(self, category, reason):
        assert category in FAILURES, category
        super().__init__(reason); self.category = category


# ---------- shared time-parameterisation ----------
def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def retime(path, limits, rate=60.0):
    """Minimum-jerk timing along the joint path's arc length (largest single-joint change per segment), duration set so
    the peak joint speed equals limits['max_joint_speed'] (minimum-jerk peak = 1.875 x mean). Zero speed and
    acceleration at both ends. Same for every backend."""
    P = np.asarray(path, float)
    if len(P) == 1:
        P = np.vstack([P, P])
    seg = np.max(np.abs(np.diff(P, axis=0)), axis=1); s_knots = np.concatenate([[0.0], np.cumsum(seg)]); total = s_knots[-1]
    vmax = float(limits.get("max_joint_speed", 0.5))
    T = max(float(limits.get("min_duration_s", 1.0)), 1.875 * total / vmax) if total > 0 else 0.0
    n = max(2, int(np.ceil(T * rate)) + 1)
    t = np.linspace(0.0, T, n)
    s = min_jerk(t / T if T > 0 else np.ones_like(t)) * total
    q = np.stack([np.interp(s, s_knots, P[:, j]) for j in range(P.shape[1])], axis=1)
    return {"t": t.round(5).tolist(), "q": q.round(6).tolist()}


# ---------- shared metrics ----------
def metrics(traj, dh=None):
    t = np.asarray(traj["t"], float); q = np.asarray(traj["q"], float)
    out = dict(duration_s=round(float(t[-1]), 3), joint_path_rad=round(float(np.sum(np.abs(np.diff(q, axis=0)))), 4),
               joint_displacement_rad=np.abs(q[-1] - q[0]).round(4).tolist())
    tool = np.array([kin.fk(qq, dh)[:3, 3] if dh else kin.fk(qq)[:3, 3] for qq in q])
    out["tool_path_m"] = round(float(np.sum(np.linalg.norm(np.diff(tool, axis=0), axis=1))), 4)
    if len(t) > 3 and t[-1] > 0:
        dt = np.diff(t).mean(); v = np.diff(q, axis=0) / dt; a = np.diff(v, axis=0) / dt; j = np.diff(a, axis=0) / dt
        out.update(peak_speed_rad_s=round(float(np.abs(v).max()), 4), peak_accel_rad_s2=round(float(np.abs(a).max()), 4),
                   peak_jerk_rad_s3=round(float(np.abs(j).max()), 3) if len(j) else 0.0)
    return out


def polish_ik(q, T_goal, dh, iters=10):
    """Newton steps on OUR kinematics so the flange lands exactly on T_goal: the same refinement every backend applies to its
    own IK solution (RoboDK's model is ~2 mm off ours; cuRobo / MoveIt stop at their solver tolerance). Returns (q, pos err m,
    rot err rad)."""
    import cv2

    def err(q):
        T = kin.fk(q, dh); dr, _ = cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)
        return np.r_[T_goal[:3, 3] - T[:3, 3], dr.ravel()]
    q = np.array(q, float)
    for _ in range(iters):
        e = err(q)
        if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-6:
            break
        J = np.zeros((6, 6)); h = 1e-6
        for i in range(6):
            dq = q.copy(); dq[i] += h; J[:, i] = (err(q) - err(dq)) / h
        q = q + np.linalg.lstsq(J, e, rcond=None)[0]
    e = err(q)
    return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


# ---------- shared executor ----------
def execute(traj, cfg, node_name="s5_execute"):
    """Stream the trajectory to /joint_command at its own times, then wait until the arm is still and compare."""
    import rclpy
    from sensor_msgs.msg import JointState
    import cell
    from ros_io import RobotIO
    rclpy.init(); io = RobotIO(node_name, command=True)
    try:
        if not io.wait_for(lambda: io.q is not None, 10):
            raise RuntimeError("no /joint_states: is Isaac playing on domain 33?")
        t = np.asarray(traj["t"]); q = np.asarray(traj["q"])
        start_err = float(np.max(np.abs(io.q - q[0])))
        if start_err > 0.02:
            raise RuntimeError(f"arm is {np.degrees(start_err):.1f} deg away from the trajectory start: replan from where it is")
        # contact stop: the arm falling this far behind the command means it is blocked; hold where it is instead of pushing.
        # Following error against the command of contact_lag_s ago (command -> joint feedback takes ~0.15-0.2 s here), else
        # a fast move trips it with no contact (2026-10-28: base joint 8.8 deg behind at 0.5 rad/s, nothing touched)
        pc = cfg.get("planner") or {}
        stop_at, lat = float(pc.get("contact_stop_rad", 0.15)), float(pc.get("contact_lag_s", 0.2))
        msg = JointState(); msg.name = cell.ARM_JOINTS; t0 = time.time(); trace = []; stopped = None
        for k in range(len(t)):
            while time.time() - t0 < t[k]:
                rclpy.spin_once(io, timeout_sec=0.0); time.sleep(0.002)
            tl = max(0.0, time.time() - t0 - lat)
            lag = io.q - np.array([np.interp(tl, t, q[:, i]) for i in range(q.shape[1])])
            trace.append([round(time.time() - t0, 3)] + q[k].round(5).tolist() + io.q.round(5).tolist())
            if np.max(np.abs(lag)) > stop_at:
                j = int(np.argmax(np.abs(lag)))
                stopped = f"contact stop at {time.time() - t0:.2f} s: {cell.ARM_JOINTS[j]} {np.degrees(lag[j]):+.1f} deg off the command"
                msg.position = io.q.tolist(); io.pub.publish(msg)
                break
            msg.position = q[k].tolist(); io.pub.publish(msg)
        still = io.wait_still(cfg["capture"]["max_joint_speed"])
        err = float(np.max(np.abs(io.q - q[-1])))
        return dict(arrived=bool(stopped is None and still and err < 0.002), max_error_rad=round(err, 5),
                    executed_s=round(time.time() - t0, 2), final_joints=io.q.round(5).tolist(), stopped=stopped,
                    trace_columns="t, commanded q x6, actual q x6", trace=trace)
    finally:
        rclpy.shutdown()


# ---------- backends ----------
class InterimPlanner(Planner):
    """Baseline: straight line in joint space, no IK, no collision checking (what s1 did). Joint goals only."""
    name = "interim"

    def plan_path(self, req):
        if "joints" not in req.goal:
            raise PlanFailure("unsupported", "interim planner has no IK: give a joint goal")
        if req.motion != "free":
            raise PlanFailure("unsupported", "interim planner only makes joint-space (free) moves")
        return [list(req.start), list(req.goal["joints"])], {"collision_checked": False}


def get_planner(name, cfg, remote=False):
    """remote=True: a RemotePlanner talking to the long-lived planner daemon (lib/planner_daemon.py), so a pick's seven steps do
    not each restart the planner (cuRobo: 36 s a time). The benchmark keeps remote=False (direct, nothing between)."""
    if remote and name != "interim":
        from planner_daemon import RemotePlanner
        return RemotePlanner(name, cfg)
    if name == "interim":
        return InterimPlanner(cfg)
    if name == "robodk":
        from planner_robodk import RoboDKPlanner
        return RoboDKPlanner(cfg)
    if name == "curobo":
        from planner_curobo import CuroboPlanner
        return CuroboPlanner(cfg)
    if name == "moveit":
        from planner_moveit import MoveItPlanner
        return MoveItPlanner(cfg)
    raise ValueError(f"unknown planner '{name}' (have: interim, robodk, curobo, moveit)")
