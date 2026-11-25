"""cuRobo backend (NVIDIA cuRobo, GPU trajectory optimisation) behind the one planner interface (lib/planning.py).

cuRobo lives in its own environment (~/ip_ws/.venv_curobo) and needs CUDA_HOME for its kernel compiler, so it runs as a
worker process (lib/curobo_worker.py) started with the right environment and warmed up once; this class sends it JSON
requests. Connect/warm-up time is not counted in planning time, as for RoboDK. Planning time here includes the (local pipe)
round trip, about a millisecond; the worker's own solve time is kept in the result notes.
  free motion, pose goal   -> cuRobo plan_pose (collision-aware trajectory optimisation)
  free motion, joint goal  -> cuRobo plan_cspace
  linear motion            -> Cartesian waypoints every 1 cm / 2 deg (an exact line, as in the RoboDK backend), cuRobo's collision-aware
                              IK at each, seeded from the previous, then polished on our DH. (cuRobo's own waypoint-following
                              optimiser only succeeded with 5 mm / 2 mm tolerance on the pick's descent: operator chose the exact line.)
Collision: the robot (arm, 2F-85 open, D455 as spheres) against itself; no scene yet (same footing as the RoboDK station).
"""
import json, math, os, subprocess, sys
import numpy as np, cv2
import cell, kinematics as kin
from planning import Planner, PlanFailure

WS = cell.WS
PY = os.path.join(WS, ".venv_curobo", "bin", "python")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_worker.py")
ROBOT = os.path.join(WS, "models", "curobo_ur10", "ur10_cell.yml")


class CuroboPlanner(Planner):
    name = "curobo"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dh = cfg["robot"]["dh"]
        for p in (PY, ROBOT):
            if not os.path.exists(p):
                raise RuntimeError(f"cuRobo not set up: {p} is missing (see issues log PT-7)")
        env = dict(os.environ, CUDA_HOME=os.path.join(WS, ".cuda13"), CUDA_PATH=os.path.join(WS, ".cuda13"),
                   OMP_NUM_THREADS="3", TORCHINDUCTOR_COMPILE_THREADS="1")
        import json, scene
        sc = scene.load(cfg)          # the same obstacle list as MoveIt and RoboDK (cell.yaml scene:); meshes passed by file
        world = dict(cuboids=sc["cuboids"], meshes=[dict(name=m["name"], file=os.path.join(WS, c["file"]))
                                                    for m, c in zip(sc["meshes"], [x for x in (cfg.get("scene") or {}).get("meshes", []) if x.get("enabled", True)])])
        self.proc = subprocess.Popen(["nice", "-n", "10", PY, WORKER, ROBOT, json.dumps(world)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, env=env, cwd="/tmp")
        while True:                                            # warm-up: until the worker says ready (kernel compile on first use)
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("cuRobo worker exited during start-up (run lib/curobo_worker.py by hand to see why)")
            if line.startswith("{") and json.loads(line).get("ready"):
                break

    def polish(self, q, T_goal, iters=10):
        """Newton steps on OUR kinematics so the flange lands exactly on T_goal (same step as the RoboDK backend)."""
        def err(q):
            T = kin.fk(q, self.dh); dr, _ = cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)
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

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); self.proc.stdin.flush(); self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def __del__(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.close()

    def call(self, **req):
        self.proc.stdin.write(json.dumps(dict(cmd="plan", **req)) + "\n"); self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def plan_path(self, req):
        start = [float(v) for v in req.start]
        notes = dict(planner="cuRobo trajectory optimisation", collision="self + cell obstacles (cell.yaml scene)")
        if req.motion == "free":
            r = self.call(start=start, kind="joints", goal=[float(v) for v in req.goal["joints"]]) if "joints" in req.goal \
                else self.call(start=start, kind="pose", goal=np.asarray(req.goal["pose"], float).tolist())
        elif req.motion == "linear":
            T0 = kin.fk(start, self.dh)
            T1 = np.asarray(req.goal["pose"], float) if "pose" in req.goal else kin.fk(np.asarray(req.goal["joints"], float), self.dh)
            rvec, _ = cv2.Rodrigues(T0[:3, :3].T @ T1[:3, :3]); ang = float(np.linalg.norm(rvec))
            n = max(2, int(math.ceil(max(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / 0.01, ang / math.radians(2)))))
            wps = []
            for s in np.linspace(0, 1, n + 1)[1:]:
                T = np.eye(4); T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]; T[:3, :3] = T0[:3, :3] @ cv2.Rodrigues(rvec * s)[0]
                wps.append(T.tolist())
            notes["line_waypoints"] = n
            r = self.call(start=start, kind="line_ik", goal=wps)
            if r["ok"]:                                        # polish on our DH and check the line like the RoboDK backend does
                q, out = np.asarray(start, float), [np.asarray(start, float)]
                for k, (qk, Tk) in enumerate(zip(r["q"], wps)):
                    qk, ep, er = self.polish(np.asarray(qk, float), np.asarray(Tk, float))
                    if ep > 5e-5 or er > math.radians(0.01):
                        raise PlanFailure("no_ik", f"IK at waypoint {k + 1} would not polish to the line (error {ep * 1000:.2f} mm)")
                    if np.max(np.abs(qk - q)) > 0.2:
                        raise PlanFailure("unreachable", f"joint jump of {math.degrees(np.max(np.abs(qk - q))):.0f} deg at waypoint {k + 1} "
                                                         "(configuration flip or singularity)")
                    out.append(qk); q = qk
                notes["solve_s"] = r.get("solve_s")
                return out, notes
        else:
            raise PlanFailure("unsupported", f"motion '{req.motion}'")
        notes["solve_s"] = r.get("solve_s")
        if not r["ok"]:
            reason = r.get("reason", "")
            raise PlanFailure("collision" if "ollision" in reason else "backend_error" if "Error" in reason else "no_ik", reason)
        path = [np.asarray(q, float) for q in r["q"]]
        notes["samples"] = len(path); notes["worker_dt_s"] = r["dt"]
        return [start] + path[1:] if np.allclose(path[0], start, atol=1e-3) else [start] + path, notes
