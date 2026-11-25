#!/usr/bin/env python3
"""cuRobo worker: runs in ~/ip_ws/.venv_curobo (cuRobo + warp), talks to lib/planner_curobo.py over stdin/stdout, one JSON
object per line. Loads the robot (models/curobo_ur10/ur10_cell.yml, our UR10 + 2F-85 + D455), warms up once, then serves:
  {"cmd": "plan", "start": [6 rad], "kind": "pose"|"joints"|"line", "goal": 4x4 | [6 rad] | [4x4, ...]  (flange pose, our DH base)}
  -> {"ok": true, "q": [[6] ...], "dt": s, "solve_s": s} or {"ok": false, "reason": "..."}
cuRobo's base_link is our DH base turned 180 deg about z, so poses are flipped on the way in; joint values are the same.
"""
import json, os, sys, time, traceback
import numpy as np


def main():
    import yaml, torch
    from scipy.spatial.transform import Rotation as R
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo._src.types.robot import RobotCfg
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose
    rc = yaml.safe_load(open(sys.argv[1]))["robot_cfg"]
    from curobo.scene import Scene, Cuboid, Mesh
    world = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    # obstacles arrive in OUR base frame; cuRobo's base_link is that frame turned 180 deg about z: x, y negated, meshes rotated (w x y z = 0 0 0 1)
    scene = Scene(cuboid=[Cuboid(name=c["name"], dims=c["dims"], pose=[-c["centre"][0], -c["centre"][1], c["centre"][2], 1, 0, 0, 0])
                          for c in world.get("cuboids", [])],
                  mesh=[Mesh(name=m["name"], file_path=m["file"], pose=[0, 0, 0, 0, 0, 0, 1]) for m in world.get("meshes", [])]) if world else None
    mp = MotionPlanner(MotionPlannerCfg.create(robot=RobotCfg.create(rc, device_cfg=DeviceCfg()), scene_model=scene, use_cuda_graph=True))
    mp.warmup(enable_graph=True)
    names = rc["kinematics"]["cspace"]["joint_names"]
    flip = np.diag([-1, -1, 1, 1.0])
    frame = mp.tool_frames[0]

    def pose(T):
        T = flip @ np.asarray(T, float); q = R.from_matrix(T[:3, :3]).as_quat()          # x y z w -> w x y z
        return Pose.from_list([*T[:3, 3].tolist(), q[3], q[0], q[1], q[2]])

    def goal_of(Ts):
        """One goal (batch 1, horizon 1), or N waypoints (batch 1, horizon N) in cuRobo's [B, H, links, goalset, 3|4] layout."""
        P = [pose(T) for T in Ts]
        pos = torch.cat([p.position.reshape(1, 3) for p in P]).reshape(1, len(P), 1, 1, 3)
        quat = torch.cat([p.quaternion.reshape(1, 4) for p in P]).reshape(1, len(P), 1, 1, 4)
        return GoalToolPose(tool_frames=[frame], position=pos, quaternion=quat)

    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["cmd"] == "quit":
                break
            mp.reset_seed()                                  # no memory of the previous query (cuRobo's own benchmark does this too);
            #                                                  without it repeats of one query were answered ~15x faster than a cold plan
            start = JointState.from_position(mp.device_cfg.to_device([req["start"]]), joint_names=names)
            t0 = time.perf_counter()
            if req["kind"] == "line_ik":                   # IK (collision-aware) at each Cartesian waypoint, seeded from the previous one
                qs, q = [], np.asarray(req["start"], float)
                for k, T in enumerate(req["goal"]):
                    st = JointState.from_position(mp.device_cfg.to_device([q.tolist()]), joint_names=names)
                    ik = mp.ik_solver.solve_pose(goal_of([T]), current_state=st,
                                                 seed_config=mp.device_cfg.to_device([[q.tolist()]]), return_seeds=1)
                    if ik is None or not bool(ik.success.reshape(-1)[0].item()):
                        print(json.dumps({"ok": False, "reason": f"cuRobo IK found no collision-free solution at waypoint {k + 1} of {len(req['goal'])}",
                                          "solve_s": time.perf_counter() - t0}), flush=True)
                        break
                    q = ik.js_solution.position.cpu().numpy().reshape(-1)[:6].astype(float); qs.append(q.tolist())
                else:
                    print(json.dumps({"ok": True, "q": qs, "dt": 0.0, "solve_s": time.perf_counter() - t0}), flush=True)
                continue
            if req["kind"] == "joints":
                res = mp.plan_cspace(JointState.from_position(mp.device_cfg.to_device([req["goal"]]), joint_names=names), start, max_attempts=int(req.get("attempts", 10)))
            elif req["kind"] == "pose":
                res = mp.plan_pose(goal_of([req["goal"]]), start, max_attempts=int(req.get("attempts", 10)))
            else:
                res = mp.plan_pose_waypoints(goal_of(req["goal"]), start, path_tolerance=tuple(req.get("ptol", (0.001, 0.05))),
                                             final_tolerance=tuple(req.get("ftol", (0.0005, 0.02))),
                                             max_attempts=int(req.get("attempts", 10)),
                                             track_orientation=bool(req.get("track_orientation", True)))
            solve = time.perf_counter() - t0
            if res is None or not bool(res.success.item()):
                st = getattr(res, "status", None)
                print(json.dumps({"ok": False, "reason": f"cuRobo found no solution ({st})", "solve_s": solve}), flush=True)
                continue
            js = res.get_interpolated_plan()
            print(json.dumps({"ok": True, "q": js.position.cpu().numpy().reshape(-1, 6).tolist(),
                              "dt": float(js.dt.reshape(-1)[0]), "solve_s": solve}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "reason": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}), flush=True)


if __name__ == "__main__":
    main()
