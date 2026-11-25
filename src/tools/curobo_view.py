#!/usr/bin/env python3
"""Watch a planned motion in a browser (cuRobo's viser viewer): the arm moves, with its collision spheres and the TCP path.

Two modes in one page: "live" mirrors the REAL arm from /joint_states (ROS domain 33, so run it with `sip` sourced) and
"follow" jumps to every new plan file as it is written (e.g. by a pick) and plays it. Untick both to browse old plans.

Replays the TIMED TRAJECTORY stored in any plan file (runs/*/5_plan_*.json: written by s5 for every backend, so cuRobo, MoveIt,
RoboDK and the interim planner can all be watched, and compared by loading two plans one after the other), or the frozen
benchmark queries planned live by cuRobo is not done here: plan first (s5 --planner curobo, or the benchmark), then watch.
Nothing is sent to the arm. The model is our UR10 + 2F-85 + D455 (models/curobo_ur10/ur10_cell.yml); the viewer shows cuRobo's
frame (its base_link is our DH base turned 180 deg about z), so the arm appears turned by 180 deg about the vertical axis
relative to the cell: the motion itself is identical.
Run in the cuRobo environment, then open http://127.0.0.1:8081 :

  CUDA_HOME=~/ip_ws/.cuda13 nice -n 19 ~/ip_ws/.venv_curobo/bin/python ~/ip_ws/src/tools/curobo_view.py [--port 8081] [plan.json ...]
  (no files: lists every plan in runs/ newest first)
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, torch, yaml


def plans(files):
    out = []
    for p in files or sorted(glob.glob(os.path.expanduser("~/ip_ws/runs/*/5_plan_*.json")), key=os.path.getmtime, reverse=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("success") and d.get("trajectory"):
            r = os.path.relpath(p, os.path.expanduser("~/ip_ws/runs"))
            out.append((f"{d['planner']:7s} {r}", p, d))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="*"); ap.add_argument("--port", type=int, default=8081)
    a = ap.parse_args()
    import cell, kinematics as kin
    from curobo.viewer import ViserVisualizer
    from curobo._src.state.state_joint import JointState
    P = plans(a.files)
    if not P:
        raise SystemExit("no successful plan files found (run s5 or the benchmark first)")
    rc = yaml.safe_load(open(os.path.join(cell.WS, "models", "curobo_ur10", "ur10_cell.yml")))
    viz = ViserVisualizer(content_path=rc, connect_ip="127.0.0.1", connect_port=a.port, add_control_frames=False,
                          visualize_robot_spheres=True, add_robot_to_scene=True)
    srv, dh, flip = viz._server, cell.load()["robot"]["dh"], np.diag([-1, -1, 1, 1.0])
    names = viz.joint_names
    import scene                                              # the cell's obstacles (cell.yaml scene:, the same list the planners use), in cuRobo's frame
    sc = scene.load(cell.load())
    for c in sc["cuboids"]:
        if c["name"] == "floor":
            continue                                          # 20 m slab: not drawn
        srv.scene.add_box(f"/obstacles/{c['name']}", color=(143, 167, 207), dimensions=tuple(c["dims"]), opacity=0.45,
                          position=(-c["centre"][0], -c["centre"][1], c["centre"][2]))
    for m in sc["meshes"]:
        V = m["vertices"] * np.array([-1.0, -1.0, 1.0])
        srv.scene.add_mesh_simple(f"/obstacles/{m['name']}", vertices=V.astype(np.float32), faces=m["faces"].astype(np.uint32),
                                  color=(143, 167, 207), opacity=0.45)
    labels = [lab for lab, _, _ in P]
    pick = srv.gui.add_dropdown("plan", labels, initial_value=labels[0])
    play = srv.gui.add_checkbox("play", True); loop = srv.gui.add_checkbox("loop", True)
    speed = srv.gui.add_slider("speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    when = srv.gui.add_slider("time (s)", min=0.0, max=1.0, step=0.01, initial_value=0.0)
    info = srv.gui.add_text("info", "", disabled=True)
    live = srv.gui.add_checkbox("live arm", True); follow = srv.gui.add_checkbox("follow new plans", True)
    state = dict(k=-1, t=0.0, traj=None, last=time.time(), q=None, seen=set(p for _, p, _ in P))
    try:                                                          # real arm: /joint_states in a background thread
        import rclpy, threading
        from sensor_msgs.msg import JointState as RosJS
        rclpy.init(); node = rclpy.create_node("curobo_view")
        arm = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        def on_js(m):
            d = dict(zip(m.name, m.position))
            if all(j in d for j in arm):
                state["q"] = [d[j] for j in arm]; state["q_t"] = time.time()
        node.create_subscription(RosJS, "/joint_states", on_js, 10)
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    except Exception as e:
        print(f"[curobo view] no live arm ({e}); plan replay only", flush=True); live.value = False

    def load():
        _, path, d = P[labels.index(pick.value)]
        tr = d["trajectory"]; t, q = np.asarray(tr["t"], float), np.asarray(tr["q"], float)
        tcp = np.array([(flip @ kin.fk(x, dh))[:3, 3] for x in q])          # TCP path in cuRobo's frame
        srv.scene.add_line_segments("/tcp_path", points=np.stack([tcp[:-1], tcp[1:]], 1), colors=np.array([[122, 162, 247]] * (len(tcp) - 1))[:, None, :].repeat(2, 1) / 255.0, line_width=3.0)
        m = d.get("metrics") or {}
        info.value = f"{d['planner']}  {t[-1]:.2f} s  tool path {m.get('tool_path_m', '-')} m  plan {1000 * (d.get('planning_time_s') or 0):.0f} ms"
        when.max = float(t[-1]); state.update(traj=(t, q), t=0.0)

    pick.on_update(lambda _: load()); load()
    print(f"[curobo view] http://127.0.0.1:{a.port}  ({len(P)} plans; pick one in the panel)", flush=True)
    last_scan = 0.0
    while True:
        now = time.time(); dt, state["last"] = now - state["last"], now
        if now - last_scan > 1.0:                                 # new plan files (every pick step writes one)
            last_scan = now
            fresh = [x for x in plans(None) if x[1] not in state["seen"]]
            for lab, path, d in reversed(fresh):
                state["seen"].add(path); P.insert(0, (lab, path, d)); labels.insert(0, lab)
            if fresh:
                pick.options = tuple(labels)
                if follow.value:
                    pick.value = labels[0]; play.value = True; live.value = False
        if live.value and state.get("q") is not None and now - state.get("q_t", 0) < 2.0:
            viz.set_joint_state(JointState.from_position(torch.tensor([state["q"]], dtype=torch.float32, device="cuda"), joint_names=names))
            time.sleep(1 / 30); continue
        t, q = state["traj"]
        if play.value:
            state["t"] += dt * speed.value
            if state["t"] > t[-1]:
                state["t"] = 0.0 if loop.value else t[-1]
            when.value = float(state["t"])
        else:
            state["t"] = float(when.value)
        qi = [np.interp(state["t"], t, q[:, j]) for j in range(6)]
        viz.set_joint_state(JointState.from_position(torch.tensor([qi], dtype=torch.float32, device="cuda"), joint_names=names))
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
