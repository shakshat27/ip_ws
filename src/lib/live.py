"""Live view for s0 (--ui): camera + joints in, detection + localisation several times a second, a web page out.

Real-world inputs only, exactly as a real cell: /rgb, /depth, /camera_info, /joint_states. Each cycle takes the
newest colour + depth pair (timestamps within capture.max_pair_dt), the joint angles recorded at the image time,
runs s3's detector (cell.yaml detector: model, two-pass, depth check) and s4's measurement, and draws the result.
Commands to the cell's test fixtures go out as ROS topics (spawner: /spawn_boxes, std_msgs/String).

Served with the standard library only (no web framework):
  GET  /             the page (src/ui/s0.html)
  GET  /stream.mjpg  annotated camera feed (MJPEG)
  GET  /state        JSON: link health, timings, detections
  POST /spawn        {"count": 3, "colour": "random"}  -> /spawn_boxes "3 random"
  POST /clear        -> /spawn_boxes "clear"
  POST /view         {"paused": true|false, "two_pass": true|false}   (runtime only; cell.yaml untouched)
  POST /snapshot     the newest frame -> runs/<stamp>_snapshot/2_capture_live/ (same files as s2, so s3/s4 rerun on it)
  POST /move         {"pose": "pick_overview", "planner": "robodk"} -> s5_planner.py --execute (one arm job at a time)
  POST /task         {"task": "pick", "planner": "robodk"[, "x": .., "y": ..]} -> s0_task_manager.py --task pick --then return
                     [--near x y: that object only] (moves the arm; one job at a time)
  POST /jog          {"joint": 1..6, "step": deg} or {"axis": "x".."rz", "step": mm|deg, "frame": "base"|"tool"} -> tools/jog.py
  POST /teach        {"name": "drop_overview"} -> tools/teach_pose.py: current joints into cell.yaml capture_poses.<name>
"""
import json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import geometry as geo, kinematics as kin
from ros_io import RobotIO, stamp_s
import run_io, cell

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "s0.html")
PIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
PLANNERS = ["interim", "robodk", "moveit", "curobo"]   # backends behind s5's one interface (lib/planning.py)
COLOURS = {"normal": (247, 162, 122), "inverted": (207, 167, 143), "depth": (247, 162, 122)}   # BGR: accent, steel blue


class Live:
    def __init__(self, cfg, period=0.2):
        import s3_detect, s4_localize                       # pipeline modules (torch only loads here)
        self.s3, self.s4, self.cfg, self.period = s3_detect, s4_localize, cfg, period
        self.model = s3_detect.load_model(cfg)
        rclpy.init(); self.io = RobotIO("s0_live", camera=True)
        self.spawn_pub = self.io.create_publisher(String, "/spawn_boxes", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.run_pub = self.io.create_publisher(Bool, "/plc/run_enable", latched)      # operator tag of the mini PLC (tools/mini_plc.py)
        self.io.create_subscription(String, "/plc/state", self.on_plc, latched)
        self.ex = SingleThreadedExecutor(); self.ex.add_node(self.io)
        threading.Thread(target=self.ex.spin, daemon=True).start()
        self.lock = threading.Condition(); self.jpeg = None; self.seq = 0
        self.state = dict(camera=False, joints=False, frame_age=None, fps=0.0, detect_ms=None, detections=[], joints_source=None,
                          model=os.path.basename(os.path.dirname(os.path.expanduser(cfg["detector"]["model"]))),
                          two_pass=bool(cfg["detector"].get("two_pass")), paused=False, last_command=None, error=None,
                          poses=[],
                          jobs=[], last_snapshot=None, planners=PLANNERS, planner=(cfg.get("planner") or {}).get("backend", "interim"),
                          q_deg=None, tcp=None, belt=None)
        self.state["poses"] = self.poses()
        self.last_stamp = None; self.times = []; self.frame = None; self.job_lock = threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def on_plc(self, msg):
        try:
            p = json.loads(msg.data); self.state["belt"] = "run" if p.get("belt_running") else ("halt" if p.get("halted") else "stopping")
            self.plc_t = time.time()
        except Exception:
            pass

    # ---------- detection loop ----------
    def loop(self):
        while True:
            t_cycle = time.time()
            try:
                self.cycle()
                self.state["error"] = None
            except Exception as e:                        # keep the page alive; show the cause
                self.state["error"] = f"{type(e).__name__}: {e}"
            time.sleep(max(0.0, self.period - (time.time() - t_cycle)))

    def poses(self):
        return [k for grp in ("named_poses", "capture_poses") for k, v in (self.cfg.get(grp) or {}).items() if v and v.get("joints")]

    def cycle(self):
        io, st = self.io, self.state
        st["joints"] = io.q is not None
        if io.q is not None:                              # for the jog panel: joints and TCP (FK x tool0_T_tcp)
            T = kin.fk(io.q, self.cfg["robot"]["dh"]) @ kin.T_from_cfg(self.cfg["gripper"]["tool0_T_tcp"])
            st["q_deg"] = [round(float(np.degrees(v)), 2) for v in io.q]
            st["tcp"] = [round(float(v) * 1000, 1) for v in T[:3, 3]] + [round(float(v), 2) for v in np.degrees(kin.R_to_rpy(T[:3, :3]))]
        pair = io.fresh_pair(-1e18, self.cfg["capture"]["max_pair_dt"]) if io.info is not None else None
        st["camera"] = pair is not None
        if pair is None:
            return
        d, rgb_msg = pair; t_img = stamp_s(d.header.stamp)
        st["frame_age"] = round(time.time() - io.depth_arrival, 2)
        if t_img == self.last_stamp:
            return                                        # no new frame since the last cycle
        self.last_stamp = t_img
        depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width).copy()
        rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)[..., :3].copy()
        K = np.array(io.info.k).reshape(3, 3)
        if io.q is None:
            return
        src, q = io.joints_at(d.header.stamp, self.cfg["capture"]["max_pose_age"]); st["joints_source"] = src
        self.frame = dict(rgb=rgb, depth=depth, K=K, q=[float(v) for v in q], src=src, stamp=t_img,
                          rgb_stamp=stamp_s(rgb_msg.header.stamp), info=io.info)
        if st["paused"]:                                  # feed stays live, no detection
            self.publish(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); st.update(detections=[], detect_ms=None)
            return
        t0 = time.perf_counter()
        found = self.s3.detect_rfdetr(rgb, depth, K, q, self.cfg, self.model, two_pass=st["two_pass"])
        pts, valid = geo.backproject(depth, K, near=self.cfg["detection"]["near_cutoff"])
        B = geo.to_frame(pts, geo.base_T_camera(q, self.cfg))
        rows, img = [], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for i, f in enumerate(found):
            P = B[f["mask"] & valid]
            if not f["surface"] or len(P) < 20:
                continue
            m = self.s4.measure(P, self.cfg["surfaces"][f["surface"]]["centre"][2])
            fit = self.s4.match_type(m, self.cfg["objects"])
            size_ok = bool(fit) and fit[0] == f["cls"]
            rows.append(dict(id=i, cls=f["cls"], score=round(f["score"], 2), source=f.get("source", "normal"),
                             x=round(m["position"][0], 4), y=round(m["position"][1], 4), z=round(m["position"][2], 4),
                             yaw=round(m["yaw_deg"], 1), size_mm=[round(v * 1000) for v in (*m["footprint"], m["height"])], size_ok=size_ok))
            col = COLOURS.get(f.get("source", "normal"), COLOURS["normal"])
            cnts, _ = cv2.findContours(f["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 2, cv2.LINE_AA)
            x, y, w, h = f["bbox"]
            for k, line in enumerate((f"#{i} {f['cls']} {f['score']:.2f}", f"x {m['position'][0]:+.3f} y {m['position'][1]:+.3f} yaw {m['yaw_deg']:.0f}")):
                org = (x, max(14, y - 22 + 16 * k))
                cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (14, 15, 17), 3, cv2.LINE_AA)
                cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (233, 231, 230), 1, cv2.LINE_AA)
        ms = (time.perf_counter() - t0) * 1000
        now = time.time(); self.times = [t for t in self.times if now - t < 3] + [now]
        st.update(detect_ms=round(ms), fps=round(len(self.times) / 3, 1), detections=rows)
        self.publish(img)

    def publish(self, img):
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.jpeg, self.seq = jpg.tobytes(), self.seq + 1
            self.lock.notify_all()

    def command(self, text):
        self.spawn_pub.publish(String(data=text)); self.state["last_command"] = text

    # ---------- snapshot: the newest frame saved exactly as s2 saves a capture ----------
    def snapshot(self):
        f = self.frame
        if f is None:
            raise RuntimeError("no frame yet")
        run = run_io.new_run("snapshot"); out = run_io.step_dir(run, "2_capture_live")
        np.save(os.path.join(out, "depth.npy"), f["depth"])
        cv2.imwrite(os.path.join(out, "rgb.png"), cv2.cvtColor(f["rgb"], cv2.COLOR_RGB2BGR))
        info = f["info"]
        run_io.write_json(os.path.join(out, "camera_info.json"), dict(width=info.width, height=info.height, K=f["K"],
                          D=list(info.d), distortion_model=info.distortion_model, frame_id=info.header.frame_id))
        run_io.write_json(os.path.join(out, "capture.json"), dict(label="live", image_stamp=f["stamp"], rgb_stamp=f["rgb_stamp"],
                          joints=f["q"], joints_source=f["src"], wall_time=time.strftime("%Y-%m-%d %H:%M:%S")))
        self.state["last_snapshot"] = os.path.basename(run)
        return os.path.basename(run)

    # ---------- jobs: arm moves and tasks as separate processes, one at a time ----------
    def job(self, name, args, on_done=None):
        with self.job_lock:
            if any(j["status"] == "running" for j in self.state["jobs"]):
                raise RuntimeError("busy: another job is running")
            j = dict(id=len(self.state["jobs"]) + 1, name=name, status="running", started=time.strftime("%H:%M:%S"), seconds=0.0, last="")
            self.state["jobs"] = (self.state["jobs"] + [j])[-8:]
        threading.Thread(target=self._run_job, args=(j, args, on_done), daemon=True).start()

    def _run_job(self, j, args, on_done=None):
        t0 = time.time()
        try:
            p = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                line = line.strip()
                if line and "warning" not in line.lower() and not line.startswith(("->", "[")):
                    j["last"] = line[-160:]
                j["seconds"] = round(time.time() - t0, 1)
            j["status"] = "done" if p.wait() == 0 else "failed"
            if on_done and j["status"] == "done":
                on_done()
        except Exception as e:
            j["status"], j["last"] = "failed", f"{type(e).__name__}: {e}"
        j["seconds"] = round(time.time() - t0, 1)

    def reload(self):
        """cell.yaml changed (a pose taught): pick up the new poses for the Arm buttons and /move."""
        self.cfg = cell.load(); self.state["poses"] = self.poses()


def serve(cfg, port=8090, host="127.0.0.1"):
    live = Live(cfg)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, open(UI, "rb").read(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self.send(200, json.dumps(live.state).encode(), "application/json")
            elif self.path == "/stream.mjpg":
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store"); self.end_headers(); seen = -1
                try:
                    while True:
                        with live.lock:
                            live.lock.wait_for(lambda: live.seq != seen, timeout=5)
                            jpg, seen = live.jpeg, live.seq
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            try:
                req = json.loads(body)
                if self.path == "/spawn":
                    n = int(req.get("count", 1)); c = str(req.get("colour", "random"))
                    live.command(f"{n} {c}"); self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/belt":
                    live.run_pub.publish(Bool(data=bool(req.get("run")))); self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/clear":
                    live.command("clear"); self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/view":
                    for k in ("paused", "two_pass"):
                        if k in req:
                            live.state[k] = bool(req[k])
                    self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/snapshot":
                    run = live.snapshot(); self.send(200, json.dumps(dict(ok=True, run=run)).encode(), "application/json")
                elif self.path == "/move":
                    pose = str(req.get("pose")); planner = str(req.get("planner") or live.state["planner"])
                    if pose not in live.state["poses"]:
                        raise ValueError(f"unknown pose {pose}")
                    if planner not in PLANNERS:
                        raise ValueError(f"unknown planner {planner}")
                    live.state["planner"] = planner
                    live.job(f"{pose} ({planner})", [os.path.join(PIPE, "s5_planner.py"), "--pose", pose, "--planner", planner, "--execute"])
                    self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/task":
                    if req.get("task") != "pick":
                        raise ValueError("only task 'pick' exists")
                    planner = str(req.get("planner") or live.state["planner"])
                    if planner not in PLANNERS:
                        raise ValueError(f"unknown planner {planner}")
                    live.state["planner"] = planner
                    args = [os.path.join(PIPE, "s0_task_manager.py"), "--task", "pick", "--type", "box",
                            "--method", "rfdetr", "--planner", planner, "--then", "place", "--watch"]
                    label = f"pick ({planner})"
                    if req.get("x") is not None and req.get("y") is not None:      # a row's pick button: that object only
                        x, y = float(req["x"]), float(req["y"])
                        args.remove("--watch")                                     # a chosen object: pick it where it is now
                        args += ["--near", f"{x:.4f}", f"{y:.4f}"]; label = f"pick {x:+.3f} {y:+.3f} ({planner})"
                    live.job(label, args)
                    self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/jog":
                    step = float(req["step"])
                    if "joint" in req:
                        jn = int(req["joint"])
                        if not 1 <= jn <= 6:
                            raise ValueError("joint 1..6")
                        args, label = ["--joint", str(jn), "--step", f"{step:g}"], f"jog J{jn} {step:+g}"
                    else:
                        ax, fr = str(req["axis"]), str(req.get("frame", "base"))
                        if ax not in ("x", "y", "z", "rx", "ry", "rz") or fr not in ("base", "tool"):
                            raise ValueError("axis x y z rx ry rz, frame base|tool")
                        args, label = ["--axis", ax, "--step", f"{step:g}", "--frame", fr], f"jog {ax} {step:+g} {fr}"
                    live.job(label, [os.path.join(TOOLS, "jog.py"), *args])
                    self.send(200, b'{"ok":true}', "application/json")
                elif self.path == "/teach":
                    name = str(req.get("name", "")).strip()
                    args = [os.path.join(TOOLS, "teach_pose.py"), "--name", name]
                    for k in ("looks_at", "purpose"):
                        if req.get(k):
                            args += [f"--{k.replace('_', '-')}", str(req[k])]
                    live.job(f"teach {name}", args, on_done=live.reload)
                    self.send(200, b'{"ok":true}', "application/json")
                else:
                    self.send(404, b'{"ok":false}', "application/json")
            except Exception as e:
                self.send(400, json.dumps(dict(ok=False, error=str(e))).encode(), "application/json")

    srv = ThreadingHTTPServer((host, port), H); srv.daemon_threads = True
    print(f"[s0 ui] http://{host}:{port}  (camera /rgb /depth, joints /joint_states; spawner /spawn_boxes)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close(); rclpy.shutdown()
