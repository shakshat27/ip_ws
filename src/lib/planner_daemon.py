"""Planner daemon: keeps each planner ALIVE between pick steps.

Every pick step is its own process (s5_planner.py), and each used to build its planner from scratch: cuRobo's worker takes
~36 s to start, MoveIt's move_group ~0.7 s, so a 7-step cuRobo pick spent 259 s of 311 s starting planners (2026-11-24).
This daemon holds the real planners; `get_planner(name, cfg, remote=True)` returns a RemotePlanner that sends the PlanRequest
here and gets the PlanResult back (same retiming, same metrics: the real planner's plan() runs in the daemon).

  protocol : JSON lines over 127.0.0.1:<planner.daemon_port> (default 8765)
             {"planner": "curobo", "request": {PlanRequest fields}} -> {PlanResult fields}   |   {"cmd": "stop"}
  start    : automatic on first use (RemotePlanner starts it detached); or  python3 planner_daemon.py
  stop     : python3 planner_daemon.py stop          (frees the cuRobo GPU worker and MoveIt's move_group)
  reload   : a planner is rebuilt when cell.yaml changes (scene, limits); after editing planner CODE, stop the daemon
  idle     : exits after planner.daemon_idle_s (default 1800 s) without a request
"""
import json, os, socket, socketserver, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cell
from planning import PlanRequest, PlanResult, Planner, get_planner


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def port(cfg=None):
    return int(((cfg or cell.load()).get("planner") or {}).get("daemon_port", 8765))


class Daemon(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, cfg):
        super().__init__(("127.0.0.1", port(cfg)), Handler)
        self.planners, self.cfg_mtime, self.lock, self.last = {}, None, threading.Lock(), time.time()
        self.idle_s = float((cfg.get("planner") or {}).get("daemon_idle_s", 1800))
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while True:
            time.sleep(10)
            if time.time() - self.last > self.idle_s and not self.lock.locked():
                print("[planner daemon] idle, exiting", flush=True); self.shutdown(); return

    def planner(self, name):
        m = os.path.getmtime(cell.CFG_PATH)
        if m != self.cfg_mtime:                       # config changed: rebuild every planner (scene, limits)
            for p in self.planners.values():
                getattr(p, "close", lambda: None)()
            self.planners, self.cfg_mtime = {}, m
        if name not in self.planners:
            t0 = time.time(); self.planners[name] = get_planner(name, cell.load())
            print(f"[planner daemon] {name} up in {time.time() - t0:.1f} s", flush=True)
        return self.planners[name]

    def close_all(self):
        for p in self.planners.values():
            getattr(p, "close", lambda: None)()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line.strip():                          # a bare connect (the "is it up?" probe)
            return
        msg = json.loads(line)
        srv = self.server
        if msg.get("cmd") == "stop":
            self.wfile.write(b'{"ok": true}\n'); threading.Thread(target=srv.shutdown, daemon=True).start(); return
        with srv.lock:                                # one plan at a time (one GPU worker, one move_group)
            srv.last = time.time()
            try:
                res = srv.planner(msg["planner"]).plan(PlanRequest(**msg["request"]))
                out = _plain(res.__dict__)
            except Exception as e:
                out = _plain(PlanResult(False, msg.get("planner", "?"), "backend_error", None,
                                        notes={"reason": f"daemon: {type(e).__name__}: {e}"}).__dict__)
            srv.last = time.time()
        self.wfile.write((json.dumps(out) + "\n").encode())


class RemotePlanner(Planner):
    """Same interface as every planner; the work happens in the daemon (started on demand)."""
    def __init__(self, name, cfg):
        super().__init__(cfg); self.name = name; self.port = port(cfg); self.ensure()

    def ensure(self):
        if self.up():
            return
        import subprocess
        log = open(os.path.join(cell.WS, "logs", "planner_daemon.log"), "a")
        subprocess.Popen([sys.executable, os.path.abspath(__file__)], stdout=log, stderr=log, start_new_session=True)
        t0 = time.time()
        while not self.up():
            if time.time() - t0 > 30:
                raise RuntimeError("planner daemon did not start (see ~/ip_ws/logs/planner_daemon.log)")
            time.sleep(0.3)

    def up(self):
        with socket.socket() as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def plan(self, req):
        with socket.create_connection(("127.0.0.1", self.port), timeout=300) as s:
            s.sendall((json.dumps(dict(planner=self.name, request=_plain(req.__dict__))) + "\n").encode())
            line = s.makefile().readline()
        if not line:
            return PlanResult(False, self.name, "backend_error", None, notes={"reason": "planner daemon closed the connection"})
        return PlanResult(**json.loads(line))


def main():
    cfg = cell.load()
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        try:
            with socket.create_connection(("127.0.0.1", port(cfg)), timeout=3) as s:
                s.sendall(b'{"cmd": "stop"}\n'); print("planner daemon:", s.makefile().readline().strip())
        except OSError:
            print("planner daemon: not running")
        return
    srv = Daemon(cfg)
    print(f"[planner daemon] up on 127.0.0.1:{port(cfg)}", flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.close_all(); srv.server_close()


if __name__ == "__main__":
    main()
