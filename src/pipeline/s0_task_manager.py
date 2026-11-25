#!/usr/bin/env python3
"""Script 0 - TASK MANAGER. Decides what happens next and calls the other scripts in order.

It moves nothing and computes no geometry itself: it runs scripts 2-5 and the gripper tool as separate processes
(each prints its own IN / OUT) and makes the decisions between them, logging each decision and
its reason in <run>/0_task.json.

Task 'pick':
  5 move to the capture pose  ->  2 capture  ->  3 detect  ->  4 localize
  -> decide: which object (requested type, nearest the robot base = the belt end, with a safe finger placement)
  -> 4b grasp poses  ->  fingers to the approach opening  ->  5 free move to pre-grasp  ->  5 straight descent
  -> close + grasp check (fingers stop short = object between them)  ->  5 straight lift  ->  hold check
  -> --then return (default): put it back, release, straight up, back to the capture pose; --then hold: keep it
  Any failed move or empty grip: open the gripper and back straight up, outcome 'failed'.
  Every move goes through script 5 with the chosen planner (--planner, default cell.yaml planner.backend).

  sipv
  python3 ~/ip_ws/src/pipeline/s0_task_manager.py --task pick --type box [--planner robodk] [--then return|hold] [--no-move]

Front end (--ui): live camera feed with detection + localisation, spawn controls, tables (lib/live.py, ui/s0.html).
Needs the training environment (the detector): `sipv`, then open http://127.0.0.1:8090
  python3 ~/ip_ws/src/pipeline/s0_task_manager.py --ui [--port 8090]
"""
import argparse, math, os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import cell, run_io

NAME = "s0 task"
HERE = os.path.dirname(os.path.abspath(__file__))


def call(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print(f"\n[{NAME}] -> {script} {' '.join(args)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[{NAME}] {script} failed (exit {r.returncode}); stopping")


def run_step(script, *args):
    """Like call, but returns the exit code so the task can react (back off, report) instead of stopping."""
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print(f"\n[{NAME}] -> {script} {' '.join(args)}", flush=True)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick", choices=["pick"])
    ap.add_argument("--type", default="box", help="object type to pick, from cell.yaml objects")
    ap.add_argument("--pose", default="pick_overview")
    ap.add_argument("--no-move", action="store_true", help="capture from where the arm is now")
    ap.add_argument("--method", default="rfdetr", choices=["rfdetr", "geometry"],
                    help="detector for script 3: the trained model (cell.yaml detector; needs `sipv`) or the depth-only baseline")
    ap.add_argument("--planner", default=None, help="motion planner backend for every move; default cell.yaml planner.backend")
    ap.add_argument("--then", default="return", choices=["return", "hold", "drop", "place"],
                    help="after the pick: put the object back and return to the capture pose, keep holding it, or drop it in the basket (cell.yaml drop:) and go home")
    ap.add_argument("--near", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="pick THIS object: the one within --near-tol of (x, y) in the robot base frame (e.g. a row clicked on "
                         "the s0 page); refused, not replaced, if it is missing or has no safe finger placement")
    ap.add_argument("--near-tol", type=float, default=0.03, help="m")
    ap.add_argument("--watch", action="store_true", help="wait at the capture pose with the belt running until a box is in view, then halt the belt and pick (camera-triggered stop)")
    ap.add_argument("--no-plc", action="store_true", help="do not ask the mini PLC (tools/mini_plc.py) to hold the belt while picking")
    ap.add_argument("--ui", action="store_true", help="serve the live front end instead of running a task")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    if a.ui:
        sys.path.insert(0, HERE)                  # s3_detect / s4_localize for the live loop
        import live
        live.serve(cell.load(), port=a.port); return
    cfg = cell.load(); planner = a.planner or (cfg.get("planner") or {}).get("backend", "interim")
    run = run_io.new_run(a.task); rn = os.path.basename(run)
    log = dict(task=a.task, type=a.type, pose=a.pose, planner=planner, then=a.then, started=time.strftime("%Y-%m-%d %H:%M:%S"), decisions=[])
    run_io.show(NAME, "IN", [f"task {a.task}, object type {a.type}, capture pose {a.pose}, planner {planner}, then {a.then}", f"run {run}"])

    def decide(step, result, reason):
        log["decisions"].append(dict(step=step, result=result, reason=reason))
        print(f"\n[{NAME}] decision ({step}): {reason}", flush=True)

    def plc(value, wait):
        """The robot only writes a tag: pick_request true = hold the belt (waits until the PLC reports it halted, then the settle
        time), false = release. The PLC (tools/mini_plc.py) owns the belt. A missing PLC stops the task rather than picking off a moving belt."""
        rc = run_step("../tools/plc_tag.py", "pick_request", value, "--wait", wait)
        if value == "true":
            if rc != 0:
                decide("belt hold", False, "the PLC did not confirm the belt halted (is tools/mini_plc.py running?)")
                return False
            time.sleep(float((cfg.get("plc") or {}).get("settle_s", 0.5)))
            decide("belt hold", True, "pick_request written; the PLC halted the belt")
        return True

    def finish(outcome, lines):
        log["outcome"] = outcome; log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run_io.write_json(os.path.join(run, "0_task.json"), log)
        run_io.show(NAME, "OUT", lines + [f"{run}/0_task.json", f"outcome: {outcome}"])

    if not a.no_plc and a.task == "pick":
        import atexit
        atexit.register(lambda: subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "plc_tag.py"), "pick_request", "false"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))    # the belt always runs again when the task ends, however it ends
        if not a.watch and not plc("true", "halted"):
            return finish("failed", ["belt not halted by the PLC; nothing moved"])

    def move(label, *goal, motion="free"):
        return run_step("s5_planner.py", *goal, "--motion", motion, "--planner", planner, "--execute", "--run", rn, "--label", label) == 0

    def gripper(*args):
        return run_step("../tools/gripper.py", *args, "--hold", "1.5")

    if not a.no_move and not move("to_capture", "--pose", a.pose):
        return finish("failed", ["could not reach the capture pose"])
    if a.watch:
        # camera-triggered stop: the belt keeps running; look until a box is in view (sim time runs slower than wall time, so
        # a fixed wait cannot know where the box is), then the PLC halts the belt and the box is measured again standing still
        t_end = time.time() + float((cfg.get("plc") or {}).get("watch_timeout_s", 120)); seen = False
        while time.time() < t_end and not seen:
            call("s2_capture.py", "--label", a.pose, "--run", rn)
            call("s3_detect.py", "--run", rn, "--capture", f"2_capture_{a.pose}", "--method", a.method)
            call("s4_localize.py", "--run", rn, "--label", a.pose)
            seen = any(o["type"] == a.type for o in run_io.read_json(os.path.join(run, f"4_localize_{a.pose}", "objects.json"))["objects"])
        if not seen:
            return finish("nothing to pick", [f"no '{a.type}' came into view within the watch time"])
        decide("watch", True, f"a '{a.type}' is in view")
        if not a.no_plc and not plc("true", "halted"):
            return finish("failed", ["belt not halted by the PLC"])
    call("s2_capture.py", "--label", a.pose, "--run", rn)
    call("s3_detect.py", "--run", rn, "--capture", f"2_capture_{a.pose}", "--method", a.method)
    call("s4_localize.py", "--run", rn, "--label", a.pose)

    # choose: the requested type, nearest the robot base (the belt end) first; skip any without a safe finger placement
    objs = run_io.read_json(os.path.join(run, f"4_localize_{a.pose}", "objects.json"))["objects"]
    wanted = sorted([o for o in objs if o["type"] == a.type], key=lambda o: math.hypot(o["position"][0], o["position"][1]))
    if not wanted:
        decide("choose object", None, f"no '{a.type}' among {len(objs)} object(s): {[o['type'] for o in objs]}")
        return finish("nothing to pick", [f"no '{a.type}' found ({len(objs)} object(s) seen)"])
    if a.near:                                     # the operator chose one: only that one, never a substitute
        d = lambda o: math.hypot(o["position"][0] - a.near[0], o["position"][1] - a.near[1])
        sel = min(wanted, key=d)
        if d(sel) > a.near_tol:
            decide("choose object", None, f"no '{a.type}' within {a.near_tol * 1000:.0f} mm of the selected position "
                   f"({a.near[0]:+.3f}, {a.near[1]:+.3f}); nearest is #{sel['id']} at {d(sel) * 1000:.0f} mm")
            return finish("selected object not found", [f"nothing within {a.near_tol * 1000:.0f} mm of ({a.near[0]:+.3f}, {a.near[1]:+.3f})"])
        wanted = [sel]
    g = None
    for o in wanted:
        call("s4b_grasp.py", "--run", rn, "--label", a.pose, "--object", str(o["id"]))
        cand = run_io.read_json(os.path.join(run, f"4b_grasp_{a.pose}", "grasp.json"))
        unsafe = [n for n in cand["notes"] if n.startswith("NO SAFE")]
        if unsafe or "grasp" not in cand:
            decide("choose object", None, f"#{o['id']} skipped: {unsafe[0] if unsafe else cand['notes']}")
            continue
        g = cand
        why = (f"the one selected at x {a.near[0]:+.3f} y {a.near[1]:+.3f} (found "
               f"{math.hypot(o['position'][0] - a.near[0], o['position'][1] - a.near[1]) * 1000:.0f} mm from it" if a.near
               else f"the nearest '{a.type}' to the robot base with a safe finger placement ({math.hypot(*o['position'][:2]):.3f} m")
        decide("choose object", o["id"], f"#{o['id']}, {why}; finger clearance "
               + ("none nearby" if g["finger_clearance_m"] is None else f"{g['finger_clearance_m'] * 1000:+.1f} mm") + ")")
        break
    if g is None:
        if a.near:
            return finish("selected object not pickable", ["the selected object has no safe finger placement (see the decision above)"])
        return finish("nothing to pick", [f"{len(wanted)} '{a.type}' seen, none with a safe finger placement"])
    run_io.write_json(os.path.join(run, "0_grasp_chosen.json"), g)
    tool = lambda key: [*map(str, g[key]["xyz"]), *map(str, g[key]["rpy"])]

    def back_off(why):
        gripper("open"); move("back_off", "--tool0", *tool("pre_grasp"), motion="linear")
        return finish("failed", [why, "gripper opened, arm backed straight up to the pre-grasp pose"])

    # pick: fingers to the approach opening, above the object, straight down, close, check, straight up, check
    gripper("--pos", str(g["finger_approach_rad"]))
    if not move("pre_grasp", "--tool0", *tool("pre_grasp")):
        return finish("failed", ["could not reach the pre-grasp pose"])
    if not move("grasp", "--tool0", *tool("grasp"), motion="linear"):
        return back_off("descent did not arrive (contact stop or planning failure)")
    # grasp check: the fingers must stop where this object's width puts them (gripper model), not merely short of closed
    check = ("--pos", str(g["finger_close_rad"]), "--expect-hold", "--expect-rad", str(g["finger_contact_rad"]))
    why = {2: "the fingers closed fully: nothing between them",
           3: "the fingers stopped far from the object's width: blocked by something else"}
    rc = gripper(*check)
    if rc != 0:
        decide("grasp check", False, why.get(rc, f"gripper tool exit {rc}"))
        return back_off(f"grasp check failed: {why.get(rc, rc)}")
    decide("grasp check", True, f"the fingers stopped at the object's width (about {g['finger_contact_rad']} rad): object between them")
    if not move("lift", "--tool0", *tool("lift"), motion="linear"):
        return finish("failed", ["lift did not arrive; holding the object where the arm stopped"])
    rc = gripper(*check)
    if rc != 0:
        decide("hold check", False, "after the lift: " + why.get(rc, f"gripper tool exit {rc}"))
        gripper("open"); return finish("failed", ["object lost during the lift"])
    decide("hold check", True, "after the lift the fingers still stop at the object's width: object held")

    if a.then == "hold":
        return finish("picked", [f"object #{g['object']['id']} held at the lift pose ({g['lift']['xyz']} m)"])
    if a.then in ("drop", "place"):          # "drop" kept as an alias: both place the box in the basket
        # up to the release height straight above the pick (linear), over the basket (free), open, home. Heights: the fingertips
        # (TCP) end `clearance` above the basket rim; tool0 is tcp_z_offset higher. Same orientation as the grasp.
        dp = cfg["drop"]; z0 = cfg["gripper"]["tool0_T_tcp"]["xyz"][2]
        zt = dp["rim_z"] + dp["clearance"] + z0
        over_pick = [g["lift"]["xyz"][0], g["lift"]["xyz"][1], zt]
        over_basket = [dp["centre"][0], dp["centre"][1], zt]
        rpy = g["lift"]["rpy"]
        # over the basket the tool keeps its grasp orientation turned by the base swing between the pick and the basket, so the
        # move is a base swing and every planner's IK lands near the start (with the grasp yaw unchanged the wrist had to flip:
        # MoveIt went 10 to 57 rad of joint path for this move, 2026-11-24)
        swing = math.atan2(over_basket[1], over_basket[0]) - math.atan2(over_pick[1], over_pick[0])
        rpy_drop = [rpy[0], rpy[1], math.atan2(math.sin(rpy[2] + swing), math.cos(rpy[2] + swing))]
        # place, not drop: straight down until the box bottom is place_gap above the basket floor (tool0 height from the grasp:
        # the TCP sat (tcp - object centre) above the box centre), open, straight up, home. If the descent meets something first
        # (another box), the contact stop ends it there and the box is released where it rests, as a force-limited place would.
        o = g["object"]; tcp_above_centre = g["tcp_target"][2] - o["position"][2]
        z_place = dp["floor_z"] + dp.get("place_gap", 0.005) + o["height"] / 2 + tcp_above_centre + z0
        at_place = [over_basket[0], over_basket[1], z_place]
        ok = (move("rise", "--tool0", *map(str, over_pick), *map(str, rpy), motion="linear")
              and move("over_basket", "--tool0", *map(str, over_basket), *map(str, rpy_drop)))
        if ok:
            down = move("place_down", "--tool0", *map(str, at_place), *map(str, rpy_drop), motion="linear")
            decide("place", True, "set down on the basket floor" if down else "the descent ended just short of the target (box touched down): released there")
            ok = (gripper("open") == 0 and move("place_up", "--tool0", *map(str, over_basket), *map(str, rpy_drop), motion="linear")
                  and move("to_home", "--pose", "home"))
        return finish("picked and placed" if ok else "picked; place failed",
                      [f"object #{g['object']['id']} picked and placed in the basket at {dp['centre']} (tool0 z {z_place:.3f} m at release)",
                       "arm back at the home pose" if ok else "the place sequence stopped; see the log above"])
    # return: put it back where it was, release, straight up, back to the capture pose (keeps the scene for the next test)
    ok = (move("place_back", "--tool0", *tool("grasp"), motion="linear") and gripper("--pos", str(g["finger_approach_rad"])) == 0
          and move("leave", "--tool0", *tool("lift"), motion="linear") and gripper("open") == 0 and move("to_capture_end", "--pose", a.pose))
    return finish("picked and returned" if ok else "picked; return failed",
                  [f"object #{g['object']['id']} picked, lifted {cfg['grasp']['approach_height'] * 100:.0f} cm and held",
                   "put back where it was, arm back at the capture pose" if ok else "the return sequence stopped; see the log above"])


if __name__ == "__main__":
    main()
