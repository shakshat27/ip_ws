# UR10 cell: start-up from a cold reboot to a successful pick

Aliases come from `~/.ip_aliases` (loaded by `~/.bashrc`; list in `~/Desktop/alias.txt`). Use one terminal per step and keep them open.

## 1. Start (in this order)

| # | terminal | command | ready when |
|---|---|---|---|
| 1 | mini PLC (owns the belt) | `plc` | log shows `mini PLC up` and `belt RUN` |
| 2 | Isaac Sim with our cell | `isaaccell` | window shows the cell, arm upright, timeline playing (about 40 s) |
| 3 | dashboard | `s0ui` | open http://127.0.0.1:8090, live camera image, planner dropdown filled |

Why this order: the pick task refuses to start unless the PLC answers; the dashboard subscribes to Isaac's camera when it starts (see section 4).

Planners start themselves on first use, inside one long-lived planner daemon (`~/ip_ws/logs/planner_daemon.log`): MoveIt 2 (its own move_group, domain 71, under 1 s) and cuRobo (worker, about 37 s, only on the very first move; after that a pick takes about 45 s). The daemon exits after 30 minutes idle. Stop it by hand before closing Isaac for GPU-heavy work, or after changing planner code: `sip; python3 ~/ip_ws/src/lib/planner_daemon.py stop`. RoboDK is the exception: open it first with `cd ~/RoboDK && ./RoboDK-Start.sh ~/ip_ws/cell.rdk`, and it needs an active license.

## 2. Check

```bash
sip; ros2 topic list | grep -E "joint_states|rgb|plc"
```

Expect `/joint_states`, `/rgb`, `/plc/state`. Dashboard: belt label shows `run` or `halt`.

## 3. A successful pick (dashboard, the demo)

1. Press **belt run** (label `run`).
2. Press **spawn** (count 1): a box appears at the conveyor start and travels towards the robot.
3. Choose the planner (MoveIt 2 or cuRobo; RoboDK needs its license) and press **pick** right away.
4. The robot moves to the view pose and watches the belt. When the box comes into view, the PLC halts the belt, the robot measures the box, grasps, lifts, checks the hold, swings over `Basket01`, lowers the box in, releases, lifts away and goes home; the belt runs again. Success is `outcome: picked and placed` in the job row.
5. Waiting time: the box needs about 1.5 to 2 minutes to reach the camera (the simulation runs slower than real time). For a quicker demo, spawn it near the camera instead: `sip; ros2 topic pub --once /spawn_boxes std_msgs/msg/String "{data: '1 random at 3.90 0.60'}"`.

Same from a terminal: `sipv; python3 ~/ip_ws/src/pipeline/s0_task_manager.py --task pick --type box --planner moveit --then place --watch` (`--then return` puts the box back where it was, `--then hold` keeps it).

## 4. When something is wrong

| symptom | cause and fix |
|---|---|
| no `/joint_states`, `/rgb`, `/tf` | Isaac loaded its graphs disabled: start Isaac with `isaaccell`, not the plain `isaac` |
| dashboard image frozen after an Isaac restart | restart the dashboard: find its PID with `ss -ltnp \| grep 8090`, `kill -INT <pid>`, then `kill -9 <pid>` if it is still there after 5 s, then `s0ui` |
| `belt not halted by the PLC` | the PLC is not running: `plc` |
| first move of a session takes 40 s | cuRobo is starting in the planner daemon; later moves are immediate |
| `contact stop ... off the command` | the arm lagged its command (slow sim) or touched something; retry the move |
| RoboDK `Invalid license: API calls are limited` | license deactivated: activate under Help > License |
| nothing detected | check the box is in the camera image; if the robot was moved in `cell.usd`, re-teach the belt: view pose, then `sipv; python3 ~/ip_ws/src/tools/teach_surface.py --name belt_a --write` (pick the big level plane with `--plane N`) |

Never use `pkill -f` or `pgrep -f` to stop processes; use the PID.
