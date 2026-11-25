# ip_ws documentation

| File | Contents |
|---|---|
| [issues_and_discrepancies.md](issues_and_discrepancies.md) | Every issue, mismatch and limitation found, with evidence, impact and status; deviations from the problem statement |

Related, outside this folder:

| Where | Contents |
|---|---|
| `~/ip_ws/config/cell.yaml` | The cell description every script reads: calibration, surfaces, poses, detector, planner |
| `~/ip_ws/src/` | `pipeline/` s0–s5, `lib/` shared code (planner interface: `planning.py`), `tools/` offline tools, `ui/` s0 page |
| `~/ip_ws/evals/` | Detector evaluations (`summary.json`, per-detection CSV); `bench_20261021/run.sh` re-runs the two-pass benchmark |
| `~/ip_ws/runs/` | Pipeline and planner run folders (inputs and outputs of every step) |
| `~/ip_ws/report/2026-10-21/` | Status report figures and the script that makes them |
| `~/ros2_ws/action_graph_scripts/UR10_SpawnBoxes.py` | Box spawner script node for `/Graphs/SpawnBoxes` |
