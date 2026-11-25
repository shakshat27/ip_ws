# Isaac startup (--exec): open ~/ip_ws/cell.usd, wait for it to load, switch the action graphs on, press play.
# Found 2026-11-24: this Isaac loads every graph DISABLED (graph.is_disabled() True, OnPlaybackTick frame stuck at 0, no
# publishers), so nothing reached ROS until graph.set_disabled(False). Cause of the disabled state not found.
import asyncio, os, omni.usd, omni.timeline, omni.kit.app, omni.graph.core as og
async def _go():
    ctx = omni.usd.get_context(); app = omni.kit.app.get_app()
    ok, err = await ctx.open_stage_async(os.path.expanduser("~/ip_ws/cell.usd"))
    print("[open_cell] opened", ok, err, flush=True)
    for _ in range(240):                     # let payloads and the graphs settle (~4 s at 60 Hz)
        await app.next_update_async()
    omni.timeline.get_timeline_interface().play()
    for _ in range(30): await app.next_update_async()
    for g in og.get_all_graphs():
        if g.is_disabled(): g.set_disabled(False)
    print("[open_cell] playing, graphs on:", [g.get_path_to_graph() for g in og.get_all_graphs()], flush=True)
asyncio.ensure_future(_go())
