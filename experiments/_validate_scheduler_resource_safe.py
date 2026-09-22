from pathlib import Path

# Start from the validated generic-retry scheduler (91,362): no action-label
# time-sensitive cost, no FEED-after-PLANT/HARVEST rule, generic (not animal)
# alternate packing search, opening still uses untouched legacy scheduler.
base = Path("experiments/_validate_scheduler_generic_retry.py").read_text()
exec(compile(base, "_validate_scheduler_generic_retry.py", "exec"), {})

# ---------------------------------------------------------------------------
# Post-opening router: mandatory/optional may affect ADMISSION, but does not
# constrain execution order inside a worker queue.
# ---------------------------------------------------------------------------
scheduler = Path("agents/scheduler_post.py")
text = scheduler.read_text()
old = '''                # Buckets are already priority-sorted; only the two new
                # neighbours can violate the invariant after insertion.
                if insertion and priorities[id(bucket[insertion - 1])] > priority:
                    continue
                if insertion < len(bucket) and priority > priorities[id(bucket[insertion])]:
                    continue
'''
if text.count(old) != 1:
    raise SystemExit(f"route class-order anchor count={text.count(old)}")
text = text.replace(old, "", 1)
scheduler.write_text(text)

# ---------------------------------------------------------------------------
# Morning resource plan: after opening, do not assume a same-day WHEAT harvest
# happens before FEED.  The route is free to order complete tasks by geometry,
# so feed must already be funded independently of that order.
# ---------------------------------------------------------------------------
farm_tasks = Path("agents/farm_tasks.py")
ft = farm_tasks.read_text()
old = '''def purchase_orders(
    obs, targets, active_positions, available_money=None, available_wheat=None,
    replant_same_crop=False,
):
'''
new = '''def purchase_orders(
    obs, targets, active_positions, available_money=None, available_wheat=None,
    replant_same_crop=False, credit_same_day_wheat=True,
):
'''
if ft.count(old) != 1:
    raise SystemExit(f"purchase signature anchor count={ft.count(old)}")
ft = ft.replace(old, new, 1)
old = '''    wheat_needed = max(
        0,
        live_animals + pending_feed + new_animal_feed
        - wheat_on_hand - wheat_incoming,
    )
'''
new = '''    wheat_credit = wheat_incoming if credit_same_day_wheat else 0
    wheat_needed = max(
        0,
        live_animals + pending_feed + new_animal_feed
        - wheat_on_hand - wheat_credit,
    )
'''
if ft.count(old) != 1:
    raise SystemExit(f"wheat credit anchor count={ft.count(old)}")
ft = ft.replace(old, new, 1)
farm_tasks.write_text(ft)

# Wire the post-opening policy through the morning market planner and remove
# reactive post-opening feed purchases. Opening intentionally keeps legacy
# just-in-time refinance/feed behavior.
agent = Path("agents/expansion_agent.py")
a = agent.read_text()
old = '''    mandatory_hand_target=None,
):
'''
new = '''    mandatory_hand_target=None,
    credit_same_day_wheat=True,
):
'''
if a.count(old) != 1:
    raise SystemExit(f"hire-buy signature anchor count={a.count(old)}")
a = a.replace(old, new, 1)
old = '''            replant_same_crop=replant_same_crop,
        )
'''
new = '''            replant_same_crop=replant_same_crop,
            credit_same_day_wheat=credit_same_day_wheat,
        )
'''
# There should be exactly one purchase_orders call with this anchor in helper.
if a.count(old) < 1:
    raise SystemExit("purchase call anchor missing")
a = a.replace(old, new, 1)
old = '''                mandatory_hand_target=mandatory_hand_target,
            )
'''
new = '''                mandatory_hand_target=mandatory_hand_target,
                credit_same_day_wheat=state["opening_active"],
            )
'''
if a.count(old) != 1:
    raise SystemExit(f"hour0 hire-buy call anchor count={a.count(old)}")
a = a.replace(old, new, 1)
old = '''        if day < effective_end:
            market += feed_wheat_order(
                obs, committed_feed_targets, _active_positions(farm)
            )
'''
new = '''        if state["opening_active"] and day < effective_end:
            market += feed_wheat_order(
                obs, committed_feed_targets, _active_positions(farm)
            )
'''
if a.count(old) != 1:
    raise SystemExit(f"intraday feed anchor count={a.count(old)}")
a = a.replace(old, new, 1)
agent.write_text(a)
