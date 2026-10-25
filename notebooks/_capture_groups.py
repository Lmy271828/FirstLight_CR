"""Deep capture of causal-group observation research with per-round logging."""

import json
from collections import defaultdict

nb = json.load(open("notebooks/get_familiar_with_firstlight_cr.ipynb", encoding="utf-8"))
ns = {}
for i, cell in enumerate(nb["cells"]):
    if i > 9:  # 只执行到 §3 观测解剖（env/new_match/act_now/CARD_NAMES 均已就绪）
        break
    if cell["cell_type"] != "code":
        continue
    print(f"--- code cell {i} ---")
    exec(compile(cell["source"], f"<cell{i}>", "exec"), ns)

env = ns["env"]
act_now = ns["act_now"]
ActionV1 = ns["ActionV1"]
CARD_NAMES = ns["CARD_NAMES"]
SWARM_NAMES = {"SkeletonArmy", "GoblinGang", "MinionHorde", "Minions", "Skeletons", "Witch", "Graveyard"}

def causal_key(entity):
    return entity.causal_group.handle if entity.causal_group is not None else None

def cluster_snapshot(obs):
    clusters = defaultdict(list)
    for entity in obs.entities:
        clusters[causal_key(entity)].append(entity)
    return clusters

best = None
for round_index in range(20):
    mask = env.action_mask(0)
    native_now = env.native.observe()
    rp0 = next(p for p in native_now["players"] if p["owner"] == 0)
    hand = {c["handIndex"]: c for c in rp0["hand"]}
    reason_dump = dict(mask.reasons)

    best_pick = None
    for i, ok in enumerate(mask.hand_slots):
        if not ok or str(i) not in mask.placement_masks:
            continue
        card = hand.get(i)
        if card is None:
            continue
        name = CARD_NAMES.get(card["cardId"], "")
        rows = mask.placement_masks[str(i)]["row_major"]
        cells = [(cx, cy) for cy, row in enumerate(rows) for cx, v in enumerate(row) if v]
        if not cells:
            continue
        is_swarm = name in SWARM_NAMES and len(cells) < 576
        score = (0 if is_swarm else 1, -card["cost"])
        cx, cy = cells[len(cells) // 2]
        pick = (score, i, (cx, cy), card, name)
        if best_pick is None or pick[0] < best_pick[0]:
            best_pick = pick

    played = "-"
    biggest = "-"
    if best_pick is not None:
        _, slot, (cx, cy), card, name = best_pick
        played = f"{name}({card['cost']})"
        act_now(ActionV1.play(owner=0, hand_slot=slot, grid=(cx, cy), card_id=card["cardId"]))
        snap_obs, *_ = act_now(ActionV1.wait(0, ticks=2))
        clusters = cluster_snapshot(snap_obs)
        biggest = max((len(v) for v in clusters.values()), default=0)
        if best is None or biggest > best[0]:
            best = (biggest, dict(clusters), snap_obs)
    else:
        act_now(ActionV1.wait(0, ticks=10))

    print(f"[round {round_index:2d}] tick={env.native.observe()['tick']:>4} 圣水={rp0['elixir']:>5.2f} "
          f"hand={[(c['handIndex'], CARD_NAMES.get(c['cardId'], c['cardId']), c['cost']) for c in rp0['hand']]} "
          f"slots={mask.hand_slots} ready={reason_dump.get('decision_ready')} "
          f"出牌={played} 本轮最大组={biggest}")

print()
if best is None:
    raise SystemExit("未捕获任何因果组——查看上方逐轮日志")
size, clusters, snap_obs = best
print(f"========== 深度剖析：历史最大瞬间（{size} 个实体同组） ==========")
for key, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    ref = members[0].causal_group
    src = CARD_NAMES.get(ref.source_card_id, ref.source_card_id)
    print(f"组 handle = {ref.handle}")
    print(f"  kind = {ref.kind.value} | 来源卡 = {src} | parent_entity = {ref.parent_entity_id} | version = {ref.version}")
    for m in members:
        hp = "-" if m.hitpoints is None else int(m.hitpoints)
        px = round(float(m.position[0]))
        py = round(float(m.position[1]))
        print(f"    实体 {m.entity_id}: {CARD_NAMES.get(m.card_id, m.card_id)} | kind={m.entity_kind} | hp={hp} | pos=({px},{py})")
    print()

print("========== 实体总表 ==========")
print("entities 总数:", len(snap_obs.entities))
kinds = {}
for e in snap_obs.entities:
    kinds[e.entity_kind] = kinds.get(e.entity_kind, 0) + 1
print("实体种类分布:", kinds)
