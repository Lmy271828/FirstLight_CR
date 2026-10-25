"""One-off generator for notebooks/get_familiar_with_firstlight_cr.ipynb."""

import json


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


cells = []

cells.append(md("""# FirstLight CR 入门：像熟悉 Gymnasium 一样熟悉本地对战 API

本 notebook 把 FirstLight CR 的 Python API 与你熟悉的 Gymnasium 概念一一对应。
跑完前三个部分你就能：**启动一场可视化对局 → 越过热身期 → 读取合法动作 → 亲手下一张牌**。

## 前置条件（跑不通请先完成 `docs/getting_started.md`）

1. 离线引擎已安装进专用 MuMu 实例（`install_offline_engine.ps1` 成功）；
2. 本 notebook 必须使用仓库 `.venv` 的 Jupyter 内核（依赖 torch/pyarrow 等）；
3. MuMu 实例在运行。引擎进程由 §1 的代码自动拉起（内部调用 `native_runner/start_offline.ps1`）。

## Gymnasium ↔ FirstLight 对照表

| Gymnasium | FirstLight CR | 差异说明 |
| --- | --- | --- |
| `env.reset()` | `env.reset(match_config=..., options=...)` | 返回**双方**观测字典 `{0: obs, 1: obs}`（PettingZoo parallel 风格）；单方视角用 `step_actor()` |
| `obs, reward = env.step(a)` | `obs, r, term, trunc, info = env.step({0: [...], 1: [...]})` | 动作是字典；每方每个决策可提交 1~3 个复合动作；**`step` 会阻塞到下一个决策边界** |
| `action_space.sample()` | `env.action_mask(owner)` | 没有采样器，只有**带原因的合法性掩码**（含每槽 18×32 落点矩阵）；越界动作抛 `BattleEnvError` |
| 引擎内置物理 | **真实游戏引擎**（probe 劫持的 `libg.so`） | Python 不实现任何游戏逻辑，引擎是唯一事实源 |
| 帧率 | 20 Hz 逻辑 tick，本 notebook 用 5 tick 决策间隔 | 训练时 4 决策/秒；可视化模式 0.25×~4× 变速 |
| 单智能体 | 双方零和（奖励自动反对称化） | 本教程只驱动 owner 0，对手保持等待 |

锚点常量：`FIRST_PLAYABLE_TICK = 90`（热身期后的第一个可决策 tick）。"""))

cells.append(code("""import json
import random
import socket
import sys
import time
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "native_runner").is_dir():
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
print("Python:", sys.executable)
assert "venv" in sys.executable.lower(), "请使用仓库 .venv 的 Jupyter 内核运行本 notebook"

from native_runner.battle_env import BattleEnvError, BattleEnvV1
from native_runner.contracts import ActionV1
from dataclasses import replace

from native_runner.cr_native_env import NativeClashEnv, RunnerError
from native_runner.match_factory import MatchConfig

# 观测研究牌组：以"群卡"为主（一次部署产生多个实体），用于观察因果组结构
SWARM_DECK = (
    26000012,  # Skeleton Army 骷髅军团（一次 ~15 个骷髅）
    26000041,  # Goblin Gang 哥布林团伙（3 哥布林 + 2 飞刀哥布林）
    26000022,  # Minion Horde 亡灵大军（6 亡灵）
    26000010,  # Skeletons 骷髅兵（3~4 个）
    26000007,  # Witch 巫婆（持续召唤骷髅）
    28000010,  # Graveyard 骷髅墓园（持续在敌方场地生成骷髅）
    28000000,  # Fireball 火球（法术对照：无实体群）
    26000000,  # Knight 骑士（单体对照）
)
CARD_NAMES = {
    c["card_id"]: c["name"]
    for c in json.load(open(REPO / "native_runner" / "data" / "competitive" / "card_support.json", encoding="utf-8"))[
        "supported_cards"
    ]
}

HOST, PORT = "127.0.0.1", 26789

def engine_ready(host: str = HOST, port: int = PORT) -> bool:
    with socket.socket() as sock:
        sock.settimeout(2.0)
        return sock.connect_ex((host, port)) == 0

def ensure_engine(restarts: int = 1) -> None:
    \"\"\"探活 + attest 试连；失败则强制重启引擎（长会话后偶发 attest 响应异常）。\"\"\"
    import subprocess

    for attempt in range(restarts + 1):
        try:
            probe_env = NativeClashEnv(HOST, PORT, timeout=10.0)
            probe_env.wait_ready(timeout=20.0)
            probe_env.attest()
            probe_env.close_transport()
            print("引擎就绪（attest 通过）。")
            return
        except Exception as error:
            print(f"引擎未就绪（{type(error).__name__}: {str(error)[:80]}），正在重启离线引擎…")
            subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(REPO / "native_runner" / "start_offline.ps1"),
                ],
                check=False,
            )
    raise SystemExit("引擎多次重启后仍未就绪——请检查 MuMu 实例与 .env 配置。")

ensure_engine()"""))

cells.append(md("""## §1 启动对局，越过热身期，停在第一个可玩状态

三种执行模式，本 notebook 全程使用 **native-render**（MuMu 窗口里能看到对局画面）：

| 模式 | 用途 | 时间语义 |
| --- | --- | --- |
| `native-render` | 可视化交互（本 notebook） | 实时节奏（0.25×~4× 变速），可随时冻结 |
| `headless` | 训练/批量采集 | 无渲染，`step N` 一次推进 N tick，超实时 |
| 常驻多槽 (resident) | 一个引擎进程并行多局 | 二进制批量通道 |

**三个关键事实**：

1. **暂停与倍速是引擎状态，不是屏幕按钮**：Python 通过 TCP 向 probe 发
   `pause / resume / speed 0.25|0.5|1|2|4`，probe 直接改写引擎控制器。`native.pause()` 冻结引擎、
   `native.resume()` 继续——任何时候想停下来观察都可以。
2. **`env.step_actor(...)` 会阻塞到下一个决策边界**：和 Gymnasium 一样"提交动作、等结果"，
   只是等待的是引擎的真实时间。热身期（tick 0~90）里引擎不受理出牌，用 WAIT 逐步推进即可。
3. **本节的 `new_match()` 在结尾把对局冻结在 tick 90**：cell 运行完毕后你可以无限期观察 MuMu 窗口，
   时间不会流逝；下一次执行动作格时，`act_now(...)` 会自动恢复运行、同步时钟并出手。
4. **`env.action_mask(owner)` 不需要传观测**：环境内部自己维护引擎状态。

下面的代码：创建一场固定种子（seed=42）的对局（创建即暂停）→ 恢复实时运行 → 循环 WAIT 越过热身期
→ 冻结在 tick 90。重复运行本格会以同一 seed 重新开局（这也是后面做确定性实验的基础）。"""))

cells.append(code("""SEED = 42

RESET_OPTIONS = {
    "render_mode": "native-render",                 # MuMu 窗口可见
    "pause_native_render": True,                    # 创建即暂停，由我们决定何时开始
    "allow_initial_remaining_runtime_rejections": True,
    "decision_ticks": 5,                            # 每 5 tick（250ms 游戏时间）一个决策点
    "event_driven_decisions": False,
}

def new_match(seed: int = SEED, deck0: tuple = SWARM_DECK):
    \"\"\"重开一局并越过热身期。

    deck0 默认为群卡研究组态（见 SWARM_DECK），deck1 保持标准牌组。
    返回 (reset 时刻的双视角观测 dict, owner 0 热身期后的观测, 原生状态字典)。
    reset 时刻的双视角用于取双方各自的精确初始圣水（FAIR 契约：只能从
    各自的视角里读到自己的 exact 值）。
    \"\"\"
    observations, info = env.reset(
        match_config=MatchConfig(
            seed=seed,
            deck0=deck0,
            level_cap=11,
            minimum_card_level=11,
            king_tower_level=11,
            owner0_name="Notebook-0",
            owner1_name="Notebook-1",
        ),
        options=RESET_OPTIONS,
    )
    env.native.set_speed(1.0)
    env.native.resume()                              # 引擎开始实时运行
    while True:                                       # WAIT 循环：阻塞式越过 tick 0~90 热身期
        obs0, reward, terminated, truncated, info = env.step_actor(0, {0: [ActionV1.wait(0)]})
        if obs0.tick >= 90 or terminated or truncated:
            break
    env.native.pause()                                # 冻结在 tick 90：给你无限的观察时间
    raw = env.native.observe()                        # 原生层 JSON（含手牌费用等细节）
    return observations, obs0, raw

env = BattleEnvV1()

def act_now(action):
    \"\"\"回合制人机同步垫片：恢复运行 → 同步内部时钟 → 出手 → 重新冻结。

    引擎在实时运行时，人在 Jupyter 里读文档的几十秒里对局已经前进；
    step_actor 用的是环境内部缓存的 tick（上次 step 结束时刻）来计算命令的
    绝对执行 tick——直接提交会带着"过去的执行 tick"而被拒绝
    （RunnerError: native hand action was not queued before its absolute execute tick）。
    因此每个动作格都通过本函数提交：出手之后战场重新冻结，交还给你无限的观察时间。
    \"\"\"
    env.native.resume()                          # 解冻
    env.step_actor(0, {0: [ActionV1.wait(0)]})   # 丢弃式等待：重新同步内部时钟
    for attempt in range(3):
        # 每次重试递增执行偏移（1→3→5 tick）：后台窗口的渲染帧是猝发推进的，
        # 一次帧猝发可能跨过太近的执行边界；偏移越大余量越足。
        offset = max(1, int(action.execute_offset_ticks or 0)) + attempt * 2
        action_try = replace(action, execute_offset_ticks=offset)
        try:
            result = env.step_actor(0, {0: [action_try]})   # 立刻提交真正的动作
            break
        except RunnerError as error:
            # 提交与引擎 tick 之间存在竞态：偶发"执行时刻已过"。重新同步后重试。
            if "not queued" not in str(error) or attempt == 2:
                raise
            env.step_actor(0, {0: [ActionV1.wait(0)]})
    env.native.pause()                           # 重新冻结，交还观察权
    return result

reset_observations, obs0, native_state = new_match()

p0 = next(p for p in obs0.players if p.owner == 0)
print("原生 tick:", native_state["tick"], "（>= 90 即可出牌）")
print("我方圣水:", p0.elixir_exact)
print("手牌（卡 ID 元组）:", p0.hand)
print("原生层手牌明细（含费用）:")
for card in next(p for p in native_state["players"] if p["owner"] == 0)["hand"]:
    print(f"  slot {card['handIndex']}: card_id={card['cardId']} cost={card['cost']}")
# 提示：切到 MuMu 窗口，对局画面应冻结在开局（tick 90）——时间不会流逝；
# 之后的动作格会通过 act_now 自动恢复运行、出手、并交还给你一个冻结的战场。"""))

cells.append(md("""## §2 熟悉动作：先读掩码，再出手

FirstLight 的合法性哲学与 Gymnasium 不同：**动作空间不是枚举，而是带原因的掩码**。

- `env.action_mask(0)` 返回 `ActionMaskV1`：
  - `hand_slots: tuple[bool, bool, bool, bool]` —— 四个手牌槽当前是否可出；
  - `placement_masks[str(slot)]["row_major"][y][x]` —— 该槽在 18×32 网格上每个格是否可落；
  - `reasons` —— **为什么**（`insufficient_elixir` / 槽已空 / …）；
- `env.step_actor(...)` 对越界动作抛 `BattleEnvError`——掩码是日常依据，异常是最后防线。

> **⚠️ 冻结-恢复节奏**：`new_match()` 结束、以及每一次 `act_now(...)` 出手后，对局都处于**冻结**状态——
> 你可以无限期观察，时间不会流逝。`act_now` 负责完整的节奏：恢复运行 → 同步内部时钟 → 出手 → 重新冻结。
> 如果绕过 `act_now` 直接 `step_actor`：冻结状态下会轮询超时
> （`RunnerError: ... decision interval timed out`）；引擎运行中隔了很久才提交则会报
> `not queued before its absolute execute tick`。

我们刻意演示三条路径：**读掩码 → 合法出牌 → 故意违规并解读报错**。"""))

cells.append(code("""mask = env.action_mask(0)
print("动作类型可用性:", dict(mask.kinds))
print("手牌槽可用性:", mask.hand_slots)
for slot, entry in mask.placement_masks.items():
    legal_cells = sum(sum(1 for v in row if v) for row in entry["row_major"])
    print(f"槽 {slot}: 可落点 {legal_cells}/576 格, effective_card_id={entry.get('effective_card_id')}")
print("原因明细:", dict(mask.reasons))"""))

cells.append(code("""# 重新读一次掩码（自由运行下引擎状态在变），选一个合法 (槽, 格) 组合 —— 这就是"动作采样"的正确姿势
mask = env.action_mask(0)
legal_slot = next(i for i, ok in enumerate(mask.hand_slots) if ok)
grid_rows = mask.placement_masks[str(legal_slot)]["row_major"]
legal_cells = [(x, y) for y, row in enumerate(grid_rows) for x, v in enumerate(row) if v]
x, y = random.choice(legal_cells)

native_now = env.native.observe()
rp0 = next(p for p in native_now["players"] if p["owner"] == 0)
hand_card = next(c for c in rp0["hand"] if c["handIndex"] == legal_slot)
action = ActionV1.play(owner=0, hand_slot=legal_slot, grid=(x, y), card_id=hand_card["cardId"])
print(f"出牌: slot={legal_slot} card={hand_card['cardId']} cost={hand_card['cost']} -> ({x}, {y})")

obs0, reward, terminated, truncated, info = act_now(action)
p0 = next(p for p in obs0.players if p.owner == 0)
print(f"reward={reward} terminated={terminated} tick={obs0.tick} 圣水剩余={p0.elixir_exact}")"""))

cells.append(code("""# 故意违规 1：读一个被掩码槽位的"为什么"
# 注意：placement_masks 只为"手牌列表里存在该卡"的槽提供落点条目；
# 刚打完牌的瞬间（下一张还在发牌途中），该槽是 not_in_hand 状态，没有条目。
mask = env.action_mask(0)
reasons = dict(mask.reasons).get("slots", {})
masked_slots = {
    i: reasons.get(str(i), "槽已空（卡牌已打出或尚未到手）")
    for i, ok in enumerate(mask.hand_slots)
    if not ok
}
print("被掩码的槽位与原因:", masked_slots if masked_slots else "当前全部可出，等圣水消耗后再试")

# 故意违规 2：把一支部队扔进对方角落 (0, 0) —— 部署领土对 owner 0 永远不含 (0, 0)
# 用"可落点 < 576"挑出领土受限的部队槽（法术可砸全图，(0, 0) 对它们是合法的）。
# 先判 ok、再确认该槽有落点条目（不在支持集的牌没有条目），最后才取 entry。
troop_slot = None
for i, ok in enumerate(mask.hand_slots):
    if not ok or str(i) not in mask.placement_masks:
        continue
    entry = mask.placement_masks[str(i)]
    if sum(sum(1 for v in row if v) for row in entry["row_major"]) < 576:
        troop_slot, troop_entry = i, entry
        break

if troop_slot is None:
    print("当前没有可出的部队槽（圣水不足？）——稍后重跑本格")
else:
    corner_was_legal = bool(troop_entry["row_major"][0][0])
    native_now = env.native.observe()
    rp0 = next(p for p in native_now["players"] if p["owner"] == 0)
    troop_card = next(c for c in rp0["hand"] if c["handIndex"] == troop_slot)
    act = ActionV1.play(owner=0, hand_slot=troop_slot, grid=(0, 0), card_id=troop_card["cardId"])
    try:
        act_now(act)
        print("(0, 0) 被接受 —— 意外结果，请把输出发给作者核对")
    except BattleEnvError as error:
        print("捕获 BattleEnvError:", error)
    print(f"（槽 {troop_slot} 掩码中 (0,0) 原值 = {corner_was_legal}）")

# 等待型动作：WAIT 推进一个决策周期；WAIT_UNTIL_AFFORDABLE 睡到某槽付得起为止
obs0, reward, terminated, truncated, info = act_now(ActionV1.wait(0))
p0 = next(p for p in obs0.players if p.owner == 0)
print("等待后 tick:", obs0.tick, "圣水:", p0.elixir_exact)"""))

cells.append(md("""## §3 观测解剖：`ObservationV1` 里都有什么

`reset` / `step_actor` 返回的 `ObservationV1` 是**单个视角**的公平观测（FAIR 契约：绝不包含对手私有状态，
`tier` 字段标记可见性等级）。先让 Python 自己告诉我们它有什么字段，再逐个读取："""))

cells.append(code("""import dataclasses

print("ObservationV1 字段:", [f.name for f in dataclasses.fields(obs0)])
print("tier:", obs0.tier, "| tick:", obs0.tick, "| owner:", obs0.owner)

p0 = next(p for p in obs0.players if p.owner == 0)
p1 = next(p for p in obs0.players if p.owner == 1)
print()
print("我方玩家: 圣水(exact)", p0.elixir_exact, "| 手牌(卡ID)", p0.hand, "| 皇冠", p0.crowns)
print("对手: 圣水(exact) =", p1.elixir_exact, "← None 就是 FAIR 契约：精确圣水不公开；估计见 obs0.opponent_belief")
print()
print("塔数量:", len(obs0.towers), "| 场上实体数:", len(obs0.entities), "| 事件数:", len(obs0.events))
if obs0.entities:
    entity = obs0.entities[0]
    print("实体字段:", [f.name for f in dataclasses.fields(entity)])
print("opponent_belief:", type(obs0.opponent_belief).__name__)"""))

cells.append(md("""### §3.5 观测的原始形态：一张**扁平实体表**

在进入层级化 token 之前，先看引擎实际给你的东西。`env.native.observe()` 返回的原生 JSON 里，
战场是 `objects` 数组——**每行一个对象，没有任何嵌套容器**。塔、部队、建筑全部混在一张表里，
"哪些骷髅来自同一次骷髅军团"这类关系不体现为结构，只体现为每行的外键字段
（`source_entity` 指向派生者、`visible_target` 指向当前目标）。

下面的代码打几支部队上场，然后把**原生层**与**环境层**的两张扁平表都打印出来。
注意观察：① 塔的 `cardId=-1`（塔不是卡牌）；② 坐标是原生单位（x∈[0,18000], y∈[0,32000]，每格 100）；
③ `entities` 表里 `source_entity`/`visible_target` 就是建模时"关系边"的原料。"""))

cells.append(code("""# 打群卡上场（骷髅军团/哥布林团伙/亡灵大军……），每次出牌后立即快照，
# 保留"成员数最多的因果组"瞬间——大群会跨桥阵亡，必须在落地时抓住它。
from collections import defaultdict

best = None   # (最大组成员数, clusters, obs 快照)

def causal_key(entity):
    return entity.causal_group.handle if entity.causal_group is not None else None

def cluster_snapshot(obs):
    clusters = defaultdict(list)
    for entity in obs.entities:
        clusters[causal_key(entity)].append(entity)
    return clusters

SWARM_NAMES = {"SkeletonArmy", "GoblinGang", "MinionHorde", "Minions", "Skeletons", "Witch"}

def pick_play(mask):
    \"\"\"选一个可出动作：群卡优先、贵卡优先（群越大越值得快照）。返回 (action, card) 或 None。\"\"\"
    native_now = env.native.observe()
    rp0 = next(p for p in native_now["players"] if p["owner"] == 0)
    best_pick = None
    for i, ok in enumerate(mask.hand_slots):
        if not ok or str(i) not in mask.placement_masks:
            continue
        card = next((c for c in rp0["hand"] if c["handIndex"] == i), None)
        if card is None:
            continue
        rows = mask.placement_masks[str(i)]["row_major"]
        cells = [(cx, cy) for cy, row in enumerate(rows) for cx, v in enumerate(row) if v]
        if not cells:
            continue
        name = CARD_NAMES.get(card["cardId"], "")
        score = (0 if name in SWARM_NAMES else 1, -card["cost"])
        cx, cy = cells[len(cells) // 2]
        pick = (score, i, (cx, cy), card)
        if best_pick is None or pick[0] < best_pick[0]:
            best_pick = pick
    if best_pick is None:
        return None
    _, slot, (cx, cy), card = best_pick
    return ActionV1.play(owner=0, hand_slot=slot, grid=(cx, cy), card_id=card["cardId"])

for round_index in range(12):
    play = pick_play(env.action_mask(0))
    if play is not None:
        act_now(play)
        snap_obs, *_ = act_now(ActionV1.wait(0, ticks=2))     # 等 2 tick 让部署落地
        clusters = cluster_snapshot(snap_obs)
        size = max((len(v) for v in clusters.values()), default=0)
        if best is None or size > best[0]:
            best = (size, clusters, snap_obs)
    act_now(ActionV1.wait(0, ticks=20))

size, clusters, snap_obs = best
print(f"=== 因果组聚类（历史最大瞬间：{size} 个实体同组）===")
for key, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    ref = members[0].causal_group
    kind = ref.kind.value if ref is not None else "none"
    src = CARD_NAMES.get(ref.source_card_id, ref.source_card_id) if ref is not None else "-"
    names = [CARD_NAMES.get(m.card_id, m.card_id) for m in members]
    label = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
    print(f"  [{kind}] 来源={src} | owner={members[0].owner} | {len(members)} 个实体 | {label}")

obs0 = snap_obs
raw_now = env.native.observe()
print()
print("=== 原生层 objects 扁平表（该瞬间的快照）===")
print("objects 总数:", len(raw_now["objects"]), "（探针上限 256）")
print()
print("=== 环境层 entities 扁平表（该瞬间，前 16 行，含因果外键）===")
print(f"{'eid':>5} {'owner':>5} {'card_id':>9} {'kind':>10} {'hp/max':>13} {'source':>6} {'target':>6}")
for entity in list(obs0.entities)[:16]:
    src = entity.source_entity if entity.source_entity is not None else "-"
    tgt = entity.visible_target if entity.visible_target is not None else "-"
    hp = "-" if entity.hitpoints is None else f"{int(entity.hitpoints)}/{int(entity.max_hitpoints or 0)}"
    name = CARD_NAMES.get(entity.card_id, entity.card_id)
    print(f"{entity.entity_id:>5} {entity.owner:>5} {str(name):>9} {entity.entity_kind:>10} "
          f"{hp:>13} {str(src):>6} {str(tgt):>6}")
print("entities 总数:", len(obs0.entities))
print()
print("关键理解：数据本身是扁平的；'组'层级是建模侧用 causal_group/source_entity 等外键重建的——")
print("引擎不提供嵌套结构，但提供了重建任意结构所需的全部外键。群卡让这一结构清晰可见。")"""))

cells.append(md("""## §4 作者的观测分词器：`UniversalObservationTensorizerV4`

如果 NLP 的 tokenizer 把文本切成 token id 序列，那么 V4 的"观测分词器"就是把一帧战场切成
**六组结构化 token 集合 + 空间平面 + 动作候选**。词表来自编译目录（§`card_features` 那套内容寻址资产）：
122 张卡、技能、实体原型、效果各有自己的词表（PAD=0 / UNKNOWN=1 / 真实卡从 2 起按 ID 排序）。

**构造**：`production_semantic_bundle()` 加载全部编译目录（哈希校验）→
`build_episode_tensorizer_v4(env.episode_config, actor_owner=0)` 组装出绑定我方牌组的分词器
（内置一个 `DeterministicPublicTracker` 维护对手信念）。

**`tensorize(obs)` 的内部流水线**（源码顺序）：卡牌集合 → 塔 → 实体组 → 生效中效果 → 事件 →
动作候选 → 关系边 → 标量 → 空间平面 → 上一帧动作，最终打包成 `UniversalSemanticBatchV4`：

| 字段 | 形状（B=1） | 内容 |
| --- | --- | --- |
| `match_scalars` | `[1, 18]` | 18 个命名标量（剩余时间、双方圣水/皇冠、溢出计数、是否真红方…） |
| `own_cards` / `opponent_cards` | CardSet | 卡牌词表 ID + 运行时形态 + 角色位(精英/进化) + 运行时特征 + 掩码 |
| `towers` | TowerSet | 6 座塔固定槽位 |
| `groups` | BattleGroupSet | 场上实体组（每实体再展开子 token） |
| `active_effects` / `events` | Set | 生效中效果 token / 事件 token |
| `relation_edges` | EdgeSet | 组↔塔↔卡之间的关系边（供注意力读取） |
| `explicit_spatial_planes` | `[1, 14, 32, 18]` | 空间平面：己方通道 0-6 / 敌方 7-13，每面含地面/空军/建筑/弹道 4 类密度 |
| `previous_action` | Prev | 上一帧动作（循环策略的记忆输入之一） |
| `candidates` | ≤6 行 × 24 字段 | **动作候选**：variant、卡词表 ID、目标模式、落点、圣水、费用…… |

**精度保护细节**（`to_storage` 源码注释）：打包成 fp16 存储时，`candidates.elixir` 与 `candidates.cost`
被强制保持 fp32——"原生有效圣水 4.999 在 fp16 里会进位成 5.0，采样就可能放行一次总计恰好 5 圣水
的双开，而权威的原生解码器随后会拒绝它"。这类细节就是发行 checkpoint 能稳定自博弈的原因之一。"""))

cells.append(code("""from native_runner.training.v4.factory import build_episode_tensorizer_v4

tensorizer = build_episode_tensorizer_v4(env.episode_config, actor_owner=0)

# 分词器要"绑定一局"：tracker 需要双方的精确初始圣水（offline_duel 同款做法）。
# 因此重开一局，在 tick 1 从各自的视角抓取精确圣水，随后越过高模期再张量化。
reset_observations, obs0, native_state = new_match()
initial_elixir = {
    owner: next(p.elixir_exact for p in reset_observations[owner].players if p.owner == owner)
    for owner in (0, 1)
}
tensorizer.start_episode(reset_observations[0], initial_elixir=initial_elixir)

import dataclasses
batch = tensorizer.tensorize(obs0)
print(type(batch).__name__, "| batch_size:", batch.batch_size)
print("batch 字段:", [f.name for f in dataclasses.fields(batch)])
for name in ("match_scalars", "explicit_spatial_planes"):
    print(f"  {name}: {tuple(getattr(batch, name).shape)}")
print("  own_cards:", tuple(batch.own_cards.mask.shape), "| candidates:", tuple(batch.candidates.mask.shape))"""))

cells.append(code("""from native_runner.training.v4.config import MATCH_SCALAR_NAMES

# ① 18 个命名标量：打印非零项
for name, value in zip(MATCH_SCALAR_NAMES, batch.match_scalars[0].tolist()):
    if abs(value) > 1e-9:
        print(f"  {name} = {value:.4f}")

# ② 卡牌词表 ID 反查卡名（PAD=0, UNKNOWN=1, 真实卡 = 2 + 目录排序序号）
raw_ids = tensorizer.catalog.raw_card_ids
def card_name(vocab_id: int) -> str:
    return raw_ids[int(vocab_id) - 2] if 2 <= int(vocab_id) < 2 + len(raw_ids) else f"vocab{int(vocab_id)}"

active_rows = batch.own_cards.mask[0].nonzero().flatten().tolist()
print("own_cards:", [(int(batch.own_cards.card_vocab_id[0, r]), card_name(batch.own_cards.card_vocab_id[0, r])) for r in active_rows])

# ③ 空间平面：每通道非零格数（通道 0-6 己方 / 7-13 敌方；每面内 0=地面 1=空军 2=建筑 3=弹道）
planes = batch.explicit_spatial_planes[0]
for ch in range(planes.shape[0]):
    nonzero = int((planes[ch] != 0).sum())
    if nonzero:
        print(f"  通道 {ch} ({'己方' if ch < 7 else '敌方'}): {nonzero} 格非零")"""))

cells.append(code("""# ④ 动作候选：模型真正"选择"的对象 —— 每个决策点最多 6 个
active_candidates = int(batch.candidates.mask[0].sum())
print("候选数:", active_candidates, "/ 容量", batch.candidates.mask.shape[1])
for row in range(active_candidates):
    vocab_id = int(batch.candidates.effective_card_vocab_id[0, row])
    print(
        f"  候选 {row}: variant={int(batch.candidates.variant[0, row])}"
        f" card={card_name(vocab_id)} form={int(batch.candidates.effective_form[0, row])}"
        f" hand_slot={int(batch.candidates.native_hand_slot[0, row])}"
        f" target_mode={int(batch.candidates.target_mode[0, row])}"
    )
print("候选 elixir（fp32 保护字段）:", batch.candidates.elixir.tolist())
print("候选 cost（fp32 保护字段）:", batch.candidates.cost.tolist() if batch.candidates.cost.numel() <= 6 else "…")"""))

cells.append(md("""## 路线图（后续 cell）

1. ~~观测解剖~~ ✅ §3
2. ~~V4 观测分词器~~ ✅ §4
3. **奖励与终止** —— 潜在函数塑形（皇冠差 + 塔血）与零和反对称化；
4. **确定性实验** —— 同 seed 重置 + 相同动作序列 ⇒ 逐 bit 相同的对局（本项目最大卖点）；
5. **headless 快进** —— 不渲染一次推进上千 tick，吞吐量与训练一致；
6. **快照/回带** —— `snapshot-create` / `restore` 无损回滚到任意决策点；
7. **接入自己的 Actor-Critic** —— 用 §4 的 `tensorize` 作为特征入口，把 `candidates` 变成你的动作头。"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for i, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        compile(cell["source"], f"<cell {i}>", "exec")

path = "notebooks/get_familiar_with_firstlight_cr.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print("written:", path, "cells:", len(cells))
