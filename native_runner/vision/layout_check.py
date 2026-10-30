"""全屏布局校验：MuMu 开机、训练开跑前跑一次，人工确认两个裁剪框。

用法:
    python -m native_runner.vision.layout_check [serial]

输出 runs 旁边一张 fullframe_layout_check.png：绿框 = ARENA_CROP（战场
18x32 网格投影，喂给世界模型），紫框 = BOTTOM_CROP（手牌+圣水条，P1
视觉通道源）。人工核对：绿框正套战场、紫框正套手牌区、底部为单侧
手牌（个人对战 HUD，非回放观战布局）。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from .capture import BOTTOM_CROP, ARENA_CROP, capture_png, scale_crop


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else None
    png = capture_png("adb", serial)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for rect_ref, color, label in (
        (ARENA_CROP, (0, 255, 0), "ARENA_CROP (world model input)"),
        (BOTTOM_CROP, (255, 0, 255), "BOTTOM_CROP (P1 elixir/hand source)"),
    ):
        rect = scale_crop(rect_ref, img.size)
        draw.rectangle(rect, outline=color, width=3)
        draw.text((rect[0] + 6, rect[1] + 6), label, fill=color)
    out_path = Path(__file__).resolve().parents[3] / "fullframe_layout_check.png"
    out.save(out_path)
    print(f"saved {out_path} (screen {img.size})")
    print("check: green rect == battlefield; magenta rect == own hand/elixir strip;")
    print("       bottom shows ONE hand (personal view, not replay spectator layout)")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
