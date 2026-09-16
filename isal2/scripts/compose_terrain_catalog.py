"""Arrange Isaac Sim terrain captures into labeled contact sheets."""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

folder = Path(sys.argv[1])
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
font_path = "C:/Windows/Fonts/msyh.ttc"
font = lambda size: ImageFont.truetype(font_path, size)
labels = {
    "inv_pyramid_stairs_25": ("下行阶梯 · 踏面 25 cm", "step"),
    "inv_pyramid_stairs_35": ("下行阶梯 · 踏面 35 cm", "step"),
    "pyramid_stairs_25": ("上行阶梯 · 踏面 25 cm", "step"),
    "pyramid_stairs_35": ("上行阶梯 · 踏面 35 cm", "step"),
    "slope": ("上坡", "坡度参数 0.1 → 0.3"),
    "inv_slope": ("下坡", "坡度参数 0.1 → 0.3"),
    "grid": ("随机方块 · 连续底面", "grid"),
    "random_rough": ("随机粗糙地面", "起伏 -2～4 cm；量化步长 2 cm"),
    "flat": ("平地", "无障碍"),
    "high_platform": ("双层凹坑 / 平台", "pit depth 参数 10 → 30 cm"),
    "star": ("星形窄梁", "梁宽 40 → 25 cm；梁高 / 坑深 10 m"),
    "gap": ("环形沟", "沟宽 10 → 30 cm；中央平台 2 m"),
    "stepping_stones": ("踏石", "石宽 40 → 30 cm；间距 10 → 20 cm"),
}
for preset in ("rough", "rough_hard"):
    entries = [x for x in manifest["terrains"] if x["preset"] == preset]
    cols, cell_w, cell_h, margin = 3, 560, 630, 30
    rows = (len(entries) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w + 2 * margin, rows * cell_h + 190), "#eef2f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 22), f"ISAL2  /  {preset}", fill="#132838", font=font(43))
    renderer = "Isaac Sim RTX" if manifest.get("renderer") == "Isaac Sim RTX" else "原始网格 / MuJoCo 离屏"
    draw.text((margin, 83), f"当前配置实际生成 · {renderer} · 每块 8 × 8 m · 难度 0.5 · 种子 42", fill="#435566", font=font(22))
    draw.text((margin, 119), "卡片等大展示各类型，不代表采样比例；颜色仅用于区分。参数箭头表示从容易到困难。", fill="#435566", font=font(22))
    for i, item in enumerate(entries):
        x = margin + i % cols * cell_w
        y = 172 + i // cols * cell_h
        draw.rounded_rectangle((x, y, x + cell_w - 16, y + cell_h - 18), 15, fill="white")
        name = item["name"]
        title, detail = labels[name]
        draw.text((x + 15, y + 13), title, fill="#173749", font=font(25))
        draw.text((x + cell_w - 93, y + 15), f'{item["proportion"]:.0%}', fill="#9a5c20" if preset.endswith("hard") else "#23657c", font=font(25))
        draw.text((x + 15, y + 50), name, fill="#647383", font=font(18))
        filename = item["image"]
        if name in ("star", "stepping_stones"):
            filename = filename.replace(".png", "__top.png")
        pic = Image.open(folder / filename).convert("RGB").resize((528, 462), Image.Resampling.LANCZOS)
        canvas.paste(pic, (x + 8, y + 83))
        if detail == "step":
            detail = "台阶高 5 → 20 cm" if preset.endswith("hard") else "台阶高 5 → 15 cm"
        if detail == "grid":
            detail = "方块宽 45 cm；高度参数 5 → " + ("20 cm" if preset.endswith("hard") else "15 cm")
        draw.text((x + 15, y + 554), detail, fill="#324c60", font=font(21))
        if name == "stepping_stones":
            draw.text((x + 15, y + 580), "坑深使用 Isaac Lab 默认值：10 m", fill="#9a5c20", font=font(19))
        elif name == "star":
            draw.text((x + 15, y + 580), "俯视图；12 根贯穿梁，中央平台直径 2 m", fill="#647383", font=font(18))
    output = folder / f"{preset}_catalog.png"
    canvas.save(output)
    print(output)
