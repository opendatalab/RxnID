# -*- coding: utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path

# --- 配置 ---
REVERSIBLE_FILES_JSON = ""
INPUT_JSON_PATH = ""
OUTPUT_JSONL_PATH = ""
IMAGE_BASE_PATH = ""
# 备用图片目录：主目录找不到时回退到此处
IMAGE_FALLBACK_BASE_PATH = ""
# 将路径不对（找不到图片）的 item 记录到此 JSON，便于后续重新生成
MISSING_ITEMS_JSON_PATH = ""
# 统一对齐 transform_yolo_detections_v2.py 的“复制式增强”配置（不做像素变换，仅样本重复）
# 基础倍数：对所有样本最低复制倍数（>=1）
BASE_DUPLICATION_FACTOR = 1
# 显式“可逆反应”文件名列表触发的复制
DUPLICATE_EXPLICIT_REVERSIBLE = True
EXPLICIT_REVERSIBLE_FACTOR = 1
# 反应布局左右/上下“反转”触发的复制
DUPLICATE_HORIZONTAL_REVERSED = True
HORIZONTAL_REVERSED_FACTOR = 1
DUPLICATE_VERTICAL_REVERSED = True
VERTICAL_REVERSED_FACTOR = 1
INCLUDE_IDT_LIST = True
REVERSE_KEEP_RATIO = 1  # 保留统计字段，实际复制倍数由上方 *REVERSED_FACTOR 控制

SYSTEM_PROMPT_TEMPLATE = """
You parse chemical reaction diagrams. Return only a JSON list.
Available IDTs (use these exact strings for structures):
{idt_list}
Rules:
- Classify into 'reactants', 'conditions', 'products'.
- Use {{"idt":"<id>"}} for known structures (from IDTs above); otherwise use {{"text":"<content>"}}.
- Pay attention to the arrow direction in the diagram when determining roles.
Example:
```json
[
  {{"reactants": [{{"idt": "E-3"}}, {{"text": "H2O"}}],
    "conditions": [{{"text": "heat"}}, {{"text": "80%"}}],
    "products": [{{"idt": "5"}}]
  }}
]
```
"""

USER_PROMPT = "<image>\nReturn only the JSON list."


def build_system_prompt(unique_identifiers):
    idt_list_str = json.dumps(unique_identifiers, ensure_ascii=False, separators=(",", ":"))
    return SYSTEM_PROMPT_TEMPLATE.format(idt_list=idt_list_str)


def find_image_path(file_name: str, base_path: Path) -> str:
    p = base_path / file_name
    return str(p) if p.exists() else ""


def create_reaction_part(bbox: dict) -> dict | None:
    if not bbox:
        return None
    text = str(bbox.get('text', '')).strip()
    if text:
        return {"text": text}
    ids = bbox.get('identifiers')
    if ids and isinstance(ids, list) and ids[0]:
        content = str(ids[0]).strip()
        if content:
            return {"idt": content}
    return None

def _is_explicit_reversible(file_name: str, reversible_patterns: set[str]) -> bool:
    base = Path(file_name).name
    stem = Path(file_name).stem
    for pat in reversible_patterns:
        if not pat:
            continue
        if pat in file_name or pat in base or pat in stem:
            return True
    return False


def _center_from_bbox(b):
    try:
        x, y, w, h = b.get('bbox', [None, None, None, None])
        if None in (x, y, w, h):
            return None
        return (x + w / 2.0, y + h / 2.0)
    except Exception:
        return None


def _has_reverse_layout(image_info: dict, bboxes_map: dict) -> tuple[bool, bool]:
    for reaction in image_info.get('reactions', []):
        react_b = []
        prod_b = []
        for rid in reaction.get('reactants', []):
            b = bboxes_map.get(rid, {})
            if 'bbox' in b:
                react_b.append(b['bbox'])
        for pid in reaction.get('products', []):
            b = bboxes_map.get(pid, {})
            if 'bbox' in b:
                prod_b.append(b['bbox'])
        if not react_b or not prod_b:
            continue
        rx = sum(x + w/2 for x,y,w,h in react_b) / len(react_b)
        ry = sum(y + h/2 for x,y,w,h in react_b) / len(react_b)
        px = sum(x + w/2 for x,y,w,h in prod_b) / len(prod_b)
        py = sum(y + h/2 for x,y,w,h in prod_b) / len(prod_b)
        avg_rw = sum(w for x,y,w,h in react_b) / len(react_b)
        avg_rh = sum(h for x,y,w,h in react_b) / len(react_b)
        dx = rx - px
        dy = ry - py
        horiz = (dx > max(avg_rw, 60)) and (abs(dy) < min(avg_rh * 0.4, 30))
        vert  = (dy > max(avg_rh, 60)) and (abs(dx) < min(avg_rw * 0.4, 30))
        if horiz or vert:
            return horiz, vert
    return False, False


def create_training_data():
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载输入失败: {e}")
        return
    if REVERSIBLE_FILES_JSON and os.path.exists(REVERSIBLE_FILES_JSON):
        try:
            with open(REVERSIBLE_FILES_JSON, 'r', encoding='utf-8') as f:
                reversible_files_set = set(json.load(f))
        except Exception as e:
            print(f"[WARN] 加载 reversible 文件失败，将按空集合处理: {e}")
            reversible_files_set = set()
    else:
        reversible_files_set = set()

    image_base_path = Path(IMAGE_BASE_PATH)
    fallback_image_base_path = Path(IMAGE_FALLBACK_BASE_PATH or IMAGE_BASE_PATH)
    output_parent = os.path.dirname(OUTPUT_JSONL_PATH)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    output_count = 0
    base_count = 0
    dup_explicit_count = 0
    dup_h_reverse_count = 0
    dup_v_reverse_count = 0
    missing_img_count = 0
    missing_img_examples = []
    missing_items = []  # 收集找不到图片的 image_info
    with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f_out:
        for image_info in tqdm(data.get('images', []), desc="Creating mix_idt training set"):
            file_name = image_info.get('file_name')
            if not file_name:
                continue
            img_path = find_image_path(Path(file_name).name, image_base_path)
            if not img_path:
                # 主目录未找到时，尝试备用目录
                img_path = find_image_path(Path(file_name).name, fallback_image_base_path)
            if not img_path:
                missing_img_count += 1
                if len(missing_img_examples) < 10:
                    missing_img_examples.append(str(Path(file_name).name))
                # 记录完整条目，便于后续重新生成
                missing_items.append(image_info)
                continue

            bboxes_map = {bbox['id']: bbox for bbox in image_info.get('bboxes', []) if 'id' in bbox}

            # 只收集“没有非空文本”的结构/文本框上的identifier，避免与文本优先规则冲突
            unique_identifiers = sorted({
                str((bbox.get('identifiers') or [""])[0]).strip()
                for bbox in image_info.get('bboxes', [])
                if bbox.get('category_id') in [1, 2]
                   and (bbox.get('identifiers') or [""])[0]
                   and not str(bbox.get('text', '')).strip()
            })

            system_prompt = build_system_prompt(unique_identifiers if INCLUDE_IDT_LIST else [])

            # 显式可逆文件名命中
            need_dup_explicit = _is_explicit_reversible(file_name, reversible_files_set)
            # 反转布局检测（左右/上下）
            has_h_reverse, has_v_reverse = _has_reverse_layout(image_info, bboxes_map)

            # 复制倍数对齐 transform_yolo_detections_v2：取各触发因子的最大值，并乘以基础倍数
            duplication_factor = max(1, BASE_DUPLICATION_FACTOR)
            if DUPLICATE_EXPLICIT_REVERSIBLE and need_dup_explicit:
                duplication_factor = max(duplication_factor, EXPLICIT_REVERSIBLE_FACTOR)
                dup_explicit_count += 1
            if DUPLICATE_HORIZONTAL_REVERSED and has_h_reverse:
                duplication_factor = max(duplication_factor, HORIZONTAL_REVERSED_FACTOR)
                dup_h_reverse_count += 1
            if DUPLICATE_VERTICAL_REVERSED and has_v_reverse:
                duplication_factor = max(duplication_factor, VERTICAL_REVERSED_FACTOR)
                dup_v_reverse_count += 1

            base_count += 1

            for _ in range(duplication_factor):
                assistant_content_list = []
                for reaction in image_info.get('reactions', []):
                    new_reaction = {"reactants": [], "conditions": [], "products": []}
                    for role in ["reactants", "conditions", "products"]:
                        for bbox_id in reaction.get(role, []):
                            part = create_reaction_part(bboxes_map.get(bbox_id))
                            if part:
                                new_reaction[role].append(part)
                    if any(new_reaction.values()):
                        assistant_content_list.append(new_reaction)
                if not assistant_content_list:
                    continue
                assistant_content_str = json.dumps(assistant_content_list, ensure_ascii=False, separators=(",", ":"))

                item = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": USER_PROMPT},
                        {"role": "assistant", "content": assistant_content_str},
                    ],
                    "images": [img_path],
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_count += 1

    print(f"完成，生成 {output_count} 条样本 -> {OUTPUT_JSONL_PATH}")
    print("--- Augmentation stats (mix_idt) ---")
    print(f"Base images: {base_count}")
    print(f"Triggered explicit reversible: {dup_explicit_count}")
    print(f"Triggered horizontal reversed: {dup_h_reverse_count}")
    print(f"Triggered vertical reversed: {dup_v_reverse_count}")
    print(f"[STATS] Missing images: {missing_img_count}")
    if missing_img_examples:
        print(f"[STATS] Missing image examples (up to 10): {missing_img_examples}")
    # 输出缺失项 JSON
    try:
        if missing_items:
            missing_items_path = MISSING_ITEMS_JSON_PATH or str(Path(OUTPUT_JSONL_PATH).with_suffix(".missing_items.json"))
            missing_parent = os.path.dirname(missing_items_path)
            if missing_parent:
                os.makedirs(missing_parent, exist_ok=True)
            with open(missing_items_path, 'w', encoding='utf-8') as mf:
                json.dump({"images": missing_items}, mf, ensure_ascii=False, indent=2)
            print(f"[WRITE] Missing items JSON -> {missing_items_path} (count={len(missing_items)})")
        else:
            print("[WRITE] No missing items. Skip writing missing-items JSON.")
    except Exception as e:
        print(f"[WARN] 写入缺失项 JSON 失败: {e}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Create IdtVP SFT JSONL from Mid-Mapper annotated JSON.")
    parser.add_argument("--input_json_path", required=True, help="Final JSON with molecule identifiers.")
    parser.add_argument("--output_jsonl_path", required=True, help="Output SFT JSONL path.")
    parser.add_argument("--image_base_path", required=True, help="Directory containing clean IDT-rendered images.")
    parser.add_argument("--image_fallback_base_path", default="", help="Optional fallback image directory.")
    parser.add_argument("--missing_items_json_path", default="", help="Where to write records whose image path is missing.")
    parser.add_argument("--reversible_files_json", default="", help="Optional JSON list of reversible-reaction file patterns.")
    parser.add_argument("--base_duplication_factor", type=int, default=1)
    parser.add_argument("--duplicate_explicit_reversible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explicit_reversible_factor", type=int, default=1)
    parser.add_argument("--duplicate_horizontal_reversed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizontal_reversed_factor", type=int, default=1)
    parser.add_argument("--duplicate_vertical_reversed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vertical_reversed_factor", type=int, default=1)
    parser.add_argument("--include_idt_list", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _apply_args(args):
    global REVERSIBLE_FILES_JSON, INPUT_JSON_PATH, OUTPUT_JSONL_PATH
    global IMAGE_BASE_PATH, IMAGE_FALLBACK_BASE_PATH, MISSING_ITEMS_JSON_PATH
    global BASE_DUPLICATION_FACTOR, DUPLICATE_EXPLICIT_REVERSIBLE, EXPLICIT_REVERSIBLE_FACTOR
    global DUPLICATE_HORIZONTAL_REVERSED, HORIZONTAL_REVERSED_FACTOR
    global DUPLICATE_VERTICAL_REVERSED, VERTICAL_REVERSED_FACTOR, INCLUDE_IDT_LIST

    REVERSIBLE_FILES_JSON = args.reversible_files_json
    INPUT_JSON_PATH = args.input_json_path
    OUTPUT_JSONL_PATH = args.output_jsonl_path
    IMAGE_BASE_PATH = args.image_base_path
    IMAGE_FALLBACK_BASE_PATH = args.image_fallback_base_path or args.image_base_path
    MISSING_ITEMS_JSON_PATH = args.missing_items_json_path
    BASE_DUPLICATION_FACTOR = args.base_duplication_factor
    DUPLICATE_EXPLICIT_REVERSIBLE = args.duplicate_explicit_reversible
    EXPLICIT_REVERSIBLE_FACTOR = args.explicit_reversible_factor
    DUPLICATE_HORIZONTAL_REVERSED = args.duplicate_horizontal_reversed
    HORIZONTAL_REVERSED_FACTOR = args.horizontal_reversed_factor
    DUPLICATE_VERTICAL_REVERSED = args.duplicate_vertical_reversed
    VERTICAL_REVERSED_FACTOR = args.vertical_reversed_factor
    INCLUDE_IDT_LIST = args.include_idt_list


if __name__ == '__main__':
    _apply_args(_parse_args())
    create_training_data()
