import argparse
import json, re, os, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

CONFIG = {
    "JSON_RESULTS_DIR": "",
    "LOCAL_OUTPUT_ROOT": "outputs/mid_mapper",
    "IMAGE_EXTS": [".jpg", ".jpeg", ".png"],
    "IMAGE_ROOT": "",
    "NEW_JSON_FORMAT": True,
    "DRY_RUN": False,
    "FONT_PATH": "",
    "IDT_MIN_FONT": 14,
    "IDT_MAX_FONT": 42,
    "IDT_IMG_HEIGHT_FONT_RATIO": 0.02,
    "IDT_PADDING": 2,
    "IDT_ALLOWED_SIDES": ["bottom", "left", "right"],
    "IDT_SIDE_GAP": 5,
    "IDT_VERTICAL_GAP": 5,
    "IDT_CAND_INK_MAX_RATIO": 0.01,
    "IDT_INTRUDE_DARK_THRESHOLD": 180,
    "PREVIEW": True,
    "PREVIEW_TEXT_COLOR_VARIANTS": [("black",(0,0,0)), ("red",(255,0,0))],
    "OUT_IMAGE_DIR_ADDED": "annotated_images_added",
    "DEBUG_LOG": True,
    "IDT_MAX_CHAR_LEN": 0,
    "USE_BOLD_FONT": True,
    "FONT_BOLD_PATH": "",
    "BOLD_DRAW_STROKE": 1,
    "IDT_CHAR_SPACING_RATIO": 0.08,
    "IDT_CHAR_SPACING_MIN": 1,
    "IDT_CHAR_SPACING_MAX": 6,
    "IDT_ENSURE_FIT": True,
    "IDT_BOX_MIN_W": 0,
    "IDT_BOX_MIN_H": 0,
    "IDT_REPORT_OVERFLOW": True,
    "IDT_TRY_INSIDE": True,
    "IDT_INSIDE_MARGIN": 4,
    "IDT_SIDE_OFFSETS": [-0.35, 0.0, 0.35],
    "IDT_INNER_MAX_INK_RATIO": 0.005,
    "IDT_INNER_GRID_USE": True,
    "IDT_INNER_GRID_POSITIONS": [0.15,0.5,0.85],
    "IDT_CENTER_FIRST": False,
    "IDT_EDGE_CLEARANCE": 6,
    "IDT_VERTICAL_GAP_NEAR": 4,
    "IDT_SIDE_GAP_NEAR": 5,
    "IDT_AVOID_TOP": True,
    "IDT_INSIDE_PREFER_BOTTOM": True,
    "IDT_INSIDE_BOTTOM_WEIGHT": 0.6,
    "IDT_FONT_SHRINK_STEP": 2,
    "PREVIEW_HIGHLIGHT_MOL_WITH_IDT": True,
    "PREVIEW_MOL_ORIG_HAS_IDT_COLOR": (255,165,0),
    "PREVIEW_MOL_ORIG_NO_IDT_COLOR": (0,0,255),
    "IDT_INSIDE_HUG_EDGE": True,
    "IDT_CENTER_SKIP_RATIO": 0.55,
    "IDT_INSIDE_HUG_ORDER": [
        "bottom-right","bottom-left","bottom-center",
        "right-center","left-center"
    ],
    "FORCE_SIMPLE_NUMERIC": False,
    "IDT_SIDE_STRIP_WIDTH": 3,
    "IDT_SIDE_STRIP_MAX_INK": 0.04,
    "IDT_SECOND_VIEW_PLAIN": True,
    "IDT_STRADDLE_ENABLE": True,
    "IDT_STRADDLE_SIDES": ["bottom"],
    "IDT_STRADDLE_FRACTION_BOTTOM": 0.45,
    "IDT_STRADDLE_FRACTION_LEFT": 0.50,
    "IDT_STRADDLE_FRACTION_RIGHT": 0.50,
    "IDT_EDGE_BAND_PIXELS": 6,
    "IDT_EDGE_BAND_MAX_INK": 0.07,
    "IDT_BOTTOM_INSIDE_OFFSETS": [0.0, -0.22, 0.22],
    "IDT_OUT_BOTTOM_OFFSETS": [0.0, -0.2, 0.2],
    "IDT_ALLOW_TOP_FALLBACK": True,
    "IDT_INSIDE_ADJ_BAND_FRAC": 0.6,
    "IDT_INSIDE_ADJ_MAX_INK": 0.08,
    "IDT_INSIDE_BOTTOM_ABOVE_FRAC": 0.55,
    "IDT_INSIDE_BOTTOM_ABOVE_MAX_INK": 0.05,
    "IDT_OUT_BOTTOM_BELOW_FRAC": 0.9,
    "IDT_OUT_BOTTOM_BELOW_MAX_INK": 0.06,
    "IDT_OUT_BOTTOM_MIN_GAP": 4,
    "ALWAYS_PREVIEW": True,
    "ALWAYS_SAVE_JSON": True,
    "IDT_FALLBACK_MAX_LEN": 2,
    "IDT_PREFER_EXTERNAL_BOTTOM": True,

    # 合并模式
    "MERGED_INPUT_JSON": "",
    "OUTPUT_JSON_MERGED": "",
    "IMAGE_ALT_ROOTS": [],
    "INPUT_DPI": 400,
    "TARGET_DPI": 400,
    "MOL_CATEGORY_ID": 1,
    "IDT_CATEGORY_ID": 2,
    "SKIP_CATEGORY_ID": 3,
    "PREVIEW_OUT_DIR": "merged_previews",
    "PREVIEW_OUT_DIR_ANNOTATED": "",
    "PREVIEW_OUT_DIR_CLEAN": "",

    # === 新增 强制放置相关配置 ===
    "IDT_FORCE_PLACE": True,                # 启用强制放置阶段
    "IDT_MIN_FONT_FORCE": 8,                # 强制阶段允许的最小字号
    "IDT_FORCE_PLACEMENT_STRATEGY": ["bottom_center","inside_center","top_center"], # 强制放置尝试顺序
    "IDT_FORCE_IGNORE_INK": True,           # 强制放置时忽略墨迹检测
    "IDT_FORCE_ALLOW_OVERLAP": False,       # 强制放置是否允许与其他 obstacle 重叠
    "IDT_FORCE_OVERLAP_PADDING": 1,         # 强制放置允许的最小像素间隔
}

def _normalize_filename(fn: str) -> str:
    if not fn:
        return fn
    if fn.startswith("figure/"):
        fn = fn[len("figure/"):]
    elif fn.startswith("table/"):
        fn = fn[len("table/"):]
    base = os.path.splitext(fn)[0]
    return base + ".png"

def find_image_path(file_name: str, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        p = root / file_name
        if p.exists():
            return p
    base, _ = os.path.splitext(file_name)
    for root in roots:
        for ext in ['.png', '.jpg', '.jpeg']:
            p = root / (base + ext)
            if p.exists():
                return p
    return None

# ==============================================================================
# 命名方案
# ==============================================================================
def analyze_scheme(identifiers: List[str]) -> Dict[str, Any]:
    if not identifiers:
        return {"type": "numeric", "max": 0}

    if CONFIG.get("FORCE_SIMPLE_NUMERIC", False):
        nums = [int(i) for i in identifiers if i.isdecimal()]
        return {"type": "numeric", "max": max(nums) if nums else 0}

    from collections import Counter
    nums = []
    alphas = []
    alpha_prefix_nums = []
    num_prefix_alphas = []

    for s in identifiers:
        s = s.strip()
        if not s: continue
        if s.isdecimal():
            nums.append(int(s))
        elif s.isalpha():
            alphas.append(s)
        else:
            match1 = re.match(r"^([a-zA-Z]+)(\d+)$", s)
            if match1:
                alpha_prefix_nums.append((match1.group(1), int(match1.group(2))))
                continue
            match2 = re.match(r"^(\d+)([a-zA-Z]+)$", s)
            if match2:
                num_prefix_alphas.append((int(match2.group(1)), match2.group(2)))
                continue

    schemes = {
        "numeric": len(nums),
        "alpha": len(alphas),
        "alpha_prefix": len(alpha_prefix_nums),
        "num_prefix": len(num_prefix_alphas)
    }
    if not any(schemes.values()):
        return {"type": "numeric", "max": 0}
    dominant_scheme = max(schemes, key=schemes.get)
    try:
        if dominant_scheme == "numeric":
            return {"type": "numeric", "max": max(nums) if nums else 0}
        if dominant_scheme == "alpha":
            return {"type": "alpha", "max": max(alphas) if alphas else ''}
        if dominant_scheme == "alpha_prefix":
            from collections import Counter
            prefix_counts = Counter(p for p, n in alpha_prefix_nums)
            common_prefix = prefix_counts.most_common(1)[0][0]
            max_num = max(n for p, n in alpha_prefix_nums if p == common_prefix)
            return {"type": "alpha_prefix", "prefix": common_prefix, "max": max_num}
        if dominant_scheme == "num_prefix":
            from collections import Counter
            suffix_counts = Counter(s for n, s in num_prefix_alphas)
            common_suffix = suffix_counts.most_common(1)[0][0]
            max_num = max(n for n, s in num_prefix_alphas if s == common_suffix)
            return {"type": "num_prefix", "suffix": common_suffix, "max": max_num}
    except (IndexError, ValueError):
        return {"type": "numeric", "max": 0}
    return {"type": "numeric", "max": 0}

def next_label(scheme: Dict[str, Any]) -> str:
    scheme_type = scheme.get("type", "numeric")
    if scheme_type == "numeric":
        scheme["max"] += 1
        return str(scheme["max"])
    elif scheme_type == "alpha":
        if not scheme["max"]:
            scheme["max"] = 'a'
        else:
            if scheme["max"] == 'z': scheme["max"] = 'aa'
            elif scheme["max"] == 'Z': scheme["max"] = 'AA'
            else:
                last_char = scheme["max"][-1]
                base = scheme["max"][:-1]
                if 'a' <= last_char < 'z' or 'A' <= last_char < 'Z':
                    scheme["max"] = base + chr(ord(last_char) + 1)
                else:
                    scheme["max"] += 'a' if 'a' <= last_char <= 'z' else 'A'
        return scheme["max"]
    elif scheme_type == "alpha_prefix":
        scheme["max"] += 1
        return f"{scheme['prefix']}{scheme['max']}"
    elif scheme_type == "num_prefix":
        scheme["max"] += 1
        return f"{scheme['max']}{scheme['suffix']}"
    return ""

# ==============================================================================
# 图像/字体
# ==============================================================================
_gray_images_cache = {}
def _get_gray_image(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode == 'L':
        return pil_img
    if id(pil_img) in _gray_images_cache:
        return _gray_images_cache[id(pil_img)]
    gray_img = pil_img.convert('L')
    _gray_images_cache[id(pil_img)] = gray_img
    return gray_img

_font_cache = {}
def load_font(size: int) -> ImageFont.FreeTypeFont:
    size = int(size)
    use_bold = CONFIG.get("USE_BOLD_FONT", False)
    font_path = CONFIG["FONT_PATH"]
    if use_bold and os.path.exists(CONFIG["FONT_BOLD_PATH"]):
        font_path = CONFIG["FONT_BOLD_PATH"]
    cache_key = (font_path or "DejaVuSans.ttf", size)
    if cache_key in _font_cache:
        return _font_cache[cache_key]
    candidates = [font_path] if font_path else []
    candidates.append("DejaVuSans.ttf")
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, size)
            _font_cache[cache_key] = font
            return font
        except (OSError, IOError):
            continue
    if CONFIG["DEBUG_LOG"] and font_path:
        print(f"[WARN] 字体 '{font_path}' 加载失败, 使用默认字体")
    font = ImageFont.load_default()
    _font_cache[cache_key] = font
    return font

def calc_char_spacing(font_size: int) -> int:
    ratio = CONFIG.get("IDT_CHAR_SPACING_RATIO", 0)
    if ratio <= 0: return 0
    min_sp = CONFIG.get("IDT_CHAR_SPACING_MIN", 1)
    max_sp = CONFIG.get("IDT_CHAR_SPACING_MAX", 4)
    spacing = int(round(font_size * ratio))
    return max(min_sp, min(max_sp, spacing))

def measure_text_with_spacing(font: ImageFont.FreeTypeFont, text: str, font_size: int, spacing: int = 0) -> Tuple[int, int]:
    if not text: return 0, 0
    if spacing == 0 and not CONFIG.get("BOLD_DRAW_STROKE", 0) > 0:
        try:
            l, t, r, b = font.getbbox(text)
            return r - l, b - t
        except (TypeError, AttributeError):
            return font.getsize(text)
    canvas = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(canvas)
    stroke = CONFIG.get("BOLD_DRAW_STROKE", 0)
    total_width = 0
    max_height = 0
    if hasattr(font, 'getbbox'):
        for i, char in enumerate(text):
            l, t, r, b = font.getbbox(char)
            char_w = r - l
            char_h = b - t
            total_width += char_w
            if i < len(text) - 1:
                total_width += spacing
            max_height = max(max_height, char_h)
        if stroke > 0:
            total_width += stroke * 2
            max_height += stroke * 2
    else:
        for i, char in enumerate(text):
            char_w, char_h = draw.textsize(char, font=font)
            total_width += char_w
            if i < len(text) - 1:
                total_width += spacing
            max_height = max(max_height, char_h)
        if stroke > 0:
            total_width += stroke * 2
            max_height += stroke * 2
    return total_width, max_height

def draw_text_with_spacing(draw: ImageDraw.ImageDraw, pos: Tuple[float, float], text: str, font: ImageFont.FreeTypeFont, fill, spacing: int = 0, stroke: int = 0, stroke_fill=None):
    x, y = pos
    if spacing == 0 and stroke == 0:
        draw.text((x, y), text, font=font, fill=fill)
        return
    current_x = x
    for char in text:
        draw.text((current_x, y), char, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        try:
            l, t, r, b = font.getbbox(char)
            char_w = r - l
        except (TypeError, AttributeError):
            char_w, _ = font.getsize(char)
        current_x += char_w + spacing

def check_ink(gray_img: Image.Image, box: Tuple[int, int, int, int], threshold: int) -> float:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2: return 1.0
    cropped = gray_img.crop(box)
    stat = cropped.getextrema()
    if not stat or stat[0] >= threshold: return 0.0
    count = sum(1 for pixel in cropped.getdata() if pixel < threshold)
    return count / (cropped.width * cropped.height)

# ==============================================================================
# 放置逻辑
# ==============================================================================
def find_placement_enhanced(
    struct_box: Dict[str, Any], box_w: int, box_h: int, img_w: int, img_h: int,
    obstacles: List[Tuple[int, int, int, int]], gray_img: Image.Image,
    self_obstacle: Tuple[int, int, int, int], try_inside: bool = True
) -> Optional[Tuple[int, int]]:
    sx, sy, sw, sh = struct_box["x"], struct_box["y"], struct_box["width"], struct_box["height"]
    dark_thresh = CONFIG["IDT_INTRUDE_DARK_THRESHOLD"]

    def is_overlapping(x1, y1, w1, h1, check_obstacles):
        for ox1, oy1, ox2, oy2 in check_obstacles:
            if not (x1 + w1 <= ox1 or x1 >= ox2 or y1 + h1 <= oy1 or y1 >= oy2):
                return True
        return False

    def try_positions(positions: List[Dict[str, Any]], other_obstacles: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
        for cand_info in positions:
            x, y = int(cand_info["pos"][0]), int(cand_info["pos"][1])
            cand_type = cand_info["type"]
            if x < 0 or y < 0 or x + box_w > img_w or y + box_h > img_h: continue
            if is_overlapping(x, y, box_w, box_h, other_obstacles): continue
            cand_box = (x, y, x + box_w, y + box_h)
            ink_ratio = check_ink(gray_img, cand_box, dark_thresh)
            if ink_ratio <= CONFIG.get("IDT_CAND_INK_MAX_RATIO", 0.01):
                if cand_type.startswith("out-bottom"):
                    below_h = int(box_h * CONFIG.get("IDT_OUT_BOTTOM_BELOW_FRAC", 0.9))
                    below_box = (x, y + box_h, x + box_w, y + box_h + below_h)
                    if check_ink(gray_img, below_box, dark_thresh) > CONFIG.get("IDT_OUT_BOTTOM_BELOW_MAX_INK", 0.06):
                        continue
                    return x, y + CONFIG.get("IDT_OUT_BOTTOM_MIN_GAP", 4)
                if cand_type.startswith("in-"):
                    adj_h = int(box_h * CONFIG.get("IDT_INSIDE_ADJ_BAND_FRAC", 0.6))
                    adj_box_above = (x, y - adj_h, x + box_w, y)
                    if check_ink(gray_img, adj_box_above, dark_thresh) > CONFIG.get("IDT_INSIDE_ADJ_MAX_INK", 0.08):
                        continue
                    if "bottom" in cand_type and CONFIG.get("IDT_INSIDE_BOTTOM_ABOVE_MAX_INK", 0) > 0:
                        above_h = int(box_h * CONFIG.get("IDT_INSIDE_BOTTOM_ABOVE_FRAC", 0.55))
                        above_box = (x, y - above_h, x + box_w, y)
                        if check_ink(gray_img, above_box, dark_thresh) > CONFIG["IDT_INSIDE_BOTTOM_ABOVE_MAX_INK"]:
                            continue
                return x, y
        return None

    external_bottom_cands = []
    internal_cands = []
    external_side_cands = []
    other_obstacles = [obs for obs in obstacles if obs != self_obstacle]
    margin = CONFIG["IDT_INSIDE_MARGIN"]

    if CONFIG.get("IDT_STRADDLE_ENABLE", False) and "bottom" in CONFIG.get("IDT_STRADDLE_SIDES", []):
        frac = CONFIG.get("IDT_STRADDLE_FRACTION_BOTTOM", 0.45)
        y = sy + sh - int(box_h * frac)
        for offset_ratio in CONFIG.get("IDT_BOTTOM_INSIDE_OFFSETS", [0.0]):
            x = sx + (sw - box_w) / 2 + offset_ratio * sw
            external_bottom_cands.append({"pos": (x, y), "type": "out-bottom-straddle"})
    y = sy + sh + CONFIG.get("IDT_VERTICAL_GAP_NEAR", 2)
    for offset_ratio in CONFIG.get("IDT_OUT_BOTTOM_OFFSETS", [0.0]):
        x = sx + (sw - box_w) / 2 + offset_ratio * sw
        external_bottom_cands.append({"pos": (x, y), "type": "out-bottom-near"})
    y = sy + sh + CONFIG.get("IDT_VERTICAL_GAP", 5)
    for offset_ratio in CONFIG.get("IDT_SIDE_OFFSETS", [0.0]):
        x = sx + (sw - box_w) / 2 + offset_ratio * sw
        external_bottom_cands.append({"pos": (x, y), "type": "out-bottom-far"})

    if try_inside and CONFIG.get("IDT_TRY_INSIDE", False) and box_w < sw and box_h < sh:
        temp_internal = []
        if CONFIG.get("IDT_INSIDE_HUG_EDGE", False):
            skip_center = box_w / sw < CONFIG.get("IDT_CENTER_SKIP_RATIO", 0.55) and box_h / sh < CONFIG.get("IDT_CENTER_SKIP_RATIO", 0.55)
            order = CONFIG.get("IDT_INSIDE_HUG_ORDER", [])
            hug_cands = {
                "bottom-right": (sx + sw - box_w - margin, sy + sh - box_h - margin),
                "bottom-left": (sx + margin, sy + sh - box_h - margin),
                "bottom-center": (sx + (sw - box_w) / 2, sy + sh - box_h - margin),
                "right-center": (sx + sw - box_w - margin, sy + (sh - box_h) / 2),
                "left-center": (sx + margin, sy + (sh - box_h) / 2),
                "center": (sx + (sw - box_w) / 2, sy + (sh - box_h) / 2)
            }
            if skip_center: hug_cands.pop("center", None)
            for pos_key in order:
                if pos_key in hug_cands:
                    temp_internal.append({"pos": hug_cands[pos_key], "type": f"in-{pos_key}"})
        if CONFIG.get("IDT_INNER_GRID_USE", False):
            for ry in CONFIG.get("IDT_INNER_GRID_POSITIONS", [0.5]):
                for rx in CONFIG.get("IDT_INNER_GRID_POSITIONS", [0.5]):
                    temp_internal.append({"pos": (sx + (sw - box_w) * rx, sy + (sh - box_h) * ry), "type": "in-grid"})
        if not CONFIG.get("IDT_CENTER_FIRST", False):
            temp_internal.append({"pos": (sx + (sw - box_w) / 2, sy + (sh - box_h) / 2), "type": "in-center"})
        else:
            temp_internal.insert(0, {"pos": (sx + (sw - box_w) / 2, sy + (sh - box_h) / 2), "type": "in-center"})
        unique_cands = {}
        for cand in temp_internal:
            x, y = int(cand["pos"][0]), int(cand["pos"][1])
            if not (sx + margin <= x and sy + margin <= y and x + box_w <= sx + sw - margin and y + box_h <= sy + sh - margin):
                continue
            unique_cands[(x, y)] = cand
        internal_cands = list(unique_cands.values())

    for side in ["left", "right"]:
        if side not in CONFIG.get("IDT_ALLOWED_SIDES", ["bottom", "left", "right"]): continue
        x = sx - box_w - CONFIG.get("IDT_SIDE_GAP_NEAR", 3) if side == "left" else sx + sw + CONFIG.get("IDT_SIDE_GAP_NEAR", 3)
        for offset_ratio in CONFIG.get("IDT_SIDE_OFFSETS", [0.0]):
            y = sy + (sh - box_h) / 2 + offset_ratio * sh
            external_side_cands.append({"pos": (x, y), "type": f"out-{side}"})

    if CONFIG.get("IDT_PREFER_EXTERNAL_BOTTOM", True):
        pos = try_positions(external_bottom_cands, other_obstacles)
        if pos: return pos
        pos = try_positions(internal_cands, other_obstacles)
        if pos: return pos
        pos = try_positions(external_side_cands, other_obstacles)
        if pos: return pos
    else:
        pos = try_positions(internal_cands, other_obstacles)
        if pos: return pos
        pos = try_positions(external_bottom_cands, other_obstacles)
        if pos: return pos
        pos = try_positions(external_side_cands, other_obstacles)
        if pos: return pos
    return None

def force_place_label(
    struct_box: Dict[str, Any], box_w: int, box_h: int, img_w: int, img_h: int,
    obstacles: List[Tuple[int,int,int,int]], self_obstacle: Tuple[int,int,int,int]
) -> Optional[Tuple[int,int]]:
    """
    强制放置：忽略墨迹，但严格避免与其他障碍重叠。
    策略顺序由 CONFIG["IDT_FORCE_PLACEMENT_STRATEGY"] 控制。
    如果所有策略都重叠，则返回 None。
    """
    sx, sy, sw, sh = struct_box["x"], struct_box["y"], struct_box["width"], struct_box["height"]
    pad_overlap = CONFIG.get("IDT_FORCE_OVERLAP_PADDING", 1)
    strategies = CONFIG.get("IDT_FORCE_PLACEMENT_STRATEGY", ["bottom_center","inside_center","top_center"])

    def overlaps(x, y, w, h):
        # 使用 self_obstacle 过滤掉自身
        other_obstacles = [obs for obs in obstacles if obs != self_obstacle]
        for ox1, oy1, ox2, oy2 in other_obstacles:
            if not (x + w + pad_overlap <= ox1 or x >= ox2 + pad_overlap or y + h + pad_overlap <= oy1 or y >= oy2 + pad_overlap):
                return True
        return False

    candidates: List[Tuple[int,int]] = []
    for st in strategies:
        if st == "bottom_center":
            x = int(sx + (sw - box_w)/2)
            y = int(sy + sh + CONFIG.get("IDT_VERTICAL_GAP", 4))
            candidates.append((x,y))
        elif st == "inside_center":
            x = int(sx + (sw - box_w)/2)
            y = int(sy + (sh - box_h)/2)
            candidates.append((x,y))
        elif st == "top_center":
            x = int(max(0, sx + (sw - box_w)/2))
            y = int(max(0, sy - box_h - CONFIG.get("IDT_VERTICAL_GAP", 4)))
            candidates.append((x,y))
        elif st == "right_center":
            x = int(sx + sw + CONFIG.get("IDT_SIDE_GAP", 4))
            y = int(sy + (sh - box_h)/2)
            candidates.append((x,y))
        elif st == "left_center":
            x = int(max(0, sx - box_w - CONFIG.get("IDT_SIDE_GAP", 4)))
            y = int(sy + (sh - box_h)/2)
            candidates.append((x,y))

    final_cands = []
    for (x,y) in candidates:
        if x < 0: x = 0
        if y < 0: y = 0
        if x + box_w > img_w: x = max(0, img_w - box_w)
        if y + box_h > img_h: y = max(0, img_h - box_h)
        final_cands.append((x,y))

    # 尝试找到一个不重叠的位置
    for (x,y) in final_cands:
        if not overlaps(x,y,box_w,box_h):
            return (x,y)
            
    # 如果所有策略都失败（重叠），则返回 None，由调用者处理最终保障
    return None

def estimate_font_size(img_path: Path, img_h: int, existing_idts: List[str], bboxes: List[Dict[str, Any]] = None) -> int:
    # 策略1：根据 bbox 平均高度估算 identifier 字号
    bbox_based_fs = 0
    if bboxes:
        valid_heights = []
        for b in bboxes:
            # 过滤掉极小的框或明显的非分子框
            if "bbox" in b:
                _, _, _, h = b["bbox"]
                if h > 20: valid_heights.append(h)
        
        if valid_heights:
            avg_h = sum(valid_heights) / len(valid_heights)
            # 经验比例：Identifier 高度约为分子框平均高度的 18%
            bbox_based_fs = int(avg_h * 0.18)
            if CONFIG["DEBUG_LOG"]:
                 print(f"  [BBox估算] 平均高度: {avg_h:.1f}, 估算字号: {bbox_based_fs}")

    # 策略2：基于图片高度的保底估算
    # 稍微调大默认比例，从 0.02 -> 0.025
    ratio = 0.025 
    img_h_fs = int(img_h * ratio)
    
    # 综合取舍：如果 bbox 估算有效，取两者中较大的一个
    fs = max(bbox_based_fs, img_h_fs) if bbox_based_fs > 0 else img_h_fs

    min_fs = CONFIG.get("IDT_MIN_FONT", 14)
    max_fs = CONFIG.get("IDT_MAX_FONT", 42)
    final_fs = max(min_fs, min(max_fs, fs))
    
    if CONFIG["DEBUG_LOG"]:
        print(f"  [最终估算] 字号: {final_fs} (BBox: {bbox_based_fs}, ImgH: {img_h_fs})")
        
    return final_fs

def render_preview(img_data: Dict[str, Any], pil_img: Image.Image, new_placements: List[Dict[str, Any]]):
    # 核心修复：确保只要开启预览，就一定会执行，不再因为 new_placements 为空而提前返回。
    # 这将保证所有图片都会被处理并保存。
    if not CONFIG.get("PREVIEW", False) and not CONFIG.get("ALWAYS_PREVIEW", False):
        return

    annotated_out_dir = Path(CONFIG["PREVIEW_OUT_DIR_ANNOTATED"])
    clean_out_dir = Path(CONFIG["PREVIEW_OUT_DIR_CLEAN"])
    annotated_out_dir.mkdir(parents=True, exist_ok=True)
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    mol_newly_assigned_color = CONFIG.get("PREVIEW_MOL_ORIG_HAS_IDT_COLOR", (255, 165, 0))
    mol_other_color = CONFIG.get("PREVIEW_MOL_ORIG_NO_IDT_COLOR", (0, 0, 255))
    idt_new_text_color = (255, 0, 0)
    idt_clean_text_color = (0, 0, 0)
    base_name = Path(img_data["file_name"]).stem
    
    # --- 生成 annotated_img ---
    annotated_img = pil_img.copy()
    draw_annotated = ImageDraw.Draw(annotated_img)
    # 绘制分子框，并在左上角标注框的 id
    for idx, bbox in enumerate(img_data.get("bboxes", [])):
        if bbox.get("category_id") == CONFIG["MOL_CATEGORY_ID"] and 'bbox' in bbox:
            # _has_new_idt 标记是在处理流程中添加的，用于区分新旧
            color = mol_newly_assigned_color if '_has_new_idt' in bbox else mol_other_color
            x, y, w, h = bbox['bbox']
            draw_annotated.rectangle([x, y, x + w, y + h], outline=color, width=2)
            
            # 在左上角标注框的 id
            bbox_id = bbox.get("id") or bbox.get("order") or idx
            id_text = str(bbox_id)
            # 使用较小的字体显示 id
            id_font_size = max(12, min(20, int(pil_img.height * 0.015)))
            id_font = load_font(id_font_size)
            # 计算文本尺寸
            try:
                id_bbox = id_font.getbbox(id_text)
                id_w = id_bbox[2] - id_bbox[0]
                id_h = id_bbox[3] - id_bbox[1]
            except (TypeError, AttributeError):
                id_w, id_h = id_font.getsize(id_text)
            
            # 在左上角绘制背景框和文本
            padding = 3
            bg_x1 = x
            bg_y1 = y
            bg_x2 = x + id_w + padding * 2
            bg_y2 = y + id_h + padding * 2
            # 绘制白色背景框
            draw_annotated.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
            # 绘制 id 文本
            text_x = x + padding
            text_y = y + padding
            draw_annotated.text((text_x, text_y), id_text, fill=(0, 0, 0), font=id_font)
    # 绘制新增的 IDT
    for placement in new_placements:
        label = placement["label"]
        fs = placement["font_size"]
        fnt = load_font(fs)
        spacing = placement["char_spacing"]
        stroke = CONFIG.get("BOLD_DRAW_STROKE", 0)
        tx, ty = placement["pos"]
        draw_text_with_spacing(draw_annotated, (tx, ty), label, fnt, fill=idt_new_text_color, spacing=spacing, stroke=stroke, stroke_fill=idt_new_text_color)
    
    annotated_path = annotated_out_dir / f"{base_name}.png"
    annotated_img.save(annotated_path)
    if CONFIG["DEBUG_LOG"]:
        log_msg = f"  [预览] 已保存 (带标注): {annotated_path}"
        if not new_placements:
            log_msg += " (无新增IDT)"
        print(log_msg)

    # --- 生成 clean_img ---
    clean_img = pil_img.copy()
    draw_clean = ImageDraw.Draw(clean_img)
    # 只绘制新增的 IDT
    for placement in new_placements:
        label = placement["label"]
        fs = placement["font_size"]
        fnt = load_font(fs)
        spacing = placement["char_spacing"]
        stroke = CONFIG.get("BOLD_DRAW_STROKE", 0)
        tx, ty = placement["pos"]
        draw_text_with_spacing(draw_clean, (tx, ty), label, fnt, fill=idt_clean_text_color, spacing=spacing, stroke=stroke, stroke_fill=idt_clean_text_color)
    
    clean_path = clean_out_dir / f"{base_name}.png"
    clean_img.save(clean_path)
    if CONFIG["DEBUG_LOG"]:
        log_msg = f"  [预览] 已保存 (干净版): {clean_path}"
        if not new_placements:
            log_msg += " (无新增IDT)"
        print(log_msg)

def scale_bbox_in_place(bbox: Dict[str, Any], factor: float):
    try:
        if "bbox" in bbox and isinstance(bbox["bbox"], (list, tuple)) and len(bbox["bbox"]) == 4:
            x, y, w, h = bbox["bbox"]
            bbox["bbox"] = [
                int(round(x * factor)),
                int(round(y * factor)),
                int(round(w * factor)),
                int(round(h * factor)),
            ]
        elif all(k in bbox for k in ("x", "y", "width", "height")):
            bbox["x"] = int(round(bbox["x"] * factor))
            bbox["y"] = int(round(bbox["y"] * factor))
            bbox["width"] = int(round(bbox["width"] * factor))
            bbox["height"] = int(round(bbox["height"] * factor))
    except (TypeError, ValueError):
        pass

def run_full_pipeline():
    merged_path = CONFIG["MERGED_INPUT_JSON"]
    out_path = CONFIG["OUTPUT_JSON_MERGED"]
    if not merged_path:
        print("[ERROR] Missing input JSON. Pass --merged_input_json.")
        return
    if not out_path:
        print("[ERROR] Missing output JSON. Pass --output_json_merged.")
        return
    if not CONFIG["IMAGE_ROOT"]:
        print("[ERROR] Missing image root. Pass --image_root.")
        return
    if not os.path.isfile(merged_path):
        print(f"[ERROR] 输入 JSON 不存在: {merged_path}"); return
    with open(merged_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_root = Path(CONFIG["IMAGE_ROOT"])
    alt_roots = [Path(p) for p in CONFIG.get("IMAGE_ALT_ROOTS", [])]
    all_roots = [image_root] + alt_roots
    mol_cid = CONFIG["MOL_CATEGORY_ID"]
    idt_cid = CONFIG["IDT_CATEGORY_ID"]
    skip_cid = CONFIG["SKIP_CATEGORY_ID"]
    scale_factor = CONFIG["TARGET_DPI"] / CONFIG["INPUT_DPI"]
    total_images = len(data.get("images", []))
    skipped_images_count = 0
    print(f"开始处理 {total_images} 张图片...")

    for idx, img_data in enumerate(data.get("images", [])):
        print(f"\n--- [进度: {idx + 1}/{total_images}] 正在处理: {img_data.get('file_name', 'N/A')} ---")
        if not isinstance(img_data, dict): continue
        img_data["file_name"] = _normalize_filename(img_data.get("file_name", ""))
        if not img_data.get("_dpi_scaled"):
            for bbox in img_data.get("bboxes", []):
                scale_bbox_in_place(bbox, scale_factor)
            img_data["_dpi_scaled"] = True

        img_path = find_image_path(img_data["file_name"], all_roots)
        if not img_path:
            print(f"  [!!!警告!!!] 找不到图片文件: {img_data['file_name']}，已跳过。")
            skipped_images_count += 1
            continue
        
        print(f"  [信息] 图片已找到并加载: {img_path}")

        all_current_identifiers = []
        mols_to_process = []
        all_obstacles = []

        # 修复：首先遍历一次，收集所有已存在的标识符
        for bbox in img_data.get("bboxes", []):
            if 'identifiers' in bbox and bbox['identifiers']:
                ids = bbox['identifiers']
                if not isinstance(ids, list):
                    ids = [str(ids)]
                # 将有效的、非空的标识符添加到列表中
                all_current_identifiers.extend(id_str for id_str in ids if id_str)
        
        # 使用 set 去重，确保唯一性
        all_current_identifiers = list(set(all_current_identifiers))
        if all_current_identifiers:
            print(f"  [信息] 发现已存在的 IDT: {sorted(all_current_identifiers)}")

        for bbox in img_data.get("bboxes", []):
            if 'identifiers' in bbox and not isinstance(bbox['identifiers'], list):
                bbox['identifiers'] = [str(bbox['identifiers'])]

        for bbox in img_data.get("bboxes", []):
            if 'bbox' in bbox:
                x, y, w, h = bbox['bbox']
                all_obstacles.append((x, y, x + w, y + h))
            
            ids = bbox.get("identifiers")
            has_id = ids and any(ids)
            
            raw_text = bbox.get("text", "")
            has_text = raw_text and str(raw_text).strip()

            category_id = bbox.get("category_id")
            is_forbidden_category = category_id in [2, 3, 4]

            # 核心筛选逻辑：只有当所有条件都不满足时，才加入待处理列表
            if not has_id and not has_text and not is_forbidden_category:
                mols_to_process.append(bbox)

        if not mols_to_process:
            print("  [信息] 此图片中没有需要添加 IDT 的目标。数据将按原样保留。")
            try:
                pil_img = Image.open(img_path).convert("RGB")
                render_preview(img_data, pil_img, [])  # 即使没有新增IDT，也生成预览图
                pil_img.close()
                print(f"  [信息] 图片已保存到预览文件夹 (无新增IDT)。")
            except Exception as e:
                if CONFIG["DEBUG_LOG"]: print(f"[PREVIEW_ERR] {e}")
            continue
        
        print(f"  [信息] 发现 {len(mols_to_process)} 个目标需要添加 IDT。")

        scheme = analyze_scheme(all_current_identifiers)
        pil_img = None
        temp_placements_for_preview = []
        fallback_chars = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + \
                         [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
                         [str(i) for i in range(10)]

        try:
            pil_img = Image.open(img_path).convert("RGB")
            gray_img = _get_gray_image(pil_img)
            W, H = pil_img.size
            fs_base = estimate_font_size(img_path, H, all_current_identifiers, img_data.get("bboxes", []))
            for bbox in img_data.get("bboxes", []):
                if bbox.get("category_id") == CONFIG["IDT_CATEGORY_ID"]:
                    bbox["font_size"] = fs_base
                    bbox["char_spacing"] = calc_char_spacing(fs_base)

            for mol_bbox in mols_to_process:
                label = next_label(scheme)
                fallback_len = CONFIG.get("IDT_FALLBACK_MAX_LEN", 0)
                if fallback_len > 0 and len(label) > fallback_len:
                    for char in fallback_chars:
                        if char not in all_current_identifiers:
                            label = char
                            break
                while label in all_current_identifiers:
                    label = next_label(scheme)

                placed = False
                phase = "normal"
                attempt_fs = fs_base
                min_font_normal = CONFIG.get("IDT_MIN_FONT", 14)
                min_font_force_allow = CONFIG.get("IDT_MIN_FONT_FORCE", 8)

                # --- 阶段1 & 2: 正常放置，并逐步缩小字号直到最小允许字号 ---
                while attempt_fs >= min_font_force_allow:
                    fnt = load_font(attempt_fs)
                    char_spacing = calc_char_spacing(attempt_fs)
                    tw, th = measure_text_with_spacing(fnt, label, attempt_fs, char_spacing)
                    if tw <= 0 or th <= 0: break
                    pad = CONFIG["IDT_PADDING"]
                    bw, bh = int(math.ceil(tw + pad * 2)), int(math.ceil(th + pad * 2))
                    x, y, w, h = mol_bbox['bbox']
                    struct_box = {"x": x, "y": y, "width": w, "height": h}
                    self_obstacle = (x, y, x + w, y + h)
                    
                    pos = find_placement_enhanced(struct_box, bw, bh, W, H, all_obstacles, gray_img, self_obstacle, try_inside=True)
                    
                    if pos:
                        ix, iy = pos
                        mol_bbox['identifiers'] = [label]
                        mol_bbox['_has_new_idt'] = True
                        all_current_identifiers.append(label)
                        temp_placements_for_preview.append({
                            "label": label, "pos": (ix + pad, iy + pad),
                            "font_size": attempt_fs, "char_spacing": char_spacing
                        })
                        all_obstacles.append((ix, iy, ix + bw, iy + bh))
                        placed = True
                        if attempt_fs < min_font_normal and CONFIG["DEBUG_LOG"]:
                             print(f"  [SHRINK] {img_data['file_name']} 放置 {label} 成功 字号={attempt_fs}")
                        break
                    
                    attempt_fs -= CONFIG.get("IDT_FONT_SHRINK_STEP", 2)

                # --- 阶段3: 强制放置 (如果前面所有尝试都失败) ---
                if not placed:
                    phase = "force"
                    attempt_fs = min_font_force_allow
                    if attempt_fs < 4: attempt_fs = 4
                    
                    fnt = load_font(attempt_fs)
                    char_spacing = calc_char_spacing(attempt_fs)
                    tw, th = measure_text_with_spacing(fnt, label, attempt_fs, char_spacing)
                    pad = CONFIG["IDT_PADDING"]
                    bw, bh = int(math.ceil(tw + pad * 2)), int(math.ceil(th + pad * 2))
                    x, y, w, h = mol_bbox['bbox']
                    struct_box = {"x": x, "y": y, "width": w, "height": h}
                    self_obstacle = (x, y, x + w, y + h)
                    
                    # 尝试强制策略
                    force_pos = force_place_label(struct_box, bw, bh, W, H, all_obstacles, self_obstacle)
                    
                    # --- 阶段4: 最终保障 - 像素级扫描 (仅当强制策略也找不到不重叠位置时) ---
                    if not force_pos:
                        if CONFIG["DEBUG_LOG"]:
                            print(f"  [SCAN] {img_data['file_name']} 强制策略失败，启动最终像素扫描寻找空隙...")
                        
                        other_obstacles = [obs for obs in all_obstacles if obs != self_obstacle]
                        
                        def is_safe(cand_x, cand_y):
                            # 检查图像边界
                            if not (0 <= cand_x and cand_x + bw <= W and 0 <= cand_y and cand_y + bh <= H):
                                return False
                            # 检查与其他障碍物的重叠
                            for ox1, oy1, ox2, oy2 in other_obstacles:
                                if not (cand_x + bw <= ox1 or cand_x >= ox2 or cand_y + bh <= oy1 or cand_y >= oy2):
                                    return False
                            return True

                        # 从分子框周围螺旋向外扫描
                        scan_step = 4
                        for r in range(0, max(W, H), scan_step):
                            # 扫描 (x-r, y-r) 到 (x+r, y+r) 的矩形边框
                            for i in range(-r, r + 1, scan_step):
                                # 顶部和底部边
                                cand_x, cand_y = x + i, y - r
                                if is_safe(cand_x, cand_y): force_pos = (cand_x, cand_y); break
                                cand_y = y + r
                                if is_safe(cand_x, cand_y): force_pos = (cand_x, cand_y); break
                                # 左侧和右侧边
                                cand_x, cand_y = x - r, y + i
                                if is_safe(cand_x, cand_y): force_pos = (cand_x, cand_y); break
                                cand_x = x + r
                                if is_safe(cand_x, cand_y): force_pos = (cand_x, cand_y); break
                            if force_pos: break
                        
                        if force_pos and CONFIG["DEBUG_LOG"]:
                            print(f"  [SCAN-SUCCESS] 扫描找到安全位置 at {force_pos}")

                    if force_pos:
                        ix, iy = force_pos
                        mol_bbox['identifiers'] = [label]
                        mol_bbox['_has_new_idt'] = True
                        all_current_identifiers.append(label)
                        temp_placements_for_preview.append({
                            "label": label, "pos": (ix + pad, iy + pad),
                            "font_size": attempt_fs, "char_spacing": char_spacing
                        })
                        all_obstacles.append((ix, iy, ix + bw, iy + bh))
                        placed = True
                        if CONFIG["DEBUG_LOG"]:
                            print(f"  [FORCE/SCAN] {img_data['file_name']} 最终放置 {label} 成功 字号={attempt_fs}")
                    else:
                        # 此处理论上不应到达，除非图像被完全填满以至于一个像素的空隙都没有
                        print(f"[FATAL-ERROR] 图片 {img_data['file_name']} 完全没有空间放置IDT '{label}'。这是一个极端情况，请检查图片。")


            render_preview(img_data, pil_img, temp_placements_for_preview)

        finally:
            if pil_img:
                pil_img.close()
            if id(pil_img) in _gray_images_cache:
                del _gray_images_cache[id(pil_img)]

    data_to_save = data
    if CONFIG.get("ALWAYS_SAVE_JSON", False):
        data_to_save = json.loads(json.dumps(data))
    for img in data_to_save.get("images", []):
        img.pop("_dpi_scaled", None)
        for bbox in img.get("bboxes", []):
            bbox.pop("_is_new", None)
            bbox.pop("font_size", None)
            bbox.pop("char_spacing", None)
            bbox.pop("_has_new_idt", None)
            has_id = bbox.get("identifiers") and any(bbox.get("identifiers", []))
            if has_id and "text" in bbox:
                del bbox["text"]

    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    print("\n========== 处理完成 ==========")
    print(f"总共处理了 {total_images} 条图片记录。")
    if skipped_images_count > 0:
        print(f"其中 {skipped_images_count} 张图片因文件未找到而被跳过。")
    print(f"JSON 文件已更新并保存到: {out_path}")
    annotated_preview_path = CONFIG.get("PREVIEW_OUT_DIR_ANNOTATED")
    clean_preview_path = CONFIG.get("PREVIEW_OUT_DIR_CLEAN")
    print(f"带标注的预览图保存在: {annotated_preview_path}")
    print(f"仅含新增IDT文本的预览图保存在: {clean_preview_path}")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Assign or render molecule identifiers after Mid-Mapper recognition."
    )
    parser.add_argument("--merged_input_json", required=True, help="JSON with Mid-Mapper identifiers written to bboxes.")
    parser.add_argument("--image_root", required=True, help="Directory containing original reaction images.")
    parser.add_argument("--output_json_merged", required=True, help="Final JSON path with complete identifiers.")
    parser.add_argument("--output_root", default="outputs/mid_mapper", help="Root directory for generated previews.")
    parser.add_argument("--preview_out_dir_annotated", default="", help="Preview images with bboxes and newly added IDTs.")
    parser.add_argument("--preview_out_dir_clean", default="", help="Clean images with newly added IDTs rendered.")
    parser.add_argument("--image_alt_roots", nargs="*", default=[], help="Optional fallback image directories.")
    parser.add_argument("--font_path", default="", help="Optional TrueType font path.")
    parser.add_argument("--font_bold_path", default="", help="Optional bold TrueType font path.")
    parser.add_argument("--input_dpi", type=int, default=400)
    parser.add_argument("--target_dpi", type=int, default=400)
    parser.add_argument("--debug_log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _apply_args(args):
    output_root = Path(args.output_root)
    annotated_dir = args.preview_out_dir_annotated or str(output_root / "annotated_previews")
    clean_dir = args.preview_out_dir_clean or str(output_root / "clean_previews")

    CONFIG.update({
        "JSON_RESULTS_DIR": args.merged_input_json,
        "LOCAL_OUTPUT_ROOT": args.output_root,
        "IMAGE_ROOT": args.image_root,
        "FONT_PATH": args.font_path,
        "FONT_BOLD_PATH": args.font_bold_path,
        "USE_BOLD_FONT": bool(args.font_bold_path),
        "MERGED_INPUT_JSON": args.merged_input_json,
        "OUTPUT_JSON_MERGED": args.output_json_merged,
        "IMAGE_ALT_ROOTS": args.image_alt_roots,
        "INPUT_DPI": args.input_dpi,
        "TARGET_DPI": args.target_dpi,
        "PREVIEW_OUT_DIR_ANNOTATED": annotated_dir,
        "PREVIEW_OUT_DIR_CLEAN": clean_dir,
        "DEBUG_LOG": args.debug_log,
    })


def main():
    args = _parse_args()
    _apply_args(args)
    run_full_pipeline()

if __name__ == "__main__":
    main()
