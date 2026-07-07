import argparse
import ast
import json
import os
import sys
import time

from PIL import Image, ImageDraw, ImageFont


def load_prompt_template():
    system_prompt = 'You are a scientific paper-reading assistant.'
    user_prompt = """Task: Identify the identifiers corresponding to the molecular formulas in the blue box area in the scientific research paper image.

Instructions:

1.The image is a page from a scientific paper featuring 2D molecular structures enclosed in blue boxes.
2.Each blue box has a visible box index (black background, white text) in the top-left corner.
3.Key Objective: Match each mol box index to its corresponding molecular identifier(s) as stated in the paper.
 a.If a molecule has no associated identifier, return None for that entry.
 b.The identifier is usually displayed in bold and appears below the molecular formula in the figure.
 c.If there are no molecular marked with blue boxes in the figure, '[]' should be returned.
4.Output Format:
```json
[
    {
        "mol bbox index": "box index of the detected molecule",
        "identifier": ["identifier1", "identifier2", ...]
    },
    {
        "mol bbox index": "box index of the detected molecule",
        "identifier": None
    }
]
```
5.Requirements:
 a.Strictly adhere to the JSON structure; no additional text or formatting is allowed.
 b.Use `None` (not a string) for missing identifiers."""

    return system_prompt, user_prompt

def draw_identifier_num(num, xmin, ymin, draw, font, padding=2):
    bbox = draw.textbbox((0, 0), str(num), font=font)
    # 计算文本的宽度和高度
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    fill_x1 = xmin
    fill_y1 = ymin - 20
    fill_x2 = fill_x1 + text_width + 2 * padding
    fill_y2 = fill_y1 + text_height + 2 * padding

    # 绘制填充区域（黑色背景）
    draw.rectangle((fill_x1, fill_y1, fill_x2, fill_y2), fill="black")
    
    # 计算文本的绘制位置（居中）
    text_x = fill_x1 + padding
    text_y = fill_y1 - 3

    # 绘制数字文本（白色）
    draw.text((text_x, text_y), str(num), fill="white", font=font)

def group_bboxes_by_row(bboxes):
    """
        根据 y 坐标将边界框分组为行
        :param bboxes: 排序后的边界框列表
        :return: 分组后的边界框列表
    """
    rows = []
    if not bboxes:
        return rows
    current_row = [bboxes[0]]
    for bbox in bboxes[1:]:
        w, h = current_row[0][2] - current_row[0][0], current_row[0][3] - current_row[0][1]
        if bbox[1] <= current_row[0][1] + h / 2 and bbox[1] >= current_row[0][1] - h / 2:
            current_row.append(bbox)
        else:
            rows.append(current_row)
            current_row = [bbox]
    rows.append(current_row)
    return rows

def get_reading_order(bboxes):
    """
        生成边界框的阅读顺序
        :param bboxes: 边界框列表
        :return: 阅读顺序的边界框列表
    """
    sorted_bboxes = sorted(bboxes, key=lambda box: (box[1], box[0]))
    rows = group_bboxes_by_row(sorted_bboxes)
    reading_order = []
    for row in rows:
        reading_order.extend(sorted(row, key=lambda bbox: bbox[0]))
    return reading_order

            
def parse_vlm_response(response):
    response = response.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Some checkpoints emit Python-style None instead of JSON null.
        return ast.literal_eval(response.replace('null', 'None'))

def get_vlm_single_result(args, image_path, message_history, retry=10):
    
    system_prompt, user_prompt = load_prompt_template()
    
    # 构造prompt
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]
    messages.append({'role': 'user', 'content': user_prompt})
    user_instruction = []
    
    instruction = 'A image will be provided below. Please follow the given instructions to formulate your response.'
    user_instruction.append({'type': 'text', 'text': instruction})
    if image_path:
        user_instruction.append({"type":"image"})

    
    messages.append({'role': 'user', 'content': user_instruction})
    message_history.extend(messages)
    
    # 创建聊天补全请求
    for _ in range(1, retry + 1):
        try:
            # Preprocess the inputs
            text_prompt = args.processor.apply_chat_template(message_history, add_generation_prompt=True)
            inputs = args.processor(
                text=[text_prompt],
                images=[Image.open(img_path) for img_path in [image_path]],
                padding=True,
                return_tensors="pt",
            )
            # 关键：device_map="auto" 时不要强制搬到单一 cuda，保持 CPU 让 HF/Accelerate 按映射分发
            if not getattr(args.model, "hf_device_map", None):
                model_device = next(args.model.parameters()).device
                inputs = inputs.to(model_device)
            # Inference: Generation of the output
            infer_start = time.time()
            output_ids = args.model.generate(**inputs, max_new_tokens=8192)
            infer_elapsed = time.time() - infer_start
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            model_reply = args.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print('模型调用成功')
            print(f"[TIME] 单张推理耗时: {infer_elapsed:.4f}s image={os.path.basename(image_path)}")
            message_history.append({'role': 'assistant', 'content': model_reply})
            return model_reply, infer_elapsed
        
        except Exception as e:
            print("模型调用失败:", str(e))
            time.sleep(5) # 如果调用失败，程序睡眠5秒钟
            
    return '', 0.0

def get_vlm_coref_result(args, image_path, response_root_dir):
    message_history = []
    reply_multi = []
    reply, infer_elapsed = get_vlm_single_result(args, image_path, message_history)
    print('coref第一轮回复', reply)
    reply_multi.append('coref第一轮回复' + reply + '\n')
    response_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    with open(os.path.join(response_root_dir, response_name), 'w') as f:
        f.write(''.join(reply_multi))
    return reply, infer_elapsed

def save_mol_det_identifier_result(image_name, width, height, save_dir, response_root_dir,
                                   mol_idx2coord, mol_det_identifier, ordered_mol_ids):
    # 去掉前缀 figure/ 或 table/
    if image_name.startswith('figure/'):
        image_name = image_name[len('figure/'):]
    if image_name.startswith('table/'):
        image_name = image_name[len('table/'):]
    save_json_path = os.path.join(save_dir, '.'.join(image_name.split('.')[:-1])) + '_mol_det_identifier.json'

    # 解析模型回复
    try:
        parsed = parse_vlm_response(mol_det_identifier)
        if not isinstance(parsed, list):
            parsed = []
    except Exception:
        parsed = []

    # 收集: 原始 bbox id -> identifiers(list)
    mol_to_ids = {}
    for elem in parsed:
        try:
            mol_idx = int(elem.get('mol bbox index'))
        except Exception:
            continue
        ids = elem.get('identifier')
        if ids is None:
            ids = []
        if isinstance(ids, (list, tuple)):
            ids = [str(x) for x in ids if isinstance(x, (str, int, float))]
        else:
            ids = []
        mol_to_ids[mol_idx] = ids

    result_list = []
    corefs = []        # [identifier_order, mol_order]
    corefs_bbox = []   # [identifier_order, original_bbox_id]
    idt_strings = []
    mol_order_map = {}  # original bbox id -> sequential order (1-based)

    # 分子框: 使用 ordered_mol_ids 的顺序构造
    for seq_order, mol_id in enumerate(ordered_mol_ids, start=1):
        box = mol_idx2coord.get(mol_id)
        if not box:
            continue
        x1, y1, x2, y2 = box
        x = int(round(x1)); y = int(round(y1))
        w = int(round(x2 - x1)); h = int(round(y2 - y1))
        result_list.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "attribute": "Mol",
            "order": seq_order,
            "bbox_id": mol_id  # 原始 bbox id
        })
        mol_order_map[mol_id] = seq_order

    # identifiers: 接在分子框之后
    next_order = len(result_list) + 1
    for mol_id in ordered_mol_ids:
        ids = mol_to_ids.get(mol_id, [])
        mol_order = mol_order_map.get(mol_id)
        if not mol_order:
            continue
        for id_str in ids:
            result_list.append({
                "attribute": "Idt",
                "order": next_order,
                "textAttribute": id_str
            })
            corefs.append([next_order, mol_order])
            corefs_bbox.append([next_order, mol_id])
            idt_strings.append(id_str)
            next_order += 1

    json_dict = {
        "file_name": image_name,
        "width": width,
        "height": height,
        "result": result_list,
        "corefs": corefs,              # 旧方式 (identifier_order, mol_order)
        "corefs_bbox": corefs_bbox,    # 新方式 (identifier_order, 原始 bbox id)
        "mol_id_to_order": mol_order_map,  # 映射 (原始 bbox id -> mol_order)
        "idt_content": list(dict.fromkeys(idt_strings))
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False)
    # 新增：返回解析后的编号->identifier列表映射
    return {k: mol_to_ids.get(k, []) for k in ordered_mol_ids}

def _compute_dynamic_font(bboxes, min_font_size, font_step):
    if not bboxes:
        return 36
    sizes = []
    for x1, y1, x2, y2 in bboxes:
        sizes.append(min(x2 - x1, y2 - y1))
    min_bbox_size = min(sizes)
    if min_bbox_size < 50:
        font_size = max(int(min_bbox_size * 0.4), min_font_size)
    elif min_bbox_size < 100:
        font_size = max(int(min_bbox_size * 0.3), min_font_size + font_step)
    else:
        font_size = max(int(min_bbox_size * 0.2), min_font_size + 2 * font_step)
    return min(font_size, 48)

def _load_font(size_candidate):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size_candidate)
    except Exception:
        return ImageFont.load_default()

def process(args, idt_data):
    image_root_dir = args.image_root_dir
    response_root_dir = args.response_root_dir
    middle_root_dir = args.middle_root_dir
    result_root_dir = args.result_root_dir
    line_width = args.line_width

    images = idt_data.get('images', [])
    total_images = len(images)
    print(f"[INFO] Loaded JSON with images={total_images}")
    sys.stdout.flush()

    processed_images = 0
    missing_image_count = 0
    skipped_reaction_empty = 0
    model_call_count = 0
    total_infer_time = 0.0
    timed_model_call_count = 0

    # 新增：用于最终写回（修改原 JSON 的副本）
    updated_images = images  # 直接在原结构上修改其 bboxes

    for idx, img_item in enumerate(updated_images):
        file_name = img_item.get('file_name')
        if not file_name:
            continue

        # 跳过 reactions 为空列表的图片
        if 'reactions' in img_item and isinstance(img_item['reactions'], list) and len(img_item['reactions']) == 0:
            skipped_reaction_empty += 1
            continue

        if idx % 50 == 0:
            print(f"[PROGRESS] {idx}/{total_images}")
            sys.stdout.flush()

        base_original = os.path.basename(file_name)
        base_no_prefix = base_original.split('/')[-1]
        name_no_ext, ext = os.path.splitext(base_no_prefix)

        # 支持 jpg/png 互换
        candidates = [
            os.path.join(image_root_dir, file_name),
            os.path.join(image_root_dir, base_no_prefix)
        ]
        if ext.lower() == '.jpg':
            candidates.append(os.path.join(image_root_dir, name_no_ext + '.png'))
        if ext.lower() == '.png':
            candidates.append(os.path.join(image_root_dir, name_no_ext + '.jpg'))

        img_path = None
        for cand in candidates:
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            missing_image_count += 1
            print(f"[WARN] Missing image for entry #{idx}: {file_name}")
            sys.stdout.flush()
            continue

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] 打开失败 {img_path}: {e}")
            continue

        # 收集可绘制 bbox：跳过 (text非空) 或 (category_id==3)
        candidate = []
        for b in img_item.get('bboxes', []):
            if b.get('category_id') == 3:
                continue
            t = b.get('text')
            if isinstance(t, str) and t.strip():
                continue
            bb = b.get('bbox')
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            x, y, w, h = bb
            candidate.append([x, y, x + w, y + h, b])  # 末尾附带引用

        if not candidate:
            # 没框：保存原图（不推理）
            out_image_name = f"{idx}_{name_no_ext}{ext}"
            middle_image_path = os.path.join(middle_root_dir, out_image_name)
            image.save(middle_image_path)
            processed_images += 1
            continue

        # 阅读顺序排序
        coords_only = [c[:4] for c in candidate]
        ordered_coords = get_reading_order(coords_only)
        ordered_full = []
        used = set()
        for oc in ordered_coords:
            for c in candidate:
                if c[:4] == oc and id(c) not in used:
                    ordered_full.append(c)
                    used.add(id(c))
                    break

        # 动态字体
        font_size = _compute_dynamic_font([c[:4] for c in ordered_full], args.min_font_size, args.font_step)
        font = _load_font(font_size)

        # 绘制并建立 index -> bbox 映射
        draw = ImageDraw.Draw(image)
        index_to_bbox_obj = {}
        mol_idx2coord = {}
        for i, c in enumerate(ordered_full, start=1):
            x1, y1, x2, y2, bbox_obj = c
            draw.rectangle((x1, y1, x2, y2), outline="blue", width=line_width)
            draw_identifier_num(i, x1, y1 + 20, draw, font)
            index_to_bbox_obj[i] = bbox_obj
            mol_idx2coord[i] = [x1, y1, x2, y2]

        out_image_name = f"{idx}_{name_no_ext}{ext}"
        middle_image_path = os.path.join(middle_root_dir, out_image_name)
        image.save(middle_image_path)

        # 模型推理
        if args.dry_run:
            reply = "[]"
            infer_elapsed = 0.0
            print(f"[DRY] Skip model call: {middle_image_path}")
        else:
            reply, infer_elapsed = get_vlm_coref_result(args, middle_image_path, response_root_dir)
            model_call_count += 1
            if infer_elapsed > 0:
                total_infer_time += infer_elapsed
                timed_model_call_count += 1
        print(f"[INFO] 完成绘制与推理: {middle_image_path} boxes={len(ordered_full)}")
        sys.stdout.flush()

        # 保存额外结构文件（保留原有函数）
        ordered_indices = list(range(1, len(ordered_full) + 1))
        idx_to_ids = save_mol_det_identifier_result(base_no_prefix, image.width, image.height,
                                                    result_root_dir, response_root_dir,
                                                    mol_idx2coord, reply, ordered_indices)

        # 将识别结果写回原 JSON (identifiers)
        for idx_k, bbox_obj in index_to_bbox_obj.items():
            ids_list = idx_to_ids.get(idx_k, [])
            bbox_obj['identifiers'] = ids_list  # 新增字段

        processed_images += 1

    print(f"[SUMMARY] total_images={total_images} processed={processed_images} missing_images={missing_image_count} skipped_reaction_empty={skipped_reaction_empty} model_calls={model_call_count}")
    if timed_model_call_count > 0:
        avg_infer_time = total_infer_time / timed_model_call_count
        print(f"[TIME_SUMMARY] timed_model_calls={timed_model_call_count} total_infer_time={total_infer_time:.4f}s avg_infer_time={avg_infer_time:.4f}s/image")
    else:
        print("[TIME_SUMMARY] timed_model_calls=0 total_infer_time=0.0000s avg_infer_time=0.0000s/image")
    sys.stdout.flush()

def main(args):
    os.makedirs(args.middle_root_dir, exist_ok=True)
    os.makedirs(args.response_root_dir, exist_ok=True)
    os.makedirs(args.result_root_dir, exist_ok=True)

    # 读取整合 JSON
    try:
        with open(args.idt_json_path, 'r', encoding='utf-8') as f:
            idt_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取 JSON 失败: {e}")
        return

    if not idt_data.get('images'):
        print("[ERROR] JSON 中没有 images，直接退出.")
        return

    if args.dry_run:
        print("[DRY] 跳过模型加载。")
        args.model = None
        args.processor = None
    else:
        if not args.model_path:
            print("[ERROR] --model_path is required unless --dry_run is set.")
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            args.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            args.model.eval()
            args.processor = AutoProcessor.from_pretrained(args.model_path)
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            return

    process(args, idt_data)
    # 写回更新后的 JSON（含 identifiers）
    if args.updated_json_path:
        try:
            with open(args.updated_json_path, 'w', encoding='utf-8') as wf:
                json.dump(idt_data, wf, ensure_ascii=False, indent=2)
            print(f"[INFO] 已输出更新后的 JSON -> {args.updated_json_path}")
        except Exception as e:
            print(f"[WARN] 写入更新 JSON 失败: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_dir', type=str, required=True)
    parser.add_argument('--idt_json_path', type=str, required=True)
    parser.add_argument('--response_root_dir', type=str, required=True)
    parser.add_argument('--middle_root_dir', type=str, required=True)
    parser.add_argument('--result_root_dir', type=str, required=True)
    parser.add_argument('--updated_json_path', type=str, default='',
                        help='写回新增 identifiers 字段后的完整 JSON 输出路径')
    parser.add_argument('--model_path', type=str, required=False, default='songjhPKU/Mid-Mapper')
    parser.add_argument('--line_width', type=int, default=3)
    # 去除 scale_factor / image_downscale / dpi 相关参数（不再缩放）
    parser.add_argument('--min_font_size', type=int, default=24)
    parser.add_argument('--font_step', type=int, default=12)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--log_file', type=str, default='')
    args = parser.parse_args()
    if args.log_file:
        class Tee:
            def __init__(self, path):
                self.f = open(path, 'w', encoding='utf-8')
                self.stdout = sys.stdout
            def write(self, data):
                self.stdout.write(data)
                self.f.write(data)
            def flush(self):
                self.stdout.flush()
                self.f.flush()
        sys.stdout = Tee(args.log_file)
    main(args)
