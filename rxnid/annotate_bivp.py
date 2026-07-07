#!/usr/bin/env python3

import json
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
import re
import time
import argparse
def draw_identifier_num(num, xmin, ymin, draw, font, padding=2):
    bbox = draw.textbbox((0, 0), str(num), font=font)
    # 计算文本的宽度和高度
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 计算黑底框的位置和大小，确保它紧密地包裹住字体
    fill_x1 = xmin  # 黑框的左上角 X 坐标
    fill_y1 = ymin - 20 # 黑框的顶部 Y 坐标（紧密包裹字体）
    fill_x2 = fill_x1 + text_width + 2 * padding  # 黑框的右下角 X 坐标
    # fill_y2 = fill_y1 + text_height + 10 * padding  # 黑框的底部 Y 坐标
    fill_y2 = fill_y1 + 1.8*text_height   # 黑框的底部 Y 坐标
    # 绘制黑底框（确保它位于边界框内，并且足够大来包裹文字）
    draw.rectangle((fill_x1, fill_y1, fill_x2, fill_y2), fill="black")

    # 计算文本的绘制位置，确保其在黑框内居中
    text_x = fill_x1 + padding  # 文本的 X 坐标
    text_y = fill_y1 + padding  # 文本的 Y 坐标

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

def filter_bboxes_by_confidence(mol_det_dict, threshold):
    """
    根据置信度阈值过滤边界框
    :param mol_det_dict: 检测结果字典
    :param threshold: 置信度阈值
    :return: 过滤后的边界框列表
    """
    filtered_bboxes = []
    for elem in mol_det_dict:
        if elem['confidence'] >= threshold:
            filtered_bboxes.append(elem['bbox_xyxy'])
    return filtered_bboxes

def process_with_threshold(args, threshold):
    image_root_dir = args.image_root_dir
    det_json_root_dir = args.det_json_root_dir
    middle_root_dir = args.middle_root_dir
    line_width = args.line_width

    # 为当前阈值创建对应的输出目录
    threshold_dir = os.path.join(middle_root_dir, f"threshold_{threshold}")
    os.makedirs(threshold_dir, exist_ok=True)
    image_name_list = os.listdir(image_root_dir)

    for image_name in image_name_list:
        print("处理图片：", image_name)
        image_path = os.path.join(image_root_dir, image_name)
        # 尝试不同的图片后缀来找到对应的json文件
        json_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_json_path = os.path.join(det_json_root_dir, image_name.replace(ext, '.json'))
            if os.path.exists(potential_json_path):
                json_path = potential_json_path
                break
        # 如果没有找到对应的json文件，跳过当前图片
        if json_path is None:
            continue
        
        # 读取yolo分子式检测结果
        with open(json_path, 'r') as f:
            mol_det_dict = json.load(f)
        
        image = Image.open(image_path)
        
        # 根据置信度阈值过滤边界框
        bboxes = filter_bboxes_by_confidence(mol_det_dict, threshold)
        
        # 计算动态字体大小（根据最小边界框的大小）
        if bboxes:
            # 计算每个边界框的宽度和高度
            bbox_sizes = []
            for box in bboxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                bbox_sizes.append(min(width, height))  # 取宽度和高度的较小值
            
            # 根据最小边界框的尺寸设置字体大小，采用更合理的比例
            min_bbox_size = min(bbox_sizes)
            
            if min_bbox_size < 50:
                font_size = max(int(min_bbox_size * 0.4), args.min_font_size)  # 小框用较大比例，最小16
            elif min_bbox_size < 100:
                font_size = max(int(min_bbox_size * 0.3), args.min_font_size + args.font_step)  # 中等框用中等比例，最小20
            else:
                font_size = max(int(min_bbox_size * 0.2), args.min_font_size + 2 * args.font_step)  # 大框用较小比例，最小24
            
            # 设置字体大小的上限，避免过大
            font_size = min(font_size, 48)
        else:
            # 如果没有边界框，使用默认字体大小
            font_size = 36
            
        font = ImageFont.load_default(size=font_size)  # 设置动态字体大小
        
        middle_image = image.copy()
        draw = ImageDraw.Draw(middle_image)
        
        # 记录mol边界框序号与mol边界框的对应字典
        mol_idx2coord = {}
        
        # 将bboxes按从左到右从上到下进行排序
        bboxes = get_reading_order(bboxes)
        
        norm_bboxes = []
        for i, box in enumerate(bboxes):
            x1_mol, y1_mol, x2_mol, y2_mol = box
            x1_mol = x1_mol * args.dpi / 200
            x2_mol = x2_mol * args.dpi / 200
            y1_mol = y1_mol * args.dpi / 200
            y2_mol = y2_mol * args.dpi / 200
            mol_idx2coord[i + 1] = [x1_mol, y1_mol, x2_mol, y2_mol]
            draw.rectangle((x1_mol, y1_mol, x2_mol, y2_mol), outline="blue", width=line_width)
            draw_identifier_num(i + 1, x1_mol, y1_mol + 20, draw, font)
            norm_bboxes.append([x1_mol / image.width, y1_mol / image.height, x2_mol / image.width, y2_mol / image.height])
        
        # 保存绘制后的图像到对应阈值的目录
        middle_image_path = os.path.join(threshold_dir, image_name)
        middle_image.save(middle_image_path)

def process(args):
    process_with_threshold(args, args.confidence_threshold)

def main(args):
    os.makedirs(args.middle_root_dir, exist_ok=True)
    process(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw BIVP-style indexed bounding boxes on reaction diagrams.")
    parser.add_argument('--det_json_root_dir', type=str, required=True, help='Directory containing detector JSON files.')
    parser.add_argument('--middle_root_dir', type=str, required=True, help='Directory for annotated output images.')
    parser.add_argument('--image_root_dir', type=str, required=True, help='Directory containing raw input images.')
    parser.add_argument('--confidence_threshold', type=float, default=0.5) # YOLO框置信度阈值，默认为0.5，在此阈值之下的框不绘制
    parser.add_argument('--dpi', type=int, default=400) # 图片的dpi，默认为400,使用dpi/200决定缩放比例
    parser.add_argument('--line_width', type=int, default=4) # 框线的粗细，默认为4
    parser.add_argument('--min_font_size', type=int, default=24) # 最小字体大小，默认为24，这个字体大小是针对于最小框的，更大尺寸的框会在此基础上动态调整，依次+font_step
    parser.add_argument('--font_step', type=int, default=12) # 字体大小步长，默认为12
    args = parser.parse_args()
    main(args)
