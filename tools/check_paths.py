#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 jsonl 文件中每条记录的图片路径是否存在
如果路径不存在，记录到文件中
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def check_image_paths(jsonl_file, output_file=None):
    """
    检查 jsonl 文件中每条记录的图片路径是否存在
    
    Args:
        jsonl_file: jsonl 文件路径
        output_file: 输出文件路径，如果为 None，则使用默认路径
    """
    if output_file is None:
        # 默认输出文件名为原文件名 + _missing_paths.txt
        base_name = Path(jsonl_file).stem
        output_file = Path(jsonl_file).parent / f"{base_name}_missing_paths.txt"
    
    missing_paths = []
    total_records = 0
    total_images = 0
    missing_count = 0
    
    print(f"开始检查文件: {jsonl_file}")
    print(f"输出文件: {output_file}")
    
    # 读取 jsonl 文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        # 先统计总行数（用于进度条）
        lines = f.readlines()
        total_lines = len(lines)
    
    # 逐行处理
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="检查进度"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                total_records += 1
                
                # 检查 images 字段
                if 'images' in record and isinstance(record['images'], list):
                    for img_path in record['images']:
                        total_images += 1
                        if not os.path.exists(img_path):
                            missing_count += 1
                            missing_paths.append({
                                'line_num': line_num,
                                'path': img_path
                            })
                            print(f"第 {line_num} 行: 路径不存在 - {img_path}")
                else:
                    print(f"警告: 第 {line_num} 行没有 'images' 字段或格式不正确")
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    # 保存不存在的路径
    if missing_paths:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 不存在的图片路径列表\n")
            f.write(f"# 总记录数: {total_records}\n")
            f.write(f"# 总图片数: {total_images}\n")
            f.write(f"# 不存在路径数: {missing_count}\n")
            miss_rate = missing_count / total_images * 100 if total_images else 0
            f.write(f"# 缺失率: {miss_rate:.2f}%\n")
            f.write(f"{'='*80}\n\n")
            
            for item in missing_paths:
                f.write(f"行号: {item['line_num']}\n")
                f.write(f"路径: {item['path']}\n")
                f.write(f"{'-'*80}\n")
        
        print(f"\n检查完成!")
        print(f"总记录数: {total_records}")
        print(f"总图片数: {total_images}")
        print(f"不存在路径数: {missing_count}")
        miss_rate = missing_count / total_images * 100 if total_images else 0
        print(f"缺失率: {miss_rate:.2f}%")
        print(f"不存在的路径已保存到: {output_file}")
    else:
        print(f"\n检查完成!")
        print(f"总记录数: {total_records}")
        print(f"总图片数: {total_images}")
        print(f"所有路径都存在！✓")
        # 即使没有缺失路径，也创建一个文件记录统计信息
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 路径检查结果\n")
            f.write(f"# 总记录数: {total_records}\n")
            f.write(f"# 总图片数: {total_images}\n")
            f.write(f"# 不存在路径数: 0\n")
            f.write(f"# 所有路径都存在！✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether image paths in an IdtVP JSONL file exist.")
    parser.add_argument("jsonl_file", help="Input JSONL file with an images field.")
    parser.add_argument("--output_file", default=None, help="Missing-path report. Defaults to <jsonl_stem>_missing_paths.txt.")
    args = parser.parse_args()

    check_image_paths(args.jsonl_file, args.output_file)
