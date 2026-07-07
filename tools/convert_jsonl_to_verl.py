#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IdtVP数据集预处理脚本（直接转换版本）
将已划分好的train和val的JSONL文件转换为verl可用的parquet格式

作者: 自动生成
日期: 2025
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSONL文件中加载数据
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        List[Dict]: 加载的数据列表
    """
    data = []
    print(f"📖 正在读取文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析每一行的JSON数据
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
    
    print(f"✅ 成功加载 {len(data)} 条数据")
    return data


def transform_to_verl_format(raw_data: List[Dict[str, Any]], data_source: str = "IDTVP_naive") -> List[Dict[str, Any]]:
    """
    将原始数据转换为verl需要的格式
    
    verl格式要求:
    {
        "data_source": str,           # 数据集标识符，用于奖励函数路由
        "prompt": List[Dict],          # 对话格式的prompt
        "reward_model": {              # 奖励相关信息
            "style": "rule",           # "rule"表示基于规则的奖励
            "ground_truth": str        # 正确答案
        }
    }
    
    Args:
        raw_data: 原始JSONL数据
        data_source: 数据集标识符，默认为"IDTVP_naive"
        
    Returns:
        List[Dict]: 转换后的verl格式数据
    """
    verl_data = []
    
    print(f"🔄 开始转换数据格式...")
    
    for idx, item in enumerate(tqdm(raw_data, desc="转换进度")):
        try:
            # ============================================
            # 步骤1: 提取原始数据的关键字段
            # ============================================
            messages = item.get("messages", [])
            images = item.get("images", [])
            
            # 检查数据完整性
            if not messages or len(messages) < 3:
                print(f"⚠️  警告: 第 {idx} 条数据messages字段不完整，跳过")
                continue
            
            # ============================================
            # 步骤2: 构造prompt (对话格式)
            # ============================================
            # IdtVP数据集包含: system消息、user消息、assistant消息
            # 对于verl，我们只需要system + user作为prompt
            # assistant的内容作为ground_truth
            
            prompt_messages = []
            ground_truth = None
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    # 保留system消息（任务描述）
                    prompt_messages.append({
                        "role": "system",
                        "content": content
                    })
                    
                elif role == "user":
                    # 保留user消息（包含<image>标记和问题）
                    # 注意：verl会在数据集中自动处理<image>标记
                    prompt_messages.append({
                        "role": "user",
                        "content": content
                    })
                    
                elif role == "assistant":
                    # assistant的内容作为ground_truth（正确答案）
                    ground_truth = content
            
            # 检查是否提取到了ground_truth
            if ground_truth is None:
                print(f"⚠️  警告: 第 {idx} 条数据没有assistant回答，跳过")
                continue
            
            # ============================================
            # 步骤3: 构造verl格式的数据项
            # ============================================
            verl_item = {
                # 🔴 重要: data_source用于奖励函数路由
                # 必须与奖励函数中的判断条件匹配
                "data_source": data_source,
                
                # 🔴 重要: prompt必须是对话格式的列表
                "prompt": prompt_messages,
                
                # 🔴 重要: reward_model包含评估所需的信息
                "reward_model": {
                    "style": "rule",           # "rule"表示基于规则的奖励函数
                    "ground_truth": ground_truth  # 正确答案，供奖励函数使用
                },
                
                # 🔴 重要: images字段用于多模态数据
                # verl会自动处理images，将图片路径传递给模型
                "images": images
            }
            
            # ============================================
            # 步骤4: （可选）添加额外信息
            # ============================================
            # 如果未来需要添加额外信息用于奖励函数或分析，可以在这里添加
            # verl_item["extra_info"] = {
            #     "original_index": idx,
            #     "num_images": len(images),
            #     "response_length": len(ground_truth)
            # }
            
            verl_data.append(verl_item)
            
        except Exception as e:
            print(f"⚠️  警告: 处理第 {idx} 条数据时出错: {e}")
            continue
    
    print(f"✅ 转换完成，有效数据: {len(verl_data)} 条")
    return verl_data


def save_to_parquet(
    data: List[Dict[str, Any]], 
    output_path: str,
    dataset_name: str = "dataset"
):
    """
    将数据保存为parquet格式
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
        dataset_name: 数据集名称（用于日志输出）
    """
    print(f"💾 正在保存{dataset_name}到: {output_path}")
    
    # 使用datasets库保存（推荐）
    # datasets库会自动处理复杂的数据类型
    try:
        dataset = Dataset.from_list(data)
        dataset.to_parquet(output_path)
        print(f"✅ {dataset_name}保存成功 ({len(data)} 条数据)")
        
        # 输出数据示例
        if len(data) > 0:
            print(f"\n📋 {dataset_name}数据示例:")
            print(f"   data_source: {data[0].get('data_source')}")
            print(f"   prompt条数: {len(data[0].get('prompt', []))}")
            print(f"   images数量: {len(data[0].get('images', []))}")
            if data[0].get('images'):
                print(f"   第一张图片: {data[0]['images'][0]}")
            print(f"   ground_truth长度: {len(data[0]['reward_model']['ground_truth'])} 字符")
            
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        raise


def process_single_file(
    input_file: str,
    output_file: str,
    data_source: str,
    dataset_name: str
):
    """
    处理单个JSONL文件：加载、转换、保存
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的parquet文件路径
        data_source: 数据集标识符
        dataset_name: 数据集名称（用于日志）
    """
    print(f"\n{'='*60}")
    print(f"📦 处理 {dataset_name}")
    print(f"{'='*60}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 加载原始数据
    raw_data = load_jsonl(input_file)
    
    if len(raw_data) == 0:
        raise ValueError(f"{dataset_name}没有成功加载任何数据！")
    
    # 转换数据格式
    verl_data = transform_to_verl_format(raw_data, data_source=data_source)
    
    if len(verl_data) == 0:
        raise ValueError(f"{dataset_name}转换后没有有效数据！")
    
    # 保存为parquet格式
    save_to_parquet(verl_data, output_file, dataset_name)
    
    return len(raw_data), len(verl_data)


def main():
    """主函数：完整的数据处理流程"""
    
    # ============================================
    # 步骤1: 解析命令行参数
    # ============================================
    parser = argparse.ArgumentParser(
        description="将已划分好的IdtVP train/val JSONL文件转换为verl可用的parquet格式"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="训练集JSONL文件路径"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="验证集JSONL文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/parquet",
        help="输出目录"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="IDTVP_naive",
        help="数据集标识符（用于奖励函数路由）"
    )
    parser.add_argument(
        "--train_output_name",
        type=str,
        default="train.parquet",
        help="训练集输出文件名（默认: train.parquet）"
    )
    parser.add_argument(
        "--val_output_name",
        type=str,
        default="val.parquet",
        help="验证集输出文件名（默认: val.parquet）"
    )
    
    args = parser.parse_args()
    
    # ============================================
    # 步骤2: 验证输入输出路径
    # ============================================
    print("=" * 60)
    print("📦 IdtVP数据集转换工具（直接转换版本）")
    print("=" * 60)
    print(f"训练集文件: {args.train_file}")
    print(f"验证集文件: {args.val_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据源标识: {args.data_source}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============================================
    # 步骤3: 处理训练集
    # ============================================
    train_output_path = os.path.join(args.output_dir, args.train_output_name)
    train_raw_count, train_valid_count = process_single_file(
        input_file=args.train_file,
        output_file=train_output_path,
        data_source=args.data_source,
        dataset_name="训练集"
    )
    
    # ============================================
    # 步骤4: 处理验证集
    # ============================================
    val_output_path = os.path.join(args.output_dir, args.val_output_name)
    val_raw_count, val_valid_count = process_single_file(
        input_file=args.val_file,
        output_file=val_output_path,
        data_source=args.data_source,
        dataset_name="验证集"
    )
    
    # ============================================
    # 步骤5: 输出统计信息
    # ============================================
    print("\n" + "=" * 60)
    print("📊 数据处理完成！统计信息:")
    print("=" * 60)
    print(f"训练集:")
    print(f"  原始数据: {train_raw_count} 条")
    print(f"  有效数据: {train_valid_count} 条")
    print(f"  数据完整性: {train_valid_count/train_raw_count*100:.2f}%")
    print(f"  输出文件: {train_output_path}")
    print(f"\n验证集:")
    print(f"  原始数据: {val_raw_count} 条")
    print(f"  有效数据: {val_valid_count} 条")
    print(f"  数据完整性: {val_valid_count/val_raw_count*100:.2f}%")
    print(f"  输出文件: {val_output_path}")
    print(f"\n总计:")
    print(f"  原始数据总数: {train_raw_count + val_raw_count} 条")
    print(f"  有效数据总数: {train_valid_count + val_valid_count} 条")
    print("=" * 60)
    
    # ============================================
    # 步骤6: 提示下一步操作
    # ============================================
    print("\n💡 下一步操作:")
    print("1. 编写奖励函数，处理 data_source='{}' 的数据".format(args.data_source))
    print("2. 在训练脚本中使用以下配置:")
    print(f"   data.train_files={train_output_path}")
    print(f"   data.val_files={val_output_path}")
    print(f"   reward_model.custom_reward_function.path=<你的奖励函数.py>")
    print("=" * 60)


if __name__ == "__main__":
    main()
