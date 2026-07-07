#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDTVP化学反应识别任务的奖励函数

基于JSON结构匹配的Soft Match和Hybrid Match评估
用于verl的GRPO训练

数据格式：
- 每个反应包含 reactants/conditions/products
- 每个元素可以是：{"idt": "<content>"} 或 {"text": "<content>"}
- idt可以包含多个指代符，用逗号连接，如 {"idt": "1a,2a"}

匹配规则：
- Soft Match: 只比较identifier，忽略text
- Hybrid Match: identifier + text都要匹配
- 对于identifier元素：两个元素的指代符列表有交集就算匹配成功

作者: 基于BIVP评测逻辑重新设计
日期: 2025
"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
from scipy.optimize import linear_sum_assignment
import numpy as np


# ============================================
# 全局配置
# ============================================
COND_TEXT_EDIT_DIST_THRES = 0.20  # 条件文本允许的最大归一化编辑距离


# ============================================
# 文本处理工具
# ============================================
def extract_json_from_markdown(text: str) -> str:
    """
    从可能包含markdown代码块的文本中提取JSON
    
    支持的格式：
    1. ```json ... ```
    2. ``` ... ```
    3. 纯JSON文本
    
    Args:
        text: 输入文本
    
    Returns:
        提取的JSON字符串
    """
    text = text.strip()
    
    # 尝试提取markdown代码块
    # 匹配 ```json ... ``` 或 ``` ... ```
    import re
    
    # 模式1: ```json ... ```
    pattern1 = r'```json\s*\n(.*?)\n```'
    match = re.search(pattern1, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 模式2: ``` ... ```
    pattern2 = r'```\s*\n(.*?)\n```'
    match = re.search(pattern2, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 模式3: 没有换行的情况 ```json...``` 或 ```...```
    pattern3 = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern3, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果没有代码块标记，返回原文本
    return text


def levenshtein_distance(a: str, b: str) -> int:
    """计算Levenshtein编辑距离"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr.append(prev[j - 1])
            else:
                curr.append(1 + min(prev[j - 1], prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """归一化编辑距离 ∈ [0,1]，0表示完全相同"""
    max_len = max(len(a), len(b), 1)
    return levenshtein_distance(a, b) / max_len


def normalize_text(txt: str) -> str:
    """
    化学文本归一化：处理全角、上下标、符号等
    """
    if not txt:
        return ""
    txt = txt.strip()

    # 全角 → 半角
    txt = txt.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    txt = txt.translate(str.maketrans(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ", 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ))
    txt = txt.translate(str.maketrans(
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ", 
        "abcdefghijklmnopqrstuvwxyz"
    ))
    txt = txt.translate(str.maketrans("＋－×÷＝（）【】｛｝［］", "+-*/=()[]{}[]"))

    # Unicode下标/上标
    sub_map = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "ₐ": "a", "ₑ": "e", "ᵢ": "i", "ₒ": "o", 
        "ᵣ": "r", "ᵤ": "u", "ᵥ": "v", "ₓ": "x"
    }
    sup_map = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
        "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j",
        "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n", "ᵒ": "o",
        "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u",
        "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z"
    }
    for k, v in sub_map.items():
        txt = txt.replace(k, v)
    for k, v in sup_map.items():
        txt = txt.replace(k, v)

    # 符号统一
    txt = txt.replace("×", "*").replace("÷", "/")
    txt = txt.replace("≤", "<=").replace("≥", ">=")
    txt = txt.replace("≠", "!=").replace("≈", "~")
    txt = txt.replace("→", "->").replace("←", "<-").replace("↔", "<->")
    txt = txt.replace("⟶", "->").replace("⟵", "<-").replace("⟷", "<->")
    txt = txt.replace("（", "(").replace("）", ")")
    txt = txt.replace("【", "[").replace("】", "]")
    txt = txt.replace("｛", "{").replace("｝", "}")
    
    # 删除所有空白字符
    txt = re.sub(r"\s+", "", txt)
    return txt.lower()


def normalize_identifier(idt: str) -> str:
    """
    归一化单个identifier字符串
    去除空白，转小写
    """
    if not idt:
        return ""
    return idt.strip().lower()


# ============================================
# 提取工具函数
# ============================================
def parse_identifier_string(idt_str: str) -> Set[str]:
    """
    解析identifier字符串，返回identifier集合
    
    支持逗号分隔的多个identifier，如 "1a,2a" -> {"1a", "2a"}
    
    Args:
        idt_str: identifier字符串
    
    Returns:
        归一化后的identifier集合
    """
    if not idt_str:
        return set()
    
    # 按逗号分割
    parts = idt_str.split(",")
    result = set()
    for part in parts:
        normalized = normalize_identifier(part)
        if normalized:
            result.add(normalized)
    return result


def extract_identifier_elements(items: List[Dict]) -> List[Set[str]]:
    """
    从反应部分提取所有idt元素
    
    每个idt元素是一个集合（因为可能包含多个逗号分隔的identifier）
    
    Args:
        items: 反应部分的元素列表
    
    Returns:
        idt集合的列表，每个元素对应一个idt结构
    """
    identifier_sets = []
    for item in items:
        idt_value = item.get("idt")
        if idt_value is not None:
            idt_set = parse_identifier_string(str(idt_value))
            if idt_set:
                identifier_sets.append(idt_set)
    return identifier_sets


def extract_all_identifiers(items: List[Dict]) -> Set[str]:
    """
    从反应部分提取所有idt，展平成一个集合
    
    Args:
        items: 反应部分的元素列表
    
    Returns:
        所有idt的集合（展平后）
    """
    all_idts = set()
    for item in items:
        idt_value = item.get("idt")
        if idt_value is not None:
            idt_set = parse_identifier_string(str(idt_value))
            all_idts.update(idt_set)
    return all_idts


def extract_texts(items: List[Dict]) -> List[str]:
    """
    从反应部分提取所有text字段（归一化后）
    
    Args:
        items: 反应部分的元素列表
    
    Returns:
        归一化后的text列表
    """
    texts = []
    for item in items:
        if "text" in item:
            text = item.get("text", "").strip()
            if text:
                texts.append(normalize_text(text))
    return texts


def canonical_text_string(texts: List[str]) -> str:
    """
    将文本列表转换为字符多重集的顺序无关字符串
    用于条件文本的比较
    
    Args:
        texts: 归一化后的文本列表
    
    Returns:
        排序后的字符串
    """
    if not texts:
        return ""
    
    # 合并所有文本的字符并排序
    all_chars = []
    for text in texts:
        all_chars.extend(list(text))
    all_chars.sort()
    return "".join(all_chars)


# ============================================
# Identifier元素匹配
# ============================================
def identifier_elements_match(gold_elements: List[Set[str]], pred_elements: List[Set[str]]) -> bool:
    """
    比较两个identifier元素列表是否匹配
    
    使用匈牙利算法进行一对一匹配
    两个元素有交集就算匹配成功
    
    Args:
        gold_elements: 标注的identifier元素列表
        pred_elements: 预测的identifier元素列表
    
    Returns:
        是否匹配
    """
    n_gold = len(gold_elements)
    n_pred = len(pred_elements)
    
    # 特殊情况：都为空
    if n_gold == 0 and n_pred == 0:
        return True
    
    # 数量不同，直接不匹配
    if n_gold != n_pred:
        return False
    
    # 构造cost矩阵：0表示有交集（匹配），1表示无交集（不匹配）
    cost = np.ones((n_gold, n_pred))
    for i, gold_set in enumerate(gold_elements):
        for j, pred_set in enumerate(pred_elements):
            # 有交集就算匹配
            if gold_set & pred_set:
                cost[i, j] = 0
    
    # 匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # 检查是否所有都匹配上了
    total_cost = sum(cost[r, c] for r, c in zip(row_ind, col_ind))
    return total_cost == 0


# ============================================
# Soft Match: 仅比较identifier
# ============================================
def compare_soft(gold_reaction: Dict, pred_reaction: Dict) -> bool:
    """
    Soft Match: 仅比较identifier
    
    规则：
    1. 提取reactants和conditions中的所有identifier元素
    2. 提取products中的所有identifier元素
    3. 使用匈牙利算法进行元素级匹配（有交集即匹配）
    
    Args:
        gold_reaction: 标注反应
        pred_reaction: 预测反应
    
    Returns:
        是否匹配
    """
    # 提取反应物侧的identifier元素（reactants + conditions中的identifier）
    gold_reactant_elements = extract_identifier_elements(gold_reaction.get("reactants", []))
    gold_cond_idt_elements = extract_identifier_elements(gold_reaction.get("conditions", []))
    gold_r_elements = gold_reactant_elements + gold_cond_idt_elements
    
    pred_reactant_elements = extract_identifier_elements(pred_reaction.get("reactants", []))
    pred_cond_idt_elements = extract_identifier_elements(pred_reaction.get("conditions", []))
    pred_r_elements = pred_reactant_elements + pred_cond_idt_elements
    
    # 提取产物侧的identifier元素
    gold_p_elements = extract_identifier_elements(gold_reaction.get("products", []))
    pred_p_elements = extract_identifier_elements(pred_reaction.get("products", []))
    
    # 比较反应物侧
    if not identifier_elements_match(gold_r_elements, pred_r_elements):
        return False
    
    # 比较产物侧
    if not identifier_elements_match(gold_p_elements, pred_p_elements):
        return False
    
    return True


# ============================================
# Hybrid Match: identifier + text
# ============================================
def compare_hybrid(gold_reaction: Dict, pred_reaction: Dict) -> bool:
    """
    Hybrid Match: 比较identifier + text
    
    规则：
    1. 反应物和产物：
       - identifier元素必须匹配（使用交集匹配）
       - text也要匹配（使用集合比较）
    2. 条件：
       - identifier元素必须匹配
       - text使用字符多重集+编辑距离阈值
    
    Args:
        gold_reaction: 标注反应
        pred_reaction: 预测反应
    
    Returns:
        是否匹配
    """
    # 1. 比较反应物和产物
    for role in ("reactants", "products"):
        gold_items = gold_reaction.get(role, [])
        pred_items = pred_reaction.get(role, [])
        
        # 提取identifier元素
        gold_idt_elements = extract_identifier_elements(gold_items)
        pred_idt_elements = extract_identifier_elements(pred_items)
        
        # identifier元素必须匹配
        if not identifier_elements_match(gold_idt_elements, pred_idt_elements):
            return False
        
        # 提取text（归一化后）
        gold_texts = extract_texts(gold_items)
        pred_texts = extract_texts(pred_items)
        
        # text也要匹配（作为多重集比较，因为顺序可能不同）
        if Counter(gold_texts) != Counter(pred_texts):
            return False
    
    # 2. 比较条件
    gold_conds = gold_reaction.get("conditions", [])
    pred_conds = pred_reaction.get("conditions", [])
    
    # 条件中的identifier元素必须匹配
    gold_cond_idt_elements = extract_identifier_elements(gold_conds)
    pred_cond_idt_elements = extract_identifier_elements(pred_conds)
    
    if not identifier_elements_match(gold_cond_idt_elements, pred_cond_idt_elements):
        return False
    
    # 条件文本使用字符多重集+编辑距离
    gold_cond_texts = extract_texts(gold_conds)
    pred_cond_texts = extract_texts(pred_conds)
    
    gold_cond_str = canonical_text_string(gold_cond_texts)
    pred_cond_str = canonical_text_string(pred_cond_texts)
    
    # 一个有文本一个没有
    if bool(gold_cond_str) != bool(pred_cond_str):
        return False
    
    # 都有文本时检查编辑距离
    if gold_cond_str and pred_cond_str:
        if normalized_edit_distance(gold_cond_str, pred_cond_str) > COND_TEXT_EDIT_DIST_THRES:
            return False
    
    return True


# ============================================
# 反应列表匹配
# ============================================
def match_reactions(
    gold_reactions: List[Dict],
    pred_reactions: List[Dict],
    match_type: str
) -> Tuple[int, int, int]:
    """
    匹配两个反应列表，计算TP/FP/FN
    
    使用匈牙利算法进行一对一匹配
    
    Args:
        gold_reactions: 标注反应列表
        pred_reactions: 预测反应列表
        match_type: "soft" 或 "hybrid"
    
    Returns:
        (tp, fp, fn)
    """
    if not gold_reactions and not pred_reactions:
        return 0, 0, 0
    
    # 选择比较函数
    compare_fn = compare_soft if match_type == "soft" else compare_hybrid
    
    n_gold = len(gold_reactions)
    n_pred = len(pred_reactions)
    
    # 构造cost矩阵：0表示匹配，1表示不匹配
    cost = np.ones((n_gold, n_pred))
    for i, gold_rxn in enumerate(gold_reactions):
        for j, pred_rxn in enumerate(pred_reactions):
            if compare_fn(gold_rxn, pred_rxn):
                cost[i, j] = 0
    
    # 匈牙利算法求解最优匹配
    if n_gold > 0 and n_pred > 0:
        row_ind, col_ind = linear_sum_assignment(cost)
        # 只统计cost=0的匹配
        tp = sum(1 for r, c in zip(row_ind, col_ind) if cost[r, c] == 0)
    else:
        tp = 0
    
    fp = n_pred - tp  # 预测了但未匹配上
    fn = n_gold - tp  # 标注了但未被预测到
    
    return tp, fp, fn


def compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    """
    计算Precision和Recall
    
    Returns:
        (precision, recall)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    计算Precision、Recall和F1（调和平均）

    F1 = 2 * (P * R) / (P + R)

    Returns:
        (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


# ============================================
# 主奖励函数
# ============================================
def compute_idtvp_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    # 可配置的权重参数
    soft_weight: float = 0.5,      # Soft Match的权重
    hybrid_weight: float = 1.0,    # Hybrid Match的权重
    normalize: bool = True,        # 是否归一化到[0,1]
    **kwargs
) -> Dict[str, Any]:
    """
    IDTVP化学反应识别的奖励函数
    
    奖励计算方式:
    - Soft Match: F1_soft * soft_weight
    - Hybrid Match: F1_hybrid * hybrid_weight
    - 总奖励 = Soft分数 + Hybrid分数
    
    如果normalize=True，则最终归一化到[0,1]
    
    Args:
        data_source: 数据集标识符，应该是"IDTVP_naive"
        solution_str: 模型生成的回答（JSON字符串）
        ground_truth: 标注的正确答案（JSON字符串）
        extra_info: 额外信息（未使用）
        soft_weight: Soft Match权重
        hybrid_weight: Hybrid Match权重
        normalize: 是否归一化
        **kwargs: 其他参数
    
    Returns:
        Dict包含详细指标和最终score
    """
    try:
        # ====================================
        # 步骤1: 解析JSON
        # ====================================
        # 先尝试从markdown中提取JSON
        solution_str_clean = extract_json_from_markdown(solution_str)
        
        try:
            pred_reactions = json.loads(solution_str_clean)
        except json.JSONDecodeError as e:
            # JSON解析失败，返回0分
            # 注意：键顺序必须与成功路径完全一致，否则多worker concat时会报AssertionError
            return {
                "score": 0.0,
                "error": f"Invalid prediction JSON: {str(e)}",
                "soft_precision": 0.0,
                "soft_recall": 0.0,
                "soft_f1": 0.0,
                "soft_score": 0.0,
                "soft_tp": 0,
                "soft_fp": 0,
                "soft_fn": 0,
                "hybrid_precision": 0.0,
                "hybrid_recall": 0.0,
                "hybrid_f1": 0.0,
                "hybrid_score": 0.0,
                "hybrid_tp": 0,
                "hybrid_fp": 0,
                "hybrid_fn": 0,
            }
        
        # ground_truth也尝试从markdown中提取
        ground_truth_clean = extract_json_from_markdown(ground_truth)
        
        try:
            gold_reactions = json.loads(ground_truth_clean)
        except json.JSONDecodeError as e:
            # ground_truth解析失败（不应该发生）
            # 注意：键顺序必须与成功路径完全一致，否则多worker concat时会报AssertionError
            return {
                "score": 0.0,
                "error": f"Invalid ground_truth JSON: {str(e)}",
                "soft_precision": 0.0,
                "soft_recall": 0.0,
                "soft_f1": 0.0,
                "soft_score": 0.0,
                "soft_tp": 0,
                "soft_fp": 0,
                "soft_fn": 0,
                "hybrid_precision": 0.0,
                "hybrid_recall": 0.0,
                "hybrid_f1": 0.0,
                "hybrid_score": 0.0,
                "hybrid_tp": 0,
                "hybrid_fp": 0,
                "hybrid_fn": 0,
            }
        
        # 确保是列表格式
        if not isinstance(pred_reactions, list):
            pred_reactions = [pred_reactions] if pred_reactions else []
        if not isinstance(gold_reactions, list):
            gold_reactions = [gold_reactions] if gold_reactions else []
        
        # ====================================
        # 步骤2: 计算Soft Match
        # ====================================
        soft_tp, soft_fp, soft_fn = match_reactions(
            gold_reactions, pred_reactions, match_type="soft"
        )
        soft_precision, soft_recall, soft_f1 = compute_precision_recall_f1(soft_tp, soft_fp, soft_fn)

        # Soft分数: F1（调和平均）
        soft_score = soft_f1 * soft_weight
        
        # ====================================
        # 步骤3: 计算Hybrid Match
        # ====================================
        hybrid_tp, hybrid_fp, hybrid_fn = match_reactions(
            gold_reactions, pred_reactions, match_type="hybrid"
        )
        hybrid_precision, hybrid_recall, hybrid_f1 = compute_precision_recall_f1(hybrid_tp, hybrid_fp, hybrid_fn)

        # Hybrid分数: F1（调和平均）
        hybrid_score = hybrid_f1 * hybrid_weight
        
        # ====================================
        # 步骤4: 计算总分
        # ====================================
        total_score = soft_score + hybrid_score
        
        # ====================================
        # 步骤5: 归一化（可选）
        # ====================================
        if normalize:
            # 最大可能分数:
            # soft_weight * 1.0 + hybrid_weight * 1.0
            max_possible = soft_weight * 1.0 + hybrid_weight * 1.0
            if max_possible > 0:
                total_score = total_score / max_possible
        
        # ====================================
        # 步骤6: 返回详细结果
        # ====================================
        return {
            # 主要奖励分数
            "score": float(total_score),
            
            # 错误信息（成功时为空字符串，保持字段一致性）
            "error": "",
            
            # Soft Match详细指标
            "soft_precision": float(soft_precision),
            "soft_recall": float(soft_recall),
            "soft_f1": float(soft_f1),
            "soft_score": float(soft_score),
            "soft_tp": int(soft_tp),
            "soft_fp": int(soft_fp),
            "soft_fn": int(soft_fn),
            
            # Hybrid Match详细指标
            "hybrid_precision": float(hybrid_precision),
            "hybrid_recall": float(hybrid_recall),
            "hybrid_f1": float(hybrid_f1),
            "hybrid_score": float(hybrid_score),
            "hybrid_tp": int(hybrid_tp),
            "hybrid_fp": int(hybrid_fp),
            "hybrid_fn": int(hybrid_fn),
        }
        
    except Exception as e:
        # 任何其他错误都返回0分
        # 注意：键顺序必须与成功路径完全一致，否则多worker concat时会报AssertionError
        return {
            "score": 0.0,
            "error": f"Unexpected error: {str(e)}",
            "soft_precision": 0.0,
            "soft_recall": 0.0,
            "soft_f1": 0.0,
            "soft_score": 0.0,
            "soft_tp": 0,
            "soft_fp": 0,
            "soft_fn": 0,
            "hybrid_precision": 0.0,
            "hybrid_recall": 0.0,
            "hybrid_f1": 0.0,
            "hybrid_score": 0.0,
            "hybrid_tp": 0,
            "hybrid_fp": 0,
            "hybrid_fn": 0,
        }


# ============================================
# 推荐的配置版本
# ============================================
def compute_idtvp_reward_v1(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    版本1: 原始设计
    Soft: (P+R)/2 * 0.5 → [0, 0.5]
    Hybrid: (P+R) * 1.0 → [0, 2.0]
    Total: [0, 2.5], 归一化后 [0, 1]
    
    Soft权重更大，鼓励完整匹配
    """
    return compute_idtvp_reward(
        data_source, solution_str, ground_truth, extra_info,
        soft_weight=1.0,
        hybrid_weight=0,
        normalize=True,
        **kwargs
    )


def compute_idtvp_reward_v2(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    版本2: 平衡设计
    Soft: (P+R)/2 * 1.0 → [0, 1.0]
    Hybrid: (P+R) * 1.0 → [0, 2.0]
    Total: [0, 3.0], 归一化后 [0, 1]
    
    Soft和Hybrid更平衡
    """
    return compute_idtvp_reward(
        data_source, solution_str, ground_truth, extra_info,
        soft_weight=1.0,
        hybrid_weight=1.0,
        normalize=True,
        **kwargs
    )


def compute_idtvp_reward_v3(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    版本3: 强调Hybrid
    Soft: (P+R)/2 * 0.25 → [0, 0.25]
    Hybrid: (P+R) * 1.5 → [0, 3.0]
    Total: [0, 3.25], 归一化后 [0, 1]
    
    更强调完整的Hybrid匹配
    """
    return compute_idtvp_reward(
        data_source, solution_str, ground_truth, extra_info,
        soft_weight=0.0,
        hybrid_weight=1.0,
        normalize=True,
        **kwargs
    )


# ============================================
# 默认导出（用于verl）
# ============================================
# 默认使用版本1
compute_idtvp_reward_default = compute_idtvp_reward_v1


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("IDTVP奖励函数测试")
    print("=" * 70)
    
    # ====================================
    # 测试1: 完美匹配
    # ====================================
    gold = json.dumps([{
        "reactants": [{"idt": "1"}, {"text": "H2O"}],
        "conditions": [{"text": "heat"}, {"text": "100°C"}],
        "products": [{"idt": "2"}]
    }])
    
    pred_perfect = json.dumps([{
        "reactants": [{"idt": "1"}, {"text": "H2O"}],
        "conditions": [{"text": "heat"}, {"text": "100°C"}],
        "products": [{"idt": "2"}]
    }])
    
    result = compute_idtvp_reward_v1("IDTVP_naive", pred_perfect, gold)
    print("\n【测试1: 完美匹配】")
    print(f"  总分: {result['score']:.4f}")
    print(f"  Soft - P/R: {result['soft_precision']:.2f} / {result['soft_recall']:.2f} → 分数: {result['soft_score']:.4f}")
    print(f"  Hybrid - P/R: {result['hybrid_precision']:.2f} / {result['hybrid_recall']:.2f} → 分数: {result['hybrid_score']:.4f}")
    print(f"  Soft TP/FP/FN: {result['soft_tp']}/{result['soft_fp']}/{result['soft_fn']}")
    print(f"  Hybrid TP/FP/FN: {result['hybrid_tp']}/{result['hybrid_fp']}/{result['hybrid_fn']}")
    
    # ====================================
    # 测试2: 仅identifier匹配（缺text）
    # ====================================
    pred_partial = json.dumps([{
        "reactants": [{"idt": "1"}],  # 缺少H2O
        "conditions": [],  # 缺少条件
        "products": [{"idt": "2"}]
    }])
    
    result2 = compute_idtvp_reward_v1("IDTVP_naive", pred_partial, gold)
    print("\n【测试2: 仅identifier匹配（缺text和conditions）】")
    print(f"  总分: {result2['score']:.4f}")
    print(f"  Soft - P/R: {result2['soft_precision']:.2f} / {result2['soft_recall']:.2f} → 分数: {result2['soft_score']:.4f}")
    print(f"  Hybrid - P/R: {result2['hybrid_precision']:.2f} / {result2['hybrid_recall']:.2f} → 分数: {result2['hybrid_score']:.4f}")
    print(f"  解释: Soft能匹配(只看identifier), Hybrid不能匹配(还要看text)")
    
    # ====================================
    # 测试3: identifier错误
    # ====================================
    pred_wrong = json.dumps([{
        "reactants": [{"idt": "3"}],  # 错误的identifier
        "conditions": [{"text": "heat"}],
        "products": [{"idt": "4"}]  # 错误的identifier
    }])
    
    result3 = compute_idtvp_reward_v1("IDTVP_naive", pred_wrong, gold)
    print("\n【测试3: identifier完全错误】")
    print(f"  总分: {result3['score']:.4f}")
    print(f"  Soft - P/R: {result3['soft_precision']:.2f} / {result3['soft_recall']:.2f} → 分数: {result3['soft_score']:.4f}")
    print(f"  Hybrid - P/R: {result3['hybrid_precision']:.2f} / {result3['hybrid_recall']:.2f} → 分数: {result3['hybrid_score']:.4f}")
    print(f"  解释: identifier不匹配，Soft和Hybrid都是0分")
    
    # ====================================
    # 测试4: 多identifier交集匹配
    # ====================================
    gold_multi_idt = json.dumps([{
        "reactants": [{"idt": "1a,2a"}],  # 包含多个identifier
        "conditions": [],
        "products": [{"idt": "3"}]
    }])
    
    pred_multi_idt = json.dumps([{
        "reactants": [{"idt": "1a"}],  # 只有一个，但有交集
        "conditions": [],
        "products": [{"idt": "3"}]
    }])
    
    result4 = compute_idtvp_reward_v1("IDTVP_naive", pred_multi_idt, gold_multi_idt)
    print("\n【测试4: 多identifier交集匹配】")
    print(f"  Gold reactants: idt='1a,2a', Pred reactants: idt='1a'")
    print(f"  总分: {result4['score']:.4f}")
    print(f"  Soft - P/R: {result4['soft_precision']:.2f} / {result4['soft_recall']:.2f}")
    print(f"  Hybrid - P/R: {result4['hybrid_precision']:.2f} / {result4['hybrid_recall']:.2f} → 分数: {result4['hybrid_score']:.4f}")
    print(f"  解释: '1a'和'1a,2a'有交集'1a'，所以算匹配成功")
    
    # ====================================
    # 测试5: 多个反应
    # ====================================
    gold_multi = json.dumps([
        {
            "reactants": [{"idt": "1"}],
            "conditions": [{"text": "MeOH, r.t."}],
            "products": [{"idt": "2"}]
        },
        {
            "reactants": [{"idt": "2"}],
            "conditions": [{"text": "KOt-Bu"}, {"text": "MeOH, r.t."}],
            "products": [{"idt": "3"}]
        }
    ])
    
    # 预测对了第一个，第二个identifier错误
    pred_multi = json.dumps([
        {
            "reactants": [{"idt": "1"}],
            "conditions": [{"text": "MeOH, r.t."}],
            "products": [{"idt": "2"}]
        },
        {
            "reactants": [{"idt": "2"}],
            "conditions": [{"text": "KOt-Bu"}],
            "products": [{"idt": "3"}]  # 错误
        }
    ])
    
    result5 = compute_idtvp_reward_v1("IDTVP_naive", pred_multi, gold_multi)
    print("\n【测试5: 多个反应（1对1错）】")
    print(f"  总分: {result5['score']:.4f}")
    print(f"  Soft - P/R: {result5['soft_precision']:.2f} / {result5['soft_recall']:.2f}")
    print(f"  Soft TP/FP/FN: {result5['soft_tp']}/{result5['soft_fp']}/{result5['soft_fn']}")
    print(f"  Hybrid - P/R: {result5['hybrid_precision']:.2f} / {result5['hybrid_recall']:.2f}")
    print(f"  Hybrid TP/FP/FN: {result5['hybrid_tp']}/{result5['hybrid_fp']}/{result5['hybrid_fn']}")
    
    # ====================================
    # 测试6: JSON格式错误
    # ====================================
    pred_invalid = "This is not valid JSON"
    result6 = compute_idtvp_reward_v1("IDTVP_naive", pred_invalid, gold)
    print("\n【测试6: 无效JSON】")
    print(f"  总分: {result6['score']:.4f}")
    print(f"  错误信息: {result6.get('error', 'N/A')}")
    
    # ====================================
    # 测试7: Markdown代码块格式
    # ====================================
    gold_markdown = json.dumps([{
        "reactants": [{"idt": "1"}],
        "conditions": [],
        "products": [{"idt": "2"}]
    }])
    
    pred_markdown = '```json\n[{"reactants": [{"idt": "1"}], "conditions": [], "products": [{"idt": "2"}]}]\n```'
    
    result7 = compute_idtvp_reward_v1("IDTVP_naive", pred_markdown, gold_markdown)
    print("\n【测试7: Markdown代码块格式】")
    print(f"  Pred使用```json...```包裹")
    print(f"  总分: {result7['score']:.4f}")
    print(f"  解释: 支持从markdown代码块中提取JSON")
    
    # ====================================
    # 测试8: 实际示例格式
    # ====================================
    gold_real = json.dumps([{
        "reactants": [{"idt": "1"}],
        "conditions": [
            {"text": "FeCl₃ (0.65 eq)\nDCE, 70 °C, 4 h"},
            {"text": "R²YYR² (1 eq)"}
        ],
        "products": [{"idt": "2"}]
    }])
    
    pred_real = json.dumps([{
        "reactants": [{"idt": "1"}],
        "conditions": [
            {"text": "FeCl3 (0.65 eq) DCE, 70 C, 4 h"},  # 略有不同
            {"text": "R2YYR2 (1 eq)"}  # 略有不同
        ],
        "products": [{"idt": "2"}]
    }])
    
    result8 = compute_idtvp_reward_v1("IDTVP_naive", pred_real, gold_real)
    print("\n【测试8: 实际格式示例（条件文本略有不同）】")
    print(f"  总分: {result8['score']:.4f}")
    print(f"  Soft - P/R: {result8['soft_precision']:.2f} / {result8['soft_recall']:.2f}")
    print(f"  Hybrid - P/R: {result8['hybrid_precision']:.2f} / {result8['hybrid_recall']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    
    # ====================================
    