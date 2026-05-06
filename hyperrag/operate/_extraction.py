import asyncio
import json
import re
import os
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from typing import Union, List, Dict, Set, Tuple, Optional

import numpy as np
import psutil

from ..utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
)
from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from ..prompt import GRAPH_FIELD_SEP, PROMPTS
from ..llm import openai_embedding
from ._config import DualDimensionConfig, EMB_MODEL, EMB_API_KEY, EMB_BASE_URL, normalize_entity_name, match_entity_name, strip_leading_articles


def parse_entity_array_robust(entity_array_str):
    """智能解析实体数组，处理各种引号问题

    优先使用分号作为实体分隔符（避免实体名中的逗号被误解析），
    如果没有分号则回退到逗号分隔。支持引号包裹的实体名。
    当LLM输出缺少闭合方括号时，自动补全以增强鲁棒性。

    Args:
        entity_array_str: 实体数组字符串

    Returns:
        解析后的实体列表，如果解析失败则返回None
    """
    start_bracket = entity_array_str.find('[')
    end_bracket = entity_array_str.rfind(']')

    if start_bracket == -1:
        return None

    if end_bracket == -1 or end_bracket <= start_bracket:
        array_content = entity_array_str[start_bracket + 1:]
        logger.debug(f"实体数组缺少闭合方括号，自动补全: {entity_array_str[:80]}")
    else:
        array_content = entity_array_str[start_bracket + 1:end_bracket]
    array_content = array_content.replace('" + "', '')

    has_semicolon = ';' in array_content

    entities = []
    current_entity = []
    in_quotes = False
    quote_char = None

    i = 0
    while i < len(array_content):
        char = array_content[i]

        if char == '\\' and i + 1 < len(array_content):
            current_entity.append(char)
            current_entity.append(array_content[i + 1])
            i += 2
            continue

        if char in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            else:
                current_entity.append(char)
        elif has_semicolon and char == ';' and not in_quotes:
            entity_str = ''.join(current_entity).strip()
            if entity_str:
                entities.append(entity_str)
            current_entity = []
        elif not has_semicolon and char == ',' and not in_quotes:
            entity_str = ''.join(current_entity).strip()
            if entity_str:
                entities.append(entity_str)
            current_entity = []
        elif char in ' \t\n\r' and not in_quotes:
            if current_entity:
                current_entity.append(char)
        else:
            current_entity.append(char)

        i += 1

    entity_str = ''.join(current_entity).strip()
    if entity_str:
        entities.append(entity_str)

    cleaned_entities = []
    for entity in entities:
        entity = entity.strip().strip('"\'')
        if entity:
            cleaned_entities.append(entity)

    return cleaned_entities if cleaned_entities else None


async def _batch_handle_entity_summaries(
    entity_data_list: list[tuple],
    global_config: dict,
) -> list[str]:
    """批量处理实体描述摘要，减少LLM调用次数

    Args:
        entity_data_list: [(entity_name, description), ...]
        global_config: 全局配置

    Returns:
        摘要列表
    """
    if not entity_data_list:
        return []

    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    to_summarize = []
    results = {}

    for entity_name, description in entity_data_list:
        tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
        if len(tokens) < summary_max_tokens:
            results[entity_name] = description
        else:
            use_description = decode_tokens_by_tiktoken(
                tokens[:llm_max_tokens], model_name=tiktoken_model_name
            )
            to_summarize.append((entity_name, use_description))

    if not to_summarize:
        return [results[item[0]] for item in entity_data_list]

    batch_size = global_config.get("batch_summary_size", 5)
    all_summaries = {}

    for i in range(0, len(to_summarize), batch_size):
        batch = to_summarize[i:i + batch_size]

        entities_str = ""
        for idx, (entity_name, description) in enumerate(batch):
            entities_str += f"{idx + 1}. Entity: {entity_name}\n"
            entities_str += f"   Descriptions: {description[:500]}...\n\n"

        batch_prompt = PROMPTS["batch_entity_summary"].format(entities=entities_str)

        logger.info(f"Batch summarizing {len(batch)} entities")

        try:
            response = await use_llm_func(batch_prompt, max_tokens=summary_max_tokens * len(batch))
            if response:
                for idx, (entity_name, _) in enumerate(batch):
                    patterns = [
                        rf"Entity\s*{idx + 1}[\.:\)]\s*(.+?)(?=Entity\s*{idx + 2}|\Z)",
                        rf"{idx + 1}[\.\)]\s*(.+?)(?={idx + 2}[\.\)]|\Z)",
                    ]
                    summary = None
                    for pattern in patterns:
                        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                        if match:
                            summary = match.group(1).strip()
                            break

                    if not summary:
                        summary = batch[idx][1][:500]

                    all_summaries[entity_name] = summary
            else:
                for entity_name, description in batch:
                    all_summaries[entity_name] = description[:500]
        except Exception as e:
            logger.error(f"Batch summary failed: {e}")
            for entity_name, description in batch:
                all_summaries[entity_name] = description[:500]

    results.update(all_summaries)
    return [results[item[0]] for item in entity_data_list]


async def _batch_handle_relation_summaries(
    relation_data_list: list[tuple],
    global_config: dict,
) -> list[str]:
    """批量处理关系描述摘要

    Args:
        relation_data_list: [(relation_id, description), ...]
        global_config: 全局配置

    Returns:
        [description_summary, ...]
    """
    if not relation_data_list:
        return []

    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    desc_max_tokens = global_config["relation_summary_to_max_tokens"]

    to_summarize_desc = []
    results = {}

    for item in relation_data_list:
        relation_id = item[0]
        description = item[1]
        desc_tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
        if len(desc_tokens) < desc_max_tokens:
            results[relation_id] = description
        else:
            use_description = decode_tokens_by_tiktoken(
                desc_tokens[:llm_max_tokens], model_name=tiktoken_model_name
            )
            to_summarize_desc.append((relation_id, use_description))
            results[relation_id] = None

    batch_size = global_config.get("batch_summary_size", 5)

    if to_summarize_desc:
        for i in range(0, len(to_summarize_desc), batch_size):
            batch = to_summarize_desc[i:i + batch_size]
            relationships_str = ""
            for idx, (rel_id, description) in enumerate(batch):
                relationships_str += f"{idx + 1}. Relationship {rel_id}:\n"
                relationships_str += f"   Descriptions: {description[:400]}...\n\n"

            batch_prompt = PROMPTS["batch_relation_description_summary"].format(relationships=relationships_str)
            logger.info(f"Batch summarizing {len(batch)} relation descriptions")

            try:
                response = await use_llm_func(batch_prompt, max_tokens=desc_max_tokens * len(batch))
                if response:
                    for idx, (rel_id, original_desc) in enumerate(batch):
                        patterns = [
                            rf"{idx + 1}[\.:\)]\s*(.+?)(?={idx + 2}[\.:\)]|\Z)",
                            rf"\[{idx + 1}\]\s*:\s*(.+?)(?=\[{idx + 2}\]|\Z)",
                        ]
                        summary = None
                        for pattern in patterns:
                            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                            if match:
                                summary = match.group(1).strip()
                                break

                        if not summary:
                            summary = original_desc[:400]

                        results[rel_id] = summary
            except Exception as e:
                logger.error(f"Batch relation desc summary failed: {e}")
                for rel_id, original_desc in batch:
                    if results[rel_id] is None:
                        results[rel_id] = original_desc[:400]

    for rel_id in results:
        if results[rel_id] is None:
            results[rel_id] = ""

    return [results[item[0]] for item in relation_data_list]


def build_information_layer(msgs, entities, chunk_key, max_simplex_dimension=5):
    """构建信息层：每个MSG转换为一个极大单纯形

    当MSG实体数超过维度上限时，拆分为多个重叠的单纯形而非截断，
    确保不丢失任何实体间的关系。

    Args:
        msgs: MSG列表，每个包含text, entities, completeness
        entities: 实体字典 {name: data}
        chunk_key: 文本块标识
        max_simplex_dimension: 单纯形维度硬上限

    Returns:
        信息层单纯形列表（仅包含极大单纯形）
    """
    information_layer = []
    for msg in msgs:
        entity_list = sorted(set(msg["entities"]))
        if len(entity_list) < 2:
            continue

        if len(entity_list) - 1 > max_simplex_dimension:
            max_count = max_simplex_dimension + 1
            logger.info(
                f"MSG实体数{len(entity_list)}超过维度上限{max_simplex_dimension}, "
                f"拆分为多个重叠单纯形: {entity_list}"
            )
            core = entity_list[:max_count]
            information_layer.append({
                "id": compute_mdhash_id(str(core), prefix=f"simplex-{len(core)-1}-"),
                "entities": core,
                "dimension": len(core) - 1,
                "is_maximal": True,
                "layer": "information",
                "description": msg["text"],
                "completeness": msg["completeness"],
                "source_id": chunk_key,
                "importance": max(
                    (entities.get(e, {}).get("importance", 0.5) for e in core),
                    default=0.5
                ),
            })
            remaining = entity_list[max_count:]
            OVERLAP = 2
            anchor = core[:OVERLAP]
            i = 0
            while i < len(remaining):
                batch = anchor + remaining[i:i + max_count - OVERLAP]
                if len(batch) < 2:
                    batch = anchor[:1] + remaining[i:i + 1]
                if len(batch) >= 2:
                    information_layer.append({
                        "id": compute_mdhash_id(str(batch), prefix=f"simplex-{len(batch)-1}-"),
                        "entities": batch,
                        "dimension": len(batch) - 1,
                        "is_maximal": True,
                        "layer": "information",
                        "description": msg["text"],
                        "completeness": max(0.5, msg["completeness"] - 0.1),
                        "source_id": chunk_key,
                        "importance": max(
                            (entities.get(e, {}).get("importance", 0.5) for e in batch),
                            default=0.5
                        ),
                    })
                i += max_count - OVERLAP
        else:
            simplex = {
                "id": compute_mdhash_id(str(entity_list), prefix=f"simplex-{len(entity_list)-1}-"),
                "entities": entity_list,
                "dimension": len(entity_list) - 1,
                "is_maximal": True,
                "layer": "information",
                "description": msg["text"],
                "completeness": msg["completeness"],
                "source_id": chunk_key,
                "importance": max(
                    (entities.get(e, {}).get("importance", 0.5) for e in entity_list),
                    default=0.5
                ),
            }
            information_layer.append(simplex)

    information_layer = compute_maximal_flags(information_layer)
    return information_layer


def fuzzy_merge_entities(merged_entities, all_msgs, all_relations=None):
    """基于纯算法模糊匹配合并疑似重复的实体

    采用严格的纯算法策略，不依赖任何硬编码词典，仅合并以下情况：
    1. 仅标点符号差异（如 "SAEXPLORATION HOLDINGS, INC." ↔ "SAEXPLORATION HOLDINGS INC."）

    不合并的情况（即使看起来相似）：
    - 子集关系但差异词为实义词（如 "BANK OF NEW YORK" ≠ "THE BANK OF NEW YORK"）
    - 数字/金额不同（如 "$2.8 MILLION" ≠ "$125.0 MILLION"）
    - 部分重叠但非子集关系（如 "SECURED PARTY" ≠ "SECURED CREDITOR"）

    设计原则（参考基线模型HyperGraphRAG）：
    构建阶段仅做确定性归一化（大写+下划线转空格+去多余空格），
    模糊匹配交由检索阶段的向量相似度（VDB cosine similarity）处理。
    本函数仅作为安全网，处理normalize_entity_name无法覆盖的标点差异。
    子集关系的实体合并（如停用词差异）不在构建阶段处理，
    而是通过Prompt约束LLM输出一致性 + 检索阶段VDB语义匹配来隐式解决。

    低频实体合并到高频实体中，并同步更新MSG和relation中的实体引用。
    合并后对rename_map进行传递闭包解析，防止连锁合并导致KeyError。

    Args:
        merged_entities: 已合并的实体字典 {name: data}
        all_msgs: MSG列表，需同步更新实体引用
        all_relations: 关系列表，需同步更新subject/object引用

    Returns:
        (merged_entities, all_msgs, all_relations) 去重后的实体字典、MSG列表和关系列表
    """

    def _is_punctuation_only_diff(name1, name2):
        """判断两个名称是否仅标点符号不同

        移除所有非字母数字字符后比较，纯算法无词典依赖。
        例如 "HOLDINGS, INC." 和 "HOLDINGS INC." 会被判定为相同。

        Args:
            name1: 第一个实体名称
            name2: 第二个实体名称

        Returns:
            True如果仅标点差异，False否则
        """
        s1 = ''.join(c for c in name1 if c.isalnum() or c.isspace())
        s2 = ''.join(c for c in name2 if c.isalnum() or c.isspace())
        return s1.upper().strip() == s2.upper().strip()

    def _has_number_diff(name1, name2):
        """判断两个名称是否包含不同的数字

        防止数字不同的实体被错误合并（如金额、日期、条款号）。
        纯正则匹配，无词典依赖。

        Args:
            name1: 第一个实体名称
            name2: 第二个实体名称

        Returns:
            True如果包含不同数字，False否则
        """
        nums1 = set(re.findall(r'\d+\.?\d*', name1))
        nums2 = set(re.findall(r'\d+\.?\d*', name2))
        if nums1 and nums2 and nums1 != nums2:
            return True
        return False

    def _entity_similarity(name1, name2):
        """计算两个实体名称的相似度

        匹配策略（按优先级）：
        1. 仅标点差异 → 1.0（确定相同）
        2. 去冠词后完全相同 → 0.95（高度可信）
        3. 去冠词后包含关系且词数差≤1 → 0.8（较可信）
        数字不同的实体直接返回0.0，防止金额/日期等误合并。

        Args:
            name1: 第一个实体名称
            name2: 第二个实体名称

        Returns:
            相似度值，1.0表示确定相同，0.0表示不同
        """
        if _has_number_diff(name1, name2):
            return 0.0

        if _is_punctuation_only_diff(name1, name2):
            return 1.0

        norm1 = normalize_entity_name(name1)
        norm2 = normalize_entity_name(name2)
        stripped1 = strip_leading_articles(norm1)
        stripped2 = strip_leading_articles(norm2)

        if stripped1 and stripped2 and stripped1 == stripped2:
            return 0.95

        if stripped1 and stripped2:
            words1 = stripped1.split()
            words2 = stripped2.split()
            if len(words1) >= 2 and len(words2) >= 2:
                if stripped1 in stripped2 or stripped2 in stripped1:
                    shorter_len = min(len(words1), len(words2))
                    longer_len = max(len(words1), len(words2))
                    if longer_len - shorter_len <= 1:
                        return 0.8

        return 0.0

    entity_names = list(merged_entities.keys())
    n = len(entity_names)
    if n < 2:
        return merged_entities, all_msgs, all_relations

    rename_map = {}
    checked = set()

    for i in range(n):
        name_i = entity_names[i]
        if name_i in rename_map or name_i in checked:
            continue
        for j in range(i + 1, n):
            name_j = entity_names[j]
            if name_j in rename_map or name_j in checked:
                continue

            sim = _entity_similarity(name_i, name_j)
            if sim < 0.45:
                continue

            freq_i = merged_entities[name_i].get("frequency", 0)
            freq_j = merged_entities[name_j].get("frequency", 0)

            if freq_i >= freq_j:
                canonical, duplicate = name_i, name_j
            else:
                canonical, duplicate = name_j, name_i

            rename_map[duplicate] = canonical
            checked.add(duplicate)
            logger.info(
                f"模糊匹配去重: '{duplicate}' -> '{canonical}'"
                f" (sim={sim:.3f}, freq={freq_j}->{freq_i})"
            )

    if not rename_map:
        return merged_entities, all_msgs, all_relations

    resolved_rename = {}
    for duplicate, canonical in rename_map.items():
        target = canonical
        while target in rename_map:
            target = rename_map[target]
        resolved_rename[duplicate] = target

    for duplicate, canonical in resolved_rename.items():
        if canonical not in merged_entities:
            logger.warning(f"模糊匹配跳过: canonical '{canonical}' 不存在，'{duplicate}' 保留")
            continue
        src = merged_entities[duplicate]
        dst = merged_entities[canonical]

        src_descs = set(dst["description"].split(GRAPH_FIELD_SEP)) if dst["description"] else set()
        new_descs = set(src["description"].split(GRAPH_FIELD_SEP)) if src["description"] else set()
        combined = src_descs | new_descs
        dst["description"] = GRAPH_FIELD_SEP.join(sorted(combined))

        src_sources = set(dst["source_id"].split(GRAPH_FIELD_SEP)) if dst["source_id"] else set()
        new_sources = set(src["source_id"].split(GRAPH_FIELD_SEP)) if src["source_id"] else set()
        dst["source_id"] = GRAPH_FIELD_SEP.join(sorted(src_sources | new_sources))

        dst["frequency"] = dst.get("frequency", 0) + src.get("frequency", 0)

        imp_dst = dst.get("importance", 0.5)
        imp_src = src.get("importance", 0.5)
        total_freq = dst["frequency"]
        dst["importance"] = (imp_dst * (total_freq - src.get("frequency", 1)) + imp_src * src.get("frequency", 1)) / total_freq

        del merged_entities[duplicate]

    for msg in all_msgs:
        msg["entities"] = list(dict.fromkeys(resolved_rename.get(e, e) for e in msg["entities"]))

    if all_relations:
        for rel in all_relations:
            rel["subject"] = resolved_rename.get(rel["subject"], rel["subject"])
            rel["object"] = resolved_rename.get(rel["object"], rel["object"])

    logger.info(f"模糊匹配去重完成: 合并{len(resolved_rename)}对重复实体")

    return merged_entities, all_msgs, all_relations


def repair_split_entities(merged_entities, all_msgs, all_relations=None):
    """修复因逗号分隔符误解析导致的实体拆分问题

    当LLM使用逗号作为实体列表分隔符时，实体名内部的逗号会导致拆分错误。
    例如 "HENRY M. MILLER, JR." 被拆分为 "HENRY M. MILLER" 和 "JR."，
    而 "MIZUHO BANK, LTD." 被拆分为 "MIZUHO BANK" 和 "LTD."。

    修复策略：对于MSG中不存在于实体数据库的实体，尝试将相邻的缺失实体
    用 ", " 拼接后检查是否匹配已知实体。纯算法方法，不依赖任何词典。

    Args:
        merged_entities: 已合并的实体字典 {name: data}
        all_msgs: MSG列表
        all_relations: 关系列表，需同步更新subject/object引用

    Returns:
        (merged_entities, all_msgs, all_relations) 修复后的实体字典、MSG列表和关系列表
    """
    entity_names_set = set(merged_entities.keys())
    total_repaired = 0
    rename_map = {}

    for msg in all_msgs:
        entities = msg["entities"]
        if not entities:
            continue

        missing_indices = [i for i, e in enumerate(entities) if e not in entity_names_set]
        if not missing_indices:
            continue

        repaired = True
        while repaired:
            repaired = False
            current_entities = msg["entities"]
            current_missing = [i for i, e in enumerate(current_entities) if e not in entity_names_set]

            for idx in current_missing:
                if idx + 1 < len(current_entities):
                    candidate = current_entities[idx] + ", " + current_entities[idx + 1]
                    if candidate in entity_names_set:
                        old_name = current_entities[idx]
                        rename_map[old_name] = candidate
                        current_entities[idx] = candidate
                        current_entities.pop(idx + 1)
                        repaired = True
                        total_repaired += 1
                        break

                if idx > 0:
                    candidate = current_entities[idx - 1] + ", " + current_entities[idx]
                    if candidate in entity_names_set:
                        old_name = current_entities[idx]
                        rename_map[old_name] = candidate
                        current_entities[idx - 1] = candidate
                        current_entities.pop(idx)
                        repaired = True
                        total_repaired += 1
                        break

        msg["entities"] = list(dict.fromkeys(current_entities))

    if all_relations and rename_map:
        for rel in all_relations:
            rel["subject"] = rename_map.get(rel["subject"], rel["subject"])
            rel["object"] = rename_map.get(rel["object"], rel["object"])

    if total_repaired > 0:
        logger.info(f"实体拆分修复完成: 修复{total_repaired}处逗号拆分错误")

    return merged_entities, all_msgs, all_relations


def build_entity_coboundary(entities, information_layer):
    """构建实体的coboundary：该实体所属的MSG列表

    当MSG引用了不存在于实体字典中的实体时，先尝试通过
    match_entity_name 进行多策略模糊匹配（去冠词、包含关系等），
    匹配失败才自动补建最小实体记录，避免因LLM命名不一致
    导致重复创建实体。

    Args:
        entities: 实体字典
        information_layer: 信息层MSG列表

    Returns:
        更新后的实体字典
    """
    for name in entities:
        entities[name]["coboundary"] = []

    normalized_entity_map = {normalize_entity_name(k): k for k in entities}
    entity_name_set = set(entities.keys())

    auto_created_count = 0
    for msg in information_layer:
        resolved_entities = []
        for entity_name in msg["entities"]:
            matched = match_entity_name(entity_name, entity_name_set, normalized_entity_map)
            if matched is not None:
                entities[matched]["coboundary"].append(msg["id"])
                resolved_entities.append(matched)
                if matched != entity_name:
                    logger.debug(
                        f"MSG {msg['id']} 实体 '{entity_name}' "
                        f"通过模糊匹配到 '{matched}'"
                    )
            else:
                entities[entity_name] = {
                    "entity_name": entity_name,
                    "entity_type": "concept",
                    "description": msg.get("description", ""),
                    "source_id": msg.get("source_id", ""),
                    "additional_properties": "",
                    "frequency": 1,
                    "importance": 0.3,
                    "coboundary": [msg["id"]],
                }
                entity_name_set.add(entity_name)
                normalized_entity_map[normalize_entity_name(entity_name)] = entity_name
                resolved_entities.append(entity_name)
                auto_created_count += 1
                logger.info(
                    f"MSG {msg['id']} 引用了不存在的实体 '{entity_name}'，"
                    f"已自动补建实体记录"
                )

        msg["entities"] = resolved_entities
        msg["dimension"] = len(msg["entities"]) - 1
        if len(msg["entities"]) < 2:
            logger.warning(f"MSG {msg['id']} 过滤后实体数<2，需关注")

    if auto_created_count > 0:
        logger.info(f"自动补建 {auto_created_count} 个缺失实体记录")

    return entities


def build_msg_boundary(information_layer, entities):
    """构建MSG的boundary：该MSG包含的实体ID列表

    Args:
        information_layer: 信息层MSG列表
        entities: 实体字典

    Returns:
        更新后的信息层列表
    """
    for msg in information_layer:
        msg["boundary"] = []
        for entity_name in msg["entities"]:
            entity_id = compute_mdhash_id(entity_name, prefix="simplex-0-")
            msg["boundary"].append(entity_id)

    return information_layer


def compute_maximal_flags(information_layer):
    """计算每个单纯形的is_maximal标记

    如果一个单纯形的实体集合是另一个单纯形实体集合的真子集，
    则该单纯形不是极大单纯形。仅保留真正的极大单纯形用于后续处理，
    非极大单纯形的描述信息合并到包含它的极大单纯形中。

    特殊规则：一维单纯形(dimension=1, 2个实体)即使被更高维单纯形包含也不移除，
    因为低维二元关系携带独立的语义信息（如特定的谓词或上下文），
    在检索时提供更精确的匹配粒度。

    Args:
        information_layer: 信息层单纯形列表

    Returns:
        仅包含极大单纯形的列表（已更新is_maximal标记）
    """
    if not information_layer:
        return information_layer

    entity_sets = [frozenset(s["entities"]) for s in information_layer]
    dims = [s.get("dimension", len(s["entities"]) - 1) for s in information_layer]
    n = len(information_layer)
    is_maximal = [True] * n

    for i in range(n):
        if not is_maximal[i]:
            continue
        if dims[i] <= 1:
            continue
        for j in range(n):
            if i == j or not is_maximal[j]:
                continue
            if entity_sets[i] < entity_sets[j]:
                is_maximal[i] = False
                break

    maximal_layer = []
    for i, simplex in enumerate(information_layer):
        simplex["is_maximal"] = is_maximal[i]
        if is_maximal[i]:
            maximal_layer.append(simplex)
        else:
            logger.debug(
                f"单纯形 {simplex['id']} 非极大"
                f"(entities={simplex['entities']})，已移除"
            )

    removed = n - len(maximal_layer)
    if removed > 0:
        logger.info(
            f"极大单纯形过滤: {len(maximal_layer)}/{n}"
            f" (移除{removed}个非极大单纯形)"
        )

    return maximal_layer


def build_bipartite_laplacian(information_layer, entities):
    """计算二部图简化Laplacian矩阵

    定义直接边界映射 ∂: C_max → C₀，从极大单纯形直接映射到0-单纯形。
    L_entity = BᵀB（实体-实体共现拓扑）
    L_msg = BBᵀ（MSG-MSG共享实体拓扑）

    Args:
        information_layer: 信息层MSG列表
        entities: 实体字典

    Returns:
        (L_entity, L_msg, entity_index, msg_index) 元组
    """
    entity_names = sorted(entities.keys())
    msg_ids = [msg["id"] for msg in information_layer]
    entity_index = {name: i for i, name in enumerate(entity_names)}
    msg_index = {mid: i for i, mid in enumerate(msg_ids)}

    n_entities = len(entity_names)
    n_msgs = len(msg_ids)

    B = np.zeros((n_msgs, n_entities), dtype=np.float32)
    for msg in information_layer:
        i = msg_index[msg["id"]]
        for entity_name in msg["entities"]:
            if entity_name in entity_index:
                j = entity_index[entity_name]
                B[i, j] = 1.0

    L_entity = B.T @ B
    L_msg = B @ B.T

    return L_entity, L_msg, entity_index, msg_index


async def _handle_single_msg_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """解析MSG记录（内嵌实体元数据），同时提取实体记录

    新格式: ("mcss"|<description>|[<e1#type1#imp1>;<e2#type2#imp2>]|<completeness>)
    每个实体条目用#分隔子字段: entity_name#entity_type#importance
    实体的original_text直接使用MSG的description作为上下文

    使用从后往前解析策略，避免描述中的|导致错误分割。
    实体数>6时拆分为多个重叠MSG。

    Args:
        record_attributes: 解析后的属性列表
        chunk_key: 文本块标识

    Returns:
        (msg_list, entity_list) 元组；解析失败返回None
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"mcss"':
        return None

    completeness = 0.75
    last_val = clean_str(record_attributes[-1])
    if is_float_regex(last_val):
        completeness = float(last_val)
        entity_list_str = record_attributes[-2]
    else:
        entity_list_str = record_attributes[-1]

    if is_float_regex(last_val):
        description_parts = record_attributes[1:-2]
    else:
        description_parts = record_attributes[1:-1]
    description = clean_str(" | ".join(description_parts))

    entity_entries = parse_entity_array_robust(entity_list_str)
    if entity_entries is None or len(entity_entries) < 1:
        logger.warning(f"MSG实体列表解析失败: {entity_list_str}")
        return None

    parsed_entities = []
    extracted_entity_records = []
    for entry in entity_entries:
        entry = clean_str(entry)
        if "#" in entry:
            parts = entry.split("#", 2)
            ent_name = normalize_entity_name(clean_str(parts[0]))
            ent_type = clean_str(parts[1]).strip().lower() if len(parts) > 1 else "concept"
            ent_imp_str = clean_str(parts[2]).strip() if len(parts) > 2 else "0.5"
            ent_imp = float(ent_imp_str) if is_float_regex(ent_imp_str) else 0.5
        else:
            ent_name = normalize_entity_name(entry)
            ent_type = "concept"
            ent_imp = 0.5

        if not ent_name.strip() or ent_name.upper() in ("NULL", "NONE", "N/A", "NAN", "UNDEFINED"):
            continue
        MAX_ENTITY_NAME_LEN = 80
        if len(ent_name) > MAX_ENTITY_NAME_LEN:
            ent_name = ent_name[:MAX_ENTITY_NAME_LEN].strip()

        parsed_entities.append(ent_name)
        extracted_entity_records.append(dict(
            type="entity",
            entity_name=ent_name,
            entity_type=ent_type,
            description=description,
            source_id=chunk_key,
            importance=ent_imp,
        ))

    if len(parsed_entities) < 2:
        logger.info(f"MSG实体数<2({len(parsed_entities)}个), 降级为entity记录: {parsed_entities}")
        return ([], extracted_entity_records)

    MAX_MSG_ENTITY_COUNT = 6

    if len(parsed_entities) <= MAX_MSG_ENTITY_COUNT:
        return ([dict(
            type="mcss",
            text=description,
            entities=parsed_entities,
            completeness=completeness,
            source_id=chunk_key,
        )], extracted_entity_records)

    logger.info(
        f"MSG实体数>{MAX_MSG_ENTITY_COUNT}({len(parsed_entities)}个), "
        f"拆分为多个重叠MSG: {parsed_entities}"
    )

    result_msgs = []
    core_entities = parsed_entities[:MAX_MSG_ENTITY_COUNT]
    result_msgs.append(dict(
        type="mcss",
        text=description,
        entities=core_entities,
        completeness=completeness,
        source_id=chunk_key,
    ))

    remaining = parsed_entities[MAX_MSG_ENTITY_COUNT:]
    OVERLAP_SIZE = 2
    overlap_anchor = core_entities[:OVERLAP_SIZE]

    i = 0
    while i < len(remaining):
        batch = overlap_anchor + remaining[i:i + MAX_MSG_ENTITY_COUNT - OVERLAP_SIZE]
        if len(batch) < 2:
            batch = overlap_anchor[:1] + remaining[i:i + 1]
        if len(batch) >= 2:
            result_msgs.append(dict(
                type="mcss",
                text=description,
                entities=batch,
                completeness=max(0.5, completeness - 0.1),
                source_id=chunk_key,
            ))
        i += MAX_MSG_ENTITY_COUNT - OVERLAP_SIZE

    return (result_msgs, extracted_entity_records)


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """解析entity记录

    格式: ("entity"|<name>|<type>|<original_text>|<importance>)
    使用从后往前解析策略，避免original_text中的|导致错误分割

    Args:
        record_attributes: 解析后的属性列表
        chunk_key: 文本块标识

    Returns:
        实体字典或None
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    entity_name = normalize_entity_name(clean_str(record_attributes[1]))
    if not entity_name.strip() or entity_name.upper() in ("NULL", "NONE", "N/A", "NAN", "UNDEFINED"):
        return None

    # 从后往前解析：最后一段可能是importance（浮点数）
    importance = 0.1
    last_val = clean_str(record_attributes[-1])
    if is_float_regex(last_val):
        importance = float(last_val)
        entity_type = clean_str(record_attributes[2]).strip()
        # 中间所有内容拼接为描述
        description_parts = record_attributes[3:-1]
        description = clean_str(" | ".join(description_parts)) if description_parts else ""
    else:
        entity_type = clean_str(record_attributes[2]).strip()
        description_parts = record_attributes[3:]
        description = clean_str(" | ".join(description_parts)) if description_parts else ""

    source_id = chunk_key

    return dict(
        type="entity",
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        additional_properties="",
        importance=importance,
    )


async def _handle_single_relation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """解析二元关系记录

    格式: ("relation"|<subject>|<predicate>|<object>|<original_text>|<importance>)
    subject/predicate/object为短文本不含|，original_text可能含|
    使用从后往前解析策略，避免original_text中的|导致错误分割

    Args:
        record_attributes: 解析后的属性列表
        chunk_key: 文本块标识

    Returns:
        关系字典或None
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relation"':
        return None

    subject = normalize_entity_name(clean_str(record_attributes[1]))
    predicate = clean_str(record_attributes[2]).strip().lower()
    obj = normalize_entity_name(clean_str(record_attributes[3]))

    if not subject.strip() or not predicate.strip() or not obj.strip():
        return None
    if subject.upper() in ("NULL", "NONE", "N/A", "NAN", "UNDEFINED"):
        return None
    if obj.upper() in ("NULL", "NONE", "N/A", "NAN", "UNDEFINED"):
        return None

    importance = 0.5
    last_val = clean_str(record_attributes[-1])
    if is_float_regex(last_val):
        importance = float(last_val)
        description_parts = record_attributes[4:-1]
    else:
        description_parts = record_attributes[4:]
    description = clean_str(" | ".join(description_parts)) if description_parts else ""

    return dict(
        type="relation",
        subject=subject,
        predicate=predicate,
        object=obj,
        description=description,
        source_id=chunk_key,
        importance=importance,
    )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    simplex_storage=None,
) -> None:
    """MSG提取主函数

    流程：
    1. 文本分块 → chunk_list
    2. 并发LLM调用 → 每个chunk独立提取MSG+entity
    3. 解析LLM输出: "mcss"记录→MSG, "entity"记录→实体
    4. 跨chunk去重: 同名实体→描述去重拼接，类型多数投票；相同实体列表的MSG→合并描述
    5. 构建信息层（MSG→极大单纯形）
    6. 构建二部图关联（实体↔MSG）
    7. 计算简化Laplacian（L_entity, L_msg）
    8. 计算嵌入向量
    9. 存储到entities_vdb + relationships_vdb + SimplexStorage
    """
    start_time = time.time()
    begin_time = datetime.now()
    logger.info("Starting extract_entities function (MSG mode)")

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    cpu_count = os.cpu_count()
    memory_available = psutil.virtual_memory().available / (1024**3)
    resource_limit = min(cpu_count * 2, int(memory_available / 2), 32)
    config_limit = global_config.get("max_parallel_chunks", 10)
    max_concurrency = min(config_limit, resource_limit)
    logger.info(
        f"Setting max concurrency to {max_concurrency} "
        f"(config={config_limit}, resource_limit={resource_limit})"
    )

    def preprocess_chunks(chunks):
        processed = []
        processed_contents = set()
        for chunk_key, chunk in chunks.items():
            content = chunk["content"]
            if content in processed_contents:
                logger.debug(f"Skipping duplicate chunk: {chunk_key}")
                continue
            processed_contents.add(content)
            processed.append((chunk_key, chunk))
        return processed

    ordered_chunks = preprocess_chunks(chunks)
    logger.info(f"Preprocessed {len(ordered_chunks)} chunks for processing")

    entity_extract_prompt = PROMPTS["entity_extraction"]
    example_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    )
    example_prompt = PROMPTS["entity_extraction_examples"][0]
    example_str = example_prompt.format(**example_base)

    context_base = dict(
        language=PROMPTS["DEFAULT_LANGUAGE"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=example_str,
    )
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_msgs = 0
    already_relations = 0

    processed_chunks = set()
    failed_chunks = set()

    llm_cache = {}
    _pending_llm_calls = {}

    async def cached_llm_call(prompt, history_messages=None, skip_cache=False):
        history_key = str(history_messages) if history_messages else "none"
        cache_key = f"{hash(prompt)}_{hash(history_key)}"

        if not skip_cache and cache_key in llm_cache:
            logger.info(f"Using cached LLM result for prompt length: {len(prompt)}")
            return llm_cache[cache_key]

        if not skip_cache and cache_key in _pending_llm_calls:
            logger.info(f"Waiting for in-flight LLM call (prompt length: {len(prompt)})")
            return await _pending_llm_calls[cache_key]

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if cache_key not in _pending_llm_calls:
            _pending_llm_calls[cache_key] = future

        try:
            logger.info(f"New LLM call for prompt length: {len(prompt)} and history length: {len(history_messages) if history_messages else 0}")

            last_error = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = await asyncio.wait_for(
                        use_llm_func(prompt, history_messages=history_messages),
                        timeout=180
                    )
                    logger.info(f"LLM call completed successfully, caching result")
                    llm_cache[cache_key] = result
                    if not future.done():
                        future.set_result(result)
                    return result
                except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                    last_error = e
                    logger.warning(f"LLM call timed out or cancelled (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        if not future.done():
                            future.set_exception(e)
                        raise
                    await asyncio.sleep(min(2 ** attempt + 1, 30))
                except Exception as e:
                    last_error = e
                    logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        if not future.done():
                            future.set_exception(e)
                        raise
                    await asyncio.sleep(min(2 ** attempt + 1, 30))
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            _pending_llm_calls.pop(cache_key, None)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_msgs, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        if chunk_key in processed_chunks:
            logger.info(f"Skipping already processed chunk: {chunk_key}")
            return None

        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)

        try:
            logger.info(f"Processing chunk {chunk_key}...")
            logger.info(f"Prompt length: {len(hint_prompt)} chars")
            final_result = await cached_llm_call(hint_prompt)
            if final_result is None:
                logger.warning(f"LLM returned None for chunk {chunk_key}")
                failed_chunks.add(chunk_key)
                return None

            processed_chunks.add(chunk_key)

            logger.info(f"LLM returned {len(final_result)} chars for chunk {chunk_key}")

            try:
                logger.debug(f"LLM raw output for chunk {chunk_key}: {final_result[:500]}...")
            except UnicodeEncodeError:
                logger.debug(f"LLM raw output for chunk {chunk_key}: {final_result[:500].encode('gbk', errors='replace').decode('gbk')}...")

            import re
            lower_result = final_result.lower()
            if re.search(r'\berror:\b', lower_result):
                try:
                    logger.error(f"LLM returned error: {final_result}")
                except UnicodeEncodeError:
                    logger.error(f"LLM returned error: {final_result.encode('gbk', errors='replace').decode('gbk')}")
                return None

        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.error(f"LLM call timed out or cancelled for chunk {chunk_key}: {e}")
            failed_chunks.add(chunk_key)
            return None
        except Exception as e:
            logger.error(f"Error calling LLM for chunk {chunk_key}: {e}")
            import traceback
            traceback.print_exc()
            failed_chunks.add(chunk_key)
            return None

        if entity_extract_max_gleaning > 0:
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

            content_length = len(content)
            max_gleaning = min(entity_extract_max_gleaning, 2 if content_length < 1000 else entity_extract_max_gleaning)

            for now_glean_index in range(max_gleaning):
                logger.info(f"Calling LLM for chunk {chunk_key} (gleaning round {now_glean_index + 1})...")
                glean_result = await cached_llm_call(continue_prompt, history)
                if glean_result is None:
                    break

                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await cached_llm_call(
                    if_loop_prompt, history
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        logger.debug(f"LLM raw output length: {len(final_result)}, records count: {len(records)}")

        unique_records = []
        seen_records = set()
        for record in records:
            if record.strip():
                record_str = record.strip()
                if record_str not in seen_records:
                    seen_records.add(record_str)
                    unique_records.append(record)

        if not unique_records:
            logger.warning(f"LLM returned empty extraction for chunk {chunk_key}, retrying once...")
            try:
                retry_prompt = hint_prompt + "\n\nPlease carefully re-examine the text and extract all MSG records and entities. If the text contains any facts, actions, or relationships involving multiple entities, output them as MSG records."
                retry_result = await cached_llm_call(retry_prompt, skip_cache=True)
                if retry_result and len(retry_result.strip()) > 10:
                    final_result = retry_result
                    records = split_string_by_multi_markers(
                        final_result,
                        [context_base["record_delimiter"], context_base["completion_delimiter"]],
                    )
                    for record in records:
                        if record.strip():
                            record_str = record.strip()
                            if record_str not in seen_records:
                                seen_records.add(record_str)
                                unique_records.append(record)
                    logger.info(f"Retry extracted {len(unique_records)} records for chunk {chunk_key}")
            except Exception as e:
                logger.warning(f"Retry extraction failed for chunk {chunk_key}: {e}")

        local_msgs = []
        local_entities = defaultdict(list)
        local_relations = []

        for record in unique_records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = re.split(r'\s*\|\s*', record)

            msg_result = await _handle_single_msg_extraction(record_attributes, chunk_key)
            if msg_result is not None:
                msg_list, entity_records = msg_result
                local_msgs.extend(msg_list)
                for ent in entity_records:
                    local_entities[ent["entity_name"]].append(ent)
                continue

            relation_data = await _handle_single_relation_extraction(record_attributes, chunk_key)
            if relation_data is not None:
                local_relations.append(relation_data)
                continue

            entity_data = await _handle_single_entity_extraction(record_attributes, chunk_key)
            if entity_data is not None:
                entity_key = entity_data["entity_name"]
                local_entities[entity_key].append(entity_data)
                continue

        # 预填充relation中引用的实体：参照HyperGraphRAG设计，
        # 将relation中出现的实体也纳入local_entities，避免后续匹配时
        # 因LLM未在MSG中记录这些实体而触发自动补建。
        # HyperGraphRAG中每个实体都有独立的("entity"|name|type|desc|score)记录，
        # 此处等效处理：从relation的subject/object创建实体记录。
        if local_relations:
            entity_name_set = set(local_entities.keys())
            prefill_count = 0
            for rel in local_relations:
                for field in ("subject", "object"):
                    ename = rel[field]
                    if ename and ename not in entity_name_set:
                        norm = normalize_entity_name(ename)
                        # 检查归一化后是否已有匹配实体
                        found = False
                        for existing_name in entity_name_set:
                            if normalize_entity_name(existing_name) == norm:
                                rel[field] = existing_name
                                found = True
                                break
                        if not found:
                            # 使用relation的description作为实体描述上下文
                            rel_desc = rel.get("description", "")
                            local_entities[ename].append(dict(
                                type="entity",
                                entity_name=ename,
                                entity_type="concept",
                                description=rel_desc,
                                source_id=chunk_key,
                                importance=0.3,
                            ))
                            entity_name_set.add(ename)
                            prefill_count += 1
            if prefill_count > 0:
                logger.debug(f"Chunk内预填充 {prefill_count} 个relation引用的实体到local_entities")

        if local_msgs:
            entity_name_set = set(local_entities.keys())
            normalized_entity_map = {normalize_entity_name(k): k for k in entity_name_set}
            auto_created = 0
            for msg in local_msgs:
                resolved = []
                for ename in msg["entities"]:
                    matched = match_entity_name(ename, entity_name_set, normalized_entity_map)
                    if matched is not None:
                        resolved.append(matched)
                    else:
                        resolved.append(ename)
                        if ename not in local_entities:
                            local_entities[ename].append(dict(
                                type="entity",
                                entity_name=ename,
                                entity_type="concept",
                                description=msg.get("text", ""),
                                source_id=chunk_key,
                                importance=0.3,
                            ))
                            entity_name_set.add(ename)
                            norm = normalize_entity_name(ename)
                            normalized_entity_map[norm] = ename
                            auto_created += 1
                            logger.debug(
                                f"MSG实体 '{ename}' 未匹配到已有实体，"
                                f"候选集: {list(entity_name_set)[:10]}..."
                            )
                msg["entities"] = resolved
            if auto_created > 0:
                logger.info(f"Chunk内自动补建 {auto_created} 个MSG引用但未提取的实体")

        if local_relations:
            entity_name_set = set(local_entities.keys())
            normalized_entity_map = {normalize_entity_name(k): k for k in entity_name_set}
            rel_auto_created = 0
            for rel in local_relations:
                for field in ("subject", "object"):
                    ename = rel[field]
                    matched = match_entity_name(ename, entity_name_set, normalized_entity_map)
                    if matched is not None:
                        if matched != ename:
                            rel[field] = matched
                    else:
                        local_entities[ename].append(dict(
                            type="entity",
                            entity_name=ename,
                            entity_type="concept",
                            description=rel.get("description", ""),
                            source_id=chunk_key,
                            importance=0.3,
                        ))
                        entity_name_set.add(ename)
                        norm = normalize_entity_name(ename)
                        normalized_entity_map[norm] = ename
                        rel_auto_created += 1
                        logger.debug(
                            f"Relation实体 '{ename}' 未匹配到已有实体，"
                            f"候选集: {list(entity_name_set)[:10]}..."
                        )
            if rel_auto_created > 0:
                logger.info(f"Chunk内自动补建 {rel_auto_created} 个relation引用但未提取的实体")

        logger.info(f"Extracted {len(local_msgs)} MSGs, {len(local_entities)} entities, {len(local_relations)} relations for chunk {chunk_key}")

        already_processed += 1
        already_entities += len(local_entities)
        already_msgs += len(local_msgs)
        already_relations += len(local_relations)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]

        current_time = datetime.now()
        elapsed = current_time - begin_time
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        percent = (already_processed / max(len(ordered_chunks), 1)) * 100
        bar_length = int(50 * already_processed // max(len(ordered_chunks), 1))
        bar = '█' * bar_length + '-' * (50 - bar_length)
        sys.stdout.write(
            f'\n\r|{bar}| {percent:.2f}% |{hours:02}:{minutes:02}:{seconds:02}| {now_ticks} Processed, {already_entities} entities, {already_msgs} MSGs, {already_relations} relations \n'
        )
        sys.stdout.flush()

        return local_msgs, dict(local_entities), local_relations

    async def worker(queue, results):
        while not queue.empty():
            chunk = await queue.get()
            try:
                result = await _process_single_content(chunk)
                results.append(result)
            except asyncio.CancelledError:
                logger.warning("Worker cancelled during chunk processing")
                queue.task_done()
                raise
            except Exception as e:
                logger.error(f"Worker error during chunk processing: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results.append(None)
            finally:
                queue.task_done()

    try:
        queue = asyncio.Queue()
        for chunk in ordered_chunks:
            await queue.put(chunk)

        results = []
        tasks = []
        for _ in range(max_concurrency):
            task = asyncio.create_task(worker(queue, results))
            tasks.append(task)

        await queue.join()
        for task in tasks:
            task.cancel()
    except Exception as e:
        logger.error(f"Error during async processing: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 合并所有chunk的提取结果
    all_msgs = []
    all_entities = defaultdict(list)
    all_relations = []
    entity_frequency = Counter()
    entity_sources = defaultdict(set)
    msg_sources = defaultdict(set)

    for result in results:
        if result is None:
            continue

        local_msgs, local_entities, local_relations = result

        if local_msgs:
            for msg in local_msgs:
                all_msgs.append(msg)
                source_id = msg.get("source_id", "")
                if source_id:
                    entity_key = tuple(sorted(msg["entities"]))
                    msg_sources[entity_key].add(source_id)

        if local_entities:
            for entity_name, entity_list in local_entities.items():
                for entity_data in entity_list:
                    all_entities[entity_name].append(entity_data)
                    entity_frequency[entity_name] += 1
                    source_id = entity_data.get("source_id", "")
                    if source_id:
                        entity_sources[entity_name].add(source_id)

        if local_relations:
            all_relations.extend(local_relations)

    logger.info(f"Total extracted: {len(all_msgs)} MSGs, {len(all_entities)} unique entities, {len(all_relations)} relations")

    # 跨chunk去重：实体合并（仅合并当前批次新提取的数据，与已有数据的合并由upsert_simplex处理）
    merged_entities = {}
    for entity_name, entity_list in all_entities.items():
        all_entity_types = [dp["entity_type"] for dp in entity_list]
        entity_type = sorted(
            Counter(all_entity_types).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0] if all_entity_types else "organization"

        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in entity_list]))
        )

        additional_properties = GRAPH_FIELD_SEP.join(
            sorted(set(
                [dp.get("additional_properties", "") for dp in entity_list]
            ))
        )

        source_ids = entity_sources.get(entity_name, set())
        source_id = GRAPH_FIELD_SEP.join(
            set([dp["source_id"] for dp in entity_list] + list(source_ids))
        )

        importance_values = [dp.get("importance", 0.5) for dp in entity_list]
        avg_importance = sum(importance_values) / len(importance_values) if importance_values else 0.5

        frequency = entity_frequency.get(entity_name, 0)

        merged_entities[entity_name] = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_id": source_id,
            "additional_properties": additional_properties,
            "frequency": frequency,
            "importance": avg_importance,
            "coboundary": [],
        }

    # 模糊匹配去重：合并拼写变体实体（如UNCREDITED AGENT → INTERCREDITOR AGENT）
    merged_entities, all_msgs, all_relations = fuzzy_merge_entities(merged_entities, all_msgs, all_relations)

    # 修复逗号拆分错误：LLM使用逗号分隔时实体名内逗号导致的误拆分
    merged_entities, all_msgs, all_relations = repair_split_entities(merged_entities, all_msgs, all_relations)

    # 跨chunk去重：MSG合并
    merged_msgs = []
    msg_dedup = {}
    for msg in all_msgs:
        entity_key = tuple(sorted(msg["entities"]))
        if entity_key in msg_dedup:
            existing = msg_dedup[entity_key]
            existing_descs = set()
            if existing["text"]:
                existing_descs.add(existing["text"])
            if msg["text"]:
                existing_descs.add(msg["text"])
            existing["text"] = GRAPH_FIELD_SEP.join(sorted(existing_descs))
            existing["completeness"] = max(existing["completeness"], msg["completeness"])
            existing["importance"] = max(
                existing.get("importance", 0.5),
                msg.get("importance", 0.5)
            )
            source_ids = set()
            if existing.get("source_id"):
                source_ids.update(split_string_by_multi_markers(existing["source_id"], [GRAPH_FIELD_SEP]))
            if msg.get("source_id"):
                source_ids.add(msg["source_id"])
            existing["source_id"] = GRAPH_FIELD_SEP.join(source_ids)
        else:
            msg_dedup[entity_key] = dict(msg)

    merged_msgs = list(msg_dedup.values())
    logger.info(f"After dedup: {len(merged_msgs)} MSGs, {len(merged_entities)} entities")

    # 跨chunk去重：二元关系合并
    merged_relations = {}
    for rel in all_relations:
        rel_key = (rel["subject"], rel["predicate"], rel["object"])
        if rel_key in merged_relations:
            existing = merged_relations[rel_key]
            existing_descs = set()
            if existing["description"]:
                existing_descs.add(existing["description"])
            if rel["description"]:
                existing_descs.add(rel["description"])
            existing["description"] = GRAPH_FIELD_SEP.join(sorted(existing_descs))
            existing["importance"] = max(existing["importance"], rel["importance"])
            source_ids = set()
            if existing.get("source_id"):
                source_ids.update(split_string_by_multi_markers(existing["source_id"], [GRAPH_FIELD_SEP]))
            if rel.get("source_id"):
                source_ids.add(rel["source_id"])
            existing["source_id"] = GRAPH_FIELD_SEP.join(source_ids)
        else:
            merged_relations[rel_key] = dict(rel)

    logger.info(f"After dedup: {len(merged_relations)} unique relations")

    # 批量摘要
    entity_summary_data = [(name, data["description"]) for name, data in merged_entities.items()]
    entity_ap_data = [(name, data["additional_properties"]) for name, data in merged_entities.items()]

    desc_summaries = await _batch_handle_entity_summaries(entity_summary_data, global_config)
    ap_summaries = await _batch_handle_entity_summaries(entity_ap_data, global_config)

    for idx, (entity_name, data) in enumerate(merged_entities.items()):
        data["description"] = desc_summaries[idx]
        data["additional_properties"] = ap_summaries[idx]

    filtered_entities = merged_entities
    filtered_msgs = merged_msgs
    filtered_relations = list(merged_relations.values())
    logger.info(f"Kept all: {len(filtered_entities)} entities, {len(filtered_msgs)} MSGs, {len(filtered_relations)} relations")

    # 阶段2：本地代码转换
    logger.info("Building information layer (MSG → maximal simplices)...")
    information_layer = []
    for msg in filtered_msgs:
        entity_list = sorted(set(msg["entities"]))
        if len(entity_list) < 2:
            continue
        simplex = {
            "id": compute_mdhash_id(str(entity_list), prefix=f"simplex-{len(entity_list)-1}-"),
            "entities": entity_list,
            "dimension": len(entity_list) - 1,
            "is_maximal": True,
            "layer": "information",
            "description": msg.get("text", ""),
            "completeness": msg.get("completeness", 0.75),
            "source_id": msg.get("source_id", ""),
            "importance": max(
                (filtered_entities.get(e, {}).get("importance", 0.5) for e in entity_list),
                default=0.5
            ),
        }
        information_layer.append(simplex)

    logger.info("Computing maximal simplex flags...")
    information_layer = compute_maximal_flags(information_layer)

    # 构建关系层：二元关系作为1-单纯形
    logger.info("Building relation layer (binary relations → 1-simplices)...")
    relation_layer = []
    for rel in filtered_relations:
        entity_list = sorted([rel["subject"], rel["object"]])
        if len(entity_list) < 2:
            continue
        if entity_list[0] == entity_list[1]:
            continue
        simplex = {
            "id": compute_mdhash_id(f"{rel['subject']}|{rel['predicate']}|{rel['object']}", prefix="simplex-1-"),
            "entities": entity_list,
            "dimension": 1,
            "is_maximal": True,
            "layer": "relation",
            "predicate": rel["predicate"],
            "description": rel.get("description", ""),
            "source_id": rel.get("source_id", ""),
            "importance": rel.get("importance", 0.5),
        }
        relation_layer.append(simplex)
    logger.info(f"Relation layer: {len(relation_layer)} 1-simplices from binary relations")

    # 合并信息层和关系层用于二部图构建
    combined_layer = information_layer + relation_layer

    logger.info("Building bipartite associations (entity ↔ MSG + relations)...")
    filtered_entities = build_entity_coboundary(filtered_entities, combined_layer)
    combined_layer = build_msg_boundary(combined_layer, filtered_entities)

    logger.info("Computing bipartite Laplacian (L_entity, L_msg)...")
    L_entity, L_msg, entity_index, msg_index = build_bipartite_laplacian(combined_layer, filtered_entities)
    logger.info(f"Laplacian computed: L_entity shape={L_entity.shape}, L_msg shape={L_msg.shape}")

    # 计算嵌入向量
    logger.info("Computing embeddings...")
    embedding_texts = []
    embedding_targets = []

    for name, data in filtered_entities.items():
        text = f"{name} {data.get('description', '')}"
        embedding_texts.append(text)
        embedding_targets.append(("entity", name))

    for msg in information_layer:
        text = " ".join(msg["entities"]) + " " + msg.get("description", "")
        embedding_texts.append(text)
        embedding_targets.append(("msg", msg["id"]))

    for rel_sx in relation_layer:
        text = f"{rel_sx['predicate']} " + " ".join(rel_sx["entities"]) + " " + rel_sx.get("description", "")
        embedding_texts.append(text)
        embedding_targets.append(("relation", rel_sx["id"]))

    if embedding_texts and hasattr(simplex_storage, 'embedding_func') and simplex_storage.embedding_func:
        batch_size = 100
        for i in range(0, len(embedding_texts), batch_size):
            batch_texts = embedding_texts[i:i+batch_size]
            try:
                embeddings = await simplex_storage.embedding_func(batch_texts)
                for j, emb in enumerate(embeddings):
                    if i + j < len(embedding_targets):
                        target_type, target_id = embedding_targets[i + j]
                        if target_type == "entity":
                            filtered_entities[target_id]["embedding"] = emb.tolist()
                        elif target_type == "msg":
                            for msg in information_layer:
                                if msg["id"] == target_id:
                                    msg["embedding"] = emb.tolist()
                                    break
                        elif target_type == "relation":
                            for rel_sx in relation_layer:
                                if rel_sx["id"] == target_id:
                                    rel_sx["embedding"] = emb.tolist()
                                    break
            except Exception as e:
                logger.error(f"Error computing embeddings for batch: {e}")

    # 阶段3：存储
    # 存储实体到entities_vdb
    if entity_vdb is not None and filtered_entities:
        logger.info(f"Upserting {len(filtered_entities)} entities to entity_vdb")
        data_for_vdb = {}
        for name, data in filtered_entities.items():
            content = f"{name} {data.get('description', '')} {data.get('additional_properties', '')}".strip()
            data_for_vdb[compute_mdhash_id(name, prefix="ent-")] = {
                "content": content,
                "entity_name": name,
                "entity_type": data.get("entity_type", ""),
                "description": data.get("description", ""),
                "additional_properties": data.get("additional_properties", ""),
                "frequency": data.get("frequency", 1),
                "source_id": data.get("source_id", ""),
                "importance": data.get("importance", 0),
            }
        try:
            await entity_vdb.upsert(data_for_vdb)
        except Exception as e:
            logger.error(f"Error storing entities: {e}")

    # 存储信息层MSG到relationships_vdb
    if relationships_vdb is not None and information_layer:
        logger.info(f"Upserting {len(information_layer)} MSGs to relationships_vdb")
        data_for_vdb = {}
        for msg in information_layer:
            content = " ".join(filter(None, [
                msg.get("description", ""),
                " ".join(msg["entities"]),
            ]))
            data_for_vdb[msg["id"]] = {
                "id_set": msg["entities"],
                "content": content,
                "frequency": 1,
                "source_id": msg.get("source_id", ""),
                "importance": msg.get("importance", 0),
                "dimension": msg.get("dimension", len(msg["entities"]) - 1),
            }
        try:
            await relationships_vdb.upsert(data_for_vdb)
        except Exception as e:
            logger.error(f"Error storing MSGs: {e}")

    # 存储二元关系到relationships_vdb
    if relationships_vdb is not None and relation_layer:
        logger.info(f"Upserting {len(relation_layer)} binary relations to relationships_vdb")
        data_for_vdb = {}
        for rel_sx in relation_layer:
            content = " ".join(filter(None, [
                rel_sx.get("predicate", ""),
                rel_sx.get("description", ""),
                " ".join(rel_sx["entities"]),
            ]))
            data_for_vdb[rel_sx["id"]] = {
                "id_set": rel_sx["entities"],
                "content": content,
                "frequency": 1,
                "source_id": rel_sx.get("source_id", ""),
                "importance": rel_sx.get("importance", 0),
                "dimension": 1,
                "predicate": rel_sx.get("predicate", ""),
            }
        try:
            await relationships_vdb.upsert(data_for_vdb)
        except Exception as e:
            logger.error(f"Error storing binary relations: {e}")

    # 存储到SimplexStorage
    if simplex_storage is not None:
        logger.info(f"Storing {len(information_layer)} MSGs + {len(relation_layer)} relations + {len(filtered_entities)} entities to SimplexStorage")

        # 存储0-单纯形（实体）
        for name, data in filtered_entities.items():
            simplex_id = compute_mdhash_id(name, prefix="simplex-0-")
            simplex_data = {
                "id": simplex_id,
                "type": "simplex",
                "dimension": 0,
                "entities": [name],
                "entity_name": name,
                "entity_type": data.get("entity_type", "organization"),
                "description": data.get("description", ""),
                "source_id": data.get("source_id", ""),
                "additional_properties": data.get("additional_properties", ""),
                "frequency": data.get("frequency", 1),
                "importance": data.get("importance", 0.5),
                "boundary": [],
                "coboundary": data.get("coboundary", []),
            }
            if "embedding" in data:
                simplex_data["embedding"] = data["embedding"]
            try:
                await simplex_storage.upsert_simplex(simplex_id, simplex_data)
            except Exception as e:
                logger.error(f"Error storing entity simplex {name}: {e}")

        # 存储信息层MSG（极大单纯形）
        for msg in information_layer:
            simplex_data = {
                "id": msg["id"],
                "type": "simplex",
                "dimension": msg["dimension"],
                "entities": msg["entities"],
                "is_maximal": True,
                "layer": "information",
                "description": msg.get("description", ""),
                "completeness": msg.get("completeness", 0.75),
                "source_id": msg.get("source_id", ""),
                "importance": msg.get("importance", 0.5),
                "frequency": 1,
                "boundary": msg.get("boundary", []),
                "coboundary": [],
            }
            if "embedding" in msg:
                simplex_data["embedding"] = msg["embedding"]
            try:
                await simplex_storage.upsert_simplex(msg["id"], simplex_data)
            except Exception as e:
                logger.error(f"Error storing MSG simplex {msg['id']}: {e}")

        # 存储关系层（二元关系1-单纯形）
        for rel_sx in relation_layer:
            simplex_data = {
                "id": rel_sx["id"],
                "type": "simplex",
                "dimension": 1,
                "entities": rel_sx["entities"],
                "is_maximal": True,
                "layer": "relation",
                "predicate": rel_sx.get("predicate", ""),
                "description": rel_sx.get("description", ""),
                "source_id": rel_sx.get("source_id", ""),
                "importance": rel_sx.get("importance", 0.5),
                "frequency": 1,
                "boundary": rel_sx.get("boundary", []),
                "coboundary": [],
            }
            if "embedding" in rel_sx:
                simplex_data["embedding"] = rel_sx["embedding"]
            try:
                await simplex_storage.upsert_simplex(rel_sx["id"], simplex_data)
            except Exception as e:
                logger.error(f"Error storing relation simplex {rel_sx['id']}: {e}")

        # 缓存Laplacian矩阵
        try:
            simplex_storage.cache_laplacian("L_entity", L_entity)
            simplex_storage.cache_laplacian("L_msg", L_msg)
            simplex_storage.cache_index("entity_index", entity_index)
            simplex_storage.cache_index("msg_index", msg_index)
            logger.info("Cached Laplacian matrices and indices to SimplexStorage")
        except Exception as e:
            logger.error(f"Error caching Laplacian: {e}")

    # 更新文本块的提取记录
    if chunks:
        for chunk_key in chunks:
            chunk_entities = set()
            chunk_msgs = set()
            chunk_relations = set()
            for name, data in filtered_entities.items():
                source_id = data.get("source_id", "")
                if chunk_key in source_id.split(GRAPH_FIELD_SEP):
                    chunk_entities.add(name)
            for msg in filtered_msgs:
                source_id = msg.get("source_id", "")
                if chunk_key in source_id.split(GRAPH_FIELD_SEP):
                    chunk_msgs.add(tuple(sorted(msg["entities"])))
            for rel in filtered_relations:
                source_id = rel.get("source_id", "")
                if chunk_key in source_id.split(GRAPH_FIELD_SEP):
                    chunk_relations.add((rel["subject"], rel["predicate"], rel["object"]))
            if chunk_entities or chunk_msgs or chunk_relations:
                chunks[chunk_key]['extracted_entities'] = list(chunk_entities)
                chunks[chunk_key]['extracted_relations'] = [str(m) for m in chunk_msgs]
                chunks[chunk_key]['extracted_binary_relations'] = [str(r) for r in chunk_relations]

    logger.info("Completed extract_entities function successfully (MSG mode)")
    logger.info(f"extract_entities function took {time.time() - start_time:.2f} seconds to complete")
    if failed_chunks:
        logger.warning(f"Failed to extract entities for {len(failed_chunks)} chunks: {failed_chunks}")
    return failed_chunks
