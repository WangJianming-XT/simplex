import asyncio
import json
import re
import os
import time
import warnings
from collections import defaultdict, Counter
from contextlib import asynccontextmanager
from typing import Union, List, Dict, Set, Optional

import numpy as np

from ..utils import logger
from ..base import BaseKVStorage, BaseVectorStorage, TextChunkSchema, QueryParam
from ..prompt import GRAPH_FIELD_SEP, PROMPTS
from ..llm import openai_embedding
from ..utils import process_combine_contexts, compute_mdhash_id, encode_string_by_tiktoken
from ._config import DualDimensionConfig, EMB_MODEL, EMB_API_KEY, EMB_BASE_URL, semantic_similarity, normalize_entity_name
from ._simplicial_complex import HeterogeneousSimplicialComplex, get_simplex_entities, calculate_simplex_score
from ._retriever import SimplicialRAGRetriever, compute_semantic_similarity


async def _extract_query_entities(query: str, global_config: dict) -> tuple:
    """从查询中提取实体和关系（统一入口）

    使用统一的normalize_entity_name进行实体名称标准化，
    解决extraction用大写、retrieval用小写导致匹配失败的问题。

    Args:
        query: 查询文本
        global_config: 全局配置

    Returns:
        (extracted_entities, extracted_relations)
    """
    extracted_entities = []
    extracted_relations = []

    use_llm_func = global_config.get("llm_model_func")
    enable_llm_extraction = global_config.get("enable_llm_keyword_extraction", True)

    if use_llm_func and not callable(use_llm_func):
        logger.error(f"Invalid use_llm_func: {use_llm_func}")
        use_llm_func = None

    if not (enable_llm_extraction and use_llm_func and callable(use_llm_func)):
        logger.info("LLM extraction not enabled or invalid LLM function")
        min_entity_len = DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH
        possible_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', query)
        for entity_name in possible_entities:
            if len(entity_name) < min_entity_len:
                continue
            normalized_name = normalize_entity_name(entity_name)
            extracted_entities.append({'type': 'Entity', 'name': normalized_name, 'description': ''})
        return extracted_entities, extracted_relations

    try:
        if "query_entity_extraction" not in PROMPTS:
            from ..prompt import PROMPTS as RELOADED_PROMPTS
            if "query_entity_extraction" in RELOADED_PROMPTS:
                PROMPTS.update(RELOADED_PROMPTS)

        if "query_entity_extraction" not in PROMPTS:
            min_entity_len = DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH
            possible_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', query)
            for entity_name in possible_entities:
                if len(entity_name) < min_entity_len:
                    continue
                normalized_name = normalize_entity_name(entity_name)
                extracted_entities.append({'type': 'Entity', 'name': normalized_name, 'description': ''})
            return extracted_entities, extracted_relations

        if not isinstance(query, str):
            query = str(query) if query else ""

        prompt_template = PROMPTS["query_entity_extraction"]
        extract_prompt = prompt_template.replace('{query}', query)

        try:
            extracted_data = await use_llm_func(extract_prompt)
        except Exception as llm_e:
            logger.error(f"LLM function call failed: {llm_e}")
            extracted_data = None

        if not extracted_data or not isinstance(extracted_data, str) or len(extracted_data.strip()) == 0:
            logger.warning("LLM returned empty or invalid response")
            return extracted_entities, extracted_relations

        cleaned_data = extracted_data.strip()
        if cleaned_data.startswith('```json'):
            cleaned_data = cleaned_data[7:].strip()
            if cleaned_data.endswith('```'):
                cleaned_data = cleaned_data[:-3].strip()
        elif cleaned_data.startswith('```'):
            cleaned_data = cleaned_data[3:].strip()
            if cleaned_data.endswith('```'):
                cleaned_data = cleaned_data[:-3].strip()

        if not cleaned_data.startswith('{'):
            json_start = cleaned_data.find('{')
            json_end = cleaned_data.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                cleaned_data = cleaned_data[json_start:json_end]
            else:
                logger.warning(f"LLM returned invalid JSON format: {cleaned_data[:100]}")
                return extracted_entities, extracted_relations

        cleaned_data = cleaned_data.replace('{{', '{').replace('}}', '}')

        try:
            parsed_data = json.loads(cleaned_data)
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error for: {cleaned_data[:100]}")
            return extracted_entities, extracted_relations

        raw_entities = parsed_data.get("entities", [])
        highest_simplex = parsed_data.get("highest_simplex", {})
        raw_simplices = parsed_data.get("simplices", [])
        raw_relations = parsed_data.get("relations", [])

        for ent in raw_entities:
            if isinstance(ent, dict):
                entity_name = ent.get('name', '')
                normalized_name = normalize_entity_name(entity_name)
                if len(normalized_name) < DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH:
                    logger.debug(f"跳过过短实体: '{normalized_name}' (len={len(normalized_name)})")
                    continue
                extracted_entities.append({
                    'type': ent.get('type', 'Entity'),
                    'name': normalized_name,
                    'description': ent.get('description', '')
                })
            elif isinstance(ent, str):
                normalized_name = normalize_entity_name(ent)
                if len(normalized_name) < DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH:
                    logger.debug(f"跳过过短实体: '{normalized_name}' (len={len(normalized_name)})")
                    continue
                extracted_entities.append({'type': 'Entity', 'name': normalized_name, 'description': ''})

        if highest_simplex and isinstance(highest_simplex, dict) and 'entities' in highest_simplex:
            entities_list = highest_simplex.get('entities', [])
            normalized_entities = [normalize_entity_name(ent) for ent in entities_list]
            relation_desc = highest_simplex.get('description', '')
            dimension = highest_simplex.get('dimension', len(normalized_entities) - 1 if normalized_entities else 0)
            if normalized_entities:
                extracted_relations.append({
                    'entities': normalized_entities,
                    'description': relation_desc,
                    'dimension': dimension,
                    'is_highest': True
                })

        if raw_simplices:
            highest_entity_set = set()
            if extracted_relations:
                highest_entity_set = set(extracted_relations[0].get('entities', []))

            for item in raw_simplices:
                if isinstance(item, dict):
                    entities_list = item.get('entities', [])
                    normalized_entities = [normalize_entity_name(ent) for ent in entities_list]
                    relation_desc = item.get('description', '')
                    dimension = item.get('dimension', len(normalized_entities) - 1 if normalized_entities else 0)
                    if normalized_entities and len(normalized_entities) >= 2:
                        item_entity_set = set(normalized_entities)
                        if item_entity_set != highest_entity_set and not item_entity_set.issubset(highest_entity_set):
                            extracted_relations.append({
                                'entities': normalized_entities,
                                'description': relation_desc,
                                'dimension': dimension
                            })

        if not highest_simplex and not raw_simplices and raw_relations:
            for item in raw_relations:
                if isinstance(item, dict):
                    entities_list = item.get('entities', [])
                    normalized_entities = [normalize_entity_name(ent) for ent in entities_list]
                    relation_desc = item.get('description', '')
                    dimension = item.get('dimension', len(normalized_entities) - 1 if normalized_entities else 0)
                    if normalized_entities:
                        extracted_relations.append({
                            'entities': normalized_entities,
                            'description': relation_desc,
                            'dimension': dimension
                        })
                elif isinstance(item, list):
                    normalized_entities = [normalize_entity_name(ent) for ent in item]
                    extracted_relations.append({
                        'entities': normalized_entities,
                        'description': '',
                        'dimension': len(normalized_entities) - 1 if normalized_entities else 0
                    })

        # 关系数量上限：防止过多低维关系淹没检索
        # 优先保留高维关系（highest_simplex），再按维度降序保留
        if len(extracted_relations) > 5:
            extracted_relations.sort(key=lambda r: r.get('dimension', 0), reverse=True)
            highest_rel = [r for r in extracted_relations if r.get('is_highest', False)]
            other_rels = [r for r in extracted_relations if not r.get('is_highest', False)]
            extracted_relations = highest_rel + other_rels[:4]

        logger.info(f"Extracted {len(extracted_entities)} entities and {len(extracted_relations)} relations from query")

    except Exception as e:
        logger.warning(f"Error in LLM extraction: {e}")
        import traceback
        traceback.print_exc()

    return extracted_entities, extracted_relations


async def _build_hsc_from_storage(simplex_storage) -> HeterogeneousSimplicialComplex:
    """从存储中构建异质单纯复形

    Args:
        simplex_storage: 复形存储对象

    Returns:
        构建好的HSC实例
    """
    hsc = HeterogeneousSimplicialComplex()

    try:
        cached_data = await simplex_storage.get_cached_laplacians()
        hsc.L0 = cached_data['L0']
        hsc.L1 = cached_data['L1']
        hsc.nodes = cached_data['nodes']
        hsc.simplices = cached_data['simplices']
    except Exception as e:
        logger.warning(f"Failed to load cached Laplacians: {e}")

    if not hsc.nodes or not hsc.simplices:
        logger.warning("缓存加载后HSC数据为空，从simplex_storage重建HSC")
        all_simplices = await simplex_storage.get_all_simplices()
        # 预构建0-simplex实体类型查找表，确保每个节点获取正确的entity_type
        entity_type_map = {}
        for simplex_id, simplex_data in all_simplices:
            dim = simplex_data.get('dimension', 0)
            if dim == 0:
                entity_name = simplex_data.get('entity_name', '')
                etype = simplex_data.get('entity_type', 'Entity')
                if entity_name and entity_name not in entity_type_map:
                    entity_type_map[entity_name] = etype
        # 预构建实体面查找表，加速高维复形的boundary回退计算
        entity_face_map = {}
        for simplex_id, simplex_data in all_simplices:
            hsc.simplices[simplex_id] = simplex_data
            nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
            dim = simplex_data.get('dimension', 0)
            if dim >= 1:
                map_key = tuple(sorted(nodes))
                entity_face_map[(dim, map_key)] = simplex_id
            for node in nodes:
                node_entity_type = entity_type_map.get(node, 'Entity')
                if node not in hsc.nodes:
                    hsc.nodes[node] = {'type': node_entity_type, 'vector': []}
                if node not in hsc.simplices:
                    hsc.simplices[node] = {
                        'id': node,
                        'dimension': 0,
                        'entities': [node],
                        'nodes': [node],
                        'type': node_entity_type,
                        'entity_type': node_entity_type,
                        'boundary': [],
                        'coboundary': [],
                        'importance': simplex_data.get('importance', 1.0),
                        'frequency': simplex_data.get('frequency', 1)
                    }
            if dim >= 1:
                for node in nodes:
                    if node in hsc.simplices:
                        if simplex_id not in hsc.simplices[node].get('coboundary', []):
                            hsc.simplices[node]['coboundary'].append(simplex_id)
                if dim == 1 and nodes:
                    existing_boundary = simplex_data.get('boundary', [])
                    if existing_boundary:
                        hsc.simplices[simplex_id]['boundary'] = existing_boundary
                    else:
                        hsc.simplices[simplex_id]['boundary'] = list(nodes)
                elif dim >= 2:
                    boundary_ids = simplex_data.get('boundary', [])
                    hsc.simplices[simplex_id]['boundary'] = boundary_ids
                    for bid in boundary_ids:
                        if bid in hsc.simplices:
                            if simplex_id not in hsc.simplices[bid].get('coboundary', []):
                                hsc.simplices[bid]['coboundary'].append(simplex_id)
        hsc.build_incidence_matrices()
        hsc.compute_hodge_laplacians()
        logger.info(f"HSC重建完成：nodes={len(hsc.nodes)}, simplices={len(hsc.simplices)}")

    return hsc


async def _match_entities_to_hsc(extracted_entities, hsc, simplex_storage, entities_vdb=None, embedding_func=None, embedding_cache=None) -> tuple:
    """并行双维度实体匹配：拓扑维度 ‖ 语义维度 → 交叉验证融合

    架构原则：
    1. 拓扑维度（字符串匹配）和语义维度（嵌入检索）并行执行
    2. 两个维度各自独立产生候选集
    3. 交叉验证：两维度都支持的候选获得最高置信度
    4. 子串包含关系自动检测修饰词，动态调整语义验证阈值

    Args:
        extracted_entities: 提取的实体列表
        hsc: 异质单纯复形
        simplex_storage: 复形存储
        entities_vdb: 实体向量数据库（可选，用于语义匹配回退）
        embedding_func: 嵌入函数（可选，用于语义验证）
        embedding_cache: 预计算的嵌入向量缓存（可选，dict[simplex_id, np.ndarray]，
            来自SimplicialRAGRetriever._simplex_embedding_cache，避免对已知实体重复调用API）

    Returns:
        (query_vertices, matched_entity_names, virtual_node_ids)
    """
    query_vertices = []
    matched_entity_names = set()
    virtual_node_ids = set()

    vertex_descriptions = {}
    entity_embeddings = {}
    for simplex_id, simplex_data in hsc.simplices.items():
        nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
        description = simplex_data.get('description', '')
        for node in nodes:
            if node not in vertex_descriptions:
                vertex_descriptions[node] = description
        simplex_emb = None
        if embedding_cache is not None and simplex_id in embedding_cache:
            simplex_emb = embedding_cache[simplex_id]
        elif simplex_data.get('embedding') is not None:
            try:
                simplex_emb = np.array(simplex_data['embedding'], dtype=np.float32)
            except (ValueError, TypeError):
                pass
        if simplex_emb is not None and simplex_emb.ndim == 1 and len(simplex_emb) > 0:
            for node in nodes:
                if node not in entity_embeddings:
                    entity_embeddings[node] = simplex_emb

    _local_emb_cache = {}

    def _cosine_sim(emb_a, emb_b):
        emb_a = np.asarray(emb_a, dtype=np.float32).ravel()
        emb_b = np.asarray(emb_b, dtype=np.float32).ravel()
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a > 0 and norm_b > 0:
            return float(emb_a @ emb_b / (norm_a * norm_b))
        return None

    async def _get_embedding(text):
        if text in _local_emb_cache:
            return _local_emb_cache[text]
        if text in entity_embeddings:
            emb = entity_embeddings[text]
            _local_emb_cache[text] = emb
            return emb
        if embedding_cache is not None and text in embedding_cache:
            emb = np.array(embedding_cache[text], dtype=np.float32)
            _local_emb_cache[text] = emb
            return emb
        if embedding_func is None:
            return None
        try:
            embeddings = await embedding_func([text])
            if embeddings is not None and len(embeddings) >= 1:
                emb = np.array(embeddings[0], dtype=np.float32)
                _local_emb_cache[text] = emb
                return emb
        except Exception as e:
            logger.debug(f"嵌入计算失败: {e}")
        return None

    async def _batch_get_embeddings(texts):
        """批量获取嵌入向量，优先使用持久化缓存，其次API批量调用"""
        results = {}
        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(texts):
            if text in _local_emb_cache:
                results[i] = _local_emb_cache[text]
            elif text in entity_embeddings:
                emb = entity_embeddings[text]
                _local_emb_cache[text] = emb
                results[i] = emb
            elif embedding_cache is not None and text in embedding_cache:
                emb = np.array(embedding_cache[text], dtype=np.float32)
                _local_emb_cache[text] = emb
                results[i] = emb
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        if uncached_texts and embedding_func is not None:
            try:
                batch_embeddings = await embedding_func(uncached_texts)
                if batch_embeddings is not None and len(batch_embeddings) == len(uncached_texts):
                    for j, emb in enumerate(batch_embeddings):
                        emb_arr = np.array(emb, dtype=np.float32)
                        idx = uncached_indices[j]
                        results[idx] = emb_arr
                        _local_emb_cache[uncached_texts[j]] = emb_arr
            except Exception as e:
                logger.debug(f"批量嵌入计算失败: {e}")
        return results

    async def match_entity(entity):
        """并行双通道实体匹配：拓扑通道 ‖ 语义通道 → 交叉验证"""
        entity_name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
        entity_type = entity.get('type', 'Entity') if isinstance(entity, dict) else 'Entity'
        entity_desc = entity.get('description', '') if isinstance(entity, dict) else ''

        if not entity_name or not entity_name.strip():
            return None

        normalized_query = normalize_entity_name(entity_name)

        # ===== 通道1：拓扑维度（字符串匹配）=====
        topo_match = None
        topo_score = 0
        topo_candidates = []

        entity_simplices = await simplex_storage.get_simplices_by_entity(entity_name)
        if not entity_simplices:
            entity_simplices = await simplex_storage.get_simplices_by_entity(normalized_query)

        if entity_simplices:
            for simplex_id, simplex_data in entity_simplices:
                simplex_entities = simplex_data.get('entities', [])
                for simplex_entity in simplex_entities:
                    if normalize_entity_name(simplex_entity) == normalized_query:
                        topo_match = simplex_entity
                        topo_score = 1.0
                        break
                if topo_match:
                    break

        if not topo_match:
            min_substr_len = DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH
            for node_id in hsc.nodes.keys():
                if node_id in matched_entity_names:
                    continue
                name_score = 0.0
                if isinstance(node_id, str):
                    normalized_node = normalize_entity_name(node_id)
                    if normalized_node == normalized_query:
                        name_score = 1.0
                    elif len(normalized_query) >= min_substr_len and normalized_query in normalized_node and len(normalized_node) > 0:
                        match_ratio = len(normalized_query) / len(normalized_node)
                        if match_ratio >= 0.5:
                            name_score = 0.7 + match_ratio * 0.3
                    elif len(normalized_node) >= min_substr_len and normalized_node in normalized_query and len(normalized_query) > 0:
                        match_ratio = len(normalized_node) / len(normalized_query)
                        if match_ratio >= 0.5:
                            name_score = 0.6 + match_ratio * 0.3

                desc_score = 0.0
                if entity_desc and vertex_descriptions.get(node_id):
                    node_desc = vertex_descriptions[node_id]
                    desc_words = set(entity_desc.lower().split())
                    node_words = set(node_desc.lower().split())
                    if desc_words and node_words:
                        overlap = len(desc_words.intersection(node_words))
                        desc_score = overlap / max(len(desc_words), len(node_words))

                total_score = 0.6 * name_score + 0.4 * desc_score
                if total_score >= 0.4:
                    topo_candidates.append((node_id, total_score, name_score))

            # 候选上限：按得分排序后只保留top-5，避免大数据集下模糊匹配泛滥
            if len(topo_candidates) > 5:
                topo_candidates.sort(key=lambda x: x[1], reverse=True)
                topo_candidates = topo_candidates[:5]

        # ===== 通道2：语义维度（VDB嵌入检索，独立于拓扑通道）=====
        sem_match = None
        sem_score = 0
        if entities_vdb is not None:
            try:
                vdb_query_text = entity_name
                if entity_desc:
                    vdb_query_text = f"{entity_name} {entity_desc}"
                vdb_results = await entities_vdb.query(vdb_query_text, top_k=DualDimensionConfig.PARALLEL_ENTITY_SEM_TOP_K)
                if vdb_results:
                    for vdb_result in vdb_results:
                        if isinstance(vdb_result, dict):
                            vdb_entity_name = vdb_result.get("entity_name", "")
                            vdb_score = vdb_result.get("distance", 0)
                        elif isinstance(vdb_result, (list, tuple)) and len(vdb_result) >= 2:
                            vdb_entity_name, vdb_score = vdb_result[0], vdb_result[1]
                        else:
                            continue
                        if not vdb_entity_name:
                            continue
                        vdb_entity_name_normalized = normalize_entity_name(vdb_entity_name)
                        matched_node = None
                        if vdb_entity_name in hsc.nodes:
                            matched_node = vdb_entity_name
                        elif vdb_entity_name_normalized in hsc.nodes:
                            matched_node = vdb_entity_name_normalized
                        else:
                            for node_id in hsc.nodes:
                                if normalize_entity_name(str(node_id)) == vdb_entity_name_normalized:
                                    matched_node = node_id
                                    break
                        if not matched_node and entity_desc:
                            vdb_desc_words = set(vdb_entity_name.lower().split())
                            if entity_desc:
                                vdb_desc_words.update(entity_desc.lower().split())
                            best_desc_match = None
                            best_desc_overlap = 0.0
                            for node_id in hsc.nodes:
                                node_desc = vertex_descriptions.get(node_id, '')
                                if not node_desc:
                                    continue
                                node_words = set(node_desc.lower().split())
                                if not node_words:
                                    continue
                                overlap = len(vdb_desc_words.intersection(node_words))
                                overlap_ratio = overlap / max(len(vdb_desc_words), len(node_words), 1)
                                if overlap_ratio > best_desc_overlap and overlap_ratio >= 0.4:
                                    best_desc_overlap = overlap_ratio
                                    best_desc_match = node_id
                            if best_desc_match:
                                matched_node = best_desc_match
                                logger.info(f"VDB实体 '{vdb_entity_name}' HSC精确匹配失败，"
                                            f"通过description模糊匹配到 '{best_desc_match}' (重叠率={best_desc_overlap:.2f})")
                        if matched_node and matched_node not in matched_entity_names:
                            if vdb_score >= 0.5:
                                sem_match = matched_node
                                sem_score = vdb_score
                                break
                            elif vdb_score >= 0.3 and sem_match is None:
                                sem_match = matched_node
                                sem_score = vdb_score
            except Exception as e:
                logger.warning(f"实体VDB语义检索失败: {e}")

        # ===== 交叉验证融合 =====
        best_match = None
        best_score = 0
        best_semantic_score = None

        if topo_score >= 1.0:
            best_match = topo_match
            best_score = 1.0
            best_semantic_score = 1.0
        elif topo_match and sem_match and topo_match == sem_match:
            best_match = topo_match
            best_score = topo_score * 0.6 + sem_score * 0.4
            best_semantic_score = sem_score
            logger.info(f"实体 '{entity_name}' 双维度对撞匹配到 '{best_match}' "
                        f"(拓扑分={topo_score:.2f}, 语义分={sem_score:.2f})")
        elif topo_candidates:
            topo_candidates.sort(key=lambda x: x[1], reverse=True)
            base_threshold = DualDimensionConfig.ENTITY_MATCH_SEMANTIC_VERIFY_THRESHOLD

            query_emb_text = entity_name
            if entity_desc:
                query_emb_text = f"{entity_name} {entity_desc}"
            batch_texts = [query_emb_text]
            cand_text_map = {}
            modifier_texts = []
            for cid, _, cname_score in topo_candidates:
                if cname_score < 1.0:
                    if cid in entity_embeddings or cid in _local_emb_cache:
                        cand_text_map[cid] = cid
                    else:
                        cand_desc = vertex_descriptions.get(cid, '')
                        cand_text = f"{cid} {cand_desc}" if cand_desc else cid
                        cand_text_map[cid] = cand_text
                    batch_texts.append(cand_text_map[cid])
                    normalized_candidate = normalize_entity_name(cid)
                    is_sub = (normalized_query in normalized_candidate or normalized_candidate in normalized_query)
                    if is_sub and normalized_query != normalized_candidate:
                        longer = normalized_candidate if len(normalized_query) < len(normalized_candidate) else normalized_query
                        shorter = normalized_query if len(normalized_query) < len(normalized_candidate) else normalized_candidate
                        modifier = longer.replace(shorter, "").strip()
                        if len(modifier) / max(len(longer), 1) > 0.2 and modifier not in batch_texts:
                            modifier_texts.append(modifier)
                            batch_texts.append(modifier)

            batch_embs = await _batch_get_embeddings(batch_texts)
            query_emb = batch_embs.get(0)

            for candidate_id, candidate_score, candidate_name_score in topo_candidates:
                if candidate_name_score >= 1.0:
                    best_match = candidate_id
                    best_score = candidate_score
                    best_semantic_score = 1.0
                    break

                cand_lookup = cand_text_map.get(candidate_id, candidate_id)
                cand_idx = batch_texts.index(cand_lookup) if cand_lookup in batch_texts else None
                cand_emb = batch_embs.get(cand_idx) if cand_idx is not None else None

                if query_emb is not None and cand_emb is not None:
                    sem_sim = _cosine_sim(query_emb, cand_emb)
                else:
                    sem_sim = None

                if sem_sim is not None:
                    normalized_candidate = normalize_entity_name(candidate_id)
                    is_substring_match = (normalized_query in normalized_candidate or
                                          normalized_candidate in normalized_query)

                    is_plural_variant = False
                    if normalized_query != normalized_candidate:
                        q_words = normalized_query.split()
                        c_words = normalized_candidate.split()
                        if len(q_words) == len(c_words):
                            all_plural = True
                            for qw, cw in zip(q_words, c_words):
                                if qw == cw:
                                    continue
                                if qw == cw + 'S' or qw == cw + 'ES':
                                    continue
                                if qw.endswith('IES') and cw.endswith('Y') and qw[:-3] == cw[:-1]:
                                    continue
                                if cw == qw + 'S' or cw == qw + 'ES':
                                    continue
                                if cw.endswith('IES') and qw.endswith('Y') and cw[:-3] == qw[:-1]:
                                    continue
                                all_plural = False
                                break
                            is_plural_variant = all_plural

                    if is_plural_variant:
                        verify_threshold = 0.4
                        logger.info(f"单复数变体检测：'{entity_name}'与'{candidate_id}'为单复数关系，"
                                    f"语义验证阈值降至{verify_threshold:.2f}")
                    elif is_substring_match and normalized_query != normalized_candidate:
                        query_is_shorter = len(normalized_query) < len(normalized_candidate)
                        longer = normalized_candidate if query_is_shorter else normalized_query
                        shorter = normalized_query if query_is_shorter else normalized_candidate
                        modifier = longer.replace(shorter, "").strip()
                        modifier_ratio = len(modifier) / max(len(longer), 1)

                        if modifier_ratio > 0.2:
                            mod_idx = batch_texts.index(modifier) if modifier in batch_texts else None
                            mod_emb = batch_embs.get(mod_idx) if mod_idx is not None else None
                            modifier_sem_sim = _cosine_sim(query_emb, mod_emb) if query_emb is not None and mod_emb is not None else None
                            if query_is_shorter:
                                if modifier_sem_sim is not None and modifier_sem_sim > 0.5:
                                    verify_threshold = base_threshold
                                else:
                                    verify_threshold = min(base_threshold + 0.25, 0.95)
                                    logger.info(f"子串包含修饰词验证：'{modifier}'与'{entity_name}'语义相似度"
                                                f"={'%.2f' % modifier_sem_sim if modifier_sem_sim else 'N/A'}"
                                                f"，修饰词改变核心含义，阈值提升至{verify_threshold:.2f}")
                            else:
                                if modifier_sem_sim is not None and modifier_sem_sim > 0.5:
                                    verify_threshold = base_threshold
                                    logger.info(f"候选泛化验证：'{candidate_id}'是'{entity_name}'的子串"
                                                f"（丢失限定词'{modifier}'），但限定词语义相关"
                                                f"(相似度={modifier_sem_sim:.2f})，保持阈值{verify_threshold:.2f}")
                                else:
                                    verify_threshold = min(base_threshold + 0.2, 0.95)
                                    logger.info(f"候选泛化验证：'{candidate_id}'是'{entity_name}'的子串"
                                                f"（丢失限定词'{modifier}'，语义相似度"
                                                f"={'%.2f' % modifier_sem_sim if modifier_sem_sim else 'N/A'}"
                                                f"），阈值提升至{verify_threshold:.2f}")
                        else:
                            verify_threshold = base_threshold
                    else:
                        verify_threshold = base_threshold

                    if sem_sim >= verify_threshold:
                        best_match = candidate_id
                        best_score = candidate_score * 0.6 + sem_sim * 0.4
                        best_semantic_score = sem_sim
                        logger.info(f"实体 '{entity_name}' 模糊匹配到 '{candidate_id}' "
                                    f"(字符串分={candidate_name_score:.2f}, "
                                    f"语义验证通过={sem_sim:.2f}>={verify_threshold:.2f})")
                        break
                    else:
                        logger.info(f"实体 '{entity_name}' 模糊匹配到 '{candidate_id}' "
                                    f"被语义验证拒绝 (字符串分={candidate_name_score:.2f}, "
                                    f"语义相似度={sem_sim:.2f}<{verify_threshold:.2f})")
                else:
                    logger.info(f"实体 '{entity_name}' 模糊匹配到 '{candidate_id}' "
                                f"无法计算语义相似度，跳过该候选")

        if not best_match and sem_match:
            best_match = sem_match
            best_score = sem_score * 0.7
            best_semantic_score = sem_score
            logger.info(f"实体 '{entity_name}' 仅语义通道匹配到 '{sem_match}' (分数: {sem_score:.2f})")

        if best_match:
            is_low = best_score < 0.4
            result = {
                'type': entity_type,
                'id': best_match,
                'original_name': entity_name,
                'description': entity_desc,
                'match_score': best_score,
                'is_low_confidence': is_low,
                'semantic_verify_score': best_semantic_score,
            }
            matched_entity_names.add(best_match)
            dim_info = ""
            if best_semantic_score is not None:
                dim_info = f", 语义验证={best_semantic_score:.2f}"
            logger.info(f"实体 '{entity_name}' 综合匹配到 '{best_match}' "
                        f"(分数: {best_score:.2f}{dim_info})")
            return result

        normalized_query_ent = normalize_entity_name(entity_name)
        sub_entity_words = normalized_query_ent.replace('-', ' ').split()
        sub_matched_results = []

        if len(sub_entity_words) >= 2:
            combined_match = None
            for node_id in hsc.nodes.keys():
                if normalize_entity_name(node_id) == normalized_query_ent:
                    combined_match = node_id
                    break
            if not combined_match:
                for simplex_key in hsc.simplices.keys():
                    if normalize_entity_name(simplex_key) == normalized_query_ent:
                        combined_match = simplex_key
                        break
            if combined_match:
                logger.info(f"复合实体组合匹配：'{entity_name}' 整体匹配到HSC实体 '{combined_match}'")
                result = {
                    'type': entity_type,
                    'id': combined_match,
                    'original_name': entity_name,
                    'description': entity_desc,
                    'match_score': 0.9,
                    'is_low_confidence': False,
                    'semantic_verify_score': None,
                    'is_combined_match': True,
                }
                matched_entity_names.add(combined_match)
                return result

            for sub_word in sub_entity_words:
                if len(sub_word) < DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH:
                    continue
                sub_simplices = await simplex_storage.get_simplices_by_entity(sub_word)
                if sub_simplices:
                    for simplex_id, simplex_data in sub_simplices:
                        simplex_entities = simplex_data.get('entities', [])
                        for se in simplex_entities:
                            if normalize_entity_name(se) == sub_word:
                                sub_matched_results.append({
                                    'type': entity_type,
                                    'id': se,
                                    'original_name': entity_name,
                                    'description': entity_desc,
                                    'match_score': 0.5,
                                    'is_sub_match': True,
                                    'sub_word': sub_word,
                                    'is_low_confidence': False,
                                    'semantic_verify_score': None,
                                })
                                matched_entity_names.add(se)
                                logger.info(f"复合实体拆分：'{entity_name}' 的子词 '{sub_word}' 匹配到HSC实体 '{se}'")
                                break
                        if any(r['sub_word'] == sub_word for r in sub_matched_results):
                            break
                else:
                    for node_id in hsc.nodes.keys():
                        if normalize_entity_name(node_id) == sub_word:
                            sub_matched_results.append({
                                'type': entity_type,
                                'id': node_id,
                                'original_name': entity_name,
                                'description': entity_desc,
                                'match_score': 0.5,
                                'is_sub_match': True,
                                'sub_word': sub_word,
                                'is_low_confidence': False,
                                'semantic_verify_score': None,
                            })
                            matched_entity_names.add(node_id)
                            logger.info(f"复合实体拆分：'{entity_name}' 的子词 '{sub_word}' 匹配到HSC节点 '{node_id}'")
                            break

        if sub_matched_results:
            best_sub = max(sub_matched_results, key=lambda r: r['match_score'])
            best_match = best_sub['id']
            best_score = best_sub['match_score']
            best_semantic_score = None
            for r in sub_matched_results:
                if r['id'] != best_match and r not in query_vertices:
                    query_vertices.append(r)
            logger.info(f"复合实体拆分：'{entity_name}' 拆分为 {len(sub_matched_results)} 个子实体匹配，"
                        f"主匹配='{best_match}'，附加={len(sub_matched_results)-1}个")
            result = {
                'type': entity_type,
                'id': best_match,
                'original_name': entity_name,
                'description': entity_desc,
                'match_score': best_score,
                'is_low_confidence': True,
                'semantic_verify_score': best_semantic_score,
                'is_sub_match': True,
            }
            matched_entity_names.add(best_match)
            return result

        virtual_id = normalize_entity_name(entity_name)
        virtual_node_ids.add(virtual_id)
        proxy_node_ids = []
        if entities_vdb is not None:
            try:
                proxy_query_text = entity_name
                if entity_desc:
                    proxy_query_text = f"{entity_name} {entity_desc}"
                proxy_results = await entities_vdb.query(proxy_query_text, top_k=DualDimensionConfig.ENTITY_PROXY_TOP_K * 2)
                if proxy_results:
                    for pr in proxy_results:
                        if isinstance(pr, dict):
                            pr_id = pr.get("id", "")
                            pr_score = pr.get("distance", 0)
                        elif isinstance(pr, (list, tuple)) and len(pr) >= 2:
                            pr_id, pr_score = pr[0], pr[1]
                        else:
                            continue
                        if pr_id in hsc.nodes and pr_id not in matched_entity_names:
                            proxy_node_ids.append((pr_id, pr_score))
                            if len(proxy_node_ids) >= DualDimensionConfig.ENTITY_PROXY_TOP_K:
                                break
                    if proxy_node_ids:
                        logger.info(f"虚拟节点 '{entity_name}' 通过VDB语义检索找到 {len(proxy_node_ids)} 个代理节点: "
                                    f"{[(pid, f'{pscore:.2f}') for pid, pscore in proxy_node_ids]}")
            except Exception as e:
                logger.debug(f"虚拟节点代理检索失败: {e}")

        primary_proxy = proxy_node_ids[0][0] if proxy_node_ids else None
        all_proxy_coboundaries = []
        for proxy_pid, _ in proxy_node_ids:
            if proxy_pid in hsc.simplices:
                proxy_simplices = await simplex_storage.get_simplices_by_entity(proxy_pid)
                for sid, sdata in proxy_simplices:
                    if sid not in all_proxy_coboundaries:
                        all_proxy_coboundaries.append(sid)

        virtual_simplex = {
            'id': virtual_id,
            'dimension': 0,
            'entities': [virtual_id],
            'nodes': [virtual_id],
            'type': entity_type,
            'entity_type': entity_type,
            'boundary': [],
            'coboundary': all_proxy_coboundaries[:DualDimensionConfig.VDB_ONE_HOP_COBOUNDARY_LIMIT * DualDimensionConfig.ENTITY_PROXY_TOP_K],
            'importance': DualDimensionConfig.ENTITY_MATCH_VIRTUAL_CONFIDENCE,
            'frequency': 1,
            'is_virtual': True
        }
        if primary_proxy:
            virtual_simplex['proxy_node_id'] = primary_proxy
        if proxy_node_ids:
            virtual_simplex['proxy_node_ids'] = [pid for pid, _ in proxy_node_ids]

        result = {
            'type': entity_type,
            'id': virtual_id,
            'original_name': entity_name,
            'description': entity_desc,
            'match_score': DualDimensionConfig.ENTITY_MATCH_VIRTUAL_CONFIDENCE,
            'is_virtual': True,
            'virtual_simplex_data': virtual_simplex
        }
        matched_entity_names.add(virtual_id)
        proxy_info = f'(代理={primary_proxy}' + (f', 共{len(proxy_node_ids)}个代理)' if len(proxy_node_ids) > 1 else ')')
        logger.info(f"实体 '{entity_name}' 未匹配到HSC，标记为虚拟节点 '{virtual_id}'"
                    f"{proxy_info if proxy_node_ids else '(无代理)'}")
        return result

    if extracted_entities:
        logger.info(f"开始匹配 {len(extracted_entities)} 个提取实体...")
        match_tasks = [asyncio.create_task(match_entity(entity)) for entity in extracted_entities]
        match_results = await asyncio.gather(*match_tasks, return_exceptions=True)

        for result in match_results:
            if isinstance(result, Exception):
                logger.error(f"Error matching entity: {result}")
                continue
            if result is not None:
                query_vertices.append(result)

    virtual_vertices = [v for v in query_vertices if isinstance(v, dict) and v.get('is_virtual')]
    if len(virtual_vertices) >= 1:
        all_query_word_sets = []
        for v in query_vertices:
            if not isinstance(v, dict):
                continue
            vname = normalize_entity_name(v.get('original_name', v.get('id', '')))
            words = [w for w in vname.replace('-', ' ').split() if len(w) >= DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH]
            if words:
                all_query_word_sets.append((v, set(words)))
        if len(all_query_word_sets) >= 2:
            all_words = set()
            for _, ws in all_query_word_sets:
                all_words.update(ws)
            combined_match = None
            if all_words:
                all_hsc_names = set(hsc.nodes.keys()) | set(hsc.simplices.keys())
                for hsc_name in all_hsc_names:
                    hsc_word_set = set(normalize_entity_name(hsc_name).replace('-', ' ').split())
                    if all_words == hsc_word_set:
                        combined_match = hsc_name
                        break
            if combined_match:
                merged_virtual_ids = set()
                for v in virtual_vertices:
                    vid = v.get('original_name', '')
                    merged_virtual_ids.add(vid)
                    if vid in virtual_node_ids:
                        virtual_node_ids.discard(vid)
                query_vertices = [v for v in query_vertices if not (isinstance(v, dict) and v.get('is_virtual') and v.get('original_name', '') in merged_virtual_ids)]
                subsumed_real = []
                for v in list(query_vertices):
                    if isinstance(v, dict) and not v.get('is_virtual'):
                        vname = normalize_entity_name(v.get('original_name', v.get('id', '')))
                        v_words = set(vname.replace('-', ' ').split())
                        if v_words.issubset(all_words) and v_words != all_words:
                            subsumed_real.append(v)
                            if v.get('id', '') in matched_entity_names:
                                matched_entity_names.discard(v.get('id', ''))
                for v in subsumed_real:
                    query_vertices.remove(v)
                combined_result = {
                    'type': 'Entity',
                    'id': combined_match,
                    'original_name': ' '.join(sorted(all_words)),
                    'description': '',
                    'match_score': 0.9,
                    'is_low_confidence': False,
                    'semantic_verify_score': None,
                    'is_combined_match': True,
                }
                query_vertices.append(combined_result)
                matched_entity_names.add(combined_match)
                logger.info(f"虚拟节点组合合并：虚拟节点{merged_virtual_ids}+被包含真实节点{[v.get('original_name','') for v in subsumed_real]} "
                            f"合并匹配到HSC实体 '{combined_match}' (分数: 0.90)")

    matched_names = [v.get('id', v.get('original_name', '?')) for v in query_vertices if isinstance(v, dict)]
    virtual_names = [v.get('original_name', '?') for v in query_vertices if isinstance(v, dict) and v.get('is_virtual')]
    real_names = [v.get('original_name', '?') for v in query_vertices if isinstance(v, dict) and not v.get('is_virtual')]
    logger.info(f"实体匹配结果：{len(query_vertices)}个实体，"
                f"真实匹配={len(real_names)}个{real_names}，"
                f"虚拟节点={len(virtual_names)}个{virtual_names}")

    return query_vertices, matched_entity_names, virtual_node_ids


async def _collect_text_chunks(ranked_simplices, text_chunks_db, total_chunks_limit=50, max_context_tokens=None, query_text=None, entity_count=0, relation_count=0, hsc=None, retriever=None, embedding_func=None, query_vertices=None, simplex_storage=None, entities_vdb=None, relationships_vdb=None, ll_keywords="", hl_keywords="") -> tuple:
    """从检索到的复形中收集相关文本块

    改进策略：种子优先 + 精度优先，解决大数据集下噪声爆炸问题。

    收集优先级（从高到低）：
    1. 种子实体直接source_id（最高优先级，确保答案所在chunk不被遗漏）
    2. Local路径：实体VDB检索 → 仅收集直接source_id（不做一跳扩展）
    3. Global路径：关系VDB检索 → 仅收集直接source_id（不做一跳扩展）
    4. 拓扑补充：ranked_simplices的source_id（仅种子复形）

    排名策略：
    - 种子chunk优先，VDB order次之，被多路径触达的chunk排前面
    - 加入chunk内容与查询的语义相关性作为排序因子

    Args:
        ranked_simplices: 排序后的复形列表
        text_chunks_db: 文本块数据库
        total_chunks_limit: 最大文本块数量（软上限）
        max_context_tokens: 最大上下文token数
        query_text: 查询文本
        entity_count: 查询实体数量
        relation_count: 查询关系数量
        hsc: 异质单纯复形实例
        retriever: 检索器实例
        embedding_func: 嵌入函数
        query_vertices: 查询匹配的种子实体列表
        simplex_storage: 复形存储
        entities_vdb: 实体向量数据库
        relationships_vdb: 关系向量数据库（Global路径）
        ll_keywords: LLM提取的低层关键词（实体名，逗号分隔），用于Local路径VDB查询
        hl_keywords: LLM提取的高层关键词（关系描述，逗号分隔），用于Global路径VDB查询

    Returns:
        (related_chunks, source_types)
    """
    if entity_count > 0 or relation_count > 0:
        dynamic_limit = DualDimensionConfig.compute_chunk_budget(
            entity_count, relation_count, total_chunks_limit
        )
        total_chunks_limit = min(total_chunks_limit, dynamic_limit)

    related_chunks = []
    source_ids = set()
    source_types = {}
    total_tokens = 0

    _invalid_sources = {'entity_extraction', 'relation_extraction', 'high_order_extraction'}

    async def collect_chunk(source_id):
        try:
            chunk_data = await text_chunks_db.get_by_id(source_id)
            if chunk_data and "content" in chunk_data:
                return chunk_data["content"]
        except Exception as e:
            logger.warning(f"Error retrieving chunk for source_id {source_id}: {e}")
        return None

    def _parse_source_ids(source_val):
        """解析source字段为source_id列表"""
        if not source_val:
            return []
        if isinstance(source_val, str):
            return [s.strip() for s in source_val.split('<SEP>') if s.strip() and s.strip() not in _invalid_sources]
        elif isinstance(source_val, list):
            return [str(s).strip() for s in source_val if s and str(s).strip() not in _invalid_sources]
        return []

    source_ref_count = Counter()
    source_type_map = {}
    source_is_seed = defaultdict(bool)
    source_vdb_order = {}
    source_relation_counts = Counter()
    source_semantic_score = {}

    # ===== 阶段0（最高优先级）：种子实体直接source_id =====
    # 种子实体是查询直接匹配到的实体，其source_id对应的chunk最可能包含答案
    seed_source_ids = set()
    if query_vertices and hsc is not None:
        for v in query_vertices:
            if not isinstance(v, dict) or v.get('is_virtual'):
                continue
            vid = v.get('id', '')
            if not vid or vid not in hsc.simplices:
                continue
            vdata = hsc.simplices[vid]
            for key in ['source', 'source_id']:
                for src_id in _parse_source_ids(vdata.get(key)):
                    seed_source_ids.add(src_id)
                    source_is_seed[src_id] = True
                    source_ref_count[src_id] += 2
                    if src_id not in source_type_map:
                        source_type_map[src_id] = 'seed_entity'
                    if src_id not in source_vdb_order:
                        source_vdb_order[src_id] = 0
            coboundary = vdata.get('coboundary', [])
            for cb_id in coboundary[:1]:
                cb_data = hsc.simplices.get(cb_id, {})
                if not cb_data:
                    continue
                for key in ['source', 'source_id']:
                    for src_id in _parse_source_ids(cb_data.get(key)):
                        if src_id not in seed_source_ids:
                            seed_source_ids.add(src_id)
                            source_is_seed[src_id] = True
                            source_ref_count[src_id] += 1
                            if src_id not in source_type_map:
                                source_type_map[src_id] = 'seed_entity_one_hop'
                            if src_id not in source_vdb_order:
                                source_vdb_order[src_id] = 0

    logger.info(f"种子实体直接收集：{len(seed_source_ids)}个source_id")

    # ===== 阶段1A：Local路径 — 仅收集实体直接source_id，不做一跳扩展 =====
    local_query = ll_keywords if ll_keywords else query_text
    vdb_entity_count = 0
    if entities_vdb is not None and local_query and hsc is not None:
        try:
            entity_results = await entities_vdb.query(local_query, top_k=20)
            logger.info(f"Local路径：entities_vdb.query('{local_query[:50]}')返回 {len(entity_results) if entity_results else 0} 个结果")
            if entity_results:
                for order_idx, vdb_result in enumerate(entity_results):
                    if isinstance(vdb_result, dict):
                        entity_name = vdb_result.get("entity_name", vdb_result.get("id", ""))
                        vdb_score = vdb_result.get("distance", 0)
                    elif isinstance(vdb_result, (list, tuple)) and len(vdb_result) >= 2:
                        entity_name, vdb_score = vdb_result[0], vdb_result[1]
                    else:
                        continue
                    if vdb_score < DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                        continue
                    matched_node = None
                    if entity_name in hsc.simplices:
                        matched_node = entity_name
                    else:
                        normalized = normalize_entity_name(entity_name)
                        for node_id in hsc.nodes:
                            if normalize_entity_name(str(node_id)) == normalized:
                                matched_node = node_id
                                break
                    if not matched_node:
                        continue
                    node_data = hsc.simplices.get(matched_node, {})
                    for key in ['source', 'source_id']:
                        for src_id in _parse_source_ids(node_data.get(key)):
                            source_ref_count[src_id] += 1
                            if src_id not in source_type_map:
                                source_type_map[src_id] = 'vdb_local'
                            if src_id not in source_vdb_order:
                                source_vdb_order[src_id] = order_idx
                            vdb_entity_count += 1
        except Exception as e:
            logger.warning(f"Local路径chunk收集失败: {e}")

    # ===== 阶段1B：Global路径 — 仅收集关系直接source_id，不做一跳扩展 =====
    global_query = hl_keywords if hl_keywords else query_text
    vdb_relation_count = 0
    if relationships_vdb is not None and global_query and hsc is not None:
        try:
            relation_results = await relationships_vdb.query(global_query, top_k=20)
            logger.info(f"Global路径：relationships_vdb.query('{global_query[:50]}')返回 {len(relation_results) if relation_results else 0} 个结果")
            if relation_results:
                for order_idx, vdb_result in enumerate(relation_results):
                    if isinstance(vdb_result, dict):
                        rel_id = vdb_result.get("id", vdb_result.get("relationship_id", ""))
                        vdb_score = vdb_result.get("distance", 0)
                    elif isinstance(vdb_result, (list, tuple)) and len(vdb_result) >= 2:
                        rel_id, vdb_score = vdb_result[0], vdb_result[1]
                    else:
                        continue
                    if vdb_score < DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                        continue
                    matched_simplex = None
                    if rel_id in hsc.simplices:
                        matched_simplex = rel_id
                    else:
                        for sid in hsc.simplices:
                            if str(sid) == str(rel_id):
                                matched_simplex = sid
                                break
                    if not matched_simplex:
                        continue
                    simplex_data = hsc.simplices.get(matched_simplex, {})
                    for key in ['source', 'source_id']:
                        for src_id in _parse_source_ids(simplex_data.get(key)):
                            source_ref_count[src_id] += 1
                            if src_id not in source_type_map:
                                source_type_map[src_id] = 'vdb_global'
                            if src_id not in source_vdb_order:
                                source_vdb_order[src_id] = order_idx + 20
                            vdb_relation_count += 1
        except Exception as e:
            logger.warning(f"Global路径chunk收集失败: {e}")

    # ===== 阶段1C：拓扑补充 — 仅收集种子复形的source_id =====
    topo_source_count = 0
    for simplex_data in ranked_simplices:
        if isinstance(simplex_data, tuple):
            simplex_data = simplex_data[1]
        is_seed = simplex_data.get('is_seed', False)
        if not is_seed:
            continue
        source_list = simplex_data.get("source_id") or simplex_data.get("source")
        for src_id in _parse_source_ids(source_list):
            source_ref_count[src_id] += 1
            if src_id not in source_type_map:
                source_type_map[src_id] = 'topo_seed_simplex'
            source_is_seed[src_id] = True
            topo_source_count += 1

    all_candidate_ids = set(source_ref_count.keys()) | seed_source_ids

    logger.info(f"chunk候选集：种子={len(seed_source_ids)}个，"
                f"Local路径={vdb_entity_count}个引用，"
                f"Global路径={vdb_relation_count}个引用，"
                f"拓扑种子={topo_source_count}个引用，"
                f"唯一source_id={len(all_candidate_ids)}个")

    # ===== 阶段2：批量获取chunk内容 =====
    collect_tasks = []
    for source_id in all_candidate_ids:
        if source_id in source_ids:
            continue
        source_ids.add(source_id)
        task = asyncio.create_task(collect_chunk(source_id))
        collect_tasks.append((task, source_id))

    chunk_contents = {}
    if collect_tasks:
        collect_results = await asyncio.gather(*[task for task, _ in collect_tasks], return_exceptions=True)
        for i, result in enumerate(collect_results):
            if result is None or isinstance(result, Exception):
                continue
            _, source_id = collect_tasks[i]
            chunk_contents[source_id] = result

    # ===== 阶段3：排名 — 种子优先 + VDB order + 语义相关性 =====
    query_embedding = None
    if query_text and retriever is not None:
        try:
            query_embedding = await retriever._get_query_embedding(query_text)
        except Exception:
            pass

    scored_chunks = []
    for source_id, content in chunk_contents.items():
        ref_count = source_ref_count.get(source_id, 0)
        is_seed = source_id in seed_source_ids or source_is_seed.get(source_id, False)
        order = source_vdb_order.get(source_id, 999)
        rel_counts = source_relation_counts.get(source_id, 0)
        source_type = source_type_map.get(source_id, 'unknown')
        is_vdb_path = source_type in ('vdb_local', 'vdb_global')

        semantic_score = 0.0
        if query_embedding is not None and retriever is not None:
            chunk_emb = retriever._chunk_embedding_cache.get(source_id)
            if chunk_emb is None and embedding_func is not None:
                try:
                    chunk_embs = await embedding_func([content])
                    if chunk_embs and len(chunk_embs) > 0:
                        chunk_emb = np.array(chunk_embs[0], dtype=np.float32)
                        retriever._chunk_embedding_cache[source_id] = chunk_emb
                except Exception:
                    pass
            if chunk_emb is not None:
                q_norm = float(np.linalg.norm(query_embedding))
                c_norm = float(np.linalg.norm(chunk_emb))
                if q_norm > 0 and c_norm > 0:
                    semantic_score = float(query_embedding @ chunk_emb / (q_norm * c_norm))
        source_semantic_score[source_id] = semantic_score

        scored_chunks.append((source_id, order, rel_counts, ref_count, is_seed, is_vdb_path, source_type, content, semantic_score))

    # 排序：种子优先 → 语义相关性降序 → VDB order升序 → ref_count降序
    scored_chunks.sort(key=lambda x: (
        0 if x[4] else 1,
        -x[8],
        x[1],
        -x[3],
    ))

    top5 = scored_chunks[:5]
    logger.info(f"chunk排名完成：{len(scored_chunks)}个候选，"
                f"Top5: {[f'{s[0][:20]}:seed={s[4]},sem={s[8]:.3f},order={s[1]},ref={s[3]}' for s in top5]}")

    # ===== 阶段4：动态配额选择chunk =====
    score_distribution = [max(s[8], 1.0 / (s[1] + 1)) for s in scored_chunks]
    adaptive_limit = DualDimensionConfig.compute_adaptive_chunk_limit(
        len(scored_chunks), score_distribution
    )
    effective_limit = min(adaptive_limit, total_chunks_limit)
    logger.info(f"动态配额：自适应上限={adaptive_limit}，预算上限={total_chunks_limit}，有效上限={effective_limit}")

    seed_chunks = []
    vdb_chunks = []
    topo_chunks = []

    for source_id, order, rel_counts, ref_count, is_seed_chunk, is_vdb_path, source_type, content, sem_score in scored_chunks:
        chunk_tokens = 0
        if max_context_tokens is not None:
            chunk_tokens = len(encode_string_by_tiktoken(content))

        if is_seed_chunk or source_type == 'seed_entity' or source_type == 'seed_entity_one_hop':
            seed_chunks.append((source_id, content, source_type, chunk_tokens))
        elif is_vdb_path:
            if len(vdb_chunks) < effective_limit:
                vdb_chunks.append((source_id, content, source_type, chunk_tokens))
        else:
            if len(vdb_chunks) + len(topo_chunks) < effective_limit:
                topo_chunks.append((source_id, content, source_type, chunk_tokens))

    ordered_chunks = seed_chunks + vdb_chunks + topo_chunks

    for source_id, content, source_type, chunk_tokens in ordered_chunks:
        if max_context_tokens is not None:
            if total_tokens + chunk_tokens > max_context_tokens:
                break
            total_tokens += chunk_tokens

        related_chunks.append(content)
        if source_type not in source_types:
            source_types[source_type] = []
        source_types[source_type].append(source_id)

        if len(related_chunks) >= effective_limit:
            break

    seed_count = len(seed_chunks[:len(related_chunks)])
    vdb_count = len(vdb_chunks[:max(0, len(related_chunks) - seed_count)])
    topo_count = len(related_chunks) - seed_count - vdb_count
    logger.info(f"上下文收集完成：{len(related_chunks)} chunks, {total_tokens} tokens, "
                f"种子={seed_count}, VDB路径={vdb_count}, 拓扑补充={topo_count}")
    return related_chunks, source_types


@asynccontextmanager
async def virtual_node_scope(hsc, query_vertices):
    """虚拟节点生命周期管理器

    确保虚拟节点在查询结束后被正确清理，
    即使查询过程中发生异常也不会污染共享HSC。
    清理后标记矩阵需要重建，保证后续查询的Laplacian一致性。

    对于有代理节点的虚拟节点，将代理节点的coboundary复形中
    添加虚拟节点作为boundary成员，使拓扑扩散能触达这些复形。

    Args:
        hsc: 异质单纯复形
        query_vertices: 查询顶点列表（含虚拟节点信息）
    """
    virtual_node_ids = set()
    patched_simplices = set()
    try:
        for vertex in query_vertices:
            if vertex.get('is_virtual') and 'virtual_simplex_data' in vertex:
                virtual_id = vertex['id']
                hsc.nodes[virtual_id] = {'type': vertex.get('type', 'Entity'), 'vector': []}
                hsc.simplices[virtual_id] = vertex['virtual_simplex_data']
                virtual_node_ids.add(virtual_id)

                proxy_node_id = vertex.get('virtual_simplex_data', {}).get('proxy_node_id')
                if proxy_node_id and proxy_node_id in hsc.simplices:
                    proxy_coboundary = hsc.simplices[proxy_node_id].get('coboundary', [])
                    for cb_sid in proxy_coboundary:
                        if cb_sid in hsc.simplices:
                            cb_data = hsc.simplices[cb_sid]
                            cb_boundary = cb_data.get('boundary', [])
                            if virtual_id not in cb_boundary:
                                cb_boundary.append(virtual_id)
                                cb_data['boundary'] = cb_boundary
                                patched_simplices.add((cb_sid, 'boundary'))
                            cb_entities = cb_data.get('entities', [])
                            if virtual_id not in cb_entities:
                                cb_entities.append(virtual_id)
                                cb_data['entities'] = cb_entities
                                patched_simplices.add((cb_sid, 'entities'))
                    hsc.simplices[virtual_id]['coboundary'] = list(proxy_coboundary)

        if virtual_node_ids:
            logger.info(f"虚拟节点注入：{len(virtual_node_ids)} 个节点，"
                        f"修补 {len(patched_simplices)} 处拓扑连接")
        yield virtual_node_ids
    finally:
        for cb_sid, field in patched_simplices:
            if cb_sid in hsc.simplices:
                cb_data = hsc.simplices[cb_sid]
                for virtual_id in virtual_node_ids:
                    if field == 'boundary':
                        cb_data['boundary'] = [x for x in cb_data.get('boundary', []) if x != virtual_id]
                    elif field == 'entities':
                        cb_data['entities'] = [x for x in cb_data.get('entities', []) if x != virtual_id]
        for virtual_id in virtual_node_ids:
            hsc.nodes.pop(virtual_id, None)
            hsc.simplices.pop(virtual_id, None)
        if virtual_node_ids:
            hsc.B_matrices = {}
            hsc.L_matrices = {}
            if hasattr(hsc, '_retriever') and hsc._retriever is not None:
                for vid in virtual_node_ids:
                    hsc._retriever._simplex_embedding_cache.pop(vid, None)
            logger.info(f"虚拟节点清理：已移除 {len(virtual_node_ids)} 个节点，"
                        f"还原 {len(patched_simplices)} 处拓扑连接，矩阵缓存已失效")


def _iterative_coboundary_expand(hsc, seed_id, base_score, max_depth=None, decay=None):
    """从种子复形出发，沿coboundary逐层向上扩展到高维复形

    改进：使用配置参数控制衰减和深度，增加语义过滤。
    衰减系数从0.8降为0.5，使3层后得分仅剩0.125，大幅减少噪声。
    每层扩展后用查询嵌入做余弦相似度过滤，低于阈值的直接丢弃。

    Args:
        hsc: 异质单纯复形实例
        seed_id: 种子复形ID
        base_score: 种子复形的语义相似度得分
        max_depth: 最大扩展深度（None则使用配置值）
        decay: 每层衰减系数（None则使用配置值）

    Returns:
        dict: {simplex_id: {'score': float, 'source': str}} 扩展得到的高维复形
    """
    if max_depth is None:
        max_depth = DualDimensionConfig.COBOUNDARY_EXPAND_MAX_DEPTH_ENTITY
    if decay is None:
        decay = DualDimensionConfig.COBOUNDARY_EXPAND_DECAY

    expanded = {}
    current_frontier = {seed_id: base_score}
    visited = {seed_id}

    for depth in range(1, max_depth + 1):
        next_frontier = {}
        for sid, s_score in current_frontier.items():
            coboundary = hsc.simplices.get(sid, {}).get('coboundary', [])
            seed_dim = hsc.simplices.get(sid, {}).get('dimension', 0)
            for higher_id in coboundary:
                if higher_id in visited or higher_id not in hsc.simplices:
                    continue
                higher_dim = hsc.simplices[higher_id].get('dimension', 0)
                if higher_dim <= seed_dim:
                    continue
                decayed_score = s_score * decay
                if decayed_score >= DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                    existing = expanded.get(higher_id)
                    if existing is None or decayed_score > existing['score']:
                        expanded[higher_id] = {
                            'score': decayed_score,
                            'source': f'coboundary_depth_{depth}'
                        }
                    if higher_id not in next_frontier or decayed_score > next_frontier[higher_id]:
                        next_frontier[higher_id] = decayed_score
                    visited.add(higher_id)
        current_frontier = next_frontier
        if not current_frontier:
            break

    return expanded


async def _semantic_vector_retrieve(
    query: str,
    entities_vdb,
    relationships_vdb,
    hsc,
    top_k: int = 20
) -> dict:
    """基于向量数据库的语义内容维度检索

    使用实体和关系的向量数据库进行语义搜索，
    找到与查询语义相关的复形，作为语义内容维度的补充。

    改进：VDB命中低维复形后，通过多层迭代coboundary扩展
    逐级上升到高维复形（2维面、3维体等），弥补VDB仅存储
    0维实体和1维关系导致的语义覆盖不足。

    Args:
        query: 查询文本
        entities_vdb: 实体向量数据库
        relationships_vdb: 关系向量数据库
        hsc: 异质单纯复形
        top_k: 每个向量数据库返回的最大结果数

    Returns:
        {simplex_id: {'score': float, 'source': str}} 字典
    """
    semantic_results = {}

    if entities_vdb is not None:
        try:
            entity_results = await entities_vdb.query(query, top_k=top_k)
            if entity_results:
                for result in entity_results:
                    if isinstance(result, dict):
                        entity_id = result.get("id", "")
                        score = result.get("distance", 0)
                        entity_name = result.get("entity_name", "")
                    elif isinstance(result, (list, tuple)) and len(result) >= 2:
                        entity_id, score = result[0], result[1]
                        entity_name = ""
                    else:
                        continue

                    # VDB中存储的ID格式可能与HSC simplices的key格式不同
                    # 映射策略：1)直接ID 2)entity_name在HSC nodes中查找 3)哈希ID回退
                    matched_id = None
                    if entity_id and entity_id in hsc.simplices:
                        matched_id = entity_id
                    if not matched_id and entity_name:
                        normalized_name = normalize_entity_name(entity_name)
                        for node_id in hsc.nodes:
                            if normalize_entity_name(str(node_id)) == normalized_name:
                                matched_id = node_id
                                break
                    if not matched_id and entity_name:
                        normalized_name = normalize_entity_name(entity_name)
                        hsc_simplex_id = compute_mdhash_id(normalized_name, prefix="simplex-0-")
                        if hsc_simplex_id in hsc.simplices:
                            matched_id = hsc_simplex_id

                    if matched_id and score >= DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                        semantic_results[matched_id] = {
                            'score': score,
                            'source': 'entities_vdb'
                        }

                    # 多层迭代coboundary扩展：从0维实体逐级上升到高维复形
                    if matched_id and score >= DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                        expanded = _iterative_coboundary_expand(
                            hsc, matched_id, score,
                            max_depth=DualDimensionConfig.COBOUNDARY_EXPAND_MAX_DEPTH_ENTITY,
                            decay=DualDimensionConfig.COBOUNDARY_EXPAND_DECAY
                        )
                        for exp_id, exp_info in expanded.items():
                            if exp_id not in semantic_results:
                                semantic_results[exp_id] = exp_info
                            elif exp_info['score'] > semantic_results[exp_id]['score']:
                                semantic_results[exp_id] = exp_info
        except Exception as e:
            logger.warning(f"实体向量数据库语义检索失败: {e}")

    if relationships_vdb is not None:
        try:
            rel_results = await relationships_vdb.query(query, top_k=top_k)
            if rel_results:
                for result in rel_results:
                    if isinstance(result, dict):
                        rel_id = result.get("id", "")
                        score = result.get("distance", 0)
                        rel_entities = result.get("entities", [])
                    elif isinstance(result, (list, tuple)) and len(result) >= 2:
                        rel_id, score = result[0], result[1]
                        rel_entities = []
                    else:
                        continue

                    # VDB中关系ID格式可能与HSC simplices的key格式不同
                    # 映射策略：1)直接ID 2)通过关系实体在HSC中查找 3)哈希ID回退
                    matched_id = None
                    if rel_id and rel_id in hsc.simplices:
                        matched_id = rel_id
                    if not matched_id and rel_entities and isinstance(rel_entities, list):
                        normalized_ents = sorted([normalize_entity_name(e) for e in rel_entities])
                        for sid, sdata in hsc.simplices.items():
                            if sdata.get('dimension', 0) >= 1:
                                s_nodes = sdata.get('nodes', sdata.get('entities', []))
                                s_normalized = sorted([normalize_entity_name(n) for n in s_nodes])
                                if s_normalized == normalized_ents:
                                    matched_id = sid
                                    break
                    if not matched_id and rel_entities and isinstance(rel_entities, list):
                        normalized_ents = sorted([normalize_entity_name(e) for e in rel_entities])
                        for dim in range(len(normalized_ents) - 1, 0, -1):
                            candidate_id = compute_mdhash_id(str(normalized_ents), prefix=f"simplex-{dim}-")
                            if candidate_id in hsc.simplices:
                                matched_id = candidate_id
                                break

                    if matched_id and score >= DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                        if matched_id not in semantic_results or score > semantic_results[matched_id]['score']:
                            semantic_results[matched_id] = {
                                'score': score,
                                'source': 'relationships_vdb'
                            }

                    # 多层迭代coboundary扩展：从1维关系逐级上升到更高维复形
                    if matched_id and score >= DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                        expanded = _iterative_coboundary_expand(
                            hsc, matched_id, score,
                            max_depth=DualDimensionConfig.COBOUNDARY_EXPAND_MAX_DEPTH_RELATION,
                            decay=DualDimensionConfig.COBOUNDARY_EXPAND_DECAY
                        )
                        for exp_id, exp_info in expanded.items():
                            if exp_id not in semantic_results:
                                semantic_results[exp_id] = exp_info
                            elif exp_info['score'] > semantic_results[exp_id]['score']:
                                semantic_results[exp_id] = exp_info
        except Exception as e:
            logger.warning(f"关系向量数据库语义检索失败: {e}")

    logger.info(f"向量数据库语义检索：找到 {len(semantic_results)} 个语义相关复形")
    return semantic_results


async def topology_retrieval(
    query,
    simplex_storage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    global_config: dict,
):
    """MSG拓扑检索 - 2×2检索结构

    检索维度：
    |            | 实体节点              | MSG边                 |
    |------------|----------------------|----------------------|
    | 语义检索    | VDB查entity          | VDB查relationship    |
    | 拓扑检索    | L_entity扩散         | L_msg扩散            |

    流程：
    1. 从查询中提取实体和关系
    2. 构建异质单纯复形 + 加载二部图Laplacian
    3. 实体匹配
    4. 2×2并行检索
    5. 语义加权融合：score = α·semantic + (1-α)·topology
    6. 收集相关文本块

    Args:
        query: 查询文本
        simplex_storage: 复形存储
        entities_vdb: 实体向量数据库
        relationships_vdb: 关系向量数据库
        text_chunks_db: 文本块数据库
        global_config: 全局配置

    Returns:
        检索结果字典
    """
    max_simplices = global_config.get("max_simplices", 100)
    total_chunks_limit = global_config.get("max_topology_chunks", 50)

    # 步骤1和步骤2并行：提取实体和构建HSC互不依赖
    extract_task = asyncio.create_task(_extract_query_entities(query, global_config))
    hsc_task = asyncio.create_task(_build_hsc_from_storage(simplex_storage))

    extract_result = None
    hsc = None

    done, pending = await asyncio.wait(
        [extract_task, hsc_task],
        return_when=asyncio.ALL_COMPLETED
    )

    for task in done:
        if task is extract_task:
            try:
                extract_result = task.result()
            except Exception as e:
                logger.error(f"实体提取失败: {e}")
        elif task is hsc_task:
            try:
                hsc = task.result()
            except Exception as e:
                logger.error(f"HSC构建失败: {e}")

    if extract_result is None:
        logger.warning("实体提取失败，使用空结果")
        extracted_entities, extracted_relations = [], []
    else:
        extracted_entities, extracted_relations = extract_result

    if hsc is None:
        logger.error("HSC构建失败，无法继续检索")
        return {
            "ranked_simplices": [],
            "prompt_instructions": [],
            "related_chunks": [],
            "source_types": {},
            "simplex_counts": {},
            "structured_entities": [],
            "structured_simplices": [],
        }

    # 加载二部图Laplacian
    L_entity = None
    L_msg = None
    entity_index = None
    msg_index = None

    if simplex_storage is not None:
        L_entity = simplex_storage.load_laplacian("L_entity")
        L_msg = simplex_storage.load_laplacian("L_msg")
        entity_index = simplex_storage.load_index("entity_index")
        msg_index = simplex_storage.load_index("msg_index")

    if L_entity is None or L_msg is None:
        logger.info("二部图Laplacian未缓存，从HSC缓存中获取...")
        try:
            cached_data = await simplex_storage.get_cached_laplacians()
            L_entity = cached_data.get('L_entity')
            L_msg = cached_data.get('L_msg')
            entity_index = cached_data.get('entity_index')
            msg_index = cached_data.get('msg_index')
        except Exception as e:
            logger.warning(f"从缓存获取二部图Laplacian失败: {e}")

    # 步骤3：初始化检索器
    if not hasattr(hsc, '_retriever') or hsc._retriever is None:
        hsc._retriever = SimplicialRAGRetriever(hsc)
    retriever = hsc._retriever

    if not hasattr(hsc, 'B_matrices') or not hsc.B_matrices:
        if not retriever._laplacian_built:
            hsc.build_dynamic_incidence_matrices()
            hsc.compute_dynamic_hodge_laplacians()
            retriever._laplacian_built = True

    # 步骤4：实体匹配
    query_vertices, matched_entity_names, virtual_node_ids = await _match_entities_to_hsc(
        extracted_entities, hsc, simplex_storage, entities_vdb,
        embedding_func=global_config.get("embedding_func"),
        embedding_cache=retriever._simplex_embedding_cache
    )

    async with virtual_node_scope(hsc, query_vertices) as virtual_node_ids:
        # 构建查询关系
        query_partial_relations = []
        if extracted_relations:
            for relation in extracted_relations:
                relation_entities = relation.get('entities', [])
                matched_entities = []
                unmatched_entity_names = []

                for entity_name in relation_entities:
                    found = False
                    for vertex in query_vertices:
                        v_name = vertex.get('original_name', '')
                        v_id = vertex.get('id', '')
                        if v_name == entity_name or v_id == entity_name:
                            matched_entities.append(v_id)
                            found = True
                            break
                    if not found:
                        normalized_name = normalize_entity_name(entity_name)
                        for node_id in hsc.nodes.keys():
                            if isinstance(node_id, str) and normalize_entity_name(node_id) == normalized_name:
                                matched_entities.append(node_id)
                                found = True
                                break
                        if not found:
                            for vertex in query_vertices:
                                if vertex.get('is_virtual') and normalize_entity_name(vertex.get('original_name', '')) == normalized_name:
                                    matched_entities.append(vertex['id'])
                                    found = True
                                    break
                    if not found:
                        unmatched_entity_names.append(entity_name)

                relation_dimension = relation.get('dimension', len(relation_entities) - 1)
                match_ratio = len(matched_entities) / max(len(relation_entities), 1)

                if len(matched_entities) >= 2 or (relation_dimension == 1 and len(matched_entities) >= 1):
                    query_partial_relations.append({
                        'entities': matched_entities,
                        'description': relation.get('description', ''),
                        'dimension': relation_dimension,
                        'match_ratio': match_ratio
                    })
                elif len(matched_entities) >= 1 and match_ratio >= 0.3:
                    query_partial_relations.append({
                        'entities': matched_entities,
                        'description': relation.get('description', ''),
                        'dimension': relation_dimension,
                        'match_ratio': match_ratio,
                        'is_partial': True
                    })

        logger.info(f"Query vertices: {len(query_vertices)}, Query relations: {len(query_partial_relations)}")

        vertex_ids = [v['id'] for v in query_vertices if 'id' in v]

        # 虚拟节点降权
        real_vertices = [v for v in query_vertices if not v.get('is_virtual', False)]
        virtual_vertices = [v for v in query_vertices if v.get('is_virtual', False)]
        if virtual_vertices:
            logger.info(f"虚拟节点处理：{len(virtual_vertices)}个虚拟节点参与扩散但降权，"
                        f"{len(real_vertices)}个真实节点正常权重")
        for v in virtual_vertices:
            v['_virtual_weight_factor'] = 0.3

        # ===== 步骤5：2×2并行检索 =====
        # 定义融合权重α（语义权重）
        alpha = global_config.get("semantic_fusion_alpha", 0.5)
        logger.info(f"2×2检索融合权重：α(语义)={alpha:.2f}, 1-α(拓扑)={1-alpha:.2f}")

        # --- 5.1 语义×实体：VDB查entity ---
        async def _semantic_entity_retrieve():
            """语义检索实体节点：通过entities_vdb查询"""
            results = {}
            if entities_vdb is None:
                return results
            try:
                entity_results = await entities_vdb.query(query, top_k=DualDimensionConfig.PARALLEL_ENTITY_SEM_TOP_K)
                if entity_results:
                    for result in entity_results:
                        if isinstance(result, dict):
                            entity_name = result.get("entity_name", "")
                            score = result.get("distance", 0)
                        elif isinstance(result, (list, tuple)) and len(result) >= 2:
                            entity_name, score = result[0], result[1]
                        else:
                            continue
                        if not entity_name or score < DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                            continue

                        # 映射到HSC simplex
                        matched_id = None
                        normalized_name = normalize_entity_name(entity_name)
                        simplex_id = compute_mdhash_id(normalized_name, prefix="simplex-0-")
                        if simplex_id in hsc.simplices:
                            matched_id = simplex_id
                        elif entity_name in hsc.simplices:
                            matched_id = entity_name
                        else:
                            for node_id in hsc.nodes:
                                if normalize_entity_name(str(node_id)) == normalized_name:
                                    matched_id = node_id
                                    break

                        if matched_id:
                            results[matched_id] = {
                                'score': score,
                                'source': 'semantic_entity',
                                'dimension': 0,
                                'is_seed': matched_id in vertex_ids,
                            }

                            # coboundary扩展到MSG
                            expanded = _iterative_coboundary_expand(
                                hsc, matched_id, score,
                                max_depth=DualDimensionConfig.COBOUNDARY_EXPAND_MAX_DEPTH_ENTITY,
                                decay=DualDimensionConfig.COBOUNDARY_EXPAND_DECAY
                            )
                            for exp_id, exp_info in expanded.items():
                                if exp_id not in results:
                                    results[exp_id] = {
                                        'score': exp_info['score'],
                                        'source': 'semantic_entity_expand',
                                        'dimension': hsc.simplices.get(exp_id, {}).get('dimension', 0),
                                        'is_seed': exp_id in vertex_ids,
                                    }
            except Exception as e:
                logger.warning(f"语义×实体检索失败: {e}")
            return results

        # --- 5.2 语义×MSG：VDB查relationship ---
        async def _semantic_msg_retrieve():
            """语义检索MSG：通过relationships_vdb查询"""
            results = {}
            if relationships_vdb is None:
                return results
            try:
                rel_results = await relationships_vdb.query(query, top_k=DualDimensionConfig.PARALLEL_SEM_TOP_K)
                if rel_results:
                    for result in rel_results:
                        if isinstance(result, dict):
                            rel_id = result.get("id", "")
                            score = result.get("distance", 0)
                            rel_entities = result.get("id_set", [])
                        elif isinstance(result, (list, tuple)) and len(result) >= 2:
                            rel_id, score = result[0], result[1]
                            rel_entities = []
                        else:
                            continue

                        if score < DualDimensionConfig.SEMANTIC_MIN_SIMILARITY:
                            continue

                        # 映射到HSC simplex
                        matched_id = None
                        if rel_id in hsc.simplices:
                            matched_id = rel_id
                        elif rel_entities and isinstance(rel_entities, list):
                            normalized_ents = sorted([normalize_entity_name(e) for e in rel_entities])
                            for dim in range(len(normalized_ents) - 1, 0, -1):
                                candidate_id = compute_mdhash_id(str(normalized_ents), prefix=f"simplex-{dim}-")
                                if candidate_id in hsc.simplices:
                                    matched_id = candidate_id
                                    break

                        if not matched_id and rel_entities:
                            for sid, sdata in hsc.simplices.items():
                                if sdata.get('dimension', 0) >= 1:
                                    s_nodes = sdata.get('nodes', sdata.get('entities', []))
                                    s_normalized = sorted([normalize_entity_name(n) for n in s_nodes])
                                    if s_normalized == sorted([normalize_entity_name(e) for e in rel_entities]):
                                        matched_id = sid
                                        break

                        if matched_id:
                            results[matched_id] = {
                                'score': score,
                                'source': 'semantic_msg',
                                'dimension': hsc.simplices.get(matched_id, {}).get('dimension', 1),
                                'is_seed': matched_id in vertex_ids,
                            }

                            # coboundary扩展
                            expanded = _iterative_coboundary_expand(
                                hsc, matched_id, score,
                                max_depth=DualDimensionConfig.COBOUNDARY_EXPAND_MAX_DEPTH_RELATION,
                                decay=DualDimensionConfig.COBOUNDARY_EXPAND_DECAY
                            )
                            for exp_id, exp_info in expanded.items():
                                if exp_id not in results:
                                    results[exp_id] = {
                                        'score': exp_info['score'],
                                        'source': 'semantic_msg_expand',
                                        'dimension': hsc.simplices.get(exp_id, {}).get('dimension', 0),
                                        'is_seed': exp_id in vertex_ids,
                                    }
            except Exception as e:
                logger.warning(f"语义×MSG检索失败: {e}")
            return results

        # --- 5.3 拓扑×实体：L_entity扩散 ---
        async def _topology_entity_retrieve():
            """拓扑检索实体节点：L_entity扩散"""
            results = {}
            if L_entity is None or entity_index is None:
                logger.info("L_entity未就绪，跳过拓扑×实体检索")
                return results

            try:
                n = L_entity.shape[0]
                seed_vector = np.zeros(n, dtype=np.float32)
                seed_count = 0

                for vertex in query_vertices:
                    vid = vertex.get('id', '')
                    if not vid:
                        continue
                    normalized = normalize_entity_name(vid)
                    if normalized in entity_index:
                        idx = entity_index[normalized]
                        weight = 1.0
                        if vertex.get('is_virtual', False):
                            weight = 0.3
                        seed_vector[idx] = weight
                        seed_count += 1
                    elif vid in entity_index:
                        idx = entity_index[vid]
                        weight = 1.0
                        if vertex.get('is_virtual', False):
                            weight = 0.3
                        seed_vector[idx] = weight
                        seed_count += 1

                if seed_count == 0:
                    logger.info("无种子实体可进行L_entity扩散")
                    return results

                # 多步扩散：x_{t+1} = (I - β·L_entity) @ x_t
                beta = global_config.get("topology_diffusion_beta", 0.15)
                steps = global_config.get("topology_diffusion_steps", 3)
                x = seed_vector.copy()
                I = np.eye(n, dtype=np.float32)

                for _ in range(steps):
                    x = (I - beta * L_entity) @ x
                    x = np.clip(x, 0, None)

                # 收集扩散结果：自适应阈值替代固定0.01，避免大数据集噪声泛滥
                inv_entity_index = {v: k for k, v in entity_index.items()}
                positive_scores = x[x > 0]
                if len(positive_scores) > 10:
                    threshold = max(0.05, float(np.percentile(positive_scores, 75)))
                else:
                    threshold = 0.05
                logger.info(f"L_entity扩散自适应阈值：{threshold:.4f}（正得分节点数={len(positive_scores)}）")

                for idx in range(n):
                    if x[idx] > threshold:
                        entity_name = inv_entity_index.get(idx)
                        if entity_name:
                            # 映射到HSC simplex
                            simplex_id = compute_mdhash_id(entity_name, prefix="simplex-0-")
                            matched_id = simplex_id if simplex_id in hsc.simplices else None
                            if not matched_id and entity_name in hsc.simplices:
                                matched_id = entity_name

                            if matched_id:
                                results[matched_id] = {
                                    'score': float(x[idx]),
                                    'source': 'topology_entity',
                                    'dimension': 0,
                                    'is_seed': matched_id in vertex_ids,
                                }

                                # coboundary扩展到MSG
                                expanded = _iterative_coboundary_expand(
                                    hsc, matched_id, float(x[idx]),
                                    max_depth=DualDimensionConfig.COBOUNDARY_EXPAND_MAX_DEPTH_ENTITY,
                                    decay=DualDimensionConfig.COBOUNDARY_EXPAND_DECAY
                                )
                                for exp_id, exp_info in expanded.items():
                                    if exp_id not in results:
                                        results[exp_id] = {
                                            'score': exp_info['score'],
                                            'source': 'topology_entity_expand',
                                            'dimension': hsc.simplices.get(exp_id, {}).get('dimension', 0),
                                            'is_seed': exp_id in vertex_ids,
                                        }

                logger.info(f"L_entity扩散：{seed_count}个种子 → {len(results)}个实体/MSG")
            except Exception as e:
                logger.warning(f"拓扑×实体检索失败: {e}")
                import traceback
                traceback.print_exc()
            return results

        # --- 5.4 拓扑×MSG：L_msg扩散 ---
        async def _topology_msg_retrieve():
            """拓扑检索MSG：L_msg扩散"""
            results = {}
            if L_msg is None or msg_index is None:
                logger.info("L_msg未就绪，跳过拓扑×MSG检索")
                return results

            try:
                n = L_msg.shape[0]
                seed_vector = np.zeros(n, dtype=np.float32)
                seed_count = 0

                # 从种子实体找到所属MSG作为扩散种子
                for vertex in query_vertices:
                    vid = vertex.get('id', '')
                    if not vid:
                        continue
                    if vid in hsc.simplices:
                        coboundary = hsc.simplices[vid].get('coboundary', [])
                        weight = 1.0
                        if vertex.get('is_virtual', False):
                            weight = 0.3
                        for cb_id in coboundary:
                            if cb_id in msg_index:
                                idx = msg_index[cb_id]
                                seed_vector[idx] = max(seed_vector[idx], weight)
                                seed_count += 1

                # 也从查询关系中找MSG种子
                for relation in query_partial_relations:
                    rel_entities = relation.get('entities', [])
                    normalized_ents = sorted([normalize_entity_name(e) for e in rel_entities])
                    for dim in range(len(normalized_ents) - 1, 0, -1):
                        candidate_id = compute_mdhash_id(str(normalized_ents), prefix=f"simplex-{dim}-")
                        if candidate_id in msg_index:
                            idx = msg_index[candidate_id]
                            match_ratio = relation.get('match_ratio', 1.0)
                            seed_vector[idx] = max(seed_vector[idx], match_ratio)
                            seed_count += 1
                            break

                if seed_count == 0:
                    logger.info("无种子MSG可进行L_msg扩散")
                    return results

                # 多步扩散
                beta = global_config.get("topology_diffusion_beta", 0.15)
                steps = global_config.get("topology_diffusion_steps", 3)
                x = seed_vector.copy()
                I = np.eye(n, dtype=np.float32)

                for _ in range(steps):
                    x = (I - beta * L_msg) @ x
                    x = np.clip(x, 0, None)

                # 收集扩散结果：自适应阈值替代固定0.01
                inv_msg_index = {v: k for k, v in msg_index.items()}
                positive_scores = x[x > 0]
                if len(positive_scores) > 10:
                    threshold = max(0.05, float(np.percentile(positive_scores, 75)))
                else:
                    threshold = 0.05
                logger.info(f"L_msg扩散自适应阈值：{threshold:.4f}（正得分节点数={len(positive_scores)}）")

                for idx in range(n):
                    if x[idx] > threshold:
                        msg_id = inv_msg_index.get(idx)
                        if msg_id and msg_id in hsc.simplices:
                            results[msg_id] = {
                                'score': float(x[idx]),
                                'source': 'topology_msg',
                                'dimension': hsc.simplices[msg_id].get('dimension', 1),
                                'is_seed': msg_id in vertex_ids,
                            }

                logger.info(f"L_msg扩散：{seed_count}个种子 → {len(results)}个MSG")
            except Exception as e:
                logger.warning(f"拓扑×MSG检索失败: {e}")
                import traceback
                traceback.print_exc()
            return results

        # 并行执行2×2检索
        sem_entity_res, sem_msg_res, topo_entity_res, topo_msg_res = await asyncio.gather(
            _semantic_entity_retrieve(),
            _semantic_msg_retrieve(),
            _topology_entity_retrieve(),
            _topology_msg_retrieve(),
        )

        logger.info(f"2×2检索结果：语义×实体={len(sem_entity_res)}, 语义×MSG={len(sem_msg_res)}, "
                     f"拓扑×实体={len(topo_entity_res)}, 拓扑×MSG={len(topo_msg_res)}")

        # ===== 步骤5.5：语义加权融合 =====
        # 对每个simplex，合并四个通道的分数
        # score = α·semantic + (1-α)·topology
        # semantic = max(语义×实体, 语义×MSG)
        # topology = max(拓扑×实体, 拓扑×MSG)

        fused_simplices = {}

        all_simplex_ids = set()
        all_simplex_ids.update(sem_entity_res.keys())
        all_simplex_ids.update(sem_msg_res.keys())
        all_simplex_ids.update(topo_entity_res.keys())
        all_simplex_ids.update(topo_msg_res.keys())

        for sid in all_simplex_ids:
            sem_e = sem_entity_res.get(sid, {}).get('score', 0)
            sem_m = sem_msg_res.get(sid, {}).get('score', 0)
            topo_e = topo_entity_res.get(sid, {}).get('score', 0)
            topo_m = topo_msg_res.get(sid, {}).get('score', 0)

            semantic_score = max(sem_e, sem_m)
            topology_score = max(topo_e, topo_m)

            fused_score = alpha * semantic_score + (1 - alpha) * topology_score

            # 碰撞加成：语义和拓扑都命中的simplex获得额外权重
            has_semantic = sem_e > 0 or sem_m > 0
            has_topology = topo_e > 0 or topo_m > 0
            if has_semantic and has_topology:
                fused_score *= DualDimensionConfig.PARALLEL_COLLISION_BOOST

            # 确定最佳来源信息
            best_source = 'unknown'
            best_dim = 0
            for res_dict in [sem_entity_res, sem_msg_res, topo_entity_res, topo_msg_res]:
                if sid in res_dict:
                    info = res_dict[sid]
                    if info.get('source', 'unknown') != 'unknown':
                        best_source = info['source']
                    if info.get('dimension', 0) > best_dim:
                        best_dim = info['dimension']

            is_seed = False
            for res_dict in [sem_entity_res, sem_msg_res, topo_entity_res, topo_msg_res]:
                if sid in res_dict and res_dict[sid].get('is_seed', False):
                    is_seed = True
                    break

            sdata = hsc.simplices.get(sid, {})
            fused_simplices[sid] = {
                'simplex_id': sid,
                'fusion_score': fused_score,
                'semantic_score': semantic_score,
                'topology_score': topology_score,
                'fusion_type': 'collision' if (has_semantic and has_topology) else ('semantic_only' if has_semantic else 'topology_only'),
                'source': best_source,
                'dimension': best_dim,
                'is_seed': is_seed,
                'all_vertices': sdata.get('nodes', sdata.get('entities', [])),
                'description': sdata.get('description', ''),
                'entity_type': sdata.get('entity_type', sdata.get('type', '')),
                'importance': sdata.get('importance', 0.5),
                'frequency': sdata.get('frequency', 1),
                'source_id': sdata.get('source_id', sdata.get('source', '')),
                'coboundary': sdata.get('coboundary', []),
                'boundary': sdata.get('boundary', []),
                'layer': sdata.get('layer', ''),
                'is_maximal': sdata.get('is_maximal', False),
                'completeness': sdata.get('completeness', 0.75),
            }

        # 按融合分数排序
        sorted_simplices = sorted(fused_simplices.values(), key=lambda x: x['fusion_score'], reverse=True)

        # 维度配额截断
        dim_groups = defaultdict(list)
        for s in sorted_simplices:
            dim_groups[s.get('dimension', 0)].append(s)

        present_dims = sorted(dim_groups.keys(), reverse=True)
        if len(sorted_simplices) > max_simplices * 1.5:
            result = []
            remaining_quota = max_simplices
            high_dim_quota = int(max_simplices * 0.4)
            low_dim_quota = max_simplices - high_dim_quota

            for dim in present_dims:
                if dim >= 2:
                    quota = max(1, int(high_dim_quota / max(1, sum(1 for d in present_dims if d >= 2))))
                    taken = min(quota, len(dim_groups[dim]), remaining_quota)
                    result.extend(dim_groups[dim][:taken])
                    remaining_quota -= taken
                else:
                    quota = max(1, int(low_dim_quota / max(1, sum(1 for d in present_dims if d < 2))))
                    taken = min(quota, len(dim_groups[dim]), remaining_quota)
                    result.extend(dim_groups[dim][:taken])
                    remaining_quota -= taken

            if remaining_quota > 0:
                used_ids = {s.get('simplex_id') for s in result}
                leftovers = [s for s in sorted_simplices if s.get('simplex_id') not in used_ids]
                leftovers.sort(key=lambda x: x['fusion_score'], reverse=True)
                result.extend(leftovers[:remaining_quota])

            retrieved_simplices = result
        else:
            retrieved_simplices = sorted_simplices[:max_simplices]

        collision_count = sum(1 for s in retrieved_simplices if s.get('fusion_type') == 'collision')
        semantic_only_count = sum(1 for s in retrieved_simplices if s.get('fusion_type') == 'semantic_only')
        topology_only_count = sum(1 for s in retrieved_simplices if s.get('fusion_type') == 'topology_only')
        logger.info(f"融合结果：{len(retrieved_simplices)}个复形 "
                    f"(碰撞={collision_count}, 语义独有={semantic_only_count}, 拓扑独有={topology_only_count})")

        # 步骤7：生成提示指令
        prompt_instructions = []

        entity_details_for_prompt = []
        for v in query_vertices:
            vid = v.get('id', '')
            if not vid:
                continue
            vtype = v.get('type', '')
            if vid in hsc.simplices:
                hsc_data = hsc.simplices[vid]
                if not vtype:
                    vtype = hsc_data.get('entity_type', hsc_data.get('type', ''))
            type_str = f" [{vtype}]" if vtype else ""
            entity_details_for_prompt.append(f"{vid}{type_str}")
        if entity_details_for_prompt:
            prompt_instructions.append(
                f"Matched entities: {'; '.join(entity_details_for_prompt)}"
            )

        relation_descriptions = []
        for rel in query_partial_relations:
            entities_str = ', '.join(rel.get('entities', []))
            if entities_str:
                relation_descriptions.append(f"({entities_str})")
        if relation_descriptions:
            prompt_instructions.append(
                f"Key relationships: {'; '.join(relation_descriptions)}. "
                f"Verify all claims against Sources."
            )

        # 步骤8：收集文本块
        ll_keywords_str = ", ".join([
            v.get('original_name', v.get('id', ''))
            for v in query_vertices
            if isinstance(v, dict) and not v.get('is_virtual')
        ]).upper()
        hl_keywords_str = ", ".join([
            f"<hyperedge>{rel.get('description', '')}"
            for rel in query_partial_relations
            if rel.get('description')
        ])

        max_context_tokens = global_config.get("max_context_tokens", DualDimensionConfig.MAX_CONTEXT_TOKENS)
        _embedding_func = global_config.get("embedding_func")
        related_chunks, source_types = await _collect_text_chunks(
            retrieved_simplices, text_chunks_db, total_chunks_limit, max_context_tokens, query,
            entity_count=len(query_vertices),
            relation_count=len(query_partial_relations),
            hsc=hsc,
            retriever=retriever,
            embedding_func=_embedding_func,
            query_vertices=query_vertices,
            simplex_storage=simplex_storage,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            ll_keywords=ll_keywords_str,
            hl_keywords=hl_keywords_str,
        )

        # 回退：如果拓扑收集为空，通过种子实体直接收集
        if not related_chunks:
            logger.info("拓扑收集为空，尝试通过种子实体回退收集所有关联chunk")
            try:
                for vertex in query_vertices:
                    vid = vertex.get('id', '')
                    if not vid or vid not in hsc.simplices:
                        continue
                    v_simplices = await simplex_storage.get_simplices_by_entity(vid)
                    for simplex_id, simplex_data in v_simplices:
                        source_list = simplex_data.get('source_id') or simplex_data.get('source')
                        if source_list:
                            if isinstance(source_list, list):
                                source_ids_list = source_list
                            elif isinstance(source_list, str):
                                source_ids_list = [s.strip() for s in source_list.split('<SEP>') if s.strip()]
                            else:
                                source_ids_list = []
                            for src_id in source_ids_list:
                                chunk_data = await text_chunks_db.get_by_id(src_id)
                                if chunk_data and "content" in chunk_data:
                                    content = chunk_data["content"]
                                    if content not in related_chunks:
                                        related_chunks.append(content)
                        if len(related_chunks) >= total_chunks_limit:
                            break
                    if len(related_chunks) >= total_chunks_limit:
                        break
                if related_chunks:
                    logger.info(f"种子实体回退：收集到 {len(related_chunks)} 个chunk")
            except Exception as e:
                logger.debug(f"种子实体回退收集失败: {e}")

        simplex_counts = defaultdict(int)
        for s in retrieved_simplices:
            simplex_counts[s.get('dimension', 0)] += 1

        seed_count = sum(1 for s in retrieved_simplices if s.get('is_seed', False))

        logger.info(
            f"Topology retrieval completed: {len(retrieved_simplices)} simplices "
            f"(including {seed_count} seed), {len(related_chunks)} chunks"
        )

        structured_entities = []
        seen_entity_ids = set()
        for s in retrieved_simplices:
            if s.get('dimension', 0) == 0 or s.get('is_seed', False):
                entity_id = s.get('simplex_id', '')
                if entity_id and entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    entity_names = s.get('all_vertices', s.get('entities', []))
                    entity_name = entity_names[0] if isinstance(entity_names, list) and entity_names else entity_id
                    structured_entities.append({
                        'name': entity_name,
                        'type': s.get('entity_type', s.get('type', 'Entity')),
                        'description': s.get('description', ''),
                        'is_seed': s.get('is_seed', False)
                    })

        structured_simplices = []
        for s in retrieved_simplices:
            dim = s.get('dimension', 0)
            if dim >= 1:
                structured_simplices.append({
                    'entities': s.get('all_vertices', s.get('entities', [])),
                    'dimension': dim,
                    'description': s.get('description', ''),
                    'is_seed': s.get('is_seed', False),
                    'layer': s.get('layer', ''),
                    'is_maximal': s.get('is_maximal', False),
                    'completeness': s.get('completeness', 0.75),
                })

    return {
        "ranked_simplices": [(s.get('simplex_id'), s) for s in retrieved_simplices],
        "prompt_instructions": prompt_instructions,
        "related_chunks": related_chunks,
        "source_diversity": source_types,
        "simplex_counts": dict(simplex_counts),
        "structured_entities": structured_entities,
        "structured_simplices": structured_simplices
    }


def combine_contexts(relation_context, entity_context):
    """合并关系上下文和实体上下文"""

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        simplices_match = re.search(
            r"-----Simplices-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        simplices = simplices_match.group(1) if simplices_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, simplices, sources

    if relation_context is None:
        warnings.warn(
            "High Level context is None. Return empty High_Level entity/simplex/source"
        )
        hl_entities, hl_simplices, hl_sources = "", "", ""
    else:
        hl_entities, hl_simplices, hl_sources = extract_sections(relation_context)

    if entity_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low_Level entity/simplex/source"
        )
        ll_entities, ll_simplices, ll_sources = "", "", ""
    else:
        ll_entities, ll_simplices, ll_sources = extract_sections(entity_context)

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_simplices = process_combine_contexts(
        hl_simplices, ll_simplices
    )
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Simplices-----
```csv
{combined_simplices}
```
-----Sources-----
```csv
{combined_sources}
```
"""

def remove_after_sources(input_string: str) -> str:
    """删除字符串中 '-----Sources-----' 及其之后的所有内容"""
    index = input_string.find("-----Sources-----")
    if index != -1:
        return input_string[:index]
    return input_string
