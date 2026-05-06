import asyncio
import json
import re
import os
from collections import defaultdict
from typing import List, Dict, Set, Optional

import numpy as np

from ..utils import logger, compute_mdhash_id
from ..base import BaseKVStorage, TextChunkSchema
from ..prompt import GRAPH_FIELD_SEP, PROMPTS
from ._config import DualDimensionConfig, EMB_MODEL, EMB_API_KEY, EMB_BASE_URL, semantic_similarity
from ._simplicial_complex import get_simplex_entities

async def semantic_verification(
    candidates,
    simplex_storage,
    text_chunks_db,
    global_config,
):
    """
    Semantic verification and status labeling - Balanced version
    1. Void and face determination with balanced threshold
    2. Topological label storage
    3. Combine quality score and LLM verification
    """
    use_llm_func = global_config.get("llm_model_func")
    verified_candidates = []
    
    # 合理的验证数量
    max_verification = global_config.get("max_semantic_verification", 10000)
    
    # 并行计算候选者质量分数
    async def score_candidate(candidate):
        entities = candidate.get("entities", [])
        if len(entities) < 3:
            return None
        
        # 计算候选者质量分数
        quality_score = await _calculate_candidate_quality(entities, simplex_storage)
        return (candidate, quality_score)
    
    # 并行处理候选者
    score_tasks = []
    for candidate in candidates:
        task = asyncio.create_task(score_candidate(candidate))
        score_tasks.append(task)
    
    score_results = await asyncio.gather(*score_tasks, return_exceptions=True)
    
    # 过滤结果
    scored_candidates = []
    for result in score_results:
        if isinstance(result, Exception):
            logger.error(f"Error scoring candidate: {result}")
            continue
        if result is not None:
            scored_candidates.append(result)
    
    # 按质量分数降序排序，优先验证高质量候选者
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 限制验证数量
    candidates_to_verify = scored_candidates[:max_verification]
    
    logger.info(f"Semantic verification: {len(candidates_to_verify)} candidates selected from {len(scored_candidates)} total")
    
    # 并行验证候选者
    async def verify_candidate(candidate, quality_score):
        entities = candidate.get("entities", [])
        
        # 预验证：检查实体间是否存在直接关联
        pre_verification_passed = await _pre_verify_candidate(entities, simplex_storage)
        
        is_verified = False
        verification_reason = ""
        
        # 策略：高质量分数直接通过，中等质量需要LLM验证，低质量直接跳过
        if quality_score >= 0.6:
            # 高质量：直接标记为验证通过
            is_verified = True
            verification_reason = f"High quality score: {quality_score:.3f}"
        elif quality_score >= 0.3 and pre_verification_passed:
            # 中等质量且预验证通过：需要LLM验证
            if use_llm_func:
                # 并行获取相关文本块
                async def get_related_chunks(entity):
                    entity_chunks = []
                    entity_simplices = await simplex_storage.get_simplices_by_entity(entity)
                    for simplex_id, simplex_data in entity_simplices:
                        source_id = simplex_data.get("source_id")
                        if source_id:
                            chunk_data = await text_chunks_db.get_by_id(source_id)
                            if chunk_data and "content" in chunk_data:
                                entity_chunks.append(chunk_data["content"])
                    return entity_chunks
                
                entity_chunk_tasks = []
                for entity in entities:
                    task = asyncio.create_task(get_related_chunks(entity))
                    entity_chunk_tasks.append(task)
                
                entity_chunk_results = await asyncio.gather(*entity_chunk_tasks, return_exceptions=True)
                
                related_chunks = []
                for result in entity_chunk_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error getting related chunks: {result}")
                        continue
                    related_chunks.extend(result)
                
                related_chunks = list(set(related_chunks))
                
                if len(related_chunks) >= 1:
                    context = "\n".join(related_chunks[:2])
                    
                    # 构建带上下文的实体组评估提示词
                    prompt = PROMPTS["entity_group_evaluation_with_context"].format(entities=entities, context=context)
                    
                    try:
                        result = await use_llm_func(prompt)
                        if result and "yes" in result.lower():
                            is_verified = True
                            verification_reason = f"LLM verified, quality score: {quality_score:.3f}"
                        else:
                            verification_reason = f"LLM said No, quality score: {quality_score:.3f}"
                    except Exception:
                        # LLM验证失败时，基于质量分数决定
                        if quality_score >= 0.4:
                            is_verified = True
                            verification_reason = f"LLM failed, medium quality score: {quality_score:.3f}"
                        else:
                            verification_reason = f"LLM failed and low quality score: {quality_score:.3f}"
                else:
                    # 没有上下文时，基于质量分数决定
                    if quality_score >= 0.4:
                        is_verified = True
                        verification_reason = f"No context, medium quality score: {quality_score:.3f}"
                    else:
                        verification_reason = f"No context and low quality score: {quality_score:.3f}"
            else:
                # 没有LLM函数时，基于质量分数
                if quality_score >= 0.4:
                    is_verified = True
                    verification_reason = f"No LLM, medium quality score: {quality_score:.3f}"
                else:
                    verification_reason = f"No LLM and low quality score: {quality_score:.3f}"
        else:
            # 低质量或预验证失败
            verification_reason = f"Low quality or pre-verification failed: {quality_score:.3f}"
        
        # Create or update simplex
        simplex_id = compute_mdhash_id(str(sorted(entities)), prefix=f"simplex-{len(entities)-1}-")
        simplex_data = {
            "type": "simplex",
            "dimension": len(entities) - 1,
            "entities": entities,
            "source": candidate.get("source", "semantic_verification"),
            "verification_reason": verification_reason,
            "quality_score": quality_score,
        }
        
        # Store simplex
        await simplex_storage.upsert_simplex(simplex_id, simplex_data)
        return simplex_data, is_verified
    
    # 并行验证候选者
    verify_tasks = []
    for candidate, quality_score in candidates_to_verify:
        task = asyncio.create_task(verify_candidate(candidate, quality_score))
        verify_tasks.append(task)
    
    verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)
    
    verified_count = 0
    unverified_count = 0
    
    # 处理验证结果
    for result in verify_results:
        if isinstance(result, Exception):
            logger.error(f"Error verifying candidate: {result}")
            continue
        if result is not None:
            simplex_data, is_verified = result
            verified_candidates.append(simplex_data)
            if is_verified:
                verified_count += 1
            else:
                unverified_count += 1
    
    logger.info(f"Semantic verification completed: {len(verified_candidates)} candidates verified, "
                f"{verified_count} passed, {unverified_count} rejected")
    return verified_candidates

async def _calculate_candidate_quality(entities, simplex_storage):
    """计算候选者质量分数"""
    if len(entities) < 2:
        return 0.0
    
    # 并行获取每个实体的来源
    async def get_entity_sources(entity):
        simplices = await simplex_storage.get_simplices_by_entity(entity)
        sources = set()
        for _, data in simplices:
            if data.get("source_id"):
                sources.add(data["source_id"])
        return entity, sources
    
    # 并行获取所有实体的来源
    source_tasks = []
    for entity in entities:
        task = asyncio.create_task(get_entity_sources(entity))
        source_tasks.append(task)
    
    source_results = await asyncio.gather(*source_tasks, return_exceptions=True)
    
    # 构建实体到来源的映射
    entity_to_sources = {}
    for result in source_results:
        if isinstance(result, Exception):
            logger.error(f"Error getting entity sources: {result}")
            continue
        if result is not None:
            entity, sources = result
            entity_to_sources[entity] = sources
    
    # 基于实体间的连接数计算质量分数
    connection_count = 0
    
    # 并行计算实体对的连接数
    async def calculate_connection(entity1, entity2):
        sources1 = entity_to_sources.get(entity1, set())
        sources2 = entity_to_sources.get(entity2, set())
        common_sources = sources1.intersection(sources2)
        return len(common_sources)
    
    # 生成所有实体对
    entity_pairs = []
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            entity_pairs.append((entity1, entity2))
    
    # 并行计算所有实体对的连接数
    connection_tasks = []
    for entity1, entity2 in entity_pairs:
        task = asyncio.create_task(calculate_connection(entity1, entity2))
        connection_tasks.append(task)
    
    connection_results = await asyncio.gather(*connection_tasks, return_exceptions=True)
    
    # 汇总连接数
    for result in connection_results:
        if isinstance(result, Exception):
            logger.error(f"Error calculating connection: {result}")
            continue
        if isinstance(result, int):
            connection_count += result
    
    # 归一化分数
    max_possible_connections = len(entities) * (len(entities) - 1) / 2
    quality_score = min(connection_count / max(max_possible_connections, 1), 1.0)
    
    # 为高维单纯形提供额外的质量分数加成
    # 维度越高，加成越多，鼓励使用更高维度的结构
    dimension = len(entities) - 1
    dimension_bonus = dimension * 0.1
    quality_score = min(quality_score + dimension_bonus, 1.0)
    
    return quality_score


async def _pre_verify_candidate(entities, simplex_storage):
    """快速预验证候选者"""
    if len(entities) < 2:
        return False
    
    # 并行获取每个实体的来源
    async def get_entity_sources(entity):
        simplices = await simplex_storage.get_simplices_by_entity(entity)
        sources = set()
        for _, data in simplices:
            if data.get("source_id"):
                sources.add(data["source_id"])
        return entity, sources
    
    # 并行获取所有实体的来源
    source_tasks = []
    for entity in entities:
        task = asyncio.create_task(get_entity_sources(entity))
        source_tasks.append(task)
    
    source_results = await asyncio.gather(*source_tasks, return_exceptions=True)
    
    # 构建实体到来源的映射
    entity_to_sources = {}
    for result in source_results:
        if isinstance(result, Exception):
            logger.error(f"Error getting entity sources: {result}")
            continue
        if result is not None:
            entity, sources = result
            entity_to_sources[entity] = sources
    
    # 检查是否存在共同的来源
    async def check_common_sources(entity1, entity2):
        sources1 = entity_to_sources.get(entity1, set())
        sources2 = entity_to_sources.get(entity2, set())
        common_sources = sources1.intersection(sources2)
        return len(common_sources) > 0
    
    # 生成所有实体对
    entity_pairs = []
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            entity_pairs.append((entity1, entity2))
    
    # 并行检查所有实体对的共同来源
    check_tasks = []
    for entity1, entity2 in entity_pairs:
        task = asyncio.create_task(check_common_sources(entity1, entity2))
        check_tasks.append(task)
    
    check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
    
    # 检查是否有任何实体对有共同来源
    for result in check_results:
        if isinstance(result, Exception):
            logger.error(f"Error checking common sources: {result}")
            continue
        if result:
            return True
    
    return False




