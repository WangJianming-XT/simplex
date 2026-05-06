import asyncio
import re
from collections import defaultdict, deque
from typing import List, Dict, Set, Union

import numpy as np

from ..utils import logger
from ..llm import openai_embedding
from ._config import DualDimensionConfig, EMB_MODEL, EMB_API_KEY, EMB_BASE_URL, normalize_entity_name
from ._simplicial_complex import HeterogeneousSimplicialComplex, get_simplex_entities, calculate_simplex_score


async def compute_semantic_similarity(query_text: str, simplex_texts: list, simplex_ids: list) -> dict:
    """计算查询文本与复形文本的语义相似度

    Args:
        query_text: 查询文本
        simplex_texts: 复形文本列表
        simplex_ids: 复形ID列表（与simplex_texts对应）

    Returns:
        dict: 复形ID到相似度的映射
    """
    if not query_text or not simplex_texts or not simplex_ids:
        logger.warning("compute_semantic_similarity: Invalid input parameters")
        return {}

    try:
        max_retries = 3
        retry_delay = 2
        embeddings = None

        for attempt in range(max_retries):
            try:
                logger.info(f"尝试计算嵌入 (第 {attempt+1}/{max_retries} 次)")
                embeddings = await openai_embedding(
                    [query_text] + simplex_texts,
                    model=EMB_MODEL,
                    api_key=EMB_API_KEY,
                    base_url=EMB_BASE_URL
                )
                logger.info(f"嵌入计算成功，嵌入数量: {len(embeddings)}")
                break
            except Exception as e:
                logger.error(f"计算嵌入失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)

        if embeddings is not None and len(embeddings) > 0:
            try:
                query_embedding = embeddings[0]
                simplex_embeddings = embeddings[1:]

                logger.info(f"开始计算余弦相似度，复形嵌入数: {len(simplex_embeddings)}")
                similarities = {}
                for i, simplex_id in enumerate(simplex_ids):
                    if i < len(simplex_embeddings):
                        q_emb = np.asarray(query_embedding, dtype=np.float32).ravel()
                        s_emb = np.asarray(simplex_embeddings[i], dtype=np.float32).ravel()
                        query_norm = float(np.linalg.norm(q_emb))
                        simplex_norm = float(np.linalg.norm(s_emb))
                        if query_norm > 0 and simplex_norm > 0:
                            similarity = float(np.dot(q_emb, s_emb) / (query_norm * simplex_norm))
                            similarities[simplex_id] = similarity
                        else:
                            logger.warning(f"compute_semantic_similarity: Zero norm detected for simplex {simplex_id}")
                            similarities[simplex_id] = 0.0

                logger.info(f"余弦相似度计算完成，共 {len(similarities)} 个复形")
                return similarities

            except Exception as e:
                logger.error(f"计算余弦相似度失败: {e}")
                return {}
        else:
            logger.warning("compute_semantic_similarity: All retries failed, returning empty result")
            return {}

    except Exception as e:
        logger.error(f"compute_semantic_similarity: Unexpected error: {e}")
        return {}


class SimplicialRAGRetriever:
    """基于双维度并行的拓扑检索器

    实现"分发 - 扩散 - 对撞 - 补全"的四步走逻辑：
    - 分发：从查询中提取语义顶点（维度A）和结构模式（维度B）
    - 扩散：基于Hodge Laplacian进行拓扑能量扩散
    - 对撞：两个维度的能量在公共复形上重合时确认相关性
    - 补全：通过拓扑链补全和维度提升获取完整上下文

    双维度融合：
    - 拓扑结构维度：基于图谱结构关系、连接方式及层级组织
    - 语义内容维度：基于文本语义相似度、上下文理解及概念关联
    """

    def __init__(self, hsc: HeterogeneousSimplicialComplex):
        self.hsc = hsc
        self._sub_complex_cache = {}
        self._entity_inverted_index = None
        self._index_built = False
        self._simplex_embedding_cache = {}
        self._simplex_embedding_version = 0
        self._simplex_text_cache = {}
        self._query_embedding_cache = {}
        self._chunk_embedding_cache = {}
        self._adjacency_cache = None
        self._adjacency_cache_version = 0
        self._laplacian_built = False
        self._preload_persistent_embeddings()

    def _preload_persistent_embeddings(self):
        """初始化时预加载所有持久化嵌入向量到内存缓存

        从HSC复形数据的embedding字段批量加载到_simplex_embedding_cache，
        避免语义检索时逐个查找和重复API调用。
        仅加载有效嵌入（非None且维度正确），跳过无嵌入的复形。
        """
        loaded = 0
        skipped = 0
        for simplex_id, simplex_data in self.hsc.simplices.items():
            stored_embedding = simplex_data.get('embedding')
            if stored_embedding is not None:
                try:
                    emb_array = np.array(stored_embedding, dtype=np.float32)
                    if emb_array.ndim == 1 and len(emb_array) > 0:
                        self._simplex_embedding_cache[simplex_id] = emb_array
                        loaded += 1
                    else:
                        skipped += 1
                except (ValueError, TypeError):
                    skipped += 1
            else:
                skipped += 1
        logger.info(f"预加载持久化嵌入：成功 {loaded} 个，跳过 {skipped} 个（无嵌入或格式异常）")

    def _build_simplex_text_cache(self):
        """构建复形文本缓存，避免每次查询重复构建文本列表

        缓存覆盖所有维度的复形（dim≥0），包括实体复形。
        实体复形的description可能包含关键语义信息，
        仅跳过无文本内容的复形。
        """
        current_total_count = len(self.hsc.simplices)
        if self._simplex_text_cache and len(self._simplex_text_cache) == current_total_count:
            return
        self._simplex_text_cache = {}
        for simplex_id, simplex_data in self.hsc.simplices.items():
            entities = simplex_data.get('entities', simplex_data.get('nodes', []))
            description = simplex_data.get('description', '')
            text = ' '.join(entities) if isinstance(entities, list) else str(entities)
            if description:
                text += ' ' + description
            if text.strip():
                self._simplex_text_cache[simplex_id] = text

    async def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """获取查询文本的嵌入向量（带缓存，LRU淘汰）

        同一查询文本在单次检索流程中可能被多次使用，
        缓存后只需调用一次嵌入API，消除重复计算。
        当缓存超过上限时淘汰最早的条目，防止内存无界增长。

        Args:
            query_text: 查询文本

        Returns:
            查询文本的嵌入向量
        """
        cache_key = query_text.strip()
        if cache_key in self._query_embedding_cache:
            return self._query_embedding_cache[cache_key]
        # LRU淘汰：缓存超过100条时清除最早的一半
        if len(self._query_embedding_cache) > 100:
            keys_to_remove = list(self._query_embedding_cache.keys())[:50]
            for k in keys_to_remove:
                del self._query_embedding_cache[k]
            logger.info(f"查询嵌入缓存LRU淘汰：移除 {len(keys_to_remove)} 条，"
                        f"剩余 {len(self._query_embedding_cache)} 条")
        try:
            embeddings = await openai_embedding(
                [query_text], model=EMB_MODEL, api_key=EMB_API_KEY, base_url=EMB_BASE_URL
            )
            if embeddings is not None and len(embeddings) > 0:
                self._query_embedding_cache[cache_key] = embeddings[0]
                return embeddings[0]
        except Exception as e:
            logger.error(f"查询嵌入计算失败: {e}")
        return None

    async def _get_simplex_embeddings_batch(self, simplex_ids: list, simplex_texts: list) -> dict:
        """批量获取复形嵌入向量（仅使用缓存，不调用API）

        嵌入向量在初始化时已从持久化文件预加载到内存缓存。
        无持久化嵌入的复形直接跳过，避免运行时API调用导致限流。
        仅当显式调用_compute_missing_embeddings时才会补充计算缺失嵌入。

        Args:
            simplex_ids: 复形ID列表
            simplex_texts: 复形文本列表（未使用，保留接口兼容）

        Returns:
            {simplex_id: embedding_vector} 字典
        """
        result = {}
        missed = 0

        for simplex_id in simplex_ids:
            if simplex_id in self._simplex_embedding_cache:
                result[simplex_id] = self._simplex_embedding_cache[simplex_id]
            else:
                missed += 1

        if missed > 0:
            logger.info(f"复形嵌入缓存：命中 {len(result)} 个，跳过 {missed} 个（无持久化嵌入）")

        return result

    async def get_chunk_embeddings_batch(self, chunk_ids: list, chunk_texts: list, embedding_func=None) -> dict:
        """批量获取chunk嵌入向量（优先使用缓存，缓存未命中时调用API并缓存结果）

        与_get_simplex_embeddings_batch不同，chunk嵌入不在初始化时预加载，
        而是按需计算并缓存。同一chunk在多次查询间共享缓存，
        避免重复调用嵌入API。

        Args:
            chunk_ids: chunk ID列表
            chunk_texts: chunk文本列表（缓存未命中时用于API调用）
            embedding_func: 嵌入函数（缓存未命中时必须提供）

        Returns:
            {chunk_id: embedding_vector} 字典
        """
        result = {}
        uncached_ids = []
        uncached_texts = []
        uncached_indices = []

        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id in self._chunk_embedding_cache:
                result[chunk_id] = self._chunk_embedding_cache[chunk_id]
            else:
                uncached_ids.append(chunk_id)
                uncached_texts.append(chunk_texts[i])
                uncached_indices.append(i)

        if uncached_ids and embedding_func is not None:
            # 分批调用嵌入API，每批最多10个文本，避免触发429限流
            batch_size = 10
            all_embeddings = {}
            for batch_start in range(0, len(uncached_ids), batch_size):
                batch_ids = uncached_ids[batch_start:batch_start + batch_size]
                batch_texts = uncached_texts[batch_start:batch_start + batch_size]
                try:
                    batch_embeddings = await embedding_func(batch_texts)
                    if batch_embeddings is not None and len(batch_embeddings) == len(batch_ids):
                        for j, emb in enumerate(batch_embeddings):
                            emb_array = np.array(emb, dtype=np.float32)
                            if emb_array.ndim == 1 and len(emb_array) > 0:
                                self._chunk_embedding_cache[batch_ids[j]] = emb_array
                                all_embeddings[batch_ids[j]] = emb_array
                except Exception as e:
                    logger.warning(f"chunk嵌入批量计算失败(批次{batch_start//batch_size+1}): {e}")
                    break
            result.update(all_embeddings)

        if len(result) < len(chunk_ids):
            logger.info(f"chunk嵌入缓存：命中 {len(result)} 个，未命中 {len(chunk_ids) - len(result)} 个")

        return result

    def clear_embedding_cache(self):
        """清除所有缓存（数据更新时调用）"""
        self._simplex_embedding_cache.clear()
        self._simplex_text_cache.clear()
        self._query_embedding_cache.clear()
        self._chunk_embedding_cache.clear()
        self._simplex_embedding_version += 1
        self._adjacency_cache = None
        self._adjacency_cache_version = 0
        self._laplacian_built = False

    def _build_entity_inverted_index(self):
        """构建实体到单纯形的倒排索引，加速维度B的关系匹配

        使用统一的normalize_entity_name进行标准化，
        解决extraction用大写、retrieval用小写导致匹配失败的问题。
        """
        if self._index_built:
            return
        self._entity_inverted_index = defaultdict(dict)
        for simplex_id, simplex_data in self.hsc.simplices.items():
            entities = simplex_data.get('entities', [])
            for entity in entities:
                normalized = normalize_entity_name(entity)
                if normalized:
                    self._entity_inverted_index[normalized][simplex_id] = simplex_data
        self._index_built = True
        logger.info(f"实体倒排索引构建完成，共 {len(self._entity_inverted_index)} 个实体条目")

    def _lookup_simplices_by_entity(self, entity_name: str) -> Dict[str, dict]:
        """通过倒排索引快速查找包含指定实体的所有单纯形

        增强模糊匹配：排除语义相反的实体对（如SECURED/UNSECURED），
        避免子串包含关系导致的错误匹配。

        Args:
            entity_name: 实体名称

        Returns:
            {simplex_id: simplex_data} 字典
        """
        if not self._index_built:
            self._build_entity_inverted_index()
        normalized = normalize_entity_name(entity_name)
        exact_matches = self._entity_inverted_index.get(normalized, {})
        if exact_matches:
            return exact_matches
        fuzzy_matches = {}
        min_substr_len = DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH
        fuzzy_min_ratio = DualDimensionConfig.FUZZY_MATCH_MIN_RATIO
        for idx_entity, simplices in self._entity_inverted_index.items():
            if len(normalized) >= min_substr_len and normalized in idx_entity and len(idx_entity) > 0:
                match_ratio = len(normalized) / len(idx_entity)
                if match_ratio >= fuzzy_min_ratio:
                    fuzzy_matches.update(simplices)
            elif len(idx_entity) >= min_substr_len and idx_entity in normalized and len(normalized) > 0:
                match_ratio = len(idx_entity) / len(normalized)
                if match_ratio >= fuzzy_min_ratio:
                    fuzzy_matches.update(simplices)
        return fuzzy_matches

    def _entity_match(self, query_entity: str, simplex_entity: str) -> bool:
        """检查查询实体是否与复形中的实体匹配

        匹配策略：精确匹配 > 词边界子串匹配 > 词集合子集匹配 > 模糊匹配
        使用统一的normalize_entity_name进行标准化

        Args:
            query_entity: 查询实体名称
            simplex_entity: 复形中的实体名称

        Returns:
            是否匹配
        """
        if not query_entity or not simplex_entity:
            return False
        query_str = normalize_entity_name(query_entity)
        simplex_str = normalize_entity_name(simplex_entity)
        if not query_str or not simplex_str:
            return False
        if query_str == simplex_str:
            return True
        min_length = DualDimensionConfig.MIN_SUBSTRING_MATCH_LENGTH
        if len(query_str) >= min_length and len(simplex_str) >= min_length:
            pattern = r'\b' + re.escape(query_str) + r'\b'
            if re.search(pattern, simplex_str, re.IGNORECASE):
                return True
        query_words = set(query_str.split())
        simplex_words = set(simplex_str.split())
        if query_words and query_words.issubset(simplex_words):
            return True
        if len(query_str) < min_length or len(simplex_str) < min_length:
            return False
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, query_str, simplex_str).ratio()
        return similarity > DualDimensionConfig.ENTITY_MATCH_LEVENSHTEIN_THRESHOLD

    def _calculate_match_score(self, relation_entities: list, simplex_nodes: list) -> tuple:
        """计算关系实体与复形节点的匹配得分

        使用自适应阈值替代固定阈值，根据实体数量动态调整匹配条件。

        Args:
            relation_entities: 关系中的实体列表
            simplex_nodes: 复形中的节点列表

        Returns:
            (是否匹配, 得分, 匹配数, 精确匹配数)
        """
        if not relation_entities:
            return (False, 0, 0, 0)
        match_count = 0
        exact_match_count = 0
        for rel_ent in relation_entities:
            for simp_ent in simplex_nodes:
                if self._entity_match(rel_ent, simp_ent):
                    match_count += 1
                    query_str = normalize_entity_name(rel_ent)
                    simplex_str = normalize_entity_name(simp_ent)
                    if query_str == simplex_str:
                        exact_match_count += 1

        match_ratio = match_count / len(relation_entities)
        score = 0
        score += match_ratio * 60
        score += exact_match_count * 8
        score += (1 if match_count == len(relation_entities) else 0) * 32

        thresholds = DualDimensionConfig.get_match_thresholds(len(relation_entities))
        if len(relation_entities) == 1:
            matches = match_count >= 1 and score >= thresholds["score_threshold"]
        else:
            matches = match_ratio >= thresholds["match_ratio"]

        return (matches, score, match_count, exact_match_count)

    def _match_relations_to_simplices(
        self,
        enhanced_relations: list
    ) -> tuple:
        """维度B：将查询关系匹配到单纯形

        Args:
            enhanced_relations: 增强后的关系列表

        Returns:
            (relation_to_simplices, seed_edge_ids, seed_high_dim_simplices)
        """
        relation_to_simplices = defaultdict(set)
        seed_edge_ids = []
        seed_high_dim_simplices = {}

        if not enhanced_relations:
            return relation_to_simplices, seed_edge_ids, seed_high_dim_simplices

        edge_match_scores = []
        high_dim_match_scores = defaultdict(list)

        for relation in enhanced_relations:
            if isinstance(relation, dict):
                relation_entities = relation.get('entities', [])
                relation_dim = relation.get('dimension', len(relation_entities) - 1 if relation_entities else 0)
                is_partial = relation.get('is_partial', False)
                match_ratio = relation.get('match_ratio', 1.0)
            elif isinstance(relation, list):
                relation_entities = relation
                relation_dim = len(relation) - 1 if relation else 0
                is_partial = False
                match_ratio = 1.0
            else:
                continue

            normalized_entities = [normalize_entity_name(e) for e in relation_entities]
            relation_key = tuple(sorted(normalized_entities))

            if relation_key in self._sub_complex_cache:
                cached_pattern = self._sub_complex_cache[relation_key]
                for simplex_id in cached_pattern.get('precomputed_coboundary', set()):
                    relation_to_simplices[simplex_id].add(tuple(relation_entities))
            else:
                candidate_simplex_ids = set()
                for entity in relation_entities:
                    entity_matches = self._lookup_simplices_by_entity(entity)
                    candidate_simplex_ids.update(entity_matches.keys())

                for simplex_id in candidate_simplex_ids:
                    simplex_data = self.hsc.simplices.get(simplex_id, {})
                    if not simplex_data:
                        continue
                    simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))

                    matches, score, match_count, exact_match_count = self._calculate_match_score(
                        relation_entities, simplex_nodes
                    )

                    if matches:
                        relation_to_simplices[simplex_id].add(tuple(relation_entities))
                        simplex_dim = simplex_data.get('dimension', 0)
                        adjusted_score = score * match_ratio if is_partial else score
                        if simplex_dim == 1:
                            edge_match_scores.append((-adjusted_score, -match_count, -exact_match_count, simplex_id))
                        elif simplex_dim >= 2:
                            high_dim_match_scores[simplex_dim].append((-adjusted_score, -match_count, -exact_match_count, simplex_id))

        edge_match_scores.sort()
        seed_edge_ids = [simplex_id for (_, _, _, simplex_id) in edge_match_scores]

        for dim, scores_list in high_dim_match_scores.items():
            scores_list.sort()
            seed_high_dim_simplices[dim] = [simplex_id for (_, _, _, simplex_id) in scores_list]

        total_high_dim = sum(len(ids) for ids in seed_high_dim_simplices.values())
        logger.info(f"Stream B（结构模式）：匹配到 {len(relation_to_simplices)} 个关系单纯形，"
                    f"其中 {len(seed_edge_ids)} 个边（dim=1），{total_high_dim} 个高维复形（dim≥2）")

        return relation_to_simplices, seed_edge_ids, seed_high_dim_simplices

    async def _execute_diffusion(
        self,
        seed_nodes: list,
        seed_edge_ids: list,
        seed_high_dim_simplices: dict,
        relation_to_simplices: dict,
        query_text: str = None
    ) -> tuple:
        """双维度扩散：拓扑能量扩散 + 语义引导修正

        拓扑维度：Hodge Laplacian能量扩散，沿图结构传播种子节点能量
        语义维度：用查询嵌入对扩散得分进行语义加权修正，
        语义相关的节点得分提升，语义无关的噪声节点得分衰减。

        Args:
            seed_nodes: 种子节点列表
            seed_edge_ids: 种子边列表
            seed_high_dim_simplices: 高维种子复形
            relation_to_simplices: 关系到复形的映射
            query_text: 查询文本（用于语义引导修正）

        Returns:
            (diffused_node_scores, diffused_edge_scores, diffused_high_dim_scores)
        """
        if not self._laplacian_built:
            self.hsc.build_dynamic_incidence_matrices()
            self.hsc.compute_dynamic_hodge_laplacians()
            self._laplacian_built = True

        all_seeds = defaultdict(list)

        if seed_nodes:
            all_seeds[0].extend(seed_nodes)
        if seed_edge_ids:
            all_seeds[1].extend(seed_edge_ids)
        if seed_high_dim_simplices:
            for dim, simplex_ids in seed_high_dim_simplices.items():
                all_seeds[dim].extend(simplex_ids)

        if relation_to_simplices:
            for simplex_id in relation_to_simplices.keys():
                simplex_data = self.hsc.simplices.get(simplex_id, {})
                dim = simplex_data.get('dimension', 0)
                all_seeds[dim].append(simplex_id)

            for dim in sorted(all_seeds.keys()):
                all_seeds[dim] = list(set(all_seeds[dim]))
                logger.info(f"  dim={dim}: {len(all_seeds[dim])} 个复形种子")

        diffused_node_scores = {}
        diffused_edge_scores = {}
        diffused_high_dim_scores = {}

        # 多维度扩散并行执行：各维度使用独立的拉普拉斯矩阵，互不依赖
        diffusion_tasks = []
        if 0 in all_seeds and all_seeds[0]:
            diffusion_tasks.append(('node', all_seeds[0], 0))
        if 1 in all_seeds and all_seeds[1]:
            diffusion_tasks.append(('edge', all_seeds[1], 1))
        for dim in all_seeds:
            if dim >= 2 and all_seeds[dim]:
                diffusion_tasks.append((f'high_{dim}', all_seeds[dim], dim))

        if diffusion_tasks:
            loop = asyncio.get_running_loop()
            diffusion_futures = []
            for name, seeds, dim in diffusion_tasks:
                future = loop.run_in_executor(None, self.hsc.dynamic_diffusion, seeds, dim)
                diffusion_futures.append((name, future))

            for name, future in diffusion_futures:
                try:
                    scores = await future
                    if name == 'node':
                        diffused_node_scores = scores
                        logger.info(f"Stream A（dim=0）扩散完成，获得 {len(scores)} 个节点的拓扑得分")
                    elif name == 'edge':
                        diffused_edge_scores = scores
                        logger.info(f"Stream B（dim=1）扩散完成，获得 {len(scores)} 个边的拓扑得分")
                    else:
                        diffused_high_dim_scores.update(scores)
                        logger.info(f"Stream C ({name}) 扩散完成，获得 {len(scores)} 个高维复形的拓扑得分")
                except Exception as e:
                    logger.warning(f"维度 {name} 扩散失败: {e}")

        # 语义引导修正：用查询嵌入对扩散得分进行向量化加权
        if query_text and self._simplex_embedding_cache:
            query_embedding = await self._get_query_embedding(query_text)
            if query_embedding is not None:
                q_norm = float(np.linalg.norm(query_embedding))
                if q_norm > 0:
                    sem_adjusted_nodes = 0
                    sem_adjusted_edges = 0
                    sem_adjusted_high = 0

                    for score_dict, label in [
                        (diffused_node_scores, 'node'),
                        (diffused_edge_scores, 'edge'),
                        (diffused_high_dim_scores, 'high')
                    ]:
                        if not score_dict:
                            continue
                        cached_ids = [sid for sid in score_dict if sid in self._simplex_embedding_cache]
                        if not cached_ids:
                            continue
                        emb_matrix = np.stack([self._simplex_embedding_cache[sid] for sid in cached_ids])
                        emb_norms = np.linalg.norm(emb_matrix, axis=1)
                        valid_mask = emb_norms > 0
                        if not np.any(valid_mask):
                            continue
                        valid_ids = [sid for sid, m in zip(cached_ids, valid_mask) if m]
                        valid_embs = emb_matrix[valid_mask]
                        valid_norms = emb_norms[valid_mask]
                        cos_sims = valid_embs @ query_embedding / (valid_norms * q_norm)
                        sem_weights = 0.2 + 0.8 * np.maximum(0, cos_sims)
                        for sid, w in zip(valid_ids, sem_weights):
                            score_dict[sid] *= float(w)
                        if label == 'node':
                            sem_adjusted_nodes = len(valid_ids)
                        elif label == 'edge':
                            sem_adjusted_edges = len(valid_ids)
                        else:
                            sem_adjusted_high += len(valid_ids)

                    logger.info(f"语义引导扩散修正：节点{sem_adjusted_nodes}个, "
                                f"边{sem_adjusted_edges}个, 高维{sem_adjusted_high}个")

        return diffused_node_scores, diffused_edge_scores, diffused_high_dim_scores

    async def _compute_coboundary_contraction(
        self,
        vertex_ids: list,
        type_filter: list = None,
        query_text: str = None
    ) -> tuple:
        """步骤3：上边界收缩 - 维度A通过B^T向上辐射寻找包含自己的面

        使用自适应覆盖阈值替代固定阈值，避免参数过多。
        使用原始查询文本计算语义相似度，而非实体ID拼接。

        Args:
            vertex_ids: 查询顶点ID列表
            type_filter: 类型过滤列表
            query_text: 原始查询文本（用于计算语义相似度，替代vertex_ids拼接）

        Returns:
            (candidate_simplices, simplex_coverage, strict_intersection,
             simplex_matched_vertices, simplex_similarity)
        """
        vertex_coboundaries = []
        for vertex_id in vertex_ids:
            vertex_coboundary = self.hsc.get_upper_adjacent([vertex_id], current_dim=0)
            if type_filter:
                filtered_coboundary = set()
                for simplex_id in vertex_coboundary:
                    simplex_data = self.hsc.simplices.get(simplex_id, {})
                    simplex_type = simplex_data.get('type', '')
                    if simplex_type in type_filter:
                        filtered_coboundary.add(simplex_id)
                vertex_coboundary = filtered_coboundary
            vertex_coboundaries.append(vertex_coboundary)

        strict_intersection = set()
        if vertex_coboundaries:
            strict_intersection = set.intersection(*vertex_coboundaries)

        candidate_simplices = set()
        simplex_matched_vertices = defaultdict(set)
        max_coboundary_per_vertex = DualDimensionConfig.MAX_COBOUNDARY_PER_VERTEX

        for i, coboundary in enumerate(vertex_coboundaries):
            coboundary_list = list(coboundary)
            coboundary_list.sort(
                key=lambda sid: self.hsc.simplices.get(sid, {}).get('dimension', 0),
                reverse=True
            )
            limited_coboundary = coboundary_list[:max_coboundary_per_vertex]
            candidate_simplices.update(limited_coboundary)
            for simplex_id in limited_coboundary:
                simplex_matched_vertices[simplex_id].add(vertex_ids[i])

        logger.info(f"维度A严格交集大小: {len(strict_intersection)}")
        logger.info(f"维度A候选复形数: {len(candidate_simplices)}")

        simplex_coverage = {}
        for simplex_id in candidate_simplices:
            simplex_data = self.hsc.simplices.get(simplex_id, {})
            simplex_vertices = set(simplex_data.get('nodes', simplex_data.get('entities', [])))
            covered_vertices = simplex_vertices.intersection(set(vertex_ids))
            coverage_count = len(covered_vertices)
            coverage_ratio = coverage_count / len(vertex_ids) if vertex_ids else 0
            dim = simplex_data.get('dimension', 0)
            boundary = simplex_data.get('boundary', [])
            coboundary = simplex_data.get('coboundary', [])
            structural_complexity = len(boundary) + len(coboundary)
            importance = simplex_data.get('importance', 1.0)

            simplex_coverage[simplex_id] = {
                'count': coverage_count,
                'ratio': coverage_ratio,
                'dimension': dim,
                'structural_complexity': structural_complexity,
                'importance': importance,
                'covered_vertices': covered_vertices
            }

        effective_query_text = query_text if query_text else " ".join(vertex_ids)
        simplex_similarity = {}
        if candidate_simplices and effective_query_text:
            simplex_texts = []
            simplex_ids = []
            for simplex_id in candidate_simplices:
                simplex_data = self.hsc.simplices.get(simplex_id, {})
                simplex_entities = simplex_data.get('nodes', simplex_data.get('entities', []))
                simplex_text = " ".join(simplex_entities)
                if simplex_text:
                    simplex_texts.append(simplex_text)
                    simplex_ids.append(simplex_id)

            if simplex_texts:
                logger.info(f"开始计算语义相似度，查询文本: '{effective_query_text[:50]}...', 复形数: {len(simplex_texts)}")
                query_embedding = await self._get_query_embedding(effective_query_text)
                if query_embedding is not None:
                    simplex_embeddings = await self._get_simplex_embeddings_batch(simplex_ids, simplex_texts)
                    if simplex_embeddings:
                        emb_ids_list = list(simplex_embeddings.keys())
                        emb_mat = np.stack([simplex_embeddings[sid] for sid in emb_ids_list])
                        emb_n = np.linalg.norm(emb_mat, axis=1)
                        q_n = float(np.linalg.norm(query_embedding))
                        if q_n > 0:
                            valid = emb_n > 0
                            scores = np.zeros(len(emb_ids_list))
                            scores[valid] = emb_mat[valid] @ query_embedding / (emb_n[valid] * q_n)
                            for i, sid in enumerate(emb_ids_list):
                                if scores[i] > 0:
                                    simplex_similarity[sid] = float(scores[i])
                else:
                    simplex_similarity = await compute_semantic_similarity(effective_query_text, simplex_texts, simplex_ids)

        return (candidate_simplices, simplex_coverage, strict_intersection,
                simplex_matched_vertices, simplex_similarity)

    def _filter_candidates(
        self,
        vertex_ids: list,
        simplex_coverage: dict,
        strict_intersection: set,
        simplex_similarity: dict,
        candidate_simplices: set
    ) -> list:
        """基于覆盖度和语义相似度筛选候选复形

        使用自适应阈值替代固定阈值：
        - 覆盖阈值根据查询顶点数量自适应计算
        - 语义相似度阈值根据分数分布自适应计算
        - 保留数量根据候选总数自适应计算

        Args:
            vertex_ids: 查询顶点ID列表
            simplex_coverage: 复形覆盖度信息
            strict_intersection: 严格交集
            simplex_similarity: 语义相似度
            candidate_simplices: 候选复形集合

        Returns:
            筛选后的复形列表 [(simplex_id, coverage_info), ...]
        """
        filtered_simplices = []

        coverage_threshold = DualDimensionConfig.get_coverage_threshold(len(vertex_ids))
        similarity_scores = list(simplex_similarity.values()) if simplex_similarity else None
        semantic_threshold = DualDimensionConfig.get_semantic_threshold(similarity_scores)

        for simplex_id in strict_intersection:
            if simplex_id in simplex_coverage:
                sem_score = simplex_similarity.get(simplex_id, 0)
                relaxed_threshold = max(0.1, semantic_threshold - 0.1) if simplex_similarity else 0
                if sem_score >= relaxed_threshold or not simplex_similarity:
                    filtered_simplices.append((simplex_id, simplex_coverage[simplex_id]))

        if len(vertex_ids) == 1:
            sorted_candidates = sorted(
                [(sid, cov) for sid, cov in simplex_coverage.items() if sid not in strict_intersection],
                key=lambda x: (
                    x[1]['dimension'],
                    simplex_similarity.get(x[0], 0),
                    -x[1]['structural_complexity'],
                    x[1]['importance']
                ),
                reverse=True
            )
            filtered_candidates = [c for c in sorted_candidates if simplex_similarity.get(c[0], 0) >= semantic_threshold]
            if filtered_candidates:
                filtered_simplices.extend(filtered_candidates)
            else:
                filtered_simplices.extend(sorted_candidates[:max(5, len(sorted_candidates))])
        else:
            candidate_list = []
            for simplex_id, coverage in simplex_coverage.items():
                if simplex_id not in strict_intersection and (
                    coverage['count'] >= 1 or
                    coverage['ratio'] >= coverage_threshold):
                    if simplex_similarity.get(simplex_id, 0) >= semantic_threshold:
                        candidate_list.append((simplex_id, coverage))

            candidate_list.sort(key=lambda x: (
                x[1]['dimension'],
                simplex_similarity.get(x[0], 0),
                x[1]['count'],
                x[1]['importance'],
                x[1]['structural_complexity'],
                x[1]['ratio']
            ), reverse=True)

            filtered_simplices.extend(candidate_list)

        filtered_simplices.sort(key=lambda x: (
            x[0] in strict_intersection,
            x[1]['count'],
            x[1]['dimension'],
            x[1]['importance'],
            x[1]['structural_complexity'],
            x[1]['ratio']
        ), reverse=True)

        return filtered_simplices

    def _compute_common_coboundary(self, vertex_ids: list) -> set:
        """计算查询顶点的共同上边界复形

        Args:
            vertex_ids: 查询顶点ID列表

        Returns:
            共同上边界复形集合
        """
        node_coboundaries = []
        for node_id in vertex_ids:
            if node_id in self.hsc.simplices:
                node_data = self.hsc.simplices[node_id]
                node_coboundary = node_data.get('coboundary', [])
                node_coboundaries.append(set(node_coboundary))
            else:
                node_coboundary = set()
                for simplex_id, simplex_data in self.hsc.simplices.items():
                    dim = simplex_data.get('dimension', 0)
                    if dim > 0:
                        simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
                        if node_id in simplex_nodes:
                            node_coboundary.add(simplex_id)
                node_coboundaries.append(node_coboundary)

        if node_coboundaries:
            common = set.intersection(*node_coboundaries)
            logger.info(f"维度A：找到 {len(common)} 个共同上边界复形")
            return common
        return set()

    def _fusion_dual_dimensions(
        self,
        candidate_simplices: set,
        common_coboundary: set,
        pattern_simplices: set,
        vertex_ids: list,
        relation_to_simplices: dict,
        diffused_high_dim_scores: dict,
        coboundary_threshold: float,
        vertex_quality: float = None,
        relation_quality: float = None,
        simplex_similarity: dict = None
    ) -> tuple:
        """三维度融合：拓扑维度A（实体）+ 拓扑维度B（关系）+ 语义维度

        在原有拓扑A/B双维度融合基础上，融入语义相似度分量，
        确保语义相关但拓扑得分低的复形不会被过早淘汰。

        Args:
            candidate_simplices: 维度A候选复形
            common_coboundary: 共同上边界
            pattern_simplices: 维度B模式匹配复形
            vertex_ids: 查询顶点
            relation_to_simplices: 关系到复形映射
            diffused_high_dim_scores: 高维扩散得分
            coboundary_threshold: 对撞阈值
            vertex_quality: 维度A匹配质量（0~1），None则自动计算
            relation_quality: 维度B匹配质量（0~1），None则自动计算
            simplex_similarity: 语义相似度字典（来自_compute_coboundary_contraction）

        Returns:
            (fusion_simplices, weighted_scores, simplex_scores)
        """
        simplex_scores = {}

        for simplex_id in pattern_simplices:
            simplex_data = self.hsc.simplices.get(simplex_id, {})
            simplex_data_copy = dict(simplex_data)
            simplex_data_copy['simplex_id'] = simplex_id
            b_score = calculate_simplex_score(simplex_data_copy, None, relation_to_simplices, score_type='B')
            simplex_scores[simplex_id] = b_score

        for simplex_id in common_coboundary:
            simplex_data = self.hsc.simplices.get(simplex_id, {})
            simplex_data_copy = dict(simplex_data)
            simplex_data_copy['simplex_id'] = simplex_id
            a_score = calculate_simplex_score(simplex_data_copy, vertex_ids, None, score_type='A')
            simplex_scores[simplex_id] = a_score

        all_retrieved_simplices = set()
        all_retrieved_simplices.update(candidate_simplices)
        all_retrieved_simplices.update(common_coboundary)
        all_retrieved_simplices.update(pattern_simplices)

        if vertex_quality is None:
            a_simplices = set(candidate_simplices) | set(common_coboundary)
            vertex_quality = len(a_simplices) / max(len(vertex_ids), 1) if vertex_ids else 0.0
            vertex_quality = min(1.0, vertex_quality)
        if relation_quality is None:
            relation_quality = min(1.0, len(relation_to_simplices) / max(len(pattern_simplices), 1)) if pattern_simplices else 0.0

        weight_a, weight_b = DualDimensionConfig.compute_dynamic_fusion_weights(
            vertex_quality, relation_quality
        )

        weighted_scores = {}
        if simplex_similarity:
            sem_quality = self._compute_semantic_quality(simplex_similarity)
            topo_quality_for_fusion = (vertex_quality + relation_quality) / 2.0 if (vertex_quality + relation_quality) > 0 else 0.5
            topo_weight, sem_weight = DualDimensionConfig.compute_topology_semantic_weights(
                topo_quality_for_fusion, sem_quality
            )
        else:
            topo_weight = 1.0
            sem_weight = 0.0

        # 双维度均失效时的回退策略：优先使用扩散得分，其次均分权重
        if weight_a == 0.0 and weight_b == 0.0:
            if diffused_high_dim_scores:
                logger.warning("双维度均失效（vertex_quality=0, relation_quality=0），"
                               "启用回退策略：使用拓扑扩散得分作为排序依据")
                for sid, d_score in diffused_high_dim_scores.items():
                    simplex_data = self.hsc.simplices.get(sid, {})
                    importance = simplex_data.get('importance', 1.0)
                    weighted_scores[sid] = topo_weight * d_score * importance + sem_weight * simplex_similarity.get(sid, 0)
                if weighted_scores:
                    scores_values = list(weighted_scores.values())
                    adaptive_threshold = max(coboundary_threshold, float(np.percentile(scores_values, 30))) if scores_values else coboundary_threshold
                    fusion_simplices = {
                        sid for sid, score in weighted_scores.items()
                        if score >= adaptive_threshold
                    }
                    if fusion_simplices:
                        logger.info(f"扩散回退策略生效：筛选出 {len(fusion_simplices)} 个复形")
                        return fusion_simplices, weighted_scores, simplex_scores
            logger.warning("双维度均失效且无扩散得分，使用均分权重回退")
            weight_a = 0.5
            weight_b = 0.5

        logger.info(f"加权融合：维度A复形数: {len(set(candidate_simplices) | set(common_coboundary))}, "
                    f"维度B复形数: {len(pattern_simplices)}, 融合后总数: {len(all_retrieved_simplices)}, "
                    f"动态权重: A={weight_a:.2f}, B={weight_b:.2f} "
                    f"(质量: A={vertex_quality:.2f}, B={relation_quality:.2f})")

        for simplex_id in all_retrieved_simplices:
            simplex_data = self.hsc.simplices.get(simplex_id, {})
            simplex_data_copy = dict(simplex_data)
            simplex_data_copy['simplex_id'] = simplex_id

            a_score = 0
            if simplex_id in common_coboundary or simplex_id in candidate_simplices:
                a_score = calculate_simplex_score(simplex_data_copy, vertex_ids, None, score_type='A')

            b_score = 0
            if simplex_id in pattern_simplices:
                b_score = calculate_simplex_score(simplex_data_copy, None, relation_to_simplices, score_type='B')

            topo_score = weight_a * a_score + weight_b * b_score
            sem_score = simplex_similarity.get(simplex_id, 0) if simplex_similarity else 0
            weighted_scores[simplex_id] = topo_weight * topo_score + sem_weight * sem_score

        if weighted_scores:
            scores_values = list(weighted_scores.values())
            adaptive_threshold = max(coboundary_threshold, float(np.percentile(scores_values, 30)))
        else:
            adaptive_threshold = coboundary_threshold

        fusion_simplices = {
            sid for sid, score in weighted_scores.items()
            if score >= adaptive_threshold
        }

        if not fusion_simplices and weighted_scores:
            sorted_fallback = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
            fallback_count = min(max(len(sorted_fallback) // 5, 3), len(sorted_fallback))
            fusion_simplices = {sid for sid, _ in sorted_fallback[:fallback_count]}
            logger.info(f"降低阈值回退：取top-{fallback_count}复形")

        sorted_simplices = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"加权融合：基于阈值筛选后复形数: {len(fusion_simplices)}")
        if sorted_simplices:
            top_simplex = sorted_simplices[0]
            logger.info(f"最高分复形: {top_simplex[0]} (得分: {top_simplex[1]:.2f})")

        if diffused_high_dim_scores:
            logger.info(f"融合高维扩散结果：{len(diffused_high_dim_scores)} 个高维复形")
            high_dim_candidates = {}

            for sid, score in diffused_high_dim_scores.items():
                simplex_data = self.hsc.simplices.get(sid, {})
                simplex_vertices = set(simplex_data.get('nodes', simplex_data.get('entities', [])))
                covered_vertices = simplex_vertices.intersection(set(vertex_ids))
                coverage_count = len(covered_vertices)
                importance = simplex_data.get('importance', 1.0)

                if coverage_count > 0:
                    combined_score = score * (coverage_count ** 2) * importance
                    high_dim_candidates[sid] = combined_score

            sorted_high_dim = sorted(high_dim_candidates.items(), key=lambda x: x[1], reverse=True)
            top_high_dim = set(sid for sid, _ in sorted_high_dim[:10])

            if top_high_dim:
                new_fusion = fusion_simplices | top_high_dim
                logger.info(f"融合后复形数: {len(fusion_simplices)} -> {len(new_fusion)}")
                fusion_simplices = new_fusion

        return fusion_simplices, weighted_scores, simplex_scores

    def _try_elevation(self, vertex_ids, pattern_simplices, common_coboundary, seed_edge_ids):
        """维度提升：将多个低维复形拼接成高维复形

        Args:
            vertex_ids: 查询顶点
            pattern_simplices: 模式匹配复形
            common_coboundary: 共同上边界
            seed_edge_ids: 种子边ID

        Returns:
            提升后的复形集合
        """
        elevated_simplices = set()

        for edge_simplex_id in seed_edge_ids:
            edge_simplex_data = self.hsc.simplices.get(edge_simplex_id, {})
            edge_simplex_entities_set = set(edge_simplex_data.get('entities', []))

            for pattern_id in pattern_simplices:
                pattern_data = self.hsc.simplices.get(pattern_id, {})
                pattern_entities = pattern_data.get('entities', [])
                if pattern_entities and set(pattern_entities).issubset(edge_simplex_entities_set):
                    elevated_simplices.add(edge_simplex_id)

        if not elevated_simplices:
            for simp1_id in pattern_simplices:
                simp1_data = self.hsc.simplices.get(simp1_id, {})
                simp1_entities = set(simp1_data.get('entities', []))

                for simp2_id in common_coboundary:
                    if simp1_id == simp2_id:
                        continue
                    simp2_data = self.hsc.simplices.get(simp2_id, {})
                    simp2_entities = set(simp2_data.get('entities', []))

                    common_vertices = simp1_entities & simp2_entities
                    if common_vertices:
                        merged_entities = simp1_entities | simp2_entities
                        for candidate_id in simp1_data.get('coboundary', []):
                            candidate_data = self.hsc.simplices.get(candidate_id, {})
                            if not candidate_data:
                                continue
                            candidate_entities = set(candidate_data.get('entities', []))
                            if merged_entities.issubset(candidate_entities):
                                elevated_simplices.add(candidate_id)

        if not elevated_simplices:
            logger.info("执行迭代维度提升...")
            elevated_simplices = self._iterative_dimension_elevation(
                vertex_ids, pattern_simplices, common_coboundary, max_iterations=3
            )

        return elevated_simplices

    def _iterative_dimension_elevation(
        self,
        vertex_ids: List[str],
        pattern_simplices: set,
        common_coboundary: set,
        max_iterations: int = 2
    ) -> set:
        """迭代维度提升：多次迭代提升维度直到找到满意的答案

        Args:
            vertex_ids: 查询顶点列表
            pattern_simplices: 模式匹配的复形集合
            common_coboundary: 共同上边界复形集合
            max_iterations: 最大迭代次数

        Returns:
            提升后的复形集合
        """
        logger.info(f"开始迭代维度提升，初始pattern_simplices: {len(pattern_simplices)}, common_coboundary: {len(common_coboundary)}")

        candidate_simplices = pattern_simplices | common_coboundary
        if not candidate_simplices:
            return set()

        simplices_by_dim = defaultdict(list)
        hsc_max_dim = 0
        for simplex_id, simplex_data in self.hsc.simplices.items():
            dim = simplex_data.get('dimension', 0)
            simplices_by_dim[dim].append((simplex_id, simplex_data))
            hsc_max_dim = max(hsc_max_dim, dim)

        elevated = set()
        current_candidates = candidate_simplices.copy()

        for iteration in range(max_iterations):
            logger.info(f"迭代 {iteration + 1}/{max_iterations}: 当前候选复形 {len(current_candidates)}")

            if not current_candidates:
                break

            candidate_entities = set()
            for simp_id in current_candidates:
                simp_data = self.hsc.simplices.get(simp_id, {})
                entities = simp_data.get('entities', [])
                candidate_entities.update(entities)

            next_candidates = set()
            current_max_dim = 0

            for simp_id in current_candidates:
                simp_data = self.hsc.simplices.get(simp_id, {})
                dim = simp_data.get('dimension', 0)
                current_max_dim = max(current_max_dim, dim)

            for simp_id in current_candidates:
                simp_data = self.hsc.simplices.get(simp_id, {})
                coboundary = simp_data.get('coboundary', [])
                for higher_simplex_id in coboundary:
                    higher_data = self.hsc.simplices.get(higher_simplex_id, {})
                    higher_dim = higher_data.get('dimension', 0)
                    if higher_dim > current_max_dim:
                        higher_entities = set(higher_data.get('entities', []))
                        if candidate_entities.issubset(higher_entities):
                            next_candidates.add(higher_simplex_id)
                            elevated.add(higher_simplex_id)

            if not next_candidates:
                coboundary_pool = set()
                for simp_id in current_candidates:
                    simp_data = self.hsc.simplices.get(simp_id, {})
                    coboundary_pool.update(simp_data.get('coboundary', []))
                for higher_id in coboundary_pool:
                    higher_data = self.hsc.simplices.get(higher_id, {})
                    if not higher_data:
                        continue
                    higher_dim = higher_data.get('dimension', 0)
                    if higher_dim > current_max_dim:
                        higher_entities = set(higher_data.get('entities', []))
                        if candidate_entities.issubset(higher_entities):
                            next_candidates.add(higher_id)
                            elevated.add(higher_id)

            if next_candidates and next_candidates != current_candidates:
                current_candidates = next_candidates
            else:
                break

        logger.info(f"迭代维度提升完成，找到 {len(elevated)} 个提升后的复形")
        return elevated

    def _build_adjacency_cache(self):
        """预计算并缓存复形邻接表，避免每次查询重复构建

        基于boundary/coboundary关系构建邻接表，供find_topological_chain使用。
        当HSC复形数据变化时（_adjacency_cache_version递增），自动重建缓存。
        """
        current_version = len(self.hsc.simplices)
        if self._adjacency_cache is not None and self._adjacency_cache_version == current_version:
            return

        simplex_adjacency = defaultdict(set)
        simplex_nodes = {}
        for simplex_id, simplex_data in self.hsc.simplices.items():
            nodes = set(simplex_data.get('nodes', simplex_data.get('entities', [])))
            simplex_nodes[simplex_id] = nodes

        for simplex_id, simplex_data in self.hsc.simplices.items():
            boundary = simplex_data.get('boundary', [])
            for sub_simplex_id in boundary:
                if sub_simplex_id in self.hsc.simplices:
                    simplex_adjacency[simplex_id].add(sub_simplex_id)
                    simplex_adjacency[sub_simplex_id].add(simplex_id)
            coboundary = simplex_data.get('coboundary', [])
            for super_simplex_id in coboundary:
                if super_simplex_id in self.hsc.simplices:
                    simplex_adjacency[simplex_id].add(super_simplex_id)
                    simplex_adjacency[super_simplex_id].add(simplex_id)

        if not simplex_adjacency:
            simplex_list = list(self.hsc.simplices.keys())
            for i, sid1 in enumerate(simplex_list):
                for sid2 in simplex_list[i+1:]:
                    if simplex_nodes[sid1] & simplex_nodes[sid2]:
                        simplex_adjacency[sid1].add(sid2)
                        simplex_adjacency[sid2].add(sid1)

        self._adjacency_cache = (simplex_adjacency, simplex_nodes)
        self._adjacency_cache_version = current_version
        logger.info(f"邻接表缓存构建完成：{len(simplex_adjacency)} 个复形有邻接关系")

    def find_topological_chain(self, vertex_ids: List[int], max_hops: int = 2) -> List[Dict]:
        """单纯形链的拓扑连通 - 多源BFS优化版（使用预计算邻接表缓存）

        从所有查询顶点同时出发进行BFS搜索，
        当不同源顶点的搜索路径在同一节点交汇时，
        该节点即为连接多个查询顶点的拓扑链节点。

        改进：max_hops从3降为2，减少BFS搜索空间；
        增加匹配顶点数过滤，仅保留至少匹配1个查询顶点的复形。

        Args:
            vertex_ids: 查询顶点ID列表
            max_hops: 最大跳数（默认2）

        Returns:
            拓扑链结果列表
        """
        if len(vertex_ids) < 2:
            return []

        self._build_adjacency_cache()
        simplex_adjacency, simplex_nodes = self._adjacency_cache

        vertex_to_simplices = defaultdict(set)
        for simplex_id, nodes in simplex_nodes.items():
            for vertex_id in vertex_ids:
                if vertex_id in nodes:
                    vertex_to_simplices[vertex_id].add(simplex_id)

        source_reachability = defaultdict(set)
        queue = deque()
        for vid in vertex_ids:
            source_reachability[vid].add(vid)
            for sid in vertex_to_simplices.get(vid, set()):
                source_reachability[sid].add(vid)
                queue.append((sid, vid, 0))

        all_path_simplices = set()
        visited = set()

        while queue:
            current, source, hops = queue.popleft()

            if (current, source) in visited:
                continue
            visited.add((current, source))

            if len(source_reachability[current]) >= 2:
                all_path_simplices.add(current)

            if hops >= max_hops:
                continue

            for neighbor in simplex_adjacency.get(current, set()):
                source_reachability[neighbor].add(source)
                queue.append((neighbor, source, hops + 1))

        chain_results = []
        vertex_id_set = set(vertex_ids)
        for simplex_id in all_path_simplices:
            simplex_data = self.hsc.simplices[simplex_id]
            nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
            matched_verts = [v for v in vertex_ids if v in nodes]
            if not matched_verts:
                continue
            chain_results.append({
                'simplex_id': simplex_id,
                'dimension': simplex_data.get('dimension', 0),
                'level_hg': simplex_data.get('level_hg', simplex_data.get('importance', 'lower')),
                'all_vertices': nodes,
                'matched_vertices': matched_verts,
                'source': simplex_data.get('source', simplex_data.get('source_id', '')),
                'description': simplex_data.get('description', '')
            })

        logger.info(f"多源BFS拓扑链找到 {len(chain_results)} 个单纯形，覆盖 {len(vertex_ids)} 个查询顶点")
        return chain_results

    def _build_completion_results(
        self,
        fusion_simplices: set,
        vertex_ids: list,
        simplex_scores: dict,
        simplex_matched_vertices: dict,
        diffused_node_scores: dict
    ) -> list:
        """构建补全结果列表

        Args:
            fusion_simplices: 融合后的复形集合
            vertex_ids: 查询顶点
            simplex_scores: 复形得分
            simplex_matched_vertices: 匹配的顶点
            diffused_node_scores: 扩散得分

        Returns:
            补全结果列表
        """
        completion_results = []
        vertex_id_set = set(vertex_ids)

        for simplex_id in fusion_simplices:
            simplex_data = self.hsc.simplices[simplex_id]
            simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
            simplex_type_map = simplex_data.get('node_types', {})

            missing_vertices = []
            for node in simplex_nodes:
                if node not in vertex_id_set:
                    node_type = simplex_type_map.get(node, 'unknown')
                    missing_vertices.append({
                        'id': node,
                        'type': node_type,
                        'is_answer_candidate': node_type in ['Value', 'State', 'Number', 'Attribute']
                    })

            coboundary_score = simplex_scores.get(simplex_id, 0)
            matched_count = len(simplex_matched_vertices.get(simplex_id, set()))

            diffusion_score = 0
            if simplex_nodes:
                node_scores = [diffused_node_scores.get(vid, 0) for vid in simplex_nodes]
                if node_scores:
                    diffusion_score = sum(node_scores) / len(node_scores)

            result = {
                'simplex_id': simplex_id,
                'dimension': simplex_data.get('dimension', 0),
                'level_hg': simplex_data.get('level_hg', simplex_data.get('importance', 'lower')),
                'missing_vertices': missing_vertices,
                'all_vertices': simplex_nodes,
                'matched_vertices': list(simplex_matched_vertices.get(simplex_id, set())),
                'coboundary_score': coboundary_score,
                'diffusion_score': diffusion_score,
                'source': simplex_data.get('source', simplex_data.get('source_id', '')),
                'description': simplex_data.get('description', '')
            }
            result.update(simplex_data)
            completion_results.append(result)

        return completion_results

    def _inject_seed_simplices(
        self,
        completion_results: list,
        seed_node_ids: list,
        seed_edge_ids: list,
        diffused_node_scores: dict
    ) -> list:
        """注入种子节点和种子复形到检索结果中

        种子节点：查询实体对应的0维复形，保留实体属性和类型信息
        种子复形：查询关系直接匹配的1维边复形，保留直接关联的结构信息
        这些复形标记为is_seed=True，在子复形移除和截断时受保护

        Args:
            completion_results: 已有的补全结果列表
            seed_node_ids: 种子节点ID列表（查询实体对应的0维复形ID）
            seed_edge_ids: 种子边ID列表（查询关系直接匹配的1维复形ID）
            diffused_node_scores: 扩散得分字典

        Returns:
            注入种子后的补全结果列表
        """
        existing_ids = {r.get('simplex_id') for r in completion_results}
        seed_results = []

        for node_id in seed_node_ids:
            if node_id in existing_ids:
                for r in completion_results:
                    if r.get('simplex_id') == node_id:
                        r['is_seed'] = True
                continue

            simplex_data = self.hsc.simplices.get(node_id)
            if not simplex_data:
                simplex_data = {
                    'id': node_id,
                    'dimension': 0,
                    'entities': [node_id],
                    'nodes': [node_id],
                    'type': 'Entity',
                    'entity_type': 'Entity',
                    'boundary': [],
                    'coboundary': [],
                    'importance': 1.0,
                    'frequency': 1
                }

            # 种子节点（0维）通常没有source字段
            # 从其coboundary（包含该节点的1维边复形）中收集source
            node_source = set()
            direct_source = simplex_data.get('source', simplex_data.get('source_id', ''))
            if direct_source:
                node_source.update(self._parse_source_ids(direct_source))
            coboundary = simplex_data.get('coboundary', [])
            for edge_id in coboundary:
                edge_data = self.hsc.simplices.get(edge_id, {})
                edge_source = edge_data.get('source', edge_data.get('source_id', ''))
                if edge_source:
                    node_source.update(self._parse_source_ids(edge_source))

            diffusion_score = diffused_node_scores.get(node_id, 0)
            seed_result = {
                'simplex_id': node_id,
                'dimension': 0,
                'level_hg': simplex_data.get('level_hg', simplex_data.get('importance', 'lower')),
                'missing_vertices': [],
                'all_vertices': [node_id],
                'matched_vertices': [node_id],
                'coboundary_score': 0,
                'diffusion_score': diffusion_score,
                'source': "<SEP>".join(sorted(node_source)) if node_source else '',
                'source_id': "<SEP>".join(sorted(node_source)) if node_source else '',
                'description': simplex_data.get('description', ''),
                'is_seed': True
            }
            seed_results.append(seed_result)

        for edge_id in seed_edge_ids:
            if edge_id in existing_ids:
                for r in completion_results:
                    if r.get('simplex_id') == edge_id:
                        r['is_seed'] = True
                continue

            simplex_data = self.hsc.simplices.get(edge_id)
            if not simplex_data:
                continue

            simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
            diffusion_score = 0
            if simplex_nodes:
                node_scores = [diffused_node_scores.get(vid, 0) for vid in simplex_nodes]
                if node_scores:
                    diffusion_score = sum(node_scores) / len(node_scores)

            seed_result = {
                'simplex_id': edge_id,
                'dimension': simplex_data.get('dimension', 1),
                'level_hg': simplex_data.get('level_hg', simplex_data.get('importance', 'lower')),
                'missing_vertices': [],
                'all_vertices': simplex_nodes,
                'matched_vertices': simplex_nodes,
                'coboundary_score': 0,
                'diffusion_score': diffusion_score,
                'source': simplex_data.get('source', simplex_data.get('source_id', '')),
                'description': simplex_data.get('description', ''),
                'is_seed': True
            }
            seed_result.update(simplex_data)
            seed_results.append(seed_result)

        if seed_results:
            logger.info(f"注入 {sum(1 for r in seed_results if r['dimension'] == 0)} 个种子节点，"
                        f"{sum(1 for r in seed_results if r['dimension'] >= 1)} 个种子复形")

        return completion_results + seed_results

    @staticmethod
    def _parse_source_ids(source_val) -> list:
        """解析复形的source字段为source_id列表

        source字段可能是单个chunk_id、<SEP>分隔的多个chunk_id、或列表

        Args:
            source_val: source字段值

        Returns:
            source_id列表
        """
        if not source_val:
            return []
        if isinstance(source_val, list):
            return [s for s in source_val if s]
        if isinstance(source_val, str):
            return [s.strip() for s in source_val.split("<SEP>") if s.strip()]
        return []

    def _remove_sub_simplices(self, completion_results: list) -> list:
        """移除子复形，保留父复形，并将子复形的source_id合并到父复形

        核心原则：最终回答依赖原始文本块，移除子复形时必须确保其source_id
        不丢失。当子复形的实体集合是父复形的子集时，将子复形的source_id
        合并到父复形，同时给父复形和对应文本块加分。

        种子实体（dim-0）不再强制保护：若其被高维复形包含，source会合并到
        父复形，信息不丢失；若其是孤立的（无父复形包含），则保留。

        Args:
            completion_results: 补全结果列表

        Returns:
            去除子复形后的结果列表
        """
        if len(completion_results) <= 1:
            return completion_results

        simplex_entities_map = {}
        for result in completion_results:
            simplex_id = result.get('simplex_id', id(result))
            all_entities = frozenset(result.get('all_vertices', result.get('entities', [])))
            simplex_entities_map[simplex_id] = {
                'all_entities': all_entities,
                'matched_entities': frozenset(result.get('matched_vertices', [])),
                'result': result,
                'dimension': result.get('dimension', 0)
            }

        sorted_sids = sorted(simplex_entities_map.keys(),
                             key=lambda sid: len(simplex_entities_map[sid]['all_entities']),
                             reverse=True)

        subsets_to_remove = set()
        parent_bonus = defaultdict(float)
        parent_merged_sources = defaultdict(set)
        seed_has_parent = set()

        for i, sid1 in enumerate(sorted_sids):
            if sid1 in subsets_to_remove:
                continue
            data1 = simplex_entities_map[sid1]
            if not data1['all_entities']:
                continue
            for j in range(i + 1, len(sorted_sids)):
                sid2 = sorted_sids[j]
                if sid2 in subsets_to_remove:
                    continue
                data2 = simplex_entities_map[sid2]
                if not data2['all_entities']:
                    continue
                if (data2['all_entities'].issubset(data1['all_entities']) and
                    len(data2['all_entities']) < len(data1['all_entities']) and
                    (data2['matched_entities'].issubset(data1['matched_entities']) or
                     data2['dimension'] < data1['dimension'])):
                    subsets_to_remove.add(sid2)
                    dim_diff = data1['dimension'] - data2['dimension']
                    parent_bonus[sid1] += 0.1 + dim_diff * 0.03

                    child_sources = self._parse_source_ids(
                        data2['result'].get('source') or data2['result'].get('source_id')
                    )
                    parent_merged_sources[sid1].update(child_sources)

                    if data2['result'].get('is_seed', False):
                        seed_has_parent.add(sid2)

        for sid, bonus in parent_bonus.items():
            if sid in simplex_entities_map:
                parent_result = simplex_entities_map[sid]['result']
                parent_result['level_hg'] = parent_result.get('level_hg', 0.5) + bonus
                parent_result['parent_bonus'] = bonus

        for sid, merged_sources in parent_merged_sources.items():
            if sid in simplex_entities_map:
                parent_result = simplex_entities_map[sid]['result']
                existing_sources = set(self._parse_source_ids(
                    parent_result.get('source') or parent_result.get('source_id')
                ))
                all_sources = existing_sources | merged_sources
                if all_sources:
                    parent_result['source'] = "<SEP>".join(sorted(all_sources))
                    parent_result['source_id'] = parent_result['source']

        orphan_seeds = set()
        for sid, data in simplex_entities_map.items():
            if data['result'].get('is_seed', False) and sid not in seed_has_parent and sid in subsets_to_remove:
                orphan_seeds.add(sid)

        if orphan_seeds:
            subsets_to_remove -= orphan_seeds
            logger.info(f"保留 {len(orphan_seeds)} 个孤立种子实体（无父复形包含）")

        if subsets_to_remove:
            original_count = len(completion_results)
            completion_results = [r for r in completion_results if r.get('simplex_id', id(r)) not in subsets_to_remove]
            removed_count = original_count - len(completion_results)
            total_merged = sum(len(v) for v in parent_merged_sources.values())
            logger.info(f"移除了 {removed_count} 个子复形，合并 {total_merged} 个source_id "
                        f"到 {len(parent_bonus)} 个父复形")

        return completion_results

    async def _semantic_retrieve(self, query_text: str, top_k: int = None, seed_vertex_ids: list = None) -> dict:
        """语义内容维度检索：基于嵌入余弦相似度检索与查询语义相关的复形

        优化策略：当提供种子节点时，优先计算种子节点拓扑邻近范围内的复形，
        仅当邻近范围不足时才扩展到全量复形，避免每次查询对全量复形做暴力扫描。

        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数（None则使用配置默认值）
            seed_vertex_ids: 查询种子节点ID列表（用于拓扑邻近性预过滤和加分）

        Returns:
            {simplex_id: semantic_score} 字典，按相似度降序排列
        """
        if not query_text or not query_text.strip():
            logger.warning("_semantic_retrieve: 查询文本为空，跳过语义检索")
            return {}

        if top_k is None:
            top_k = DualDimensionConfig.SEMANTIC_RETRIEVE_TOP_K

        self._build_simplex_text_cache()

        candidate_ids = None
        if seed_vertex_ids:
            seed_set = set(seed_vertex_ids)
            topo_nearby_ids = set()
            for vid in seed_vertex_ids:
                if vid in self.hsc.simplices:
                    coboundary = self.hsc.simplices[vid].get('coboundary', [])
                    topo_nearby_ids.update(coboundary)
                    for edge_id in coboundary:
                        if edge_id in self.hsc.simplices:
                            edge_coboundary = self.hsc.simplices[edge_id].get('coboundary', [])
                            topo_nearby_ids.update(edge_coboundary)
            topo_nearby_ids.discard(None)
            if len(topo_nearby_ids) >= top_k * 0.5:
                candidate_ids = topo_nearby_ids
                logger.info(f"语义检索预过滤：种子节点拓扑邻近范围内 {len(candidate_ids)} 个候选复形")

        if candidate_ids is None:
            candidate_ids = set(self._simplex_text_cache.keys())
            logger.info(f"语义检索：邻近范围不足，使用全量 {len(candidate_ids)} 个复形")

        simplex_ids = [sid for sid in candidate_ids if sid in self._simplex_text_cache]
        simplex_texts = [self._simplex_text_cache[sid] for sid in simplex_ids]

        if not simplex_texts:
            logger.warning("_semantic_retrieve: HSC中没有可检索的复形文本")
            return {}

        logger.info(f"语义内容维度：计算 {len(simplex_texts)} 个复形的语义相似度（使用嵌入缓存）")

        query_embedding = await self._get_query_embedding(query_text)
        if query_embedding is None:
            logger.warning("_semantic_retrieve: 查询嵌入计算失败，跳过语义检索")
            return {}

        simplex_embeddings = await self._get_simplex_embeddings_batch(simplex_ids, simplex_texts)
        if not simplex_embeddings:
            logger.warning("_semantic_retrieve: 无可用复形嵌入")
            return {}

        emb_ids = list(simplex_embeddings.keys())
        emb_matrix = np.stack([simplex_embeddings[sid] for sid in emb_ids])
        emb_norms = np.linalg.norm(emb_matrix, axis=1)
        query_norm = float(np.linalg.norm(query_embedding))

        if query_norm <= 0 or bool(np.all(emb_norms <= 0)):
            return {}

        valid_mask = emb_norms > 0
        cos_scores = np.zeros(len(emb_ids))
        cos_scores[valid_mask] = emb_matrix[valid_mask] @ query_embedding / (emb_norms[valid_mask] * query_norm)

        similarities = {emb_ids[i]: float(cos_scores[i]) for i in range(len(emb_ids)) if cos_scores[i] > 0}

        logger.info(f"语义内容维度：矩阵批量计算完成，有效相似度 {len(similarities)} 个 "
                    f"(候选 {len(simplex_ids)}，有嵌入 {len(simplex_embeddings)})")

        if not similarities:
            logger.warning("_semantic_retrieve: 语义相似度计算失败")
            return {}

        if seed_vertex_ids and similarities:
            seed_set = set(seed_vertex_ids)
            topo_boosted = 0
            for sid in list(similarities.keys()):
                simplex_data = self.hsc.simplices.get(sid, {})
                simplex_nodes = set(simplex_data.get('nodes', simplex_data.get('entities', [])))
                if simplex_nodes & seed_set:
                    similarities[sid] *= 1.3
                    topo_boosted += 1
                else:
                    coboundary = simplex_data.get('coboundary', [])
                    boundary = simplex_data.get('boundary', [])
                    for neighbor_id in coboundary + boundary:
                        neighbor_data = self.hsc.simplices.get(neighbor_id, {})
                        neighbor_nodes = set(neighbor_data.get('nodes', neighbor_data.get('entities', [])))
                        if neighbor_nodes & seed_set:
                            similarities[sid] *= 1.15
                            topo_boosted += 1
                            break
            if topo_boosted > 0:
                logger.info(f"语义检索拓扑加分：{topo_boosted} 个复形获得拓扑邻近性加分")

        min_similarity = DualDimensionConfig.SEMANTIC_MIN_SIMILARITY
        filtered_sims = {sid: score for sid, score in similarities.items()
                         if score >= min_similarity}

        if not filtered_sims and similarities:
            sorted_all = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            fallback_count = min(max(top_k // 5, 3), len(sorted_all))
            filtered_sims = dict(sorted_all[:fallback_count])
            logger.info(f"语义内容维度：无复形达到阈值{min_similarity}，"
                        f"回退取top-{fallback_count}（最高分{sorted_all[0][1]:.3f}）")

        sorted_sims = sorted(filtered_sims.items(), key=lambda x: x[1], reverse=True)
        result = dict(sorted_sims[:top_k])

        logger.info(f"语义内容维度：检索到 {len(result)} 个语义相关复形 "
                    f"(阈值={min_similarity}, top_k={top_k})")
        if result:
            top_score = list(result.values())[0]
            logger.info(f"最高语义相似度: {top_score:.3f}")

        return result

    @staticmethod
    def _normalize_scores(scores_dict: dict) -> dict:
        """分位数归一化得分，消除极端值影响

        使用95分位数替代最大值进行归一化，避免单个极端高分
        压缩其他所有复形的归一化得分。当得分不足5个时回退到最大值归一化。
        支持负值得分：先做min-max平移到非负区间，再除以参考值。
        零值保护：当所有得分相同时返回均匀分布。

        Args:
            scores_dict: {id: score} 字典

        Returns:
            归一化后的 {id: normalized_score} 字典，值域[0, 1]
        """
        if not scores_dict:
            return {}
        values = np.array(list(scores_dict.values()))
        # 平移到非负区间：减去最小值
        min_val = float(np.min(values))
        shifted = {k: v - min_val for k, v in scores_dict.items()}
        shifted_values = np.array(list(shifted.values()))
        if len(shifted_values) >= 5:
            reference = float(np.percentile(shifted_values, 95))
        else:
            reference = float(np.max(shifted_values))
        if reference <= 0:
            # 所有得分相同，返回均匀分布
            n = len(scores_dict)
            return {k: 1.0 / n for k in scores_dict}
        return {k: min(v / reference, 1.0) for k, v in shifted.items()}

    def _compute_topology_quality(self, topology_simplices, vertex_ids, weighted_scores) -> float:
        """多因子拓扑维度质量评估

        综合覆盖率、得分集中度和维度丰富度三个因子，
        比单一数量比例更准确地反映拓扑维度的匹配质量。

        Args:
            topology_simplices: 拓扑维度检索到的复形集合
            vertex_ids: 查询顶点ID列表
            weighted_scores: 拓扑维度加权得分

        Returns:
            拓扑维度质量得分（0~1）
        """
        if not topology_simplices or not vertex_ids:
            return 0.0

        # 覆盖率：查询顶点中被拓扑结果覆盖的比例
        # 而非拓扑结果数量与查询顶点数量的比例
        covered_vertices = set()
        for sid in topology_simplices:
            if sid in self.hsc.simplices:
                simplex_data = self.hsc.simplices[sid]
                nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
                for node in nodes:
                    if node in vertex_ids:
                        covered_vertices.add(node)
        coverage = len(covered_vertices) / max(len(vertex_ids), 1)

        concentration = 0.0
        if weighted_scores:
            scores = list(weighted_scores.values())
            if scores:
                mean_score = np.mean(scores)
                if mean_score > 0:
                    concentration = sum(1 for s in scores if s > mean_score) / len(scores)

        dims_covered = set()
        for sid in topology_simplices:
            if sid in self.hsc.simplices:
                dims_covered.add(self.hsc.simplices[sid].get('dimension', 0))
        diversity = min(1.0, len(dims_covered) / 3.0) if dims_covered else 0.0

        return 0.5 * coverage + 0.3 * concentration + 0.2 * diversity

    def _compute_semantic_quality(self, semantic_scores) -> float:
        """多因子语义维度质量评估

        综合高相似度占比和得分分布形态两个因子，
        比简单的高分计数更准确。

        Args:
            semantic_scores: 语义维度得分字典

        Returns:
            语义维度质量得分（0~1）
        """
        if not semantic_scores:
            return 0.0
        values = list(semantic_scores.values())
        high_ratio = sum(1 for s in values if s >= 0.5) / max(len(values), 1)
        mean_score = np.mean(values)
        distribution_quality = min(1.0, mean_score * 2)
        return 0.6 * high_ratio + 0.4 * distribution_quality

    def _fusion_topology_semantic(
        self,
        topology_simplices: set,
        topology_weighted_scores: dict,
        semantic_scores: dict,
        vertex_ids: list,
        coboundary_threshold: float = 0.5
    ) -> tuple:
        """融合拓扑结构维度和语义内容维度的检索结果

        拓扑维度提供结构连贯性：通过图结构连接找到相关复形
        语义维度提供内容相关性：通过文本相似度找到相关复形
        两个维度互补，融合后提供更全面的检索结果。

        融合策略：
        1. 收集两个维度的所有候选复形
        2. 对每个复形计算拓扑得分和语义得分
        3. 根据维度质量动态调整权重
        4. 加权融合得到最终得分

        Args:
            topology_simplices: 拓扑维度检索到的复形集合
            topology_weighted_scores: 拓扑维度加权得分
            semantic_scores: 语义维度得分
            vertex_ids: 查询顶点ID列表
            coboundary_threshold: 融合阈值

        Returns:
            (fusion_simplices, final_scores) 融合后的复形集合和得分
        """
        all_simplices = set(topology_simplices) | set(semantic_scores.keys())

        if not all_simplices:
            return set(), {}

        topology_quality = self._compute_topology_quality(
            topology_simplices, vertex_ids, topology_weighted_scores
        )
        semantic_quality = self._compute_semantic_quality(semantic_scores)

        weight_topo, weight_sem = DualDimensionConfig.compute_topology_semantic_weights(
            topology_quality, semantic_quality
        )

        logger.info(f"拓扑-语义融合：拓扑复形数={len(topology_simplices)}, "
                    f"语义复形数={len(semantic_scores)}, "
                    f"权重: topo={weight_topo:.2f}, sem={weight_sem:.2f} "
                    f"(质量: topo={topology_quality:.2f}, sem={semantic_quality:.2f})")

        topo_normalized = self._normalize_scores(topology_weighted_scores)
        sem_normalized = self._normalize_scores(semantic_scores)

        final_scores = {}
        sem_only_simplices = all_simplices - topology_simplices
        for simplex_id in all_simplices:
            topo_score = topo_normalized.get(simplex_id, 0.0)
            sem_score = sem_normalized.get(simplex_id, 0.0)
            final_scores[simplex_id] = weight_topo * topo_score + weight_sem * sem_score

        if final_scores:
            scores_values = list(final_scores.values())
            p20 = float(np.percentile(scores_values, 20))
            adaptive_threshold = min(coboundary_threshold, p20)
            p5 = float(np.percentile(scores_values, 5))
            adaptive_threshold = max(adaptive_threshold, p5)
        else:
            adaptive_threshold = coboundary_threshold

        fusion_simplices = {
            sid for sid, score in final_scores.items()
            if score >= adaptive_threshold
        }

        for sid in sem_only_simplices:
            if sid in sem_normalized and sem_normalized[sid] >= 0.5:
                fusion_simplices.add(sid)

        if not fusion_simplices and final_scores:
            min_score = min(final_scores.values())
            fusion_simplices = {
                sid for sid, score in final_scores.items()
                if score >= min_score
            }
            logger.info(f"降低阈值到 {min_score:.4f}，筛选后复形数: {len(fusion_simplices)}")

        logger.info(f"拓扑-语义融合完成：融合后复形数={len(fusion_simplices)}")

        return fusion_simplices, final_scores

    def _truncate_with_seed_protection(self, completion_results: list, max_total: int) -> list:
        """基于重要性的截断：综合维度和得分排序，不强制保留种子

        种子实体不再强制优先，因为：
        1. 被高维复形包含的种子实体，其source已合并到父复形，信息不丢失
        2. 孤立种子实体已在_remove_sub_simplices中被保护
        3. 实体信息已通过[Retrieved Entities]单独注入prompt

        Args:
            completion_results: 已排序的补全结果列表
            max_total: 最大总数

        Returns:
            截断后的结果列表
        """
        completion_results.sort(key=lambda x: (
            x['dimension'],
            x.get('diffusion_score', 0) + x.get('coboundary_score', 0),
            len(x.get('matched_vertices', [])),
        ), reverse=True)

        return completion_results[:max_total]

    async def dual_dimension_retrieve(
        self,
        query_vertices: List[dict],
        query_partial_relations: List[dict],
        coboundary_threshold: float = 0.5,
        type_filter: List[str] = None,
        query_text: str = None
    ) -> dict:
        """拓扑结构维度检索核心：仅执行拓扑A/B维度融合，不包含语义检索

        架构设计：
        语义检索和拓扑-语义对撞融合统一由外层 topology_retrieval 处理，
        避免语义信号在多层融合中被反复稀释。
        本方法只负责拓扑结构维度内部的实体维度A + 关系维度B融合。

        流程：
        1. 分发：实体匹配（维度A）+ 关系匹配（维度B）
        2. 扩散：Hodge Laplacian拓扑能量扩散
        3. 对撞：上边界收缩 + 维度A/B动态权重融合
        4. 补全：维度提升 → 拓扑链补全 → Common Coboundary

        Args:
            query_vertices: 查询顶点列表（可包含_virtual_weight_factor降权标记）
            query_partial_relations: 查询中的部分关系列表
            coboundary_threshold: 对撞阈值
            type_filter: 类型过滤列表
            query_text: 查询文本（仅用于扩散中的语义引导修正和上边界收缩的语义相似度）

        Returns:
            包含查询实体、补全结果和拓扑融合得分的字典
        """
        vertex_ids = [v['id'] for v in query_vertices if 'id' in v and v.get('id', '').strip()]

        # 构建虚拟节点降权映射：虚拟节点在扩散结果中降权
        virtual_weight_map = {}
        for v in query_vertices:
            vid = v.get('id', '')
            factor = v.get('_virtual_weight_factor', 1.0)
            if factor < 1.0 and vid:
                virtual_weight_map[vid] = factor

        if not vertex_ids:
            logger.warning("No valid query vertices provided for dual_dimension_retrieve")
            return {'query_entities': query_vertices, 'completion_results': [], 'topology_weighted_scores': {}}

        enhanced_relations = query_partial_relations.copy()

        logger.info(f"开始拓扑维度检索，查询顶点数: {len(vertex_ids)}, "
                    f"原始关系数: {len(query_partial_relations)}, "
                    f"查询文本: {'有' if query_text else '无'}")

        # ===== 步骤1：分发 - 实体匹配和关系匹配 =====
        seed_nodes = vertex_ids
        logger.info(f"拓扑结构维度 - 语义点火：激活 {len(seed_nodes)} 个种子节点")

        relation_to_simplices, seed_edge_ids, seed_high_dim_simplices = \
            self._match_relations_to_simplices(enhanced_relations)

        # ===== 步骤2：扩散 - Hodge Laplacian拓扑能量扩散 =====
        diffused_node_scores, diffused_edge_scores, diffused_high_dim_scores = \
            await self._execute_diffusion(seed_nodes, seed_edge_ids, seed_high_dim_simplices, relation_to_simplices, query_text=query_text)

        # 虚拟节点降权：对扩散结果中虚拟种子节点直接贡献的得分进行衰减
        if virtual_weight_map:
            for vid, factor in virtual_weight_map.items():
                if vid in diffused_node_scores:
                    diffused_node_scores[vid] *= factor
            logger.info(f"虚拟节点降权：{len(virtual_weight_map)}个虚拟节点的扩散得分乘以0.3")

        # ===== 步骤3：对撞 - 上边界收缩 + 拓扑内部维度A/B融合 =====
        (candidate_simplices, simplex_coverage, strict_intersection,
         simplex_matched_vertices, simplex_similarity) = \
            await self._compute_coboundary_contraction(vertex_ids, type_filter, query_text=query_text)

        filtered_simplices = self._filter_candidates(
            vertex_ids, simplex_coverage, strict_intersection, simplex_similarity, candidate_simplices
        )

        filtered_candidate_set = {sid for sid, _ in filtered_simplices} | strict_intersection

        pattern_simplices = set(relation_to_simplices.keys())
        common_coboundary = self._compute_common_coboundary(vertex_ids)

        matched_vertex_count = sum(1 for v in vertex_ids if v in self.hsc.nodes)
        vertex_quality = matched_vertex_count / max(len(vertex_ids), 1)
        relation_quality = min(1.0, len(relation_to_simplices) / max(len(query_partial_relations), 1)) if query_partial_relations else 0.0

        # 拓扑维度内部融合（实体维度A + 关系维度B），不再融合语义维度
        topology_fusion_simplices, topology_weighted_scores, simplex_scores = self._fusion_dual_dimensions(
            filtered_candidate_set, common_coboundary, pattern_simplices,
            vertex_ids, relation_to_simplices, diffused_high_dim_scores, coboundary_threshold,
            vertex_quality=vertex_quality, relation_quality=relation_quality,
            simplex_similarity=simplex_similarity
        )

        logger.info(f"拓扑结构维度：检索到 {len(topology_fusion_simplices)} 个拓扑相关复形")

        fusion_simplices = topology_fusion_simplices

        # ===== 步骤4：补全 =====
        if not fusion_simplices:
            logger.info("无交集，尝试维度提升...")
            elevated_simplices = self._try_elevation(
                vertex_ids, pattern_simplices, common_coboundary, seed_edge_ids
            )
            if elevated_simplices:
                fusion_simplices = elevated_simplices
                logger.info(f"维度提升完成：获得 {len(fusion_simplices)} 个提升后的复形")

        if not fusion_simplices:
            logger.info("无交集，尝试拓扑链补全...")
            chain_results = self.find_topological_chain(vertex_ids, max_hops=2)

            if chain_results:
                final_result = {
                    'query_entities': query_vertices,
                    'completion_results': chain_results,
                    'topology_weighted_scores': topology_weighted_scores
                }
                logger.info(f"拓扑链补全：返回 {len(chain_results)} 个单纯形")
                return final_result

            fusion_simplices = common_coboundary
            logger.info(f"使用Common Coboundary作为最终候选: {len(fusion_simplices)} 个")

        completion_results = self._build_completion_results(
            fusion_simplices, vertex_ids, simplex_scores,
            simplex_matched_vertices, diffused_node_scores
        )

        if simplex_similarity:
            for result in completion_results:
                sid = result.get('simplex_id')
                if sid in simplex_similarity:
                    result['semantic_score'] = simplex_similarity[sid]

        completion_results = self._inject_seed_simplices(
            completion_results, vertex_ids, seed_edge_ids, diffused_node_scores
        )

        completion_results = self._remove_sub_simplices(completion_results)

        completion_results.sort(key=lambda x: (
            x['dimension'],
            x.get('diffusion_score', 0) + x.get('coboundary_score', 0),
            x.get('semantic_score', 0),
            1 if x['level_hg'] == 'high' or (isinstance(x['level_hg'], (int, float)) and x['level_hg'] > 0.7) else 0,
            len(x.get('matched_vertices', [])),
        ), reverse=True)

        final_result = {
            'query_entities': query_vertices,
            'completion_results': completion_results,
            'topology_weighted_scores': topology_weighted_scores
        }

        dim_counts = defaultdict(int)
        for result in completion_results:
            dim_counts[result['dimension']] += 1
        dim_info = ", ".join([f"dim={dim}: {count}" for dim, count in sorted(dim_counts.items(), reverse=True)])

        sem_count = sum(1 for r in completion_results if r.get('semantic_score', 0) > 0)
        logger.info(f"拓扑维度检索完成，返回 {len(completion_results)} 个复形结果（{dim_info}），"
                    f"其中 {sem_count} 个包含语义得分")
        return final_result

    def cache_frequent_patterns(self, min_frequency: int = 3):
        """子复形缓存机制：缓存高频出现的结构模式

        Args:
            min_frequency: 最低出现频率阈值
        """
        pattern_frequency = defaultdict(int)
        pattern_data = {}

        for simplex_id, simplex_data in self.hsc.simplices.items():
            nodes = tuple(sorted(simplex_data.get('nodes', simplex_data.get('entities', []))))
            if len(nodes) >= 2:
                pattern_frequency[nodes] += 1
                pattern_data[nodes] = simplex_data

        self._sub_complex_cache = {}
        for pattern, freq in pattern_frequency.items():
            if freq >= min_frequency:
                self._sub_complex_cache[pattern] = {
                    'frequency': freq,
                    'data': pattern_data[pattern],
                    'precomputed_coboundary': self._precompute_coboundary(pattern)
                }

        logger.info(f"子复形缓存完成，缓存了 {len(self._sub_complex_cache)} 个高频模式（频率 >= {min_frequency}）")

    def _precompute_coboundary(self, nodes: tuple) -> Set[str]:
        """预计算给定节点组合的上边界"""
        coboundary = set()
        for simplex_id, simplex_data in self.hsc.simplices.items():
            simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
            if all(node in simplex_nodes for node in nodes):
                coboundary.add(simplex_id)
        return coboundary

    def clear_cache(self):
        """清除子复形缓存"""
        self._sub_complex_cache.clear()
        logger.info("子复形缓存已清除")
