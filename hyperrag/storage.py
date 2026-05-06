import asyncio
import html
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List, Set, Tuple, Optional, Dict
import numpy as np
import scipy.sparse as sp
import concurrent.futures
import time
import threading
import psutil
from nano_vectordb import NanoVectorDB
from .simplex_tree import SimplexTree
from .utils import load_json, logger, write_json, compute_mdhash_id
from .operate._config import normalize_entity_name
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseSimplexStorage
)


class ReadWriteLock:
    """读写锁实现：允许多个读操作并行，写操作独占

    使用threading.Condition实现，支持：
    - 多个读者同时持有读锁
    - 写者独占写锁
    - 写者优先策略：有写者等待时，新读者排队，避免写者饥饿
    - 可重入读锁：同一线程可多次获取读锁（支持递归调用）
    - 可重入写锁：同一线程持有写锁时可再次获取写锁或读锁（锁降级安全）
    """

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._writer_waiting = 0
        self._reader_owners = {}  # thread_id -> count，支持可重入读锁
        self._writer_owner = None  # 持有写锁的线程ID
        self._writer_count = 0  # 写锁重入计数

    def acquire_read(self):
        tid = threading.get_ident()
        with self._cond:
            if tid in self._reader_owners:
                self._reader_owners[tid] += 1
                return
            if self._writer_owner == tid:
                self._reader_owners[tid] = 1
                self._readers += 1
                return
            while self._writers > 0 or self._writer_waiting > 0:
                self._cond.wait()
            self._readers += 1
            self._reader_owners[tid] = 1

    def release_read(self):
        tid = threading.get_ident()
        with self._cond:
            if tid in self._reader_owners:
                self._reader_owners[tid] -= 1
                if self._reader_owners[tid] == 0:
                    del self._reader_owners[tid]
                    self._readers -= 1
                    if self._readers == 0:
                        self._cond.notify_all()

    def acquire_write(self):
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner == tid:
                self._writer_count += 1
                return
            self._writer_waiting += 1
            while self._readers > 0 or self._writers > 0:
                if self._writer_owner == tid:
                    break
                self._cond.wait()
            self._writer_waiting -= 1
            self._writers += 1
            self._writer_owner = tid
            self._writer_count = 1

    def release_write(self):
        tid = threading.get_ident()
        with self._cond:
            if self._writer_owner != tid:
                return
            self._writer_count -= 1
            if self._writer_count == 0:
                self._writers -= 1
                self._writer_owner = None
                self._cond.notify_all()

    class _ReadContext:
        def __init__(self, rwlock):
            self._rwlock = rwlock
        def __enter__(self):
            self._rwlock.acquire_read()
            return self
        def __exit__(self, *args):
            self._rwlock.release_read()

    class _WriteContext:
        def __init__(self, rwlock):
            self._rwlock = rwlock
        def __enter__(self):
            self._rwlock.acquire_write()
            return self
        def __exit__(self, *args):
            self._rwlock.release_write()

    def read_lock(self):
        return self._ReadContext(self)

    def write_lock(self):
        return self._WriteContext(self)

"""
存储模块

本模块实现了基于单纯形树(Simplex Tree)的复形存储系统，
提供高效的复形管理、查询和拓扑关系计算功能。

核心功能：
- 复形的存储和检索
- 边界和上边界计算
- 实体到复形的双向索引
- 拓扑能量扩散
- 拉普拉斯矩阵计算

使用单纯形树作为存储后端，相比传统的HypergraphDB，
提供了更高效的空间使用和更快的查询性能。
"""



@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        # 使用namespace作为子目录
        storage_dir = os.path.join(working_dir, "rag")
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self._file_name = os.path.join(storage_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)
        return data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        storage_dir = os.path.join(working_dir, "rag")
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self._client_file_name = os.path.join(
            storage_dir, f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )
        
        # ===== 性能优化：查询缓存 =====
        self._query_cache = {}  # 查询缓存: {query_hash: (timestamp, results)}
        self._cache_size_limit = 1000  # 缓存最大条目数
        self._cache_ttl = 300  # 缓存有效期（秒）
        import hashlib
        self._hash_func = hashlib.md5
        
        # ===== 性能优化：预加载数据到内存 =====
        self._preloaded = False
        self._preload_data = {}  # 预加载的数据索引

    def _get_query_hash(self, query: str, top_k: int) -> str:
        """计算查询的哈希值用于缓存"""
        return self._hash_func(f"{query}:{top_k}".encode()).hexdigest()

    def _cleanup_cache(self):
        """清理过期或超出限制的缓存"""
        import time
        current_time = time.time()
        
        # 清理过期缓存
        expired_keys = [
            key for key, (timestamp, _) in self._query_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._query_cache[key]
        
        # 如果缓存超出限制，保留最近使用的
        if len(self._query_cache) > self._cache_size_limit:
            # 按时间戳排序，保留最新的
            sorted_items = sorted(
                self._query_cache.items(),
                key=lambda x: x[1][0],
                reverse=True
            )[:self._cache_size_limit // 2]
            self._query_cache = dict(sorted_items)

    async def _preload_index(self):
        """预加载索引到内存，加速查询"""
        if self._preloaded:
            return
        
        logger.info(f"Preloading index for {self.namespace}...")
        import time
        start_time = time.time()
        
        try:
            storage = getattr(self._client, '_NanoVectorDB__storage', None)
            if storage is None:
                storage = getattr(self._client, '_db', None)
            if storage is not None:
                all_data = storage.get('data', [])
                if isinstance(all_data, dict):
                    all_data = all_data.get('datas', [])
            else:
                all_data = []
        except Exception as e:
            logger.warning(f"预加载索引时访问存储失败: {e}")
            all_data = []

        for item in all_data:
            if '__id__' in item:
                self._preload_data[item['__id__']] = item
        
        self._preloaded = True
        end_time = time.time()
        logger.info(f"Preloaded {len(self._preload_data)} items in {end_time - start_time:.2f} seconds")

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        
        # 清空缓存，因为数据已更新
        self._query_cache.clear()
        self._preloaded = False
        
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        
        # 优化：动态调整批次大小，根据数据量和可用内存
        memory_available = psutil.virtual_memory().available / (1024**3)
        dynamic_batch_size = min(
            self._max_batch_size,
            max(8, int(memory_available * 10))  # 每GB内存分配10个批次
        )
        
        batches = [
            contents[i : i + dynamic_batch_size]
            for i in range(0, len(contents), dynamic_batch_size)
        ]
        
        # 限制并发数，避免内存溢出
        max_concurrent_batches = min(len(batches), 8)
        results = []
        
        for i in range(0, len(batches), max_concurrent_batches):
            batch_group = batches[i:i+max_concurrent_batches]
            embeddings_list = await asyncio.gather(
                *[self.embedding_func(batch) for batch in batch_group]
            )
            
            # 合并结果并插入
            merged_data = []
            base_idx = i * dynamic_batch_size
            for j, batch_embeddings in enumerate(embeddings_list):
                for k, embedding in enumerate(batch_embeddings):
                    idx = base_idx + j * dynamic_batch_size + k
                    if idx < len(list_data):
                        list_data[idx]["__vector__"] = embedding
                        merged_data.append(list_data[idx])
            
            if merged_data:
                results.extend(self._client.upsert(datas=merged_data))
        
        return results

    async def query(self, query: str, top_k=5):
        # 检查缓存
        query_hash = self._get_query_hash(query, top_k)
        current_time = time.time()
        
        if query_hash in self._query_cache:
            cache_time, cached_results = self._query_cache[query_hash]
            if current_time - cache_time < self._cache_ttl:
                logger.debug(f"Cache hit for query: {query[:30]}...")
                return cached_results
        
        # 预加载索引（首次查询时）
        if not self._preloaded:
            await self._preload_index()
        
        # 查询嵌入
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        
        # 优化：使用预加载的数据进行快速过滤
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        
        # 更新缓存
        self._query_cache[query_hash] = (current_time, results)
        self._cleanup_cache()
        
        return results

    async def batch_query(self, queries: list[str], top_k=5):
        """批量查询优化"""
        if not queries:
            return []
        
        results = []
        uncached_queries = []
        query_indices = []
        
        # 检查缓存
        for i, query in enumerate(queries):
            query_hash = self._get_query_hash(query, top_k)
            if query_hash in self._query_cache:
                cache_time, cached_results = self._query_cache[query_hash]
                if time.time() - cache_time < self._cache_ttl:
                    results.append(cached_results)
                    continue
            
            uncached_queries.append(query)
            query_indices.append(i)
            results.append(None)  # 占位
        
        # 批量处理未缓存的查询
        if uncached_queries:
            # 批量计算嵌入
            embeddings = await self.embedding_func(uncached_queries)
            
            # 批量查询
            current_time = time.time()
            for i, (query, embedding) in enumerate(zip(uncached_queries, embeddings)):
                query_results = self._client.query(
                    query=embedding,
                    top_k=top_k,
                    better_than_threshold=self.cosine_better_than_threshold,
                )
                query_results = [
                    {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} 
                    for dp in query_results
                ]
                
                # 更新结果列表和缓存
                results[query_indices[i]] = query_results
                query_hash = self._get_query_hash(query, top_k)
                self._query_cache[query_hash] = (current_time, query_results)
            
            self._cleanup_cache()
        
        return results

    async def index_done_callback(self):
        self._client.save()
        logger.info(f"Saved NanoVectorDB for {self.namespace}")


@dataclass
class SimplexStorage(BaseSimplexStorage):
    embedding_func: Optional[Any] = None

    async def _handle_simplex_summary(
        self,
        name: str,
        description: str,
        summary_type: str = "entity_description",
    ) -> str:
        """对过长的描述/属性调用LLM生成摘要

        对齐HyperGraphRAG-main的_handle_entity_relation_summary逻辑，
        当合并后的文本超过token阈值时，调用LLM进行压缩摘要。

        Args:
            name: 实体或关系名称
            description: 需要摘要的文本内容
            summary_type: 摘要类型，支持 "entity_description"/"entity_ap"/"relation_description"

        Returns:
            摘要后的文本或原始文本（如果不需要摘要）
        """
        from .utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken
        from .prompt import GRAPH_FIELD_SEP, PROMPTS

        global_config = self.global_config
        use_llm_func = global_config.get("llm_model_func")
        if use_llm_func is None:
            return description

        tiktoken_model_name = global_config.get("tiktoken_model_name", "gpt-4o-mini")
        llm_max_tokens = global_config.get("llm_model_max_token_size", 32768)

        if summary_type == "entity_description":
            summary_max_tokens = global_config.get("entity_summary_to_max_tokens", 500)
        elif summary_type == "entity_ap":
            summary_max_tokens = global_config.get("entity_additional_properties_to_max_tokens", 250)
        elif summary_type == "relation_description":
            summary_max_tokens = global_config.get("relation_summary_to_max_tokens", 750)
        else:
            summary_max_tokens = 500

        tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
        if len(tokens) < summary_max_tokens:
            return description

        prompt_map = {
            "entity_description": "summarize_entity_descriptions",
            "entity_ap": "summarize_entity_additional_properties",
            "relation_description": "summarize_relation_descriptions",
        }
        prompt_key = prompt_map.get(summary_type, "summarize_entity_descriptions")
        prompt_template = PROMPTS.get(prompt_key)
        if prompt_template is None:
            return description

        use_description = decode_tokens_by_tiktoken(
            tokens[:llm_max_tokens], model_name=tiktoken_model_name
        )

        language = global_config.get("addon_params", {}).get(
            "language", PROMPTS.get("DEFAULT_LANGUAGE", "English")
        )

        if summary_type == "entity_description":
            context_base = dict(
                entity_name=name,
                description_list=use_description.split(GRAPH_FIELD_SEP),
                language=language,
            )
        elif summary_type == "entity_ap":
            context_base = dict(
                entity_name=name,
                additional_properties_list=use_description.split(GRAPH_FIELD_SEP),
                language=language,
            )
        elif summary_type == "relation_description":
            context_base = dict(
                relation_name=name,
                relation_description_list=use_description.split(GRAPH_FIELD_SEP),
                language=language,
            )
        else:
            context_base = dict(
                entity_name=name,
                description_list=use_description.split(GRAPH_FIELD_SEP),
                language=language,
            )

        use_prompt = prompt_template.format(**context_base)
        logger.debug(f"Trigger simplex summary: {name} (type={summary_type})")
        try:
            summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
            return summary
        except Exception as e:
            logger.error(f"LLM summary failed for {name}: {e}")
            return description

    def _get_vertices_from_node(self, node) -> list:
        """从单纯形树节点获取从根到叶的顶点路径"""
        vertices = []
        current = node
        while current != self._hg.root:
            vertices.append(current.vertex)
            current = current.parent
        return list(reversed(vertices))

    @staticmethod
    def load_hypergraph(file_name) -> SimplexTree:
        if os.path.exists(file_name):
            import json
            import numpy as np
            pre_simplex_tree = SimplexTree()
            
            # 加载嵌入向量
            embeddings = {}
            # 尝试加载 .npy 文件
            embedding_file = file_name.replace('.json', '_embeddings.npy')
            if not os.path.exists(embedding_file):
                # 尝试加载 .npz 文件
                embedding_file = file_name.replace('.json', '_embeddings.npy.npz')
            
            if os.path.exists(embedding_file):
                try:
                    with np.load(embedding_file) as data:
                        for key in data.files:
                            embeddings[key] = data[key].tolist()
                    logger.info(f"Loaded {len(embeddings)} embeddings from binary file: {embedding_file}")
                except Exception as e:
                    logger.error(f"Error loading embeddings: {e}")
            
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 处理新的存储结构
                if 'entities' in data:
                    # 加载实体数据
                    for entity_name, entity_data in data['entities'].items():
                        # 获取嵌入向量
                        embedding_id = entity_data.get('embedding_id')
                        embedding = embeddings.get(embedding_id) if embedding_id else entity_data.get('embedding')
                        # 构建完整的顶点数据
                        simplex_data = {
                            'id': entity_data.get('id'),
                            'type': 'simplex',
                            'dimension': 0,
                            'entities': [entity_name],
                            'entity_name': entity_name,
                            'entity_type': entity_data.get('entity_type', entity_data.get('type', 'Entity')),
                            'description': entity_data.get('description', ''),
                            'source': entity_data.get('source', 'unknown'),
                            'source_id': entity_data.get('source_id', entity_data.get('source', '')),
                            'importance': entity_data.get('importance', 0.5),
                            'frequency': entity_data.get('frequency', 1),
                            'embedding': embedding,
                            'embedding_id': entity_data.get('embedding_id'),
                            'boundary': entity_data.get('boundary', []),
                            'coboundary': entity_data.get('coboundary', [])
                        }
                        # 插入到单纯形树
                        pre_simplex_tree.insert([entity_name], simplex_data, simplex_data.get('id'))
                
                if 'relations' in data:
                    # 加载关系数据
                    for relation_id, relation_data in data['relations'].items():
                        entities = relation_data.get('entities', [])
                        if entities:
                            # 确保entities按排序顺序加载，与SimplexTree内部路径一致
                            entities = sorted(entities)
                            # 获取嵌入向量
                            embedding_id = relation_data.get('embedding_id')
                            embedding = embeddings.get(embedding_id) if embedding_id else relation_data.get('embedding')
                            # 构建复形数据
                            simplex_data = {
                                'id': relation_id,
                                'type': 'simplex',
                                'dimension': relation_data.get('dimension', 1),
                                'entities': entities,
                                'description': relation_data.get('description', ''),
                                'source': relation_data.get('source', 'unknown'),
                                'source_id': relation_data.get('source_id', relation_data.get('source', '')),
                                'importance': relation_data.get('importance', 0.5),
                                'frequency': relation_data.get('frequency', 1),
                                'embedding': embedding,
                                'embedding_id': relation_data.get('embedding_id'),
                                'boundary': relation_data.get('boundary', []),
                                'coboundary': relation_data.get('coboundary', []),
                                'layer': relation_data.get('layer', ''),
                                'is_maximal': relation_data.get('is_maximal', False),
                                'completeness': relation_data.get('completeness', 0.75)
                            }
                            # 插入到单纯形树
                            pre_simplex_tree.insert(entities, simplex_data, relation_id)
                
                # 兼容旧的存储结构
                if 'vertices' in data:
                    for vertex_id, vertex_data in data['vertices'].items():
                        vertex_data['id'] = vertex_data.get('id', vertex_id)
                        pre_simplex_tree.insert([vertex_id], vertex_data, vertex_data.get('id'))
                if 'edges' in data:
                    for edge_tuple_str, edge_data in data['edges'].items():
                        edge_tuple = tuple(json.loads(edge_tuple_str))
                        edge_data['id'] = edge_data.get('id', str(edge_tuple))
                        pre_simplex_tree.insert(list(edge_tuple), edge_data, edge_data.get('id'))
            
            return pre_simplex_tree
        return None

    @staticmethod
    def write_hypergraph(hypergraph: SimplexTree, file_name):
        import json
        import numpy as np
        import time
        
        start_time = time.time()
        logger.info(f"Starting to write simplex tree to {file_name}")
        
        # 准备要保存的数据
        data = {
            'entities': {},
            'relations': {}
        }
        
        # 分离嵌入向量
        embeddings = {}
        embedding_counter = 0
        
        # 处理所有单纯形
        all_simplices = hypergraph.get_all_simplices()
        total_simplices = len(all_simplices)
        logger.info(f"Processing {total_simplices} simplices")
        
        processed_simplices = 0
        for vertices, simplex_data in all_simplices:
            dimension = simplex_data.get('dimension', 0)
            if dimension == 0:
                # 处理实体（0-simplex）
                entity_name = vertices[0] if vertices else 'unknown'
                embedding = simplex_data.get('embedding')
                embedding_id = None
                if embedding is not None:
                    embedding_id = f"embedding_{embedding_counter}"
                    embeddings[embedding_id] = embedding
                    embedding_counter += 1
                entity_data = {
                    'id': simplex_data.get('id'),
                    'entity_type': simplex_data.get('entity_type'),
                    'description': simplex_data.get('description'),
                    'source_id': simplex_data.get('source_id', simplex_data.get('source', '')),
                    'importance': simplex_data.get('importance'),
                    'frequency': simplex_data.get('frequency'),
                    'embedding_id': embedding_id,
                    'dimension': 0,
                    'boundary': simplex_data.get('boundary'),
                    'coboundary': simplex_data.get('coboundary')
                }
                if simplex_data.get('source') is not None:
                    entity_data['source'] = simplex_data.get('source')
                data['entities'][entity_name] = entity_data
            else:
                # 处理关系（1-simplex及以上）
                relation_id = simplex_data.get('id', str(vertices))
                embedding = simplex_data.get('embedding')
                embedding_id = None
                if embedding is not None:
                    embedding_id = f"embedding_{embedding_counter}"
                    embeddings[embedding_id] = embedding
                    embedding_counter += 1
                relation_data = {
                    'id': relation_id,
                    'type': simplex_data.get('type'),
                    'entities': vertices,
                    'dimension': dimension,
                    'description': simplex_data.get('description'),
                    'source_id': simplex_data.get('source_id', simplex_data.get('source', '')),
                    'importance': simplex_data.get('importance'),
                    'frequency': simplex_data.get('frequency'),
                    'embedding_id': embedding_id,
                    'boundary': simplex_data.get('boundary'),
                    'coboundary': simplex_data.get('coboundary'),
                    'layer': simplex_data.get('layer', ''),
                    'is_maximal': simplex_data.get('is_maximal', False),
                    'completeness': simplex_data.get('completeness', 0.75)
                }
                if simplex_data.get('source') is not None:
                    relation_data['source'] = simplex_data.get('source')
                data['relations'][relation_id] = relation_data
            
            processed_simplices += 1
            if processed_simplices % 100 == 0:
                logger.info(f"Processed {processed_simplices}/{total_simplices} simplices")
        
        logger.info(
            f"Processed simplex tree with {len(data['entities'])} entities, {len(data['relations'])} relations"
        )
        
        # 保存到JSON文件
        json_start_time = time.time()
        logger.info(f"Starting to save JSON file")
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        json_end_time = time.time()
        logger.info(f"Saved JSON file in {json_end_time - json_start_time:.2f} seconds")
        
        # 保存嵌入向量到二进制文件
        if embeddings:
            embedding_start_time = time.time()
            logger.info(f"Starting to save {len(embeddings)} embeddings")
            embedding_file = file_name.replace('.json', '_embeddings.npy')
            # 转换为numpy数组并保存
            embedding_dict = {}
            for emb_id, emb in embeddings.items():
                embedding_dict[emb_id] = np.array(emb, dtype=np.float32)
            np.savez_compressed(embedding_file, **embedding_dict)
            embedding_end_time = time.time()
            logger.info(f"Saved {len(embeddings)} embeddings to binary file: {embedding_file} in {embedding_end_time - embedding_start_time:.2f} seconds")
        
        end_time = time.time()
        logger.info(f"Completed writing hypergraph in {end_time - start_time:.2f} seconds")

    def __post_init__(self):
        # 将存储文件改为 .json 格式，使用rag作为子目录
        self._hgdb_dir = os.path.join(
            self.global_config["working_dir"], "rag"
        )
        # 确保目录存在
        if not os.path.exists(self._hgdb_dir):
            os.makedirs(self._hgdb_dir)
        self._hgdb_file = os.path.join(
            self._hgdb_dir, f"simplex_{self.namespace}.json"
        )
        # 加载或创建 SimplexTree
        preloaded_simplex_tree = SimplexStorage.load_hypergraph(self._hgdb_file)
        self._hg = preloaded_simplex_tree or SimplexTree()
        # 计算单纯形数量
        simplex_count = self._hg.size()
        logger.info(f"Load Simplex storage {self.namespace} with {simplex_count} simplices")
        
        # 初始化拉普拉斯矩阵缓存
        self._laplacian_cache = None
        self._cache_timestamp = 0
        self._data_version = simplex_count  # 基于数据量生成版本号，避免进程重启后归零
        
        # 初始化双向索引结构
        self._entity_to_simplices = defaultdict(list)  # 实体到复形的映射
        self._simplex_to_entities = dict()  # 复形到实体的映射
        self._index_cache = dict()  # 索引缓存，提高查询效率
        self._cache_size_limit = 10000  # 缓存大小限制，防止内存溢出
        
        # 初始化倒排索引，加速按字段查询
        self._dimension_index = defaultdict(set)  # 维度 -> 单纯形ID集合
        self._verification_status_index = defaultdict(set)  # 校验状态 -> 单纯形ID集合
        self._chunk_id_index = defaultdict(set)  # chunk_id -> 单纯形ID集合
        
        # 构建初始索引
        self._build_indexes()
        
        # 添加锁机制，确保数据一致性
        self._lock = ReadWriteLock()  # 读写锁：读操作并行，写操作独占
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """规范化实体名称，使用全局统一的标准化函数

        委托给_config.normalize_entity_name，确保与extraction和retrieval
        使用相同的标准化策略（大写+去空格），解决名称匹配不一致的问题。

        Args:
            entity_name: 原始实体名称

        Returns:
            规范化后的实体名称
        """
        return normalize_entity_name(entity_name)
    
    def _match_entity(self, entity_name: str, candidate_entity: str) -> bool:
        """
        匹配实体名称，考虑规范化和部分匹配
        
        Args:
            entity_name: 原始实体名称
            candidate_entity: 候选实体名称
            
        Returns:
            是否匹配
        """
        if not entity_name or not candidate_entity:
            return False
        
        # 规范化两个实体名称
        normalized_entity = self._normalize_entity_name(entity_name)
        normalized_candidate = self._normalize_entity_name(candidate_entity)
        
        # 完全匹配
        if normalized_entity == normalized_candidate:
            return True
        
        # 部分匹配（子字符串）
        if normalized_entity in normalized_candidate or normalized_candidate in normalized_entity:
            return True
        
        # 单词级匹配
        entity_words = set(normalized_entity.split())
        candidate_words = set(normalized_candidate.split())
        
        # 如果有共同的单词，并且其中一个是另一个的子集
        if entity_words.intersection(candidate_words) and (
            entity_words.issubset(candidate_words) or candidate_words.issubset(entity_words)
        ):
            return True
        
        return False
    
    def _build_indexes(self):
        """
        构建实体到复形的双向索引及倒排索引
        """
        logger.info("Building entity-simplex indexes...")
        
        self._entity_to_simplices.clear()
        self._simplex_to_entities.clear()
        self._dimension_index.clear()
        self._verification_status_index.clear()
        self._chunk_id_index.clear()
        
        all_simplices = self._hg.get_all_simplices()
        for vertices, simplex_data in all_simplices:
            simplex_id = simplex_data.get('id')
            if not simplex_id:
                continue
            
            entities = vertices
            for entity in entities:
                self._entity_to_simplices[entity].append(simplex_id)
            self._simplex_to_entities[simplex_id] = entities
            
            self._dimension_index[simplex_data.get('dimension', 0)].add(simplex_id)
            
            vs = simplex_data.get('verification_status')
            if vs:
                self._verification_status_index[vs].add(simplex_id)
            
            for cid in simplex_data.get('chunk_ids', []):
                self._chunk_id_index[cid].add(simplex_id)
            source_id = simplex_data.get('source_id', '')
            if source_id:
                for sid in source_id.split('<SEP>'):
                    sid = sid.strip()
                    if sid:
                        self._chunk_id_index[sid].add(simplex_id)
        
        logger.info(f"Built indexes: {len(self._entity_to_simplices)} entities mapped to {len(self._simplex_to_entities)} simplices")
        
        self._consistency_check_enabled = True

    def _add_to_inverted_indexes(self, simplex_id: str, simplex_data: dict):
        """将单纯形添加到倒排索引中"""
        self._dimension_index[simplex_data.get('dimension', 0)].add(simplex_id)
        vs = simplex_data.get('verification_status')
        if vs:
            self._verification_status_index[vs].add(simplex_id)
        for cid in simplex_data.get('chunk_ids', []):
            self._chunk_id_index[cid].add(simplex_id)
        source_id = simplex_data.get('source_id', '')
        if source_id:
            for sid in source_id.split('<SEP>'):
                sid = sid.strip()
                if sid:
                    self._chunk_id_index[sid].add(simplex_id)

    def _remove_from_inverted_indexes(self, simplex_id: str, simplex_data: dict):
        """从倒排索引中移除单纯形"""
        dim = simplex_data.get('dimension', 0)
        self._dimension_index[dim].discard(simplex_id)
        if not self._dimension_index[dim]:
            del self._dimension_index[dim]
        vs = simplex_data.get('verification_status')
        if vs and vs in self._verification_status_index:
            self._verification_status_index[vs].discard(simplex_id)
            if not self._verification_status_index[vs]:
                del self._verification_status_index[vs]
        for cid in simplex_data.get('chunk_ids', []):
            if cid in self._chunk_id_index:
                self._chunk_id_index[cid].discard(simplex_id)
                if not self._chunk_id_index[cid]:
                    del self._chunk_id_index[cid]
        source_id = simplex_data.get('source_id', '')
        if source_id:
            for sid in source_id.split('<SEP>'):
                sid = sid.strip()
                if sid and sid in self._chunk_id_index:
                    self._chunk_id_index[sid].discard(simplex_id)
                    if not self._chunk_id_index[sid]:
                        del self._chunk_id_index[sid]

    def _get_laplacian_cache_file(self) -> str:
        """获取拉普拉斯矩阵缓存文件路径"""
        return os.path.join(self._hgdb_dir, f"simplex_{self.namespace}_laplacian_cache.npz")

    def _load_laplacian_cache_from_file(self) -> dict:
        """从文件加载拉普拉斯矩阵缓存（使用JSON序列化避免pickle安全风险）"""
        cache_file = self._get_laplacian_cache_file()
        npz_file = cache_file
        json_file = cache_file.replace('.npz', '.json')
        if not os.path.exists(npz_file):
            return None
        
        try:
            logger.info(f"Loading Laplacian cache from file: {npz_file}")
            start_time = time.time()
            
            with np.load(npz_file, allow_pickle=False) as data:
                version = int(data['version'])
                if version != self._data_version:
                    logger.info(f"Cache version mismatch: {version} vs {self._data_version}")
                    return None
                
                L0 = sp.csr_matrix((data['L0_data'], data['L0_indices'], data['L0_indptr']), shape=tuple(data['L0_shape']))
                L1 = sp.csr_matrix((data['L1_data'], data['L1_indices'], data['L1_indptr']), shape=tuple(data['L1_shape']))
            
            nodes = {}
            simplices = {}
            if os.path.exists(json_file):
                import json
                logger.info(f"Loading nodes/simplices from JSON file: {json_file}")
                json_start = time.time()
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    nodes = meta.get('nodes', {})
                    simplices = meta.get('simplices', {})
                json_elapsed = time.time() - json_start
                logger.info(f"JSON file loaded in {json_elapsed:.2f}s (nodes={len(nodes)}, simplices={len(simplices)})")
            else:
                logger.warning(f"JSON cache file not found: {json_file}")
            
            if not nodes or not simplices:
                logger.warning(
                    f"Laplacian cache file exists but nodes/simplices are empty "
                    f"(nodes={len(nodes)}, simplices={len(simplices)}), forcing rebuild"
                )
                return None
            
            end_time = time.time()
            logger.info(f"Loaded Laplacian cache in {end_time - start_time:.2f} seconds")
            
            return {
                'version': version,
                'L0': L0,
                'L1': L1,
                'nodes': nodes,
                'simplices': simplices,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to load Laplacian cache from file: {e}")
            return None

    def _save_laplacian_cache_to_file(self, cache: dict):
        """将拉普拉斯矩阵缓存保存到文件（使用JSON序列化避免pickle安全风险，原子写入防损坏）"""
        cache_file = self._get_laplacian_cache_file()
        json_file = cache_file.replace('.npz', '.json')
        try:
            logger.info(f"Saving Laplacian cache to file: {cache_file}")
            start_time = time.time()
            
            L0 = cache['L0']
            L1 = cache['L1']
            
            np.savez_compressed(
                cache_file,
                version=np.array(cache['version']),
                L0_data=L0.data,
                L0_indices=L0.indices,
                L0_indptr=L0.indptr,
                L0_shape=np.array(L0.shape),
                L1_data=L1.data,
                L1_indices=L1.indices,
                L1_indptr=L1.indptr,
                L1_shape=np.array(L1.shape),
            )
            
            import json, tempfile
            # 原子写入：先写临时文件，再重命名，防止并发读写导致文件损坏
            temp_json = json_file + '.tmp'
            try:
                with open(temp_json, 'w', encoding='utf-8') as f:
                    json.dump({
                        'nodes': cache.get('nodes', {}),
                        'simplices': cache.get('simplices', {}),
                    }, f, ensure_ascii=False, default=str)
                # 原子重命名（Windows需要先删除目标文件）
                if os.path.exists(json_file):
                    os.remove(json_file)
                os.rename(temp_json, json_file)
            except Exception:
                if os.path.exists(temp_json):
                    os.remove(temp_json)
                raise
            
            end_time = time.time()
            logger.info(f"Saved Laplacian cache in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to save Laplacian cache to file: {e}")

    async def get_cached_laplacians(self):
        """获取缓存的拉普拉斯矩阵（二部图简化Laplacian）

        优先从内存缓存加载，其次从文件加载，最后从SimplexStorage中重建。
        返回结构包含 L_entity, L_msg, entity_index, msg_index 以及传统的 nodes/simplices。
        """
        current_version = self._data_version

        if self._laplacian_cache and self._laplacian_cache.get('version') == current_version:
            if self._laplacian_cache.get('simplices') and self._laplacian_cache.get('nodes'):
                logger.debug("Using in-memory Laplacian cache")
                return self._laplacian_cache
            else:
                logger.warning("In-memory Laplacian cache has empty simplices/nodes, forcing rebuild")
                self._laplacian_cache = None

        file_cache = self._load_laplacian_cache_from_file()
        if file_cache and file_cache.get('version') == current_version:
            if file_cache.get('simplices') and file_cache.get('nodes'):
                logger.info("Using file-based Laplacian cache")
                self._laplacian_cache = file_cache
                return self._laplacian_cache
            else:
                logger.warning("File-based Laplacian cache has empty simplices/nodes, forcing rebuild")

        logger.info("Building bipartite Laplacian matrices from SimplexStorage...")
        start_time = time.time()

        from .operate import HeterogeneousSimplicialComplex
        hsc = HeterogeneousSimplicialComplex()

        all_simplices = await self.get_all_simplices()

        for simplex_id, simplex_data in all_simplices:
            hsc.simplices[simplex_id] = simplex_data

        for simplex_id, simplex_data in all_simplices:
            nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
            entity_type = simplex_data.get('entity_type', 'Entity')
            for node in nodes:
                if node not in hsc.nodes:
                    hsc.nodes[node] = {'type': entity_type, 'vector': []}
                if node not in hsc.simplices:
                    hsc.simplices[node] = {
                        'id': node,
                        'dimension': 0,
                        'entities': [node],
                        'nodes': [node],
                        'type': entity_type,
                        'entity_type': entity_type,
                        'boundary': [],
                        'coboundary': [],
                        'importance': simplex_data.get('importance', 1.0),
                        'frequency': simplex_data.get('frequency', 1)
                    }

        for simplex_id, simplex_data in all_simplices:
            dim = simplex_data.get('dimension', 0)
            if dim >= 1:
                entities = simplex_data.get('nodes', simplex_data.get('entities', []))
                if not entities or len(entities) < 2:
                    continue
                boundary_ids = []
                try:
                    from itertools import combinations
                    sorted_entities = sorted(entities)
                    for sub_entities in combinations(sorted_entities, len(sorted_entities) - 1):
                        sub_node = self._hg.find(list(sub_entities))
                        if sub_node and sub_node.simplex_data:
                            sub_id = sub_node.simplex_data.get('id')
                            if sub_id and sub_id not in boundary_ids:
                                boundary_ids.append(sub_id)
                except Exception as e:
                    logger.error(f"计算复形 {simplex_id} 的boundary失败: {e}")
                hsc.simplices[simplex_id]['boundary'] = boundary_ids

        for simplex_id, simplex_data in all_simplices:
            dim = simplex_data.get('dimension', 0)
            entities = simplex_data.get('nodes', simplex_data.get('entities', []))
            if not entities:
                continue
            coboundary_ids = []
            try:
                sorted_entities = sorted(entities)
                coboundary_vertices_list = self._hg.get_coboundary(sorted_entities)
                for coboundary_vertices in coboundary_vertices_list:
                    coboundary_node = self._hg.find(coboundary_vertices)
                    if coboundary_node and coboundary_node.simplex_data:
                        coboundary_id = coboundary_node.simplex_data.get('id')
                        if coboundary_id and coboundary_id not in coboundary_ids:
                            coboundary_ids.append(coboundary_id)
            except Exception as e:
                logger.error(f"计算复形 {simplex_id} 的coboundary失败: {e}")
            hsc.simplices[simplex_id]['coboundary'] = coboundary_ids

        hsc.build_incidence_matrices()
        hsc.compute_hodge_laplacians()

        # 构建二部图Laplacian
        entity_names = sorted(hsc.nodes.keys())
        msg_simplices = [(sid, sdata) for sid, sdata in hsc.simplices.items()
                         if sdata.get('dimension', 0) >= 1 and sdata.get('is_maximal', False)]
        msg_ids = [sid for sid, _ in msg_simplices]
        entity_index = {name: i for i, name in enumerate(entity_names)}
        msg_index = {mid: i for i, mid in enumerate(msg_ids)}

        n_entities = len(entity_names)
        n_msgs = len(msg_ids)

        B = np.zeros((n_msgs, n_entities), dtype=np.float32)
        for i, (sid, sdata) in enumerate(msg_simplices):
            entities = sdata.get('entities', sdata.get('nodes', []))
            for entity_name in entities:
                if entity_name in entity_index:
                    B[i, entity_index[entity_name]] = 1.0

        L_entity = B.T @ B
        L_msg = B @ B.T

        self._laplacian_cache = {
            'version': current_version,
            'L0': hsc.L0,
            'L1': hsc.L1,
            'L_entity': L_entity,
            'L_msg': L_msg,
            'entity_index': entity_index,
            'msg_index': msg_index,
            'nodes': hsc.nodes,
            'simplices': hsc.simplices,
            'timestamp': time.time()
        }

        asyncio.create_task(self._async_save_laplacian_cache())

        end_time = time.time()
        logger.info(f"Built bipartite Laplacian matrices in {end_time - start_time:.2f} seconds "
                     f"(L_entity: {L_entity.shape}, L_msg: {L_msg.shape})")
        return self._laplacian_cache

    def cache_laplacian(self, name: str, matrix: np.ndarray):
        """缓存Laplacian矩阵到内存

        Args:
            name: 矩阵名称，如"L_entity"、"L_msg"
            matrix: numpy数组
        """
        if not hasattr(self, '_bipartite_laplacian_cache'):
            self._bipartite_laplacian_cache = {}
        self._bipartite_laplacian_cache[name] = matrix
        logger.info(f"Cached Laplacian matrix '{name}' with shape {matrix.shape}")

        laplacian_file = os.path.join(self._hgdb_dir, f"bipartite_{name}.npy")
        try:
            np.save(laplacian_file, matrix)
            logger.info(f"Saved Laplacian matrix '{name}' to {laplacian_file}")
        except Exception as e:
            logger.error(f"Error saving Laplacian matrix '{name}': {e}")

    def load_laplacian(self, name: str) -> Optional[np.ndarray]:
        """从缓存加载Laplacian矩阵

        Args:
            name: 矩阵名称

        Returns:
            numpy数组或None
        """
        if hasattr(self, '_bipartite_laplacian_cache') and name in self._bipartite_laplacian_cache:
            return self._bipartite_laplacian_cache[name]

        laplacian_file = os.path.join(self._hgdb_dir, f"bipartite_{name}.npy")
        if os.path.exists(laplacian_file):
            try:
                matrix = np.load(laplacian_file)
                if not hasattr(self, '_bipartite_laplacian_cache'):
                    self._bipartite_laplacian_cache = {}
                self._bipartite_laplacian_cache[name] = matrix
                logger.info(f"Loaded Laplacian matrix '{name}' from {laplacian_file}")
                return matrix
            except Exception as e:
                logger.error(f"Error loading Laplacian matrix '{name}': {e}")
        return None

    def cache_index(self, name: str, index: dict):
        """缓存索引映射到文件

        Args:
            name: 索引名称，如"entity_index"、"msg_index"
            index: 字典映射
        """
        if not hasattr(self, '_bipartite_index_cache'):
            self._bipartite_index_cache = {}
        self._bipartite_index_cache[name] = index
        logger.info(f"Cached index '{name}' with {len(index)} entries")

        index_file = os.path.join(self._hgdb_dir, f"bipartite_{name}.json")
        try:
            import json
            str_index = {str(k): int(v) if isinstance(v, (int, np.integer)) else v for k, v in index.items()}
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(str_index, f, ensure_ascii=False)
            logger.info(f"Saved index '{name}' to {index_file}")
        except Exception as e:
            logger.error(f"Error saving index '{name}': {e}")

    def load_index(self, name: str) -> Optional[dict]:
        """从文件加载索引映射

        Args:
            name: 索引名称

        Returns:
            字典映射或None
        """
        if hasattr(self, '_bipartite_index_cache') and name in self._bipartite_index_cache:
            return self._bipartite_index_cache[name]

        index_file = os.path.join(self._hgdb_dir, f"bipartite_{name}.json")
        if os.path.exists(index_file):
            try:
                import json
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                if not hasattr(self, '_bipartite_index_cache'):
                    self._bipartite_index_cache = {}
                self._bipartite_index_cache[name] = index
                logger.info(f"Loaded index '{name}' from {index_file} with {len(index)} entries")
                return index
            except Exception as e:
                logger.error(f"Error loading index '{name}': {e}")
        return None

    async def _async_save_laplacian_cache(self):
        """异步保存拉普拉斯矩阵缓存到文件"""
        if self._laplacian_cache:
            # 在线程池中执行，避免阻塞事件循环
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                await asyncio.get_running_loop().run_in_executor(
                    executor,
                    self._save_laplacian_cache_to_file,
                    self._laplacian_cache
                )

    def _invalidate_cache(self):
        """使缓存失效"""
        self._data_version += 1
        self._laplacian_cache = None
        self._index_cache.clear()
    
    def _cleanup_cache(self):
        """
        清理缓存，确保缓存大小不超过限制
        """
        if len(self._index_cache) > self._cache_size_limit:
            # 保留最近使用的缓存项
            # 这里简单实现为保留一半的缓存项
            items = list(self._index_cache.items())
            self._index_cache = dict(items[:self._cache_size_limit // 2])
            logger.debug(f"Cleaned up cache, size: {len(self._index_cache)}")
    
    async def check_consistency(self):
        """
        检查复形结构的一致性
        
        Returns:
            bool: 一致性检查是否通过
        """
        if not self._consistency_check_enabled:
            logger.debug("Consistency check disabled")
            return True
        
        logger.info("Starting consistency check...")
        issues = []
        
        # 收集所有复形ID和顶点ID
        all_simplex_ids = set()
        all_vertex_ids = set()
        
        with self._lock.read_lock():
            # 收集所有复形ID
            all_simplices = self._hg.get_all_simplices()
            for vertices, simplex_data in all_simplices:
                simplex_id = simplex_data.get("id")
                if simplex_id:
                    all_simplex_ids.add(simplex_id)
                    # 收集0维复形作为顶点
                    if simplex_data.get("dimension", -1) == 0 and vertices:
                        all_vertex_ids.add(vertices[0])
            
            # 检查所有复形的边界是否存在
            for vertices, simplex_data in all_simplices:
                simplex_id = simplex_data.get("id")
                if not simplex_id:
                    continue
                boundary = simplex_data.get("boundary", [])
                dimension = simplex_data.get("dimension", 0)
                
                # 检查边界中的复形是否存在
                for sub_id in boundary:
                    # 检查是否是0-单纯形
                    if dimension == 1:
                        # 对于边，边界应该是顶点
                        if sub_id not in all_vertex_ids:
                            issues.append(f"Boundary simplex {sub_id} (vertex) not found for simplex {simplex_id}")
                    else:
                        # 对于高维复形，边界应该是低维复形
                        if sub_id not in all_simplex_ids:
                            issues.append(f"Boundary simplex {sub_id} not found for simplex {simplex_id}")
                
                # 检查上边界中的复形是否存在
                coboundary = simplex_data.get("coboundary", [])
                for higher_id in coboundary:
                    if higher_id not in all_simplex_ids:
                        issues.append(f"Coboundary simplex {higher_id} not found for simplex {simplex_id}")
            
            # 检查所有0-单纯形的上边界
            dim_0_simplices = self._hg.get_simplices_by_dimension(0)
            for vertices, vertex_data in dim_0_simplices:
                vertex_id = vertex_data.get("id")
                if not vertex_id:
                    continue
                coboundary = vertex_data.get("coboundary", [])
                for higher_id in coboundary:
                    if higher_id not in all_simplex_ids:
                        issues.append(f"Coboundary simplex {higher_id} not found for vertex {vertex_id}")
        
        if issues:
            logger.error(f"Consistency check failed with {len(issues)} issues:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("Consistency check passed")
            return True
    
    def enable_consistency_check(self, enable: bool):
        """
        启用或禁用一致性检查
        
        Args:
            enable: 是否启用一致性检查
        """
        self._consistency_check_enabled = enable
        logger.info(f"Consistency check {'enabled' if enable else 'disabled'}")

    async def add_simplex(self, simplex_id, simplex_data):
        """添加单纯形并使缓存失效"""
        result = await super().add_simplex(simplex_id, simplex_data)
        self._invalidate_cache()
        return result

    async def delete_simplex(self, simplex_id):
        """删除单纯形并使缓存失效"""
        result = await super().delete_simplex(simplex_id)
        self._invalidate_cache()
        return result

    async def index_done_callback(self):
        logger.info(f"Starting to save SimplexStorage to {self._hgdb_file}")
        start_time = time.time()
        
        try:
            # 使用线程池执行同步的write_hypergraph方法，避免阻塞事件循环
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                await asyncio.get_running_loop().run_in_executor(
                    executor,
                    SimplexStorage.write_hypergraph,
                    self._hg,
                    self._hgdb_file
                )
            
            end_time = time.time()
            logger.info(f"Completed saving SimplexStorage in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error saving SimplexStorage: {e}")
            import traceback
            traceback.print_exc()

    async def query(self, query: str, top_k, extracted_entities: list = None, extracted_relations: list = None):
        """
        基于查询的复形检索，利用倒排索引加速并持久化嵌入向量
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            extracted_entities: 从查询中提取的实体列表
            extracted_relations: 从查询中提取的关系列表
            
        Returns:
            按相关性排序的复形列表
        """
        if not self.embedding_func:
            logger.warning("No embedding_func provided, falling back to basic retrieval")
            all_simplices = await self.get_all_simplices()
            return all_simplices[:top_k]
        
        enhanced_query = query
        if extracted_entities:
            enhanced_query = f"{enhanced_query} {' '.join(extracted_entities)}"
        if extracted_relations:
            enhanced_query = f"{enhanced_query} {' '.join(extracted_relations)}"
        
        query_embedding = await self.embedding_func([enhanced_query])
        query_embedding = query_embedding[0]
        
        need_embedding = []
        all_simplices = self._hg.get_all_simplices()
        for vertices, simplex_data in all_simplices:
            if "embedding" not in simplex_data:
                simplex_text = " ".join(vertices)
                if "description" in simplex_data:
                    simplex_text = f"{simplex_text} {simplex_data['description']}"
                need_embedding.append((vertices, simplex_data, simplex_text))
        
        if need_embedding:
            texts = [text for _, _, text in need_embedding]
            try:
                cpu_count = os.cpu_count() or 4
                max_concurrent_batches = max(2, min(cpu_count, 16))
                batch_size = 100
                
                batches = []
                for i in range(0, len(texts), batch_size):
                    batches.append(texts[i:i+batch_size])
                
                async def process_batch(batch_texts):
                    return await self.embedding_func(batch_texts)
                
                tasks = []
                for i in range(0, len(batches), max_concurrent_batches):
                    batch_group = batches[i:i+max_concurrent_batches]
                    group_tasks = [process_batch(batch) for batch in batch_group]
                    group_results = await asyncio.gather(*group_tasks)
                    tasks.extend(group_results)
                
                embeddings = []
                for batch_result in tasks:
                    embeddings.extend(batch_result)
                
                for i, (vertices, simplex_data, _) in enumerate(need_embedding):
                    if i < len(embeddings):
                        simplex_embedding = embeddings[i]
                        simplex_data["embedding"] = simplex_embedding.tolist()
                        simplex_id = simplex_data.get('id')
                        self._hg.insert(vertices, simplex_data, simplex_id)
                        self._add_to_inverted_indexes(simplex_id, simplex_data)
                
                logger.info(f"Batch computed and persisted embeddings for {len(need_embedding)} simplices")
                await self.index_done_callback()
            except Exception as e:
                logger.error(f"Error batch computing embeddings: {e}")
        
        async def compute_similarity(vertices, simplex_data):
            simplex_embedding = simplex_data.get("embedding")
            if simplex_embedding is not None:
                from numpy.linalg import norm
                q_emb = np.asarray(query_embedding, dtype=np.float32).ravel()
                s_emb = np.asarray(simplex_embedding, dtype=np.float32).ravel()
                q_norm = float(norm(q_emb))
                s_norm = float(norm(s_emb))
                if q_norm > 0 and s_norm > 0:
                    similarity = float(np.dot(q_emb, s_emb) / (q_norm * s_norm))
                else:
                    similarity = 0.0
                simplex_id = simplex_data.get("id")
                if simplex_id:
                    return (simplex_id, simplex_data, similarity)
            return None
        
        all_simplices = self._hg.get_all_simplices()
        tasks = [compute_similarity(vertices, simplex_data) for vertices, simplex_data in all_simplices]
        
        cpu_count = os.cpu_count() or 4
        max_concurrent_tasks = max(4, min(cpu_count * 2, 32))
        
        similarities = []
        for i in range(0, len(tasks), max_concurrent_tasks):
            batch_tasks = tasks[i:i+max_concurrent_tasks]
            batch_results = await asyncio.gather(*batch_tasks)
            batch_similarities = [result for result in batch_results if result is not None]
            similarities.extend(batch_similarities)
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        results = [(simplex_id, simplex_data) for simplex_id, simplex_data, _ in similarities[:top_k]]
        logger.info(f"Retrieved {len(results)} simplices from simplex storage based on query")
        return results

    async def has_simplex(self, simplex_id: Any) -> bool:
        # 使用单纯形树的find_by_id方法检查
        node = self._hg.find_by_id(simplex_id)
        return node is not None and node.simplex_data is not None

    async def get_simplex(self, simplex_id: Any, default: Any = None):
        # 使用单纯形树的find_by_id方法获取
        node = self._hg.find_by_id(simplex_id)
        if node and node.simplex_data:
            return node.simplex_data
        return default

    async def upsert_simplex(self, simplex_id: Any, simplex_data: Optional[Dict] = None):
        if simplex_data is None:
            simplex_data = {}
        
        # 计算复形维度
        dimension = simplex_data.get("dimension")
        if dimension is None:
            entities = simplex_data.get("entities", [])
            if entities:
                dimension = len(entities) - 1
            else:
                dimension = 0
        
        # 检查是否是增量更新
        existing_simplex = await self.get_simplex(simplex_id)
        is_update = existing_simplex is not None
        
        # 增量更新时合并已有数据（参照基线模型_merge_nodes_then_upsert逻辑）
        if is_update and existing_simplex is not None:
            from collections import Counter
            from .utils import split_string_by_multi_markers
            GRAPH_FIELD_SEP = "<SEP>"
            
            if dimension == 0:
                # 0-单纯形（实体）合并：实体类型投票、描述去重拼接、来源ID去重拼接
                already_entity_types = []
                already_source_ids = []
                already_description = []
                already_additional_properties = []
                
                if existing_simplex.get("entity_type"):
                    already_entity_types.append(existing_simplex["entity_type"])
                if existing_simplex.get("source_id"):
                    already_source_ids.extend(
                        split_string_by_multi_markers(existing_simplex["source_id"], [GRAPH_FIELD_SEP])
                    )
                if existing_simplex.get("description"):
                    already_description.append(existing_simplex["description"])
                if existing_simplex.get("additional_properties"):
                    already_additional_properties.append(existing_simplex["additional_properties"])
                
                # 实体类型：多数投票
                new_entity_type = simplex_data.get("entity_type", "Entity")
                all_entity_types = [new_entity_type] + already_entity_types
                if all_entity_types:
                    entity_type = sorted(
                        Counter(all_entity_types).items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[0][0]
                    simplex_data["entity_type"] = entity_type
                
                # 描述：去重拼接 + LLM摘要（对齐HyperGraphRAG-main的_merge_nodes_then_upsert）
                new_description = simplex_data.get("description", "")
                all_descriptions = [new_description] + already_description
                merged_description = GRAPH_FIELD_SEP.join(sorted(set(all_descriptions)))
                entity_name = simplex_data.get("entity_name", simplex_id)
                simplex_data["description"] = await self._handle_simplex_summary(
                    entity_name, merged_description, "entity_description"
                )
                
                # 附加属性：去重拼接 + LLM摘要
                new_ap = simplex_data.get("additional_properties", "")
                all_ap = [new_ap] + already_additional_properties
                merged_ap = GRAPH_FIELD_SEP.join(sorted(set(all_ap)))
                simplex_data["additional_properties"] = await self._handle_simplex_summary(
                    entity_name, merged_ap, "entity_ap"
                )
                
                # 来源ID：去重拼接
                new_source_id = simplex_data.get("source_id", "")
                all_source_ids = [new_source_id] + already_source_ids
                merged_source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))
                simplex_data["source_id"] = merged_source_id
                
                # 频率：累加
                existing_freq = existing_simplex.get("frequency", 0)
                new_freq = simplex_data.get("frequency", 1)
                simplex_data["frequency"] = existing_freq + new_freq
                
                # 重要性：加权平均
                existing_importance = existing_simplex.get("importance", 0.5)
                new_importance = simplex_data.get("importance", 0.5)
                simplex_data["importance"] = (existing_importance * existing_freq + new_importance * new_freq) / max(existing_freq + new_freq, 1)
                
                # 保留已有的coboundary信息
                if existing_simplex.get("coboundary"):
                    simplex_data["coboundary"] = existing_simplex["coboundary"]
                
                logger.info(f"Merged 0-simplex data for update: {simplex_id}")
            else:
                # n-单纯形（关系）合并：描述去重拼接、来源ID去重拼接
                already_description = []
                already_source_ids = []
                
                if existing_simplex.get("description"):
                    already_description.append(existing_simplex["description"])
                if existing_simplex.get("source_id"):
                    already_source_ids.extend(
                        split_string_by_multi_markers(existing_simplex["source_id"], [GRAPH_FIELD_SEP])
                    )
                
                # 描述：去重拼接 + LLM摘要（对齐HyperGraphRAG-main的_merge_edges_then_upsert）
                new_description = simplex_data.get("description", "")
                all_descriptions = [new_description] + already_description
                merged_description = GRAPH_FIELD_SEP.join(sorted(set(all_descriptions)))
                relation_name = str(simplex_data.get("entities", simplex_id))
                simplex_data["description"] = await self._handle_simplex_summary(
                    relation_name, merged_description, "relation_description"
                )
                
                # 来源ID：去重拼接
                new_source_id = simplex_data.get("source_id", "")
                all_source_ids = [new_source_id] + already_source_ids
                merged_source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))
                simplex_data["source_id"] = merged_source_id
                
                # 频率：累加
                existing_freq = existing_simplex.get("frequency", 0)
                new_freq = simplex_data.get("frequency", 1)
                simplex_data["frequency"] = existing_freq + new_freq
                
                # 重要性：加权平均
                existing_importance = existing_simplex.get("importance", 0.5)
                new_importance = simplex_data.get("importance", 0.5)
                simplex_data["importance"] = (existing_importance * existing_freq + new_importance * new_freq) / max(existing_freq + new_freq, 1)
                
                # 保留已有的boundary和coboundary信息
                if existing_simplex.get("boundary"):
                    simplex_data["boundary"] = existing_simplex["boundary"]
                if existing_simplex.get("coboundary"):
                    simplex_data["coboundary"] = existing_simplex["coboundary"]
                
                logger.info(f"Merged {dimension}-simplex data for update: {simplex_id}")
        
        # 存储所有维度的复形
        entities = simplex_data.get("entities", [])
        if entities:
            # 构建实体元组
            entity_tuple = []
            for entity in entities:
                # 直接使用实体名称
                entity_tuple.append(entity)
            
            # 创建复形连接，关联所有实体
            entity_tuple = tuple(sorted(entity_tuple))
            # 同步更新simplex_data中的entities字段为排序后的顺序，确保与树路径一致
            simplex_data["entities"] = list(entity_tuple)
            # 添加 id 到复形数据
            simplex_data["id"] = simplex_id
            
            # 嵌入向量已经在批量处理时计算，这里不再重复计算
            # 如果没有嵌入向量，且有嵌入函数，则计算嵌入向量
            if self.embedding_func and "embedding" not in simplex_data:
                simplex_text = " ".join(entity_tuple)
                
                with self._lock.read_lock():
                    for entity in entity_tuple:
                        vertex_node = self._hg.find([entity])
                        if vertex_node and vertex_node.simplex_data and "description" in vertex_node.simplex_data:
                            simplex_text = f"{simplex_text} {vertex_node.simplex_data['description']}"
                
                if "description" in simplex_data:
                    simplex_text = f"{simplex_text} {simplex_data['description']}"
                
                embedding = await self.embedding_func([simplex_text])
                simplex_data["embedding"] = embedding[0].tolist()
            
            try:
                with self._lock.write_lock():
                    if is_update:
                        old_node = self._hg.find_by_id(simplex_id)
                        if old_node:
                            old_entities = self._get_vertices_from_node(old_node)
                            for entity in old_entities:
                                if simplex_id in self._entity_to_simplices[entity]:
                                    self._entity_to_simplices[entity].remove(simplex_id)
                            if simplex_id in self._simplex_to_entities:
                                del self._simplex_to_entities[simplex_id]
                            self._remove_from_inverted_indexes(simplex_id, old_node.simplex_data)
                            self._hg.remove(old_entities)
                            logger.info(f"Removed old simplex for update: {simplex_id}")
                    
                    # 不再通过代码自动补全闭包面，由LLM提取时自行保证强闭包性质
                    
                    # 添加新复形
                    self._hg.insert(entity_tuple, simplex_data, simplex_id)
                    
                    # 更新双向索引
                    for entity in entity_tuple:
                        if simplex_id not in self._entity_to_simplices[entity]:
                            self._entity_to_simplices[entity].append(simplex_id)
                    self._simplex_to_entities[simplex_id] = list(entity_tuple)
                    self._add_to_inverted_indexes(simplex_id, simplex_data)
                    
                    # 计算边界（boundary）- 指向实际存储的面单纯形
                    # 系统仅存储极大单纯形和0-单纯形（实体节点），中间面不单独存储
                    # boundary包含：1) 已存储的(k-1)-维面单纯形 2) 所有0-单纯形（实体节点）
                    # 0-单纯形必须加入boundary，否则_update_coboundary无法更新实体的coboundary
                    boundary = []
                    if dimension > 0:
                        try:
                            from itertools import combinations
                            for sub_entities in combinations(list(entity_tuple), len(entity_tuple) - 1):
                                sub_node = self._hg.find(list(sub_entities))
                                if sub_node and sub_node.simplex_data:
                                    sub_id = sub_node.simplex_data.get('id')
                                    if sub_id:
                                        boundary.append(sub_id)
                            for entity in entity_tuple:
                                entity_node = self._hg.find([entity])
                                if entity_node and entity_node.simplex_data:
                                    entity_id = entity_node.simplex_data.get('id')
                                    if entity_id and entity_id not in boundary:
                                        boundary.append(entity_id)
                        except Exception as e:
                            logger.error(f"Error computing boundary for simplex {simplex_id}: {e}")
                    
                    # 更新单纯形的边界信息
                    simplex_data["boundary"] = boundary
                    # 重新插入以更新边界信息
                    self._hg.insert(entity_tuple, simplex_data, simplex_id)
                    
                    # 计算上边界（coboundary）- 改进：包含所有高维父复形，利用双向索引提高效率
                    coboundary = []
                    try:
                        # 使用单纯形树的get_coboundary方法获取上边界
                        # 传入排序后的实体列表，确保与单纯形树中的存储一致
                        coboundary_vertices_list = self._hg.get_coboundary(list(entity_tuple))
                        
                        # 转换为单纯形ID
                        for coboundary_vertices in coboundary_vertices_list:
                            coboundary_node = self._hg.find(coboundary_vertices)
                            if coboundary_node and coboundary_node.simplex_data:
                                coboundary_id = coboundary_node.simplex_data.get("id")
                                if coboundary_id and coboundary_id not in coboundary:
                                    coboundary.append(coboundary_id)
                    except Exception as e:
                        logger.error(f"Error computing coboundary for simplex {simplex_id}: {e}")
                    
                    # 更新单纯形的上边界信息
                    simplex_data["coboundary"] = coboundary
                    # 重新插入以更新上边界信息
                    self._hg.insert(entity_tuple, simplex_data, simplex_id)
                
                # 更新面的上边界（coboundary）
                # boundary包含所有已存在的(n-1)-面的ID
                for vertex_id in boundary:
                    await self._update_coboundary(vertex_id, simplex_id)
            except Exception as e:
                logger.error(f"Error {'updating' if is_update else 'storing'} simplex: {e}")
        elif dimension == 0:
            # 处理0-单纯形（单个实体，无entities字段的边界情况）
            entity_name = simplex_data.get("entity_name")
            if entity_name:
                # 增量更新时合并已有数据
                if is_update and existing_simplex is not None:
                    from collections import Counter as _Counter
                    from .utils import split_string_by_multi_markers as _split_str
                    _SEP = "<SEP>"
                    
                    already_entity_types = []
                    already_source_ids = []
                    already_description = []
                    already_additional_properties = []
                    
                    if existing_simplex.get("entity_type"):
                        already_entity_types.append(existing_simplex["entity_type"])
                    if existing_simplex.get("source_id"):
                        already_source_ids.extend(
                            _split_str(existing_simplex["source_id"], [_SEP])
                        )
                    if existing_simplex.get("description"):
                        already_description.append(existing_simplex["description"])
                    if existing_simplex.get("additional_properties"):
                        already_additional_properties.append(existing_simplex["additional_properties"])
                    
                    new_entity_type = simplex_data.get("entity_type", "Entity")
                    all_entity_types = [new_entity_type] + already_entity_types
                    if all_entity_types:
                        simplex_data["entity_type"] = sorted(
                            _Counter(all_entity_types).items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[0][0]
                    
                    # 描述：去重拼接 + LLM摘要
                    new_description = simplex_data.get("description", "")
                    all_descriptions = [new_description] + already_description
                    merged_description = _SEP.join(sorted(set(all_descriptions)))
                    simplex_data["description"] = await self._handle_simplex_summary(
                        entity_name, merged_description, "entity_description"
                    )
                    
                    # 附加属性：去重拼接 + LLM摘要
                    new_ap = simplex_data.get("additional_properties", "")
                    all_ap = [new_ap] + already_additional_properties
                    merged_ap = _SEP.join(sorted(set(all_ap)))
                    simplex_data["additional_properties"] = await self._handle_simplex_summary(
                        entity_name, merged_ap, "entity_ap"
                    )
                    
                    new_source_id = simplex_data.get("source_id", "")
                    all_source_ids = [new_source_id] + already_source_ids
                    simplex_data["source_id"] = _SEP.join(set(all_source_ids))
                    
                    existing_freq = existing_simplex.get("frequency", 0)
                    new_freq = simplex_data.get("frequency", 1)
                    simplex_data["frequency"] = existing_freq + new_freq
                    
                    existing_importance = existing_simplex.get("importance", 0.5)
                    new_importance = simplex_data.get("importance", 0.5)
                    simplex_data["importance"] = (existing_importance * existing_freq + new_importance * new_freq) / max(existing_freq + new_freq, 1)
                    
                    if existing_simplex.get("coboundary"):
                        simplex_data["coboundary"] = existing_simplex["coboundary"]
                    
                    logger.info(f"Merged 0-simplex data (no-entities path) for update: {simplex_id}")
                
                # 确保添加维度信息
                simplex_data["dimension"] = 0
                simplex_data["boundary"] = []
                simplex_data["coboundary"] = simplex_data.get("coboundary", [])
                # 使用insert方法插入到单纯形树
                try:
                    with self._lock.write_lock():
                        simplex_id_val = simplex_data.get("id", entity_name)
                        if is_update:
                            self._remove_from_inverted_indexes(simplex_id_val, simplex_data)
                        self._entity_to_simplices[entity_name].append(simplex_id_val)
                        self._simplex_to_entities[simplex_id_val] = [entity_name]
                        self._hg.insert([entity_name], simplex_data, simplex_id_val)
                        self._add_to_inverted_indexes(simplex_id_val, simplex_data)
                        logger.info(f"{'Updated' if is_update else 'Stored'} 0-simplex: {entity_name}")
                except Exception as e:
                    logger.error(f"Error {'updating' if is_update else 'storing'} 0-simplex: {e}")
        else:
            logger.debug(f"Skipping simplex with no entities: {simplex_id}")
        
        # 使缓存失效
        self._invalidate_cache()
        return simplex_data
    
    async def _update_coboundary(self, simplex_id: str, coboundary_id: str):
        """
        更新复形的上边界信息
        
        Args:
            simplex_id: 复形ID
            coboundary_id: 上边界复形ID
        """
        with self._lock.write_lock():
            simplex_node = self._hg.find_by_id(simplex_id)
            coboundary_node = self._hg.find_by_id(coboundary_id)
            
            if not (simplex_node and simplex_node.simplex_data and coboundary_node and coboundary_node.simplex_data):
                return
            
            simplex_dim = simplex_node.simplex_data.get('dimension', 0)
            coboundary_dim = coboundary_node.simplex_data.get('dimension', 0)
            
            if coboundary_dim <= simplex_dim:
                logger.debug(f"Skipping coboundary update: {coboundary_id}(dim={coboundary_dim}) is not higher than {simplex_id}(dim={simplex_dim})")
                return
            
            if "coboundary" not in simplex_node.simplex_data:
                simplex_node.simplex_data["coboundary"] = []
            if coboundary_id not in simplex_node.simplex_data["coboundary"]:
                simplex_node.simplex_data["coboundary"].append(coboundary_id)
                vertices = self._get_vertices_from_node(simplex_node)
                self._hg.insert(vertices, simplex_node.simplex_data, simplex_id)

    async def remove_simplex(self, simplex_id: Any):
        """
        删除复形并更新相关复形的边界信息
        
        Args:
            simplex_id: 要删除的复形ID
        """
        # 查找要删除的复形
        target_node = self._hg.find_by_id(simplex_id)
        if target_node and target_node.simplex_data:
            dimension = target_node.simplex_data.get("dimension", 0)
            boundary = target_node.simplex_data.get("boundary", [])
            
            target_vertices = self._get_vertices_from_node(target_node)
            
            with self._lock.write_lock():
                self._remove_from_inverted_indexes(simplex_id, target_node.simplex_data)
                for entity in target_vertices:
                    if simplex_id in self._entity_to_simplices[entity]:
                        self._entity_to_simplices[entity].remove(simplex_id)
                if simplex_id in self._simplex_to_entities:
                    del self._simplex_to_entities[simplex_id]
                self._hg.remove(target_vertices)
            logger.info(f"Removed simplex (dimension {dimension}): {simplex_id}")
            
            # 更新低维复形的上边界
            for sub_id in boundary:
                await self._remove_from_coboundary(sub_id, simplex_id)
            
            # 更新高维复形的边界
            # 遍历所有高维复形，检查是否包含当前复形
            with self._lock.write_lock():
                all_simplices = self._hg.get_all_simplices()
                for higher_vertices, higher_data in all_simplices:
                    higher_dim = higher_data.get("dimension", 0)
                    if higher_dim > dimension:
                        higher_boundary = higher_data.get("boundary", [])
                        if simplex_id in higher_boundary:
                            higher_boundary.remove(simplex_id)
                            higher_data["boundary"] = higher_boundary
                            # 重新插入以更新数据
                            self._hg.insert(higher_vertices, higher_data, higher_data.get('id'))
                            logger.debug(f"Updated boundary for higher simplex: {higher_data.get('id')}")
        else:
            # 尝试删除0-单纯形（顶点）
            # 使用find_by_id查找该复形
            target_node = self._hg.find_by_id(simplex_id)
            if target_node and target_node.simplex_data:
                # 获取上边界
                coboundary = target_node.simplex_data.get("coboundary", [])
                
                target_vertices = self._get_vertices_from_node(target_node)
                
                self._hg.remove(target_vertices)
                logger.info(f"Removed 0-simplex: {simplex_id}")
                
                # 更新相关复形
                for higher_id in coboundary:
                    # 查找包含该顶点的复形
                    higher_node = self._hg.find_by_id(higher_id)
                    if higher_node and higher_node.simplex_data:
                        # 获取复形的顶点
                        higher_vertices = self._get_vertices_from_node(higher_node)
                        if simplex_id in higher_vertices:
                            higher_vertices.remove(simplex_id)
                            # 如果实体列表为空，删除复形
                            if not higher_vertices:
                                old_data = dict(higher_node.simplex_data)
                                self._hg.remove(list(old_data.get("entities", old_data.get("nodes", []))))
                                logger.info(f"Removed simplex {higher_id} because it became empty")
                            else:
                                # 重新计算复形的边界
                                new_dimension = len(higher_vertices) - 1
                                new_boundary = []
                                if new_dimension > 0:
                                    from itertools import combinations
                                    for sub_entities in combinations(higher_vertices, new_dimension):
                                        sub_id = compute_mdhash_id(str(sorted(sub_entities)), prefix=f"simplex-{new_dimension-1}-")
                                        new_boundary.append(sub_id)
                                
                                # 更新复形数据
                                higher_node.simplex_data["entities"] = higher_vertices
                                higher_node.simplex_data["dimension"] = new_dimension
                                higher_node.simplex_data["boundary"] = new_boundary
                                
                                # 重新插入以更新数据
                                self._hg.insert(higher_vertices, higher_node.simplex_data, higher_id)
                                logger.info(f"Updated simplex {higher_id} after removing vertex {simplex_id}")
        
        # 使缓存失效
        self._invalidate_cache()
                    
    async def _remove_from_coboundary(self, simplex_id: str, coboundary_id: str):
        """
        从复形的上边界中移除指定的复形ID
        
        Args:
            simplex_id: 复形ID
            coboundary_id: 要移除的上边界复形ID
        """
        with self._lock.write_lock():
            # 使用单纯形树的find_by_id方法查找复形
            simplex_node = self._hg.find_by_id(simplex_id)
            if simplex_node and simplex_node.simplex_data:
                if "coboundary" in simplex_node.simplex_data and coboundary_id in simplex_node.simplex_data["coboundary"]:
                    simplex_node.simplex_data["coboundary"].remove(coboundary_id)
                    vertices = self._get_vertices_from_node(simplex_node)
                    self._hg.insert(vertices, simplex_node.simplex_data, simplex_id)
                    logger.debug(f"Removed coboundary {coboundary_id} from simplex {simplex_id}")
    
    async def delete_simplex(self, simplex_id: Any):
        # 为了兼容性，添加delete_simplex方法作为remove_simplex的别名
        return await self.remove_simplex(simplex_id)

    async def get_all_simplices(self):
        result = []
        # 使用单纯形树的get_all_simplices方法获取所有单纯形
        all_simplices = self._hg.get_all_simplices()
        for vertices, simplex_data in all_simplices:
            simplex_id = simplex_data.get("id")
            if simplex_id:
                result.append((simplex_id, simplex_data))
        
        logger.debug(f"Retrieved {len(result)} simplices from storage")
        return result

    async def get_simplices_by_entity(self, entity_name: str) -> List[Tuple[str, Dict]]:
        """
        利用双向索引快速查找包含特定实体的所有复形
        优先使用双向索引，若索引未命中则回退到单纯形树顶点查询
        
        Args:
            entity_name: 实体名称
            
        Returns:
            包含该实体的复形列表
        """
        cache_key = f"entity:{entity_name}"
        if cache_key in self._index_cache:
            logger.debug(f"Using cached results for entity: {entity_name}")
            return self._index_cache[cache_key]
        
        result = []
        entity_simplices = self._entity_to_simplices.get(entity_name, [])
        
        if entity_simplices:
            with self._lock.read_lock():
                for simplex_id in entity_simplices:
                    node = self._hg.find_by_id(simplex_id)
                    if node and node.simplex_data:
                        result.append((simplex_id, node.simplex_data))
        else:
            simplices = self._hg.get_simplices_by_vertex(entity_name)
            for vertices, simplex_data in simplices:
                simplex_id = simplex_data.get("id")
                if simplex_id:
                    result.append((simplex_id, simplex_data))
        
        self._index_cache[cache_key] = result
        self._cleanup_cache()
        logger.debug(f"Found {len(result)} simplices containing entity: {entity_name}")
        return result
    
    async def get_simplices_by_dimension(self, dimension: int):
        result = []
        # 使用单纯形树的get_simplices_by_dimension方法获取指定维度的单纯形
        simplices = self._hg.get_simplices_by_dimension(dimension)
        for vertices, simplex_data in simplices:
            simplex_id = simplex_data.get("id")
            if simplex_id:
                # 确保复形数据包含dimension字段
                if "dimension" not in simplex_data:
                    simplex_data["dimension"] = dimension
                result.append((simplex_id, simplex_data))
        
        logger.debug(f"Found {len(result)} simplices of dimension {dimension}")
        return result

    async def get_simplices_by_verification_status(self, status: str):
        """根据语义校验状态获取单纯形 - 对应2.md中的第三阶段"""
        result = []
        for sid in self._verification_status_index.get(status, set()):
            node = self._hg.find_by_id(sid)
            if node and node.simplex_data:
                result.append((sid, node.simplex_data))
        return result

    async def get_simplices_by_chunk_id(self, chunk_id: str):
        """根据Chunk ID获取相关的所有单纯形 - 对应2.md中的元数据绑定"""
        result = []
        for sid in self._chunk_id_index.get(chunk_id, set()):
            node = self._hg.find_by_id(sid)
            if node and node.simplex_data:
                result.append((sid, node.simplex_data))
        return result
    
    async def batch_upsert_simplices(self, simplices: List[Tuple[Any, Dict]]):
        """
        批量存储复形，支持异步并行处理和增量更新

        强闭包性质由LLM提取时保证，存储层不再自动补全缺失的面。
        插入n-单纯形时仅查找已存在的面来计算boundary。

        Args:
            simplices: 复形列表，每个元素是 (simplex_id, simplex_data) 元组
        """
        start_time = time.time()
        total_simplices = len(simplices)
        logger.info(f"Starting batch upsert of {total_simplices} simplices")
        
        if not simplices:
            logger.info("No simplices to upsert")
            return
        
        # 主批次大小：将整个数据集分成多个主批次处理
        main_batch_size = 1000  # 每个主批次处理1000个复形
        main_batches = []
        for i in range(0, total_simplices, main_batch_size):
            main_batches.append(simplices[i:i+main_batch_size])
        
        total_main_batches = len(main_batches)
        logger.info(f"Split into {total_main_batches} main batches")
        
        # 统计信息
        stats = {
            'total': total_simplices,
            'by_dimension': {},
            'updates': 0,
            'inserts': 0,
            'errors': 0
        }
        
        # 处理每个主批次
        for main_batch_idx, main_batch in enumerate(main_batches):
            logger.info(f"Processing main batch {main_batch_idx + 1}/{total_main_batches} ({len(main_batch)} simplices)")
            
            # 按维度分组，确保低维复形先处理
            simplices_by_dim = defaultdict(list)
            for simplex_id, simplex_data in main_batch:
                dimension = simplex_data.get("dimension")
                if dimension is None:
                    entities = simplex_data.get("entities", [])
                    if entities:
                        dimension = len(entities) - 1
                    else:
                        dimension = 0
                simplices_by_dim[dimension].append((simplex_id, simplex_data))
            
            # 按维度顺序处理，从低到高
            sorted_dims = sorted(simplices_by_dim.keys())
            
            for dim in sorted_dims:
                dim_simplices = simplices_by_dim[dim]
                if dim not in stats['by_dimension']:
                    stats['by_dimension'][dim] = 0
                stats['by_dimension'][dim] += len(dim_simplices)
                logger.info(f"Processing dimension {dim} with {len(dim_simplices)} simplices")
                
                # 收集需要计算嵌入向量的复形
                embedding_needed = []
                embedding_indices = []
                
                # 检查增量更新
                for i, (simplex_id, simplex_data) in enumerate(dim_simplices):
                    # 检查是否是增量更新
                    existing_simplex = await self.get_simplex(simplex_id)
                    is_update = existing_simplex is not None
                    simplex_data["is_update"] = is_update
                    
                    if is_update:
                        stats['updates'] += 1
                    else:
                        stats['inserts'] += 1
                    
                    simplex_data["id"] = simplex_id
                    
                    # 不再通过代码自动补全闭包面，由LLM提取时自行保证强闭包性质
                    # 仅查找已存在的面来计算boundary
                    boundary = []
                    if dim > 0:
                        try:
                            from itertools import combinations
                            entities = simplex_data.get("entities", [])
                            sorted_entities = sorted(entities)
                            
                            with self._lock.read_lock():
                                for sub_entities in combinations(sorted_entities, len(sorted_entities) - 1):
                                    sub_node = self._hg.find(list(sub_entities))
                                    if sub_node and sub_node.simplex_data:
                                        sub_id = sub_node.simplex_data.get('id')
                                        if sub_id:
                                            boundary.append(sub_id)
                        except Exception as e:
                            logger.error(f"Error computing boundary for simplex {simplex_id}: {e}")
                            stats['errors'] += 1
                    simplex_data["boundary"] = boundary
                    
                    coboundary = []
                    try:
                        entities = simplex_data.get("entities", [])
                        sorted_entities = sorted(entities)
                        with self._lock.read_lock():
                            coboundary_vertices_list = self._hg.get_coboundary(sorted_entities)
                            for coboundary_vertices in coboundary_vertices_list:
                                coboundary_node = self._hg.find(coboundary_vertices)
                                if coboundary_node and coboundary_node.simplex_data:
                                    coboundary_id = coboundary_node.simplex_data.get("id")
                                    if coboundary_id:
                                        coboundary.append(coboundary_id)
                    except Exception as e:
                        logger.error(f"Error computing coboundary for simplex {simplex_id}: {e}")
                        stats['errors'] += 1
                    simplex_data["coboundary"] = coboundary
                    
                    # 检查是否需要计算嵌入向量
                    if self.embedding_func and "embedding" not in simplex_data:
                        # 构建嵌入文本
                        if dim == 0:
                            embedding_text = simplex_data.get("entity_name", "")
                            # 添加实体描述
                            if "description" in simplex_data:
                                embedding_text = f"{embedding_text} {simplex_data['description']}"
                        else:
                            entity_tuple = simplex_data.get("entities", [])
                            embedding_text = " ".join(entity_tuple)
                            # 添加实体描述
                            for entity in entity_tuple:
                                try:
                                    # 使用find方法获取实体数据
                                    vertex_node = self._hg.find([entity])
                                    if vertex_node and vertex_node.simplex_data:
                                        entity_description = vertex_node.simplex_data.get("description", "")
                                        if entity_description:
                                            embedding_text = f"{embedding_text} {entity_description}"
                                except Exception as e:
                                    logger.error(f"Error getting entity data for {entity}: {e}")
                            # 添加复形描述
                            if "description" in simplex_data:
                                embedding_text = f"{embedding_text} {simplex_data['description']}"
                        
                        if embedding_text:
                            embedding_needed.append(embedding_text)
                            embedding_indices.append(i)
                
                # 批量计算嵌入向量
                if embedding_needed:
                    import os
                    # 根据系统CPU核心数动态调整并行度
                    cpu_count = os.cpu_count() or 4
                    max_concurrent_batches = max(2, min(cpu_count, 8))  # 降低并行度以减少内存使用
                    
                    batch_size = 50  # 减小批次大小以减少内存使用
                    batches = []
                    for i in range(0, len(embedding_needed), batch_size):
                        batch_texts = embedding_needed[i:i+batch_size]
                        batch_indices = embedding_indices[i:i+batch_size]
                        batches.append((batch_texts, batch_indices, i//batch_size + 1))
                    
                    # 并行处理函数
                    async def process_batch(batch_texts, batch_indices, batch_num):
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                logger.info(f"Computing embeddings for batch {batch_num} ({len(batch_texts)} simplices) (attempt {attempt+1}/{max_retries})")
                                embeddings = await self.embedding_func(batch_texts)
                                for j, idx in enumerate(batch_indices):
                                    if j < len(embeddings):
                                        dim_simplices[idx][1]["embedding"] = embeddings[j].tolist()
                                return True
                            except Exception as e:
                                logger.error(f"Error computing embeddings for batch {batch_num} (attempt {attempt+1}/{max_retries}): {e}")
                                if attempt < max_retries - 1:
                                    # 指数退避重试
                                    import asyncio
                                    backoff_time = 2 ** attempt  # 指数退避
                                    logger.info(f"Retrying in {backoff_time} seconds...")
                                    await asyncio.sleep(backoff_time)
                                else:
                                    logger.error(f"Max retries reached for embedding computation for batch {batch_num}")
                                    stats['errors'] += len(batch_texts)
                                    return False
                    
                    # 并行处理批次
                    for i in range(0, len(batches), max_concurrent_batches):
                        batch_group = batches[i:i+max_concurrent_batches]
                        tasks = [process_batch(texts, indices, num) for texts, indices, num in batch_group]
                        import asyncio
                        await asyncio.gather(*tasks)
                    
                    # 清理内存
                    del embedding_needed
                    del embedding_indices
                
                # 并行存储复形
                async def store_simplex(simplex_id, simplex_data):
                    try:
                        dimension = simplex_data.get("dimension")
                        if dimension is None:
                            entities = simplex_data.get("entities", [])
                            if entities:
                                dimension = len(entities) - 1
                            else:
                                dimension = 0
                        
                        is_update = simplex_data.get("is_update", False)
                        entities = simplex_data.get("entities", [])
                        
                        if entities:
                            entity_tuple = tuple(sorted(entities))
                            # 同步更新simplex_data中的entities字段为排序后的顺序
                            simplex_data["entities"] = list(entity_tuple)
                            
                            with self._lock.write_lock():
                                if is_update:
                                    old_node = self._hg.find_by_id(simplex_id)
                                    if old_node:
                                        old_vertices = self._get_vertices_from_node(old_node)
                                        for entity in old_vertices:
                                            if simplex_id in self._entity_to_simplices[entity]:
                                                self._entity_to_simplices[entity].remove(simplex_id)
                                        if simplex_id in self._simplex_to_entities:
                                            del self._simplex_to_entities[simplex_id]
                                        self._remove_from_inverted_indexes(simplex_id, old_node.simplex_data)
                                        self._hg.remove(old_vertices)
                                        logger.info(f"Removed old simplex for update: {simplex_id}")
                                
                                self._hg.insert(entity_tuple, simplex_data, simplex_id)
                                
                                for entity in entity_tuple:
                                    if simplex_id not in self._entity_to_simplices[entity]:
                                        self._entity_to_simplices[entity].append(simplex_id)
                                self._simplex_to_entities[simplex_id] = list(entity_tuple)
                                self._add_to_inverted_indexes(simplex_id, simplex_data)
                            
                            boundary = simplex_data.get("boundary", [])
                            for vertex_id in boundary:
                                try:
                                    await self._update_coboundary(vertex_id, simplex_id)
                                except Exception as e:
                                    logger.error(f"Error updating coboundary for {vertex_id}: {e}")
                                    stats['errors'] += 1
                        elif dimension == 0:
                            entity_name = simplex_data.get("entity_name")
                            # 如果没有entity_name，尝试从entities列表中获取
                            if not entity_name and entities:
                                entity_name = entities[0]
                            if entity_name:
                                # 确保添加维度信息
                                simplex_data["dimension"] = 0
                                simplex_data["boundary"] = []
                                simplex_data["coboundary"] = []
                                # 确保entity_name存在于simplex_data中
                                simplex_data["entity_name"] = entity_name
                                
                                # 使用insert方法插入到单纯形树
                                with self._lock.write_lock():
                                    self._hg.insert([entity_name], simplex_data, simplex_id)
                                logger.info(f"{'Updated' if is_update else 'Stored'} 0-simplex: {entity_name}")
                    except Exception as e:
                        logger.error(f"Error {'updating' if simplex_data.get('is_update', False) else 'storing'} simplex {simplex_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        stats['errors'] += 1
                
                # 分批次并行存储
                import os
                # 根据系统CPU核心数动态调整并行度
                cpu_count = os.cpu_count() or 4
                max_concurrent_tasks = max(2, min(cpu_count, 16))  # 降低并行度以减少内存使用
                
                batch_size = 50  # 减小批次大小以减少内存使用
                for i in range(0, len(dim_simplices), batch_size):
                    batch = dim_simplices[i:i+batch_size]
                    logger.info(f"Storing batch {i//batch_size + 1}/{(len(dim_simplices) + batch_size - 1)//batch_size} ({len(batch)} simplices)")
                    
                    tasks = []
                    for j, (simplex_id, simplex_data) in enumerate(batch):
                        tasks.append(store_simplex(simplex_id, simplex_data))
                        # 当任务数达到最大并发数时，执行一批
                        if (j + 1) % max_concurrent_tasks == 0 or (j + 1) == len(batch):
                            if tasks:
                                import asyncio
                                await asyncio.gather(*tasks, return_exceptions=True)
                                tasks = []
                    
                    # 处理剩余任务
                    if tasks:
                        import asyncio
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                # 清理内存
                del dim_simplices
            
            # 清理内存
            del simplices_by_dim
            import gc
            gc.collect()
        
        # 使缓存失效
        self._invalidate_cache()
        
        # 执行一致性检查
        if self._consistency_check_enabled:
            await self.check_consistency()
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Batch upsert completed in {execution_time:.2f} seconds")
        logger.info(f"Stats: total={stats['total']}, inserts={stats['inserts']}, updates={stats['updates']}, errors={stats['errors']}")
        logger.info(f"By dimension: {stats['by_dimension']}")
        
        # 重建双向索引
        self._build_indexes()

    async def get_chains(self, min_length: int = 2):
        """获取1-链（路径）- 对应2.md中的第二阶段"""
        # 获取所有1-单纯形（边）
        edges = []
        dim_1_simplices = self._hg.get_simplices_by_dimension(1)
        for vertices, simplex_data in dim_1_simplices:
            simplex_id = simplex_data.get("id", str(vertices))
            edges.append((simplex_id, simplex_data))
        
        # 构建图结构
        import networkx as nx
        G = nx.Graph()
        for simplex_id, data in edges:
            entities = data.get("entities", [])
            if len(entities) == 2:
                G.add_edge(entities[0], entities[1], **data)
        
        # 获取连通分量（1-链）
        chains = []
        for component in nx.connected_components(G):
            if len(component) >= min_length:
                subgraph = G.subgraph(component)
                chains.append({
                    "nodes": list(component),
                    "edges": list(subgraph.edges(data=True)),
                    "length": len(component)
                })
        return chains

    async def get_cliques(self, min_size: int = 3):
        """获取团结构（高维候选者）- 对应2.md中的第二阶段"""
        # 获取所有1-单纯形（边）
        edges = []
        dim_1_simplices = self._hg.get_simplices_by_dimension(1)
        for vertices, simplex_data in dim_1_simplices:
            edges.append((simplex_data.get("id", str(vertices)), simplex_data))
        
        # 构建图结构
        import networkx as nx
        G = nx.Graph()
        for simplex_id, data in edges:
            entities = data.get("entities", [])
            if len(entities) == 2:
                G.add_edge(entities[0], entities[1])
        
        # 查找团
        from itertools import combinations
        cliques = []
        for clique in nx.find_cliques(G):
            if len(clique) >= min_size:
                cliques.append({
                    "entities": clique,
                    "size": len(clique),
                    "potential_dimension": len(clique) - 1
                })
        return cliques
