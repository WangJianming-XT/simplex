import os
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

import time

from .operate import (
    chunking_by_token_size,
    extract_entities,
)
from .operate._config import DualDimensionConfig
from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    SimplexStorage,
)


from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
    limit_async_gen_call,
    encode_string_by_tiktoken,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


async def naive_query(
    query,
    chunks_vdb,
    text_chunks_db,
    param,
    global_config,
):
    """朴素检索模式：基于向量数据库的文本块语义检索

    直接使用查询文本在文本块向量数据库中搜索最相关的文本块，
    不经过实体提取和拓扑检索流程，适用于简单查询场景。

    Args:
        query: 查询文本
        chunks_vdb: 文本块向量数据库
        text_chunks_db: 文本块键值存储
        param: 查询参数
        global_config: 全局配置

    Returns:
        基于检索上下文生成的回答
    """
    top_k = param.top_k if hasattr(param, 'top_k') else 60
    try:
        results = await chunks_vdb.query(query, top_k=top_k)
    except Exception as e:
        logger.error(f"naive_query 向量检索失败: {e}")
        results = []

    related_chunks = []
    if results:
        for result in results:
            if isinstance(result, dict):
                chunk_id = result.get("id", "")
                score = result.get("distance", 0)
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                chunk_id, score = result[0], result[1]
            else:
                continue
            if score < 0.2:
                continue
            chunk_data = await text_chunks_db.get_by_id(chunk_id)
            if chunk_data and "content" in chunk_data:
                related_chunks.append(chunk_data["content"])

    context = "\n".join(related_chunks)
    if param.only_need_context:
        return context

    llm_func = global_config.get("llm_model_func")
    if llm_func is None:
        return context

    sys_prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the following context.\n"
        "If the context does not contain the answer, say 'The context does not provide this information.'\n\n"
        f"Context:\n{context}"
    )
    response = await llm_func(query, system_prompt=sys_prompt)
    if len(response) > len(sys_prompt):
        response = response.replace(sys_prompt, "").strip()
    return response


async def llm_query(query, param, global_config):
    """LLM直接回答模式：不检索外部知识，直接由LLM回答

    适用于不需要检索外部知识的通用问答场景，
    LLM完全依赖自身知识生成回答。

    Args:
        query: 查询文本
        param: 查询参数
        global_config: 全局配置

    Returns:
        LLM生成的回答
    """
    llm_func = global_config.get("llm_model_func")
    if llm_func is None:
        return "Error: No LLM function configured."

    sys_prompt = "You are a helpful assistant. Answer the question to the best of your ability."
    response = await llm_func(query, system_prompt=sys_prompt)
    if len(response) > len(sys_prompt):
        response = response.replace(sys_prompt, "").strip()
    return response


async def naive_query_stream(
    query,
    chunks_vdb,
    text_chunks_db,
    param,
    global_config,
):
    """朴素检索流式模式：基于向量数据库的文本块语义检索，流式输出

    Args:
        query: 查询文本
        chunks_vdb: 文本块向量数据库
        text_chunks_db: 文本块键值存储
        param: 查询参数
        global_config: 全局配置

    Yields:
        生成的token流
    """
    top_k = param.top_k if hasattr(param, 'top_k') else 60
    try:
        results = await chunks_vdb.query(query, top_k=top_k)
    except Exception as e:
        logger.error(f"naive_query_stream 向量检索失败: {e}")
        results = []

    related_chunks = []
    if results:
        for result in results:
            if isinstance(result, dict):
                chunk_id = result.get("id", "")
                score = result.get("distance", 0)
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                chunk_id, score = result[0], result[1]
            else:
                continue
            if score < 0.2:
                continue
            chunk_data = await text_chunks_db.get_by_id(chunk_id)
            if chunk_data and "content" in chunk_data:
                related_chunks.append(chunk_data["content"])

    context = "\n".join(related_chunks)
    stream_func = global_config.get("llm_model_stream_func")
    if stream_func is None:
        yield context
        return

    sys_prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the following context.\n"
        "If the context does not contain the answer, say 'The context does not provide this information.'\n\n"
        f"Context:\n{context}"
    )
    async for tok in stream_func(query, system_prompt=sys_prompt):
        yield tok


async def llm_query_stream(query, param, global_config):
    """LLM直接回答流式模式：不检索外部知识，流式输出

    Args:
        query: 查询文本
        param: 查询参数
        global_config: 全局配置

    Yields:
        生成的token流
    """
    stream_func = global_config.get("llm_model_stream_func")
    if stream_func is None:
        yield "Error: No stream function configured."
        return

    sys_prompt = "You are a helpful assistant. Answer the question to the best of your ability."
    async for tok in stream_func(query, system_prompt=sys_prompt):
        yield tok


@dataclass
class HyperRAG:
    working_dir: str = field(
        default_factory=lambda: f"./HyperRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    log_level: str = field(default_factory=lambda: str(logger.level))

    # text chunking
    chunk_token_size: int = 2000
    chunk_overlap_token_size: int = 200
    tiktoken_model_name: str = "gpt-4o-mini"
    # 语义分块参数
    use_semantic_chunking: bool = True
    semantic_chunking_threshold: float = 0.7
    semantic_chunking_min_tokens: int = 1200
    semantic_chunking_max_tokens: int = 1500
    semantic_chunking_max_chunk_size: int = 1500

    # entity extraction
    entity_extract_max_gleaning: int = 0
    entity_summary_to_max_tokens: int = 500
    entity_additional_properties_to_max_tokens: int = 250
    relation_summary_to_max_tokens: int = 750

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = ""
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    llm_model_stream_func: callable = None

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    
    # ===== 性能优化配置 =====
    # 快速模式：启用后会降低部分精度以换取速度
    fast_mode: bool = False
    
    # 批量摘要大小：每批处理的实体/关系数量
    batch_summary_size: int = 10
    
    # 并行处理的最大并发数
    max_parallel_chunks: int = 10
    
    # 是否启用批量摘要优化
    enable_batch_summaries: bool = True

    def __post_init__(self):
        # 应用快速模式配置
        if self.fast_mode:
            logger.info("Fast mode enabled: applying performance optimizations")
            self.chunk_token_size = 2000  # 更大的块减少 chunk 数量
            self.chunk_overlap_token_size = 50  # 减少重叠
            self.entity_extract_max_gleaning = 1  # 关闭多轮补充
            self.entity_summary_to_max_tokens = 200  # 减少摘要长度
            self.entity_additional_properties_to_max_tokens = 150
            self.relation_summary_to_max_tokens = 300
            self.batch_summary_size = 10  # 增大批量大小
            self.max_parallel_chunks = 10  # 增大并发数
            # 语义分块参数调整 - 使用用户设置的参数
            # 保留用户设置的参数值，不进行硬编码覆盖
        
        log_file = os.path.join(self.working_dir, "HyperRAG.log")
        set_logger(log_file)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        
        # 记录优化配置
        if self.enable_batch_summaries:
            logger.info(f"Batch summaries enabled with size: {self.batch_summary_size}")
        logger.info(f"Max parallel chunks: {self.max_parallel_chunks}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"HyperRAG init with param:\n  {_print_config}\n")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        """
            download from hgdb_path
        """
        # 移除超图数据库初始化，只使用复形数据库
        # self.chunk_entity_relation_hypergraph = self.hypergraph_storage_cls(
        #     namespace="simple_rag", global_config=asdict(self)
        # )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.simplex_storage = SimplexStorage(
            namespace="rag", 
            global_config=asdict(self),
            embedding_func=self.embedding_func
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "entity_type", "description", "additional_properties", "frequency", "source_id", "importance"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"id_set", "dimension", "description", "frequency", "source_id", "importance"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # 验证 llm_model_func 是否是可调用的
        if not callable(self.llm_model_func):
            logger.error(f"Invalid llm_model_func: {self.llm_model_func}")
            # 使用默认的 LLM 函数
            from .llm import gpt_4o_mini_complete
            self.llm_model_func = gpt_4o_mini_complete
        
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

        if getattr(self, "llm_model_stream_func", None) is not None:
            # 先把 hashing_kv 注入到 stream func（供 openai_complete_stream_if_cache 使用）
            self.llm_model_stream_func = limit_async_gen_call(self.llm_model_max_async)(
                partial(
                    self.llm_model_stream_func,
                    hashing_kv=self.llm_response_cache,
                    **self.llm_model_kwargs,
                )
            )

    def insert(self, string_or_strings):
        return asyncio.run(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            # ----------------------------------------------------------------------------
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                # 根据配置选择分块方法
                if self.use_semantic_chunking:
                    try:
                        from .operate import chunking_by_semantic
                        chunks = {
                            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                **dp,
                                "full_doc_id": doc_key,
                            }
                            for dp in chunking_by_semantic(
                                doc["content"],
                                config={
                                    'semantic_chunking_threshold': self.semantic_chunking_threshold,
                                    'semantic_chunking_min_tokens': self.semantic_chunking_min_tokens,
                                    'semantic_chunking_max_tokens': self.semantic_chunking_max_tokens,
                                    'semantic_chunking_max_chunk_size': self.semantic_chunking_max_chunk_size,
                                }
                            )
                        }
                        logger.info(f"Using semantic chunking for doc {doc_key}")
                    except Exception as e:
                        logger.error(f"Semantic chunking failed: {e}")
                        # 回退到基于token的分块
                        from .operate import chunking_by_token_size
                        chunks = {
                            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                **dp,
                                "full_doc_id": doc_key,
                            }
                            for dp in chunking_by_token_size(
                                doc["content"],
                                overlap_token_size=self.chunk_overlap_token_size,
                                max_token_size=self.chunk_token_size,
                                tiktoken_model=self.tiktoken_model_name,
                            )
                        }
                        logger.info(f"Falling back to token-based chunking for doc {doc_key}")
                else:
                    from .operate import chunking_by_token_size
                    chunks = {
                        compute_mdhash_id(dp["content"], prefix="chunk-"): {
                            **dp,
                            "full_doc_id": doc_key,
                        }
                        for dp in chunking_by_token_size(
                            doc["content"],
                            overlap_token_size=self.chunk_overlap_token_size,
                            max_token_size=self.chunk_token_size,
                            tiktoken_model=self.tiktoken_model_name,
                        )
                    }
                    logger.info(f"Using token-based chunking for doc {doc_key}")
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            # ----------------------------------------------------------------------------
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)
            # ----------------------------------------------------------------------------
            logger.info("[Entity Extraction]...")
            extract_start = time.time()
            failed_chunks = set()
            try:
                logger.info("Calling extract_entities function")
                failed_chunks = await extract_entities(
                    inserting_chunks,
                    self.entities_vdb,
                    self.relationships_vdb,
                    asdict(self),
                    simplex_storage=self.simplex_storage,
                )
                if failed_chunks is None:
                    failed_chunks = set()
                extract_end = time.time()
                logger.info(f"extract_entities completed in {extract_end - extract_start:.2f} seconds")
                
                if failed_chunks:
                    logger.warning(f"Entity extraction failed for {len(failed_chunks)} chunks, these chunks will not be stored")
                logger.info("Entity extraction completed")
            except Exception as e:
                logger.error(f"Error during entity extraction: {e}")
                import traceback
                traceback.print_exc()
                return
            # 结构发现已在实体提取阶段完成
            
            # ----------------------------------------------------------------------------
            # 移除超图数据库赋值，只使用复形数据库
            # self.chunk_entity_relation_hypergraph = maybe_new_kg
            logger.info("Final upsert of docs and chunks")
            await self.full_docs.upsert(new_docs)
            logger.info("Final upsert of full_docs completed")
            # 排除提取失败的chunk，不存入text_chunks，以便下次重新提取
            successful_chunks = {k: v for k, v in inserting_chunks.items() if k not in failed_chunks}
            if successful_chunks:
                await self.text_chunks.upsert(successful_chunks)
                logger.info(f"Final upsert of text_chunks completed ({len(successful_chunks)} successful, {len(failed_chunks)} failed)")
            else:
                logger.warning("No successful chunks to store in text_chunks")
            logger.info("All operations completed successfully")
        finally:
            logger.info("Starting finally block: calling _insert_done")
            start_time = time.time()
            try:
                await self._insert_done()
                end_time = time.time()
                logger.info(f"Completed _insert_done in {end_time - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in _insert_done: {e}")
                import traceback
                traceback.print_exc()

    async def _insert_done(self):
        start_time = time.time()
        logger.info("Starting _insert_done: saving all storage instances")
        
        # 逐个执行存储的index_done_callback，以便更好地定位问题
        storage_instances = [
            ("full_docs", self.full_docs),
            ("text_chunks", self.text_chunks),
            ("llm_response_cache", self.llm_response_cache),
            ("entities_vdb", self.entities_vdb),
            ("relationships_vdb", self.relationships_vdb),
            ("chunks_vdb", self.chunks_vdb),
            ("simplex_storage", self.simplex_storage),
        ]
        
        for name, storage_inst in storage_instances:
            if storage_inst is None:
                logger.info(f"Skipping {name}: storage instance is None")
                continue
            
            logger.info(f"Starting to save {name}")
            save_start = time.time()
            try:
                await cast(StorageNameSpace, storage_inst).index_done_callback()
                save_end = time.time()
                logger.info(f"Completed saving {name} in {save_end - save_start:.2f} seconds")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
                import traceback
                traceback.print_exc()
        
        end_time = time.time()
        logger.info(f"Completed _insert_done in {end_time - start_time:.2f} seconds")

    def query(self, query: str, param: QueryParam = QueryParam()):
        return asyncio.run(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        
        if param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "llm":
            response = await llm_query(
                query,
                param,
                asdict(self),
            )
        elif param.mode == "topology":
            # 拓扑检索 - 基于 SimplicialRAGRetriever 的实现
            from .operate import topology_retrieval
            
            # 获取拓扑检索配置
            topology_config = asdict(self)
            # 添加拓扑检索参数
            topology_config.update({
                "enable_llm_keyword_extraction": True,
                "max_topology_chunks": getattr(param, 'max_topology_chunks', 60),
                "diffusion_steps": 2,
                "max_simplices": 50,
                "min_coverage_ratio": 0.5,
                "chunks_vdb": self.chunks_vdb,
                "embedding_func": self.embedding_func,
                "llm_model_func": self.llm_model_func,
            })
            
            retrieval_result = await topology_retrieval(
                query,
                self.simplex_storage,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                topology_config,
            )
            
            # 构建CSV格式结构化上下文
            context_parts = []

            structured_entities = retrieval_result.get("structured_entities", [])
            structured_simplices = retrieval_result.get("structured_simplices", [])
            related_chunks = retrieval_result.get("related_chunks", [])

            # ===== 对齐Hybrid基线：Sources独立token预算 + 结构化信息噪声过滤 =====
            sources_token_budget = DualDimensionConfig.SOURCES_TOKEN_BUDGET
            structure_token_budget = DualDimensionConfig.STRUCTURE_TOKEN_BUDGET

            # Sources段：按token预算截断
            sources_csv_lines = ["id,content"]
            sources_tokens = len(encode_string_by_tiktoken("id,content\n"))
            for i, chunk in enumerate(related_chunks, 1):
                content = chunk.replace('"', '""').replace('\n', ' ')
                line = f'"{i}","{content}"'
                line_tokens = len(encode_string_by_tiktoken(line))
                if sources_tokens + line_tokens > sources_token_budget:
                    break
                sources_csv_lines.append(line)
                sources_tokens += line_tokens

            sources_section = "-----Sources-----\n```csv\n" + "\n".join(sources_csv_lines) + "\n```"

            # 结构化信息只保留与Sources直接相关的条目
            sources_text = " ".join(related_chunks[:len(sources_csv_lines) - 1])
            sources_entity_names = set()
            for ent in structured_entities:
                if ent['name'].upper() in sources_text.upper():
                    sources_entity_names.add(ent['name'])
            for ent in structured_entities:
                if ent.get('is_seed'):
                    sources_entity_names.add(ent['name'])

            filtered_entities = [
                ent for ent in structured_entities
                if ent['name'] in sources_entity_names
            ][:DualDimensionConfig.MAX_STRUCTURE_ENTITY_COUNT]

            filtered_simplices = []
            for simp in structured_simplices:
                simp_entities = simp.get('entities', [])
                if any(str(e) in sources_entity_names for e in simp_entities):
                    filtered_simplices.append(simp)
                elif simp.get('is_seed'):
                    filtered_simplices.append(simp)
            filtered_simplices = filtered_simplices[:DualDimensionConfig.MAX_STRUCTURE_SIMPLEX_COUNT]

            # 构建Entities CSV
            entity_section = ""
            if filtered_entities:
                entity_csv_lines = ["name,type,is_seed,description"]
                entity_tokens = len(encode_string_by_tiktoken("name,type,is_seed,description\n"))
                for ent in filtered_entities:
                    ent_type = ent.get('type', 'Entity')
                    is_seed = "yes" if ent.get('is_seed') else "no"
                    desc = (ent.get('description', '') or '').replace('"', '""').replace('\n', ' ')
                    name = ent['name'].replace('"', '""')
                    line = f'"{name}","{ent_type}","{is_seed}","{desc}"'
                    line_tokens = len(encode_string_by_tiktoken(line))
                    if entity_tokens + line_tokens > structure_token_budget // 2:
                        break
                    entity_csv_lines.append(line)
                    entity_tokens += line_tokens
                entity_section = "-----Entities-----\n```csv\n" + "\n".join(entity_csv_lines) + "\n```"

            # 构建Simplices CSV
            simplex_section = ""
            if filtered_simplices:
                simplex_csv_lines = ["dimension,entities,is_seed,description"]
                simplex_tokens = len(encode_string_by_tiktoken("dimension,entities,is_seed,description\n"))
                for simp in filtered_simplices:
                    dim = simp.get('dimension', 1)
                    entities = simp.get('entities', [])
                    is_seed = "yes" if simp.get('is_seed') else "no"
                    desc = (simp.get('description', '') or '').replace('"', '""').replace('\n', ' ')
                    ent_str = ", ".join(str(e) for e in entities).replace('"', '""')
                    dim_label = {1: "1-MSG", 2: "2-MSG", 3: "3-MSG"}.get(dim, f"{dim}-MSG")
                    line = f'"{dim_label}","{ent_str}","{is_seed}","{desc}"'
                    line_tokens = len(encode_string_by_tiktoken(line))
                    if simplex_tokens + line_tokens > structure_token_budget // 2:
                        break
                    simplex_csv_lines.append(line)
                    simplex_tokens += line_tokens
                simplex_section = "-----Simplices-----\n```csv\n" + "\n".join(simplex_csv_lines) + "\n```"

            # 组装上下文：Sources放最前面
            context_parts = [sources_section]
            if entity_section:
                context_parts.append(entity_section)
            if simplex_section:
                context_parts.append(simplex_section)

            context = "\n\n".join(context_parts)

            # 简化Prompt，不注入prompt_instructions
            use_model_func = self.llm_model_func
            from .prompt import PROMPTS
            sys_prompt = PROMPTS["topology_response_system_prompt_concise"].format(
                prompt_instructions="",
                context=context
            )
            
            # 调用LLM生成回答
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
            )
            
            # 处理响应
            if len(response) > len(sys_prompt):
                response = (
                    response.replace(sys_prompt, "")
                    .replace("user", "")
                    .replace("model", "")
                    .replace(query, "")
                    .replace("<system>", "")
                    .replace("</system>", "")
                    .strip()
                )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        if param.return_retrieval_result and param.mode == "topology":
            return {
                "response": response,
                "related_chunks": retrieval_result.get("related_chunks", []),
                "structured_entities": retrieval_result.get("structured_entities", []),
                "structured_simplices": retrieval_result.get("structured_simplices", []),
            }
        return response

    async def astream_query(self, query: str, param: QueryParam = QueryParam()):
        """
        流式查询：返回 async generator（逐 token / 逐块）
        依赖 self.llm_model_stream_func，不提供则抛错。
        """
        if self.llm_model_stream_func is None:
            raise AttributeError("llm_model_stream_func is not set, streaming is unavailable.")

        # 把 stream func 放进 global_config
        cfg = asdict(self)
        cfg["llm_model_stream_func"] = self.llm_model_stream_func

        if param.mode == "naive":
            async for tok in naive_query_stream(
                    query,
                    self.chunks_vdb,
                    self.text_chunks,
                    param,
                    cfg,
            ):
                yield tok

        elif param.mode == "llm":
            async for tok in llm_query_stream(query, param, cfg):
                yield tok

        elif param.mode == "topology":
            # 基于 SimplicialRAGRetriever 的拓扑检索流式查询
            from .operate import topology_retrieval
            
            # 获取拓扑检索配置
            topology_config = cfg.copy()
            topology_config.update({
                "enable_llm_keyword_extraction": True,
                "max_topology_chunks": getattr(param, 'max_topology_chunks', 20),
                "diffusion_steps": 2,
                "chunks_vdb": self.chunks_vdb,
                "embedding_func": self.embedding_func,
                "llm_model_func": self.llm_model_func,
            })
            
            retrieval_result = await topology_retrieval(
                query,
                self.simplex_storage,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                topology_config,
            )
            
            context_parts = []
            structured_entities = retrieval_result.get("structured_entities", [])
            if structured_entities:
                entity_csv_lines = ["name,type,is_seed,description"]
                for ent in structured_entities:
                    seed_mark = "yes" if ent.get('is_seed') else "no"
                    ent_type = ent.get('type', 'Entity')
                    desc = (ent.get('description', '') or '').replace('"', '""').replace('\n', ' ')
                    name = ent['name'].replace('"', '""')
                    entity_csv_lines.append(f'"{name}","{ent_type}","{seed_mark}","{desc}"')
                context_parts.append("-----Entities-----\n```csv\n" + "\n".join(entity_csv_lines) + "\n```")
            structured_simplices = retrieval_result.get("structured_simplices", [])
            if structured_simplices:
                simplex_csv_lines = ["dimension,entities,is_seed,description"]
                for simp in structured_simplices:
                    dim = simp.get('dimension', 1)
                    entities = simp.get('entities', [])
                    desc = (simp.get('description', '') or '').replace('"', '""').replace('\n', ' ')
                    ent_str = ", ".join(str(e) for e in entities).replace('"', '""')
                    seed_mark = "yes" if simp.get('is_seed') else "no"
                    dim_label = {1: "1-MSG", 2: "2-MSG", 3: "3-MSG"}.get(dim, f"{dim}-MSG")
                    simplex_csv_lines.append(f'"{dim_label}","{ent_str}","{seed_mark}","{desc}"')
                context_parts.append("-----Simplices-----\n```csv\n" + "\n".join(simplex_csv_lines) + "\n```")
            related_chunks = retrieval_result.get("related_chunks", [])
            if related_chunks:
                chunk_csv_lines = ["id,content"]
                for i, chunk in enumerate(related_chunks, 1):
                    content = chunk.replace('"', '""').replace('\n', ' ')
                    chunk_csv_lines.append(f'"{i}","{content}"')
                context_parts.append("-----Sources-----\n```csv\n" + "\n".join(chunk_csv_lines) + "\n```")
            context = "\n\n".join(context_parts)
            prompt_instructions = "\n".join(retrieval_result["prompt_instructions"])
            
            from .prompt import PROMPTS
            sys_prompt = PROMPTS["topology_response_system_prompt_concise"].format(
                prompt_instructions=prompt_instructions,
                context=context
            )
            
            async for tok in self.llm_model_stream_func(
                query,
                system_prompt=sys_prompt,
            ):
                yield tok

        else:
            raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()


    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).query_done_callback())
        await asyncio.gather(*tasks)
