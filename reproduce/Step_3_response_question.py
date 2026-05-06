import sys
import asyncio
import os
import json
import psutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hyperrag import HyperRAG, QueryParam
from hyperrag.utils import always_get_an_event_loop, wrap_embedding_func_with_attrs
from hyperrag.llm import openai_embedding, openai_complete_if_cache
from hyperrag.operate import topology_retrieval
from hyperrag.prompt import PROMPTS

from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_MODEL, EMB_DIM

_hsc_lock = asyncio.Lock()

async def llm_model_func(
    prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    """LLM模型函数，用于复形RAG系统
    
    设置temperature=0确保输出确定性，消除同一输入产生不同输出的随机性
    """
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=EMB_DIM, max_token_size=8192)
async def embedding_func(texts: list[str]) -> np.ndarray:
    """嵌入函数，用于复形RAG系统"""
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def extract_queries(file_path):
    """提取查询，添加错误处理和输入验证"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Queries file not found: {file_path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Queries path is not a file: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as file:
            query_list = json.load(file)
        
        if not isinstance(query_list, list):
            raise ValueError("Queries file must contain a list of queries")
        
        # 过滤空查询
        query_list = [q for q in query_list if q and isinstance(q, str) and q.strip()]
        
        return query_list
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decoding error in queries file: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Error extracting queries: {e}")
        raise




async def process_query(query_text, rag_instance, query_param):
    """处理查询"""
    try:
        # 执行查询
        result = await rag_instance.aquery(query_text, param=query_param)
        
        # 记录查询详情
        query_info = {
            "original_query": query_text,
            "parsed_query": query_text,
            "mode": query_param.mode,
            "result": result
        }
        return query_info, None
    except Exception as e:
        print(f"Error processing query '{query_text}': {e}")
        return None, {"query": query_text, "error": str(e)}


async def process_query_with_topology(query_text, rag_instance, mode="topology"):
    """使用拓扑增强检索处理查询
    
    关键修复：对HSC拓扑操作加全局锁，防止并发查询互相污染共享HSC状态。
    HSC的build_dynamic_incidence_matrices/compute_dynamic_hodge_laplacians
    会修改共享矩阵，并发执行会导致扩散得分被覆盖、检索结果不稳定。
    """
    try:
        from dataclasses import asdict
        result = None
        topology_config = asdict(rag_instance)
        topology_config.update({
            "enable_llm_keyword_extraction": True,
            "max_simplices": 80,
            "max_topology_chunks": 4,
            "max_context_tokens": 60000,
            "llm_model_func": rag_instance.llm_model_func,
            "embedding_func": rag_instance.embedding_func,
            "chunks_vdb": rag_instance.chunks_vdb,
        })
        async with _hsc_lock:
            result = await topology_retrieval(
                query_text,
                rag_instance.simplex_storage,
                rag_instance.entities_vdb,
                rag_instance.relationships_vdb,
                rag_instance.text_chunks,
                topology_config
            )
        print(f"Topology retrieval result received: {type(result).__name__}")
        print(f"  - ranked_simplices: {len(result.get('ranked_simplices', []))}")
        print(f"  - related_chunks: {len(result.get('related_chunks', []))}")
        print(f"  - prompt_instructions: {len(result.get('prompt_instructions', []))}")
        
        if result is None:
            return {"original_query": query_text, "parsed_query": query_text, "mode": mode, "result": "Error: Failed to retrieve topology information"}, None
        
        related_chunks = result.get('related_chunks', [])
        print(f"Related chunks count: {len(related_chunks)}")

        # 使用检索结果已提供的 structured_entities/structured_simplices，
        # 以CSV格式构建上下文，与Prompt期望的格式一致
        context_parts = []

        structured_entities = result.get("structured_entities", [])
        if structured_entities:
            entity_csv_lines = ["name,type,is_seed,description"]
            for ent in structured_entities:
                seed_mark = "yes" if ent.get('is_seed') else "no"
                ent_type = ent.get('type', 'Entity')
                desc = (ent.get('description', '') or '').replace('"', '""').replace('\n', ' ')
                name = ent['name'].replace('"', '""')
                entity_csv_lines.append(f'"{name}","{ent_type}","{seed_mark}","{desc}"')
            context_parts.append("-----Entities-----\n```csv\n" + "\n".join(entity_csv_lines) + "\n```")

        structured_simplices = result.get("structured_simplices", [])
        if structured_simplices:
            simplex_csv_lines = ["dimension,entities,is_seed,description"]
            for simp in structured_simplices:
                dim = simp.get('dimension', 1)
                entities = simp.get('entities', [])
                desc = (simp.get('description', '') or '').replace('"', '""').replace('\n', ' ')
                ent_str = ", ".join(str(e) for e in entities).replace('"', '""')
                seed_mark = "yes" if simp.get('is_seed') else "no"
                dim_label = {1: "Relation", 2: "Triangle", 3: "Tetrahedron"}.get(dim, f"{dim}D-Simplex")
                simplex_csv_lines.append(f'"{dim_label}","{ent_str}","{seed_mark}","{desc}"')
            context_parts.append("-----Simplices-----\n```csv\n" + "\n".join(simplex_csv_lines) + "\n```")

        if related_chunks:
            seen_content = set()
            chunk_csv_lines = ["id,content"]
            chunk_idx = 0
            for chunk in related_chunks:
                chunk_stripped = chunk.strip()
                if chunk_stripped and chunk_stripped not in seen_content:
                    seen_content.add(chunk_stripped)
                    chunk_idx += 1
                    content = chunk_stripped.replace('"', '""').replace('\n', ' ')
                    chunk_csv_lines.append(f'"{chunk_idx}","{content}"')
            context_parts.append("-----Sources-----\n```csv\n" + "\n".join(chunk_csv_lines) + "\n```")

        context = "\n\n".join(context_parts)
        entity_count = len(structured_entities)
        relation_count = len(structured_simplices)
        source_count = len(related_chunks)
        print(f"Structured context length: {len(context)}, entities={entity_count}, relations={relation_count}, sources={source_count}")
        
        if not context:
            context = "No relevant content retrieved."
        
        prompt_instructions = "\n".join(result.get('prompt_instructions', []))
        print(f"Prompt instructions length: {len(prompt_instructions)}")
        
        use_model_func = rag_instance.llm_model_func
        sys_prompt = PROMPTS["topology_response_system_prompt_concise"].format(
            prompt_instructions=prompt_instructions,
            context=context
        )
        
        try:
            final_answer = await use_model_func(
                query_text,
                system_prompt=sys_prompt,
            )
            print(f"LLM response received, length: {len(final_answer)}")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            final_answer = "Error: Failed to generate answer from LLM"
        
        if final_answer.startswith(sys_prompt):
            final_answer = final_answer[len(sys_prompt):].strip()
        final_answer = final_answer.replace("<system>", "").replace("</system>", "").strip()
        
        # 直接使用检索结果中已计算好的 simplex_counts，避免重复遍历
        raw_simplex_counts = result.get('simplex_counts', {})
        simplex_counts = {f"simplex-{k}": v for k, v in raw_simplex_counts.items()}
        # 确保所有维度都有默认值
        for d in range(4):
            key = f"simplex-{d}"
            if key not in simplex_counts:
                simplex_counts[key] = 0
        
        query_info = {
            "original_query": query_text,
            "parsed_query": query_text,
            "mode": mode,
            "result": final_answer,
            "retrieval_details": {
                "ranked_simplices": len(result.get('ranked_simplices', [])),
                "simplex_counts": simplex_counts,
                "related_chunks": len(result.get('related_chunks', []))
            }
        }
        return query_info, None
    except Exception as e:
        print(f"Error processing {mode} query '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return None, {"query": query_text, "error": str(e)}


async def process_query_with_topology_with_semaphore(query_text, rag_instance, semaphore, index, mode="topology"):
    """使用信号量控制并发的查询处理"""
    async with semaphore:
        result, error = await process_query_with_topology(query_text, rag_instance, mode=mode)
        return result, error, index

async def run_queries_and_save_to_json(
    queries, rag_instance, output_dir, question_stage, selected_modes
):
    """运行查询并保存结果（异步并行版本）"""
    # 动态计算最优并发数
    cpu_count = os.cpu_count() or 4
    memory_available = psutil.virtual_memory().available / (1024**3)
    
    # 智能计算并发数：考虑系统资源和查询数量
    # 规则：
    # 1. CPU核数 * 2（避免CPU过载）
    # 2. 可用内存 / 2GB（每个并发任务约需2GB内存）
    # 3. 查询数量（如果查询少，不需要太高并发）
    # 4. LLM服务的最大并发限制（通常API有速率限制）
    base_concurrency = min(
        cpu_count * 2, 
        int(memory_available / 2), 
        len(queries) if len(queries) < 32 else 32,
        16  # LLM API通常有限制，保守设置
    )
    
    # 如果查询数量很少，进一步降低并发以避免资源浪费
    if len(queries) <= 3:
        max_concurrency = len(queries)
    elif len(queries) <= 5:
        max_concurrency = min(base_concurrency, 4)
    else:
        max_concurrency = base_concurrency
    
    print(f"[INFO] System: {cpu_count} CPUs, {memory_available:.1f} GB available")
    print(f"[INFO] Max concurrency set to: {max_concurrency}")
    
    # 处理指定的模式
    modes = selected_modes
    
    for mode in modes:
        print(f"\nProcessing queries with {mode} mode...")
        
        # 并发控制：使用智能计算的并发数
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # 存储查询任务和原始索引
        tasks = []
        for i, query_text in enumerate(queries):
            # 创建带索引的任务，以便后续排序
            task = asyncio.create_task(
                process_query_with_topology_with_semaphore(
                    query_text, rag_instance, semaphore, i, mode=mode
                )
            )
            tasks.append(task)
        
        # 等待所有任务完成，显示进度
        print(f"Processing {len(queries)} queries in parallel with {mode} mode...")
        from tqdm.asyncio import tqdm_asyncio
        results_with_index = await tqdm_asyncio.gather(*tasks, desc=f"Processing {mode} queries", unit="query")
        
        # 按原始顺序整理结果
        sorted_results = sorted(results_with_index, key=lambda x: x[2])
        
        # 写入文件
        result_file = open(output_dir / f"{mode}_{question_stage}_stage_result.json", "w", encoding="utf-8")
        error_file = open(output_dir / f"{mode}_{question_stage}_stage_errors.json", "w", encoding="utf-8")
        result_file.write("[\n")
        error_file.write("[\n")
        
        has_results = False
        error_count = 0
        
        try:
            print(f"Processing {len(sorted_results)} results for {mode} mode...")
            for i, (result, error, _) in enumerate(sorted_results):
                print(f"Processing result {i+1}/{len(sorted_results)} for {mode} mode")
                
                if result:
                    if has_results:
                        result_file.write(",\n")
                    print(f"Writing result for query: {result.get('original_query', 'Unknown')[:50]}...")
                    json.dump(result, result_file, ensure_ascii=False, indent=4)
                    has_results = True
                elif error:
                    print(f"Writing error for query: {error.get('query', 'Unknown')[:50]}...")
                    if error_count > 0:
                        error_file.write(",\n")
                    json.dump(error, error_file, ensure_ascii=False, indent=4)
                    error_count += 1
            
            if not has_results:
                print(f"No results found for {mode} mode, writing empty object...")
                result_file.write("{}")
        finally:
            result_file.write("\n]")
            error_file.write("\n]")
            result_file.close()
            error_file.close()
            print(f"File writing completed for {mode} mode.")


def evaluate_responses(output_dir, question_stage=3):
    """评估响应质量"""
    print("\nEvaluating responses...")
    
    modes = ["topology"]
    evaluation_results = {}
    
    for mode in modes:
        result_file = output_dir / f"{mode}_{question_stage}_stage_result.json"
        if not result_file.exists():
            import glob
            stage_files = glob.glob(str(output_dir / f"{mode}_*_stage_result.json"))
            if stage_files:
                result_file = Path(sorted(stage_files)[-1])
        
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            if not isinstance(results, list):
                results = [results]
            
            total_responses = len(results)
            valid_responses = 0
            avg_response_length = 0
            
            for result in results:
                if isinstance(result, dict) and "result" in result and result["result"] is not None:
                    result_str = str(result["result"]).strip()
                    fail_phrases = [
                        "the context does not provide",
                        "the provided context does not contain",
                        "does not contain this information",
                        "error:",
                        "failed to retrieve",
                        "failed to generate",
                    ]
                    is_fail = any(phrase in result_str.lower() for phrase in fail_phrases)
                    if result_str and not is_fail:
                        valid_responses += 1
                        avg_response_length += len(result_str)
            
            if valid_responses > 0:
                avg_response_length /= valid_responses
            
            evaluation_results[mode] = {
                "total": total_responses,
                "valid": valid_responses,
                "valid_rate": valid_responses / total_responses if total_responses > 0 else 0,
                "avg_length": avg_response_length
            }
            
            print(f"{mode} mode: {valid_responses}/{total_responses} valid responses, avg length: {avg_response_length:.2f}")
    
    return evaluation_results


if __name__ == "__main__":
    try:
        # 配置参数
        data_name = "legal"  
        question_stage = 1
        # 选择要运行的模式，可根据需要修改
        # 可选值: "topology"
        # selected_modes = ["topology"]
        selected_modes = ["topology"]
        
        # 获取项目根目录（脚本所在目录的父目录）
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        
        # 为每个数据集使用单独的工作目录
        WORKING_DIR = PROJECT_ROOT / "caches" / data_name
        WORKING_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Processing questions for Simplicial Complex RAG")
        print(f"[INFO] Working directory: {WORKING_DIR}")
        print("[INFO] Using dataset-specific database storage")
        print(f"[INFO] Selected modes: {', '.join(selected_modes)}")
        
        # 输入问题 - 从原始数据集目录读取问题文件
        question_file_path = PROJECT_ROOT / "caches" / data_name / "questions" / f"{question_stage}_stage.json"
        
        if not question_file_path.exists():
            print(f"[ERROR] Questions file not found: {question_file_path}")
            print("[WARN]  Please run Step_2_extract_question.py first to generate questions")
            sys.exit(1)
        
        # 提取查询
        queries = extract_queries(question_file_path)
        print(f"[INFO] Found {len(queries)} valid queries")
        
        # 只取前十条问题
        queries = queries[:10]
        print(f"[INFO] Selected first 10 queries for processing")
        
        if not queries:
            print("[ERROR] No valid queries found")
            sys.exit(1)
        
        # 初始化HyperRAG
        print("\n[INFO] Initializing Simplicial Complex RAG...")
        # 动态计算最优并发配置
        cpu_count = os.cpu_count() or 4
        memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # 根据系统资源动态设置并发数
        optimal_llm_async = min(cpu_count * 4, int(memory_gb * 4), 64)
        optimal_embedding_async = min(cpu_count * 2, int(memory_gb * 8), 32)
        optimal_batch_size = min(cpu_count * 4, int(memory_gb * 10), 64)
        
        print(f"[INFO] System resources: {cpu_count} CPUs, {memory_gb:.1f} GB available memory")
        print(f"[INFO] Optimal concurrency: LLM={optimal_llm_async}, Embedding={optimal_embedding_async}, Batch={optimal_batch_size}")
        
        rag = HyperRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            # ===== 性能优化配置 =====
            llm_model_max_async=optimal_llm_async,       # 动态调整LLM并发数
            embedding_func_max_async=optimal_embedding_async,  # 动态调整嵌入并发数
            embedding_batch_num=optimal_batch_size,      # 动态调整嵌入批次大小
            enable_batch_summaries=True,                # 启用批量摘要优化
            batch_summary_size=20,                      # 增大批量摘要大小
        )
        print("[OK] Simplicial Complex RAG initialized successfully")
        
        # 运行查询 - 响应文件保存到原始数据集目录
        OUT_DIR = PROJECT_ROOT / "caches" / data_name / "response"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Saving responses to: {OUT_DIR}")
        
        # 使用 asyncio.run 执行异步函数
        asyncio.run(
            run_queries_and_save_to_json(
                queries,
                rag,
                OUT_DIR,
                question_stage,
                selected_modes
            )
        )
        
        # 评估结果
        print("\n[INFO] Evaluating responses...")
        evaluation_results = evaluate_responses(OUT_DIR, question_stage)
        
        # 打印评估摘要
        print("\n[RESULT] Evaluation Summary:")
        for mode, result in evaluation_results.items():
            print(f"{mode}: {result['valid']}/{result['total']} valid responses ({result['valid_rate']:.2f}), avg length: {result['avg_length']:.2f}")
        
        print("\n[OK] Question response process completed successfully!")
        print(f"\n[INFO] Next steps:")
        print(f"   1. Check the response files in: {OUT_DIR}")
        print("   2. Compare different query modes to see which works best")
        
    except Exception as e:
        print(f"\n[ERROR] Program failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
