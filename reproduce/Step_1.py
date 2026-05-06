import sys
import json
import random
import gc
import asyncio
import psutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

from hyperrag import HyperRAG
from hyperrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs
from hyperrag.llm import openai_embedding, openai_complete_if_cache

from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_MODEL, EMB_DIM


def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # 转换为MB


def log_memory_usage(stage):
    """记录内存使用情况"""
    mem_usage = get_memory_usage()
    print(f"Memory usage at {stage}: {mem_usage:.2f} MB")


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """LLM模型函数，用于复形RAG系统"""
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
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


def process_contexts(file_path, batch_size=500):
    """处理上下文文件，使用流式读取减少内存使用"""
    try:
        log_memory_usage("before loading contexts")
        
        # 使用生成器流式处理上下文，避免一次性加载全部数据
        def context_generator():
            with open(file_path, "r", encoding="utf-8") as f:
                # 尝试流式解析JSON
                content = f.read()
                # 简单处理，假设JSON是一个列表
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    # 移除首尾括号
                    content = content.strip()[1:-1]
                    # 分割元素
                    items = []
                    current_item = []
                    bracket_depth = 0
                    quote_depth = 0
                    escape = False
                    
                    for char in content:
                        if escape:
                            current_item.append(char)
                            escape = False
                        elif char == '\\':
                            current_item.append(char)
                            escape = True
                        elif char == '"':
                            quote_depth = 1 - quote_depth
                            current_item.append(char)
                        elif char == '{' or char == '[':
                            if quote_depth == 0:
                                bracket_depth += 1
                            current_item.append(char)
                        elif char == '}' or char == ']':
                            if quote_depth == 0:
                                bracket_depth -= 1
                            current_item.append(char)
                        elif char == ',' and quote_depth == 0 and bracket_depth == 0:
                            # 一个完整的项目
                            item_str = ''.join(current_item).strip()
                            if item_str:
                                try:
                                    yield json.loads(item_str)
                                except json.JSONDecodeError:
                                    pass
                            current_item = []
                        else:
                            current_item.append(char)
                    
                    # 处理最后一个项目
                    if current_item:
                        item_str = ''.join(current_item).strip()
                        if item_str:
                            try:
                                yield json.loads(item_str)
                            except json.JSONDecodeError:
                                pass
        
        # 收集上下文并分批处理
        contexts = []
        total_contexts = 0
        
        for context in context_generator():
            if isinstance(context, str):
                contexts.append(context)
                total_contexts += 1
                
                # 当达到批处理大小或处理完所有数据时，返回批次
                if len(contexts) >= batch_size:
                    yield contexts.copy()
                    contexts.clear()
                    # 清理内存
                    gc.collect()
                    log_memory_usage(f"after processing batch of {batch_size}")
        
        # 返回最后一批
        if contexts:
            yield contexts
        
        log_memory_usage("after loading contexts")
        print(f"\nProcessed {total_contexts} contexts")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decoding error in contexts file: {e}")
        raise
    except Exception as e:
        print(f"❌ Error processing contexts file: {e}")
        raise

async def insert_text(rag, context_batches, batch_size=5, max_retries=3):
    """插入文本到复形RAG系统"""
    total_contexts = 0
    for batch in context_batches:
        total_contexts += len(batch)
    
    if total_contexts == 0:
        print("❌ No valid contexts to insert")
        return
    
    print(f"Inserting {total_contexts} cleaned contexts into Simplicial Complex RAG")
    
    total_batches = 0
    
    # 计算总批次数
    for context_batch in context_batches:
        total_batches += (len(context_batch) + batch_size - 1) // batch_size
    
    current_batch = 0
    
    for context_batch in context_batches:
        # 分批次插入，避免API限制
        for i in range(0, len(context_batch), batch_size):
            batch = context_batch[i:i+batch_size]
            batch_text = "\n\n".join(batch)
            current_batch += 1
            
            print(f"\n=== Batch {current_batch}/{total_batches} ===")
            print(f"  Number of contexts: {len(batch)}")
            print(f"  First context preview: {batch[0][:100]}..." if batch else "  No contexts in batch")
            
            current_retries = 0
            while current_retries < max_retries:
                try:
                    log_memory_usage(f"before processing batch {current_batch}")
                    print(f"\nProcessing batch {current_batch}/{total_batches}...")
                    print(f"  Batch size: {len(batch)} contexts")
                    
                    # 使用复形RAG的插入方法
                    print("  Calling rag.ainsert()...")
                    start_time = time.time()
                    try:
                        result = await rag.ainsert(batch_text)
                        print(f"  rag.ainsert() returned: {result}")
                    except Exception as e:
                        print(f"  Error in rag.ainsert(): {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    end_time = time.time()
                    print(f"  rag.ainsert() completed in {end_time - start_time:.2f} seconds")
                    
                    # 清理内存
                    del batch_text
                    gc.collect()
                    log_memory_usage(f"after processing batch {current_batch}")
                    
                    print(f"✅ Batch {current_batch}/{total_batches} inserted successfully")
                    break
                except KeyError as e:
                    current_retries += 1
                    print(f"Insertion failed, retrying ({current_retries}/{max_retries}), error: Attribute error - missing key: {e}")
                    # 指数退避策略
                    wait_time = 2 ** current_retries + random.uniform(0, 1)
                    print(f"  Waiting {wait_time:.2f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                except ValueError as e:
                    current_retries += 1
                    if "dimension" in str(e).lower():
                        print(f"Insertion failed, retrying ({current_retries}/{max_retries}), error: Dimension mismatch - {e}")
                    else:
                        print(f"Insertion failed, retrying ({current_retries}/{max_retries}), error: Value error - {e}")
                    # 指数退避策略
                    wait_time = 2 ** current_retries + random.uniform(0, 1)
                    print(f"  Waiting {wait_time:.2f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    current_retries += 1
                    print(f"Insertion failed, retrying ({current_retries}/{max_retries}), error: {type(e).__name__} - {e}")
                    import traceback
                    traceback.print_exc()
                    # 指数退避策略
                    wait_time = 2 ** current_retries + random.uniform(0, 1)
                    print(f"  Waiting {wait_time:.2f} seconds before retry...")
                    await asyncio.sleep(wait_time)
            if current_retries == max_retries:
                print(f"❌ Insertion failed for batch {current_batch}")
                continue
            
            # 在批次之间添加延迟，避免速率限制
            if current_batch < total_batches:
                print(f"  Waiting 1 second before next batch...")
                await asyncio.sleep(1)


async def build_vector_database(data_name):
    """构建复形RAG向量数据库"""
    # 项目根目录（Step_1.py 位于 reproduce/ 下，向上一级即为项目根）
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # 为每个数据集创建单独的工作目录
    WORKING_DIR = PROJECT_ROOT / "caches" / data_name
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nBuilding Simplicial Complex RAG database for {data_name}")
    print(f"Working directory: {WORKING_DIR}")
    print("Using dataset-specific database storage")
    
    # 初始化HyperRAG，配置为复形RAG模式
    rag = HyperRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    
    # 处理上下文文件
    # 根据 data_name 读取正确命名的上下文文件
    contexts_file = WORKING_DIR / "contexts" / f"{data_name}_unique_contexts.json"
    try:
        log_memory_usage("before processing contexts")
        # 获取上下文批次生成器
        print("  Calling process_contexts()...")
        context_batches = process_contexts(contexts_file, batch_size=500)
        
        # 转换生成器为列表以便检查
        print("  Converting generator to list...")
        context_batches_list = list(context_batches)
        print(f"  Number of context batches: {len(context_batches_list)}")
        
        if context_batches_list:
            print("  Calling insert_text()...")
            await insert_text(rag, context_batches_list, batch_size=5)
            print("\nVector database built successfully with Simplicial Complex RAG!")
        else:
            print("\nNo valid contexts found to insert")
    except Exception as e:
        print(f"\nFailed to build vector database: {e}")
        raise


if __name__ == "__main__":
    data_name = "legal"
    try:
        asyncio.run(build_vector_database(data_name))
    except Exception as e:
        print(f"\nProgram failed: {e}")
        exit(1)
