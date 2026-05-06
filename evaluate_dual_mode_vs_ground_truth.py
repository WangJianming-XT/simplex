import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HYPER_RAG_ROOT = PROJECT_ROOT
HYPERGRAPH_RAG_ROOT = PROJECT_ROOT.parent / "HyperGraphRAG-main"

sys.path.insert(0, str(HYPER_RAG_ROOT))
sys.path.insert(0, str(HYPERGRAPH_RAG_ROOT))

import re
import json
import string
import time
import asyncio
import numpy as np
from collections import Counter
from tqdm import tqdm
from openai import OpenAI

from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMB_API_KEY, EMB_BASE_URL, EMB_MODEL, EMB_DIM

from hyperrag import HyperRAG as HyperRAG_Instance
from hyperrag import QueryParam as HyperRAG_QueryParam
from hyperrag.utils import EmbeddingFunc as HyperRAG_EmbeddingFunc, limit_async_func_call
from hyperrag.llm import openai_embedding as hyperrag_openai_embedding, openai_complete_if_cache as hyperrag_openai_complete_if_cache
from hyperrag.operate import topology_retrieval
from hyperrag.operate._config import DualDimensionConfig
from hyperrag.utils import encode_string_by_tiktoken
# from hyperrag.operate import generate_response  # 暂时注释掉CoT思维链生成方式，改用concise提示词直接生成
from hyperrag.prompt import PROMPTS

from hypergraphrag import HyperGraphRAG as HyperGraphRAG_Instance
from hypergraphrag import QueryParam as HyperGraphRAG_QueryParam
from hypergraphrag.config import get_config as get_hypergraphrag_config


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """同步调用LLM模型生成回答

    通过OpenAI API发送消息并返回模型生成的文本内容。

    Args:
        prompt: 用户输入的提示文本
        system_prompt: 系统提示词，用于设定模型角色和行为规范
        history_messages: 历史对话消息列表
        **kwargs: 透传给OpenAI API的额外参数

    Returns:
        str: 模型生成的文本内容
    """
    openai_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=LLM_MODEL, messages=messages, **kwargs
    )
    return response.choices[0].message.content


@limit_async_func_call(max_size=3, waitting_time=1.0)
async def async_llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """异步调用LLM模型生成回答（带并发限制和缓存）

    使用limit_async_func_call装饰器限制最大并发数为3，
    避免API速率限制导致请求失败。

    Args:
        prompt: 用户输入的提示文本
        system_prompt: 系统提示词
        history_messages: 历史对话消息列表
        **kwargs: 透传给openai_complete_if_cache的额外参数

    Returns:
        str: 模型生成的文本内容
    """
    return await hyperrag_openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        **kwargs,
    )


async def embedding_func(texts: list[str]):
    """异步调用嵌入模型获取文本向量

    Args:
        texts: 需要计算嵌入的文本列表

    Returns:
        嵌入向量列表
    """
    return await hyperrag_openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def get_hyper_rag_instance(data_name):
    """获取HyperRAG实例（Hyper-RAG项目，topology检索模式）

    根据数据集名称创建对应的工作目录和HyperRAG实例，
    配置LLM和嵌入函数等核心组件。

    Args:
        data_name: 数据集名称，用于确定缓存目录路径

    Returns:
        HyperRAG: 配置好的HyperRAG实例
    """
    WORKING_DIR = HYPER_RAG_ROOT / "caches" / data_name
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    rag = HyperRAG_Instance(
        working_dir=WORKING_DIR,
        llm_model_func=async_llm_model_func,
        llm_model_max_async=3,
        embedding_func=HyperRAG_EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        embedding_func_max_async=3,
    )
    return rag


def get_hypergraph_rag_instance(data_name):
    """获取HyperGraphRAG实例（HyperGraphRAG项目，hybrid检索模式）

    根据数据集名称创建对应的工作目录和HyperGraphRAG实例。
    HyperGraphRAG使用自身的config模块管理API配置，
    通过set_env_variables将my_config.py中的配置写入环境变量。

    Args:
        data_name: 数据集名称，用于确定缓存目录路径

    Returns:
        HyperGraphRAG: 配置好的HyperGraphRAG实例
    """
    config = get_hypergraphrag_config()
    config.set_env_variables()

    WORKING_DIR = HYPERGRAPH_RAG_ROOT / "caches" / data_name / "rag"
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    rag = HyperGraphRAG_Instance(working_dir=WORKING_DIR)
    return rag


def normalize_answer(answer: str) -> str:
    """对输入字符串进行标准化处理

    标准化步骤包括：
    1. 将字符串转换为小写
    2. 移除标点符号
    3. 移除冠词 "a", "an", "the"
    4. 规范化空白字符，将多个空格合并为一个

    Args:
        answer: 需要标准化的输入字符串

    Returns:
        str: 标准化后的字符串
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def calculate_f1_score(predicted, ground_truth):
    """计算词级别的F1 Score（使用Counter词频交集）

    对预测答案和真实答案分别进行标准化处理后，
    使用Counter计算词频交集，考虑重复词的贡献，
    然后基于词频交集计算精确率、召回率和F1分数。

    与基线项目evaluate_topology_vs_ground_truth.py保持一致的
    Counter计算方式，避免因set去重导致F1分数偏高。

    Args:
        predicted: 预测答案字符串
        ground_truth: 真实答案字符串

    Returns:
        float: F1 Score值，范围[0, 1]
    """
    pred_tokens = normalize_answer(predicted).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_ge_score(evaluation_result, f1_score):
    """计算Generation Evaluation (G-E)分数

    将七个评估维度的平均分与F1分数（缩放至0-10范围）取均值，
    综合反映生成答案的质量。

    Args:
        evaluation_result: 评估结果字典，包含七个维度的评分
        f1_score: F1分数，范围[0, 1]

    Returns:
        float: G-E分数，范围[0, 10]
    """
    scores = [
        float(evaluation_result.get("Comprehensiveness", {}).get("Score", 0)),
        float(evaluation_result.get("Knowledgeability", {}).get("Score", 0)),
        float(evaluation_result.get("Correctness", {}).get("Score", 0)),
        float(evaluation_result.get("Relevance", {}).get("Score", 0)),
        float(evaluation_result.get("Diversity", {}).get("Score", 0)),
        float(evaluation_result.get("Logical Coherence", {}).get("Score", 0)),
        float(evaluation_result.get("Factuality", {}).get("Score", 0))
    ]

    avg_ge = sum(scores) / len(scores)
    combined_score = (avg_ge + f1_score * 10) / 2

    return combined_score


def load_fin_jsonl(file_path):
    """加载JSONL格式的数据集文件

    支持标准JSON数组格式和JSONL逐行格式，
    自动提取input和answers字段。

    Args:
        file_path: 数据集文件路径

    Returns:
        list: 包含input和answers字段的数据列表
    """
    def extract_answer(answers_field):
        if isinstance(answers_field, list) and len(answers_field) > 0:
            return answers_field[0]
        return answers_field if answers_field else ""

    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()
        if content.startswith("["):
            try:
                items = json.loads(content)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            data.append({
                                "input": item.get("input", ""),
                                "answers": extract_answer(item.get("answers"))
                            })
            except json.JSONDecodeError:
                pass
        else:
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            data.append({
                                "input": item.get("input", ""),
                                "answers": extract_answer(item.get("answers"))
                            })
                    except json.JSONDecodeError:
                        continue
    return data


def parse_eval_json(response_text):
    """从LLM响应中解析评估结果的JSON

    LLM输出的JSON可能存在格式问题（如多余转义、缺少逗号等），
    此函数尝试多种修复策略进行解析。

    Args:
        response_text: LLM返回的原始文本

    Returns:
        dict: 解析后的评估结果字典；解析失败返回空字典
    """
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    if json_start == -1 or json_end == -1:
        return {}

    json_str = response_text[json_start:json_end]
    json_str = re.sub(r'\\(?!(["\\/bfnrt]|u[0-9a-fA-F]{4}))', r'\\\\', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    try:
        repaired = re.sub(r'\}\s*\{', '},{', json_str)
        repaired = re.sub(r'\}\s*"', '},"', repaired)
        repaired = re.sub(r'"\s*\n\s*"', '",\n"', repaired)
        return json.loads(repaired, strict=False)
    except json.JSONDecodeError as e:
        print(f"[警告] JSON解析失败，跳过本次评估解析: {e}")
        print(f"[调试] 原始JSON片段: {json_str[:300]}...")
        return {}


def build_dual_evaluation_prompt(input_text, topology_answer, hybrid_answer, ground_truth):
    """构建LLM联合评估提示词，同时评估topology和hybrid两种模式的答案

    在同一次LLM调用中，将标准答案、topology答案和hybrid答案同时
    提供给LLM，要求其以标准答案为唯一参照，分别对两种模式的答案
    从七个维度进行严格对比评估。

    评估流程要求LLM：
    1. 先提取Ground Truth的所有关键点
    2. 分别检查两个答案对每个关键点的覆盖情况
    3. 严格按评分标准打分，不允许"超出标准答案"获得高分
    4. 两个答案独立评分，互不影响

    评估维度：Comprehensiveness、Knowledgeability、Correctness、
    Relevance、Diversity、Logical Coherence、Factuality

    Args:
        input_text: 原始问题文本
        topology_answer: topology检索模式生成的答案
        hybrid_answer: hybrid检索模式生成的答案
        ground_truth: 标准答案

    Returns:
        str: 完整的联合评估提示词
    """
    return f"""
        ---Role---
        You are a strict evaluator assessing how well two Generated Answers (Answer A and Answer B) align with the Ground Truth Answer. The Ground Truth is the sole reference standard — you must evaluate both answers based solely on consistency with it, NOT on how professional or detailed either answer appears on its own.

        ---Critical Evaluation Principles---
        1. The Ground Truth Answer is the ONLY standard. "Exceeding" the Ground Truth is NOT a reason for higher scores.
        2. Information present in a Generated Answer but NOT in the Ground Truth must NOT increase any score. If such extra information is irrelevant or potentially inaccurate, it should LOWER the score.
        3. Do NOT be influenced by the length, formatting, or apparent professionalism of either Generated Answer. A concise answer that matches the Ground Truth is better than a lengthy one that diverges from it.
        4. Focus on whether the key points in the Ground Truth are correctly captured in each Generated Answer. Missing key points from the Ground Truth must result in significant score reductions.
        5. Evaluate each answer independently against the Ground Truth. Do not let one answer's quality influence the other's score.
        6. After independent scoring, provide a direct comparison: which answer better matches the Ground Truth overall, and why.

        ---Step-by-Step Evaluation Process---
        Before scoring, you MUST first:
        1. List ALL key points in the Ground Truth Answer
        2. For each key point, check whether Answer A covers it (fully / partially / missing)
        3. For each key point, check whether Answer B covers it (fully / partially / missing)
        4. List any claims in each answer that are NOT supported by the Ground Truth
        5. Directly compare: for each key point, which answer captures it more accurately?
        Then use this analysis to score each dimension for each answer.

        ---Question---
        {input_text}

        ---Ground Truth Answer---
        {ground_truth}

        ---Answer A (Topology Retrieval) to be Evaluated---
        {topology_answer}

        ---Answer B (Hybrid Retrieval) to be Evaluated---
        {hybrid_answer}

        ---Evaluation Goal---
        Evaluate how well each Generated Answer matches the Ground Truth Answer on a 0–10 integer scale. The Ground Truth is the ceiling — a perfect score means the Generated Answer faithfully captures all key points in the Ground Truth without adding unsupported information.

        ---Scoring Rubrics---
        - Comprehensiveness: Whether the Generated Answer covers ALL key points stated in the Ground Truth
          Scoring Guide (0–10):
          - 10: Every key point in the Ground Truth is fully covered; no omissions.
          - 8–9: Most key points from the Ground Truth are covered; only minor omissions.
          - 6–7: Some key points from the Ground Truth are covered, but notable points are missing.
          - 4–5: Only a few key points from the Ground Truth are mentioned; significant omissions.
          - 1–3: Most key points from the Ground Truth are missing.
          - 0: No key points from the Ground Truth are addressed at all.

        - Knowledgeability: Whether the domain knowledge in the Generated Answer is consistent with the Ground Truth
          Scoring Guide (0–10):
          - 10: All domain knowledge expressed is fully consistent with the Ground Truth; no unsupported claims.
          - 8–9: Most knowledge is consistent with the Ground Truth; only minor unsupported additions.
          - 6–7: Some knowledge aligns with the Ground Truth, but there are notable unsupported or divergent claims.
          - 4–5: Limited alignment with the Ground Truth; several unsupported or incorrect knowledge claims.
          - 1–3: Most knowledge claims diverge from or contradict the Ground Truth.
          - 0: No meaningful alignment with the Ground Truth's knowledge.

        - Correctness: Whether the factual content of the Generated Answer is correct when compared against the Ground Truth
          Scoring Guide (0–10):
          - 10: All factual claims in the Generated Answer are consistent with the Ground Truth; no contradictions.
          - 8–9: Mostly consistent with the Ground Truth; minor factual discrepancies.
          - 6–7: Partially consistent; some key facts contradict or diverge from the Ground Truth.
          - 4–5: Significant factual deviations from the Ground Truth.
          - 1–3: Most facts contradict the Ground Truth.
          - 0: Entirely contradicts the Ground Truth.

        - Relevance: Whether the Generated Answer focuses on the same key points as the Ground Truth, without irrelevant digressions
          Scoring Guide (0–10):
          - 10: The Generated Answer focuses precisely on the same key points as the Ground Truth; no irrelevant content.
          - 8–9: Mostly focused on the Ground Truth's key points; minor irrelevant additions.
          - 6–7: Addresses some Ground Truth points but includes significant irrelevant or tangential content.
          - 4–5: Much of the content is irrelevant to the Ground Truth's key points.
          - 1–3: Barely addresses the Ground Truth's key points; mostly off-topic.
          - 0: Entirely irrelevant to the Ground Truth.

        - Diversity: Whether the Generated Answer covers the same range of perspectives as the Ground Truth
          Scoring Guide (0–10):
          - 10: Covers exactly the same range of perspectives as the Ground Truth; no more, no less.
          - 8–9: Covers most of the Ground Truth's perspectives; minor differences.
          - 6–7: Covers some perspectives from the Ground Truth but misses notable angles.
          - 4–5: Limited perspective coverage compared to the Ground Truth.
          - 1–3: Very narrow perspective; misses most of the Ground Truth's angles.
          - 0: No perspective overlap with the Ground Truth.

        - Logical Coherence: Whether the Generated Answer presents information in a logically consistent way that aligns with the Ground Truth's reasoning
          Scoring Guide (0–10):
          - 10: Perfectly coherent and logically consistent with the Ground Truth's reasoning.
          - 8–9: Mostly coherent and consistent with the Ground Truth; minor logical inconsistencies.
          - 6–7: Some logical structure but inconsistencies with the Ground Truth's reasoning.
          - 4–5: Often disorganized or logically inconsistent with the Ground Truth.
          - 1–3: Poorly structured; logic contradicts the Ground Truth.
          - 0: Entirely illogical or contradicts the Ground Truth's reasoning.

        - Factuality: Whether every factual claim in the Generated Answer can be verified against the Ground Truth
          Scoring Guide (0–10):
          - 10: Every factual claim in the Generated Answer is directly supported by the Ground Truth; no unsupported claims.
          - 8–9: Most factual claims are supported by the Ground Truth; only minor unsupported claims.
          - 6–7: Some factual claims are supported by the Ground Truth; several claims lack support or are unverifiable.
          - 4–5: Many factual claims are not supported by the Ground Truth or contradict it.
          - 1–3: Most factual claims are unsupported by or contradict the Ground Truth.
          - 0: All factual claims are unsupported or contradict the Ground Truth.

        Please evaluate BOTH generated answers by strictly comparing each with the ground truth answer based on the seven criteria above. For each criterion, provide a score from 0 to 10 and a brief explanation that explicitly references which specific points from the Ground Truth are covered, missing, or contradicted.

        Output your evaluation in the following JSON format:

        {{
            "Answer_A_Topology": {{
                "Comprehensiveness": {{
                    "Explanation": "List which Ground Truth key points are covered and which are missing",
                    "Score": "A value range 0 to 10"
                }},
                "Knowledgeability": {{
                    "Explanation": "Assess whether knowledge claims are consistent with Ground Truth",
                    "Score": "A value range 0 to 10"
                }},
                "Correctness": {{
                    "Explanation": "Identify factual claims that match or contradict the Ground Truth",
                    "Score": "A value range 0 to 10"
                }},
                "Relevance": {{
                    "Explanation": "Assess whether the answer focuses on Ground Truth key points or digresses",
                    "Score": "A value range 0 to 10"
                }},
                "Diversity": {{
                    "Explanation": "Compare the range of perspectives with the Ground Truth",
                    "Score": "A value range 0 to 10"
                }},
                "Logical Coherence": {{
                    "Explanation": "Assess logical consistency with the Ground Truth's reasoning",
                    "Score": "A value range 0 to 10"
                }},
                "Factuality": {{
                    "Explanation": "List factual claims and whether each is supported by the Ground Truth",
                    "Score": "A value range 0 to 10"
                }}
            }},
            "Answer_B_Hybrid": {{
                "Comprehensiveness": {{
                    "Explanation": "List which Ground Truth key points are covered and which are missing",
                    "Score": "A value range 0 to 10"
                }},
                "Knowledgeability": {{
                    "Explanation": "Assess whether knowledge claims are consistent with Ground Truth",
                    "Score": "A value range 0 to 10"
                }},
                "Correctness": {{
                    "Explanation": "Identify factual claims that match or contradict the Ground Truth",
                    "Score": "A value range 0 to 10"
                }},
                "Relevance": {{
                    "Explanation": "Assess whether the answer focuses on Ground Truth key points or digresses",
                    "Score": "A value range 0 to 10"
                }},
                "Diversity": {{
                    "Explanation": "Compare the range of perspectives with the Ground Truth",
                    "Score": "A value range 0 to 10"
                }},
                "Logical Coherence": {{
                    "Explanation": "Assess logical consistency with the Ground Truth's reasoning",
                    "Score": "A value range 0 to 10"
                }},
                "Factuality": {{
                    "Explanation": "List factual claims and whether each is supported by the Ground Truth",
                    "Score": "A value range 0 to 10"
                }}
            }},
            "Direct_Comparison": {{
                "Better_Answer": "A or B",
                "Reason": "Explain which answer better matches the Ground Truth overall and why",
                "Key_Differences": "List the most important differences between the two answers in terms of Ground Truth coverage"
            }}
        }}

    """


async def generate_topology_answer(hyper_rag, input_text):
    """使用Hyper-RAG项目的topology检索模式生成答案

    直接调用topology_retrieval进行检索，再使用concise提示词
    （topology_response_system_prompt_concise）基于结构化上下文
    直接生成回答，不使用思维链（CoT）推理。

    改进策略（对齐Hybrid基线）：
    1. Sources token预算独立控制（8000 token），对齐Hybrid的4000-8000 token限制
    2. 结构化信息（Entities/Relationships）只保留与Sources直接相关的条目，
       且受独立token预算（4000 token）和数量上限限制，减少噪声
    3. VDB直取chunk享有绝对优先级，在token截断时不被截断
    4. 简化Prompt，移除prompt_instructions冗余信息，减少LLM认知负担

    Args:
        hyper_rag: HyperRAG实例
        input_text: 用户输入的问题文本

    Returns:
        str: 模型生成的答案文本；生成失败返回None

    Raises:
        Exception: 检索或生成过程中发生错误时抛出
    """
    topology_config = {
        "enable_llm_keyword_extraction": True,
        "max_topology_chunks": 20,
        "diffusion_steps": 2,
        "max_simplices": 60,
        "min_coverage_ratio": 0.5,
        "max_context_tokens": 60000,
        "embedding_func": hyper_rag.embedding_func,
        "llm_model_func": hyper_rag.llm_model_func,
        "chunks_vdb": hyper_rag.chunks_vdb,
    }

    retrieval_result = await topology_retrieval(
        input_text,
        hyper_rag.simplex_storage,
        hyper_rag.entities_vdb,
        hyper_rag.relationships_vdb,
        hyper_rag.text_chunks,
        topology_config,
    )

    structured_entities = retrieval_result.get("structured_entities", [])
    structured_simplices = retrieval_result.get("structured_simplices", [])
    related_chunks = retrieval_result.get("related_chunks", [])

    # ===== 改进1：Sources段独立token预算，对齐Hybrid的"少而精"策略 =====
    sources_token_budget = DualDimensionConfig.SOURCES_TOKEN_BUDGET
    structure_token_budget = DualDimensionConfig.STRUCTURE_TOKEN_BUDGET

    # Sources段：按token预算截断，保留最相关的chunk
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

    # ===== 改进2：结构化信息只保留与Sources直接相关的条目 =====
    # 从Sources中提取出现的实体名，只保留这些实体及其直接关联的simplices
    sources_text = " ".join(related_chunks[:len(sources_csv_lines) - 1])
    sources_entity_names = set()
    for ent in structured_entities:
        if ent['name'].upper() in sources_text.upper():
            sources_entity_names.add(ent['name'])

    # 种子实体也保留
    for ent in structured_entities:
        if ent.get('is_seed'):
            sources_entity_names.add(ent['name'])

    # 过滤Entities：只保留出现在Sources中或是种子的实体
    filtered_entities = [
        ent for ent in structured_entities
        if ent['name'] in sources_entity_names
    ]
    # 数量上限
    filtered_entities = filtered_entities[:DualDimensionConfig.MAX_STRUCTURE_ENTITY_COUNT]

    # 过滤Simplices：只保留其entities与Sources相关的simplices
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
        entity_csv_lines = ["name,type,description"]
        entity_tokens = len(encode_string_by_tiktoken("name,type,description\n"))
        for ent in filtered_entities:
            ent_type = ent.get('type', 'Entity')
            desc = (ent.get('description', '') or '').replace('"', '""').replace('\n', ' ')
            name = ent['name'].replace('"', '""')
            line = f'"{name}","{ent_type}","{desc}"'
            line_tokens = len(encode_string_by_tiktoken(line))
            if entity_tokens + line_tokens > structure_token_budget // 2:
                break
            entity_csv_lines.append(line)
            entity_tokens += line_tokens
        entity_section = "-----Entities-----\n```csv\n" + "\n".join(entity_csv_lines) + "\n```"

    # 构建Relationships CSV
    relationship_section = ""
    if filtered_simplices:
        simplex_csv_lines = ["id,original_text,description,related_entities"]
        simplex_tokens = len(encode_string_by_tiktoken("id,original_text,description,related_entities\n"))
        for i, simp in enumerate(filtered_simplices):
            orig_text = (simp.get('original_text', '') or '').replace('"', '""').replace('\n', ' ')
            desc = (simp.get('description', '') or '').replace('"', '""').replace('\n', ' ')
            entities = simp.get('entities', [])
            ent_str = "|".join(str(e) for e in entities).replace('"', '""')
            line = f'"{i}","{orig_text}","{desc}","{ent_str}"'
            line_tokens = len(encode_string_by_tiktoken(line))
            if simplex_tokens + line_tokens > structure_token_budget // 2:
                break
            simplex_csv_lines.append(line)
            simplex_tokens += line_tokens
        relationship_section = "-----Relationships-----\n```csv\n" + "\n".join(simplex_csv_lines) + "\n```"

    # 组装上下文：Sources放最前面（LLM对前面内容注意力更高）
    context_parts = [sources_section]
    if entity_section:
        context_parts.append(entity_section)
    if relationship_section:
        context_parts.append(relationship_section)

    structured_context = "\n\n".join(context_parts)

    # ===== 改进4：简化Prompt，不注入prompt_instructions =====
    # Hybrid基线没有prompt_instructions，只有静态RULES
    # 移除冗余的匹配实体和关系列表，减少LLM认知负担
    sys_prompt = PROMPTS["topology_response_system_prompt_concise"].format(
        prompt_instructions="",
        context=structured_context,
    )

    response = await async_llm_model_func(
        input_text,
        system_prompt=sys_prompt,
    )

    # 清理响应中可能残留的系统提示词或用户查询
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(input_text, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def generate_hybrid_answer(hypergraph_rag, input_text):
    """使用HyperGraphRAG项目的hybrid检索模式生成答案

    通过HyperGraphRAG的aquery接口，使用hybrid检索模式
    （基于知识图谱的混合检索，结合local和global模式）对问题进行检索和生成。

    Args:
        hypergraph_rag: HyperGraphRAG实例
        input_text: 用户输入的问题文本

    Returns:
        str: 模型生成的答案文本；生成失败返回None
    """
    param = HyperGraphRAG_QueryParam(mode="hybrid")
    answer = await hypergraph_rag.aquery(input_text, param)
    return answer


async def evaluate_dual_answers(input_text, topology_answer, hybrid_answer, ground_truth):
    """对topology和hybrid两种模式的答案进行联合LLM评估

    在同一次LLM调用中，将标准答案、topology答案和hybrid答案同时
    提供给LLM进行评估，确保两种模式的答案在相同上下文中被对比评价，
    提高评估的一致性和公平性。

    LLM先提取Ground Truth的关键点，再分别比对两个答案，
    最后独立评分，互不影响。

    当标准答案为空时，两种模式的所有维度评分均设为0。
    当某种模式的答案为空时，该模式的所有维度评分均设为0。

    Args:
        input_text: 原始问题文本
        topology_answer: topology检索模式生成的答案
        hybrid_answer: hybrid检索模式生成的答案
        ground_truth: 标准答案

    Returns:
        dict: 联合评估结果字典，包含以下键：
            - evaluation: LLM原始评估响应文本
            - eval_data_topology: topology模式的七维度评分字典
            - eval_data_hybrid: hybrid模式的七维度评分字典
            - F1_Score_Topology: topology答案的词级别F1分数
            - F1_Score_Hybrid: hybrid答案的词级别F1分数
            - G-E_Score_Topology: topology答案的综合评估分数
            - G-E_Score_Hybrid: hybrid答案的综合评估分数
    """
    zero_eval = {
        "Comprehensiveness": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"},
        "Knowledgeability": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"},
        "Correctness": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"},
        "Relevance": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"},
        "Diversity": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"},
        "Logical Coherence": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"},
        "Factuality": {"Explanation": "无标准答案或无生成答案，无法评估", "Score": "0"}
    }

    if not ground_truth or (not topology_answer and not hybrid_answer):
        f1_topo = calculate_f1_score(topology_answer or "", ground_truth or "") if ground_truth and topology_answer else 0.0
        f1_hyb = calculate_f1_score(hybrid_answer or "", ground_truth or "") if ground_truth and hybrid_answer else 0.0
        ge_topo = calculate_ge_score(zero_eval, f1_topo)
        ge_hyb = calculate_ge_score(zero_eval, f1_hyb)
        return {
            "evaluation": "",
            "eval_data_topology": zero_eval,
            "eval_data_hybrid": zero_eval,
            "F1_Score_Topology": f1_topo,
            "F1_Score_Hybrid": f1_hyb,
            "G-E_Score_Topology": ge_topo,
            "G-E_Score_Hybrid": ge_hyb,
        }

    prompt = build_dual_evaluation_prompt(input_text, topology_answer or "（无答案）", hybrid_answer or "（无答案）", ground_truth)
    sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating answers to questions. You will evaluate TWO answers generated by different retrieval methods by comparing each with the ground truth answer. Evaluate each answer independently and fairly.
        """

    response = llm_model_func(prompt, sys_prompt, temperature=0)
    eval_data = parse_eval_json(response)

    eval_data_topology = eval_data.get("Answer_A_Topology", {})
    eval_data_hybrid = eval_data.get("Answer_B_Hybrid", {})
    direct_comparison = eval_data.get("Direct_Comparison", {})

    if not eval_data_topology:
        eval_data_topology = zero_eval
    if not eval_data_hybrid:
        eval_data_hybrid = zero_eval

    f1_topo = calculate_f1_score(topology_answer or "", ground_truth)
    f1_hyb = calculate_f1_score(hybrid_answer or "", ground_truth)
    ge_topo = calculate_ge_score(eval_data_topology, f1_topo)
    ge_hyb = calculate_ge_score(eval_data_hybrid, f1_hyb)

    return {
        "evaluation": response,
        "eval_data_topology": eval_data_topology,
        "eval_data_hybrid": eval_data_hybrid,
        "direct_comparison": direct_comparison,
        "F1_Score_Topology": f1_topo,
        "F1_Score_Hybrid": f1_hyb,
        "G-E_Score_Topology": ge_topo,
        "G-E_Score_Hybrid": ge_hyb,
    }


MODE_A = "topology"
MODE_B = "hybrid"


async def generate_and_evaluate_dual_mode(fin_data, data_name):
    """使用两种检索模式生成答案并联合评估

    对每条数据分别使用两种检索模式生成答案：
    - topology模式：来自Hyper-RAG项目，基于单纯复形的双维度并行检索
    - hybrid模式：来自HyperGraphRAG项目，基于知识图谱的混合检索

    然后在同一次LLM调用中，将标准答案、topology答案和hybrid答案
    同时提供给LLM进行联合评估，确保两种模式的答案在相同上下文中
    被对比评价，提高评估的一致性和公平性。

    流程：
    1. 初始化两个项目的RAG实例
    2. 使用topology模式检索并生成答案
    3. 使用hybrid模式检索并生成答案
    4. 将标准答案、topology答案和hybrid答案同时提交给LLM进行联合评估
    5. 计算各自的F1 Score和G-E Score
    6. 汇总对比结果

    采用全异步实现，避免混用asyncio.run()和run_until_complete()
    导致事件循环反复创建/销毁的问题。

    Args:
        fin_data: 数据集列表，每项包含input和answers字段
        data_name: 数据集名称，用于确定缓存目录

    Returns:
        tuple: (匹配的数据列表, topology评估结果列表, hybrid评估结果列表)
    """
    hyper_rag = get_hyper_rag_instance(data_name)
    hypergraph_rag = get_hypergraph_rag_instance(data_name)

    matched_data = []
    eval_results_topology = []
    eval_results_hybrid = []

    for item in tqdm(fin_data, desc=f"Dual-mode evaluation ({MODE_A} vs {MODE_B})", total=len(fin_data)):
        time.sleep(15)
        input_text = item["input"]
        answers = item.get("answers", "")

        answer_topology = None
        answer_hybrid = None

        try:
            answer_topology = await generate_topology_answer(hyper_rag, input_text)
            if not answer_topology:
                print(f"\n[{MODE_A}] 空答案，跳过问题: {input_text[:100]}...")
            else:
                print(f"\n[{MODE_A}] 已生成答案: {input_text[:100]}...")
        except Exception as e:
            print(f"\n[{MODE_A}] 生成答案失败: {input_text[:100]}...")
            print(f"错误: {e}")

        try:
            answer_hybrid = await generate_hybrid_answer(hypergraph_rag, input_text)
            if not answer_hybrid:
                print(f"\n[{MODE_B}] 空答案，跳过问题: {input_text[:100]}...")
            else:
                print(f"\n[{MODE_B}] 已生成答案: {input_text[:100]}...")
        except Exception as e:
            print(f"\n[{MODE_B}] 生成答案失败: {input_text[:100]}...")
            print(f"错误: {e}")

        if answer_topology is None and answer_hybrid is None:
            continue

        current_data = {
            "input": input_text,
            "answers": answers,
            f"{MODE_A}_answer": answer_topology or "",
            f"{MODE_B}_answer": answer_hybrid or "",
        }
        matched_data.append(current_data)

        dual_eval = await evaluate_dual_answers(input_text, answer_topology, answer_hybrid, answers)

        if answer_topology is not None:
            eval_results_topology.append({
                "query": input_text,
                "mode": MODE_A,
                "model_answer": answer_topology,
                "ground_truth": answers,
                "eval_data": dual_eval["eval_data_topology"],
                "F1_Score": dual_eval["F1_Score_Topology"],
                "G-E_Score": dual_eval["G-E_Score_Topology"],
            })
            print(f"\n{'='*60}")
            print(f"[{MODE_A}] 问题: {input_text}")
            print(f"[标准答案] {answers}")
            print(f"[生成答案] {answer_topology}")
            print(f"[F1 Score] {dual_eval['F1_Score_Topology']:.4f}")
            print(f"[G-E Score] {dual_eval['G-E_Score_Topology']:.4f}")
            for dim_name in ["Comprehensiveness", "Knowledgeability", "Correctness", "Relevance", "Diversity", "Logical Coherence", "Factuality"]:
                dim_score = dual_eval["eval_data_topology"].get(dim_name, {}).get("Score", "N/A")
                print(f"  [{dim_name}] {dim_score}")
            print(f"{'='*60}")

        if answer_hybrid is not None:
            eval_results_hybrid.append({
                "query": input_text,
                "mode": MODE_B,
                "model_answer": answer_hybrid,
                "ground_truth": answers,
                "eval_data": dual_eval["eval_data_hybrid"],
                "F1_Score": dual_eval["F1_Score_Hybrid"],
                "G-E_Score": dual_eval["G-E_Score_Hybrid"],
            })
            print(f"\n{'='*60}")
            print(f"[{MODE_B}] 问题: {input_text}")
            print(f"[标准答案] {answers}")
            print(f"[生成答案] {answer_hybrid}")
            print(f"[F1 Score] {dual_eval['F1_Score_Hybrid']:.4f}")
            print(f"[G-E Score] {dual_eval['G-E_Score_Hybrid']:.4f}")
            for dim_name in ["Comprehensiveness", "Knowledgeability", "Correctness", "Relevance", "Diversity", "Logical Coherence", "Factuality"]:
                dim_score = dual_eval["eval_data_hybrid"].get(dim_name, {}).get("Score", "N/A")
                print(f"  [{dim_name}] {dim_score}")

        direct_comparison = dual_eval.get("direct_comparison", {})
        if direct_comparison:
            print(f"\n  [对比] 更优答案: {direct_comparison.get('Better_Answer', 'N/A')}")
            print(f"  [对比] 原因: {direct_comparison.get('Reason', 'N/A')[:200]}")
            print(f"{'='*60}")

    print(f"\n{MODE_A}: {len(eval_results_topology)} 条评估完成")
    print(f"{MODE_B}: {len(eval_results_hybrid)} 条评估完成")

    return matched_data, eval_results_topology, eval_results_hybrid


def fetch_scoring_results(responses, mode_name=""):
    """汇总并打印评估结果的各项指标平均分

    计算七个评估维度的平均分、F1 Score平均值、G-E Score平均值，
    以及所有维度的综合平均分。

    Args:
        responses: 评估结果列表
        mode_name: 模式名称，用于打印标识

    Returns:
        numpy.ndarray: 包含所有指标平均分的数组，
            顺序为7个维度 + Averaged + F1 Score + G-E Score
    """
    metric_name_list = [
        "Comprehensiveness",
        "Knowledgeability",
        "Correctness",
        "Relevance",
        "Diversity",
        "Logical Coherence",
        "Factuality",
        "Averaged",
        "F1 Score",
        "G-E Score"
    ]
    total_scores = [0] * 7
    total_f1 = 0.0
    total_ge = 0.0
    valid_results = 0

    for result in responses:
        try:
            eval_data = result.get('eval_data', {})
            total_scores[0] += float(eval_data.get("Comprehensiveness", {}).get("Score", 0) or 0)
            total_scores[1] += float(eval_data.get("Knowledgeability", {}).get("Score", 0) or 0)
            total_scores[2] += float(eval_data.get("Correctness", {}).get("Score", 0) or 0)
            total_scores[3] += float(eval_data.get("Relevance", {}).get("Score", 0) or 0)
            total_scores[4] += float(eval_data.get("Diversity", {}).get("Score", 0) or 0)
            total_scores[5] += float(eval_data.get("Logical Coherence", {}).get("Score", 0) or 0)
            total_scores[6] += float(eval_data.get("Factuality", {}).get("Score", 0) or 0)
            total_f1 += float(result.get("F1_Score", 0) or 0)
            total_ge += float(result.get("G-E_Score", 0) or 0)
            valid_results += 1
        except (ValueError, TypeError):
            continue

    if valid_results > 0:
        total_scores = np.array(total_scores) / valid_results
        average_score = np.mean(total_scores)
        avg_f1 = total_f1 / valid_results
        avg_ge = total_ge / valid_results
    else:
        total_scores = np.zeros(7)
        average_score = 0.0
        avg_f1 = 0.0
        avg_ge = 0.0

    all_scores = np.concatenate([total_scores, [average_score, avg_f1, avg_ge]])

    prefix = f"[{mode_name}] " if mode_name else ""
    for metric_name, score in zip(metric_name_list, all_scores):
        if metric_name in ["F1 Score", "G-E Score"]:
            print(f"{prefix}{metric_name:20}: {score:.4f}")
        else:
            print(f"{prefix}{metric_name:20}: {score:.2f}")
    return all_scores


def save_evaluation_to_csv(evaluation_results, output_file, mode_name=""):
    """将评估结果保存为CSV格式

    每条记录包含问题、生成答案、标准答案、七个维度评分、
    F1 Score、G-E Score和综合平均分。

    Args:
        evaluation_results: 评估结果列表
        output_file: 输出CSV文件路径
        mode_name: 模式名称，用于标识列名
    """
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Query', 'Mode', 'Final Answer', 'Ground Truth',
                      'Comprehensiveness Score', 'Knowledgeability Score',
                      'Correctness Score', 'Relevance Score',
                      'Diversity Score', 'Logical Coherence Score',
                      'Factuality Score', 'F1 Score', 'G-E Score', 'Average Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in evaluation_results:
            query = result['query']
            mode = result.get('mode', mode_name)
            final_answer = result['model_answer']
            ground_truth = result['ground_truth']
            eval_data = result.get('eval_data', {})
            f1_score = result.get('F1_Score', 0.0)
            ge_score = result.get('G-E_Score', 0.0)

            comp_score = float(eval_data.get("Comprehensiveness", {}).get("Score", 0) or 0)
            know_score = float(eval_data.get("Knowledgeability", {}).get("Score", 0) or 0)
            corr_score = float(eval_data.get("Correctness", {}).get("Score", 0) or 0)
            rele_score = float(eval_data.get("Relevance", {}).get("Score", 0) or 0)
            div_score = float(eval_data.get("Diversity", {}).get("Score", 0) or 0)
            log_score = float(eval_data.get("Logical Coherence", {}).get("Score", 0) or 0)
            fact_score = float(eval_data.get("Factuality", {}).get("Score", 0) or 0)

            avg_score = (comp_score + know_score + corr_score + rele_score + div_score + log_score + fact_score) / 7

            writer.writerow({
                'Query': query,
                'Mode': mode,
                'Final Answer': final_answer,
                'Ground Truth': ground_truth,
                'Comprehensiveness Score': comp_score,
                'Knowledgeability Score': know_score,
                'Correctness Score': corr_score,
                'Relevance Score': rele_score,
                'Diversity Score': div_score,
                'Logical Coherence Score': log_score,
                'Factuality Score': fact_score,
                'F1 Score': f1_score,
                'G-E Score': ge_score,
                'Average Score': avg_score
            })


def save_comparison_to_csv(matched_data, eval_results_a, eval_results_b, output_file):
    """将两种模式的对比评估结果保存为CSV格式

    每条记录包含问题、标准答案、两种模式的生成答案、
    以及两种模式各自的七维度评分、F1 Score和G-E Score。

    Args:
        matched_data: 匹配的数据列表
        eval_results_a: topology模式的评估结果列表
        eval_results_b: hybrid模式的评估结果列表
        output_file: 输出CSV文件路径
    """
    import csv

    eval_a_map = {r["query"]: r for r in eval_results_a}
    eval_b_map = {r["query"]: r for r in eval_results_b}

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'Query', 'Ground Truth',
            f'{MODE_A} Answer', f'{MODE_A} Comprehensiveness', f'{MODE_A} Knowledgeability',
            f'{MODE_A} Correctness', f'{MODE_A} Relevance', f'{MODE_A} Diversity',
            f'{MODE_A} Logical Coherence', f'{MODE_A} Factuality',
            f'{MODE_A} F1 Score', f'{MODE_A} G-E Score', f'{MODE_A} Average',
            f'{MODE_B} Answer', f'{MODE_B} Comprehensiveness', f'{MODE_B} Knowledgeability',
            f'{MODE_B} Correctness', f'{MODE_B} Relevance', f'{MODE_B} Diversity',
            f'{MODE_B} Logical Coherence', f'{MODE_B} Factuality',
            f'{MODE_B} F1 Score', f'{MODE_B} G-E Score', f'{MODE_B} Average',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in matched_data:
            query = item["input"]
            ground_truth = item["answers"]
            row = {'Query': query, 'Ground Truth': ground_truth}

            for mode_name, eval_map, answer_key in [
                (MODE_A, eval_a_map, f"{MODE_A}_answer"),
                (MODE_B, eval_b_map, f"{MODE_B}_answer")
            ]:
                row[f'{mode_name} Answer'] = item.get(answer_key, "")
                eval_r = eval_map.get(query, {})
                eval_data = eval_r.get('eval_data', {})
                f1 = eval_r.get('F1_Score', 0.0)
                ge = eval_r.get('G-E_Score', 0.0)

                scores = []
                for dim in ["Comprehensiveness", "Knowledgeability", "Correctness",
                            "Relevance", "Diversity", "Logical Coherence", "Factuality"]:
                    s = float(eval_data.get(dim, {}).get("Score", 0) or 0)
                    row[f'{mode_name} {dim}'] = s
                    scores.append(s)

                row[f'{mode_name} F1 Score'] = f1
                row[f'{mode_name} G-E Score'] = ge
                row[f'{mode_name} Average'] = sum(scores) / len(scores) if scores else 0.0

            writer.writerow(row)


if __name__ == "__main__":
    data_name = "legal"
    limit = 1

    jsonl_path = HYPER_RAG_ROOT / "datasets" / data_name / f"{data_name}.jsonl"
    fin_data = load_fin_jsonl(jsonl_path)
    fin_data = fin_data[:limit]
    print(f"使用 {data_name}.jsonl 数据集的前 {len(fin_data)} 条数据")
    print(f"对比模式: {MODE_A} (Hyper-RAG) vs {MODE_B} (HyperGraphRAG)")

    matched_data, eval_results_topology, eval_results_hybrid = asyncio.run(
        generate_and_evaluate_dual_mode(fin_data, data_name)
    )
    print(f"共处理 {len(matched_data)} / {len(fin_data)} 个问题")

    WORKING_DIR = HYPER_RAG_ROOT / "caches" / data_name
    OUT_DIR = WORKING_DIR / "evaluation"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_json_a = OUT_DIR / f"{MODE_A}_vs_ground_truth_{data_name}_{len(fin_data)}.json"
    with open(output_json_a, "w", encoding="utf-8") as f:
        json.dump(eval_results_topology, f, indent=4, ensure_ascii=False)
    print(f"[OK] {MODE_A} 评估结果已写入 {output_json_a}")

    output_json_b = OUT_DIR / f"{MODE_B}_vs_ground_truth_{data_name}_{len(fin_data)}.json"
    with open(output_json_b, "w", encoding="utf-8") as f:
        json.dump(eval_results_hybrid, f, indent=4, ensure_ascii=False)
    print(f"[OK] {MODE_B} 评估结果已写入 {output_json_b}")

    csv_output_a = OUT_DIR / f"{MODE_A}_vs_ground_truth_{data_name}_{len(fin_data)}.csv"
    save_evaluation_to_csv(eval_results_topology, csv_output_a, MODE_A)
    print(f"[OK] {MODE_A} 评估CSV已写入 {csv_output_a}")

    csv_output_b = OUT_DIR / f"{MODE_B}_vs_ground_truth_{data_name}_{len(fin_data)}.csv"
    save_evaluation_to_csv(eval_results_hybrid, csv_output_b, MODE_B)
    print(f"[OK] {MODE_B} 评估CSV已写入 {csv_output_b}")

    comparison_csv = OUT_DIR / f"comparison_{MODE_A}_vs_{MODE_B}_{data_name}_{len(fin_data)}.csv"
    save_comparison_to_csv(matched_data, eval_results_topology, eval_results_hybrid, comparison_csv)
    print(f"[OK] 对比CSV已写入 {comparison_csv}")

    print(f"\n[RESULT] {MODE_A} (Hyper-RAG) 评估结果:")
    fetch_scoring_results(eval_results_topology, MODE_A)
    print("=" * 80)

    print(f"\n[RESULT] {MODE_B} (HyperGraphRAG) 评估结果:")
    fetch_scoring_results(eval_results_hybrid, MODE_B)
    print("=" * 80)
