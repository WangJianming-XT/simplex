import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent))

import asyncio
import numpy as np
from hyperrag import HyperRAG, QueryParam
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from hyperrag.llm import openai_embedding, openai_complete_if_cache

# 国内中转服务的基础 URL
LLM_BASE_URL = "http://jeniya.top/v1"
# 国内中转服务的 API 密钥
LLM_API_KEY = "sk-mdKIKK4vvRoLmS0kvzAAqY6AHK6GeXdt6BRZ0rXmpvclWG3L"
# 模型名称
LLM_MODEL = "gpt-4o-mini"

# 嵌入服务的基础 URL（通常与 LLM 相同）
EMB_BASE_URL = "http://jeniya.top/v1"
# 嵌入服务的 API 密钥（通常与 LLM 相同）
EMB_API_KEY = "sk-mdKIKK4vvRoLmS0kvzAAqY6AHK6GeXdt6BRZ0rXmpvclWG3L"
# 嵌入模型名称
EMB_MODEL = "text-embedding-3-small"
# 嵌入维度
EMB_DIM = 1536

# RAG缓存路径
RAG_CACHE_DIR = Path("caches/fin/rag")

# 数据文件路径
DATA_FILE = Path("datasets/fin/fin.jsonl")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """LLM模型函数"""
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    """嵌入函数"""
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )

async def evaluate_answer_with_llm(question, generated_answer, standard_answer):
    """使用LLM评估生成的回答与标准回答的匹配度"""
    sys_prompt = """
    ---Role---
    You are an expert evaluator of AI-generated answers. You will evaluate how well a generated answer matches a standard answer for a given question.
    
    ---Evaluation Process---
    1. First, analyze the question carefully
    2. Then, compare the generated answer with the standard answer
    3. Evaluate based on the following criteria:
       - Accuracy: How well does the generated answer match the facts in the standard answer?
       - Completeness: How much of the information from the standard answer is included in the generated answer?
       - Relevance: How relevant is the generated answer to the question?
       - Clarity: How clear and well-structured is the generated answer?
    4. Provide a detailed evaluation and a final score (0-100)
    """
    
    prompt = f"""
    Please evaluate how well the generated answer matches the standard answer for the following question:
    
    Question: {question}
    
    Generated Answer:
    {generated_answer}
    
    Standard Answer:
    {standard_answer}
    
    Evaluate the generated answer based on five criteria: **Comprehensiveness**, **Accuracy**, **Relevance**, **Logical**, and **Readability**.

    - **Comprehensiveness** - 
    Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions compared to the standard answer.
    Level   | score range | description
    Level 1 | 0-20   | The answer is extremely one-sided, leaving out key parts or important aspects of the question.
    Level 2 | 20-40  | The answer has some content, but it misses many important aspects of the question and is not comprehensive enough.
    Level 3 | 40-60  | The answer is more comprehensive, covering the main aspects of the question, but there are still some omissions.
    Level 4 | 60-80  | The answer is comprehensive, covering most aspects of the question, with few omissions.
    Level 5 | 80-100 | The answer is extremely comprehensive, covering all aspects of the question with no omissions, enabling the reader to gain a complete understanding.

    - **Accuracy** - 
    Measure how accurate the answer is compared to the standard answer, including factual correctness and consistency.
    Level   | score range | description
    Level 1 | 0-20   | The answer contains significant factual errors or is largely inconsistent with the standard answer.
    Level 2 | 20-40  | The answer has some factual errors or inconsistencies with the standard answer.
    Level 3 | 40-60  | The answer is mostly accurate, with minor factual errors or inconsistencies.
    Level 4 | 60-80  | The answer is highly accurate, with very few or no factual errors.
    Level 5 | 80-100 | The answer is completely accurate and consistent with the standard answer.

    - **Relevance** - 
    Measure how relevant the answer is to the question, and how well it addresses the specific query.
    Level   | score range | description
    Level 1 | 0-20   | The answer is largely irrelevant to the question.
    Level 2 | 20-40  | The answer has some relevance but often digresses from the main question.
    Level 3 | 40-60  | The answer is mostly relevant, addressing the main question but with some irrelevant content.
    Level 4 | 60-80  | The answer is highly relevant, directly addressing the question with minimal irrelevant content.
    Level 5 | 80-100 | The answer is completely relevant, directly addressing the question without any irrelevant content.

    - **Logical** - 
    Measure whether the answer is coherent, clear, and easy to understand.
    Level   | score range | description
    Level 1 | 0-20   | The answer is illogical, incoherent, and difficult to understand.
    Level 2 | 20-40  | The answer has some logic, but it is incoherent and difficult to understand in parts.
    Level 3 | 40-60  | The answer is logically clear and the sentences are basically coherent, but there are still a few logical loopholes or unclear places.
    Level 4 | 60-80  | The answer is logical, coherent, and easy to understand.
    Level 5 | 80-100 | The answer is extremely logical, fluent and well-organized, making it easy for the reader to follow the author's thoughts.

    - **Readability** - 
    Measure whether the answer is well organized, clear in format, and easy to read.
    Level   | score range | description
    Level 1 | 0-20   | The format of the answer is confused, the writing is poorly organized and difficult to read.
    Level 2 | 20-40  | There are some problems in the format of the answer, the organizational structure of the text is not clear enough, and it is difficult to read.
    Level 3 | 40-60  | The format of the answer is basically clear, the writing structure is good, but there is still room for improvement.
    Level 4 | 60-80  | The format of the answer is clear, the writing is well organized and the reading is smooth.
    Level 5 | 80-100 | The format of the answer is very clear, the writing structure is great, the reading experience is excellent, the format is standardized and easy to understand.

   For each indicator, please give the generated answer a corresponding Level based on the description of the indicator, and then give a score according to the score range of the level.

    Provide a detailed evaluation for each criterion, then give a final score between 0 and 100, where 100 means the generated answer perfectly matches the standard answer.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Explanation": "Provide explanation here",
            "Level": "A level range 1 to 5",
            "Score": "A value range 0 to 100"
        }},
        "Accuracy": {{
            "Explanation": "Provide explanation here",
            "Level": "A level range 1 to 5",
            "Score": "A value range 0 to 100"
        }},
        "Relevance": {{
            "Explanation": "Provide explanation here",
            "Level": "A level range 1 to 5",
            "Score": "A value range 0 to 100"
        }},
        "Logical": {{
            "Explanation": "Provide explanation here",
            "Level": "A level range 1 to 5",
            "Score": "A value range 0 to 100"
        }},
        "Readability": {{
            "Explanation": "Provide explanation here",
            "Level": "A level range 1 to 5",
            "Score": "A value range 0 to 100"
        }},
        "Final Score": "A value range 0 to 100"
    }}
    """
    
    response = await llm_model_func(prompt, sys_prompt)
    return response

def parse_llm_evaluation(response):
    """解析LLM的评估结果"""
    import re
    import json
    
    # 提取JSON部分
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    
    if json_start != -1 and json_end != -1:
        json_str = response[json_start:json_end]
        # 清理JSON字符串中的转义字符问题
        json_str = re.sub(r'\\(?!(["\\/bfnrt]|u[0-9a-fA-F]{4}))', r'\\\\', json_str)
        try:
            eval_data = json.loads(json_str)
            # 构建评估结果
            evaluation = {
                "accuracy": int(eval_data.get("Accuracy", {}).get("Score", 0)),
                "completeness": int(eval_data.get("Comprehensiveness", {}).get("Score", 0)),
                "relevance": int(eval_data.get("Relevance", {}).get("Score", 0)),
                "logical": int(eval_data.get("Logical", {}).get("Score", 0)),
                "readability": int(eval_data.get("Readability", {}).get("Score", 0)),
                "clarity": int(eval_data.get("Readability", {}).get("Score", 0)),  # 保持向后兼容
                "final_score": int(eval_data.get("Final Score", 0)),
                "detailed_evaluation": "\n".join([
                    f"Comprehensiveness: {eval_data.get('Comprehensiveness', {}).get('Explanation', '')}",
                    f"Accuracy: {eval_data.get('Accuracy', {}).get('Explanation', '')}",
                    f"Relevance: {eval_data.get('Relevance', {}).get('Explanation', '')}",
                    f"Logical: {eval_data.get('Logical', {}).get('Explanation', '')}",
                    f"Readability: {eval_data.get('Readability', {}).get('Explanation', '')}"
                ]),
                "raw_response": response  # 保存原始响应，便于调试
            }
        except json.JSONDecodeError:
            # 如果解析失败，返回默认值
            evaluation = {
                "accuracy": 0,
                "completeness": 0,
                "relevance": 0,
                "logical": 0,
                "readability": 0,
                "clarity": 0,
                "final_score": 0,
                "detailed_evaluation": "Failed to parse JSON",
                "raw_response": response
            }
    else:
        # 如果没有找到JSON，返回默认值
        evaluation = {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "logical": 0,
            "readability": 0,
            "clarity": 0,
            "final_score": 0,
            "detailed_evaluation": "No JSON found",
            "raw_response": response
        }
    
    return evaluation

def load_fin_data(question_index=None, max_items=None):
    """加载fin.jsonl文件中的数据

    Args:
        question_index: 如果指定，则只返回指定索引的问题（从0开始）
        max_items: 最大加载数量，默认为None表示不限制
    """
    data = []
    count = 0
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if question_index is not None and count > question_index:
                break
            if max_items is not None and count >= max_items:
                break
            try:
                item = json.loads(line.strip())
                if all(key in item for key in ['input', 'context', 'answers']):
                    if question_index is not None and count == question_index:
                        data.append(item)
                        break
                    elif question_index is None:
                        data.append(item)
                    count += 1
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line}")
    return data

def init_rag():
    """初始化HyperRAG实例"""
    rag = HyperRAG(
        working_dir=RAG_CACHE_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
    )
    return rag

async def evaluate_topology(question_index=None, max_items=None):
    """评估topology模式的回答质量

    Args:
        question_index: 要测试的问题索引（从0开始），如果为None则测试所有数据
        max_items: 最大测试数量，默认为10
    """
    if max_items is None:
        max_items = 10
    data = load_fin_data(question_index, max_items)
    print(f"加载了 {len(data)} 条数据")

    # 初始化RAG
    rag = init_rag()
    print("RAG实例初始化成功")

    # 评估结果
    results = []

    # 处理每条数据
    for i, item in enumerate(data):
        print(f"处理第 {i+1} 条数据")
        question = item['input']
        standard_answers = item['answers']

        # 使用topology模式回答问题
        try:
            response = await rag.aquery(
                question,
                param=QueryParam(mode="topology")
            )
            
            # 对每个标准回答进行评估
            evaluations = []
            for standard_answer in standard_answers:
                # 使用LLM评估回答
                llm_eval = await evaluate_answer_with_llm(question, response, standard_answer)
                # 解析评估结果
                parsed_eval = parse_llm_evaluation(llm_eval)
                evaluations.append(parsed_eval)
            
            # 计算平均评分
            if evaluations:
                avg_final_score = sum(eval['final_score'] for eval in evaluations) / len(evaluations)
            else:
                avg_final_score = 0
            
            results.append({
                "question": question,
                "generated_answer": response,
                "standard_answers": standard_answers,
                "evaluations": evaluations,
                "avg_final_score": avg_final_score
            })
            
            print(f"问题: {question}")
            print(f"生成回答: {response}")
            print(f"标准回答: {standard_answers}")
            print(f"平均评分: {avg_final_score:.2f}")
            
            # 打印详细的评估信息
            if evaluations:
                print("详细评估:")
                # 计算五个维度的平均评分
                avg_comprehensiveness = sum(eval['completeness'] for eval in evaluations) / len(evaluations)
                avg_accuracy = sum(eval['accuracy'] for eval in evaluations) / len(evaluations)
                avg_relevance = sum(eval['relevance'] for eval in evaluations) / len(evaluations)
                avg_logical = sum(eval['logical'] for eval in evaluations) / len(evaluations)
                avg_readability = sum(eval['readability'] for eval in evaluations) / len(evaluations)
                
                print(f"Comprehensiveness: {avg_comprehensiveness:.2f}")
                print(f"Accuracy: {avg_accuracy:.2f}")
                print(f"Relevance: {avg_relevance:.2f}")
                print(f"Logical: {avg_logical:.2f}")
                print(f"Readability: {avg_readability:.2f}")
            print("-" * 80)
            
        except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "question": question,
                    "generated_answer": "Error",
                    "standard_answers": standard_answers,
                    "evaluations": [],
                    "avg_final_score": 0
                })
    
    # 计算平均评分
    avg_score = sum(r['avg_final_score'] for r in results) / len(results) if results else 0
    print(f"整体平均评分: {avg_score:.2f}")
    
    # 保存结果
    with open("evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("评估结果已保存到 evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(evaluate_topology(None, 10))
