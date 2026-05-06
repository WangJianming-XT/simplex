import re
from typing import Optional

from ..utils import logger
from ..llm import openai_complete_if_cache
from ..prompt import PROMPTS


PROMPTS["cot_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Knowledge---

{knowledge}

---Knowledge Structure---

The knowledge above is a structured context with three sections:
- -----Entities-----: CSV with columns [name, type, is_seed, description]. Entities extracted from the query and knowledge base. is_seed=yes means entity from the query.
- -----Simplices-----: CSV with columns [dimension, entities, is_seed, description]. Simplices represent relationships of varying dimensions (Relation/Triangle/Tetrahedron). is_seed=yes means directly matched. Higher dimension = stronger relationship. description contains the original text fragment from the source document.
- -----Sources-----: CSV with columns [id, content]. Original document passages — this is ground truth.

RULES:
1. Extract ALL relevant facts from the Sources that answer the question. Include every specific detail the Sources provide — exact definitions, conditions, parties, obligations.
2. Use the EXACT wording from Sources. Quote or closely paraphrase the Source text for definitions and key statements.
3. When multiple terms are asked about, address each one using only what the Sources state about it, then explain their connection if the Sources describe one.
4. Use Entities and Simplices to understand how facts are connected.
5. NEVER supplement with general knowledge or explanations not found in the Sources. If the Sources do not define a term, say so — do not provide your own definition.
6. Organize your answer clearly but do not pad it with restatements or generic background.
7. If the context does not contain relevant information, state this clearly.

---Goal---

Answer the given question.
You must first conduct reasoning inside <think_tag>...</think_tag>.
When you have the final answer, you can output the answer inside <answer_tag>...</answer_tag>.

Output format for answer:
<think_tag>
...
</think_tag>

<answer_tag>
...
</answer_tag>

---Question---

{question}
"""


async def generate_response(
    knowledge: str,
    question: str,
    llm_model_func=None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> dict:
    """基于给定知识和问题，使用思维链（Chain-of-Thought）生成回答

    通过要求模型先在 <think...</think 标签内进行推理，
    再在 <answer>...</answer> 标签内输出最终答案，
    引导模型进行逐步推理以提高回答质量。

    支持两种 LLM 调用方式：
    1. 传入 llm_model_func（可调用对象），直接调用
    2. 传入 model/api_key/base_url，使用 openai_complete_if_cache

    Args:
        knowledge: 检索到的相关知识文本
        question: 用户提出的问题
        llm_model_func: 自定义 LLM 调用函数（可选），
            签名应为 async (prompt, **kwargs) -> str
        model: 模型名称（当不传 llm_model_func 时使用）
        api_key: API 密钥（当不传 llm_model_func 时使用）
        base_url: API 基础 URL（当不传 llm_model_func 时使用）
        **kwargs: 透传给 LLM 调用函数的额外参数

    Returns:
        dict: 包含以下键的字典：
            - 'prompt': 构建的完整提示文本
            - 'generation': 模型生成的回答（含思维链推理过程），
              若调用失败则为 "[ERROR] 错误信息"
            - 'final_answer': 从生成结果中提取的最终答案
    """
    prompt = PROMPTS["cot_response"].format(
        knowledge=knowledge,
        question=question,
    )

    result = {
        "prompt": prompt,
        "generation": "",
        "final_answer": "",
    }

    try:
        if llm_model_func is not None and callable(llm_model_func):
            generation = await llm_model_func(prompt, **kwargs)
        elif model is not None:
            generation = await openai_complete_if_cache(
                model,
                prompt,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            raise ValueError(
                "必须提供 llm_model_func 或 model 参数之一"
            )
        result["generation"] = generation
    except Exception as e:
        logger.error(f"generate_response 调用失败: {e}")
        result["generation"] = f"[ERROR] {str(e)}"
        result["final_answer"] = ""
        return result

    result["final_answer"] = extract_answer_from_generation(result["generation"])
    return result


def extract_answer_from_generation(generation_text: str) -> str:
    """从思维链生成结果中提取最终答案

    依次尝试匹配以下标签格式：
    1. <answer_tag>...</answer_tag>（当前标准格式）
    2. <answer>...</answer>（旧格式兼容）

    Args:
        generation_text: 模型生成的完整文本（含思维链和最终答案）

    Returns:
        str: 提取出的最终答案文本；
             若未找到任何答案标签则返回原始生成文本
    """
    match = re.search(r'<answer_tag>\s*(.*?)\s*</answer_tag>', generation_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', generation_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generation_text
