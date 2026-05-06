import re
import sys
import json
import tiktoken
import numpy as np
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
import nltk
from difflib import SequenceMatcher

sys.path.append(str(Path(__file__).resolve().parent.parent))

from my_config import LLM_API_KEY, LLM_BASE_URL

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

LLM_MODEL = "gpt-4o-mini"

MAX_CONTEXT_TOKENS = 9000
CONTEXT_CHUNK_SIZE = 3
MAX_VALIDATE_RETRIES = 5
QUESTION_SIMILARITY_THRESHOLD = 0.55
MAX_CONTEXT_RETRIES = 10
LLM_TEMPERATURE = 0.7
QUESTIONS_PER_CONTEXT = 3


def llm_model_func(prompt, system_prompt=None, history_messages=[], temperature=LLM_TEMPERATURE, **kwargs) -> str:
    openai_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, **kwargs
    )
    return response.choices[0].message.content


def split_text_to_sentences(text):
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    sentences = nltk.sent_tokenize(clean_text)
    return [s.strip() for s in sentences if s.strip()]


def smooth_truncate(text, max_tokens, tokenizer):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    sentences = split_text_to_sentences(text)
    if not sentences:
        return tokenizer.decode(tokens[:max_tokens])

    current_tokens = 0
    selected_sentences = []

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        if current_tokens + len(sentence_tokens) <= max_tokens:
            selected_sentences.append(sentence)
            current_tokens += len(sentence_tokens)
        else:
            break

    if not selected_sentences:
        return tokenizer.decode(tokens[:max_tokens])

    return ' '.join(selected_sentences)


def select_coherent_context(chunk_ids, chunk_size=3, used_chunk_ids=None):
    """选取连贯的上下文chunk ID组，优先选择包含未使用chunk的区间

    通过计算每个候选窗口的"新鲜度"（包含多少未使用的chunk），
    优先选择新鲜度高的窗口，从而最大化上下文多样性。

    Args:
        chunk_ids: 所有chunk ID的有序列表
        chunk_size: 选取的连续chunk数量
        used_chunk_ids: 已使用过的chunk ID集合

    Returns:
        (selected_ids, selected_id_set): 选中的chunk ID列表和集合
    """
    if len(chunk_ids) <= chunk_size:
        return chunk_ids, set(chunk_ids)

    max_idx = len(chunk_ids) - chunk_size
    if max_idx <= 0:
        selected = chunk_ids[:chunk_size]
        return selected, set(selected)

    if used_chunk_ids is not None:
        best_freshness = -1
        best_candidates = []
        for start in range(max_idx + 1):
            window_ids = set(chunk_ids[start:start + chunk_size])
            freshness = len(window_ids - used_chunk_ids)
            if freshness > best_freshness:
                best_freshness = freshness
                best_candidates = [start]
            elif freshness == best_freshness:
                best_candidates.append(start)
        idx = np.random.choice(best_candidates)
    else:
        idx = np.random.randint(0, max_idx + 1)

    selected = chunk_ids[idx:idx + chunk_size]
    return selected, set(selected)


def extract_quoted_terms(text):
    """从文本中提取所有引号内的术语，用于验证问题是否引用了原文中不存在的概念"""
    terms = set()
    for match in re.finditer(r'[""\u201c\u201d\u2018\u2019]([^""\u201c\u201d\u2018\u2019]+)[""\u201c\u201d\u2018\u2019]', text):
        term = match.group(1).strip()
        term = term.rstrip(',.;:!?')
        if len(term) > 1 and not term.isdigit():
            terms.add(term)
    for match in re.finditer(r"'([^']+)'", text):
        term = match.group(1).strip()
        term = term.rstrip(',.;:!?')
        if len(term) > 1 and not term.isdigit() and term[0].isupper():
            terms.add(term)
    return terms


def validate_question_terms(question, answer, context):
    """验证问题和答案中的引号术语是否存在于参考文本中

    检查问题和答案中使用引号标注的术语（如 'Tax Credits'、'Tenant'）
    是否在参考文本中出现。如果出现不存在的术语，说明LLM产生了幻觉，
    应拒绝该问题。

    Args:
        question: 生成的问题
        answer: 生成的答案
        context: 参考文本

    Returns:
        (is_valid, hallucinated_terms)
    """
    context_lower = context.lower()
    context_terms = extract_quoted_terms(context)
    context_terms_lower = {t.lower() for t in context_terms}

    qa_text = question + " " + answer
    qa_terms = extract_quoted_terms(qa_text)

    hallucinated = []
    for term in qa_terms:
        if term.lower() not in context_terms_lower and term.lower() not in context_lower:
            hallucinated.append(term)

    return len(hallucinated) == 0, hallucinated


def compute_question_similarity(new_question, existing_questions):
    """计算新问题与已有问题列表的最大相似度

    综合使用Jaccard词集合相似度和SequenceMatcher序列相似度，
    取两者中的较大值作为最终相似度，避免单一指标的盲区。

    Args:
        new_question: 待检测的新问题
        existing_questions: 已有的问题列表

    Returns:
        最大相似度值（0.0~1.0），无已有问题时返回0.0
    """
    if not existing_questions:
        return 0.0

    new_lower = new_question.lower()
    new_words = set(new_lower.split())
    max_sim = 0.0

    for existing in existing_questions:
        existing_lower = existing.lower()
        existing_words = set(existing_lower.split())

        intersection = new_words & existing_words
        union = new_words | existing_words
        jaccard = len(intersection) / len(union) if union else 0.0

        seq_ratio = SequenceMatcher(None, new_lower, existing_lower).ratio()

        sim = max(jaccard, seq_ratio)
        max_sim = max(max_sim, sim)

    return max_sim


COMMON_SHARED_INSTRUCTIONS = """
ABSOLUTE GROUNDING RULES — VIOLATION OF ANY RULE BELOW MAKES THE QUESTION INVALID:

1. ZERO HALLUCINATION: Every term, entity name, concept, and definition mentioned in the question and answer MUST appear EXACTLY as written in the Reference. Do NOT invent, infer, derive, paraphrase, or generalize any terms that do not appear verbatim in the Reference text.

2. NO EXTERNAL KNOWLEDGE: Do NOT use your general knowledge to supplement, connect, or infer information not explicitly stated in the Reference. The question must be fully answerable using ONLY the information provided in the Reference.

3. QUOTED TERMS VERIFICATION: When you reference a defined term from the Reference (e.g., 'Secured Parties', 'Tenant'), you MUST use the EXACT term as it appears in the Reference. Do NOT create variant terms (e.g., do NOT write 'Tax Credits' if the Reference only defines 'Taxes'; do NOT write 'Related Security' if the Reference only defines 'Security Documents').

4. RELATIONSHIP GROUNDING: When asking about relationships between terms, both terms MUST be explicitly defined or mentioned in the Reference, AND the relationship between them MUST be derivable from the Reference text alone. Do NOT fabricate connections based on domain knowledge.

5. ANSWER FAITHFULNESS: The answer must contain ONLY facts stated in the Reference. Quote or closely paraphrase the Reference text. If the Reference does not provide enough information to answer a sub-question, state that explicitly rather than inventing content.
"""

STAGE_SPECIFIC_INSTRUCTIONS = {
    1: """You are designing test questions for a knowledge retrieval system. Your task is to create a SINGLE-ASPECT question based on the Reference, with a concise correct answer.

The question should test whether the system can locate and extract a specific fact, definition, or detail from the Reference.

################
Reference:
{context}
################

QUESTION DESIGN RULES:
1. Focus on a SINGLE aspect or detail from the Reference — one definition, one fact, one condition, or one relationship.
2. The question must NOT include conjunctions like "and", "or", "specifically", "particularly", "and how", "and what" that imply multiple inquiries.
3. Target specific details: exact definitions, precise conditions, named entities, numerical values, or explicit relationships stated in the Reference.
4. Avoid vague or macro-level questions. The question should require finding a precise piece of information.
5. State the question directly in one sentence. Do NOT use meta-references like "in this reference" or "in the data set".
6. The question should be in English.

{shared_instructions}

OUTPUT FORMAT:
{{
    "Question": [your question],
    "Answer": [concise correct answer based ONLY on the Reference]
}}""",

    2: """You are designing test questions for a knowledge retrieval system. Your task is to create a TWO-PART question with progressive depth based on the Reference, with a concise correct answer.

The question should test whether the system can:
- First: locate a specific fact or definition
- Then: connect it to a related fact or implication explicitly stated in the Reference

The two parts must be connected by transitional phrases like "and" or "specifically", forming a single sentence.

IMPORTANT: Both parts must reference terms and facts that EXIST in the Reference. Do NOT create a second part that requires information not in the Reference.

################
Reference:
{context}
################

QUESTION DESIGN RULES:
1. The question must contain exactly TWO interconnected sub-questions in a single sentence.
2. The second sub-question must build upon or relate to the first, creating a progressive depth.
3. Both sub-questions must be answerable from the Reference alone.
4. Target specific details: definitions, conditions, relationships, or obligations explicitly stated.
5. State the question directly. Do NOT use meta-references like "in this reference" or "in the data set".
6. The question should be in English.

{shared_instructions}

OUTPUT FORMAT:
{{
    "Question": [your two-part question],
    "Answer": [concise correct answer covering both parts, based ONLY on the Reference]
}}""",

    3: """You are designing test questions for a knowledge retrieval system. Your task is to create a THREE-PART question with progressive depth based on the Reference, with a concise correct answer.

The question should test whether the system can:
- First: locate a specific fact or definition
- Then: connect it to a related concept explicitly defined in the Reference
- Finally: explain the relationship or implication between them as stated in the Reference

The three parts must be connected by transitional phrases like "and" or "specifically", forming a single sentence.

CRITICAL: All three parts must reference terms and facts that EXIST in the Reference. The "connection" between terms must be derivable from the Reference text — do NOT invent relationships based on domain knowledge.

################
Reference:
{context}
################

QUESTION DESIGN RULES:
1. The question must contain exactly THREE interconnected sub-questions in a single sentence.
2. Each sub-question must progressively deepen: definition → related concept → their connection.
3. ALL three sub-questions must be answerable from the Reference alone.
4. When asking about relationships between terms, BOTH terms must be explicitly defined or mentioned in the Reference, AND their relationship must be derivable from the Reference.
5. Target specific details: definitions, conditions, relationships, or obligations explicitly stated.
6. State the question directly. Do NOT use meta-references like "in this reference" or "in the data set".
7. The question should be in English.

{shared_instructions}

OUTPUT FORMAT:
{{
    "Question": [your three-part question],
    "Answer": [concise correct answer covering all three parts, based ONLY on the Reference]
}}""",

    4: """You are designing test questions for a knowledge retrieval system. Your task is to create a FOUR-PART question with progressive depth based on the Reference, with a concise correct answer.

The question should test whether the system can trace a chain of related facts across multiple defined terms in the Reference:
- First: identify a definition
- Second: connect to a related defined term
- Third: explain their explicit relationship
- Fourth: identify a further implication or condition stated in the Reference

The four parts must be connected by transitional phrases, forming a single sentence.

CRITICAL: All four parts must reference terms and facts that EXIST in the Reference. Do NOT fabricate intermediate terms or relationships. If the Reference does not contain enough connected terms for a four-part question, focus on what IS present rather than inventing connections.

################
Reference:
{context}
################

QUESTION DESIGN RULES:
1. The question must contain exactly FOUR interconnected sub-questions in a single sentence.
2. Each sub-question must progressively build upon the previous one.
3. ALL four sub-questions must be answerable from the Reference alone.
4. Every term mentioned in the question must appear verbatim in the Reference.
5. Target specific details: definitions, conditions, relationships, or obligations explicitly stated.
6. State the question directly. Do NOT use meta-references like "in this reference" or "in the data set".
7. The question should be in English.

{shared_instructions}

OUTPUT FORMAT:
{{
    "Question": [your four-part question],
    "Answer": [concise correct answer covering all four parts, based ONLY on the Reference]
}}""",

    5: """You are designing test questions for a knowledge retrieval system. Your task is to create a FIVE-PART question with progressive depth based on the Reference, with a concise correct answer.

The question should test whether the system can trace an extended chain of related facts across multiple defined terms in the Reference, following the logical structure of the document.

The five parts must be connected by transitional phrases, forming a single sentence.

CRITICAL: All five parts must reference terms and facts that EXIST in the Reference. This is the most demanding question type — it requires the Reference to contain a rich network of interconnected terms. If the Reference lacks sufficient connected content, focus on the connections that DO exist rather than fabricating them.

################
Reference:
{context}
################

QUESTION DESIGN RULES:
1. The question must contain exactly FIVE interconnected sub-questions in a single sentence.
2. Each sub-question must progressively build upon the previous one.
3. ALL five sub-questions must be answerable from the Reference alone.
4. Every term mentioned in the question must appear verbatim in the Reference.
5. Target specific details: definitions, conditions, relationships, or obligations explicitly stated.
6. State the question directly. Do NOT use meta-references like "in this reference" or "in the data set".
7. The question should be in English.

{shared_instructions}

OUTPUT FORMAT:
{{
    "Question": [your five-part question],
    "Answer": [concise correct answer covering all five parts, based ONLY on the Reference]
}}"""
}


def build_prompt(question_stage, context, num_questions=1, existing_questions_for_context=None, all_existing_questions=None):
    """构建指定阶段的问题生成提示词

    当num_questions > 1时，要求LLM一次性生成多个不同角度的问题，
    并将当前上下文已有问题作为参考，避免重复。

    Args:
        question_stage: 问题阶段（1-5）
        context: 参考文本
        num_questions: 本次需要生成的问题数量
        existing_questions_for_context: 当前上下文已生成的问题列表，供LLM参考避免重复
        all_existing_questions: 全局已生成的问题列表，用于跨上下文去重提示

    Returns:
        完整的提示词字符串
    """
    template = STAGE_SPECIFIC_INSTRUCTIONS[question_stage]

    shared_content = template.format(
        context=context,
        shared_instructions=COMMON_SHARED_INSTRUCTIONS.strip()
    )

    parts = shared_content.split("OUTPUT FORMAT:")
    body = parts[0]

    if num_questions > 1:
        stage_replacements = {
            1: [
                ("create a SINGLE-ASPECT question",
                 f"create {num_questions} SINGLE-ASPECT questions, each targeting a DIFFERENT entity, concept, or detail"),
                ("Focus on a SINGLE aspect or detail from the Reference",
                 f"Each question must focus on a SINGLE aspect, but the {num_questions} questions together must cover DIFFERENT entities, concepts, or details from the Reference"),
            ],
            2: [
                ("create a TWO-PART question",
                 f"create {num_questions} TWO-PART questions, each targeting a DIFFERENT entity or concept"),
            ],
            3: [
                ("create a THREE-PART question",
                 f"create {num_questions} THREE-PART questions, each targeting a DIFFERENT entity or concept"),
            ],
            4: [
                ("create a FOUR-PART question",
                 f"create {num_questions} FOUR-PART questions, each targeting a DIFFERENT entity or concept"),
            ],
            5: [
                ("create a FIVE-PART question",
                 f"create {num_questions} FIVE-PART questions, each targeting a DIFFERENT entity or concept"),
            ],
        }
        for old_text, new_text in stage_replacements.get(question_stage, []):
            body = body.replace(old_text, new_text)

    diversity_parts = []

    if num_questions > 1:
        diversity_parts.append(f"""MULTI-QUESTION REQUIREMENT:
Generate exactly {num_questions} questions about the Reference. Each question must still follow the stage-specific format (e.g., SINGLE-ASPECT for stage 1), but the {num_questions} questions must target DIFFERENT entities, concepts, definitions, numerical values, conditions, or relationships from the Reference. Do NOT ask about the same term or detail twice.""")

    if existing_questions_for_context:
        ctx_questions_text = "\n".join(f"  - {q}" for q in existing_questions_for_context)
        diversity_parts.append(f"""EXISTING QUESTIONS FOR THIS REFERENCE:
The following questions have already been created for this Reference. Do NOT duplicate or create similar questions. Focus on DIFFERENT aspects:
{ctx_questions_text}""")

    if all_existing_questions:
        recent = all_existing_questions[-8:]
        global_questions_text = "\n".join(f"  - {q}" for q in recent)
        diversity_parts.append(f"""GLOBAL DIVERSITY NOTE:
The following questions have been generated across other References. Avoid creating questions that are too similar in topic or wording:
{global_questions_text}""")

    if num_questions > 1:
        output_section = f"""OUTPUT FORMAT (generate exactly {num_questions} questions as a JSON array):
[
    {{
        "Question": [first question from one angle],
        "Answer": [concise correct answer based ONLY on the Reference]
    }},
    {{
        "Question": [second question from a DIFFERENT angle],
        "Answer": [concise correct answer based ONLY on the Reference]
    }}
]"""
    else:
        output_section = """OUTPUT FORMAT:
{
    "Question": [your question],
    "Answer": [concise correct answer based ONLY on the Reference]
}"""

    result = body
    if diversity_parts:
        result += "\n\n" + "\n\n".join(diversity_parts) + "\n\n"
    result += output_section

    return result


def parse_qa_pairs(response, num_questions):
    """从LLM响应中解析问答对

    支持两种格式：单问题JSON对象和多问题JSON数组。
    优先尝试解析数组格式，失败后回退到正则提取。

    Args:
        response: LLM返回的原始文本
        num_questions: 期望的问题数量

    Returns:
        [(question, answer), ...] 问答对列表
    """
    pairs = []

    json_block = re.search(r'\[[\s\S]*?\]', response)
    if json_block:
        try:
            items = json.loads(json_block.group())
            for item in items:
                q = item.get("Question", "").strip()
                a = item.get("Answer", "").strip()
                if q and a:
                    pairs.append((q, a))
            if pairs:
                return pairs
        except (json.JSONDecodeError, AttributeError):
            pass

    questions = re.findall(r'"Question":\s*"(.*?)"', response)
    answers = re.findall(r'"Answer":\s*"(.*?)"', response)
    for q, a in zip(questions, answers):
        if q.strip() and a.strip():
            pairs.append((q.strip(), a.strip()))

    return pairs


encoding = tiktoken.encoding_for_model("gpt-4o")


def extract_questions_from_contexts(data_name, question_stage=1, max_questions=5):
    """从上下文中提取问题，专为复形RAG系统准备

    生成的质量保证机制：
    1. 提示词层面：通过严格的GROUNDING规则防止LLM幻觉术语
    2. 验证层面：对生成的问题和答案进行术语验证，检测引号内术语是否存在于参考文本
    3. 重试机制：验证失败时重新生成，最多重试MAX_VALIDATE_RETRIES次
    4. 上下文去重：通过chunk ID组合记录已使用的上下文，避免重复选取
    5. 问题去重：通过Jaccard+SequenceMatcher双重相似度检测，拒绝与已有问题过于相似的新问题
    6. Chunk多样化：优先选取包含未使用chunk的区间，最大化上下文覆盖面
    7. LLM多样性约束：在提示词中注入已生成问题列表，引导LLM关注不同方面

    Args:
        data_name: 数据集名称
        question_stage: 问题阶段（1-5）
        max_questions: 最大问题数量
    """
    try:
        WORKING_DIR = Path("caches") / data_name
        WORKING_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n🔧 Extracting {question_stage}-stage questions for Simplicial Complex RAG")
        print(f"📁 Working directory: {WORKING_DIR}")

        project_root = Path(__file__).resolve().parent.parent
        chunks_file = project_root / "caches" / data_name / "rag" / "kv_store_text_chunks.json"
        with open(chunks_file, mode="r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunk_ids = []
        chunk_contents = {}
        for chunk_id, chunk_info in chunks_data.items():
            if "content" in chunk_info:
                chunk_ids.append(chunk_id)
                chunk_contents[chunk_id] = chunk_info["content"]

        print(f"📊 Found {len(chunk_ids)} text chunks")

        if len(chunk_ids) == 0:
            print("❌ No text chunks found in the chunks file")
            return []

        question_list, answer_list, reference_list = [], [], []
        len_big_chunks = CONTEXT_CHUNK_SIZE

        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

        cnt = 0
        hallucination_reject_count = 0
        similarity_reject_count = 0
        context_duplicate_count = 0
        used_context_id_sets = set()
        used_chunk_ids = set()
        consecutive_context_dup = 0

        with tqdm(total=max_questions, desc=f"Extracting {question_stage}-stage questions") as pbar:
            while cnt < max_questions:
                selected_ids, selected_id_set = select_coherent_context(
                    chunk_ids, len_big_chunks, used_chunk_ids
                )
                big_chunks = [chunk_contents[cid] for cid in selected_ids]
                context = "".join(big_chunks)

                context = smooth_truncate(context, MAX_CONTEXT_TOKENS, tokenizer)

                if len(context) < 500:
                    continue

                context_key = frozenset(selected_ids)
                if context_key in used_context_id_sets:
                    context_duplicate_count += 1
                    consecutive_context_dup += 1
                    if consecutive_context_dup >= MAX_CONTEXT_RETRIES:
                        print("  ⚠️  连续多次遇到重复上下文，上下文多样性已耗尽，提前终止")
                        break
                    continue

                used_context_id_sets.add(context_key)
                consecutive_context_dup = 0
                used_chunk_ids.update(selected_id_set)

                context_questions = []
                context_answers = []
                remaining = QUESTIONS_PER_CONTEXT
                context_attempts = 0

                while remaining > 0 and cnt < max_questions and context_attempts < MAX_VALIDATE_RETRIES:
                    context_attempts += 1

                    try:
                        prompt = build_prompt(
                            question_stage, context,
                            num_questions=remaining,
                            existing_questions_for_context=context_questions if context_questions else None,
                            all_existing_questions=question_list if question_list else None
                        )
                        response = llm_model_func(prompt, temperature=LLM_TEMPERATURE)

                        qa_pairs = parse_qa_pairs(response, remaining)
                        if not qa_pairs:
                            continue

                        for q_text, a_text in qa_pairs:
                            if cnt >= max_questions:
                                break

                            is_valid, hallucinated_terms = validate_question_terms(q_text, a_text, context)
                            if not is_valid:
                                print(f"  ⚠️  幻觉术语检测: {hallucinated_terms}")
                                hallucination_reject_count += 1
                                continue

                            similarity = compute_question_similarity(q_text, question_list)
                            if similarity > QUESTION_SIMILARITY_THRESHOLD:
                                print(f"  ⚠️  问题相似度过高 ({similarity:.2f}): {q_text[:60]}...")
                                similarity_reject_count += 1
                                continue

                            question_list.append(q_text)
                            answer_list.append(a_text)
                            reference_list.append(context)
                            context_questions.append(q_text)
                            context_answers.append(a_text)
                            remaining -= 1
                            cnt += 1
                            pbar.update(1)

                    except Exception as e:
                        print(f"  ⚠️  生成错误: {e}")
                        continue

                if not context_questions:
                    hallucination_reject_count += 1

        project_root = Path(__file__).resolve().parent.parent
        questions_dir = project_root / "caches" / data_name / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)

        question_file_path = questions_dir / f"{question_stage}_stage.json"
        answer_file_path = questions_dir / f"{question_stage}_stage_answer.json"
        ref_file_path = questions_dir / f"{question_stage}_stage_ref.json"

        with open(question_file_path, "w", encoding="utf-8") as f:
            json.dump(question_list, f, ensure_ascii=False, indent=4)
        with open(answer_file_path, "w", encoding="utf-8") as f:
            json.dump(answer_list, f, ensure_ascii=False, indent=4)
        with open(ref_file_path, "w", encoding="utf-8") as f:
            json.dump(reference_list, f, ensure_ascii=False, indent=4)

        print(f"\n🎉 Extracted {len(question_list)} questions and answers")
        if hallucination_reject_count > 0:
            print(f"🚫 Rejected {hallucination_reject_count} questions due to hallucinated terms")
        if similarity_reject_count > 0:
            print(f"🔄 Rejected {similarity_reject_count} questions due to similarity with existing questions")
        if context_duplicate_count > 0:
            print(f"📋 Skipped {context_duplicate_count} duplicate contexts")
        print(f"📊 Used {len(used_chunk_ids)}/{len(chunk_ids)} unique chunks ({len(used_chunk_ids)/len(chunk_ids)*100:.1f}%)")
        print(f"📁 Questions saved to: {question_file_path}")
        print(f"📁 Answers saved to: {answer_file_path}")
        print(f"📁 References saved to: {ref_file_path}")
        return question_list
    except Exception as e:
        print(f"❌ Failed to extract questions: {e}")
        raise

if __name__ == "__main__":
    data_name = "legal"
    question_stage = 1
    max_questions = 3
    try:
        extract_questions_from_contexts(data_name, question_stage, max_questions)
    except Exception as e:
        print(f"\n❌ Program failed: {e}")
        exit(1)
