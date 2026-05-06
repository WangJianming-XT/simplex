import re
from pathlib import Path
import nltk
import numpy as np

from ..utils import (
    logger,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
)
from ._config import SentenceTransformer, cosine_similarity, model, semantic_similarity


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    if overlap_token_size >= max_token_size:
        overlap_token_size = max_token_size - 1
        logger.warning(f"overlap_token_size >= max_token_size, 调整 overlap_token_size={overlap_token_size}")
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    step = max(1, max_token_size - overlap_token_size)
    for index, start in enumerate(
        range(0, len(tokens), step)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


def split_text_to_sentences(text):
    """清洗文本并按句子分割"""
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    sentences = nltk.sent_tokenize(clean_text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunking(text, threshold, min_tokens, max_tokens, max_chunk_size):
    """基于语义相似度的分块逻辑"""
    global model
    if model is None:
        try:
            model_path = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2"
            if model_path.exists():
                model = SentenceTransformer(str(model_path))
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            return chunking_by_token_size(text, max_token_size=max_tokens)
    
    sentences = split_text_to_sentences(text)
    
    if len(sentences) == 0:
        return []
    
    sentence_tokens = [len(sentence.split()) for sentence in sentences]
    total_tokens = sum(sentence_tokens)

    try:
        embeddings = model.encode(sentences, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Failed to encode sentences: {e}")
        return chunking_by_token_size(text, max_token_size=max_tokens)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0].reshape(1, -1)
    current_token_count = sentence_tokens[0]

    for i in range(1, len(sentences)):
        next_sentence = sentences[i]
        next_embedding = embeddings[i].reshape(1, -1)
        next_token_count = sentence_tokens[i]

        should_split = False

        if current_token_count + next_token_count > max_tokens:
            should_split = True
        elif current_token_count >= max_tokens:
            should_split = True
        elif current_token_count >= 1000 or current_token_count + next_token_count >= min_tokens:
            similarity = cosine_similarity(current_embedding, next_embedding)[0][0]
            if similarity < threshold and current_token_count >= min_tokens:
                should_split = True

        if should_split:
            current_chunk_str = " ".join(current_chunk)
            chunks.append(current_chunk_str)
            current_chunk = [next_sentence]
            current_embedding = next_embedding
            current_token_count = next_token_count
        else:
            current_chunk.append(next_sentence)
            total_emb = current_embedding * (len(current_chunk) - 1) + next_embedding
            current_embedding = total_emb / len(current_chunk)
            current_token_count += next_token_count

    if current_chunk:
        if current_token_count < min_tokens and chunks:
            last_chunk = chunks.pop()
            combined_chunk = last_chunk + " " + " ".join(current_chunk)
            combined_token_count = len(last_chunk.split()) + current_token_count
            if combined_token_count <= max_tokens:
                chunks.append(combined_chunk)
            else:
                current_chunk_str = " ".join(current_chunk)
                chunks.append(last_chunk)
                chunks.append(current_chunk_str)
        else:
            current_chunk_str = " ".join(current_chunk)
            if current_token_count > max_tokens:
                sub_chunks = []
                current_sub_chunk = ""
                current_sub_token_count = 0
                for sentence in current_chunk:
                    sentence_token_count = len(sentence.split())
                    if current_sub_token_count + sentence_token_count <= max_tokens:
                        current_sub_chunk += " " + sentence if current_sub_chunk else sentence
                        current_sub_token_count += sentence_token_count
                    else:
                        if current_sub_chunk:
                            sub_chunks.append(current_sub_chunk)
                        current_sub_chunk = sentence
                        current_sub_token_count = sentence_token_count
                if current_sub_chunk:
                    sub_chunks.append(current_sub_chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk_str)

    return chunks


def chunking_by_semantic(
    content: str, config: dict
):
    """基于语义相似度的文本分块"""
    threshold = config.get('semantic_chunking_threshold', 0.5)
    min_tokens = config.get('semantic_chunking_min_tokens', 1650)
    max_tokens = config.get('semantic_chunking_max_tokens', 1750)
    max_chunk_size = config.get('semantic_chunking_max_chunk_size', 1750)
    semantic_chunks = semantic_chunking(content, threshold, min_tokens, max_tokens, max_chunk_size)
    results = []
    for index, chunk_content in enumerate(semantic_chunks):
        chunk_tokens = len(encode_string_by_tiktoken(chunk_content))
        results.append(
            {
                "tokens": chunk_tokens,
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results
