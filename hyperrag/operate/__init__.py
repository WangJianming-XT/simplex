"""
Hyper-RAG 操作模块

将原始 operate.py 拆分为以下子模块：
- _config: 配置与常量（DualDimensionConfig, EMB_MODEL 等）
- _chunking: 文本分块（chunking_by_token_size, chunking_by_semantic 等）
- _simplicial_complex: 单纯复形数据结构（HeterogeneousSimplicialComplex 等）
- _retriever: 检索引擎（SimplicialRAGRetriever, compute_semantic_similarity）
- _extraction: 实体/关系提取（extract_entities 等）
- _verification: 语义验证（semantic_verification）
- _retrieval: 检索入口与上下文组装（topology_retrieval 等）
- _generation: 思维链回答生成（generate_response, extract_answer_from_generation）

所有公开 API 均从本模块重新导出，保持向后兼容。
"""

from ._config import (
    DualDimensionConfig,
    AdaptiveThreshold,
    EMB_MODEL,
    EMB_API_KEY,
    EMB_BASE_URL,
    SentenceTransformer,
    cosine_similarity,
    model,
    semantic_similarity,
    normalize_entity_name,
)

from ._chunking import (
    chunking_by_token_size,
    chunking_by_semantic,
    split_text_to_sentences,
    semantic_chunking,
)

from ._simplicial_complex import (
    HeterogeneousSimplicialComplex,
    get_simplex_entities,
    calculate_simplex_score,
)

from ._retriever import (
    SimplicialRAGRetriever,
    compute_semantic_similarity,
)

from ._extraction import (
    parse_entity_array_robust,
    _batch_handle_entity_summaries,
    _batch_handle_relation_summaries,
    _handle_single_entity_extraction,
    _handle_single_msg_extraction,
    build_information_layer,
    build_entity_coboundary,
    build_msg_boundary,
    build_bipartite_laplacian,
    extract_entities,
)

from ._verification import (
    semantic_verification,
    _calculate_candidate_quality,
    _pre_verify_candidate,
)

from ._retrieval import (
    topology_retrieval,
    combine_contexts,
    remove_after_sources,
)

from ._generation import (
    generate_response,
    extract_answer_from_generation,
)

__all__ = [
    # ============ _config: 配置与常量 ============
    "DualDimensionConfig",      # 双维度检索配置类
    "AdaptiveThreshold",        # 自适应阈值计算器
    "EMB_MODEL",                # 嵌入模型名称
    "EMB_API_KEY",              # 嵌入API密钥
    "EMB_BASE_URL",             # 嵌入API基础URL
    "SentenceTransformer",       # SentenceTransformer模型
    "cosine_similarity",        # 余弦相似度函数
    "model",                    # 全局语义模型实例
    "semantic_similarity",      # 语义相似度计算
    "normalize_entity_name",    # 统一实体名称标准化函数

    # ============ _chunking: 文本分块 ============
    "chunking_by_token_size",    # 按token大小分块
    "chunking_by_semantic",     # 基于语义相似度的文本分块
    "split_text_to_sentences",   # 按句子分割文本
    "semantic_chunking",        # 语义分块核心逻辑

    # ============ _simplicial_complex: 单纯复形数据结构 ============
    "HeterogeneousSimplicialComplex",  # 异质单纯复形类（拓扑算子引擎）
    "get_simplex_entities",      # 统一获取复形实体列表
    "calculate_simplex_score",  # 统一计算复形得分

    # ============ _retriever: 检索引擎 ============
    "SimplicialRAGRetriever",   # 双维度并行检索器（拓扑结构维度 + 语义内容维度）
    "compute_semantic_similarity",  # 计算查询与复形文本的语义相似度

    # ============ _extraction: 实体/关系提取 ============
    "parse_entity_array_robust",  # 智能解析实体数组字符串
    "_batch_handle_entity_summaries",   # 批量处理实体描述摘要
    "_batch_handle_relation_summaries",  # 批量处理关系描述和关键词摘要
    "_handle_single_entity_extraction",  # 处理单条实体提取
    "_handle_single_msg_extraction",    # 处理单条MSG提取
    "build_information_layer",          # 构建信息层（MSG→极大单纯形）
    "build_entity_coboundary",          # 构建实体coboundary
    "build_msg_boundary",               # 构建MSG boundary
    "build_bipartite_laplacian",        # 计算二部图简化Laplacian
    "extract_entities",         # 核心实体/MSG提取函数

    # ============ _verification: 语义验证 ============
    "semantic_verification",     # 语义验证与状态标注（filled/void判定）
    "_calculate_candidate_quality",   # 计算候选者质量分数
    "_pre_verify_candidate",     # 快速预验证候选者

    # ============ _retrieval: 检索入口与上下文组装 ============
    "topology_retrieval",        # 拓扑检索（基于SimplicialRAGRetriever）
    "combine_contexts",         # 合并高层/底层上下文（去重）
    "remove_after_sources",     # 删除字符串中Sources之后的内容

    # ============ _generation: 思维链回答生成 ============
    "generate_response",              # 基于知识和问题的思维链（CoT）回答生成
    "extract_answer_from_generation", # 从CoT生成结果中提取<answer>标签内的最终答案
]
