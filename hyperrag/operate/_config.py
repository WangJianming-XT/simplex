import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from ..utils import logger
from ..llm import openai_embedding

try:
    from my_config import EMB_MODEL, EMB_API_KEY, EMB_BASE_URL
except ImportError:
    EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
    EMB_API_KEY = os.environ.get("EMB_API_KEY", "")
    EMB_BASE_URL = os.environ.get("EMB_BASE_URL", "")


class AdaptiveThreshold:
    """自适应阈值计算器 - 基于数据分布动态调整阈值，替代硬编码固定值

    核心思想：阈值应根据实际数据分布自适应调整，而非固定值。
    使用统计方法（均值、标准差、分位数）动态计算阈值，
    确保在不同数据规模和分布下都能保持合理的筛选效果。
    """

    def __init__(self):
        self._stats_cache = {}

    def compute_coverage_threshold(self, vertex_count: int) -> float:
        """根据查询顶点数量自适应计算覆盖阈值

        顶点越多，要求每个复形覆盖的比例可以适当降低，
        因为多顶点同时被覆盖的概率本身就低。
        使用对数衰减函数，确保阈值在合理范围内平滑变化。

        Args:
            vertex_count: 查询顶点数量

        Returns:
            自适应覆盖阈值
        """
        if vertex_count <= 1:
            return 1.0
        if vertex_count == 2:
            return 0.5
        return max(0.3, 1.0 / np.log2(vertex_count + 1))

    def compute_semantic_threshold(self, similarity_scores: Optional[list] = None) -> float:
        """自适应语义相似度阈值

        如果提供了相似度分数分布，则基于均值减一个标准差计算阈值；
        否则使用经验默认值。

        Args:
            similarity_scores: 候选复形的语义相似度分数列表

        Returns:
            自适应语义相似度阈值
        """
        if similarity_scores and len(similarity_scores) >= 5:
            scores = np.array(similarity_scores)
            threshold = float(np.mean(scores) - 0.5 * np.std(scores))
            return max(0.2, min(0.8, threshold))
        return 0.35

    def compute_keep_count(self, total: int, mode: str = "multi") -> int:
        """自适应保留数量

        基于候选总数动态计算保留数量，使用平方根缩放避免
        候选集过大时保留过多无关结果。

        Args:
            total: 候选总数
            mode: "single"单顶点模式或"multi"多顶点模式

        Returns:
            自适应保留数量
        """
        if total <= 0:
            return 0
        if mode == "single":
            return min(int(np.sqrt(total) * 3), total)
        return min(int(np.sqrt(total) * 5), total)

    def compute_diffusion_params(self, dim: int, simplex_count: int) -> Dict:
        """自适应扩散参数

        根据复形维度和数据规模动态调整扩散步数和扩散率。
        数据规模越大，需要更多步数和更小的扩散率以确保稳定性。
        优化后：增大扩散步数范围，使拓扑信息能传播到更远的相关复形；
        自适应扩散率与查询复杂度成反比，复杂查询需更精细的扩散。

        Args:
            dim: 复形维度
            simplex_count: 该维度的复形数量

        Returns:
            包含steps和alpha的字典
        """
        base_steps = max(3, min(6, int(np.log2(simplex_count + 1))))
        base_alpha = max(0.05, min(0.15, 0.12 / max(dim + 1, 1)))

        if simplex_count > 500:
            base_steps = max(4, min(8, base_steps + 1))
            base_alpha = max(0.03, base_alpha * 0.7)
        elif simplex_count > 100:
            base_alpha = max(0.05, base_alpha * 0.85)

        if dim >= 2:
            base_alpha *= 0.6

        return {"steps": base_steps, "alpha": base_alpha}

    def compute_match_thresholds(self, entity_count: int) -> Dict:
        """自适应匹配阈值

        根据关系中的实体数量动态调整匹配阈值。
        高阶关系（实体数多）更难完全匹配，因此需要更宽松的阈值。

        Args:
            entity_count: 关系中的实体数量

        Returns:
            包含match_ratio和score_threshold的字典
        """
        if entity_count <= 1:
            return {"match_ratio": 1.0, "score_threshold": 50}
        if entity_count == 2:
            return {"match_ratio": 0.5, "score_threshold": 40}
        return {"match_ratio": max(0.2, 1.0 / entity_count), "score_threshold": 30}


class DualDimensionConfig:
    """并行双维度检索配置类

    架构原则：
    1. 拓扑维度和语义维度独立并行检索，互不依赖
    2. 两个维度各自产生候选集和得分，通过阈值融合合并
    3. 对撞复形（两维度都支持）获得协同加分
    4. 单维度支持的复形根据另一维度质量决定保留门槛
    5. 核心参数使用自适应计算，而非固定值
    """

    _adaptive = AdaptiveThreshold()

    ENTITY_MATCH_LEVENSHTEIN_THRESHOLD = 0.85
    MIN_SUBSTRING_MATCH_LENGTH = 3

    MAX_COBOUNDARY_PER_VERTEX = 50
    MAX_COMPLETION_RESULTS = 200
    DIFFUSION_K_HOP = 1

    DIMENSION_A_WEIGHT = 0.5
    DIMENSION_B_WEIGHT = 0.5

    DIMENSION_WEIGHT_MIN = 0.2
    DIMENSION_WEIGHT_MAX = 0.8

    MAX_CONTEXT_TOKENS = 60000

    CHUNK_BUDGET_TOTAL = 60
    CHUNK_BUDGET_MIN = 5
    CHUNK_BUDGET_STRUCTURE_PER_ENTITY = 3
    CHUNK_BUDGET_STRUCTURE_PER_RELATION = 5

    SEMANTIC_RERANK_WEIGHT = 0.6
    TOPO_RERANK_WEIGHT = 0.4

    TOPO_DIMENSION_WEIGHT = 0.6
    SEMANTIC_DIMENSION_WEIGHT = 0.4
    SEMANTIC_RETRIEVE_TOP_K = 100
    SEMANTIC_MIN_SIMILARITY = 0.5
    TOPO_SEM_WEIGHT_MIN = 0.2
    TOPO_SEM_WEIGHT_MAX = 0.8

    ENTITY_MATCH_VIRTUAL_CONFIDENCE = 0.1
    ENTITY_MATCH_SEMANTIC_VERIFY_THRESHOLD = 0.6
    VIRTUAL_NODE_DIFFUSION_EXCLUDE = True
    FUZZY_MATCH_MIN_RATIO = 0.35
    VIRTUAL_RATIO_FALLBACK_THRESHOLD = 0.3

    PARALLEL_TOPO_TOP_K = 100
    PARALLEL_SEM_TOP_K = 50
    PARALLEL_COLLISION_BOOST = 1.2
    PARALLEL_TOPO_ONLY_DECAY = 0.6
    PARALLEL_SEM_ONLY_DECAY = 0.7
    PARALLEL_MIN_PER_CATEGORY = 2
    PARALLEL_ENTITY_SEM_TOP_K = 10
    PARALLEL_ENTITY_TOPO_TOP_K = 10

    COBOUNDARY_EXPAND_MAX_DEPTH_ENTITY = 1
    COBOUNDARY_EXPAND_MAX_DEPTH_RELATION = 1
    COBOUNDARY_EXPAND_DECAY = 0.4
    DIFFUSION_SCORE_THRESHOLD = 0.15
    VDB_ONE_HOP_COBOUNDARY_LIMIT = 1

    VDB_DIRECT_GUARANTEE_COUNT = 4
    VDB_SCORE_STRICT_THRESHOLD = 0.7
    QUALITY_THRESHOLD_RATIO = 0.3
    MIN_CHUNKS_GUARANTEE = 6
    MAX_CHUNKS_SOFT_LIMIT = 20
    ENTITY_PROXY_TOP_K = 3
    TOPO_ONLY_DYNAMIC_DECAY_MIN = 0.2
    SEM_ONLY_DYNAMIC_BOOST_MAX = 1.0

    SOURCES_TOKEN_BUDGET = 8000
    STRUCTURE_TOKEN_BUDGET = 4000
    MAX_STRUCTURE_ENTITY_COUNT = 15
    MAX_STRUCTURE_SIMPLEX_COUNT = 10

    # MSG相关配置参数
    MSG_COMPLETENESS_THRESHOLD = 0.5
    MSG_ENTITY_MIN_COUNT = 2
    ENTITY_IMPORTANCE_THRESHOLD = 0.02

    # 二部图Laplacian扩散参数
    TOPOLOGY_DIFFUSION_BETA = 0.15
    TOPOLOGY_DIFFUSION_STEPS = 3
    TOPOLOGY_DIFFUSION_THRESHOLD = 0.01

    # 2×2检索融合权重
    SEMANTIC_FUSION_ALPHA = 0.5

    @classmethod
    def get_coverage_threshold(cls, vertex_count: int) -> float:
        return cls._adaptive.compute_coverage_threshold(vertex_count)

    @classmethod
    def get_semantic_threshold(cls, scores: Optional[list] = None) -> float:
        return cls._adaptive.compute_semantic_threshold(scores)

    @classmethod
    def get_keep_count(cls, total: int, mode: str = "multi") -> int:
        return cls._adaptive.compute_keep_count(total, mode)

    @classmethod
    def get_diffusion_params(cls, dim: int, simplex_count: int) -> Dict:
        return cls._adaptive.compute_diffusion_params(dim, simplex_count)

    @classmethod
    def get_match_thresholds(cls, entity_count: int) -> Dict:
        return cls._adaptive.compute_match_thresholds(entity_count)

    @classmethod
    def compute_dynamic_fusion_weights(
        cls,
        vertex_quality: float,
        relation_quality: float
    ) -> tuple:
        """根据维度匹配质量动态计算融合权重（拓扑内部维度A/B融合）

        当维度A匹配质量低时自动提升维度B权重，反之亦然。
        权重被约束在[DIMENSION_WEIGHT_MIN, DIMENSION_WEIGHT_MAX]范围内，
        避免完全忽略某一维度。

        Args:
            vertex_quality: 维度A实体匹配质量（0~1）
            relation_quality: 维度B关系匹配质量（0~1）

        Returns:
            (weight_a, weight_b) 融合权重元组
            当双维度均失效时返回 (0, 0) 作为回退信号
        """
        total_quality = vertex_quality + relation_quality
        if total_quality > 0:
            weight_a = vertex_quality / total_quality
            weight_b = relation_quality / total_quality
        else:
            return (0.0, 0.0)
        weight_a = max(cls.DIMENSION_WEIGHT_MIN, min(cls.DIMENSION_WEIGHT_MAX, weight_a))
        weight_b = 1.0 - weight_a
        return weight_a, weight_b

    @classmethod
    def compute_topology_semantic_weights(
        cls,
        topology_quality: float,
        semantic_quality: float
    ) -> tuple:
        """根据拓扑和语义维度的匹配质量动态计算融合权重

        拓扑维度提供结构连贯性，语义维度提供内容相关性。
        当某一维度匹配质量低时自动提升另一维度权重，
        权重被约束在[TOPO_SEM_WEIGHT_MIN, TOPO_SEM_WEIGHT_MAX]范围内。

        Args:
            topology_quality: 拓扑维度匹配质量（0~1）
            semantic_quality: 语义维度匹配质量（0~1）

        Returns:
            (weight_topo, weight_sem) 融合权重元组
            当双维度均失效时返回 (0.5, 0.5) 作为均衡回退
        """
        total_quality = topology_quality + semantic_quality
        if total_quality > 0:
            weight_topo = topology_quality / total_quality
            weight_sem = semantic_quality / total_quality
        else:
            return (0.5, 0.5)
        weight_topo = max(cls.TOPO_SEM_WEIGHT_MIN,
                          min(cls.TOPO_SEM_WEIGHT_MAX, weight_topo))
        weight_sem = 1.0 - weight_topo
        return weight_topo, weight_sem

    @classmethod
    def compute_chunk_budget(
        cls,
        entity_count: int,
        relation_count: int,
        total_budget: int = None
    ) -> int:
        """根据查询复杂度动态分配文本块预算

        复杂查询（多实体+多关系）需要更多结构信息，简单查询需要更多文本块。
        结构信息预算与查询复杂度成正比，但不超过总预算的15%，
        确保文本块始终占主导地位——LLM回答问题依赖文本内容而非结构信息。
        当total_budget较小时（≤30），不再进一步压缩，直接返回total_budget，
        避免预算过小导致关键文本块丢失。

        Args:
            entity_count: 查询实体数量
            relation_count: 查询关系数量
            total_budget: 总预算（None则使用默认值）

        Returns:
            文本块数量预算
        """
        if total_budget is None:
            total_budget = cls.CHUNK_BUDGET_TOTAL
        if total_budget <= 30:
            return total_budget
        structure_budget = min(
            int(total_budget * 0.15),
            entity_count * cls.CHUNK_BUDGET_STRUCTURE_PER_ENTITY
            + relation_count * cls.CHUNK_BUDGET_STRUCTURE_PER_RELATION
        )
        chunk_budget = max(cls.MIN_CHUNKS_GUARANTEE, total_budget - structure_budget)
        return chunk_budget

    @classmethod
    def compute_dynamic_topo_decay(cls, topo_quality: float) -> float:
        """根据拓扑质量动态计算拓扑独有衰减系数

        拓扑质量越低，衰减越严重，避免低质量拓扑结果占据配额。
        衰减范围：[TOPO_ONLY_DYNAMIC_DECAY_MIN, PARALLEL_TOPO_ONLY_DECAY]

        Args:
            topo_quality: 拓扑维度质量（0~1）

        Returns:
            动态衰减系数
        """
        min_decay = cls.TOPO_ONLY_DYNAMIC_DECAY_MIN
        max_decay = cls.PARALLEL_TOPO_ONLY_DECAY
        return min_decay + (max_decay - min_decay) * min(1.0, topo_quality)

    @classmethod
    def compute_dynamic_sem_boost(cls, sem_quality: float) -> float:
        """根据语义质量动态计算语义独有提升系数

        语义质量越高，语义独有结果的保留比例越大。
        提升范围：[PARALLEL_SEM_ONLY_DECAY, SEM_ONLY_DYNAMIC_BOOST_MAX]

        Args:
            sem_quality: 语义维度质量（0~1）

        Returns:
            动态提升系数
        """
        min_boost = cls.PARALLEL_SEM_ONLY_DECAY
        max_boost = cls.SEM_ONLY_DYNAMIC_BOOST_MAX
        return min_boost + (max_boost - min_boost) * min(1.0, sem_quality)

    @classmethod
    def compute_adaptive_chunk_limit(cls, candidate_count: int, score_distribution: list = None) -> int:
        """根据候选数量和得分分布自适应计算chunk数量上限

        基于质量阈值截断而非硬编码上限：
        - 候选少时全部保留
        - 候选多时根据得分分布动态调整，保留得分高于阈值的chunk
        - 上限不超过MAX_CHUNKS_SOFT_LIMIT

        Args:
            candidate_count: 候选chunk总数
            score_distribution: 候选chunk得分列表（可选，用于质量阈值计算）

        Returns:
            自适应chunk数量上限
        """
        if candidate_count <= cls.MIN_CHUNKS_GUARANTEE:
            return candidate_count

        base_limit = min(cls.MAX_CHUNKS_SOFT_LIMIT, candidate_count)

        if score_distribution and len(score_distribution) >= 5:
            scores = np.array(score_distribution)
            threshold = float(np.mean(scores) - cls.QUALITY_THRESHOLD_RATIO * np.std(scores))
            above_threshold = int(np.sum(scores >= threshold))
            adaptive_limit = max(cls.MIN_CHUNKS_GUARANTEE, min(above_threshold, base_limit))
            return adaptive_limit

        return base_limit


SentenceTransformer = None
cosine_similarity = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logger.warning(f"Failed to import sentence_transformers: {e}")
    logger.warning("Using fallback implementation for semantic similarity")

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

model = None
try:
    if SentenceTransformer:
        model_path = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2"
        if model_path.exists():
            model = SentenceTransformer(str(model_path))
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.warning(f"Failed to load semantic model: {e}")


def semantic_similarity(text1, text2):
    """计算两个文本的语义相似度（替代方案）"""
    if model and cosine_similarity:
        embeddings = model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    else:
        def get_tokens(text):
            return set(re.findall(r'\w+', text.lower()))
        tokens1 = get_tokens(text1)
        tokens2 = get_tokens(text2)
        if not tokens1 and not tokens2:
            return 1.0
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))


def normalize_entity_name(entity_name: str) -> str:
    """统一实体名称标准化函数

    确定性归一化：大写 + 下划线转空格 + 去多余空格 + 常见复数后缀转单数。
    Prompt约束LLM输出单数形式作为主要防线，
    此处归一化处理少量LLM遗漏的复数输出。

    Args:
        entity_name: 原始实体名称

    Returns:
        标准化后的实体名称
    """
    if not entity_name:
        return entity_name
    normalized = str(entity_name).strip().upper()
    normalized = normalized.replace('_', ' ')
    normalized = ' '.join(normalized.split())

    IRREGULAR_PLURALS = {
        'PARTIES': 'PARTY',
        'REPRESENTATIVES': 'REPRESENTATIVE',
        'HOLDERS': 'HOLDER',
        'LENDERS': 'LENDER',
        'BORROWERS': 'BORROWER',
        'CREDITORS': 'CREDITOR',
        'OBLIGATIONS': 'OBLIGATION',
        'INSTRUMENTS': 'INSTRUMENT',
        'AGREEMENTS': 'AGREEMENT',
        'DOCUMENTS': 'DOCUMENT',
        'PROCEEDS': 'PROCEEDS',
        'EARNINGS': 'EARNINGS',
        'ASSETS': 'ASSETS',
        'FUNDS': 'FUNDS',
        'PROPERTIES': 'PROPERTY',
        'SECURITIES': 'SECURITY',
        'GUARANTEES': 'GUARANTEE',
        'FACILITIES': 'FACILITY',
        'WARRANTIES': 'WARRANTY',
        'INDENTURES': 'INDENTURE',
        'NOTICES': 'NOTICE',
        'PAYMENTS': 'PAYMENT',
        'TRANSACTIONS': 'TRANSACTION',
        'CONDITIONS': 'CONDITION',
        'REQUIREMENTS': 'REQUIREMENT',
        'RESTRICTIONS': 'RESTRICTION',
        'PROVISIONS': 'PROVISION',
        'RIGHTS': 'RIGHT',
        'REMEDIES': 'REMEDY',
        'INTERESTS': 'INTEREST',
        'SHARES': 'SHARE',
        'AMOUNTS': 'AMOUNT',
        'EXPENSES': 'EXPENSE',
        'LOSSES': 'LOSS',
        'CLAIMS': 'CLAIM',
        'COSTS': 'COST',
        'FEES': 'FEE',
        'TAXES': 'TAX',
        'CHARGES': 'CHARGE',
        'RATES': 'RATE',
        'DATES': 'DATE',
        'EVENTS': 'EVENT',
        'ACTIONS': 'ACTION',
        'PERSONS': 'PERSON',
        'ENTITIES': 'ENTITY',
        'OFFICERS': 'OFFICER',
        'DIRECTORS': 'DIRECTOR',
        'MEMBERS': 'MEMBER',
        'TRUSTEES': 'TRUSTEE',
        'AGENTS': 'AGENT',
        'MANAGERS': 'MANAGER',
        'PARTNERS': 'PARTNER',
    }

    words = normalized.split()
    singularized = []
    for word in words:
        if word in IRREGULAR_PLURALS:
            singularized.append(IRREGULAR_PLURALS[word])
        elif word.endswith('IES') and len(word) > 4:
            singularized.append(word[:-3] + 'Y')
        elif word.endswith('ES') and len(word) > 3:
            if word.endswith('SES') or word.endswith('XES') or word.endswith('ZES') or word.endswith('SHES') or word.endswith('CHES'):
                singularized.append(word[:-2])
            else:
                singularized.append(word[:-1])
        elif word.endswith('S') and not word.endswith('SS') and not word.endswith('US') and len(word) > 3:
            singularized.append(word[:-1])
        else:
            singularized.append(word)

    return ' '.join(singularized)


LEADING_ARTICLES = frozenset({"THE", "A", "AN"})


def strip_leading_articles(name: str) -> str:
    """移除实体名称开头的英文冠词

    用于实体匹配时消除冠词差异，例如
    "THE BORROWER" 和 "BORROWER" 应视为同一实体。

    Args:
        name: 已归一化（大写）的实体名称

    Returns:
        移除开头冠词后的名称
    """
    words = name.split()
    while words and words[0] in LEADING_ARTICLES:
        words.pop(0)
    return ' '.join(words)


def match_entity_name(query_name: str, candidate_names: set | dict, normalized_map: dict | None = None) -> str | None:
    """在候选实体名称集合中查找与查询名称匹配的实体

    匹配策略按优先级依次尝试：
    1. 精确匹配
    2. 归一化匹配（经 normalize_entity_name 处理后比较）
    3. 去冠词匹配（移除开头 THE/A/AN 后比较）
    4. 去冠词+归一化匹配
    5. 包含关系匹配（查询名是候选名的子串，或候选名是查询名的子串，
       但要求较短名称至少有2个词，避免单词误匹配如 "AGREEMENT" 匹配 "LOAN AGREEMENT"）
    6. 去冠词后的包含关系匹配

    Args:
        query_name: 待匹配的实体名称
        candidate_names: 候选实体名称集合
        normalized_map: 归一化名称到原始名称的映射，如未提供则自动构建

    Returns:
        匹配到的候选原始名称，未匹配返回 None
    """
    if not query_name:
        return None

    if query_name in candidate_names:
        return query_name

    if normalized_map is None:
        normalized_map = {normalize_entity_name(k): k for k in candidate_names}

    norm_query = normalize_entity_name(query_name)
    if norm_query in normalized_map:
        return normalized_map[norm_query]

    stripped_query = strip_leading_articles(norm_query)
    if stripped_query != norm_query:
        for cand in candidate_names:
            if strip_leading_articles(normalize_entity_name(cand)) == stripped_query:
                return cand

    for cand in candidate_names:
        norm_cand = normalize_entity_name(cand)
        stripped_cand = strip_leading_articles(norm_cand)
        if stripped_cand == stripped_query:
            return cand

    query_words = stripped_query.split()
    if len(query_words) >= 2:
        for cand in candidate_names:
            norm_cand = normalize_entity_name(cand)
            stripped_cand = strip_leading_articles(norm_cand)
            cand_words = stripped_cand.split()
            if len(cand_words) >= 2:
                if stripped_query in stripped_cand or stripped_cand in stripped_query:
                    shorter_len = min(len(query_words), len(cand_words))
                    longer_len = max(len(query_words), len(cand_words))
                    if longer_len - shorter_len <= 1:
                        return cand

    return None
