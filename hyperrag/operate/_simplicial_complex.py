import asyncio
from collections import defaultdict
from itertools import combinations
from typing import Union, List, Dict, Set

import numpy as np
import scipy.sparse as sp

from ..utils import logger
from ._config import DualDimensionConfig


def get_simplex_entities(simplex_data: dict) -> list:
    """统一获取复形的实体列表

    优先使用 entities 字段，如果为空则使用 nodes 字段，确保数据结构一致性。

    Args:
        simplex_data: 复形数据字典

    Returns:
        list: 实体列表
    """
    entities = simplex_data.get('entities', [])
    if entities:
        return entities
    return simplex_data.get('nodes', [])


def calculate_simplex_score(
    simplex_data: dict,
    vertex_ids: list = None,
    relation_to_simplices: dict = None,
    score_type: str = 'A'
) -> float:
    """基于Hodge理论的复形得分计算

    得分公式基于单纯复形的拓扑性质：
    - 维度贡献：高维复形包含更丰富的拓扑信息，得分应更高
    - 重要性权重：来自LLM的结构重要性评估
    - 覆盖度/匹配度：与查询的相关程度

    维度A（语义点火）得分 = importance * (1 + dim_weight) * coverage_ratio
    当coverage_ratio=0时，维度A得分大幅衰减，避免噪声污染融合结果。

    维度B（结构模式）得分 = importance * (1 + dim_weight) * match_ratio
    当match_ratio=0时，维度B得分大幅衰减。

    其中 dim_weight = log(1 + dim)，使用对数缩放避免高维复形得分过度膨胀

    Args:
        simplex_data: 复形数据字典
        vertex_ids: 查询顶点ID列表（用于维度A得分计算）
        relation_to_simplices: 关系到复形的映射（用于维度B得分计算）
        score_type: 得分类型，'A' 表示维度A，'B' 表示维度B

    Returns:
        float: 复形得分，范围[0, +inf)
    """
    dim = simplex_data.get('dimension', 0)
    importance = simplex_data.get('importance', 1.0)
    dim_weight = np.log1p(dim) + 0.5

    if score_type == 'A':
        simplex_nodes = get_simplex_entities(simplex_data)
        if vertex_ids and simplex_nodes:
            covered = len(set(simplex_nodes) & set(vertex_ids))
            coverage_ratio = covered / len(vertex_ids)
        else:
            coverage_ratio = 0.0
        if coverage_ratio == 0:
            return importance * dim_weight * 0.1
        return importance * (1 + dim_weight) * (1 + coverage_ratio)
    else:
        if relation_to_simplices:
            matched = len(relation_to_simplices.get(simplex_data.get('simplex_id'), set()))
            simplex_nodes = get_simplex_entities(simplex_data)
            match_ratio = matched / max(len(simplex_nodes), 1)
        else:
            match_ratio = 0.0
        if match_ratio == 0:
            return importance * dim_weight * 0.1
        return importance * (1 + dim_weight) * (1 + match_ratio)


class HeterogeneousSimplicialComplex:
    """异质单纯复形核心数据结构与拓扑算子引擎

    为 RAG 系统提供抗幻觉的高阶上下文补全。
    基于离散外微分理论，实现 Hodge Laplacian 算子和拓扑扩散。

    版本管理机制：通过版本号跟踪数据变更，
    确保Laplacian缓存和嵌入缓存与数据保持一致。
    """
    def __init__(self):
        self.nodes = {}
        self.simplices = {}
        self.B_matrices = {}
        self.L_matrices = {}
        self._data_version = 0
        self._laplacian_version = -1

    @property
    def data_version(self):
        """当前数据版本号"""
        return self._data_version

    def invalidate_data(self):
        """标记数据已变更，递增版本号

        在文档插入/更新/删除后调用，使Laplacian缓存失效。
        """
        self._data_version += 1
        logger.info(f"HSC数据版本更新: {self._data_version}")

    def needs_laplacian_rebuild(self):
        """检查Laplacian矩阵是否需要重建"""
        return self._laplacian_version != self._data_version

    def mark_laplacian_current(self):
        """标记Laplacian矩阵与当前数据版本一致"""
        self._laplacian_version = self._data_version

    def detect_max_dimension(self, simplices=None):
        """检测复形的最高维度"""
        target_simplices = simplices if simplices else self.simplices
        max_dim = 0
        for simplex_id, simplex_data in target_simplices.items():
            dim = simplex_data.get('dimension', 0)
            if dim > max_dim:
                max_dim = dim
        return max_dim

    def _generate_lower_faces(self, nodes, target_dim):
        """生成给定节点集合的所有 target_dim 维面"""
        return list(combinations(nodes, target_dim + 1))

    def build_dynamic_incidence_matrices(self, query_simplices=None):
        """根据查询到的复形动态构建 B 矩阵（关联矩阵）

        关联矩阵 Bk 描述了 k-1 维复形与 k 维复形之间的边界关系。
        Bk[i,j] = 1 表示第 i 个 (k-1)-单纯形是第 j 个 k-单纯形的面。

        Args:
            query_simplices: 查询到的复形子集，None则使用全量复形
        """
        target_simplices = query_simplices if query_simplices else self.simplices
        self._dynamic_simplices = target_simplices
        max_dim = self.detect_max_dimension(target_simplices)

        simplices_by_dim = defaultdict(list)
        for simplex_id, simplex_data in target_simplices.items():
            dim = simplex_data.get('dimension', 0)
            simplices_by_dim[dim].append((simplex_id, simplex_data))

        self.B_matrices = {}

        for dim in range(1, max_dim + 1):
            lower_simplices = simplices_by_dim.get(dim - 1, [])
            higher_simplices = simplices_by_dim.get(dim, [])

            if not lower_simplices or not higher_simplices:
                self.B_matrices[dim] = sp.csr_matrix((0, 0))
                continue

            lower_to_idx = {}
            for i, (simplex_id, _) in enumerate(lower_simplices):
                lower_to_idx[simplex_id] = i

            lower_entity_map = {}
            for lower_id, lower_data in lower_simplices:
                lower_entities = lower_data.get('entities', [])
                lower_key = tuple(sorted(lower_entities))
                lower_entity_map[lower_key] = lower_id

            B_data = []
            B_row = []
            B_col = []

            for higher_idx, (higher_id, higher_data) in enumerate(higher_simplices):
                boundary = higher_data.get('boundary', [])

                for lower_id in boundary:
                    if lower_id in lower_to_idx:
                        B_row.append(lower_to_idx[lower_id])
                        B_col.append(higher_idx)
                        B_data.append(1)

            shape = (len(lower_simplices), len(higher_simplices))
            if B_data:
                self.B_matrices[dim] = sp.csr_matrix((B_data, (B_row, B_col)), shape=shape)
            else:
                self.B_matrices[dim] = sp.csr_matrix(shape)

    def compute_dynamic_hodge_laplacians(self):
        """根据动态构建的 B 矩阵计算 Hodge Laplacian 矩阵

        Hodge Laplacian 定义：Lk = Bk^T @ Bk + Bk+1 @ Bk+1^T
        L0 = B1 @ B1.T （图拉普拉斯）
        Lk (k>0) = Bk^T @ Bk + Bk+1 @ Bk+1^T
        """
        self.L_matrices = {}

        max_dim = max(self.B_matrices.keys()) if self.B_matrices else 0

        for dim in range(max_dim + 1):
            if dim == 0:
                if 1 in self.B_matrices:
                    B1 = self.B_matrices[1]
                    if B1.shape[0] > 0 and B1.shape[1] > 0:
                        self.L_matrices[0] = B1 @ B1.T
                    else:
                        self.L_matrices[0] = sp.csr_matrix((0, 0))
                else:
                    self.L_matrices[0] = sp.csr_matrix((0, 0))
            else:
                L_part1 = sp.csr_matrix((0, 0))
                if dim in self.B_matrices:
                    Bk = self.B_matrices[dim]
                    if Bk.shape[0] > 0 and Bk.shape[1] > 0:
                        L_part1 = Bk.T @ Bk

                L_part2 = sp.csr_matrix((0, 0))
                if dim + 1 in self.B_matrices:
                    Bk_plus_1 = self.B_matrices[dim + 1]
                    if Bk_plus_1.shape[0] > 0 and Bk_plus_1.shape[1] > 0:
                        L_part2 = Bk_plus_1 @ Bk_plus_1.T

                has_valid_L_part1 = L_part1.shape[0] > 0 and L_part1.shape[1] > 0
                has_valid_L_part2 = L_part2.shape[0] > 0 and L_part2.shape[1] > 0

                if has_valid_L_part1 and has_valid_L_part2:
                    if L_part1.shape != L_part2.shape:
                        logger.warning(
                            f"Laplacian shape mismatch at dim={dim}: "
                            f"L_part1={L_part1.shape}, L_part2={L_part2.shape}. "
                            f"Using only the valid part."
                        )
                        if L_part1.shape[0] == L_part1.shape[1]:
                            self.L_matrices[dim] = L_part1
                        elif L_part2.shape[0] == L_part2.shape[1]:
                            self.L_matrices[dim] = L_part2
                        else:
                            self.L_matrices[dim] = sp.csr_matrix((0, 0))
                    else:
                        self.L_matrices[dim] = (L_part1 + L_part2).tocsr()
                elif has_valid_L_part1:
                    self.L_matrices[dim] = L_part1
                elif has_valid_L_part2:
                    self.L_matrices[dim] = L_part2
                else:
                    self.L_matrices[dim] = sp.csr_matrix((0, 0))

        self.mark_laplacian_current()

    def dynamic_diffusion(self, seed_simplices: List[Union[str, int]], dim: int, steps: int = None, alpha: float = None, k_hop: int = None) -> Dict[Union[str, int], float]:
        """拓扑能量扩散 - 基于Hodge Laplacian的热扩散方程

        核心公式：x(t+1) = x(t) - alpha * L @ x(t)

        修复：不再每步归一化，仅在最终结果上做归一化。
        原实现每步归一化会导致能量信息丢失，使得扩散效果退化为
        简单的邻域平均，丧失了Hodge Laplacian的拓扑过滤能力。

        Args:
            seed_simplices: 种子复形ID列表
            dim: 扩散维度
            steps: 扩散步数（None则自适应计算）
            alpha: 扩散率（None则自适应计算）
            k_hop: k-hop邻域范围

        Returns:
            复形ID到扩散得分的映射
        """
        if dim not in self.L_matrices or self.L_matrices[dim] is None or self.L_matrices[dim].shape[0] == 0:
            if dim == 0 and seed_simplices:
                logger.info(f"Laplacian matrix L0 not found or empty, returning seed nodes directly")
                return {node_id: 1.0 for node_id in seed_simplices}
            logger.warning(f"Laplacian matrix L{dim} not found or empty, skipping diffusion")
            return {}

        source_simplices = getattr(self, '_dynamic_simplices', self.simplices)

        simplices_by_dim = defaultdict(list)
        for simplex_id, simplex_data in source_simplices.items():
            simplex_dim = simplex_data.get('dimension', 0)
            simplices_by_dim[simplex_dim].append((simplex_id, simplex_data))

        target_simplices = simplices_by_dim.get(dim, [])
        simplex_count = len(target_simplices)

        if steps is None or alpha is None:
            params = DualDimensionConfig.get_diffusion_params(dim, simplex_count)
            if steps is None:
                steps = params["steps"]
            if alpha is None:
                alpha = params["alpha"]

        # 基于Gershgorin圆定理估计Laplacian矩阵的谱半径上界
        # 确保扩散率 alpha < 2/lambda_max，防止扩散发散
        # 随机采样估计避免全量计算，兼顾精度和效率
        try:
            L_full = self.L_matrices[dim]
            n_rows = L_full.shape[0]
            sample_size = min(n_rows, max(50, n_rows // 10))
            sample_indices = np.random.choice(n_rows, size=sample_size, replace=False) if n_rows > sample_size else range(n_rows)
            max_row_sum = 0.0
            for i in sample_indices:
                row = L_full.getrow(i).toarray()[0]
                row_sum = np.sum(np.abs(row))
                max_row_sum = max(max_row_sum, row_sum)
            if max_row_sum > 0:
                safe_alpha = 1.5 / max_row_sum
                alpha = min(alpha, safe_alpha)
                alpha = max(0.01, alpha)
        except Exception:
            pass

        if k_hop is None:
            k_hop = DualDimensionConfig.DIFFUSION_K_HOP

        simplex_to_idx = {simplex_id: i for i, (simplex_id, _) in enumerate(target_simplices)}

        def get_k_hop_neighbors(seed_ids, k):
            """获取种子复形的k-hop邻域"""
            neighbors = set(seed_ids)
            current = set(seed_ids)

            for _ in range(k):
                next_level = set()
                for simplex_id in current:
                    simplex_data = self.simplices.get(simplex_id, {})
                    boundary = simplex_data.get('boundary', [])
                    coboundary = simplex_data.get('coboundary', [])

                    if dim == 0:
                        for edge_id in coboundary:
                            edge_data = self.simplices.get(edge_id, {})
                            if edge_data.get('dimension', 0) == 1:
                                edge_nodes = edge_data.get('nodes', edge_data.get('entities', []))
                                for node_id in edge_nodes:
                                    if node_id != simplex_id and node_id in self.simplices:
                                        node_dim = self.simplices[node_id].get('dimension', 0)
                                        if node_dim == 0:
                                            next_level.add(node_id)
                    else:
                        for neighbor_id in boundary + coboundary:
                            neighbor_data = self.simplices.get(neighbor_id, {})
                            if neighbor_data.get('dimension', 0) == dim:
                                next_level.add(neighbor_id)

                if not next_level:
                    break

                neighbors.update(next_level)
                current = next_level

            return neighbors

        if seed_simplices:
            k_hop_neighbors = get_k_hop_neighbors(seed_simplices, k_hop)
            k_hop_neighbors = [n for n in k_hop_neighbors if n in simplex_to_idx]
            if not k_hop_neighbors:
                k_hop_neighbors = [s for s in seed_simplices if s in simplex_to_idx]
        else:
            k_hop_neighbors = [simplex_id for simplex_id, _ in target_simplices]

        if not k_hop_neighbors:
            logger.warning(f"No k-hop neighbors found for seed simplices in dimension {dim}")
            return {}

        neighbor_to_idx = {simplex_id: i for i, simplex_id in enumerate(k_hop_neighbors)}
        idx_to_neighbor = {i: simplex_id for i, simplex_id in enumerate(k_hop_neighbors)}

        L_full = self.L_matrices[dim]
        n_neighbors = len(k_hop_neighbors)

        L_neighbor = np.zeros((n_neighbors, n_neighbors))
        for i, simplex_id in enumerate(k_hop_neighbors):
            if simplex_id in simplex_to_idx:
                full_idx = simplex_to_idx[simplex_id]
                row = L_full.getrow(full_idx).toarray()[0]
                for j, neighbor_id in enumerate(k_hop_neighbors):
                    if neighbor_id in simplex_to_idx:
                        full_jdx = simplex_to_idx[neighbor_id]
                        L_neighbor[i, j] = row[full_jdx]

        L_neighbor = sp.csr_matrix(L_neighbor)

        x = np.zeros(n_neighbors)

        if seed_simplices:
            seed_count = 0
            for simplex_id in seed_simplices:
                if simplex_id in neighbor_to_idx:
                    x[neighbor_to_idx[simplex_id]] = 1.0
                    seed_count += 1

            if seed_count == 0:
                logger.warning(f"No seed simplices found in k-hop neighborhood. Using uniform initialization as fallback.")
                x = np.ones(n_neighbors) / n_neighbors
        else:
            logger.warning(f"No seed simplices provided, using uniform initialization")
            x = np.ones(n_neighbors) / n_neighbors

        if L_neighbor.shape[0] != L_neighbor.shape[1]:
            logger.warning(f"Neighborhood Laplacian matrix is not square: {L_neighbor.shape}, skipping diffusion")
            return {}
        if L_neighbor.shape[0] != len(x):
            logger.warning(f"Shape mismatch: L has {L_neighbor.shape[0]} rows but x has {len(x)} elements, skipping diffusion")
            return {}

        for step in range(steps):
            try:
                delta = L_neighbor @ x
                delta_norm = float(np.linalg.norm(delta))
                x_norm = float(np.linalg.norm(x))
                if delta_norm > 0 and x_norm > 0:
                    effective_alpha = min(alpha, 0.5 * x_norm / delta_norm)
                else:
                    effective_alpha = alpha
                x = x - effective_alpha * delta
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    logger.warning(f"数值不稳定出现在步骤 {step}，停止扩散")
                    break
            except Exception as e:
                logger.error(f"Error during diffusion: {e}")
                return {}

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        elif x_max > 0:
            x = np.ones_like(x)

        score_threshold = DualDimensionConfig.DIFFUSION_SCORE_THRESHOLD
        scores = {}
        filtered_count = 0
        for idx, score in enumerate(x):
            if idx in idx_to_neighbor:
                if score >= score_threshold:
                    scores[idx_to_neighbor[idx]] = float(score)
                else:
                    filtered_count += 1

        if filtered_count > 0:
            logger.info(f"Diffusion completed for dimension {dim}, obtained scores for {len(scores)} simplices "
                        f"in {k_hop}-hop neighborhood (filtered {filtered_count} low-score simplices < {score_threshold})")
        else:
            logger.info(f"Diffusion completed for dimension {dim}, obtained scores for {len(scores)} simplices in {k_hop}-hop neighborhood")
        return scores

    def build_incidence_matrices(self):
        """构建关联矩阵（全量版本，用于缓存）

        支持任意维度的关联矩阵构建，与 build_dynamic_incidence_matrices 逻辑一致。
        同时保留 B1/B2 属性以兼容旧代码。
        """
        simplices_by_dim = defaultdict(list)
        for simplex_id, simplex_data in self.simplices.items():
            dim = simplex_data.get('dimension', 0)
            simplices_by_dim[dim].append((simplex_id, simplex_data))

        max_dim = max(simplices_by_dim.keys()) if simplices_by_dim else 0
        self.B_matrices = {}

        for dim in range(1, max_dim + 1):
            lower_simplices = simplices_by_dim.get(dim - 1, [])
            higher_simplices = simplices_by_dim.get(dim, [])

            if not lower_simplices or not higher_simplices:
                self.B_matrices[dim] = sp.csr_matrix((0, 0))
                continue

            lower_to_idx = {sid: i for i, (sid, _) in enumerate(lower_simplices)}

            lower_entity_map = {}
            for lower_id, lower_data in lower_simplices:
                lower_entities = lower_data.get('entities', lower_data.get('nodes', []))
                lower_key = tuple(sorted(lower_entities))
                lower_entity_map[lower_key] = lower_id

            B_data = []
            B_row = []
            B_col = []

            for higher_idx, (higher_id, higher_data) in enumerate(higher_simplices):
                boundary = higher_data.get('boundary', [])
                for lower_id in boundary:
                    if lower_id in lower_to_idx:
                        B_row.append(lower_to_idx[lower_id])
                        B_col.append(higher_idx)
                        B_data.append(1)

            shape = (len(lower_simplices), len(higher_simplices))
            if B_data:
                self.B_matrices[dim] = sp.csr_matrix((B_data, (B_row, B_col)), shape=shape)
            else:
                self.B_matrices[dim] = sp.csr_matrix(shape)

        self.B1 = self.B_matrices.get(1, sp.csr_matrix((0, 0)))
        self.B2 = self.B_matrices.get(2, sp.csr_matrix((0, 0)))

    def compute_hodge_laplacians(self):
        """根据离散外微分理论计算 Hodge Laplacian 矩阵（全量版本，用于缓存）

        支持任意维度的 Hodge Laplacian 计算，与 compute_dynamic_hodge_laplacians 逻辑一致。
        Lk = Bk^T @ Bk + Bk+1 @ Bk+1^T
        同时保留 L0/L1 属性以兼容旧代码。
        """
        self.L_matrices = {}

        max_dim = max(self.B_matrices.keys()) if self.B_matrices else 0

        for dim in range(max_dim + 1):
            if dim == 0:
                if 1 in self.B_matrices:
                    B1 = self.B_matrices[1]
                    if B1.shape[0] > 0 and B1.shape[1] > 0:
                        self.L_matrices[0] = B1 @ B1.T
                    else:
                        self.L_matrices[0] = sp.csr_matrix((0, 0))
                else:
                    self.L_matrices[0] = sp.csr_matrix((0, 0))
            else:
                Bk = self.B_matrices.get(dim)
                Bk1 = self.B_matrices.get(dim + 1)

                L_part1 = sp.csr_matrix((0, 0))
                if Bk is not None and Bk.shape[0] > 0 and Bk.shape[1] > 0:
                    L_part1 = Bk.T @ Bk

                L_part2 = sp.csr_matrix((0, 0))
                if Bk1 is not None and Bk1.shape[0] > 0 and Bk1.shape[1] > 0:
                    L_part2 = Bk1 @ Bk1.T

                if L_part1.shape[0] > 0 and L_part2.shape[0] > 0:
                    self.L_matrices[dim] = L_part1 + L_part2
                elif L_part1.shape[0] > 0:
                    self.L_matrices[dim] = L_part1
                elif L_part2.shape[0] > 0:
                    self.L_matrices[dim] = L_part2
                else:
                    self.L_matrices[dim] = sp.csr_matrix((0, 0))

        self.L0 = self.L_matrices.get(0, sp.csr_matrix((0, 0)))
        self.L1 = self.L_matrices.get(1, sp.csr_matrix((0, 0)))

    def get_upper_adjacent(self, simplex_ids: List[int], current_dim: int) -> Set[int]:
        """共面收缩 (Coboundary Lookup)"""
        upper_adjacent = set()

        if current_dim == 0:
            node_coboundaries = []
            for node_id in simplex_ids:
                if node_id in self.simplices:
                    node_data = self.simplices[node_id]
                    node_coboundary = node_data.get('coboundary', [])
                    node_coboundaries.append(set(node_coboundary))
                else:
                    node_coboundary = set()
                    for simplex_id, simplex_data in self.simplices.items():
                        dim = simplex_data.get('dimension', 0)
                        if dim > 0:
                            simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))
                            if node_id in simplex_nodes:
                                node_coboundary.add(simplex_id)
                    node_coboundaries.append(node_coboundary)

            if node_coboundaries:
                upper_adjacent = set.intersection(*node_coboundaries)
        elif current_dim == 1:
            edge_coboundaries = []
            for edge_id in simplex_ids:
                if edge_id in self.simplices:
                    edge_data = self.simplices[edge_id]
                    edge_coboundary = edge_data.get('coboundary', [])
                    edge_coboundaries.append(set(edge_coboundary))
                else:
                    edge_coboundary = set()
                    for simplex_id, simplex_data in self.simplices.items():
                        dim = simplex_data.get('dimension', 0)
                        if dim > 1:
                            boundary = simplex_data.get('boundary', [])
                            if edge_id in boundary:
                                edge_coboundary.add(simplex_id)
                    edge_coboundaries.append(edge_coboundary)

            if edge_coboundaries:
                upper_adjacent = set.intersection(*edge_coboundaries)
        else:
            if len(simplex_ids) == 1:
                simplex_id = simplex_ids[0]
                if simplex_id in self.simplices:
                    simplex_data = self.simplices[simplex_id]
                    upper_adjacent = set(simplex_data.get('coboundary', []))

        return upper_adjacent

    def get_lower_adjacent(self, simplex_id: int, current_dim: int) -> Set[int]:
        """边界展开 (Boundary Lookup)"""
        lower_adjacent = set()

        if simplex_id not in self.simplices:
            return lower_adjacent

        simplex_data = self.simplices[simplex_id]

        boundary = simplex_data.get('boundary', [])
        if boundary:
            lower_adjacent.update(boundary)
        else:
            nodes = simplex_data.get('nodes', simplex_data.get('entities', []))

            for node in nodes:
                lower_adjacent.add(node)

            if current_dim >= 2:
                edge_node_map = {}
                for edge_id, edge_data in self.simplices.items():
                    edge_dim = edge_data.get('dimension', 0)
                    if edge_dim == 1:
                        edge_nodes = edge_data.get('nodes', edge_data.get('entities', []))
                        edge_key = frozenset(edge_nodes)
                        edge_node_map[edge_key] = edge_id

                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        edge_key = frozenset([nodes[i], nodes[j]])
                        if edge_key in edge_node_map:
                            lower_adjacent.add(edge_node_map[edge_key])

        return lower_adjacent
