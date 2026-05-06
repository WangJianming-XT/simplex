# Hyper-RAG 双维度检索详细分析

## 目录

1. [概述](#概述)
2. [核心数据结构](#核心数据结构)
3. [双维度定义](#双维度定义)
4. [四步走核心逻辑](#四步走核心逻辑)
5. [复形截断操作](#复形截断操作)
6. [关键算法细节](#关键算法细节)
7. [代码逐行分析](#代码逐行分析)
8. [总结](#总结)

---

## 概述

Hyper-RAG 实现了一个复杂的双维度并行检索系统，核心逻辑在 `SimplicialRAGRetriever` 类的 `dual_dimension_retrieve` 方法中。该系统利用拓扑数学理论（单纯复形、Hodge Laplacian）进行语义检索和逻辑推理。

### 函数签名

```python
def dual_dimension_retrieve(
    self,
    query_vertices: List[dict],           # 查询顶点列表
    query_partial_relations: List[List[int]],  # 查询中的部分关系
    diffusion_steps: int = 2,             # 拓扑扩散的迭代步数
    diffusion_alpha: float = 0.1,         # 扩散率参数
    coboundary_threshold: float = 0.5,    # 上边界匹配的最低阈值
    type_filter: List[str] = None         # 可选的类型过滤列表
) -> dict:
```

---

## 核心数据结构

### HeterogeneousSimplicialComplex（HSC）

```python
class HeterogeneousSimplicialComplex:
    def __init__(self):
        self.nodes = {}        # 存储异质顶点 S_0: {node_id: {'type': 'entity/time/value...', 'vector': [...]}}
        self.simplices = {}    # 存储高阶单纯形 S_k: {simplex_id: {'dimension': k, 'nodes': [...], 'level_hg': 'high'|'lower'}}
        self.B_matrices = {}   # 存储所有维度的 B 矩阵（关联矩阵）
        self.L_matrices = {}   # 存储所有维度的拉普拉斯矩阵
```

**字段说明**：
- `nodes`：存储所有实体顶点（0维单纯形），包含类型和向量
- `simplices`：存储所有高阶单纯形（边、面、三维体等），通过 `dimension` 字段区分维度
- `B_matrices`：关联矩阵，描述不同维度单纯形间的包含关系
- `L_matrices`：Hodge Laplacian 矩阵，用于拓扑能量扩散

### 单纯形的维度分类

| 维度 | 名称 | 示例 | 说明 |
|------|------|------|------|
| 0 | 顶点 (Vertex) | `{"dimension": 0, "entities": ["Stifel Financial"], "type": "Entity"}` | 单个实体 |
| 1 | 边 (Edge) | `{"dimension": 1, "entities": ["Stifel Financial", "Trust"], "description": "公司建立信托"}` | 两个实体的关系 |
| 2 | 面 (Triangle) | `{"dimension": 2, "entities": ["公司", "信托", "证券"], "description": "信托发行证券的完整上下文"}` | 三个实体形成的上下文 |
| 3 | 四面体 (Tetrahedron) | `{"dimension": 3, "entities": ["公司", "信托", "证券", "2024-01-15"], "description": "具体交易事实"}` | 四个实体形成的原子事实 |

---

## 双维度定义

### 维度一：语义点火维度（Semantic Ignition）

**目标**：解决"谁/何时/何地"的定位问题

**方法**：
1. 从查询中提取异质顶点（0-simplices），基于5W1H异质标签进行映射
2. 在S0向量索引中查找最邻近的异质顶点
3. 输出带有激活强度的顶点集合 V_seeds

**具体操作**：
- 使用 LLM 提取查询中的实体和关系（如果启用）
- 解析实体类型（Entity、Temporal、Spatial、Value/State等）
- 提取实体名称和描述

### 维度二：结构模式维度（Structural Pattern）

**目标**：解决"逻辑形状/破碎关系"的匹配问题

**方法**：
1. 提取1-simplex（边）或2-simplex（面）的特征
2. 在Sk（k≥1）索引中查找已知片段
3. 输出被初步锁定的低阶单纯形集合 Σ_patterns

**匹配规则**：
```python
def relation_matches_simplex(relation_entities: list, simplex_nodes: list) -> bool:
    # 单实体关系：必须有精确匹配
    # 多实体关系：至少60%匹配 + 至少1个精确匹配
```

---

## 四步走核心逻辑

### 步骤1：并行点火（Parallel Ignition）

#### Stream A（语义点火）
```python
seed_nodes = vertex_ids  # 查询中的实体
diffused_node_scores = self.hsc.dynamic_diffusion(
    seed_nodes, dim=0, steps=diffusion_steps, alpha=diffusion_alpha
)
```

#### Stream B（结构模式）
```python
for relation in enhanced_relations:
    for simplex_id, simplex_data in self.hsc.simplices.items():
        if relation_matches_simplex(relation_entities, simplex_nodes):
            relation_to_simplices[simplex_id].add(tuple(relation_entities))
            if dim == 1:
                seed_edge_ids.append(simplex_id)  # 边级别的种子
```

### 步骤2：拓扑扩散（Topological Diffusion）

**核心公式**：
```
x' = (I - α * Lk) x
```

**具体实现**：
```python
# 构建动态关联矩阵
self.hsc.build_dynamic_incidence_matrices()

# 计算 Hodge Laplacian 矩阵
self.hsc.compute_dynamic_hodge_laplacians()

# Stream A：节点级扩散（dim=0）
diffused_node_scores = hsc.dynamic_diffusion(seed_nodes, dim=0, steps=2, alpha=0.1)

# Stream B：边缘级扩散（dim=1）
diffused_edge_scores = hsc.dynamic_diffusion(seed_edge_ids, dim=1, steps=2, alpha=0.1)

# Stream C：高维复形扩散（dim≥2）
for dim in range(2, max_dim + 1):
    dim_scores = hsc.dynamic_diffusion(high_dim_seed_simplices, dim=dim, steps=2, alpha=0.03)
```

**扩散参数**：
| 维度 | steps | alpha | 说明 |
|------|-------|-------|------|
| dim=0 | 2 | 0.1 | 节点级扩散 |
| dim=1 | 2 | 0.1 | 边缘级扩散 |
| dim≥2 | 2 | 0.03 | 高维扩散，使用较小值避免过度扩散 |

### 步骤3：上边界收缩（Coboundary Contraction）

**核心思想**：
- 维度A（点）通过 B^T 向上辐射，寻找包含自己的面
- 计算点的共同上边界：C = ⋂_{v ∈ V_seeds} Coboundary(v)

**具体实现**：
```python
# 收集每个顶点的上边界
vertex_coboundaries = []
for vertex_id in vertex_ids:
    vertex_coboundary = hsc.get_upper_adjacent([vertex_id], current_dim=0)
    vertex_coboundaries.append(vertex_coboundary)

# 计算严格交集
strict_intersection = set.intersection(*vertex_coboundaries)

# 计算覆盖度
for simplex_id in candidate_simplices:
    simplex_vertices = set(simplex_data.get('entities', []))
    covered_vertices = simplex_vertices.intersection(set(vertex_ids))
    coverage_count = len(covered_vertices)
```

**规则化筛选逻辑**：
1. 首先添加严格交集中的复形
2. 然后添加覆盖度高的复形（至少覆盖50%的查询顶点）
3. 按优先级排序：严格交集 > 覆盖数量 > 维度 > 结构复杂度 > 覆盖比例

### 步骤4：拓扑对撞与补全（Topological Collision & Completion）

**核心思想**：
- 当两个维度的能量在某个高阶单纯形 σ 上重合时，该单纯形既满足语义相关性，又满足逻辑连通性

**综合得分计算**：
```python
# 维度A得分：覆盖度 + 维度 + 结构复杂度 + 重要性
a_score = coverage.get('count', 0) + coverage.get('dimension', 0) / 10 + \
          coverage.get('structural_complexity', 0) / 20 + coverage.get('importance', 0)

# 维度B得分：固定分数 + 维度 + 匹配关系数 + 结构复杂度 + 重要性
b_score = 1.0 + dim / 10 + matched_relations * 0.5 + \
          structural_complexity / 20 + importance

# 综合得分
simplex_scores[simplex_id] = a_score * 0.5 + b_score * 0.5
```

**补全策略**（当没有交集时）：
1. **维度提升**：将多个低维复形拼接成高维复形
2. **拓扑链补全**：寻找连接顶点的拓扑路径
3. **共同边界 fallback**：使用共同边界作为最后手段

---

## 复形截断操作

### 核心参数

#### max_simplices - 最大复形总数限制
```python
max_simplices = global_config.get("max_simplices", 100)
if len(retrieved_simplices) > max_simplices:
    retrieved_simplices = retrieved_simplices[:max_simplices]
```

#### dimension_limits - 各维度复形的最大阈值
```python
dimension_limits = global_config.get("dimension_limits", {
    0: 5,   # 顶点维度最多5个
    1: 5,   # 边维度最多5个
    2: 20,  # 面维度最多20个
    3: 15   # 三维体维度最多15个
})
```

#### max_topology_chunks - 文本块数量限制
```python
total_chunks_limit = global_config.get("max_topology_chunks", 110)
```

### 截断流程

#### 第一层截断：按维度 top-k 截断
```python
# 每个维度截断到 top-k
for dim, simplices_with_similarity in dimension_sorted_simplices.items():
    top_k = dimension_top_k.get(dim, 0)
    return dim, simplices_with_similarity[:top_k]
```

#### 第二层截断：维度优先级合并
```python
# 按维度优先级（高维优先）合并所有维度的复形
for dim in [3, 2, 1, 0]:
    for simplex_id, simplex_data, _ in dimension_sorted_simplices[dim]:
        all_related_simplices.append((simplex_id, simplex_data))
```

#### 第三层截断：按 dimension_limits 最终选择
```python
ranked_simplices = []
for dimension in [3, 2, 1, 0]:
    simplices = simplex_by_dimension.get(dimension, [])
    limit = dimension_limits.get(dimension, 0)
    selected_simplices = simplices[:limit]  # 这里进行截断
    ranked_simplices.extend(selected_simplices)
```

### 截断策略的设计思想

#### 高维优先策略
```python
# 优先高维复形的原因：
# - 高维复形包含更丰富的语义信息（多个实体的关联）
# - 3维复形代表完整的原子事实
# - 面和边的信息主要帮助定位和连接
```

#### 维度配额分配
```python
dimension_limits = {
    0: 5,   # 顶点：主要用于定位，不需要太多
    1: 5,   # 边：用于关系匹配，需要一定数量但不需要太多
    2: 20,  # 面：提供上下文信息，需要较多
    3: 15   # 三维体：核心信息载体，分配最多
}
```

---

## 关键算法细节

### 1. 动态关联矩阵构建

```python
def build_dynamic_incidence_matrices(self, query_simplices=None):
    # 按维度分组复形
    simplices_by_dim = defaultdict(list)
    for simplex_id, simplex_data in self.simplices.items():
        dim = simplex_data.get('dimension', 0)
        simplices_by_dim[dim].append((simplex_id, simplex_data))

    # 构建 B1, B2, ... Bmax_dim 矩阵
    for dim in range(1, max_dim + 1):
        lower_simplices = simplices_by_dim.get(dim - 1, [])
        higher_simplices = simplices_by_dim.get(dim, [])

        # 构建稀疏矩阵
        for higher_idx, (higher_id, higher_data) in enumerate(higher_simplices):
            boundary = higher_data.get('boundary', [])
            for lower_id in boundary:
                if lower_id in lower_to_idx:
                    B_row.append(lower_to_idx[lower_id])
                    B_col.append(higher_idx)
                    B_data.append(1)
```

### 2. Hodge Laplacian 计算

```python
def compute_dynamic_hodge_laplacians(self):
    # L0 = B1 @ B1.T
    # Lk = (Bk.T @ Bk) + (Bk+1 @ Bk+1.T)
    for dim in range(max_dim + 1):
        if dim == 0:
            self.L_matrices[0] = B1 @ B1.T
        else:
            L_part1 = Bk.T @ Bk
            L_part2 = Bk_plus_1 @ Bk_plus_1.T
            self.L_matrices[dim] = L_part1 + L_part2
```

### 3. 拓扑能量扩散

```python
def dynamic_diffusion(self, seed_ids, dim=0, steps=2, alpha=0.1, k_hop=3):
    # 初始化能量向量
    x = np.zeros(len(simplices))
    for seed_id in seed_ids:
        if seed_id in idx_map:
            x[idx_map[seed_id]] = 1.0

    # 迭代扩散
    for step in range(steps):
        # 计算拉普拉斯算子作用于当前能量
        if dim == 0:
            # 节点级扩散
            Lx = self.L_matrices[0] @ x
        else:
            # 高维扩散
            Lx = self.L_matrices[dim] @ x

        # 更新能量：x = (1 - alpha) * x + alpha * Lx
        x = (1 - alpha) * x + alpha * Lx

    return {simplex_id: x[idx_map[simplex_id]] for simplex_id in seed_ids}
```

### 4. 实体匹配函数

```python
def entity_match(query_entity: str, simplex_entity: str) -> bool:
    """检查查询实体是否与复形中的实体匹配（支持模糊匹配）"""
    if not query_entity or not simplex_entity:
        return False
    query_str = str(query_entity).strip().lower()
    simplex_str = str(simplex_entity).strip().lower()
    if not query_str or not simplex_str:
        return False
    # 精确匹配优先
    if query_str == simplex_str:
        return True
    # 支持子字符串匹配
    return query_str in simplex_str or simplex_str in query_str
```

### 5. 规则化交集计算

```python
# 计算每个复形的覆盖度和其他属性
simplex_coverage = {}
for simplex_id in candidate_simplices:
    simplex_data = self.hsc.simplices.get(simplex_id, {})
    simplex_vertices = set(simplex_data.get('nodes', simplex_data.get('entities', [])))

    # 计算覆盖的查询顶点数量和比例
    covered_vertices = simplex_vertices.intersection(set(vertex_ids))
    coverage_count = len(covered_vertices)
    coverage_ratio = coverage_count / len(vertex_ids) if vertex_ids else 0

    # 计算维度
    dim = simplex_data.get('dimension', 0)

    # 利用边界信息计算复形的结构复杂度
    boundary = simplex_data.get('boundary', [])
    coboundary = simplex_data.get('coboundary', [])
    structural_complexity = len(boundary) + len(coboundary)

    simplex_coverage[simplex_id] = {
        'count': coverage_count,
        'ratio': coverage_ratio,
        'dimension': dim,
        'structural_complexity': structural_complexity,
        'importance': simplex_data.get('importance', 1.0),
        'covered_vertices': covered_vertices
    }
```

---

## 代码逐行分析

### 函数定义与参数（第1472行）

```python
def dual_dimension_retrieve(
    self,
    query_vertices: List[dict],
    query_partial_relations: List[List[int]],
    diffusion_steps: int = 2,
    diffusion_alpha: float = 0.1,
    coboundary_threshold: float = 0.5,
    type_filter: List[str] = None
) -> dict:
```

**参数说明**：
- `query_vertices`: 查询顶点列表，如 `[{'type': 'Entity', 'id': 'node1'}, ...]`
- `query_partial_relations`: 查询中的部分关系，如 `[['node1', 'node2'], ...]`
- `diffusion_steps`: 拓扑扩散的迭代步数，默认2
- `diffusion_alpha`: 扩散率参数，默认0.1
- `coboundary_threshold`: 上边界匹配的最低阈值，默认0.5
- `type_filter`: 可选的类型过滤列表

### 步骤1：并行点火（第1472-1581行）

#### 提取查询顶点ID（第1472行）
```python
vertex_ids = [v['id'] for v in query_vertices if 'id' in v and v.get('id', '').strip()]
```
- 列表推导式，从 query_vertices 中提取所有有效的顶点ID
- 过滤条件：`'id' in v` 确保有id字段，`v.get('id', '').strip()` 确保id非空

#### 增强关系（第1474-1481行）
```python
enhanced_relations = []
enhanced_relations = query_partial_relations.copy()
```
- 直接复制查询中的关系，不做额外增强处理

#### Stream A 语义点火（第1485-1490行）
```python
seed_nodes = vertex_ids
logger.info(f"Stream A（语义点火）：激活 {len(seed_nodes)} 个种子节点")
```
- `seed_nodes` 是查询中的实体顶点，作为语义检索的种子

#### Stream B 结构匹配（第1499-1574行）
```python
# 构建单纯形索引
simplex_index = defaultdict(set)
for simplex_id, simplex_data in self.hsc.simplices.items():
    entities = tuple(sorted(simplex_data.get('entities', [])))
    dim = simplex_data.get('dimension', 0)
    simplex_index[(dim, entities)].add(simplex_id)

# 匹配关系到单纯形
for relation in enhanced_relations:
    # ... 遍历匹配逻辑
    if relation_matches_simplex(relation_entities, simplex_nodes):
        relation_to_simplices[simplex_id].add(tuple(relation_entities))
        if dim == 1:
            seed_edge_ids.append(simplex_id)
```
- 构建 `(dimension, entities_tuple)` -> set(simplex_ids) 的索引
- 支持缓存机制加速匹配

### 步骤2：拓扑扩散（第1583-1661行）

#### 构建关联矩阵和拉普拉斯矩阵（第1586-1588行）
```python
self.hsc.build_dynamic_incidence_matrices()
self.hsc.compute_dynamic_hodge_laplacians()
```
- 调用 HSC 的方法，构建描述复形间包含关系的矩阵
- 用于后续的拓扑能量扩散

#### Stream A 扩散（第1594-1598行）
```python
diffused_node_scores = self.hsc.dynamic_diffusion(
    seed_nodes, dim=0, steps=diffusion_steps, alpha=diffusion_alpha
)
```
- 在维度0（节点）上扩散

#### Stream C 高维扩散（第1608-1661行）
```python
# 收集高维种子单纯形
for edge_id in seed_edge_ids:
    edge_upper = self.hsc.get_upper_adjacent([edge_id], current_dim=1)
    for simplex_id in edge_upper:
        simplex_data = self.hsc.simplices.get(simplex_id, {})
        dim = simplex_data.get('dimension', 0)
        if dim >= 2:
            # 计算种子复形的相关性
            simplex_vertices = set(simplex_data.get('nodes', simplex_data.get('entities', [])))
            covered_vertices = simplex_vertices.intersection(set(vertex_ids))
            coverage_count = len(covered_vertices)
            if coverage_count > 0:
                seed_score = (coverage_count ** 2) + (importance * 2)
                seed_candidates[simplex_id] = seed_score

# 执行各维度扩散
for dim in range(2, max_dim + 1):
    dim_seed = [sid for sid in high_dim_seed_simplices
               if self.hsc.simplices.get(sid, {}).get('dimension', 0) == dim]
    if dim_seed:
        dim_scores = self.hsc.dynamic_diffusion(dim_seed, dim=dim, steps=2, alpha=0.03)
        diffused_high_dim_scores.update(dim_scores)
```

### 步骤3：上边界收缩（第1663-1786行）

#### 收集每个顶点的上边界（第1676-1692行）
```python
vertex_coboundaries = []
for vertex_id in vertex_ids:
    vertex_coboundary = self.hsc.get_upper_adjacent([vertex_id], current_dim=0)
    if type_filter:
        filtered_coboundary = set()
        for simplex_id in vertex_coboundary:
            simplex_data = self.hsc.simplices.get(simplex_id, {})
            simplex_type = simplex_data.get('type', '')
            if simplex_type not in type_filter:
                filtered_coboundary.add(simplex_id)
        vertex_coboundary = filtered_coboundary
    vertex_coboundaries.append(vertex_coboundary)
```

#### 计算严格交集（第1696-1698行）
```python
strict_intersection = set()
if vertex_coboundaries:
    strict_intersection = set.intersection(*vertex_coboundaries)
```
- `set.intersection(*vertex_coboundaries)` 展开参数列表，计算所有集合的交集

#### 规则化筛选（第1744-1760行）
```python
min_coverage_count = max(1, len(vertex_ids) // 2)
min_coverage_ratio = 0.5

filtered_simplices = []

# 1. 首先添加严格交集中的复形
for simplex_id in strict_intersection:
    if simplex_id in simplex_coverage:
        filtered_simplices.append((simplex_id, simplex_coverage[simplex_id]))

# 2. 然后添加覆盖度高的复形
for simplex_id, coverage in simplex_coverage.items():
    if simplex_id not in strict_intersection and (
        coverage['count'] >= min_coverage_count or
        coverage['ratio'] >= min_coverage_ratio):
        filtered_simplices.append((simplex_id, coverage))
```

### 步骤4：拓扑对撞与补全（第1788-2038行）

#### 计算综合得分（第1791-1821行）
```python
simplex_scores = {}

# 维度A得分
for simplex_id in common_coboundary:
    coverage = simplex_coverage.get(simplex_id, {})
    a_score = coverage.get('count', 0) + coverage.get('dimension', 0) / 10 + \
              coverage.get('structural_complexity', 0) / 20 + coverage.get('importance', 0)
    simplex_scores[simplex_id] = a_score * 0.5

# 维度B得分
for simplex_id in pattern_simplices:
    simplex_data = self.hsc.simplices.get(simplex_id, {})
    dim = simplex_data.get('dimension', 0)
    matched_relations = len(relation_to_simplices.get(simplex_id, set()))
    b_score = 1.0 + dim / 10 + matched_relations * 0.5 + \
              structural_complexity / 20 + importance
    simplex_scores[simplex_id] += b_score * 0.5
```

#### 维度提升策略（第1876-1927行）

**策略1：基于边的上边界提升**
```python
elevated_simplices = set()
if not intersection_simplices:
    edge_coboundary = seed_edge_ids
    for edge_simplex_id in edge_coboundary:
        edge_simplex_entities = set(self.hsc.simplices.get(edge_simplex_id, {}).get('entities', []))
        for pattern_id in pattern_simplices:
            pattern_entities = set(self.hsc.simplices.get(pattern_id, {}).get('entities', []))
            if pattern_entities.issubset(edge_simplex_entities_set):
                elevated_simplices.add(edge_simplex_id)
```

**策略2：基于共同顶点的复形拼接**
```python
if not elevated_simplices:
    for simp1_id in pattern_simplices:
        for simp2_id in common_coboundary:
            common_vertices = simp1_entities & simp2_entities
            if common_vertices:
                merged_entities = simp1_entities | simp2_entities
                for candidate_id, candidate_data in self.hsc.simplices.items():
                    if merged_entities.issubset(set(candidate_data.get('entities', []))):
                        elevated_simplices.add(candidate_id)
```

**策略3：迭代提升**
```python
if not elevated_simplices:
    elevated_simplices = self._iterative_dimension_elevation(
        vertex_ids, pattern_simplices, common_coboundary, max_iterations=3
    )
```

#### 提取缺失顶点作为答案候选（第1969-2016行）
```python
if not completion_results:
    vertex_id_set = set(vertex_ids)
    for simplex_id in intersection_simplices:
        simplex_data = self.hsc.simplices[simplex_id]
        simplex_nodes = simplex_data.get('nodes', simplex_data.get('entities', []))

        missing_vertices = []
        for node in simplex_nodes:
            if node not in vertex_id_set:
                node_type = simplex_type_map.get(node, 'unknown')
                missing_vertices.append({
                    'id': node,
                    'type': node_type,
                    'is_answer_candidate': node_type in ['Value', 'State', 'Number', 'Attribute']
                })
```

#### 最终显著性重排（第2018-2028行）
```python
completion_results.sort(key=lambda x: (
    x['dimension'],  # 优先维度高的（信息更丰富）
    1 if x['level_hg'] == 'high' or (isinstance(x['level_hg'], (int, float)) and x['level_hg'] > 0.7) else 0,
    len(x.get('missing_vertices', [])),
    x.get('coboundary_score', 0),
    x.get('diffusion_score', 0),
    len(x.get('matched_vertices', [])),
), reverse=True)
```

---

## 总结

`dual_dimension_retrieve` 函数的核心逻辑可以概括为：

```
分发 → 扩散 → 交集 → 补全 → 排序
```

1. **分发**：将查询顶点作为种子，分发到语义维度和结构维度两个并行流
2. **扩散**：利用 Hodge Laplacian 在图上进行能量扩散，传播相关性
3. **交集**：计算语义维度和结构维度的共同上边界，找到既满足语义相关又满足逻辑连通的复形
4. **补全**：通过维度提升、拓扑链补全等策略处理无交集情况
5. **排序**：多级排序确保高质量结果优先返回

### 核心优势

| 优势 | 说明 |
|------|------|
| 多维度理解 | 同时考虑语义和结构两个维度 |
| 抗幻觉能力 | 通过双维度验证，确保检索结果既满足语义相关性，又满足逻辑连通性 |
| 拓扑能量扩散 | 利用数学上的 Hodge 理论进行能量传播 |
| 规则化交集 | 处理不完整查询时的柔性匹配 |
| 迭代补全 | 通过维度提升和拓扑链补全缺失信息 |

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| diffusion_steps | 2 | 拓扑扩散的迭代步数 |
| diffusion_alpha | 0.1 | 扩散率参数 |
| coboundary_threshold | 0.5 | 上边界匹配的最低阈值 |
| max_simplices | 100 | 最大复形总数限制 |
| dimension_limits | {0:5, 1:5, 2:20, 3:15} | 各维度复形的最大阈值 |
| max_topology_chunks | 110 | 文本块数量限制 |

---

*文档生成时间：2026-04-26*
*代码版本：Hyper-RAG-main2*
