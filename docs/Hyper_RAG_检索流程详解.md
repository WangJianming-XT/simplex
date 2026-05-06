# Hyper-RAG 检索流程详解

## 一、日志解读：Stream A（dim=0）扩散完成，获得 3689 个节点的拓扑得分

### 1.1 日志含义

```
Stream A（dim=0）扩散完成，获得 3689 个节点的拓扑得分
```

这条日志表示：

- **Stream A**：检索流程中的"维度A——语义点火"通道，从查询中提取的实体节点出发进行检索
- **dim=0**：在单纯复形的 **0 维（顶点/节点维度）** 上执行扩散操作
- **3689 个节点**：在扩散过程中，种子节点的 k-hop（默认 k=3）邻域内共有 3689 个节点参与了拓扑能量扩散计算，每个节点都获得了一个拓扑得分
- **拓扑得分**：通过 Hodge Laplacian 热扩散方程迭代计算出的归一化得分，反映该节点与查询种子节点的拓扑相关程度

### 1.2 为什么是 3689 个节点？

3689 并非知识库中全部节点的数量，而是**种子节点的 k-hop 邻域内**的节点数量。具体来说：

1. 查询实体被匹配到 HSC 中的若干种子节点（如 5 个）
2. 以这些种子节点为起点，沿边（1-单纯形）向外扩展 3 跳（默认 `DIFFUSION_K_HOP=3`）
3. 所有可达的 0 维节点构成邻域子集，共 3689 个
4. 仅对这个邻域子集构建局部拉普拉斯矩阵并执行扩散，而非对全量节点

这个数字取决于：
- 知识图谱的连通密度
- 种子节点在图中的位置（中心 vs 边缘）
- k-hop 参数设置

### 1.3 日志中出现的其他 Stream

| 日志 | 含义 |
|------|------|
| `Stream A（dim=0）扩散完成，获得 N 个节点的拓扑得分` | 0 维（顶点）扩散，维度A语义点火 |
| `Stream B（dim=1）扩散完成，获得 N 个边的拓扑得分` | 1 维（边）扩散，维度B结构模式 |
| `Stream C (dim=2) 扩散完成，获得 N 个高维复形的拓扑得分` | 2 维（三角形面）扩散 |
| `Stream C (dim=3) 扩散完成，获得 N 个高维复形的拓扑得分` | 3 维（四面体）扩散 |
| `Stream C (dim=4) 扩散完成，获得 N 个高维复形的拓扑得分` | 4 维（四维单纯形）扩散 |

> 注意：当日志显示 `获得 0 个节点的拓扑得分` 时，通常意味着查询实体在 HSC 中未找到匹配的种子节点，或种子节点的邻域为空。

---

## 二、完整检索流程架构

Hyper-RAG 的检索流程遵循 **"分发 → 扩散 → 对撞 → 补全"** 四步走范式，整体管线如下：

```
用户查询
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤1：查询实体提取                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LLM 提取实体列表 + 关系列表 + 最高维单纯形           │    │
│  │  回退：正则表达式 \b[A-Z][a-zA-Z0-9_]+\b            │    │
│  │  统一标准化：normalize_entity_name（大写+去空格）      │    │
│  └─────────────────────────────────────────────────────┘    │
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤2：构建异质单纯复形（HSC）                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  优先从拉普拉斯缓存加载（L0, L1, nodes, simplices）  │    │
│  │  缓存失效时从 SimplexStorage 重建：                   │    │
│  │    - 加载所有单纯形 → 填充 boundary/coboundary        │    │
│  │    - 构建关联矩阵 Bk                                  │    │
│  │    - 计算 Hodge Laplacian Lk                          │    │
│  └─────────────────────────────────────────────────────┘    │
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤3：实体匹配                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  精确匹配 → 子串匹配 → 描述词重叠匹配                  │    │
│  │  综合得分 = 0.6 × 名称匹配分 + 0.4 × 描述匹配分       │    │
│  │  阈值 ≥ 0.3                                          │    │
│  └─────────────────────────────────────────────────────┘    │
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤4：双维度并行检索（核心）                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  4.1 分发（Parallel Ignition）                        │    │
│  │    维度A：查询实体 → seed_nodes                        │    │
│  │    维度B：查询关系 → seed_edge_ids + seed_high_dim     │    │
│  │                                                       │    │
│  │  4.2 扩散（Topological Diffusion）                    │    │
│  │    Stream A (dim=0)：节点扩散                          │    │
│  │    Stream B (dim=1)：边扩散                            │    │
│  │    Stream C (dim≥2)：高维复形扩散                      │    │
│  │                                                       │    │
│  │  4.3 对撞（Coboundary Contraction & Fusion）          │    │
│  │    维度A：上边界收缩（B^T 向上辐射）                    │    │
│  │    维度B：模式匹配约束                                  │    │
│  │    双维度加权融合：0.5×A + 0.5×B                       │    │
│  │                                                       │    │
│  │  4.4 补全（Completion）                               │    │
│  │    维度提升 → 拓扑链补全 → 子复形去重 → 显著性重排     │    │
│  └─────────────────────────────────────────────────────┘    │
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤5：能量扩散增强（二次扩散）                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  基于检索结果子集重建动态拉普拉斯矩阵                    │    │
│  │  再次执行 dim=0 扩散，按扩散得分重排序                  │    │
│  └─────────────────────────────────────────────────────┘    │
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤6：收集文本块                                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  从排序后的复形提取 source_id → 获取原始文本           │    │
│  │  限制最大文本块数量（默认50）                           │    │
│  │  无结果时回退到向量数据库检索                           │    │
│  └─────────────────────────────────────────────────────┘    │
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤7：生成回答                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  拼接文本块为上下文                                    │    │
│  │  使用拓扑增强的系统提示词                               │    │
│  │  调用 LLM 生成最终回答                                 │    │
│  └─────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────┘
```

---

## 三、各步骤详细说明

### 3.1 步骤1：查询实体提取

**源码位置**：[_retrieval.py:22-177](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retrieval.py#L22-L177)

**功能**：从用户查询文本中提取结构化信息，作为后续检索的输入。

**提取内容**：
- **实体列表（entities）**：查询中的关键名词、动词、形容词等
- **最高维单纯形（highest_simplex）**：包含所有提取实体的最全面关系
- **关系列表（relations/simplices）**：实体间的结构化关系

**提取策略**：
1. 优先使用 LLM 调用 `query_entity_extraction` 提示词进行提取
2. LLM 不可用时回退到正则表达式 `\b[A-Z][a-zA-Z0-9_]+\b`
3. 所有实体名称通过 `normalize_entity_name` 统一转大写+去空格

**示例**：
```
查询: "What is the relationship between Apple and Microsoft?"
提取结果:
  entities: ["APPLE", "MICROSOFT"]
  relations: [{entities: ["APPLE", "MICROSOFT"], dimension: 1}]
```

---

### 3.2 步骤2：构建异质单纯复形（HSC）

**源码位置**：[_retrieval.py:180-238](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retrieval.py#L180-L238)

**功能**：从持久化存储中加载或重建异质单纯复形数据结构。

**HSC 核心数据**：
| 属性 | 说明 |
|------|------|
| `nodes` | 所有 0 维顶点 `{node_id: {type, vector}}` |
| `simplices` | 所有单纯形 `{simplex_id: {dimension, entities, boundary, coboundary, ...}}` |
| `B_matrices` | 各维度的关联矩阵 `Bk` |
| `L_matrices` | 各维度的 Hodge Laplacian 矩阵 `Lk` |

**加载策略**：
1. 优先从 `simplex_storage` 的拉普拉斯缓存加载（含 L0、L1 矩阵、节点和复形数据）
2. 缓存失效时从存储重建：加载所有单纯形 → 填充 boundary/coboundary → 构建关联矩阵 → 计算 Hodge Laplacian

---

### 3.3 步骤3：实体匹配

**源码位置**：[_retrieval.py:241-351](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retrieval.py#L241-L351)

**功能**：将查询中提取的实体匹配到 HSC 中的顶点，建立查询到知识库的桥梁。

**匹配策略（优先级从高到低）**：

| 策略 | 说明 | 得分 |
|------|------|------|
| 精确匹配 | 标准化后完全相等 | 1.0 |
| 子串匹配 | 查询实体是节点名的子串或反之 | 0.6-1.0 |
| 描述词重叠 | 实体描述与节点描述的词重叠度 | 0-1.0 |

**综合得分公式**：
```
total_score = 0.6 × name_score + 0.4 × desc_score
```

**筛选条件**：`total_score ≥ 0.3`

---

### 3.4 步骤4：双维度并行检索（核心）

**源码位置**：[_retriever.py:1038-1179](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L1038-L1179)

这是整个检索流程的核心，遵循 **"分发 → 扩散 → 对撞 → 补全"** 四步走逻辑。

#### 3.4.1 分发（Parallel Ignition）

**源码位置**：[_retriever.py:1086-1090](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L1086-L1090)

将查询信息分发到两个独立维度：

| 维度 | 名称 | 输入 | 输出 |
|------|------|------|------|
| 维度A | 语义点火 | 查询实体 → `seed_nodes` | 种子节点列表 |
| 维度B | 结构模式 | 查询关系 → `seed_edge_ids` + `seed_high_dim_simplices` | 种子边 + 高维复形 |

**维度B的关系匹配**（[_retriever.py:215-295](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L215-L295)）：
- 使用倒排索引加速匹配
- 计算匹配得分（自适应阈值）
- 按得分排序，输出种子边和高维复形

#### 3.4.2 扩散（Topological Diffusion）

**源码位置**：[_retriever.py:297-358](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L297-L358) 和 [_simplicial_complex.py:226-384](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_simplicial_complex.py#L226-L384)

对每个维度执行基于 Hodge Laplacian 的拓扑能量扩散：

**扩散流程**：

```
种子节点初始化（能量=1.0）
        │
        ▼
获取 k-hop 邻域（默认 k=3）
        │
        ▼
提取邻域子拉普拉斯矩阵
        │
        ▼
热扩散迭代：x(t+1) = x(t) - α × L × x(t)
        │
        ▼
L2 归一化 → 每个节点的拓扑得分
```

**核心公式**：
```
x(t+1) = x(t) - α × L_k × x(t)
```

其中：
- `x(t)`：第 t 步的能量分布向量
- `α`：扩散率（自适应计算）
- `L_k`：k 维 Hodge Laplacian 矩阵

**自适应扩散参数**（[_config.py:85-110](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_config.py#L85-L110)）：

| 数据规模 | 扩散步数 (steps) | 扩散率 (α) |
|----------|------------------|------------|
| ≤ 100 节点 | 2 | 0.1 |
| 100-500 节点 | 2 | 0.08 |
| > 500 节点 | 3 | 0.05 |
| dim ≥ 2 | 同上 | α × 0.5 |

**物理直觉**：
- 能量从种子节点出发，沿着图的边（1-单纯形）向邻居节点扩散
- 与种子节点拓扑距离越近、连接路径越多的节点，获得的能量越高
- 本质上是**图上的热扩散过程**，利用 Hodge Laplacian 的谱性质过滤掉拓扑噪声，保留与查询语义相关的结构信息

**各 Stream 的扩散**：

| Stream | 维度 | 种子来源 | 输出 |
|--------|------|----------|------|
| Stream A | dim=0（节点） | 查询匹配的实体节点 | `diffused_node_scores` |
| Stream B | dim=1（边） | 关系匹配的边 | `diffused_edge_scores` |
| Stream C | dim≥2（高维复形） | 关系匹配的高维复形 | `diffused_high_dim_scores` |

#### 3.4.3 对撞（Coboundary Contraction & Fusion）

**源码位置**：[_retriever.py:360-685](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L360-L685)

两个维度的检索结果在公共复形上"对撞"，确认相关性。

**维度A：上边界收缩**（[_retriever.py:360-448](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L360-L448)）：
- 通过 B^T 矩阵向上辐射，寻找包含查询顶点的高维复形
- 计算覆盖度（coverage_ratio）和语义相似度
- 筛选候选复形

**双维度加权融合**（[_retriever.py:566-685](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L566-L685)）：
```
weighted_score = 0.5 × score_A + 0.5 × score_B
```

其中：
- `score_A`：维度A得分 = `importance × (1 + log(1+dim)) × (1 + coverage_ratio)`
- `score_B`：维度B得分 = `importance × (1 + log(1+dim)) × (1 + match_ratio)`

**自适应阈值筛选**：
```python
adaptive_threshold = max(coboundary_threshold, percentile(scores, 30))
```

#### 3.4.4 补全（Completion）

**源码位置**：[_retriever.py:687-1036](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L687-L1036)

当融合结果为空时，依次尝试以下补全策略：

| 策略 | 源码位置 | 说明 |
|------|----------|------|
| 维度提升 | [_retriever.py:687-816](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L687-L816) | 将多个低维复形拼接成高维复形 |
| 拓扑链补全 | [_retriever.py:818-914](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L818-L914) | BFS 搜索连接查询顶点的拓扑路径 |
| Common Coboundary | [_retriever.py:1135-1136](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L1135-L1136) | 使用查询顶点的共同上边界作为最终候选 |

**子复形去重**（[_retriever.py:980-1036](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L980-L1036)）：
- 移除被父复形完全包含的低维子复形
- 避免冗余上下文
- 为父复形添加分数奖励

**显著性重排**（[_retriever.py:1144-1166](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py#L1144-L1166)）：

排序键（优先级从高到低）：
1. `dimension`（维度越高越优先）
2. `level_hg`（高层/重要性标记）
3. `missing_vertices` 数量（缺失顶点越少越好）
4. `coboundary_score`（上边界得分）
5. `diffusion_score`（扩散得分）
6. `matched_vertices` 数量（匹配顶点越多越好）

---

### 3.5 步骤5：能量扩散增强（二次扩散）

**源码位置**：[_retrieval.py:514-534](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retrieval.py#L514-L534)

**功能**：基于检索到的复形子集，重新构建动态拉普拉斯矩阵，再次执行扩散，对检索结果进行精细化重排序。

**流程**：
1. 从检索结果中提取复形子集
2. 基于子集重建关联矩阵和拉普拉斯矩阵
3. 再次执行 `dim=0` 扩散
4. 计算每个复形的平均扩散得分
5. 按扩散得分重新排序

**与第一次扩散的区别**：
- 第一次扩散在全量 HSC 上执行，用于发现候选复形
- 第二次扩散仅在检索结果子集上执行，用于精细化排序

---

### 3.6 步骤6：收集文本块

**源码位置**：[_retrieval.py:354-423](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retrieval.py#L354-L423)

**功能**：从排序后的复形中提取原始文本内容，作为 LLM 的上下文输入。

**流程**：
1. 从排序后的复形中提取 `source_id`
2. 从 `text_chunks_db` 获取原始文本内容
3. 限制最大文本块数量（默认 50）
4. 若无结果，回退到向量数据库检索

---

### 3.7 步骤7：生成回答

**源码位置**：[hyperrag.py:506-531](file:///f:/work/Hyper-RAG-main2/hyperrag/hyperrag.py#L506-L531)（topology_aware 模式）和 [hyperrag.py:557-587](file:///f:/work/Hyper-RAG-main2/hyperrag/hyperrag.py#L557-L587)（topology 模式）

**功能**：将检索到的文本块拼接为上下文，调用 LLM 生成最终回答。

**两种模式的区别**：

| 模式 | 系统提示词 | 最大文本块数 |
|------|-----------|-------------|
| `topology_aware` | 自定义动态 Prompt | 110 |
| `topology` | `topology_response_system_prompt_concise` 模板 | 20 |

---

## 四、Hodge Laplacian 与拓扑扩散原理

### 4.1 单纯复形基础

单纯复形（Simplicial Complex）是 Hyper-RAG 的核心数据结构，它将知识图谱从传统的"节点-边"图结构扩展到高维拓扑空间：

| 维度 | 名称 | 示例 |
|------|------|------|
| dim=0 | 顶点（Vertex） | 单个实体，如 "APPLE" |
| dim=1 | 边（Edge） | 二元关系，如 ("APPLE", "MICROSOFT") |
| dim=2 | 三角形面（Triangle） | 三元关系，如 ("APPLE", "MICROSOFT", "GOOGLE") |
| dim=3 | 四面体（Tetrahedron） | 四元关系 |
| dim≥4 | 高维单纯形 | 更复杂的多实体关系 |

### 4.2 关联矩阵（Incidence Matrix）

关联矩阵 `Bk` 描述了 k-1 维复形与 k 维复形之间的边界关系：

```
Bk[i,j] = 1  表示第 i 个 (k-1)-单纯形是第 j 个 k-单纯形的面
Bk[i,j] = 0  否则
```

**源码位置**：[_simplicial_complex.py:105-166](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_simplicial_complex.py#L105-L166)

### 4.3 Hodge Laplacian 矩阵

Hodge Laplacian 是单纯复形上的核心拓扑算子：

```
L0 = B1 × B1^T          （图拉普拉斯，即经典的拉普拉斯矩阵）
Lk = Bk^T × Bk + Bk+1 × Bk+1^T    （k > 0）
```

**源码位置**：[_simplicial_complex.py:167-224](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_simplicial_complex.py#L167-L224)

**物理意义**：
- `Bk^T × Bk`：衡量"向内"的边界约束（下边界）
- `Bk+1 × Bk+1^T`：衡量"向外"的边界约束（上边界）
- 两者之和反映了复形在拓扑空间中的"曲率"信息

### 4.4 热扩散方程

扩散过程模拟了能量在单纯复形上的传播：

```
x(t+1) = x(t) - α × Lk × x(t)
```

**直觉理解**：
- 初始时刻，只有种子节点携带能量（值为 1.0）
- 每一步迭代，能量沿着拓扑结构向外扩散
- Hodge Laplacian 充当"扩散算子"，控制能量的传播方向和速率
- 与种子节点拓扑距离近、连接路径多的节点获得更高能量
- 最终归一化后，每个节点得到 [0, 1] 范围的拓扑得分

**数值稳定性保障**：
- 每步检查最大值是否超过 10^6，超过则缩放
- 检查 NaN/Inf，出现则停止扩散
- 最终结果做 L2 归一化

---

## 五、得分计算体系

### 5.1 复形得分（calculate_simplex_score）

**源码位置**：[_simplicial_complex.py:30-77](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_simplicial_complex.py#L30-L77)

| 得分类型 | 公式 | 说明 |
|----------|------|------|
| 维度A（语义点火） | `importance × (1 + log(1+dim)) × (1 + coverage_ratio)` | 基于查询实体的覆盖度 |
| 维度B（结构模式） | `importance × (1 + log(1+dim)) × (1 + match_ratio)` | 基于关系匹配度 |

**参数说明**：
- `importance`：LLM 评估的结构重要性（0-1）
- `dim`：复形维度，使用 `log(1+dim)` 对数缩放避免高维膨胀
- `coverage_ratio`：复形覆盖的查询顶点比例（维度A）
- `match_ratio`：关系匹配的实体比例（维度B）

### 5.2 扩散得分（diffusion_score）

通过 Hodge Laplacian 热扩散方程迭代计算，最终归一化后每个复形获得 [0, 1] 范围的扩散得分。

### 5.3 融合得分

```
weighted_score = 0.5 × score_A + 0.5 × score_B
```

通过自适应阈值筛选：`threshold = max(coboundary_threshold, percentile(scores, 30))`

---

## 六、自适应参数系统

**源码位置**：[_config.py:18-170](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_config.py#L18-L170)

Hyper-RAG 使用 `DualDimensionConfig` 和 `AdaptiveThreshold` 实现自适应参数计算，避免硬编码固定值：

| 参数 | 自适应策略 |
|------|-----------|
| 覆盖阈值 | `max(0.3, 1.0 / log2(vertex_count + 1))` |
| 语义相似度阈值 | `mean(scores) - 0.5 × std(scores)`，范围 [0.2, 0.8] |
| 保留数量 | `min(sqrt(total) × 5, 100)` |
| 扩散参数 | 根据数据规模和维度动态调整 |
| 匹配阈值 | 根据实体数量动态调整 |

---

## 七、关键源码文件索引

| 文件 | 职责 |
|------|------|
| [hyperrag.py](file:///f:/work/Hyper-RAG-main2/hyperrag/hyperrag.py) | 主入口类 `HyperRAG`，提供 `insert` 和 `query` 方法 |
| [_retrieval.py](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retrieval.py) | 检索统一入口 `topology_retrieval` / `topology_aware_retrieval` |
| [_retriever.py](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_retriever.py) | 核心检索器 `SimplicialRAGRetriever`，实现四步走逻辑 |
| [_simplicial_complex.py](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_simplicial_complex.py) | HSC 数据结构与拓扑算子 |
| [_config.py](file:///f:/work/Hyper-RAG-main2/hyperrag/operate/_config.py) | 自适应参数系统 `DualDimensionConfig` |
| [storage.py](file:///f:/work/Hyper-RAG-main2/hyperrag/storage.py) | 存储层（SimplexStorage） |
| [simplex_tree.py](file:///f:/work/Hyper-RAG-main2/hyperrag/simplex_tree.py) | 前缀树实现的单纯形树 |
| [prompt.py](file:///f:/work/Hyper-RAG-main2/hyperrag/prompt.py) | LLM 提示词模板 |

---

## 八、设计亮点总结

1. **双维度并行**：维度A（语义点火，从实体出发）和维度B（结构模式，从关系出发）独立检索后加权融合，互补覆盖
2. **拓扑扩散**：基于 Hodge Laplacian 的热扩散，利用单纯复形的拓扑结构传播相关性信号，超越简单的向量相似度
3. **自适应参数**：`DualDimensionConfig` 和 `AdaptiveThreshold` 根据数据规模和分布动态调整阈值、步数、扩散率等参数
4. **多级回退**：融合失败 → 维度提升 → 拓扑链补全 → 向量数据库回退，确保检索鲁棒性
5. **子复形去重**：移除被父复形完全包含的低维子复形，避免冗余上下文
6. **二次扩散**：在检索结果子集上重新执行扩散，精细化排序
