# 基于极大语义组的松弛单纯复形 RAG：完整方法流程

***

## 一、方法总览

### 1.1 核心思想

将实体识别与拓扑构造解耦：LLM 只做语义理解（提取极大语义组），本地算法做拓扑构造（构建松弛单纯复形）。

### 1.2 核心概念

**极大语义组（Maximal Semantic Group, MSG）**：一个语义事件中，所有共同参与的实体的最完整集合，满足：

- **完备性（Completeness）**：集合中所有实体共同参与同一个语义事件，不可再添加实体而不破坏语义一致性
- **极大性（Maximality）**：集合不是任何更大语义完备集合的子集

### 1.3 双层架构

| 层 | 内容 | 功能 | 包含字段 |
|---|------|------|---------|
| **信息层** | MSG（极大单纯形） | 检索的目标，信息载体 | description, source_id, embedding, importance |
| **导航层** | 实体↔MSG二部图（共享实体即桥梁） | 扩散的路径，拓扑锚点 | coboundary（实体→所属MSG列表）, boundary（MSG→包含的实体列表） |

**关键简化**：导航层不再显式创建1-单纯形（边）和2-单纯形（三角形）。MSG之间通过共享实体天然连通，无需额外建边。

### 1.4 全流程图

```
原始文本
    ↓
[阶段1] LLM 提取（MSG Prompt）
    ↓ 输出: MSG列表 + 实体列表
[阶段2] 本地代码转换
    ↓ 输出: 信息层MSG + 0-单纯形 + 二部图关联
[阶段3] 存储与索引
    ↓ 写入: entities_vdb + relationships_vdb + SimplexStorage
[阶段4] 检索（2×2结构：语义/拓扑 × 实体/MSG）
    ↓ 输出: 排序后的MSG列表 + 文本块
[阶段5] 上下文组装 + LLM回答生成
    ↓ 输出: 最终回答
```

***

## 二、阶段1：实体提取（LLM）

### 2.1 Prompt 设计

```
-Goal-
Given a text document, identify all Maximal Semantic Groups (MSG) and their constituent
entities. An MSG captures the most complete set of entities participating in a single
semantic event — output them together, NOT decomposed into pairs.
Use {language} as output language.

-Steps-
1. Divide the text into distinct semantic events — each event is a fact, action, or
   relationship involving multiple entities.
2. For each semantic event, output an MSG record:
   -- mcss_description: A sentence describing the event.
   -- entity_list: ALL entities participating in this event (comma-separated, in brackets).
   -- completeness: 0-1 score (1.0 = all participants captured).
   Format: ("mcss"{tuple_delimiter}<mcss_description>{tuple_delimiter}[<e1>,<e2>,...]{tuple_delimiter}<completeness>)
3. For each entity in an MSG, extract:
   -- entity_name: Capitalized. Keep defined terms whole (e.g., "SUPPORTING LENDERS").
   -- entity_type: One of the types listed below.
   -- original_text: EXACT text fragment — do NOT summarize.
   -- importance: 0-1 score (1.0 = central, 0.5 = supporting, 0.1 = marginal).
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<original_text>{tuple_delimiter}<importance>)
4. Return output in {language} as a single list. Use {record_delimiter} as the list delimiter.
5. When finished, output {completion_delimiter}

-Entity Types-
{entity_types}

######################
-Examples-
######################
{examples}
######################
-Real Data-
######################
Entity_types: [{entity_types}]
Text: {input_text}
######################
Output:
```

### 2.2 示例

**Example 1:**

```text
Entity_types: [organization, person, geo, event, category]

Text:
Endologix filed for Chapter 11 reorganization on January 13, 2020, in Wilmington. The Supporting Lenders committed $400 million in DIP financing under the Restructuring Support Agreement.
################
Output:
("mcss"|Endologix filed for Chapter 11 reorganization on January 13, 2020, in Wilmington|[ENDOLOGIX,CHAPTER 11 REORGANIZATION,JANUARY 13 2020,WILMINGTON]|0.95){record_delimiter}
("entity"|ENDOLOGIX|organization|Endologix filed for Chapter 11 reorganization|0.95){record_delimiter}
("entity"|CHAPTER 11 REORGANIZATION|event|Endologix filed for Chapter 11 reorganization|0.9){record_delimiter}
("entity"|JANUARY 13 2020|event|Endologix filed for Chapter 11 reorganization on January 13, 2020|0.85){record_delimiter}
("entity"|WILMINGTON|geo|Endologix filed for Chapter 11 reorganization on January 13, 2020, in Wilmington|0.8){record_delimiter}
("mcss"|The Supporting Lenders committed $400 million in DIP financing under the Restructuring Support Agreement|[SUPPORTING LENDERS,$400 MILLION,DIP FINANCING,RESTRUCTURING SUPPORT AGREEMENT]|0.95){record_delimiter}
("entity"|SUPPORTING LENDERS|organization|The Supporting Lenders committed $400 million in DIP financing|0.9){record_delimiter}
("entity"|$400 MILLION|category|The Supporting Lenders committed $400 million in DIP financing|0.9){record_delimiter}
("entity"|DIP FINANCING|event|The Supporting Lenders committed $400 million in DIP financing|0.9){record_delimiter}
("entity"|RESTRUCTURING SUPPORT AGREEMENT|category|The Supporting Lenders committed $400 million in DIP financing under the Restructuring Support Agreement|0.85)
```

**Example 2:**

```text
Entity_types: [organization, person, geo, event, category]

Text:
Apple partnered with OpenAI to integrate GPT-4 into iPhone, announced at WWDC on June 10, 2024. Microsoft, through its Azure cloud platform, provides computing infrastructure for OpenAI's models and has invested $13 billion in the company.
################
Output:
("mcss"|Apple partnered with OpenAI to integrate GPT-4 into iPhone, announced at WWDC on June 10, 2024|[APPLE,OPENAI,GPT-4,IPHONE,WWDC,JUNE 10 2024]|0.9){record_delimiter}
("entity"|APPLE|organization|Apple partnered with OpenAI to integrate GPT-4 into iPhone|0.95){record_delimiter}
("entity"|OPENAI|organization|Apple partnered with OpenAI to integrate GPT-4 into iPhone|0.9){record_delimiter}
("entity"|GPT-4|category|Apple partnered with OpenAI to integrate GPT-4 into iPhone|0.9){record_delimiter}
("entity"|IPHONE|category|Apple partnered with OpenAI to integrate GPT-4 into iPhone|0.85){record_delimiter}
("entity"|WWDC|event|Apple partnered with OpenAI to integrate GPT-4 into iPhone, announced at WWDC|0.8){record_delimiter}
("entity"|JUNE 10 2024|event|Apple partnered with OpenAI to integrate GPT-4 into iPhone, announced at WWDC on June 10, 2024|0.8){record_delimiter}
("mcss"|Microsoft, through its Azure cloud platform, provides computing infrastructure for OpenAI's models and has invested $13 billion in the company|[MICROSOFT,AZURE,OPENAI,$13 BILLION]|0.9){record_delimiter}
("entity"|MICROSOFT|organization|Microsoft provides computing infrastructure for OpenAI's models through Azure|0.9){record_delimiter}
("entity"|AZURE|category|Microsoft provides computing infrastructure for OpenAI's models through Azure|0.85){record_delimiter}
("entity"|OPENAI|organization|Microsoft provides computing infrastructure for OpenAI's models and has invested $13 billion|0.9){record_delimiter}
("entity"|$13 BILLION|category|Microsoft has invested $13 billion in OpenAI|0.85)
```

### 2.3 输出量对比

| Prompt 类型 | 同一段文本的输出记录数 |
|------------|---------------------|
| 当前单纯复形 Prompt | 30条（8个0-simplex + 12个1-simplex + 8个2-simplex + 2个3-simplex） |
| MSG Prompt | **10条**（2个mcss + 8个entity） |
| 减少比例 | **67%** |

### 2.4 提取流程（代码层面）

```
1. 文本分块 → chunk_list
2. 并发 LLM 调用 → 每个 chunk 独立提取
3. 解析 LLM 输出:
   - "mcss" 记录 → 解析为 MSG（描述 + 实体列表 + 完备性分数）
   - "entity" 记录 → 解析为实体（名称 + 类型 + 原文 + 重要性）
   - 每个 entity 自动关联到其前最近的 MSG
4. 跨 chunk 去重:
   - 同名实体 → 描述去重拼接，类型多数投票，频率累加
   - 相同实体列表的 MSG → 合并描述，importance 取最大值
5. 批量摘要 → 对描述过长的实体和MSG进行摘要
6. 重要性过滤 → 低于阈值的实体和MSG被过滤
```

***

## 三、阶段2：本地代码转换

### 3.1 转换算法总览

```
LLM 输出（MSG + 实体）
    ↓
Step 1: 构建信息层（MSG → 极大单纯形）
    ↓
Step 2: 构建二部图关联（实体 ↔ MSG）
    ↓ Step 2a: 计算每个实体的 coboundary（所属MSG列表）
    ↓ Step 2b: 计算每个MSG的 boundary（包含的实体列表）
    ↓
Step 3: 计算简化Laplacian
    ↓ L₀ = BᵀB（实体-实体Laplacian）
    ↓ L_msg = BBᵀ（MSG-MSG Laplacian）
    ↓
Step 4: 计算嵌入向量
    ↓
输出: 信息层MSG + 0-单纯形 + 二部图关联 + Laplacian
```

**核心简化**：不再显式创建1-单纯形（边）和2-单纯形（三角形）。MSG之间通过共享实体天然连通，二部图关联足以支撑拓扑扩散。

### 3.2 Step 1：构建信息层

每个 MSG 直接转换为一个极大单纯形：

```python
def build_information_layer(msgs, entities, chunk_key):
    information_layer = []
    for msg in msgs:
        entity_list = sorted(msg["entities"])
        if len(entity_list) < 2:
            continue

        simplex = {
            "id": compute_mdhash_id(str(entity_list),
                                    prefix=f"simplex-{len(entity_list)-1}-"),
            "entities": entity_list,
            "dimension": len(entity_list) - 1,
            "is_maximal": True,
            "layer": "information",
            "description": msg["text"],
            "completeness": msg["completeness"],
            "source_id": chunk_key,
            "importance": max(
                entities[e]["importance"] for e in entity_list
                if e in entities
            ),
        }
        information_layer.append(simplex)
    return information_layer
```

### 3.3 Step 2：构建二部图关联

#### Step 2a：实体的 coboundary（所属MSG列表）

```python
def build_entity_coboundary(entities, information_layer):
    for name in entities:
        entities[name]["coboundary"] = []

    for msg in information_layer:
        for entity_name in msg["entities"]:
            if entity_name in entities:
                entities[entity_name]["coboundary"].append(msg["id"])

    return entities
```

**含义**：实体的coboundary = 该实体参与的所有MSG。这是拓扑扩散的核心——从实体出发，通过coboundary找到相关MSG。

#### Step 2b：MSG的 boundary（包含的实体列表）

```python
def build_msg_boundary(information_layer, entities):
    for msg in information_layer:
        msg["boundary"] = []
        for entity_name in msg["entities"]:
            entity_id = compute_mdhash_id(entity_name, prefix="simplex-0-")
            msg["boundary"].append(entity_id)

    return information_layer
```

**含义**：MSG的boundary = 该MSG包含的所有实体ID。这是拓扑扩散的另一个方向——从MSG出发，通过boundary找到相关实体。

### 3.4 Step 3：计算简化Laplacian

定义**直接边界映射** ∂: C_max → C₀，从极大单纯形直接映射到0-单纯形，跳过所有中间维度。

```python
def build_bipartite_laplacian(information_layer, entities):
    entity_names = sorted(entities.keys())
    msg_ids = [msg["id"] for msg in information_layer]
    entity_index = {name: i for i, name in enumerate(entity_names)}
    msg_index = {mid: i for i, mid in enumerate(msg_ids)}

    n_entities = len(entity_names)
    n_msgs = len(msg_ids)

    # 关联矩阵 B: (n_msgs × n_entities)
    # B[i,j] = 1 若实体j属于MSG_i
    B = np.zeros((n_msgs, n_entities), dtype=np.float32)
    for msg in information_layer:
        i = msg_index[msg["id"]]
        for entity_name in msg["entities"]:
            if entity_name in entity_index:
                j = entity_index[entity_name]
                B[i, j] = 1.0

    # 实体-实体Laplacian: L₀ = BᵀB
    # L₀[i,j] = 实体i和j共同出现的MSG数量
    # L₀[i,i] = 实体i出现的MSG数量（度数）
    L_entity = B.T @ B

    # MSG-MSG Laplacian: L_msg = BBᵀ
    # L_msg[i,j] = MSG_i和MSG_j共享的实体数量
    # L_msg[i,i] = MSG_i包含的实体数量（度数）
    L_msg = B @ B.T

    return L_entity, L_msg, entity_index, msg_index
```

**L_entity 的含义**：
- 对角线 L_entity[i,i] = 实体i出现在多少个MSG中（度数，即该实体的"枢纽性"）
- 非对角线 L_entity[i,j] = 实体i和j共同出现在多少个MSG中（共现强度）

**L_msg 的含义**：
- 对角线 L_msg[i,i] = MSG_i包含多少个实体（度数，即该MSG的"丰富度"）
- 非对角线 L_msg[i,j] = MSG_i和MSG_j共享多少个实体（连通强度）

**与Hodge Laplacian的关系**：传统Hodge Laplacian L₀ = B₁B₁ᵀ，其中B₁是1-单纯形→0-单纯形的边界矩阵。我们的L_entity = BᵀB，其中B是MSG→实体的关联矩阵。这是Hodge Laplacian在"只保留0-单纯形和极大单纯形"时的自然简化——跳过中间维度，直接建立极大单纯形与0-单纯形之间的映射。

### 3.5 导航路径的隐式表达

去掉显式边和三角形后，导航层如何发挥"连接不同MSG的路径"功能？答案是：**共享实体就是路径，Laplacian矩阵就是邻接矩阵**。

#### 原方案：显式边作为路径

```text
APPLE ──{APPLE,OPENAI}──→ OPENAI ──{OPENAI,MICROSOFT}──→ MICROSOFT
        显式1-单纯形              显式1-单纯形
```

需要先创建边 {APPLE,OPENAI} 和 {OPENAI,MICROSOFT}，信号才能沿边扩散。

#### 新方案：共享实体即路径

```text
APPLE ── MSG₁ ── OPENAI ── MSG₂ ── MICROSOFT
        (含APPLE,    (含OPENAI,    (含MICROSOFT,
         OPENAI,..)   MICROSOFT,..) AZURE,..)
```

OPENAI本身就是桥梁——它同时属于MSG₁和MSG₂（coboundary=[MSG₁, MSG₂]），信号从APPLE出发，通过MSG₁到达OPENAI，再通过OPENAI的coboundary到达MSG₂，最终到达MICROSOFT。

#### L_entity 如何实现实体间导航

L_entity的非对角线元素 L_entity[i,j] = 实体i和j共同出现的MSG数量，**等价于实体间的邻接矩阵**：

```text
         APPLE  OPENAI  GPT-4  IPHONE  MICROSOFT  AZURE  $13B
APPLE      1      1       1      1        0        0      0
OPENAI     1      2       1      1        1        1      1    ← 度数=2，枢纽
GPT-4      1      1       1      1        0        0      0
IPHONE     1      1       1      1        0        0      0
MICROSOFT  0      1       0      0        1        1      1
AZURE      0      1       0      0        1        1      1
$13B       0      1       0      0        1        1      1
```

- APPLE和MICROSOFT之间 = 0（无直接共现，不连通）
- APPLE和OPENAI之间 = 1（共现于MSG₁，连通）
- OPENAI和MICROSOFT之间 = 1（共现于MSG₂，连通）

扩散路径：APPLE →(1跳)→ OPENAI →(1跳)→ MICROSOFT，**2跳到达**。这和原方案中"APPLE→边{APPLE,OPENAI}→OPENAI→边{OPENAI,MICROSOFT}→MICROSOFT"的效果完全一样，只是边被L_entity的非零元素隐式替代了。

#### L_msg 如何实现MSG间导航

L_msg的非对角线元素 L_msg[i,j] = MSG_i和MSG_j共享的实体数量：

```text
         MSG₁  MSG₂
MSG₁       5     1      ← 共享OPENAI
MSG₂       1     4
```

扩散路径：MSG₁ →(共享OPENAI)→ MSG₂，**1跳到达**。这等价于原方案中"MSG₁的coboundary→边{OPENAI,...}→coboundary→MSG₂"的路径，但不需要显式的中间边。

#### 显式 vs 隐式对比

| 方面 | 原方案（显式边） | 新方案（隐式路径） |
|------|----------------|------------------|
| 路径载体 | 1-单纯形 {A,B} | L_entity[A,B] > 0 |
| MSG间路径 | 边的coboundary | L_msg[MSG₁,MSG₂] > 0 |
| 桥梁机制 | 共享边连接两个MSG | 共享实体连接两个MSG |
| 扩散方式 | 沿显式边传播 | 沿Laplacian非零元素传播 |
| 扩散效果 | APPLE→OPENAI→MICROSOFT | APPLE→OPENAI→MICROSOFT（相同） |

**核心结论**：共享实体就是路径，L_entity/L_msg就是邻接矩阵，扩散沿矩阵非零元素传播——导航功能完全保留，只是从"显式建边"变成了"隐式共现"。原方案中边 {APPLE, OPENAI} 的存在是因为APPLE和OPENAI共现于MSG₁；新方案中 L_entity[APPLE, OPENAI] = 1 也是因为APPLE和OPENAI共现于MSG₁。两者表达的是同一个事实，只是表示方式不同。

### 3.6 Step 4：计算嵌入向量

```python
async def compute_embeddings(information_layer, entities, embedding_func):
    # 信息层MSG的嵌入: 实体名称拼接 + 描述 → 嵌入
    for msg in information_layer:
        text = " ".join(msg["entities"]) + " " + msg.get("description", "")
        emb = await embedding_func([text])
        msg["embedding"] = emb[0].tolist()

    # 0-单纯形的嵌入: 实体名称 + 描述 → 嵌入
    for name, data in entities.items():
        text = f"{name} {data.get('description', '')}"
        emb = await embedding_func([text])
        data["embedding"] = emb[0].tolist()
```

### 3.7 完整转换示例

**输入**：

```text
文本1: "Apple与OpenAI合作，将GPT-4集成到iPhone中，于2024年6月发布。"
文本2: "Microsoft通过Azure云服务为OpenAI提供算力支持，投资了130亿美元。"

LLM提取:
  MSG₁ = [APPLE, OPENAI, GPT-4, IPHONE, JUNE 2024]
  MSG₂ = [MICROSOFT, AZURE, OPENAI, $13 BILLION]
  实体: APPLE, OPENAI, GPT-4, IPHONE, JUNE 2024, MICROSOFT, AZURE, $13 BILLION
```

**输出**：

```text
信息层:
  MSG₁: {APPLE, OPENAI, GPT-4, IPHONE, JUNE 2024}  → 4-单纯形
    is_maximal=True, layer="information"
    description="Apple与OpenAI合作，将GPT-4集成到iPhone中"
    source_id="chunk-0x3f2a"
    embedding=[0.12, -0.34, ...]
    importance=0.95
    boundary=[simplex-0-APPLE, simplex-0-OPENAI, simplex-0-GPT-4, simplex-0-IPHONE, simplex-0-JUNE_2024]

  MSG₂: {MICROSOFT, AZURE, OPENAI, $13 BILLION}  → 3-单纯形
    is_maximal=True, layer="information"
    description="Microsoft通过Azure云服务为OpenAI提供算力支持"
    source_id="chunk-0x7b1c"
    embedding=[0.45, 0.23, ...]
    importance=0.90
    boundary=[simplex-0-MICROSOFT, simplex-0-AZURE, simplex-0-OPENAI, simplex-0-$13_BILLION]

0-单纯形:
  APPLE:    coboundary=[MSG₁], type=organization, importance=0.95
  OPENAI:   coboundary=[MSG₁, MSG₂], type=organization, importance=0.9  ← 桥梁实体
  GPT-4:    coboundary=[MSG₁], type=category, importance=0.9
  IPHONE:   coboundary=[MSG₁], type=category, importance=0.85
  JUNE 2024: coboundary=[MSG₁], type=event, importance=0.8
  MICROSOFT: coboundary=[MSG₂], type=organization, importance=0.9
  AZURE:    coboundary=[MSG₂], type=category, importance=0.85
  $13 BILLION: coboundary=[MSG₂], type=category, importance=0.85

二部图Laplacian:
  L_entity (8×8):
    APPLE-OPENAI = 1 (共现于MSG₁)
    OPENAI-MICROSOFT = 1 (共现于MSG₂，通过OPENAI桥接)
    APPLE-MICROSOFT = 0 (无共同MSG，不直接连通)
    OPENAI度数 = 2 (出现在2个MSG中，是枢纽)

  L_msg (2×2):
    MSG₁-MSG₂ = 1 (共享实体OPENAI)
    MSG₁度数 = 5, MSG₂度数 = 4
```

**二部图拓扑图**：

```text
  APPLE ────── MSG₁ ────── OPENAI ────── MSG₂ ────── MICROSOFT
  GPT-4 ────── MSG₁ ────── OPENAI ────── MSG₂ ────── AZURE
  IPHONE ───── MSG₁                        MSG₂ ───── $13 BILLION
  JUNE 2024 ── MSG₁

  OPENAI是桥梁实体：coboundary=[MSG₁, MSG₂]
  从APPLE出发: APPLE → MSG₁ → OPENAI → MSG₂ → MICROSOFT（2跳到达）
```

***

## 四、阶段3：存储与索引

### 4.1 存储架构

| 存储类型 | 存储内容 | 说明 |
|---------|---------|------|
| **entities_vdb** | 0-单纯形（实体） | 实体名称+描述的向量检索 |
| **relationships_vdb** | 信息层MSG（极大单纯形） | MSG描述+实体的向量检索 |
| **SimplexStorage** | 信息层MSG + 0-单纯形 + Laplacian缓存 | 完整拓扑结构 |
| **text_chunks** | 原始文本块 | 通过source_id关联 |

**简化**：SimplexStorage只存0-单纯形和极大单纯形，不再存1-单纯形和2-单纯形。

### 4.2 entities_vdb 存储

```python
for name, data in entities.items():
    content = f"{name} {data.get('description', '')} {data.get('additional_properties', '')}".strip()
    entities_vdb.upsert({
        "id": compute_mdhash_id(name, prefix="simplex-0-"),
        "content": content,
        "meta": {
            "entity_name": name,
            "entity_type": data["type"],
            "description": data["description"],
            "frequency": data.get("frequency", 1),
            "source_id": data.get("source_id", ""),
            "importance": data.get("importance", 0.5),
            "coboundary": data.get("coboundary", []),
        }
    })
```

### 4.3 relationships_vdb 存储

**只存信息层的MSG。**

```python
for msg in information_layer:
    content = " ".join(filter(None, [
        msg.get("description", ""),
        " ".join(msg["entities"]),
    ]))
    relationships_vdb.upsert({
        "id": msg["id"],
        "content": content,
        "meta": {
            "id_set": msg["entities"],
            "dimension": msg["dimension"],
            "description": msg["description"],
            "source_id": msg["source_id"],
            "importance": msg["importance"],
            "is_maximal": True,
            "boundary": msg.get("boundary", []),
        }
    })
```

### 4.4 SimplexStorage 存储

```python
all_simplices = {}
for s in information_layer:
    all_simplices[s["id"]] = s
for name, data in entities.items():
    sid = compute_mdhash_id(name, prefix="simplex-0-")
    all_simplices[sid] = {"id": sid, "entities": [name], "dimension": 0, **data}

simplex_storage.upsert_batch(all_simplices)

# 缓存Laplacian矩阵
simplex_storage.cache_laplacian("L_entity", L_entity)
simplex_storage.cache_laplacian("L_msg", L_msg)
simplex_storage.cache_index("entity_index", entity_index)
simplex_storage.cache_index("msg_index", msg_index)
```

### 4.5 存储结构对比

| 数据 | 当前方法（强闭包） | 新方法（松弛闭包） |
|------|------------------|------------------|
| entities_vdb | 所有0-单纯形 | 所有0-单纯形（相同） |
| relationships_vdb | 所有维度的单纯形 | **只有信息层MSG** |
| SimplexStorage | 所有维度的单纯形 | **0-单纯形 + MSG + Laplacian缓存** |
| VDB记录数 | 30条（示例文本） | **10条**（减少67%） |
| 需要计算的子复形 | C(n,2)边 + C(n,3)三角形 | **0**（无需计算） |

***

## 五、阶段4：检索

### 5.1 检索流程总览（2×2结构）

```
查询
    ↓
Step 1: 提取查询实体
    ↓
Step 2: 加载Laplacian和索引
    ↓
         ┌──────────────────────────────────────────────────┐
         │              2×2 并行检索                         │
         │                                                  │
         │         实体节点              复形（MSG）          │
         │  ┌──────────────────┬──────────────────┐         │
         │语│ 语义实体检索      │ 语义MSG检索       │         │
         │义│ entities_vdb     │ relationships_vdb │         │
         │  │ 向量搜索          │ 向量搜索          │         │
         │  ├──────────────────┼──────────────────┤         │
         │拓│ 拓扑实体扩散      │ 拓扑MSG扩散       │         │
         │扑│ L₀扩散           │ L_msg扩散         │         │
         │  │ 实体→MSG→实体    │ MSG→实体→MSG     │         │
         │  └──────────────────┴──────────────────┘         │
         └──────────────────────────────────────────────────┘
    ↓
Step 3: 语义加权融合（4路结果合并）
    ↓
Step 4: 文本块收集（从MSG的source_id提取）
    ↓
Step 5: 排序 + 截断
```

### 5.2 Step 1：提取查询实体

使用查询实体提取 Prompt（与当前逻辑一致）：

```text
查询: "Apple的AI战略如何影响Microsoft的云业务？"
提取结果:
  entities: [APPLE, MICROSOFT]
```

### 5.3 Step 2：加载Laplacian和索引

```python
L_entity = simplex_storage.load_laplacian("L_entity")
L_msg = simplex_storage.load_laplacian("L_msg")
entity_index = simplex_storage.load_index("entity_index")
msg_index = simplex_storage.load_index("msg_index")
```

### 5.4 2×2 并行检索详解

#### 语义 × 实体节点：entities_vdb 向量搜索

```python
async def semantic_entity_retrieve(query, entities_vdb, top_k=20):
    results = entities_vdb.search(query, top_k=top_k)
    entity_scores = {}
    for result in results:
        entity_name = result.meta.get("entity_name", "")
        entity_scores[entity_name] = result.score
    return entity_scores
```

**输出**：与查询语义最相关的实体列表及其相似度分数。

#### 语义 × 复形（MSG）：relationships_vdb 向量搜索

```python
async def semantic_msg_retrieve(query, relationships_vdb, top_k=20):
    results = relationships_vdb.search(query, top_k=top_k)
    msg_scores = {}
    for result in results:
        msg_id = result.id
        msg_scores[msg_id] = result.score
    return msg_scores
```

**输出**：与查询语义最相关的MSG列表及其相似度分数。

#### 拓扑 × 实体节点：L₀ 扩散

```python
async def topology_entity_diffuse(
    seed_entities, L_entity, entity_index, alpha=0.15, max_steps=3
):
    n = L_entity.shape[0]
    idx_to_name = {v: k for k, v in entity_index.items()}

    # 初始化信号：种子实体=1.0，其余=0
    x = np.zeros(n, dtype=np.float64)
    for name in seed_entities:
        if name in entity_index:
            x[entity_index[name]] = 1.0

    # 归一化度矩阵
    degrees = np.diag(L_entity)
    degrees[degrees == 0] = 1.0

    # 扩散: x(t+1) = x(t) - α * D⁻¹ * L₀ * x(t)
    for _ in range(max_steps):
        x_new = x - alpha * (L_entity @ x) / degrees
        x_new = np.maximum(x_new, 0)
        x = x_new

    # 收集结果
    entity_scores = {}
    for idx in range(n):
        if x[idx] > 0.01:
            entity_scores[idx_to_name[idx]] = float(x[idx])

    return entity_scores
```

**含义**：从种子实体出发，信号通过L₀传播——实体i的信号流向与它共现于同一MSG的实体j（L₀[i,j] > 0 表示共现）。每步衰减α，最多扩散max_steps步。

**示例**：
```text
种子: APPLE
Step 0: APPLE=1.0
Step 1: APPLE→MSG₁→OPENAI,GPT-4,IPHONE,JUNE 2024 (L₀[APPLE,*] > 0的实体)
Step 2: OPENAI→MSG₂→MICROSOFT,AZURE,$13 BILLION (OPENAI是桥梁)
Step 3: 信号继续衰减扩散...

结果: APPLE=0.85, OPENAI=0.72, GPT-4=0.65, MICROSOFT=0.42, ...
```

#### 拓扑 × 复形（MSG）：L_msg 扩散

```python
async def topology_msg_diffuse(
    seed_msg_ids, L_msg, msg_index, alpha=0.15, max_steps=2
):
    n = L_msg.shape[0]
    idx_to_id = {v: k for k, v in msg_index.items()}

    # 初始化信号：种子MSG=1.0，其余=0
    y = np.zeros(n, dtype=np.float64)
    for msg_id in seed_msg_ids:
        if msg_id in msg_index:
            y[msg_index[msg_id]] = 1.0

    # 归一化度矩阵
    degrees = np.diag(L_msg)
    degrees[degrees == 0] = 1.0

    # 扩散: y(t+1) = y(t) - α * D⁻¹ * L_msg * y(t)
    for _ in range(max_steps):
        y_new = y - alpha * (L_msg @ y) / degrees
        y_new = np.maximum(y_new, 0)
        y = y_new

    # 收集结果
    msg_scores = {}
    for idx in range(n):
        if y[idx] > 0.01:
            msg_scores[idx_to_id[idx]] = float(y[idx])

    return msg_scores
```

**含义**：从种子MSG出发，信号通过L_msg传播——MSG_i的信号流向与它共享实体的MSG_j（L_msg[i,j] > 0 表示共享实体）。共享实体越多，连接越强。

**示例**：
```text
种子: MSG₁ (通过语义检索或实体coboundary获得)
Step 0: MSG₁=1.0
Step 1: MSG₁→(共享OPENAI)→MSG₂ (L_msg[MSG₁,MSG₂] = 1)
Step 2: 信号继续衰减...

结果: MSG₁=0.85, MSG₂=0.42
```

### 5.5 Step 3：语义加权融合

**核心思想**：拓扑扩散的结果必须经过语义过滤，避免引入无关冗余信息。

```python
def fuse_results(
    sem_entity_scores,    # 语义×实体
    sem_msg_scores,       # 语义×MSG
    topo_entity_scores,   # 拓扑×实体
    topo_msg_scores,      # 拓扑×MSG
    query_embedding,      # 查询嵌入
    entities,             # 实体数据（含嵌入）
    information_layer,    # MSG数据（含嵌入）
    w_sem=0.6,            # 语义权重
    w_topo=0.4,           # 拓扑权重
):
    # --- 实体融合 ---
    all_entity_names = set(sem_entity_scores) | set(topo_entity_scores)
    fused_entities = {}

    for name in all_entity_names:
        s_sem = sem_entity_scores.get(name, 0.0)
        s_topo = topo_entity_scores.get(name, 0.0)

        # 拓扑得分需要语义加权：拓扑发现但语义无关的实体降权
        if s_topo > 0 and s_sem == 0:
            # 拓扑发现但语义未命中 → 用嵌入相似度替代
            entity_emb = entities[name].get("embedding")
            if entity_emb is not None:
                s_sem = float(
                    np.dot(entity_emb, query_embedding)
                    / (np.linalg.norm(entity_emb) * np.linalg.norm(query_embedding) + 1e-8)
                )
            s_topo *= 0.5  # 纯拓扑发现额外降权

        fused_score = w_sem * s_sem + w_topo * s_topo
        if fused_score > 0.01:
            fused_entities[name] = {
                "score": fused_score,
                "sem_score": s_sem,
                "topo_score": s_topo,
                "source": "collision" if (s_sem > 0 and s_topo > 0)
                          else "sem_only" if s_sem > 0
                          else "topo_only",
            }

    # --- MSG融合 ---
    all_msg_ids = set(sem_msg_scores) | set(topo_msg_scores)
    fused_msgs = {}

    for msg_id in all_msg_ids:
        s_sem = sem_msg_scores.get(msg_id, 0.0)
        s_topo = topo_msg_scores.get(msg_id, 0.0)

        # 拓扑发现但语义未命中的MSG → 用嵌入相似度替代
        if s_topo > 0 and s_sem == 0:
            msg_data = next((m for m in information_layer if m["id"] == msg_id), None)
            if msg_data and msg_data.get("embedding") is not None:
                msg_emb = np.array(msg_data["embedding"])
                s_sem = float(
                    np.dot(msg_emb, query_embedding)
                    / (np.linalg.norm(msg_emb) * np.linalg.norm(query_embedding) + 1e-8)
                )
            s_topo *= 0.5

        fused_score = w_sem * s_sem + w_topo * s_topo
        if fused_score > 0.01:
            fused_msgs[msg_id] = {
                "score": fused_score,
                "sem_score": s_sem,
                "topo_score": s_topo,
                "source": "collision" if (s_sem > 0 and s_topo > 0)
                          else "sem_only" if s_sem > 0
                          else "topo_only",
            }

    return fused_entities, fused_msgs
```

**三类别含义**：

| 类别 | 含义 | 得分特点 |
|------|------|---------|
| **collision** | 语义+拓扑都支持 | 最可靠，双路验证 |
| **sem_only** | 仅语义支持 | 直接相关但拓扑未达 |
| **topo_only** | 仅拓扑支持 | 拓扑可达但语义弱，额外降权50% |

**冗余控制**：topo_only类别的结果会被额外降权，且必须通过嵌入相似度验证（>0才保留），确保拓扑扩散不会引入完全无关的信息。

### 5.6 Step 4：文本块收集

**只从MSG的source_id提取文本块**，实体不直接提供文本块。

```python
def collect_text_chunks(fused_msgs, information_layer, text_chunks,
                        entities_vdb, relationships_vdb, query, top_k=60):
    chunks = []
    seen_chunk_ids = set()

    # 路径1: 从融合后的MSG直接提取source_id（最高优先级）
    sorted_msgs = sorted(fused_msgs.items(), key=lambda x: x[1]["score"], reverse=True)
    for msg_id, info in sorted_msgs:
        msg_data = next((m for m in information_layer if m["id"] == msg_id), None)
        if msg_data is None:
            continue
        source_id = msg_data.get("source_id", "")
        if source_id and source_id not in seen_chunk_ids and source_id in text_chunks:
            seen_chunk_ids.add(source_id)
            chunks.append({
                "chunk_id": source_id,
                "text": text_chunks[source_id],
                "source": "msg_fused",
                "score": info["score"],
            })

    # 路径2: 从entities_vdb补充检索
    entity_results = entities_vdb.search(query, top_k=10)
    for result in entity_results:
        source_id = result.meta.get("source_id", "")
        if source_id and source_id not in seen_chunk_ids:
            seen_chunk_ids.add(source_id)
            chunks.append({
                "chunk_id": source_id,
                "text": text_chunks.get(source_id, ""),
                "source": "entity_vdb",
                "score": result.score * 0.8,
            })

    # 路径3: 从relationships_vdb补充检索
    rel_results = relationships_vdb.search(query, top_k=10)
    for result in rel_results:
        source_id = result.meta.get("source_id", "")
        if source_id and source_id not in seen_chunk_ids:
            seen_chunk_ids.add(source_id)
            chunks.append({
                "chunk_id": source_id,
                "text": text_chunks.get(source_id, ""),
                "source": "relation_vdb",
                "score": result.score * 0.8,
            })

    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return sorted_chunks[:top_k]
```

### 5.7 检索完整示例

**查询**: "Apple的AI战略如何影响Microsoft的云业务？"

```text
Step 1: 提取查询实体 [APPLE, MICROSOFT]

Step 2: 加载Laplacian

Step 3: 2×2并行检索

  语义×实体: entities_vdb搜索
    APPLE → sim=0.92
    MICROSOFT → sim=0.88
    OPENAI → sim=0.75
    GPT-4 → sim=0.60

  语义×MSG: relationships_vdb搜索
    MSG₁ → sim=0.85
    MSG₂ → sim=0.72

  拓扑×实体: L₀扩散（种子=APPLE, MICROSOFT）
    APPLE=1.0 → OPENAI=0.72, GPT-4=0.65, IPHONE=0.60, JUNE 2024=0.55
    MICROSOFT=1.0 → AZURE=0.70, OPENAI=0.68, $13 BILLION=0.62
    合并: APPLE=1.0, MICROSOFT=1.0, OPENAI=0.72, AZURE=0.70, GPT-4=0.65, ...

  拓扑×MSG: L_msg扩散（种子=MSG₁, MSG₂，通过实体coboundary获得）
    MSG₁=1.0 → MSG₂=0.42（共享OPENAI）
    MSG₂=1.0 → MSG₁=0.35（共享OPENAI）

Step 4: 语义加权融合

  实体融合:
    APPLE: collision (sem=0.92, topo=1.0) → 0.6*0.92 + 0.4*1.0 = 0.952
    MICROSOFT: collision (sem=0.88, topo=1.0) → 0.6*0.88 + 0.4*1.0 = 0.928
    OPENAI: collision (sem=0.75, topo=0.72) → 0.6*0.75 + 0.4*0.72 = 0.738
    AZURE: topo_only (sem=0, topo=0.70) → 嵌入验证sim=0.45 → 0.6*0.45 + 0.4*0.70*0.5 = 0.410
    GPT-4: topo_only (sem=0, topo=0.65) → 嵌入验证sim=0.38 → 0.6*0.38 + 0.4*0.65*0.5 = 0.359
    IPHONE: topo_only (sem=0, topo=0.60) → 嵌入验证sim=0.25 → 0.6*0.25 + 0.4*0.60*0.5 = 0.270

  MSG融合:
    MSG₁: collision (sem=0.85, topo=1.0) → 0.6*0.85 + 0.4*1.0 = 0.910
    MSG₂: collision (sem=0.72, topo=1.0) → 0.6*0.72 + 0.4*1.0 = 0.832

Step 5: 文本块收集
  MSG₁.source_id → chunk-0x3f2a: "Apple与OpenAI合作，将GPT-4集成到iPhone中..."
  MSG₂.source_id → chunk-0x7b1c: "Microsoft通过Azure云服务为OpenAI提供算力支持..."

Step 6: 排序
  [chunk-0x3f2a (score=0.910), chunk-0x7b1c (score=0.832)]
```

***

## 六、阶段5：上下文组装与答案生成

### 6.1 上下文组装

将检索结果组装为结构化上下文，输入给 LLM：

```python
def assemble_context(fused_msgs, fused_entities, information_layer,
                     entities, text_chunks, token_limit=8000):
    context_parts = []

    # Sources段（最高优先级）：按融合得分排序的原始文本块
    sources_text = ""
    sorted_msgs = sorted(fused_msgs.items(), key=lambda x: x[1]["score"], reverse=True)
    for msg_id, info in sorted_msgs:
        msg_data = next((m for m in information_layer if m["id"] == msg_id), None)
        if msg_data is None:
            continue
        source_id = msg_data.get("source_id", "")
        chunk_text = text_chunks.get(source_id, "")
        if chunk_text:
            sources_text += chunk_text + "\n"
    context_parts.append(f"---Sources---\n{sources_text[:token_limit]}")

    # Entities段：只保留与Sources直接相关的实体
    sources_entity_names = set()
    for name, info in fused_entities.items():
        if info["source"] in ("collision", "sem_only"):
            sources_entity_names.add(name)

    entity_lines = []
    for name in sources_entity_names:
        data = entities.get(name, {})
        entity_lines.append(
            f"{name}|{data.get('type', '')}|{data.get('description', '')}"
        )
    context_parts.append(f"---Entities---\n" + "\n".join(entity_lines[:15]))

    # Simplices段：只保留MSG（信息层）
    simplex_lines = []
    for msg_id, info in sorted_msgs:
        msg_data = next((m for m in information_layer if m["id"] == msg_id), None)
        if msg_data is None:
            continue
        entities_str = ",".join(msg_data["entities"])
        simplex_lines.append(
            f"dim={msg_data['dimension']}|[{entities_str}]|{msg_data['description']}"
        )
    context_parts.append(f"---Simplices---\n" + "\n".join(simplex_lines[:10]))

    return "\n\n".join(context_parts)
```

### 6.2 答案生成 Prompt

```text
---Role---
You are a helpful assistant responding to questions about data in the tables provided.
The data is organized as a Maximal Semantic Group (MSG) structure — each MSG represents
a complete semantic event involving multiple entities.

{prompt_instructions}

---MSG Structure---
The context contains three types of data tables:
- **Sources**: Original document passages — the primary source of truth.
- **Entities**: Individual entities extracted from the documents, each with a type and description.
- **Simplices**: Maximal Semantic Groups (MSGs), each representing a complete semantic event:
  - A 1-MSG (2 entities): A pairwise relationship.
  - A 2-MSG (3 entities): A three-way relationship — three entities jointly participate in an event.
  - A 3-MSG (4 entities): A four-way relationship — four entities jointly participate in an event.
  Higher-dimensional MSGs represent more complete, more specific events.
  The `is_seed` flag marks entities/simplices directly matched to the query.

---Response Rules---
1. Prioritize extracting facts from the Sources table.
2. Use Simplices to understand how entities connect — higher-dimensional MSGs indicate
   stronger, more specific relationships.
3. Do NOT supplement knowledge beyond the Sources.
4. For multi-term questions, address each term based on the Sources separately.
```

### 6.3 答案生成流程

```python
async def generate_answer(query, context, llm_model_func):
    prompt = PROMPTS["topology_response_system_prompt"].format(
        prompt_instructions=context
    )
    response = await llm_model_func(prompt, query)
    return response
```

***

## 七、注入攻击防御机制

### 7.1 攻击模型

攻击者在文本中注入恶意实体 MALICIOUS，试图让它与合法实体关联。

### 7.2 三层防御

| 防御层 | 机制 | 效果 |
|-------|------|------|
| **构建时** | MSG由LLM语义理解提取 | 恶意实体难以进入合法MSG |
| **扩散时** | L₀扩散依赖共现关系 | 恶意实体与合法实体无共现，信号不传播 |
| **检索时** | 检索目标只有MSG | 恶意实体不在任何合法MSG中 |

### 7.3 具体攻击场景分析

**攻击1**：在文本中插入 "MALICIOUS and APPLE cooperate..."

```text
LLM 可能将 MALICIOUS 和 APPLE 放入同一个MSG
→ MSG = {MALICIOUS, APPLE}
→ L₀[MALICIOUS, APPLE] = 1（共现于同一MSG）

但:
→ MALICIOUS 在其他任何MSG中都不出现
→ MALICIOUS 的频率极低（只出现1次）
→ MALICIOUS 的 importance 极低
→ MALICIOUS 的 coboundary 只包含这1个MSG
→ L₀扩散中 MALICIOUS 的度数极低，信号难以从MALICIOUS扩散出去
→ 即使扩散到MALICIOUS，语义加权融合时嵌入相似度极低，被大幅降权
```

**攻击2**：在文本中插入 "MALICIOUS, APPLE, OPENAI cooperate..."

```text
LLM 可能创建 MSG = {MALICIOUS, APPLE, OPENAI}
→ L₀[MALICIOUS, APPLE] = 1, L₀[MALICIOUS, OPENAI] = 1

但:
→ MALICIOUS 的嵌入向量与 APPLE/OPENAI 的嵌入差异大
→ 语义加权融合时，MALICIOUS 的嵌入相似度低
→ L_msg中，该MSG与其他MSG的共享实体只有APPLE/OPENAI
→ 但APPLE/OPENAI在合法MSG中的度数高，信号分散
→ 该恶意MSG的拓扑得分被稀释
```

**攻击3**：大量注入包含 MALICIOUS 的文本

```text
MALICIOUS 的频率提高
→ MALICIOUS 出现在多个MSG中
→ 但这些MSG都是恶意注入的，彼此之间共享MALICIOUS
→ 形成一个"恶意子图"，与合法MSG的连接仅靠少量共享实体
→ L_msg中，恶意MSG与合法MSG的连接权重极低
→ 语义加权融合时，恶意MSG的嵌入与查询无关，被大幅降权
```

### 7.4 拓扑隔离定理（二部图版本）

**定理**：设 G = (V_msg ∪ V_entity, E) 为实体↔MSG二部图，其中 E = {(m, e) | 实体e属于MSG m}。对于恶意实体 m，若 m 仅出现在恶意MSG中，则 m 在 G 中的连通分量只包含恶意实体和恶意MSG。

**证明**：恶意实体 m 的 coboundary 只包含恶意MSG。恶意MSG的 boundary 只包含恶意实体（因为合法实体不出现在恶意MSG中，否则就不是纯恶意MSG）。因此从 m 出发，只能到达恶意实体和恶意MSG。□

**推论**：L₀扩散从合法实体出发，信号不会传播到恶意实体（因为不存在同时包含合法实体和恶意实体的MSG，除非LLM将它们放入同一MSG——这是唯一的攻击面，但LLM的语义理解能力构成第一层防御）。

***

## 八、与当前代码的适配关系

### 8.1 需要改动的模块

| 模块 | 改动类型 | 具体内容 |
|------|---------|---------|
| **prompt.py** | 重写 | entity_extraction Prompt 改为 MSG Prompt |
| **_extraction.py** | 重写解析逻辑 | 从解析0/1/n-simplex改为解析mcss+entity |
| **_extraction.py** | 删除 | 闭包验证函数 `_validate_closure_property` |
| **_extraction.py** | 新增 | MSG→信息层+二部图关联转换逻辑 |
| **_extraction.py** | 新增 | 简化Laplacian计算逻辑 |
| **_retrieval.py** | 重写 | 2×2检索结构（语义/拓扑 × 实体/MSG） |
| **_retrieval.py** | 新增 | L₀扩散和L_msg扩散方法 |
| **_retrieval.py** | 新增 | 语义加权融合方法 |
| **_retrieval.py** | 小改 | 文本块收集只从信息层MSG提取source_id |
| **storage.py** | 小改 | relationships_vdb 只存信息层MSG |
| **storage.py** | 新增 | Laplacian缓存机制 |

### 8.2 不需要改动的模块

| 模块 | 原因 |
|------|------|
| **_simplicial_complex.py** | Laplacian现在由本地代码直接计算，HSC类可保留用于兼容 |
| **_config.py** | 自适应参数系统不需要改动 |

### 8.3 与原方案对比

| 方面 | 原方案（显式边+三角形） | 新方案（二部图） |
|------|----------------------|----------------|
| 导航层结构 | 1-单纯形 + 2-单纯形 | 无显式子复形，二部图关联 |
| Laplacian | Hodge Laplacian (L₀, L₁) | 简化Laplacian (L_entity, L_msg) |
| 扩散方式 | L₀扩散 + L₁扩散 + 上边界收缩 | L₀扩散 + L_msg扩散 |
| 跨MSG连接 | 计算余弦相似度建边 | 共享实体天然桥接 |
| 计算量 | C(n,2)余弦相似度 + C(n,3)三角形 | 仅矩阵运算，O(n_msg × n_entity) |
| 存储量 | 0/1/2/极大单纯形 | 0/极大单纯形 + Laplacian缓存 |

***

## 九、论文创新点

### 创新点1：MSG提取范式（解耦式提取）

> 我们提出了极大语义组（MSG）的概念，将实体识别与拓扑构造解耦。传统方法要么要求LLM同时完成信息抽取和拓扑推理（导致提取衰减），要么完全忽略拓扑结构（导致关系松散）。MSG通过"语义完备性"替代"数学闭包性"，让LLM只需关注语义理解，而拓扑结构由后续算法自动构建。

### 创新点2：基于二部图关联的松弛单纯复形（简化Laplacian）

> 在MSG提取的基础上，我们定义了从极大单纯形直接到0-单纯形的边界映射 ∂: C_max → C₀，跳过所有中间维度的子复形。由此推导出简化Laplacian：L_entity = ∂ᵀ∂ 捕获实体间的共现拓扑，L_msg = ∂∂ᵀ 捕获MSG间的共享实体拓扑。这种"极简松弛"既保留了拓扑扩散能力，又避免了传统Hodge Laplacian需要显式构建1-单纯形和2-单纯形的组合爆炸问题。

### 创新点3：语义加权的拓扑检索（2×2结构）

> 我们提出了语义/拓扑 × 实体/MSG的2×2检索结构。纯拓扑扩散容易引入冗余信息，纯语义检索缺乏多跳推理能力。通过语义加权融合，拓扑扩散的结果必须经过语义相似度验证，确保只保留与查询相关的结果。三类别机制（collision/sem_only/topo_only）实现了语义与拓扑的互补而非叠加。

### 创新点4：基于拓扑隔离的注入攻击防御

> 松弛单纯复形的二部图结构天然具有拓扑隔离特性：恶意注入的实体因缺乏与合法实体的共现关系（不在同一MSG中），在L_entity中形成孤立子图，扩散信号无法到达。这种防御不依赖内容审查，而是依赖结构隔离，具有理论可证明性。

***

## 十、方法全景图

```
┌──────────────────────────────────────────────────────────────────┐
│                         原始文本                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  阶段1: LLM 提取（MSG Prompt）                                    │
│                                                                   │
│  输入: 文本块                                                     │
│  输出: MSG列表 + 实体列表                                         │
│  LLM只做: 识别语义事件 + 提取实体                                 │
│  LLM不做: 理解拓扑维度、保证闭包、枚举子关系                       │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  阶段2: 本地代码转换                                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  信息层: MSG → 极大单纯形                                  │    │
│  │  包含: description, source_id, embedding, importance      │    │
│  │  标记: is_maximal=True, layer="information"               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  二部图关联: 实体 ↔ MSG                                    │    │
│  │  entity.coboundary = 该实体所属的MSG列表                   │    │
│  │  msg.boundary = 该MSG包含的实体列表                        │    │
│  │  共享实体 = MSG之间的天然桥梁                               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  简化Laplacian:                                            │    │
│  │  L_entity = BᵀB（实体-实体共现拓扑）                       │    │
│  │  L_msg = BBᵀ（MSG-MSG共享实体拓扑）                       │    │
│  │  无需显式1-单纯形和2-单纯形                                 │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  阶段3: 存储                                                      │
│                                                                   │
│  entities_vdb ← 0-单纯形（实体）                                  │
│  relationships_vdb ← 信息层MSG（极大单纯形）                      │
│  SimplexStorage ← 0-单纯形 + MSG + Laplacian缓存                 │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  阶段4: 2×2 检索                                                  │
│                                                                   │
│  查询 → 提取实体                                                  │
│      ↓                                                            │
│  ┌────────────────────┬────────────────────┐                     │
│  │     实体节点        │     复形（MSG）     │                     │
│  ├────────────────────┼────────────────────┤                     │
│  │语义: entities_vdb  │语义: rel_vdb       │                     │
│  │     向量搜索        │     向量搜索        │                     │
│  ├────────────────────┼────────────────────┤                     │
│  │拓扑: L₀扩散        │拓扑: L_msg扩散     │                     │
│  │  实体→MSG→实体     │  MSG→实体→MSG     │                     │
│  └────────────────────┴────────────────────┘                     │
│      ↓                                                            │
│  语义加权融合（collision/sem_only/topo_only）                     │
│      ↓                                                            │
│  文本块收集（从MSG的source_id提取）                                │
│      ↓                                                            │
│  排序 + 截断                                                      │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  阶段5: 上下文组装 + LLM回答生成                                  │
│                                                                   │
│  Sources段: 原始文本块（最高优先级）                               │
│  Entities段: 与Sources相关的实体                                  │
│  Simplices段: 只包含MSG（信息层）                                 │
│                                                                   │
│  LLM生成回答 → 最终输出                                          │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  防御特性（贯穿构建和检索全过程）                                  │
│                                                                   │
│  构建时: MSG由LLM语义提取 → 恶意实体难以进入合法MSG               │
│  扩散时: L₀依赖共现关系 → 恶意实体与合法实体无共现               │
│  检索时: 检索目标只有MSG → 恶意实体不在合法MSG中                  │
│  理论保证: 二部图拓扑隔离定理 → 恶意实体形成孤立连通分量          │
└──────────────────────────────────────────────────────────────────┘
```
