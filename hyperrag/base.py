from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Any, Tuple, List, Set, Optional, Dict

from .utils import EmbeddingFunc

# 文本块Schema定义
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

# 0-单纯形（实体）Schema定义 - 对应2.md中的第一阶段
EntitySchema = TypedDict(
    "EntitySchema",
    {
        "entity_name": str,           # 实体名称
        "entity_type": str,           # 实体类型
        "description": str,           # 实体描述
        "source_id": str,             # 来源Chunk ID
        "additional_properties": str, # 附加属性
        "dimension": int,             # 维度（0-单纯形）
    },
)

# 1-单纯形（边）Schema定义 - 对应2.md中的第一阶段
EdgeSchema = TypedDict(
    "EdgeSchema",
    {
        "type": str,
        "dimension": int,
        "entities": List[str],
        "weight": float,
        "description": str,
        "source_id": str,
    },
)

# 高阶单纯形（面/体）Schema定义 - 对应2.md中的第一阶段和第三阶段
HighOrderSimplexSchema = TypedDict(
    "HighOrderSimplexSchema",
    {
        "type": str,
        "dimension": int,
        "entities": List[str],
        "weight": float,
        "description": str,
        "source_id": str,
        "chunk_ids": List[str],
        "verification_status": str,
    },
)

# 复形结构统一Schema - 对应2.md中的复形RAG架构
SimplexSchema = TypedDict(
    "SimplexSchema",
    {
        "simplex_id": str,
        "type": Literal["simplex"],
        "dimension": int,
        "entities": List[str],
        "weight": float,
        "description": str,
        "source_id": str,
        "chunk_ids": List[str],
        "verification_status": str,
        "created_at": str,
        "updated_at": str,
    },
)

T = TypeVar("T")


@dataclass
class QueryParam:
    """查询参数配置"""
    mode: Literal["naive", "llm", "topology"] = "topology"
    only_need_context: bool = False  # 是否仅返回上下文而不生成回答
    response_type: str = "Multiple Paragraphs"  # 回答格式类型
    top_k: int = 60  # 检索的top-k数量
    max_token_for_text_unit: int = 1600  # 原始文本块的最大token数
    max_token_for_entity_context: int = 300  # 实体描述的最大token数
    max_token_for_relation_context: int = 1600  # 关系描述的最大token数
    return_type: Literal["json", "text"] = "text"  # 返回类型：json或text
    return_retrieval_result: bool = False  # 是否同时返回检索结果（含原始chunks）


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError

"""
    The BaseHypergraphStorage based on hypergraph-DB
"""
@dataclass
class BaseSimplexStorage(StorageNameSpace):
    """复形存储基类 - 对应2.md中的复形RAG架构"""
    
    async def has_simplex(self, simplex_id: Any) -> bool:
        """检查单纯形是否存在"""
        raise NotImplementedError

    async def get_simplex(self, simplex_id: Any, default: Any = None):
        """获取指定单纯形"""
        raise NotImplementedError

    async def upsert_simplex(self, simplex_id: Any, simplex_data: Optional[Dict] = None):
        """插入或更新单纯形"""
        raise NotImplementedError

    async def remove_simplex(self, simplex_id: Any):
        """删除单纯形"""
        raise NotImplementedError

    async def get_all_simplices(self):
        """获取所有单纯形"""
        raise NotImplementedError

    async def get_simplices_by_dimension(self, dimension: int):
        """根据维度获取单纯形 - 对应2.md中的维度分类"""
        raise NotImplementedError

    async def get_simplices_by_entity(self, entity_id: str):
        """根据实体ID获取相关的单纯形"""
        raise NotImplementedError

    async def get_simplices_by_verification_status(self, status: str):
        """根据语义校验状态获取单纯形 - 对应2.md中的第三阶段"""
        raise NotImplementedError

    async def get_simplices_by_chunk_id(self, chunk_id: str):
        """根据Chunk ID获取相关的单纯形 - 对应2.md中的元数据绑定"""
        raise NotImplementedError

    async def get_chains(self, min_length: int = 2):
        """获取1-链（路径）- 对应2.md中的第二阶段"""
        raise NotImplementedError

    async def get_cliques(self, min_size: int = 3):
        """获取团结构（高维候选者）- 对应2.md中的第二阶段"""
        raise NotImplementedError


@dataclass
class BaseHypergraphStorage(StorageNameSpace):
    async def has_vertex(self, v_id: Any) -> bool:
        raise NotImplementedError

    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        raise NotImplementedError

    async def get_vertex(self, v_id: str, default: Any = None) :
        raise NotImplementedError

    async def get_hyperedge(self, e_tuple: Union[List, Set, Tuple], default: Any = None) :
        raise NotImplementedError

    async def get_all_vertices(self):
        raise NotImplementedError

    async def get_all_hyperedges(self):
        raise NotImplementedError

    async def get_num_of_vertices(self):
        raise NotImplementedError

    async def get_num_of_hyperedges(self):
        raise NotImplementedError

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None) :
        raise NotImplementedError

    async def upsert_hyperedge(self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None) :
        raise NotImplementedError

    async def remove_vertex(self, v_id: Any) :
        raise NotImplementedError

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]) :
        raise NotImplementedError

    async def vertex_degree(self, v_id: Any) -> int:
        raise NotImplementedError

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        raise NotImplementedError

    async def get_nbr_e_of_vertex(self, v_id: Any) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self=True) -> list:
        raise NotImplementedError
