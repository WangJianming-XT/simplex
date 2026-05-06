"""
单纯形树(Simplex Tree)数据结构实现

这是一种基于前缀树的高效存储结构，用于管理单纯形及其拓扑关系。
参考Gudhi库的实现思路，提供高效的单纯形插入、查询和边界计算功能。

核心功能：
- 高效的单纯形存储和检索
- 快速的边界和上边界计算
- 支持任意维度的单纯形
- 内存使用优化
- 支持序列化和反序列化

使用场景：
- 拓扑数据分析(TDA)
- 复杂网络分析
- 知识图谱构建
- 高维数据索引
"""


from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import json


class SimplexTreeNode:
    """
    单纯形树的节点类
    
    优化：
    - 减少节点存储的冗余信息
    - 仅在必要时存储单纯形数据
    - 使用更紧凑的数据结构
    """
    __slots__ = ['vertex', 'parent', 'children', 'simplex_data']
    
    def __init__(self, vertex: Any, parent=None):
        """
        初始化节点
        
        Args:
            vertex: 节点对应的顶点值
            parent: 父节点引用
        """
        self.vertex = vertex  # 节点对应的顶点
        self.parent = parent  # 父节点
        self.children = {}  # 子节点映射，键为顶点值，值为子节点
        self.simplex_data = None  # 存储单纯形数据，如果该节点是单纯形的终点


class SimplexTree:
    """
    单纯形树数据结构
    """
    def __init__(self):
        """
        初始化单纯形树
        """
        self.root = SimplexTreeNode(None)  # 根节点
        self.simplex_count = 0  # 单纯形数量
        self.id_to_node = {}  # ID到节点的映射
        self.vertex_to_nodes = defaultdict(list)  # 顶点到包含该顶点的节点的映射

    def insert(self, vertices: List[Any], simplex_data: Dict = None, simplex_id: str = None) -> bool:
        """
        插入单纯形
        
        Args:
            vertices: 单纯形的顶点列表，需要按顺序排列
            simplex_data: 单纯形数据
            simplex_id: 单纯形ID
            
        Returns:
            是否插入成功
        """
        if not vertices:
            return False
        
        # 确保顶点列表是排序的
        sorted_vertices = sorted(vertices)
        
        # 从根节点开始插入
        current = self.root
        for vertex in sorted_vertices:
            if vertex not in current.children:
                current.children[vertex] = SimplexTreeNode(vertex, current)
            current = current.children[vertex]
        
        # 如果该单纯形已经存在，更新数据
        if current.simplex_data is not None:
            old_id = current.simplex_data.get('id')
            current.simplex_data.update(simplex_data or {})
            if simplex_id:
                current.simplex_data['id'] = simplex_id
                self.id_to_node[simplex_id] = current
                if old_id and old_id != simplex_id:
                    self.id_to_node.pop(old_id, None)
            return True
        
        # 插入新单纯形
        current.simplex_data = simplex_data or {}
        # 将ID和维度存储在simplex_data中
        if simplex_id:
            current.simplex_data['id'] = simplex_id
            self.id_to_node[simplex_id] = current
        # 存储维度信息
        current.simplex_data['dimension'] = len(sorted_vertices) - 1
        
        # 更新顶点到节点的映射
        for vertex in sorted_vertices:
            self.vertex_to_nodes[vertex].append(current)
        
        self.simplex_count += 1
        return True

    def find(self, vertices: List[Any]) -> Optional[SimplexTreeNode]:
        """
        查找单纯形
        
        Args:
            vertices: 单纯形的顶点列表
            
        Returns:
            找到的节点，否则返回None
        """
        if not vertices:
            return None
        
        sorted_vertices = sorted(vertices)
        current = self.root
        
        for vertex in sorted_vertices:
            if vertex not in current.children:
                return None
            current = current.children[vertex]
        
        return current if current.simplex_data is not None else None

    def find_by_id(self, simplex_id: str) -> Optional[SimplexTreeNode]:
        """
        根据ID查找单纯形
        
        Args:
            simplex_id: 单纯形ID
            
        Returns:
            找到的节点，否则返回None
        """
        return self.id_to_node.get(simplex_id)

    def remove(self, vertices: List[Any]) -> bool:
        """
        删除单纯形
        
        Args:
            vertices: 单纯形的顶点列表
            
        Returns:
            是否删除成功
        """
        node = self.find(vertices)
        if not node or not node.simplex_data:
            return False
        
        # 移除ID映射
        simplex_id = node.simplex_data.get('id')
        if simplex_id:
            self.id_to_node.pop(simplex_id, None)
        
        # 移除顶点到节点的映射
        vertices = self._get_vertices(node)
        for vertex in vertices:
            if node in self.vertex_to_nodes[vertex]:
                self.vertex_to_nodes[vertex].remove(node)
        
        # 清除单纯形数据
        node.simplex_data = None
        
        # 清理空节点
        self._cleanup_empty_nodes(node)
        
        self.simplex_count -= 1
        return True

    def _cleanup_empty_nodes(self, node: SimplexTreeNode):
        """
        清理空节点
        
        Args:
            node: 要清理的节点
        """
        current = node
        while current != self.root and not current.children and current.simplex_data is None:
            parent = current.parent
            del parent.children[current.vertex]
            current = parent

    def _get_vertices(self, node: SimplexTreeNode) -> List[Any]:
        """
        获取节点对应的单纯形的顶点列表
        
        Args:
            node: 节点
            
        Returns:
            顶点列表
        """
        vertices = []
        current = node
        while current != self.root:
            vertices.append(current.vertex)
            current = current.parent
        return list(reversed(vertices))

    def get_boundary(self, vertices: List[Any]) -> List[List[Any]]:
        """
        获取单纯形的边界（所有低维子单纯形）
        
        超边复形的边界定义：
        - n-单纯形的边界包含所有低维子单纯形（从0维到n-1维）
        - 1-simplex的边界：2个0-simplex（顶点）
        - 2-simplex的边界：3个1-simplex（边）和3个0-simplex（顶点）
        - n-simplex的边界：所有k维子单纯形，其中0 ≤ k ≤ n-1
        
        Args:
            vertices: 单纯形的顶点列表
            
        Returns:
            边界单纯形的顶点列表
        """
        node = self.find(vertices)
        if not node or not node.simplex_data:
            return []
        
        # 获取单纯形维度
        dimension = node.simplex_data.get('dimension', -1)
        if dimension < 0:
            return []
        
        # 生成所有低维子单纯形（从1维到n-1维）
        from itertools import combinations
        sorted_vertices = sorted(vertices)
        boundary = []
        
        # 生成所有维度的子复形，包括1维及以上
        for k in range(1, len(sorted_vertices)):
            for sub_vertices in combinations(sorted_vertices, k):
                sub_node = self.find(list(sub_vertices))
                if sub_node and sub_node.simplex_data is not None:
                    boundary.append(list(sub_vertices))
        
        # 添加0维子复形（顶点）
        for vertex in sorted_vertices:
            vertex_node = self.find([vertex])
            if vertex_node and vertex_node.simplex_data is not None:
                boundary.append([vertex])
        
        return boundary

    def get_coboundary(self, vertices: List[Any]) -> List[List[Any]]:
        """
        获取单纯形的上边界（所有包含该单纯形的高维单纯形）
        
        超边复形的上边界定义：
        - 包含该单纯形的所有高维单纯形
        - 1-simplex的上边界：包含该边的所有2-simplex、3-simplex等
        - 0-simplex的上边界：包含该顶点的所有1-simplex、2-simplex等
        - k-simplex的上边界：所有维度大于k的单纯形，且包含该k-simplex
        
        Args:
            vertices: 单纯形的顶点列表
            
        Returns:
            上边界单纯形的顶点列表
        """
        node = self.find(vertices)
        if not node:
            return []
        
        sorted_vertices = sorted(vertices)
        vertex_set = set(sorted_vertices)
        coboundary = []
        
        # 利用 vertex_to_nodes 找到包含该单纯形所有顶点的高维单纯形
        candidate_nodes = None
        for vertex in sorted_vertices:
            nodes_set = set(self.vertex_to_nodes.get(vertex, []))
            if candidate_nodes is None:
                candidate_nodes = nodes_set
            else:
                candidate_nodes &= nodes_set
        
        if candidate_nodes is None:
            return []
        
        for candidate in candidate_nodes:
            if candidate.simplex_data is not None:
                cand_vertices = self._get_vertices(candidate)
                if len(cand_vertices) > len(sorted_vertices) and vertex_set.issubset(set(cand_vertices)):
                    coboundary.append(sorted(cand_vertices))
        
        return coboundary

    def get_all_simplices(self) -> List[Tuple[List[Any], Dict]]:
        """
        获取所有单纯形
        
        Returns:
            单纯形列表，每个元素为(顶点列表, 单纯形数据)
        """
        simplices = []
        
        def traverse(current, path):
            if current.simplex_data is not None:
                simplices.append((path.copy(), current.simplex_data))
            
            for vertex, child in current.children.items():
                path.append(vertex)
                traverse(child, path)
                path.pop()
        
        traverse(self.root, [])
        return simplices

    def get_simplices_by_dimension(self, dimension: int) -> List[Tuple[List[Any], Dict]]:
        """
        获取指定维度的所有单纯形
        
        Args:
            dimension: 维度
            
        Returns:
            单纯形列表
        """
        simplices = []
        
        def traverse(current, path):
            if current.simplex_data is not None and current.simplex_data.get('dimension', -1) == dimension:
                simplices.append((path.copy(), current.simplex_data))
            
            for vertex, child in current.children.items():
                path.append(vertex)
                traverse(child, path)
                path.pop()
        
        traverse(self.root, [])
        return simplices

    def get_simplices_by_vertex(self, vertex: Any) -> List[Tuple[List[Any], Dict]]:
        """
        获取包含指定顶点的所有单纯形
        
        Args:
            vertex: 顶点
            
        Returns:
            单纯形列表
        """
        simplices = []
        nodes = self.vertex_to_nodes.get(vertex, [])
        
        for node in nodes:
            if node.simplex_data is not None:
                vertices = self._get_vertices(node)
                simplices.append((vertices, node.simplex_data))
        
        return simplices

    def save(self, file_path: str):
        """
        保存单纯形树到文件
        
        Args:
            file_path: 文件路径
        """
        simplices = self.get_all_simplices()
        data = []
        
        for vertices, simplex_data in simplices:
            simplex_info = {
                'vertices': vertices,
                'data': simplex_data
            }
            data.append(simplex_info)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path: str):
        """
        从文件加载单纯形树
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 清空现有数据
            self.__init__()
            
            # 加载单纯形
            for simplex_info in data:
                vertices = simplex_info.get('vertices', [])
                simplex_data = simplex_info.get('data', {})
                simplex_id = simplex_data.get('id')
                self.insert(vertices, simplex_data, simplex_id)
        except Exception as e:
            print(f"Error loading simplex tree: {e}")

    def size(self) -> int:
        """
        获取单纯形数量
        
        Returns:
            单纯形数量
        """
        return self.simplex_count

    def __str__(self):
        """
        字符串表示
        """
        return f"SimplexTree with {self.simplex_count} simplices"
