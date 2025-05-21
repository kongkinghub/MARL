import numpy as np
import torch
import random

class ConsensusNetwork:
    def __init__(self, n_agents, rho=0.5, device="cpu"):
        """
        初始化共识网络
        
        Args:
            n_agents: 智能体数量N
            rho: 连通率参数ρ，控制通信网络的稠密程度，取值范围[0, 1]
            device: 计算设备
        """
        self.n_agents = n_agents
        self.rho = rho
        self.device = device
        self.total_possible_edges = 0.5 * n_agents * (n_agents - 1)
        self.expected_edges = int(self.rho * self.total_possible_edges)
        
    def generate_communication_graph(self):
        """生成时变无向图G_t = (N, E_t)"""
        # 创建邻接矩阵 (N x N)，初始为全零
        adj_matrix = np.zeros((self.n_agents, self.n_agents))
        
        # 获取所有可能的边（无向图）
        possible_edges = []
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):  # 只考虑上三角矩阵，因为是无向图
                possible_edges.append((i, j))
        
        # 根据ρ参数随机选择边
        num_edges = self.expected_edges
        selected_edges = random.sample(possible_edges, min(num_edges, len(possible_edges)))
        
        # 填充邻接矩阵（无向图，所以是对称的）
        for i, j in selected_edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
            
        return torch.tensor(adj_matrix, dtype=torch.float32, device=self.device)
    
    def calculate_weight_matrix(self, adj_matrix):
        """
        计算权重矩阵C_t
        
        按照公式:
        c_t(i,j) = 1/(1 + max[d_t(i), d_t(j)]) 如果 (i,j) ∈ E_t
        c_t(i,i) = 1 - sum_{j∈N_i} c_t(i,j) 如果 i ∈ N
        """
        n = adj_matrix.shape[0]
        weight_matrix = torch.zeros_like(adj_matrix)
        
        # 计算每个节点的度
        degrees = adj_matrix.sum(dim=1)
        
        # 计算非对角元素 c_t(i,j)
        for i in range(n):
            for j in range(n):
                if i != j and adj_matrix[i, j] > 0:
                    max_degree = torch.max(degrees[i], degrees[j])
                    weight_matrix[i, j] = 1.0 / (1.0 + max_degree)
        
        # 计算对角元素 c_t(i,i)
        for i in range(n):
            weight_matrix[i, i] = 1.0 - weight_matrix[i, :].sum()
            
        return weight_matrix