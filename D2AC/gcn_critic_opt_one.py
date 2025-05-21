import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """优化后的图卷积层，使用矩阵乘法代替循环"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.self_weight = nn.Linear(in_features, out_features)
        self.neigh_weight = nn.Linear(in_features, out_features)
        
        # 初始化权重
        self.self_weight.weight.data.normal_(0, 0.1)
        self.neigh_weight.weight.data.normal_(0, 0.1)
        
    def forward(self, x, adj):
        """
        x: 节点特征 [batch_size, n_agents, feature_dim]
        adj: 邻接矩阵 [n_agents, n_agents]
        """
        batch_size, n_agents, feature_dim = x.size()
        
        # 自连接项
        self_loop = self.self_weight(x)
        
        # 邻居聚合项 - 使用矩阵乘法而不是循环（核心优化）
        # 归一化邻接矩阵
        degree = adj.sum(dim=1).clamp(min=1)  # 防止出现零度节点
        norm_adj = adj / degree.unsqueeze(1)  # 行归一化
        
        # 一次性完成所有邻居聚合
        # [batch_size, n_agents, n_agents] × [batch_size, n_agents, feature_dim]
        aggregate = torch.bmm(
            norm_adj.unsqueeze(0).expand(batch_size, -1, -1), 
            x
        )
        
        # 应用邻居权重
        neigh_out = self.neigh_weight(aggregate)
        
        # 合并自连接和邻居项
        return F.relu(self_loop + neigh_out)


class GCNCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=128):
        super(GCNCritic, self).__init__()
        
        # 每个智能体的特征维度（状态+动作）
        self.feature_dim = state_dim + action_dim
        self.n_agents = n_agents
        
        # 创建全连接邻接矩阵（对角线为0）
        self.register_buffer('adj', self._create_adjacency_matrix(n_agents))
        
        # GCN层
        self.gcn1 = GCNLayer(self.feature_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # Q1和Q2输出层（双Q网络）
        self.q1_head = nn.Linear(hidden_dim, 1)
        self.q1_head.weight.data.normal_(0, 0.1)
        
        self.q2_head = nn.Linear(hidden_dim, 1)
        self.q2_head.weight.data.normal_(0, 0.1)
    
    def _create_adjacency_matrix(self, n_agents):
        """创建全连接邻接矩阵，对角线为0"""
        adj = torch.ones(n_agents, n_agents)
        adj.fill_diagonal_(0)
        return adj
    
    def forward(self, states, actions):
        """
        states: 联合状态 [batch_size, n_agents*state_dim]
        actions: 联合动作 [batch_size, n_agents*action_dim]
        """
        batch_size = states.size(0)
        
        # 重塑输入为[batch_size, n_agents, feature_dim]形式
        states = states.view(batch_size, self.n_agents, -1)
        actions = actions.view(batch_size, self.n_agents, -1)
        
        # 特征拼接：[batch_size, n_agents, state_dim+action_dim]
        x = torch.cat([states, actions], dim=2)
        
        # 应用GCN层
        x = self.gcn1(x, self.adj)
        x = self.gcn2(x, self.adj)
        
        # 全局平均池化，得到每个图的表示
        # [batch_size, hidden_dim]
        x = x.mean(dim=1)
        
        # 输出Q值
        q1 = self.q1_head(x)
        q2 = self.q2_head(x)
        
        return q1, q2