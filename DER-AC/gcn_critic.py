import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """严格按照文章公式 (27) 实现的图卷积层"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 自连接权重 W_s,l
        self.self_weight = nn.Linear(in_features, out_features)
        # 邻居权重 W_o,l
        self.neigh_weight = nn.Linear(in_features, out_features)
        
        # 初始化权重
        self.self_weight.weight.data.normal_(0, 0.1)
        self.neigh_weight.weight.data.normal_(0, 0.1)
        
    def forward(self, x, adj, n_agents):
        """
        x: 节点特征 [batch_size, n_agents, feature_dim]
        adj: 邻接矩阵 [n_agents, n_agents]
        n_agents: 智能体数量 N
        """
        batch_size = x.size(0)
        
        # 自连接项 h_{l-1} W_{s,l}
        self_loop = self.self_weight(x)
        
        # 邻居聚合项 (1/N) A_{adj} h_{l-1} W_{o,l}
        # 首先计算 h_{l-1} W_{o,l}
        neigh_out = self.neigh_weight(x)  # [batch_size, n_agents, out_features]
        
        # 计算 (1/N) A_{adj} h_{l-1} W_{o,l}
        norm_adj = adj / n_agents  # 直接用 1/N 归一化
        aggregate = torch.bmm(
            norm_adj.unsqueeze(0).expand(batch_size, -1, -1), 
            neigh_out
        )
        
        # 合并自连接和邻居项，并应用激活函数
        return F.relu(self_loop + aggregate)


class GCNCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=128):
        super(GCNCritic, self).__init__()
        
        # 每个智能体的特征维度（状态+动作）
        self.feature_dim = state_dim + action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        
        # 创建全连接邻接矩阵（对角线为0）
        self.register_buffer('adj', self._create_adjacency_matrix(n_agents))
        
        # GCN 层
        self.gcn1 = GCNLayer(self.feature_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # f_{v_i} 线性层，将 h^i_L 映射到标量 Q^k
        self.f_vi = nn.Linear(hidden_dim, 1)
        self.f_vi.weight.data.normal_(0, 0.1)
        
        # 双 Q 网络的第二个 f_{v_i}
        self.f_vi_2 = nn.Linear(hidden_dim, 1)
        self.f_vi_2.weight.data.normal_(0, 0.1)
    
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
        
        # 重塑输入为 [batch_size, n_agents, feature_dim] 形式
        states = states.view(batch_size, self.n_agents, -1)
        actions = actions.view(batch_size, self.n_agents, -1)
        
        # 特征拼接：[batch_size, n_agents, state_dim+action_dim]
        x = torch.cat([states, actions], dim=2)
        
        # 应用 GCN 层
        x = self.gcn1(x, self.adj, self.n_agents)
        x = self.gcn2(x, self.adj, self.n_agents)
        
        # 此时 x 的形状为 [batch_size, n_agents, hidden_dim]
        # 每个 h^i_L 是 [batch_size, hidden_dim]
        
        # 应用 f_{v_i} 线性层，映射到标量 Q^k
        # [batch_size, n_agents, hidden_dim] -> [batch_size, n_agents, 1]
        qk_1 = self.f_vi(x)
        qk_2 = self.f_vi_2(x)
        
        # 对所有 agent 的 Q^k 求和，得到 Q_{\omega^i}
        # [batch_size, n_agents, 1] -> [batch_size, 1]
        q1 = qk_1.sum(dim=1)
        q2 = qk_2.sum(dim=1)
        
        return q1, q2