import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
import random


class ActorNetwork(nn.Module):
    """
    PIC中的Actor网络 - 每个智能体拥有单独的actor
    遵循CTDE范式的分散执行部分
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_action=1.0):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        # 网络结构（两层隐藏层，与论文一致）
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.max_action * torch.tanh(self.mean(x))
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std
    
    def sample_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            return mean
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob


class GCNLayer(nn.Module):
    """
    图卷积层 - 实现置换不变性
    遵循论文中的Equation 8
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear_self = nn.Linear(in_features, out_features)  # W_self
        self.linear_other = nn.Linear(in_features, out_features)  # W_other
        
    def forward(self, x, adj):
        """
        x: [batch_size, n_agents, in_features]
        adj: [n_agents, n_agents] - 邻接矩阵
        """
        batch_size, n_agents, _ = x.size()
        
        # 自变换
        h_self = self.linear_self(x)
        
        # 其他智能体的贡献（Equation 8: 1/N * A_adj * h * W_other）
        h_other = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x) / n_agents
        h_other = self.linear_other(h_other)
        
        # 组合并激活
        h = h_self + h_other
        return F.relu(h)


class PermutationInvariantCritic(nn.Module):
    """
    置换不变评论家网络 - 使用GCN实现
    两层GCN后接平均池化，与论文一致
    """
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=64):
        super(PermutationInvariantCritic, self).__init__()
        
        self.agent_feat_dim = state_dim + action_dim
        self.n_agents = n_agents
        
        # 完全连接的邻接矩阵（非对角线为1，对角线为0）
        self.register_buffer('adj', torch.ones(n_agents, n_agents) - torch.eye(n_agents))
        
        # 两层GCN
        self.gcn1 = GCNLayer(self.agent_feat_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # 双Q网络
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, states, actions):
        batch_size = states.size(0)
        state_dim = states.size(1) // self.n_agents
        action_dim = actions.size(1) // self.n_agents
        
        # 重塑输入
        states = states.view(batch_size, self.n_agents, -1)
        actions = actions.view(batch_size, self.n_agents, -1)
        
        # 拼接状态和动作
        inputs = torch.cat([states, actions], dim=2)
        
        # 应用GCN层
        h = self.gcn1(inputs, self.adj)
        h = self.gcn2(h, self.adj)
        
        # 平均池化（置换不变）
        h = torch.mean(h, dim=1)
        
        # 计算Q值
        q1 = self.q1(h)
        q2 = self.q2(h)
        
        return q1, q2


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity, n_agents, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def push(self, states, actions, rewards, next_states, dones):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (states, actions, rewards, next_states, dones)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class PIC:
    """
    PIC算法 - 使用SAC适配的多智能体强化学习
    """
    def __init__(self, env, device="cpu", actor_lr=1e-3, critic_lr=1e-3, 
                 alpha=0.2, gamma=0.99, tau=0.005, buffer_size=1000000,
                 hidden_dim=64):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.n_agents = env.n
        
        self.state_dim = env.observation_space[0].shape[0]
        self.action_dim = env.action_space[0].shape[0]
        self.max_action = env.action_space[0].high[0]
        
        print(f"环境信息: {self.n_agents}个智能体, 状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        
        # 初始化actor网络
        self.actors = []
        self.actors_target = []
        self.actor_optimizers = []
        for _ in range(self.n_agents):
            actor = ActorNetwork(self.state_dim, self.action_dim, hidden_dim, self.max_action).to(device)
            actor_target = ActorNetwork(self.state_dim, self.action_dim, hidden_dim, self.max_action).to(device)
            actor_target.load_state_dict(actor.state_dict())
            actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
            
            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.actor_optimizers.append(actor_optimizer)
        
        # 共享的PIC评论家
        self.critic = PermutationInvariantCritic(
            self.state_dim, self.action_dim, self.n_agents, hidden_dim
        ).to(device)
        self.critic_target = PermutationInvariantCritic(
            self.state_dim, self.action_dim, self.n_agents, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size, self.n_agents, self.state_dim, self.action_dim)
        
        self.actor_losses = []
        self.critic_losses = []
        
        print("PIC算法初始化完成")
    
    def select_action(self, states, evaluate=False):
        actions = []
        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i]).to(self.device).unsqueeze(0)
            if evaluate:
                mean, _ = actor(state)
                action = mean
            else:
                action, _ = actor.sample_action(state)
            actions.append(action.detach().cpu().numpy()[0])
        return actions
    
    def update_parameters(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        # 假设全局奖励，论文MPE任务如此
        rewards_tensor = torch.FloatTensor(rewards).to(self.device).mean(dim=1, keepdim=True)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device).any(dim=1, keepdim=True)
        
        batch_size = states_tensor.size(0)
        states_flat = states_tensor.reshape(batch_size, -1)
        actions_flat = actions_tensor.reshape(batch_size, -1)
        next_states_flat = next_states_tensor.reshape(batch_size, -1)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            for i, actor_target in enumerate(self.actors_target):
                next_state_i = next_states_tensor[:, i]
                next_action_i, log_prob_i = actor_target.sample_action(next_state_i)
                next_actions.append(next_action_i)
                next_log_probs.append(log_prob_i)
            
            next_actions_flat = torch.cat(next_actions, dim=1)
            next_log_probs_mean = torch.cat(next_log_probs, dim=1).mean(dim=1, keepdim=True)
            
            target_q1, target_q2 = self.critic_target(next_states_flat, next_actions_flat)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_tensor + (1 - dones_tensor.float()) * self.gamma * \
                       (target_q - self.alpha * next_log_probs_mean)
        
        current_q1, current_q2 = self.critic(states_flat, actions_flat)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_losses.append(critic_loss.item())
        
        # 更新Actor
        total_actor_loss = 0
        for i in range(self.n_agents):
            state_i = states_tensor[:, i]
            action_i, log_prob_i = self.actors[i].sample_action(state_i)
            
            new_actions = actions_tensor.clone().reshape(batch_size, self.n_agents, -1)
            new_actions[:, i] = action_i
            new_actions_flat = new_actions.reshape(batch_size, -1)
            
            q1, q2 = self.critic(states_flat, new_actions_flat)
            q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_prob_i - q).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            total_actor_loss += actor_loss.item()
        
        self.actor_losses.append(total_actor_loss / self.n_agents)
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(), self.actors_target[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)