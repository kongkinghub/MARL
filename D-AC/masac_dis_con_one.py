import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from gcn_critic import GCNCritic
from consensus_net_one import ConsensusNetwork  # 导入共识网络类

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        
        self.in_to_y1 = nn.Linear(state_dim, hidden_dim)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        
        self.y1_to_y2 = nn.Linear(hidden_dim, hidden_dim)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.out = nn.Linear(hidden_dim, action_dim)
        self.out.weight.data.normal_(0, 0.1)
        
        self.std_out = nn.Linear(hidden_dim, action_dim)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.in_to_y1(state))
        x = F.relu(self.y1_to_y2(x))
        
        mean = self.max_action * torch.tanh(self.out(x))
        log_std = self.std_out(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # 验证数据格式
        try:
            # 确保所有数据都是正确的格式和维度
            assert isinstance(state, list), f"状态应为列表，得到{type(state)}"
            assert isinstance(action, list), f"动作应为列表，得到{type(action)}"
            assert isinstance(reward, list), f"奖励应为列表，得到{type(reward)}"
            assert isinstance(next_state, list), f"下一状态应为列表，得到{type(next_state)}"
            assert isinstance(done, list), f"完成状态应为列表，得到{type(done)}"
            
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
        
        except Exception as e:
            print(f"添加到缓冲区时出错: {e}")
            print(f"状态类型: {type(state)}, 动作类型: {type(action)}")
            print(f"奖励类型: {type(reward)}, 下一状态类型: {type(next_state)}")
            print(f"完成状态类型: {type(done)}")
            raise
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小({len(self.buffer)})小于批处理大小({batch_size})")
            
        try:
            batch_indices = np.random.choice(len(self.buffer), batch_size)
            batch = [self.buffer[i] for i in batch_indices]
            state, action, reward, next_state, done = zip(*batch)
            return state, action, reward, next_state, done
        
        except Exception as e:
            print(f"采样缓冲区时出错: {e}")
            print(f"缓冲区大小: {len(self.buffer)}, 批处理大小: {batch_size}")
            print(f"采样索引: {batch_indices if 'batch_indices' in locals() else '未定义'}")
            raise
    
    def __len__(self):
        return len(self.buffer)
    
    def size(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """检查缓冲区是否准备好进行采样"""
        return len(self.buffer) >= batch_size

class DistributedAgent:
    def __init__(self, agent_id, state_dim, action_dim, n_agents, hidden_dim, 
                 actor_lr, critic_lr, alpha_lr, gamma, tau, device, max_action, min_action):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.min_action = min_action
        
        # 策略网络 (Actor)
        self.actor = ActorNet(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # GCN价值网络 (Critic) - 注意：输入仍然是全局状态和动作
        self.critic = GCNCritic(state_dim, action_dim, n_agents, hidden_dim).to(device)
        self.target_critic = GCNCritic(state_dim, action_dim, n_agents, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 温度参数
        # self.target_entropy = -2.0  # -dim(A)
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # self.alpha = self.log_alpha.exp()
        # self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        # 温度参数 - 修改为始终为0
        self.alpha = torch.zeros(1, device=device)  # 固定为0，并且不需要requires_grad
        
        # 记录损失
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.actor(state)
            
            if evaluate:
                action = mean
            else:
                normal = Normal(mean, std)
                action = normal.sample()
                action = torch.clamp(action, self.min_action, self.max_action)
                
        return action.cpu().detach().numpy().flatten()
    
    def update_critic(self, joint_states, joint_actions, joint_next_states, joint_next_actions, 
                    rewards, dones, next_log_probs):
        """更新该智能体的价值网络"""
        try:
            with torch.no_grad():
                # 获取目标Q值
                next_q1, next_q2 = self.target_critic(joint_next_states, joint_next_actions)
                next_q = torch.min(next_q1, next_q2)
                
                # # 使用该智能体自己的奖励
                # target_q = rewards[:, self.agent_id].unsqueeze(-1) + \
                #         (1 - dones[:, self.agent_id].unsqueeze(-1)) * self.gamma * \
                #         (next_q - self.alpha.detach() * next_log_probs[:, self.agent_id].unsqueeze(-1))
                # 使用该智能体自己的奖励 - 移除熵正则化项
                target_q = rewards[:, self.agent_id].unsqueeze(-1) + \
                        (1 - dones[:, self.agent_id].unsqueeze(-1)) * self.gamma * next_q
                # 原代码: next_q - self.alpha.detach() * next_log_probs[:, self.agent_id].unsqueeze(-1)
                # 由于alpha=0，直接使用next_q
            
            # 当前Q值
            current_q1, current_q2 = self.critic(joint_states, joint_actions)
            
            # 计算critic损失
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # 优化critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 记录损失
            self.critic_losses.append(critic_loss.item())
            
            return critic_loss.item()
        
        except Exception as e:
            print(f"Agent {self.agent_id} 更新Critic时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # 返回默认值以避免中断训练
    
    def update_actor(self, joint_states, joint_actions, state):
        """更新该智能体的策略网络"""
        mean, std = self.actor(state)
        normal = Normal(mean, std)
        z = normal.rsample()  # 使用重参数化采样
        action = torch.tanh(z)
        action = torch.clamp(action, self.min_action, self.max_action)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # 创建新的联合动作，替换当前智能体的动作
        new_joint_actions = joint_actions.clone()
        new_joint_actions[:, self.agent_id*self.action_dim:(self.agent_id+1)*self.action_dim] = action
        
        # 使用自己的critic网络评估动作价值
        q1, q2 = self.critic(joint_states, new_joint_actions)
        q = torch.min(q1, q2)
        
        # 计算actor损失
        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        
        # 优化actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 移除alpha更新
        # 温度参数更新
        # alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        
        # self.alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()
        
        # # 更新alpha值
        # self.alpha = self.log_alpha.exp()
        
        # # 限制log_alpha防止发散
        # with torch.no_grad():
        #     self.log_alpha.clamp_(-5.0, 2.0)
        
        # 记录损失
        self.actor_losses.append(actor_loss.item())
        
        return action, log_prob, actor_loss.item()
    
    def soft_update_critic(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

class MASAC:
    def __init__(self, env, device="cpu", actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, 
                 hidden_dim=128, gamma=0.99, tau=0.005, buffer_size=1000000,
                 consensus_rho=0.5, use_consensus=True):
        """
        分布式多智能体SAC (Distributed MASAC)
        
        Args:
            env: 环境
            device: 用于张量操作的设备
            actor_lr: actor网络学习率
            critic_lr: critic网络学习率
            alpha_lr: 温度参数学习率
            hidden_dim: 网络隐藏层维度
            gamma: 折扣因子
            tau: 软更新系数
            buffer_size: 经验回放缓冲区大小
            consensus_rho: 通信网络连通率参数ρ
            use_consensus: 是否使用参数共识更新
        """
        self.env = env
        self.n_agents = env.n
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr  # 存储学习率
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.hidden_dim = hidden_dim  # 存储隐藏层维度
        self.buffer_size = buffer_size  # 存储缓冲区大小
        self.consensus_rho = consensus_rho  # 存储共识参数
        
        # 获取环境维度信息
        self.state_dim = env.observation_space[0].shape[0]
        self.action_dim = env.action_space[0].shape[0]
        self.max_action = env.action_space[0].high[0]
        self.min_action = env.action_space[0].low[0]
        
        print(f"环境信息: {self.n_agents}个智能体, 状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        
        # 初始化分布式智能体，每个智能体有自己的actor和critic
        self.agents = []
        for i in range(self.n_agents):
            agent = DistributedAgent(
                agent_id=i,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                n_agents=self.n_agents,
                hidden_dim=hidden_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                alpha_lr=alpha_lr,
                gamma=gamma,
                tau=tau,
                device=device,
                max_action=self.max_action,
                min_action=self.min_action
            )
            self.agents.append(agent)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 共识网络参数
        self.use_consensus = use_consensus
        self.consensus_network = ConsensusNetwork(self.n_agents, rho=consensus_rho, device=device)
        
        print(f"分布式MASAC初始化完成" + 
              (f"，参数共识已启用 (ρ={consensus_rho})" if use_consensus else ""))
    
    def take_action(self, states, explore=True):
        """选择所有智能体的动作"""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states[i], evaluate=not explore)
            actions.append(action)
        return actions
    
    def update(self, batch_size):
        """更新所有智能体的网络，然后执行参数共识更新"""
        buffer_size = len(self.replay_buffer)
        if buffer_size < batch_size:
            return None
        
        try:
            # 从回放缓冲区采样
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 转换为张量
            states_tensor = [torch.FloatTensor(np.vstack([states[j][i] for j in range(batch_size)])).to(self.device) 
                            for i in range(self.n_agents)]
            next_states_tensor = [torch.FloatTensor(np.vstack([next_states[j][i] for j in range(batch_size)])).to(self.device) 
                                for i in range(self.n_agents)]
            actions_tensor = [torch.FloatTensor(np.vstack([actions[j][i] for j in range(batch_size)])).to(self.device) 
                            for i in range(self.n_agents)]
            rewards_tensor = torch.FloatTensor(np.array([rewards[j] for j in range(batch_size)])).to(self.device)
            dones_tensor = torch.FloatTensor(np.array([dones[j] for j in range(batch_size)])).to(self.device)
            
            # 创建联合状态和动作
            joint_states = torch.cat(states_tensor, dim=1)
            joint_next_states = torch.cat(next_states_tensor, dim=1)
            joint_actions = torch.cat(actions_tensor, dim=1)
        
            # 采样所有智能体的下一个动作和对数概率
            next_actions = []
            next_log_probs_list = []
            
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    next_state = next_states_tensor[i]
                    mean, std = agent.actor(next_state)
                    normal = Normal(mean, std)
                    z = normal.sample()
                    action = torch.tanh(z)
                    action = torch.clamp(action, self.min_action, self.max_action)
                    
                    log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
                    log_prob = log_prob.sum(dim=1, keepdim=True)
                    
                    next_actions.append(action)
                    next_log_probs_list.append(log_prob)
            
            # 创建联合下一动作
            joint_next_actions = torch.cat(next_actions, dim=1)
            next_log_probs = torch.cat(next_log_probs_list, dim=1)
            
            # 更新每个智能体的critic
            for i, agent in enumerate(self.agents):
                agent.update_critic(
                    joint_states, 
                    joint_actions, 
                    joint_next_states, 
                    joint_next_actions, 
                    rewards_tensor, 
                    dones_tensor, 
                    next_log_probs
                )
            
            # 更新每个智能体的actor
            actor_losses = []
            for i, agent in enumerate(self.agents):
                # 只使用自身观察更新策略
                _, _, actor_loss = agent.update_actor(
                    joint_states,
                    joint_actions,
                    states_tensor[i]
                )
                actor_losses.append(actor_loss)
            
            # 软更新所有智能体的目标网络
            for agent in self.agents:
                agent.soft_update_critic()
            
            # 执行参数共识更新
            if self.use_consensus:
                self.perform_consensus_update()
            
            # 返回平均actor损失
            return np.mean(actor_losses)
            
        except Exception as e:
            print(f"更新出错: {e}")
            return None
    
    def perform_consensus_update(self):
        """执行参数共识更新"""
        # 为Actor和Critic分别生成通信图和权重矩阵
        actor_adj = self.consensus_network.generate_communication_graph()
        critic_adj = self.consensus_network.generate_communication_graph()
        
        actor_weights = self.consensus_network.calculate_weight_matrix(actor_adj)
        critic_weights = self.consensus_network.calculate_weight_matrix(critic_adj)
        
        # 收集所有智能体的参数
        actor_state_dicts = [agent.actor.state_dict() for agent in self.agents]
        critic_state_dicts = [agent.critic.state_dict() for agent in self.agents]
        
        # 执行参数共识更新
        self._consensus_update_networks(actor_state_dicts, actor_weights, "actor")
        self._consensus_update_networks(critic_state_dicts, critic_weights, "critic")
    
    def _consensus_update_networks(self, state_dicts, weight_matrix, network_type):
        """
        对指定类型的网络执行参数共识更新
        
        Args:
            state_dicts: 所有智能体的网络状态字典列表
            weight_matrix: 权重矩阵C_t
            network_type: 网络类型 ("actor" 或 "critic")
        """
        # 为每个智能体创建新的参数字典
        new_state_dicts = [{} for _ in range(self.n_agents)]
        
        # 对每个参数层执行共识更新
        for key in state_dicts[0].keys():
            # 跳过非参数项，如buffers
            if not isinstance(state_dicts[0][key], torch.Tensor) or not state_dicts[0][key].requires_grad:
                for i in range(self.n_agents):
                    new_state_dicts[i][key] = state_dicts[i][key].clone()
                continue
                
            # 对参数执行加权平均
            for i in range(self.n_agents):
                # 初始化为该智能体自己权重的贡献
                param_i = weight_matrix[i, i] * state_dicts[i][key].clone()
                
                # 加上来自其他智能体的贡献
                for j in range(self.n_agents):
                    if i != j and weight_matrix[i, j] > 0:
                        # 确保张量形状相同
                        if state_dicts[i][key].shape == state_dicts[j][key].shape:
                            param_i += weight_matrix[i, j] * state_dicts[j][key].clone()
                
                new_state_dicts[i][key] = param_i
        
        # 更新每个智能体的网络参数
        for i, agent in enumerate(self.agents):
            if network_type == "actor":
                agent.actor.load_state_dict(new_state_dicts[i])
            elif network_type == "critic":
                agent.critic.load_state_dict(new_state_dicts[i])
    
    def train(self, num_episodes, max_steps, batch_size, update_interval=1, eval_interval=1000, render=False):
        """训练分布式MASAC智能体"""
        episode_rewards = []
        eval_rewards = []
        
        for episode in range(num_episodes):
            states = self.env.reset()
            episode_reward = 0
            total_actor_loss = 0
            update_count = 0
            
            for step in range(max_steps):
                if render:
                    self.env.render()
                
                # 选择动作
                actions = self.take_action(states, explore=True)
                
                # 执行动作
                next_states, rewards, dones, _ = self.env.step(actions)
                
                # 存入经验回放缓冲区
                self.replay_buffer.push(states, actions, rewards, next_states, dones)
                
                # 更新状态和累积奖励
                states = next_states
                # episode_reward += np.mean(rewards)
                episode_reward += np.sum(rewards) # 对合作任务，取总和奖励
                
                # 更新网络 - 仅在缓冲区足够大时尝试更新
                if step % update_interval == 0 and len(self.replay_buffer) >= batch_size:
                    loss = self.update(batch_size)
                    if loss is not None:
                        total_actor_loss += loss
                        update_count += 1
                
                # 检查是否结束
                if any(dones):
                    break
            
            # 记录本回合奖励
            episode_rewards.append(episode_reward)
            
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_actor_loss = total_actor_loss / max(update_count, 1)
                avg_critic_losses = [np.mean(agent.critic_losses[-100:]) if agent.critic_losses else 0 
                                for agent in self.agents]
                avg_critic_loss = np.mean(avg_critic_losses)
                
                print(f"回合 {episode+1}/{num_episodes} | "
                    f"奖励: {episode_reward:.2f} | "
                    f"平均10回合: {avg_reward:.2f} | "
                    f"Actor损失: {avg_actor_loss:.4f} | "
                    f"Critic损失: {avg_critic_loss:.4f}")
            
            # 评估
            if (episode + 1) % eval_interval == 0 and len(self.replay_buffer) >= batch_size:
                eval_reward = self.evaluate(10)
                eval_rewards.append(eval_reward)
                print(f"评估 (回合 {episode+1}): 平均奖励 = {eval_reward:.2f}")
        
        return episode_rewards, eval_rewards
    
    def evaluate(self, eval_episodes=10, render=False):
        """评估训练好的智能体"""
        total_reward = 0
        
        for episode in range(eval_episodes):
            states = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            max_steps = 25  # 与训练相同
            
            while not done and step < max_steps:
                if render:
                    self.env.render()
                
                # 选择动作，不使用探索
                actions = self.take_action(states, explore=False)
                
                # 执行动作
                next_states, rewards, dones, _ = self.env.step(actions)
                
                states = next_states
                # episode_reward += np.mean(rewards)  # 对合作任务，取平均奖励
                episode_reward += np.sum(rewards) # 对合作任务，取总和奖励
                done = any(dones)
                step += 1
            
            total_reward += episode_reward
        
        return total_reward / eval_episodes
    
    def save(self, filepath):
        """保存模型参数（包含共识网络参数）"""
        save_dict = {
            'consensus_rho': self.consensus_network.rho,
            'use_consensus': self.use_consensus
        }
        
        for i, agent in enumerate(self.agents):
            save_dict[f'agent_{i}_actor'] = agent.actor.state_dict()
            save_dict[f'agent_{i}_critic'] = agent.critic.state_dict()
            save_dict[f'agent_{i}_target_critic'] = agent.target_critic.state_dict()
            # save_dict[f'agent_{i}_log_alpha'] = agent.log_alpha
            # 移除log_alpha保存
            # save_dict[f'agent_{i}_log_alpha'] = agent.log_alpha
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """加载模型参数（包含共识网络参数）"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载共识网络参数
        if 'consensus_rho' in checkpoint:
            self.consensus_network.rho = checkpoint['consensus_rho']
        if 'use_consensus' in checkpoint:
            self.use_consensus = checkpoint['use_consensus']
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint[f'agent_{i}_actor'])
            agent.critic.load_state_dict(checkpoint[f'agent_{i}_critic'])
            agent.target_critic.load_state_dict(checkpoint[f'agent_{i}_target_critic'])
            # 移除log_alpha加载
            # agent.log_alpha = checkpoint[f'agent_{i}_log_alpha']
            # agent.alpha = agent.log_alpha.exp()
        
        print(f"模型已从 {filepath} 加载")