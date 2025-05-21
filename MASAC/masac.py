import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

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

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNet, self).__init__()
        
        # Q1 network
        self.in_to_y1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        
        self.y1_to_y2 = nn.Linear(hidden_dim, hidden_dim)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.normal_(0, 0.1)
        
        # Q2 network
        self.q2_in_to_y1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        
        self.q2_y1_to_y2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.q2_out = nn.Linear(hidden_dim, 1)
        self.q2_out.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.in_to_y1(x))
        q1 = F.relu(self.y1_to_y2(q1))
        q1 = self.out(q1)
        
        # Q2
        q2 = F.relu(self.q2_in_to_y1(x))
        q2 = F.relu(self.q2_y1_to_y2(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=128, actor_lr=3e-4, critic_lr=3e-4, 
                 alpha_lr=3e-4, gamma=0.99, tau=0.005, device="cpu", max_action=1.0, min_action=-1.0):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.min_action = min_action
        
        # Actor network
        self.actor = ActorNet(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic network
        self.critic = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = CriticNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Temperature parameter
        self.target_entropy = -2.0  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
    
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
    
    def soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

class MASAC:
    def __init__(self, env, device="cpu", actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, 
                 hidden_dim=128, gamma=0.99, tau=0.005, buffer_size=1000000):
        """
        Multi-Agent Soft Actor-Critic
        
        Args:
            env: The environment
            device: Device to use for tensor operations
            actor_lr: Learning rate for actor networks
            critic_lr: Learning rate for critic networks
            alpha_lr: Learning rate for temperature parameter
            hidden_dim: Hidden dimension of networks
            gamma: Discount factor
            tau: Soft update coefficient
            buffer_size: Replay buffer size
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
        
        # Get dimensions from environment
        self.state_dim = env.observation_space[0].shape[0]
        self.action_dim = env.action_space[0].shape[0]
        self.max_action = env.action_space[0].high[0]
        self.min_action = env.action_space[0].low[0]
        
        # Joint dimensions for centralized critic
        self.joint_state_dim = self.state_dim * self.n_agents
        self.joint_action_dim = self.action_dim * self.n_agents
        
        # Initialize agents (decentralized actors)
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(
                SAC(self.state_dim, self.action_dim, hidden_dim, 
                    actor_lr, critic_lr, alpha_lr, gamma, tau, device,
                    self.max_action, self.min_action)
            )
        
        # Centralized critic
        self.critic = CriticNet(self.joint_state_dim, self.joint_action_dim, hidden_dim).to(device)
        self.target_critic = CriticNet(self.joint_state_dim, self.joint_action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration noise
        self.noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim))
    
    def take_action(self, states, explore=True):
        """Select actions for all agents"""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states[i], evaluate=not explore)
            if explore:
                action += self.noise() * 0.1  # Add exploration noise
                action = np.clip(action, self.min_action, self.max_action)
            actions.append(action)
        return actions
    
    def update(self, batch_size):
        """Update networks for all agents"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states_tensor = [torch.FloatTensor(np.vstack([states[j][i] for j in range(batch_size)])).to(self.device) 
                         for i in range(self.n_agents)]
        next_states_tensor = [torch.FloatTensor(np.vstack([next_states[j][i] for j in range(batch_size)])).to(self.device) 
                              for i in range(self.n_agents)]
        actions_tensor = [torch.FloatTensor(np.vstack([actions[j][i] for j in range(batch_size)])).to(self.device) 
                          for i in range(self.n_agents)]
        rewards_tensor = [torch.FloatTensor(np.vstack([rewards[j][i] for j in range(batch_size)])).to(self.device) 
                          for i in range(self.n_agents)]
        dones_tensor = [torch.FloatTensor(np.vstack([dones[j][i] for j in range(batch_size)])).to(self.device) 
                        for i in range(self.n_agents)]
        
        # Create joint states and actions
        joint_states = torch.cat(states_tensor, dim=1)
        joint_next_states = torch.cat(next_states_tensor, dim=1)
        joint_actions = torch.cat(actions_tensor, dim=1)
        
        # Update centralized critic
        with torch.no_grad():
            # Sample actions from all agents' policies
            next_actions = []
            next_log_probs = []
            
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
                next_log_probs.append(log_prob)
            
            # Create joint next actions
            joint_next_actions = torch.cat(next_actions, dim=1)
            joint_next_log_probs = torch.cat(next_log_probs, dim=1)
            
            # Get target Q values
            next_q1, next_q2 = self.target_critic(joint_next_states, joint_next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Average reward across agents (for cooperative tasks)
            target_reward = torch.mean(torch.stack(rewards_tensor), dim=0)
            # Average done signals across agents
            target_done = torch.mean(torch.stack(dones_tensor), dim=0)
            
            # Use mean alpha for entropy term
            mean_alpha = torch.mean(torch.stack([agent.alpha for agent in self.agents]))
            
            # Target Q value
            target_q = target_reward + (1 - target_done) * self.gamma * (
                next_q - mean_alpha * torch.mean(joint_next_log_probs))
        
        # Current Q values
        current_q1, current_q2 = self.critic(joint_states, joint_actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actors and temperature parameters
        for i, agent in enumerate(self.agents):
            # Actor update
            state = states_tensor[i]
            mean, std = agent.actor(state)
            normal = Normal(mean, std)
            z = normal.rsample()  # Use reparameterization trick
            action = torch.tanh(z)
            action = torch.clamp(action, self.min_action, self.max_action)
            
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            
            # Create joint actions with the current agent's new action
            new_joint_actions = joint_actions.clone()
            new_joint_actions[:, i*self.action_dim:(i+1)*self.action_dim] = action
            
            # Calculate Q value with new action
            q1, q2 = self.critic(joint_states, new_joint_actions)
            q = torch.min(q1, q2)
            
            # Actor loss
            actor_loss = (agent.alpha.detach() * log_prob - q).mean()
            
            # Optimize the actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # Temperature parameter update
            alpha_loss = -(agent.log_alpha * (log_prob.detach() + agent.target_entropy)).mean()
            
            agent.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            agent.alpha_optimizer.step()
            
            # Update alpha value
            agent.alpha = agent.log_alpha.exp()
            
            # Clamp log_alpha to prevent divergence
            with torch.no_grad():
                agent.log_alpha.clamp_(-5.0, 2.0)
        
        # Soft update of target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def train(self, num_episodes, max_steps, batch_size, update_interval=1, eval_interval=1000, render=False):
        """Train the MASAC agents"""
        episode_rewards = []
        eval_rewards = []  # 添加这行记录评估奖励
        
        for episode in range(num_episodes):
            states = self.env.reset()
            self.noise.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                if render:
                    self.env.render()
                
                # Select actions
                actions = self.take_action(states, explore=True)
                
                # Execute actions
                next_states, rewards, dones, _ = self.env.step(actions)
                
                # Store transition in replay buffer
                self.replay_buffer.push(states, actions, rewards, next_states, dones)
                
                # Update state and cumulative reward
                states = next_states
                # episode_reward += np.mean(rewards)  # For cooperative tasks
                episode_reward += np.sum(rewards)
                
                # Update networks
                if step % update_interval == 0:
                    self.update(batch_size)
                
                # Check if episode is done
                if any(dones):
                    break
            
            # Record episode reward
            episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Avg10: {avg_reward:.2f}")
            
            # 评估部分修改为:
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(10)
                eval_rewards.append(eval_reward)  # 添加这行记录评估奖励
                print(f"Evaluation at episode {episode+1}: Avg Reward = {eval_reward:.2f}")
        
        return episode_rewards, eval_rewards  # 修改返回值
    
    def evaluate(self, eval_episodes=10, render=False):
        """Evaluate the trained agents"""
        total_reward = 0
        
        for episode in range(eval_episodes):
            states = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            max_steps = 25  # Same as training
            
            while not done and step < max_steps:
                if render:
                    self.env.render()
                
                # Select actions without exploration
                actions = self.take_action(states, explore=False)
                
                # Execute actions
                next_states, rewards, dones, _ = self.env.step(actions)
                
                states = next_states
                # episode_reward += np.mean(rewards)  # For cooperative tasks
                episode_reward += np.sum(rewards)

                done = any(dones)
                step += 1
            
            total_reward += episode_reward
        
        return total_reward / eval_episodes
    
    def save(self, filepath):
        """Save model parameters"""
        save_dict = {
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict()
        }
        
        for i, agent in enumerate(self.agents):
            save_dict[f'actor_{i}'] = agent.actor.state_dict()
            save_dict[f'log_alpha_{i}'] = agent.log_alpha
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint[f'actor_{i}'])
            agent.log_alpha = checkpoint[f'log_alpha_{i}']
            agent.alpha = agent.log_alpha.exp()
        
        print(f"Model loaded from {filepath}")