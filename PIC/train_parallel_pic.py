import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import multiprocessing
from datetime import datetime
from tqdm import tqdm
from pic import PIC  # 导入修改后的PIC算法

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 先尝试黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 算法和环境名称
algorithm_name = 'PIC'
env_name = 'simple_formation_n6'

model_save_path = f'C:/learning/models/{env_name}/{algorithm_name}'
results_save_path = f'C:/learning/results/{env_name}/{algorithm_name}'

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

# 并行环境封装器
class ParallelEnvs:
    def __init__(self, env_name, num_envs=4):
        """创建多个并行环境实例"""
        self.envs = [make_env(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs
        # 获取第一个环境的属性
        self.n = self.envs[0].n
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def reset(self):
        """重置所有环境"""
        return [env.reset() for env in self.envs]
    
    def reset_one(self, env_idx):
        """重置单个环境"""
        return self.envs[env_idx].reset()
        
    def step(self, actions_list):
        """在所有环境中执行动作"""
        results = []
        for env, actions in zip(self.envs, actions_list):
            next_state, reward, done, info = env.step(actions)
            results.append((next_state, reward, done, info))
        
        next_states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        
        return next_states, rewards, dones, infos

    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

# 进度条函数
def progress_bar(current, total, bar_length=50, info=""):
    fraction = current / total
    arrow = '█' * int(fraction * bar_length)
    padding = ' ' * (bar_length - len(arrow))
    return f"\r[{arrow}{padding}] {current}/{total} {info}"

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

def evaluate_policy(agent, env, eval_episodes=10, max_steps=25):
    """评估策略性能"""
    total_reward = 0
    
    for _ in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += np.sum(reward)  # 对合作任务，取总和奖励
            state = next_state
            done = any(done)
            step += 1
        
        total_reward += episode_reward
    
    return total_reward / eval_episodes

def save_training_data(rewards, eval_rewards=None, eval_interval=200, metadata=None):
    """保存训练数据到CSV和NPY文件"""
    # 创建训练奖励的DataFrame
    train_data = pd.DataFrame({
        'episode': range(1, len(rewards) + 1),
        'average_return': rewards
    })
    
    # 保存到CSV
    train_data.to_csv(f"{results_save_path}/{algorithm_name}_train.csv", index=False)
    
    # 保存到NumPy数组
    np.save(f"{results_save_path}/{algorithm_name}_train.npy", np.array(rewards))
    
    # 如果有评估数据，也保存它
    if eval_rewards and len(eval_rewards) > 0:
        eval_episodes = np.linspace(1, len(rewards), len(eval_rewards), endpoint=False, dtype=int)
        eval_data = pd.DataFrame({
            'episode': eval_episodes,
            'average_return': eval_rewards
        })
        eval_data.to_csv(f"{results_save_path}/{algorithm_name}_eval.csv", index=False)
        np.save(f"{results_save_path}/{algorithm_name}_eval.npy", np.array(eval_rewards))
    
    # 保存训练元数据
    if metadata:
        with open(f"{results_save_path}/{algorithm_name}_metadata.txt", 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    print(f"训练数据已保存到 {results_save_path}")

def plot_results(rewards, eval_rewards=None, window=10, save=True, show=False):
    """绘制训练曲线，包括训练和评估奖励"""
    plt.figure(figsize=(12, 6))
    
    # 绘制每个episode的奖励
    plt.plot(rewards, 'b-', alpha=0.3, label='每回合奖励')
    
    # 绘制平滑后的奖励曲线
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed_rewards, 'b-', 
                label=f'平滑奖励 (窗口={window})')
    
    # 绘制评估奖励
    if eval_rewards and len(eval_rewards) > 0:
        eval_episodes = np.linspace(0, len(rewards)-1, len(eval_rewards), endpoint=True).astype(int)
        plt.plot(eval_episodes, eval_rewards, 'ro-', label='评估奖励')
    
    plt.xlabel('回合 (Episode)')
    plt.ylabel('平均回报 (Average Return)')
    plt.title(f'{algorithm_name} 算法在 {env_name} 环境中的训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加额外信息
    plt.figtext(0.01, 0.01, f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                           f"环境: {env_name} | 确定性策略 | 并行训练", fontsize=8)
    
    # 保存图像
    if save:
        plt.savefig(f"{results_save_path}/{algorithm_name}_training_curve.png", dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到 {results_save_path}/{algorithm_name}_training_curve.png")
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()

def main():
    # 初始化
    start_time = time.time()
    print(f"当前日期和时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户登录名: kongkinghub")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置并行环境数量
    if torch.cuda.is_available():
        # GPU训练：根据显存大小调整
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        num_envs = min(32, max(8, int(gpu_memory_gb * 2)))  # 每GB显存分配2个环境
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
    else:
        # CPU训练：根据核心数调整
        cpu_count = multiprocessing.cpu_count()
        num_envs = min(16, max(4, cpu_count - 1))  # 保留1个核心给系统
    
    # 创建并行环境
    print(f"创建 {num_envs} 个并行环境实例...")
    parallel_envs = ParallelEnvs(env_name, num_envs=num_envs)
    
    # 创建评估环境
    eval_env = make_env(env_name)
    
    # 初始化PIC智能体
    print("初始化PIC智能体...")
    agent = PIC(
        env=parallel_envs,  # 使用并行环境
        device=device,
        actor_lr=0.005,
        critic_lr=0.01, 
        alpha=0.005,
        gamma=0.95, 
        tau=0.01,
        buffer_size=375000,
        hidden_dim=128
    )
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 256
    update_interval = 1
    eval_interval = 200
    save_interval = 1000
    warmup_steps = 1000
    
    # 元数据
    metadata = {
        'algorithm': algorithm_name,
        'environment': env_name,
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'batch_size': batch_size,
        'parallel_envs': num_envs,
        'actor_lr': agent.actor_optimizers[0].param_groups[0]['lr'],
        'critic_lr': agent.critic_optimizer.param_groups[0]['lr'],
        'alpha': agent.alpha,
        'gamma': agent.gamma,
        'tau': agent.tau,
        'buffer_size': agent.replay_buffer.capacity,
        'hidden_dim': 128,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 经验收集 - 现在使用并行环境进行预填充
    print("开始并行初始经验收集...")
    states_list = parallel_envs.reset()
    collected_steps = 0
    
    with tqdm(total=warmup_steps, desc="预填充经验缓冲区") as pbar:
        while collected_steps < warmup_steps:
            # 为所有环境生成随机动作
            actions_list = []
            for _ in range(num_envs):
                actions = [
                    np.random.uniform(
                        low=parallel_envs.action_space[i].low, 
                        high=parallel_envs.action_space[i].high
                    ) for i in range(agent.n_agents)
                ]
                actions_list.append(actions)
            
            # 执行动作并获取结果
            next_states_list, rewards_list, dones_list, _ = parallel_envs.step(actions_list)
            
            # 处理每个环境的结果
            for env_idx in range(num_envs):
                # 保存经验到缓冲区
                agent.replay_buffer.push(
                    states_list[env_idx], 
                    actions_list[env_idx], 
                    rewards_list[env_idx], 
                    next_states_list[env_idx], 
                    dones_list[env_idx]
                )
                collected_steps += 1
                pbar.update(1)
                
                # 如果环境结束则重置
                if any(dones_list[env_idx]):
                    states_list[env_idx] = parallel_envs.reset_one(env_idx)
                else:
                    states_list[env_idx] = next_states_list[env_idx]
                
                if collected_steps >= warmup_steps:
                    break
    
    print(f"初始经验收集完成，缓冲区大小: {len(agent.replay_buffer)}")
    
    # 开始训练 - 并行方式
    print(f"开始并行训练，共{num_episodes}个回合...")
    episode_rewards = []
    eval_rewards = []
    
    # 初始化状态和计数器
    states_list = parallel_envs.reset()
    active_rewards = [0.0] * num_envs
    steps_in_episode = [0] * num_envs
    episode_idx = 0
    
    with tqdm(total=num_episodes, desc="训练进度") as pbar:
        while episode_idx < num_episodes:
            # 批量获取动作
            actions_list = []
            for state in states_list:
                action = agent.select_action(state, evaluate=False)
                actions_list.append(action)
            
            # 批量执行动作
            next_states_list, rewards_list, dones_list, _ = parallel_envs.step(actions_list)
            
            # 处理每个环境
            for env_idx in range(num_envs):
                # 保存经验
                agent.replay_buffer.push(
                    states_list[env_idx],
                    actions_list[env_idx],
                    rewards_list[env_idx],
                    next_states_list[env_idx],
                    dones_list[env_idx]
                )
                
                # 累计奖励
                active_rewards[env_idx] += np.sum(rewards_list[env_idx])
                steps_in_episode[env_idx] += 1
                
                # 更新网络
                if len(agent.replay_buffer) >= batch_size:
                    agent.update_parameters(batch_size)
                
                # 检查回合是否结束
                if any(dones_list[env_idx]) or steps_in_episode[env_idx] >= max_steps:
                    # 记录完成的回合
                    episode_rewards.append(active_rewards[env_idx])
                    episode_idx += 1
                    pbar.update(1)
                    
                    # 重置此环境
                    active_rewards[env_idx] = 0.0
                    steps_in_episode[env_idx] = 0
                    states_list[env_idx] = parallel_envs.reset_one(env_idx)
                    
                    # 显示进度
                    if episode_idx % 10 == 0:
                        recent_rewards = episode_rewards[-min(10, len(episode_rewards)):]
                        avg_reward = np.mean(recent_rewards)
                        avg_actor_loss = np.mean(agent.actor_losses[-100:]) if agent.actor_losses else 0
                        avg_critic_loss = np.mean(agent.critic_losses[-100:]) if agent.critic_losses else 0
                        
                        # 计算剩余时间
                        elapsed = time.time() - start_time
                        progress = episode_idx / num_episodes
                        estimated_total = elapsed / progress if progress > 0 else 0
                        remaining = estimated_total - elapsed
                        
                        pbar.set_postfix({
                            "奖励": f"{active_rewards[env_idx]:.1f}", 
                            "平均10ep": f"{avg_reward:.1f}",
                            "A损失": f"{avg_actor_loss:.4f}",
                            "C损失": f"{avg_critic_loss:.4f}",
                            "剩余": format_time(remaining)
                        })
                    
                    # 评估并保存数据
                    if episode_idx % eval_interval == 0:
                        # 暂停进度条显示
                        tqdm.write("\n执行评估...")
                        eval_reward = evaluate_policy(agent, eval_env)
                        eval_rewards.append(eval_reward)
                        tqdm.write(f"评估结果 (回合 {episode_idx}): 平均奖励 = {eval_reward:.2f}")
                        
                        # 保存训练数据
                        save_training_data(episode_rewards, eval_rewards, eval_interval=eval_interval, metadata=metadata)
                        
                        # 绘制并保存训练曲线
                        plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)
                    
                    # 保存模型
                    if episode_idx % save_interval == 0:
                        model_dict = {
                            'actors': [actor.state_dict() for actor in agent.actors],
                            'actors_target': [actor_target.state_dict() for actor_target in agent.actors_target],
                            'critic': agent.critic.state_dict(),
                            'critic_target': agent.critic_target.state_dict()
                        }
                        torch.save(model_dict, f"{model_save_path}/model_ep{episode_idx}.pt")
                        tqdm.write(f"\n模型已保存: {model_save_path}/model_ep{episode_idx}.pt")
                    
                    # 如果已达到目标回合数，退出
                    if episode_idx >= num_episodes:
                        break
                else:
                    # 继续此环境的回合
                    states_list[env_idx] = next_states_list[env_idx]
    
    # 训练结束
    print("\n训练完成!")
    total_time = time.time() - start_time
    print(f"总训练时间: {format_time(total_time)}")
    
    # 保存最终模型和数据
    model_dict = {
        'actors': [actor.state_dict() for actor in agent.actors],
        'actors_target': [actor_target.state_dict() for actor_target in agent.actors_target],
        'critic': agent.critic.state_dict(),
        'critic_target': agent.critic_target.state_dict()
    }
    torch.save(model_dict, f"{model_save_path}/final_model.pt")
    
    metadata['total_time'] = format_time(total_time)
    metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_training_data(episode_rewards, eval_rewards, eval_interval=eval_interval, metadata=metadata)
    plot_results(episode_rewards, eval_rewards, window=10, save=True, show=True)
    
    # 关闭环境
    parallel_envs.close()
    eval_env.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()