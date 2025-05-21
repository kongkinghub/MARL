import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import multiprocessing
from datetime import datetime
from tqdm import tqdm
from masac import MASAC  # 导入原始的MASAC类

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 先尝试黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加环境路径
sys.path.append('C:/')  # 调整为您的learning模块路径
from learning.envs.make_env import make_env

# 设置结果保存路径
algorithm_name = "MASAC"
env_name = "simple_triangle_n6"

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

# 扩展MASAC类，添加并行训练功能
class ParallelMASAC(MASAC):
    """扩展MASAC类，添加并行训练功能"""
    
    def train_parallel(self, parallel_envs, num_episodes=1000, max_steps=25, 
                       batch_size=64, update_interval=1, eval_interval=100):
        """使用并行环境训练智能体"""
        print(f"开始并行训练，共{num_episodes}个回合...")
        start_time = time.time()
        
        # 初始化奖励记录
        episode_rewards = []
        eval_rewards = []
        
        # 创建进度条
        pbar = tqdm(total=num_episodes, desc="训练进度")
        
        episode_idx = 0
        
        while episode_idx < num_episodes:
            # 批量收集经验并训练
            rewards_batch, completed = self._collect_experience_batch(
                parallel_envs, 
                batch_size=batch_size,
                max_steps=max_steps,
                update_interval=update_interval
            )
            
            # 记录每个完成回合的奖励
            for reward in rewards_batch:
                episode_rewards.append(reward)
                episode_idx += 1
                pbar.update(1)
                
                # 评估
                if episode_idx % eval_interval == 0:
                    pbar.set_description(f"评估中...")
                    eval_reward = self.evaluate(eval_episodes=10)
                    eval_rewards.append(eval_reward)
                    
                    # 计算平均奖励
                    recent_rewards = episode_rewards[-min(10, len(episode_rewards)):]
                    avg_reward = np.mean(recent_rewards)
                    
                    pbar.set_description(f"训练进度")
                    print(f"\n评估回合 {episode_idx}/{num_episodes} | "
                         f"评估奖励: {eval_reward:.2f} | "
                         f"最近10回合平均奖励: {avg_reward:.2f}")
                
                # 达到所需回合数后退出
                if episode_idx >= num_episodes:
                    break
        
        # 关闭进度条
        pbar.close()
        
        # 计算训练时间
        total_time = time.time() - start_time
        print(f"训练完成! 总用时: {total_time:.2f}秒")
        
        return episode_rewards, eval_rewards
    
    def _collect_experience_batch(self, parallel_envs, batch_size=64, max_steps=25, update_interval=1):
        """从并行环境收集一批经验"""
        # 初始化状态和奖励
        states_list = parallel_envs.reset()
        active_rewards = [0.0] * parallel_envs.num_envs
        steps_count = [0] * parallel_envs.num_envs
        completed_rewards = []
        steps_done = 0
        
        # 使用所有环境同时收集经验
        while len(completed_rewards) < parallel_envs.num_envs:
            # 批量选择动作
            actions_list = []
            for states in states_list:
                actions = self.take_action(states)
                actions_list.append(actions)
            
            # 批量执行动作
            next_states_list, rewards_list, dones_list, _ = parallel_envs.step(actions_list)
            
            # 处理每个环境的结果
            for env_idx in range(parallel_envs.num_envs):
                if steps_count[env_idx] >= max_steps:  # 已经完成的环境跳过
                    continue
                    
                # 保存到经验缓冲区
                self.replay_buffer.push(
                    states_list[env_idx],
                    actions_list[env_idx],
                    rewards_list[env_idx],
                    next_states_list[env_idx],
                    dones_list[env_idx]
                )
                
                # 累计奖励
                active_rewards[env_idx] += np.sum(rewards_list[env_idx])
                steps_count[env_idx] += 1
                steps_done += 1
                
                # 检查是否应当更新策略
                if len(self.replay_buffer) >= batch_size and steps_done % update_interval == 0:
                    self.update(batch_size)
                
                # 检查是否回合结束
                if any(dones_list[env_idx]) or steps_count[env_idx] >= max_steps:
                    completed_rewards.append(active_rewards[env_idx])
                    states_list[env_idx] = parallel_envs.reset_one(env_idx)
                    steps_count[env_idx] = max_steps  # 标记为已完成
                else:
                    states_list[env_idx] = next_states_list[env_idx]
        
        return completed_rewards, parallel_envs.num_envs

def save_training_data(rewards, eval_rewards=None, eval_interval=1000, metadata=None):
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

def plot_results(rewards, eval_rewards=None, window=10, save=True, show=True):
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
    plt.figtext(0.01, 0.01, f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 并行训练", fontsize=8)
    
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
    # 打印当前时间和用户信息
    start_time = time.time()
    print(f"当前日期时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户: kongkinghub")
    print("=" * 50)
    
    # 设置并行环境数量
    cpu_count = multiprocessing.cpu_count()
    num_envs = max(4, cpu_count - 1)  # 保留一个核心给系统
    
    # 创建并行环境
    print(f"创建 {num_envs} 个并行环境实例...")
    parallel_envs = ParallelEnvs(env_name, num_envs=num_envs)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 256
    update_interval = 1
    eval_interval = 1000
    
    # 创建并行MASAC智能体 - 使用扩展的并行MASAC类
    print("初始化并行MASAC智能体...")
    masac = ParallelMASAC(
        env=parallel_envs,  # 传入并行环境
        device=device,
        actor_lr=0.005,
        critic_lr=0.01, 
        alpha_lr=0.005,
        hidden_dim=128,
        gamma=0.95, 
        tau=0.01,
        buffer_size=375000
    )
    
    # 开始并行训练 - 使用特定的并行训练方法
    print(f"开始并行训练，共{num_episodes}个回合...")
    rewards, eval_rewards = masac.train_parallel(
        parallel_envs=parallel_envs,
        num_episodes=num_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        update_interval=update_interval,
        eval_interval=eval_interval
    )
    
    # 计算训练时间
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 保存训练元数据
    metadata = {
        "算法": algorithm_name,
        "环境": env_name,
        "回合数": num_episodes,
        "最大步数": max_steps,
        "批次大小": batch_size,
        "更新间隔": update_interval,
        "评估间隔": eval_interval,
        "训练时间(秒)": training_time,
        "训练设备": device,
        "智能体数量": parallel_envs.n,
        "并行环境数": num_envs,
        "最终平均奖励(最后100回合)": np.mean(rewards[-100:]),
        "最终评估奖励": eval_rewards[-1] if eval_rewards and len(eval_rewards) > 0 else "N/A",
        "训练日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存模型
    final_model_path = f"{model_save_path}/{algorithm_name}_final_model.pt"
    masac.save(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    # 保存训练数据
    save_training_data(rewards, eval_rewards, eval_interval, metadata)
    
    # 绘制训练曲线
    plot_results(rewards, eval_rewards, window=100)
    
    # 评估最终模型
    print("正在进行最终评估...")
    final_eval_reward = masac.evaluate(eval_episodes=20)
    print(f"最终评估平均奖励: {final_eval_reward:.2f}")
    
    # 关闭环境
    parallel_envs.close()
    
    print("训练和评估完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()