import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
from collections import deque
from tqdm import tqdm

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env
from masac_dis_con_one import MASAC  # 导入你的MASAC类

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 算法和环境名称
algorithm_name = 'DER-AC'
env_name = 'simple_coop_push_n6'

# 设置结果保存路径
model_save_path = f'C:/learning/models/{algorithm_name}_{env_name}'
results_save_path = f'C:/learning/results/{algorithm_name}_{env_name}'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

# 启用CUDA基准模式以优化性能（如果可用）
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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

    def render(self, env_idx=0):
        """渲染指定的环境实例"""
        return self.envs[env_idx].render()
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

# 多批次经验收集函数
def collect_experience_parallel(masac, parallel_envs, steps_per_collection=256, explore=True):
    """
    并行收集多环境经验
    
    Args:
        masac: MASAC智能体
        parallel_envs: 并行环境对象
        steps_per_collection: 每次收集的总步数
        explore: 是否使用探索
    
    Returns:
        收集的经验和平均奖励
    """
    # 初始化状态和计数器
    states_list = parallel_envs.reset()
    total_rewards = 0
    total_steps = 0
    
    while total_steps < steps_per_collection:
        # 批量生成动作
        actions_list = []
        for states in states_list:
            actions = masac.take_action(states, explore=explore)
            actions_list.append(actions)
        
        # 执行动作并获取结果
        next_states_list, rewards_list, dones_list, _ = parallel_envs.step(actions_list)
        
        # 保存经验到缓冲区
        for states, actions, rewards, next_states, dones in zip(states_list, actions_list, rewards_list, next_states_list, dones_list):
            masac.replay_buffer.push(states, actions, rewards, next_states, dones)
            total_rewards += sum(rewards)
            total_steps += 1
            
            # 重置已完成的环境
            if any(dones):
                states_list = parallel_envs.reset()
                break
        
        # 更新状态
        if not any([any(d) for d in dones_list]):
            states_list = next_states_list
    
    # 返回平均奖励
    return total_rewards / total_steps

# 修改后的批量更新函数 - 简化版本，移除了混合精度训练
def update_batch(masac, batch_size, num_updates=10):
    """
    执行多次批量更新
    
    Args:
        masac: MASAC智能体
        batch_size: 每批次的样本数
        num_updates: 更新次数
        
    Returns:
        平均损失
    """
    if len(masac.replay_buffer) < batch_size:
        return None
        
    total_loss = 0
    valid_updates = 0
    
    for _ in range(num_updates):
        loss = masac.update(batch_size)
        if loss is not None:
            total_loss += loss
            valid_updates += 1
            
    return total_loss / valid_updates if valid_updates > 0 else 0

# 事件记录和保存函数
def save_training_data(rewards, eval_rewards=None, eval_interval=1000, metadata=None):
    """保存训练数据到CSV和NPY文件"""
    train_data = pd.DataFrame({
        'episode': range(1, len(rewards) + 1),
        'average_return': rewards
    })
    
    train_data.to_csv(f"{results_save_path}/{algorithm_name}_train.csv", index=False)
    np.save(f"{results_save_path}/{algorithm_name}_train.npy", np.array(rewards))
    
    if eval_rewards and len(eval_rewards) > 0:
        eval_episodes = np.linspace(1, len(rewards), len(eval_rewards), endpoint=False, dtype=int)
        eval_data = pd.DataFrame({
            'episode': eval_episodes,
            'average_return': eval_rewards
        })
        eval_data.to_csv(f"{results_save_path}/{algorithm_name}_eval.csv", index=False)
        np.save(f"{results_save_path}/{algorithm_name}_eval.npy", np.array(eval_rewards))
    
    if metadata:
        with open(f"{results_save_path}/{algorithm_name}_metadata.txt", 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    print(f"训练数据已保存到 {results_save_path}")

def plot_results(rewards, eval_rewards=None, window=10, save=True, show=False):
    """绘制训练曲线，包括训练和评估奖励"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(rewards, 'b-', alpha=0.3, label='每回合奖励')
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed_rewards, 'b-', 
                label=f'平滑奖励 (窗口={window})')
    
    if eval_rewards and len(eval_rewards) > 0:
        eval_episodes = np.linspace(0, len(rewards)-1, len(eval_rewards), endpoint=True).astype(int)
        plt.plot(eval_episodes, eval_rewards, 'ro-', label='评估奖励')
    
    plt.xlabel('回合 (Episode)')
    plt.ylabel('平均回报 (Average Return)')
    plt.title(f'{algorithm_name} 算法在 {env_name} 环境中的训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.figtext(0.01, 0.01, f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                           f"环境: {env_name} | ρ=0.4", fontsize=8)
    
    if save:
        plt.savefig(f"{results_save_path}/{algorithm_name}_training_curve.png", dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到 {results_save_path}/{algorithm_name}_training_curve.png")
    
    if show:
        plt.show()
    else:
        plt.close()

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

def test_agent(masac, env_name, render_delay=0.2):
    """测试训练好的智能体"""
    env = make_env(env_name)
    num_test_episodes = 10
    max_steps = 25
    test_rewards = []
    
    print("\n===== 开始测试 =====")
    
    for episode in range(num_test_episodes):
        print(f"\n回合 {episode + 1}/{num_test_episodes}")
        states = env.reset()
        episode_reward = 0
        
        # 渲染初始状态
        env.render()
        time.sleep(render_delay)
        
        for step in range(max_steps):
            # 获取动作
            actions = masac.take_action(states, explore=False)
            
            # 执行动作
            next_states, rewards, dones, _ = env.step(actions)
            
            # 计算奖励
            step_reward = np.sum(rewards)
            episode_reward += step_reward
            
            # 更新状态
            states = next_states
            
            # 输出信息
            print(f"步骤 {step+1}/{max_steps} | 步骤奖励: {step_reward:.2f} | 累计奖励: {episode_reward:.2f}")
            
            # 渲染环境
            env.render()
            time.sleep(render_delay)
            
            # 如果环境结束，提前终止
            if any(dones):
                print(f"环境提前结束，步数: {step+1}")
                break
        
        test_rewards.append(episode_reward)
        print(f"回合 {episode + 1} 总奖励: {episode_reward:.2f}")
        
        # 回合之间暂停，除非是最后一个回合
        if episode < num_test_episodes - 1:
            input("按Enter键继续下一回合...")
    
    # 打印测试结果统计
    print("\n测试完成!")
    print(f"平均每回合奖励: {np.mean(test_rewards):.2f}")
    print(f"最高奖励: {np.max(test_rewards):.2f}")
    print(f"最低奖励: {np.min(test_rewards):.2f}")
    
    env.close()

def main():
    """主训练函数"""
    start_time = time.time()
    print(f"当前日期和时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户登录名: kongkinghub")
    print("=" * 50)
    
    # 检测GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"使用设备: {device} (检测到 {num_gpus} 个GPU)")
    
    # 根据CPU核心数确定并行环境数量
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    num_envs = min(16, max(4, cpu_count - 1))  # 至少4个，最多16个，保留1个核心给系统
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 512  # 使用适中的批处理大小
    eval_interval = 1000
    
    # 优化参数
    updates_per_collection = 4  # 每次收集经验后执行的更新次数
    
    print(f"创建 {num_envs} 个并行环境实例...")
    parallel_envs = ParallelEnvs(env_name, num_envs=num_envs)
    
    print("初始化分布式MASAC智能体（带参数共识）...")
    masac = MASAC(
        env=parallel_envs, 
        device=device,
        actor_lr=0.005,
        critic_lr=0.01, 
        alpha_lr=0.005,
        hidden_dim=128,
        gamma=0.95, 
        tau=0.01,
        buffer_size=375000,
        consensus_rho=0.4,
        use_consensus=True
    )
    
    # 计算每个episode需要收集的经验数量
    steps_per_collection = max_steps * num_envs
    
    # 元数据
    metadata = {
        'algorithm': algorithm_name,
        'environment': env_name,
        'num_episodes': num_episodes,
        'batch_size': batch_size,
        'parallel_envs': num_envs,
        'updates_per_collection': updates_per_collection,
        'actor_lr': masac.actor_lr,
        'critic_lr': masac.critic_lr,
        'alpha_lr': masac.alpha_lr,
        'gamma': masac.gamma,
        'tau': masac.tau,
        'buffer_size': masac.replay_buffer.capacity,
        'hidden_dim': 128,
        'consensus_rho': masac.consensus_rho,
        'use_consensus': masac.use_consensus,
        'device': str(device),
        'num_gpus': num_gpus,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("预填充经验缓冲区...")
    prefill_steps = batch_size
    avg_reward = collect_experience_parallel(masac, parallel_envs, prefill_steps, explore=True)
    print(f"预填充完成! 收集了 {len(masac.replay_buffer)} 个样本，平均奖励: {avg_reward:.2f}")
    
    print(f"开始训练，共 {num_episodes} 个回合...")
    episode_rewards = []
    eval_rewards = []
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        # 收集经验
        avg_reward = collect_experience_parallel(
            masac, 
            parallel_envs, 
            steps_per_collection, 
            explore=True
        )
        
        episode_rewards.append(avg_reward)
        
        # 进行多次批量更新
        losses = []
        for _ in range(updates_per_collection):
            loss = update_batch(masac, batch_size, num_updates=1)
            if loss is not None:
                losses.append(loss)
        
        avg_loss = np.mean(losses) if losses else 0
        
        # 计算运行状态
        elapsed = time.time() - start_time
        progress = (episode + 1) / num_episodes
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        episode_time = time.time() - episode_start
        
        # 输出进度
        if (episode + 1) % 10 == 0:
            avg_reward_10ep = np.mean(episode_rewards[-10:])
            memory_info = ""
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                memory_info = f"GPU内存: {memory_allocated:.0f}MB | "
            
            # 计算训练速度（样本/秒）
            samples_per_second = steps_per_collection / episode_time
            
            info = (f"奖励: {avg_reward:.1f} | "
                   f"10ep平均: {avg_reward_10ep:.1f} | "
                   f"损失: {avg_loss:.4f} | "
                   f"{memory_info}"
                   f"速度: {samples_per_second:.1f}样本/秒 | "
                   f"剩余: {format_time(remaining)}")
            print(f"回合 {episode+1}/{num_episodes} | {info}")
        else:
            print(f"回合 {episode+1}/{num_episodes}", end="\r", flush=True)
        
        # 定期评估和保存
        if (episode + 1) % eval_interval == 0:
            print("\n执行评估...")
            eval_reward = masac.evaluate(10)
            eval_rewards.append(eval_reward)
            print(f"评估结果 (回合 {episode+1}): 平均奖励 = {eval_reward:.2f}")
            
            masac.save(f"{model_save_path}/model_ep{episode+1}.pt")
            save_training_data(episode_rewards, eval_rewards, eval_interval, metadata)
            plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)
    
    print("\n训练完成!")
    total_time = time.time() - start_time
    print(f"总训练时间: {format_time(total_time)}")
    print(f"平均每回合用时: {format_time(total_time/num_episodes)}")
    print(f"样本吞吐量: {num_episodes * steps_per_collection / total_time:.1f} 样本/秒")
    
    # 保存最终模型
    masac.save(f"{model_save_path}/final_model.pt")
    metadata['total_time'] = format_time(total_time)
    metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_training_data(episode_rewards, eval_rewards, eval_interval, metadata)
    plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)
    
    # 关闭环境
    parallel_envs.close()
    
    # 测试训练好的模型
    print("是否测试训练好的模型? (y/n)")
    choice = input().lower()
    if choice == 'y':
        test_agent(masac, env_name)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()