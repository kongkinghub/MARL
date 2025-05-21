import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
from collections import deque
import multiprocessing
from tqdm import tqdm

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env
from masac_dis_con_one import MASAC  # 导入你的MASAC类

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 算法和环境名称
algorithm_name = 'DER-AC'
env_name = 'simple_spread_n6'

# 设置结果保存路径
model_save_path = f'C:/learning/models/{env_name}/{algorithm_name}'
results_save_path = f'C:/learning/results/{env_name}/{algorithm_name}'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

# 智能设备选择和优化
def setup_device():
    """设置并返回计算设备，同时应用相应的性能优化"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # 启用CUDA优化
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        # 尝试设置TF32精度 (Ampere及更新架构上的功能)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("启用TF32精度加速")
        except:
            pass
    else:
        device = torch.device("cpu")
        num_cores = multiprocessing.cpu_count()
        # 设置PyTorch使用所有可用核心进行并行计算
        torch.set_num_threads(num_cores)
        print(f"使用CPU训练 (检测到{num_cores}个核心)")
        
        # 尝试检测MKL加速
        mkl_available = hasattr(torch._C, '_mkldnn_enabled') and torch._C._mkldnn_enabled()
        if mkl_available:
            print("MKL加速可用")
        
    return device

# 并行环境封装器
class ParallelEnvs:
    def __init__(self, env_name, num_envs=4):
        """创建多个并行环境实例"""
        self.envs = [make_env(env_name) for _ in range(num_envs)]
        for env in self.envs:
            env.seed(0)  # 设置随机种子
            torch.manual_seed(0)
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
    
    def step_one(self, env_idx, actions):
        """在单个环境中执行动作"""
        return self.envs[env_idx].step(actions)

    def render(self, env_idx=0):
        """渲染指定的环境实例"""
        return self.envs[env_idx].render()
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

# 完全重新设计的经验收集函数，确保与非并行版本一致
def collect_experience_parallel(masac, parallel_envs, num_episodes=1, max_steps=25, explore=True):
    """
    并行收集完整回合的经验
    
    Args:
        masac: MASAC智能体
        parallel_envs: 并行环境对象
        num_episodes: 要收集的完整回合数
        max_steps: 每个回合的最大步数
        explore: 是否使用探索
        
    Returns:
        (完成回合的奖励列表, 完成的回合数, 收集的总步数)
    """
    # 初始化状态和每个环境的回合奖励
    states_list = parallel_envs.reset()
    completed_episode_rewards = []  # 存储完成回合的总奖励
    active_episode_rewards = [0.0] * parallel_envs.num_envs  # 每个环境的当前回合奖励
    steps_in_episode = [0] * parallel_envs.num_envs  # 每个环境当前回合的步数
    total_steps = 0
    completed_episodes = 0
    
    # 继续直到收集足够的完整回合
    while completed_episodes < num_episodes:
        # 批量生成动作
        actions_list = []
        for states in states_list:
            actions = masac.take_action(states, explore=explore)
            actions_list.append(actions)
        
        # 执行动作并获取结果
        next_states_list, rewards_list, dones_list, _ = parallel_envs.step(actions_list)
        
        # 处理每个环境的数据
        for env_idx in range(parallel_envs.num_envs):
            states = states_list[env_idx]
            actions = actions_list[env_idx]
            rewards = rewards_list[env_idx]
            next_states = next_states_list[env_idx]
            dones = dones_list[env_idx]
            
            # 保存到经验回放缓冲区
            masac.replay_buffer.push(states, actions, rewards, next_states, dones)
            
            # 更新该环境的回合奖励 (与非并行版本相同方式)
            active_episode_rewards[env_idx] += np.sum(rewards)
            steps_in_episode[env_idx] += 1
            total_steps += 1
            
            # 检查是否回合结束
            if any(dones) or steps_in_episode[env_idx] >= max_steps:
                # 记录完成的回合奖励
                completed_episode_rewards.append(active_episode_rewards[env_idx])
                completed_episodes += 1
                
                # 重置该环境
                active_episode_rewards[env_idx] = 0.0
                steps_in_episode[env_idx] = 0
                states_list[env_idx] = parallel_envs.reset_one(env_idx)
            else:
                # 继续该环境的回合
                states_list[env_idx] = next_states
                
            # 如果已经收集了足够的回合，就退出
            if completed_episodes >= num_episodes:
                break
    
    # 确保我们有回合奖励可以返回
    if len(completed_episode_rewards) == 0:
        return [0.0], 0, total_steps  # 返回默认列表避免空列表问题
    
    return completed_episode_rewards, completed_episodes, total_steps

# 批量更新函数
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

# 监控函数：用于定期显示系统信息
def show_system_info(device):
    """显示系统信息，包括内存使用情况和CPU/GPU利用率"""
    try:
        if torch.cuda.is_available():
            # GPU信息
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            return f"GPU内存: {allocated:.0f}MB (使用) / {reserved:.0f}MB (保留)"
        else:
            # CPU信息
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_used = memory.used / 1024**3  # GB
                memory_total = memory.total / 1024**3  # GB
                return f"CPU使用率: {cpu_percent}% | 内存: {memory_used:.1f}GB/{memory_total:.1f}GB"
            except ImportError:
                return "CPU使用中 (安装psutil库可显示更多信息)"
    except Exception as e:
        return f"信息获取失败: {e}"

def main():
    """主训练函数"""
    start_time = time.time()
    print(f"当前日期和时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户登录名: kongkinghub")
    print("=" * 50)
    
    # 设置设备（GPU或CPU）
    device = setup_device()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 根据设备类型智能调整并行度
    if torch.cuda.is_available():
        # GPU训练：根据显存大小调整
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        num_envs = min(32, max(8, int(gpu_memory_gb * 2)))  # 每GB显存分配2个环境
        batch_size = 1024  # GPU上使用更大的批处理大小
    else:
        # CPU训练：根据核心数调整
        cpu_count = multiprocessing.cpu_count()
        num_envs = min(16, max(4, cpu_count - 1))  # 保留1个核心给系统
        batch_size = 512  # CPU上使用适中的批处理大小
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    eval_interval = 1000
    
    # 优化参数
    updates_per_collection = 4  # 每次收集经验后执行的更新次数
    episodes_per_collection = 4  # 每次收集的完整回合数
    
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
    prefill_episodes = 1  # 只需要收集一个完整回合即可预填充
    prefill_rewards, prefill_episodes_completed, prefill_steps = collect_experience_parallel(
        masac,
        parallel_envs,
        num_episodes=prefill_episodes,
        max_steps=max_steps,
        explore=True
    )
    
    # 修正: 确保我们有一个初始奖励值来开始记录
    first_reward = prefill_rewards[0] if prefill_rewards and len(prefill_rewards) > 0 else 0.0
    print(f"预填充完成! 收集了 {len(masac.replay_buffer)} 个样本，首个回合奖励: {first_reward:.2f}")
    
    print(f"开始训练，共 {num_episodes} 个回合...")
    episode_rewards = [first_reward]  # 初始包含预填充回合的奖励
    eval_rewards = []
    
    # 创建进度条
    pbar = tqdm(total=num_episodes, desc="训练进度", initial=1)
    
    # 上次保存的时间
    last_save_time = time.time()
    save_interval = 900  # 15分钟自动保存一次
    
    episode_idx = 1  # 从第二个回合开始训练（因为第一个回合已作为预填充）
    
    while episode_idx < num_episodes:
        collection_start = time.time()
        
        # 收集多个回合的经验
        episodes_to_collect = min(episodes_per_collection, num_episodes - episode_idx)
        episode_rewards_batch, episodes_completed, steps_collected = collect_experience_parallel(
            masac,
            parallel_envs,
            num_episodes=episodes_to_collect,
            max_steps=max_steps,
            explore=True
        )
        
        # 记录每个完成回合的奖励
        for i in range(min(len(episode_rewards_batch), num_episodes - episode_idx)):
            episode_rewards.append(episode_rewards_batch[i])  # 修正: 添加每个单独回合的奖励
            episode_idx += 1
            pbar.update(1)
        
        # 进行多次批量更新
        losses = []
        for _ in range(updates_per_collection):
            loss = update_batch(masac, batch_size, num_updates=1)
            if loss is not None:
                losses.append(loss)
        
        avg_loss = np.mean(losses) if losses else 0
        
        # 计算运行状态
        elapsed = time.time() - start_time
        progress = episode_idx / num_episodes
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        collection_time = time.time() - collection_start
        
        # 计算最近10个回合的平均奖励
        recent_rewards = episode_rewards[-min(10, len(episode_rewards)):]
        avg_reward_10ep = np.mean(recent_rewards)
        
        # 当前回合的奖励 (最后一个添加的)
        current_reward = episode_rewards[-1] if episode_rewards else 0
        
        # 每10个回合显示详细信息
        if episode_idx % 10 == 0 or episode_idx >= num_episodes:
            samples_per_second = steps_collected / collection_time if collection_time > 0 else 0
            sys_info = show_system_info(device)
            
            pbar.clear()
            print(f"回合 {episode_idx}/{num_episodes} | "
                 f"奖励: {current_reward:.1f} | "
                 f"10ep平均: {avg_reward_10ep:.1f} | "
                 f"损失: {avg_loss:.4f} | "
                 f"速度: {samples_per_second:.1f}样本/秒 | "
                 f"{sys_info} | "
                 f"剩余: {format_time(remaining)}")
        
        # 更新进度条信息
        pbar.set_postfix({
            "奖励": f"{current_reward:.1f}", 
            "平均10ep": f"{avg_reward_10ep:.1f}", 
            "损失": f"{avg_loss:.4f}"
        })
        
        # 定期评估
        if episode_idx % eval_interval == 0:
            pbar.clear()
            print("\n执行评估...")
            eval_reward = masac.evaluate(10)
            eval_rewards.append(eval_reward)
            print(f"评估结果 (回合 {episode_idx}): 平均奖励 = {eval_reward:.2f}")
            
            masac.save(f"{model_save_path}/model_ep{episode_idx}.pt")
            save_training_data(episode_rewards, eval_rewards, eval_interval, metadata)
            plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)
        
        # 检查是否应该自动保存
        current_time = time.time()
        if current_time - last_save_time > save_interval:
            pbar.clear()
            print("\n自动保存中...")
            masac.save(f"{model_save_path}/autosave_model.pt")
            save_training_data(episode_rewards, eval_rewards, eval_interval, metadata)
            last_save_time = current_time
            print(f"自动保存完成，已训练{episode_idx}个回合")
    
    # 关闭进度条
    pbar.close()
    
    print("\n训练完成!")
    total_time = time.time() - start_time
    print(f"总训练时间: {format_time(total_time)}")
    print(f"平均每回合用时: {format_time(total_time/num_episodes)}")
    
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