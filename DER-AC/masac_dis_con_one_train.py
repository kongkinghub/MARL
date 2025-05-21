import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
from masac_dis_con_one import MASAC  # 导入修改后的MASAC类

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 先尝试黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置结果保存路径
# model_save_path = 'C:/learning/models/DER-AC_line_n6'
# results_save_path = 'C:/learning/results/DER-AC_line_n6'

# os.makedirs(model_save_path, exist_ok=True)
# os.makedirs(results_save_path, exist_ok=True)

# 算法和环境名称
algorithm_name = 'DER-AC'
env_name = 'simple_coop_push_n6'

# 设置结果保存路径
model_save_path = f'C:/learning/models/{algorithm_name}_{env_name}'
results_save_path = f'C:/learning/results/{algorithm_name}_{env_name}'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

# 简单进度条函数
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

def main():
    start_time = time.time()
    print(f"当前日期和时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户登录名: kongkinghub")
    print("=" * 50)
    
    print("创建环境...")
    env = make_env(env_name)
    env.seed(0)  # 设置随机种子
    torch.manual_seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print("初始化分布式MASAC智能体（带参数共识）...")
    masac = MASAC(
        env=env, 
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
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 256
    eval_interval = 1000
    
    # 元数据
    metadata = {
        'algorithm': algorithm_name,
        'environment': env_name,
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'batch_size': batch_size,
        'actor_lr': masac.actor_lr,
        'critic_lr': masac.critic_lr,
        'alpha_lr': masac.alpha_lr,
        'gamma': masac.gamma,
        'tau': masac.tau,
        'buffer_size': masac.buffer_size,
        'hidden_dim': masac.hidden_dim,
        'consensus_rho': masac.consensus_rho,
        'use_consensus': masac.use_consensus,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("预填充缓冲区...")
    prefill_count = 0
    states = env.reset()
    
    while len(masac.replay_buffer) < batch_size:
        actions = [
            np.random.uniform(low=masac.min_action, high=masac.max_action, size=masac.action_dim)
            for _ in range(masac.n_agents)
        ]
        
        next_states, rewards, dones, _ = env.step(actions)
        masac.replay_buffer.push(states, actions, rewards, next_states, dones)
        states = env.reset() if any(dones) else next_states
        prefill_count += 1
        
        if prefill_count % 5 == 0:
            print(progress_bar(len(masac.replay_buffer), batch_size, info="预填充"), end="", flush=True)
    
    print(f"\n预填充完成! 收集了{len(masac.replay_buffer)}个样本")
    
    print(f"开始训练，共{num_episodes}个回合...")
    episode_rewards = []
    eval_rewards = []
    
    for episode in range(num_episodes):
        episode_start = time.time()
        states = env.reset()
        episode_reward = 0
        actor_losses = []
        
        for step in range(max_steps):
            actions = masac.take_action(states, explore=True)
            next_states, rewards, dones, _ = env.step(actions)
            masac.replay_buffer.push(states, actions, rewards, next_states, dones)
            
            episode_reward += np.sum(rewards)  # 对合作任务，取总和奖励
            
            loss = masac.update(batch_size)
            if loss is not None:
                actor_losses.append(loss)
            
            states = next_states
            if any(dones):
                break
        
        episode_rewards.append(episode_reward)
        
        elapsed = time.time() - start_time
        progress = (episode + 1) / num_episodes
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        episode_time = time.time() - episode_start
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(actor_losses) if actor_losses else 0
            
            info = (f"奖励: {episode_reward:.1f} | "
                   f"平均: {avg_reward:.1f} | "
                   f"损失: {avg_loss:.4f} | "
                   f"速度: {episode_time:.1f}秒/回合 | "
                   f"剩余: {format_time(remaining)}")
            print(progress_bar(episode + 1, num_episodes, info=info), flush=True)
        else:
            print(progress_bar(episode + 1, num_episodes), end="", flush=True)
        
        if (episode + 1) % eval_interval == 0:
            print("\n执行评估...")
            eval_reward = masac.evaluate(10)
            eval_rewards.append(eval_reward)
            print(f"评估结果 (回合 {episode+1}): 平均奖励 = {eval_reward:.2f}")
            
            masac.save(f"{model_save_path}/model_ep{episode+1}.pt")
            save_training_data(episode_rewards, eval_rewards, eval_interval=eval_interval, metadata=metadata)
            plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)
    
    print("\n训练完成!")
    total_time = time.time() - start_time
    print(f"总训练时间: {format_time(total_time)}")
    print(f"平均每回合用时: {format_time(total_time/num_episodes)}")
    
    masac.save(f"{model_save_path}/final_model.pt")
    metadata['total_time'] = format_time(total_time)
    metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_training_data(episode_rewards, eval_rewards, eval_interval=eval_interval, metadata=metadata)
    plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()