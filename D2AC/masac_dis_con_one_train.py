import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import time
from datetime import datetime
from masac_dis_con_one_deterministic import MASAC  # 导入确定性策略版本的MASAC

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS'] 
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env

# 设置结果保存路径与算法标识
algorithm_name = "D2_AC"  # 使用确定性策略的名称
env_name = "simple_line_n6"
# experiment_info = "consensus_rho_0.4"
# model_save_path = f'C:/learning/models/{algorithm_name}_{env_name}_{experiment_info}'
# results_save_path = f'C:/learning/results/{algorithm_name}_{env_name}_{experiment_info}'
model_save_path = f'C:/learning/models/{algorithm_name}_{env_name}'
results_save_path = f'C:/learning/results/{algorithm_name}_{env_name}'

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

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
    plt.figtext(0.01, 0.01, f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M')} | ρ={experiment_info.split('_')[-1]} | 确定性策略", fontsize=8)
    
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
    print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: kongkinghub")
    print("=" * 50)
    
    # 创建环境
    print("创建环境...")
    env = make_env(env_name)
    env.seed(0)  # 设置随机种子
    torch.manual_seed(0)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建确定性策略版本的MASAC智能体
    print(f"初始化确定性策略{algorithm_name}智能体（参数共识ρ={experiment_info.split('_')[-1]}）...")
    masac = MASAC(
        env=env, 
        device=device,
        actor_lr=0.005,
        critic_lr=0.01, 
        alpha_lr=0.005,  # 虽然确定性版本不使用alpha，但保留参数确保接口一致
        hidden_dim=128,
        gamma=0.95, 
        tau=0.01,
        buffer_size=375000,
        consensus_rho=0.4,  # 通信网络连通率参数ρ
        use_consensus=True   # 启用参数共识更新
    )
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 256
    eval_interval = 1000
    
    # 预填充缓冲区
    print("预填充缓冲区...")
    prefill_count = 0
    states = env.reset()
    
    while len(masac.replay_buffer) < batch_size:
        # 随机动作
        actions = []
        for _ in range(masac.n_agents):
            action = np.random.uniform(low=masac.min_action, high=masac.max_action, size=masac.action_dim)
            actions.append(action)
        
        # 环境交互
        next_states, rewards, dones, _ = env.step(actions)
        masac.replay_buffer.push(states, actions, rewards, next_states, dones)
        states = env.reset() if any(dones) else next_states
        prefill_count += 1
        
        # 显示进度
        if prefill_count % 5 == 0:
            print(progress_bar(len(masac.replay_buffer), batch_size, info="预填充"), end="")
    
    print(f"\n预填充完成! 收集了{len(masac.replay_buffer)}个样本")
    
    # 训练循环
    episode_rewards = []
    eval_rewards = []
    
    print(f"开始训练，共{num_episodes}个回合...")
    
    for episode in range(num_episodes):
        episode_start = time.time()
        states = env.reset()
        episode_reward = 0
        actor_losses = []
        
        # 确定性策略特有：重置每个智能体的噪声
        for agent in masac.agents:
            agent.reset_noise()
        
        # 一个回合的交互
        for step in range(max_steps):
            # 选择动作（确定性策略+噪声探索）
            actions = masac.take_action(states, explore=True)
            
            # 执行动作
            next_states, rewards, dones, _ = env.step(actions)
            
            # 存储经验
            masac.replay_buffer.push(states, actions, rewards, next_states, dones)
            
            # 计算奖励 - 使用求和而非平均
            episode_reward += np.sum(rewards)  # 确保使用求和
            
            # 更新网络（包括参数共识更新）
            loss = masac.update(batch_size)
            if loss is not None:
                actor_losses.append(loss)
            
            # 更新状态
            states = next_states
            
            # 检查终止条件
            if any(dones):
                break
        
        # 保存奖励
        episode_rewards.append(episode_reward)
        
        # 计算训练进度信息
        elapsed = time.time() - start_time
        progress = (episode + 1) / num_episodes
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        episode_time = time.time() - episode_start
        
        # 显示进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(actor_losses) if actor_losses else 0
            
            info = (f"奖励: {episode_reward:.1f} | "
                   f"平均: {avg_reward:.1f} | "
                   f"损失: {avg_loss:.4f} | "
                   f"速度: {episode_time:.1f}秒/回合 | "
                   f"剩余: {format_time(remaining)}")
            
            print(progress_bar(episode + 1, num_episodes, info=info))
        else:
            # 简单进度条
            print(progress_bar(episode + 1, num_episodes), end="")
        
        # 评估和保存
        if (episode + 1) % eval_interval == 0:
            print("\n执行评估...")
            
            # 确定性策略评估（不使用探索）
            eval_reward = masac.evaluate(10)
            eval_rewards.append(eval_reward)
            print(f"评估结果 (回合 {episode+1}): 平均奖励 = {eval_reward:.2f}")
            
            # 保存模型和数据
            masac.save(f"{model_save_path}/model_ep{episode+1}.pt")
            
            # 绘制并保存训练曲线
            plot_results(episode_rewards, eval_rewards, window=100)
            print(f"中间结果已保存 (回合 {episode+1})")
    
    # 训练结束
    print("\n训练完成!")
    total_time = time.time() - start_time
    print(f"总训练时间: {format_time(total_time)}")
    print(f"平均每回合用时: {format_time(total_time/num_episodes)}")
    
    # 保存训练元数据
    metadata = {
        "算法": algorithm_name,
        "环境": env_name,
        "实验信息": experiment_info,
        "策略类型": "确定性策略",
        "回合数": num_episodes,
        "最大步数": max_steps,
        "批次大小": batch_size,
        "评估间隔": eval_interval,
        "训练时间(秒)": total_time,
        "训练设备": device,
        "智能体数量": env.n,
        "共识参数ρ": masac.consensus_network.rho,
        "奖励计算方式": "智能体奖励总和",
        "最终平均奖励(最后100回合)": np.mean(episode_rewards[-100:]),
        "最终评估奖励": eval_rewards[-1] if eval_rewards and len(eval_rewards) > 0 else "N/A",
        "训练日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存最终模型
    masac.save(f"{model_save_path}/final_model.pt")
    
    # 保存训练数据(CSV和NPY格式)
    save_training_data(episode_rewards, eval_rewards, eval_interval, metadata)
    
    # 绘制最终训练曲线
    plot_results(episode_rewards, eval_rewards, window=100, show=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()