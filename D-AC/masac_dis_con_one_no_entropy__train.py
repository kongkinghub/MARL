import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from datetime import datetime
from masac_dis_con_one import MASAC  # 导入分布式共识MASAC

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 先尝试黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置算法信息和保存路径
algorithm_name = "D-AC"  # 修改为反映无熵正则化特性的名称
env_name = "simple_triangle_n6"
# experiment_info = f"consensus_rho_0.7"  # 添加实验特定参数信息
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置保存路径
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

def plot_results(rewards, eval_rewards=None, window=100, save=True, show=True):
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
    plt.figtext(0.01, 0.01, f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 无熵正则化版本 (α=0)", fontsize=8)
    
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
    print(f"当前日期时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户: kongkinghub")
    print("=" * 50)
    
    # 创建环境
    print("创建环境...")
    env = make_env(env_name)
    env.seed(0)  # 设置随机种子以确保可重复性
    torch.manual_seed(0) # 设置PyTorch随机种子
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 256
    update_interval = 1
    eval_interval = 1000
    
    # 创建MASAC智能体 - 注意这是无熵正则化版本，不使用alpha_lr参数
    print("初始化分布式MASAC智能体（带参数共识，无熵正则化）...")
    masac = MASAC(
        env=env, 
        device=device,
        actor_lr=0.005,
        critic_lr=0.01, 
        # alpha_lr参数已在无熵正则化版本中移除
        hidden_dim=128,
        gamma=0.95, 
        tau=0.01,
        buffer_size=375000,
        consensus_rho=0.4,  # 通信网络连通率参数ρ
        use_consensus=True   # 启用参数共识更新
    )
    
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
    
    # 开始训练
    print(f"开始训练，共{num_episodes}个回合...")
    
    # 初始化奖励记录
    episode_rewards = []
    eval_rewards = []
    
    # 训练循环
    for episode in range(num_episodes):
        episode_start = time.time()
        states = env.reset()
        episode_reward = 0
        actor_losses = []
        
        # 单回合训练
        for step in range(max_steps):
            # 选择动作
            actions = masac.take_action(states, explore=True)
            
            # 执行动作
            next_states, rewards, dones, _ = env.step(actions)
            
            # 存储经验
            masac.replay_buffer.push(states, actions, rewards, next_states, dones)
            
            # 计算奖励
            # episode_reward += np.mean(rewards)
            episode_reward += np.sum(rewards)  # 对合作任务，取总和奖励
            
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
        progress_percent = (episode + 1) / num_episodes
        estimated_total = elapsed / progress_percent if progress_percent > 0 else 0
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
            eval_reward = masac.evaluate(10)
            eval_rewards.append(eval_reward)
            print(f"评估结果 (回合 {episode+1}): 平均奖励 = {eval_reward:.2f}")
            
            # 保存中间结果
            temp_model_path = f"{model_save_path}/model_ep{episode+1}.pt"
            masac.save(temp_model_path)
            
            # 保存训练曲线
            plot_results(episode_rewards, eval_rewards, window=100, show=False)
            
            print(f"中间模型和数据已保存 (回合 {episode+1})")
    
    # 计算训练时间
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n训练完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 保存训练元数据
    metadata = {
        "算法": algorithm_name,
        "环境": env_name,
        # "实验信息": experiment_info,
        "回合数": num_episodes,
        "最大步数": max_steps,
        "批次大小": batch_size,
        "更新间隔": update_interval,
        "评估间隔": eval_interval,
        "训练时间(秒)": training_time,
        "训练设备": device,
        "智能体数量": env.n,
        "共识参数ρ": masac.consensus_network.rho,
        "熵正则化": "关闭 (α=0)",
        "最终平均奖励(最后100回合)": np.mean(episode_rewards[-100:]),
        "最终评估奖励": eval_rewards[-1] if eval_rewards and len(eval_rewards) > 0 else "N/A",
        "训练日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存最终模型
    final_model_path = f"{model_save_path}/{algorithm_name}_final_model.pt"
    masac.save(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    # 保存训练数据
    save_training_data(episode_rewards, eval_rewards, eval_interval, metadata)
    
    # 绘制训练曲线
    plot_results(episode_rewards, eval_rewards, window=100)
    
    # 评估最终模型
    print("正在进行最终评估...")
    final_eval_reward = masac.evaluate(eval_episodes=20)
    print(f"最终评估平均奖励: {final_eval_reward:.2f}")
    
    print("训练和评估完成！")
    print(f"注意: 这是无熵正则化版本的训练结果 (α=0)，用于与标准版本比较。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()