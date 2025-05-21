import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from datetime import datetime
from masac import MASAC

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 先尝试黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加环境路径
sys.path.append('C:/')  # 调整为您的learning模块路径
from learning.envs.make_env import make_env

# 设置结果保存路径
algorithm_name = "MASAC"
env_name = 'simple_spread_n6'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


model_save_path = f'C:/learning/models/{env_name}/{algorithm_name}'
results_save_path = f'C:/learning/results/{env_name}/{algorithm_name}'

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

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
    plt.figtext(0.01, 0.01, f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=8)
    
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
    env.seed(0)  # 设置随机种子
    torch.manual_seed(0)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练参数
    num_episodes = 15000
    max_steps = 25
    batch_size = 256
    update_interval = 1
    eval_interval = 1000
    
    # 创建MASAC智能体
    print("初始化MASAC智能体...")
    masac = MASAC(
        env=env, 
        device=device,
        actor_lr=0.005,
        critic_lr=0.01, 
        alpha_lr=0.005,
        hidden_dim=128,
        gamma=0.95, 
        tau=0.01,
        buffer_size=375000
    )
    
    # 开始训练
    print(f"开始训练，共{num_episodes}个回合...")
    rewards, eval_rewards = masac.train(
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
        "智能体数量": env.n,
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
    
    print("训练和评估完成！")

if __name__ == "__main__":
    main()