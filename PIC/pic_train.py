import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
from pic import PIC  # 导入修改后的PIC算法

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 先尝试黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置结果保存路径
# model_save_path = 'C:/learning/pic/models/PIC_simple_line_n6'
# results_save_path = 'C:/learning/pic/results/PIC_simple_line_n6'
# os.makedirs(model_save_path, exist_ok=True)
# os.makedirs(results_save_path, exist_ok=True)

# 算法和环境名称
algorithm_name = 'PIC'
env_name = 'simple_spread_n6'

model_save_path = f'C:/learning/models/{env_name}/{algorithm_name}'
results_save_path = f'C:/learning/results/{env_name}/{algorithm_name}'

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
            # episode_reward += np.mean(reward)  # 假设全局奖励
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
                           f"环境: {env_name} | 确定性策略", fontsize=8)
    
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
    
    # 创建环境
    print("创建环境...")
    env = make_env(env_name)
    env.seed(0)  # 设置随机种子
    torch.manual_seed(0)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化PIC智能体
    print("初始化PIC智能体...")
    agent = PIC(
        env=env, 
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
        'actor_lr': agent.actor_optimizers[0].param_groups[0]['lr'],
        'critic_lr': agent.critic_optimizer.param_groups[0]['lr'],
        'alpha': agent.alpha,
        'gamma': agent.gamma,
        'tau': agent.tau,
        'buffer_size': agent.replay_buffer.capacity,
        'hidden_dim': 128,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 经验收集
    print("开始初始经验收集...")
    collected_steps = 0
    state = env.reset()
    
    while collected_steps < warmup_steps:
        action = [
            np.random.uniform(low=env.action_space[i].low, high=env.action_space[i].high) 
            for i in range(agent.n_agents)
        ]
        
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        collected_steps += 1
        if collected_steps % 100 == 0:
            print(f"已收集 {collected_steps}/{warmup_steps} 步经验")
            
        if any(done):
            state = env.reset()
        else:
            state = next_state
    
    print(f"初始经验收集完成，缓冲区大小: {len(agent.replay_buffer)}")
    
    # 开始训练
    print(f"开始训练，共{num_episodes}个回合...")
    episode_rewards = []
    eval_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            # episode_reward += np.mean(reward)
            episode_reward += np.sum(reward)  # 对合作任务，取总和奖励

            step += 1
            
            if len(agent.replay_buffer) >= batch_size and step % update_interval == 0:
                agent.update_parameters(batch_size)
                
            if any(done):
                break
        
        episode_rewards.append(episode_reward)
        
        # 计算训练进度信息
        elapsed = time.time() - start_time
        progress = (episode + 1) / num_episodes
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_actor_loss = np.mean(agent.actor_losses[-100:]) if agent.actor_losses else 0
            avg_critic_loss = np.mean(agent.critic_losses[-100:]) if agent.critic_losses else 0
            
            info = (f"奖励: {episode_reward:.1f} | "
                   f"平均: {avg_reward:.1f} | "
                   f"Actor损失: {avg_actor_loss:.4f} | "
                   f"Critic损失: {avg_critic_loss:.4f} | "
                   f"剩余: {format_time(remaining)}")
            
            print(progress_bar(episode + 1, num_episodes, info=info), flush=True)
        
        # 评估并保存数据
        if (episode + 1) % eval_interval == 0:
            print("\n执行评估...")
            eval_reward = evaluate_policy(agent, env)
            eval_rewards.append(eval_reward)
            print(f"评估结果 (回合 {episode+1}): 平均奖励 = {eval_reward:.2f}")
            
            # 保存训练数据
            save_training_data(episode_rewards, eval_rewards, eval_interval=eval_interval, metadata=metadata)
            
            # 绘制并保存训练曲线
            plot_results(episode_rewards, eval_rewards, window=10, save=True, show=False)
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            model_dict = {
                'actors': [actor.state_dict() for actor in agent.actors],
                'actors_target': [actor_target.state_dict() for actor_target in agent.actors_target],
                'critic': agent.critic.state_dict(),
                'critic_target': agent.critic_target.state_dict()
            }
            torch.save(model_dict, f"{model_save_path}/model_ep{episode+1}.pt")
            print(f"\n模型已保存: {model_save_path}/model_ep{episode+1}.pt")
    
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