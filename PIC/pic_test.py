import sys
import numpy as np
import torch
import time
import os
from datetime import datetime
import argparse
from pic import PIC, ActorNetwork, PermutationInvariantCritic

# 添加环境路径
sys.path.append('C:/')
from learning.envs.make_env import make_env

def test_trained_model(model_path, num_episodes=5, save_video=False):
    """测试训练好的PIC模型并可视化"""
    # 初始化
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前用户: kongkinghub")
    print("=" * 50)
    print(f"加载模型: {model_path}")
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在!")
            return
            
        # 创建环境
        env = make_env("simple_line_n6")
        print("环境已创建: simple_line_n6")
        
        # 检测设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建PIC智能体结构
        n_agents = env.n
        state_dim = env.observation_space[0].shape[0]
        action_dim = env.action_space[0].shape[0]
        hidden_dim = 128  # 与训练时相同
        
        # 创建网络
        actors = []
        for _ in range(n_agents):
            actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
            actors.append(actor)
            
        critic = PermutationInvariantCritic(state_dim, action_dim, n_agents, hidden_dim).to(device)
        
        # 加载模型参数
        checkpoint = torch.load(model_path, map_location=device)
        
        # 加载actor参数
        for i, actor in enumerate(actors):
            actor.load_state_dict(checkpoint['actors'][i])
            
        # 加载critic参数
        critic.load_state_dict(checkpoint['critic'])
        
        print("模型参数加载成功")
        
        # 设置视频录制
        if save_video:
            video_dir = "C:/learning/videos"
            os.makedirs(video_dir, exist_ok=True)
            env = make_env("simple_line_n6", render_mode="rgb_array")
            import imageio
            video_path = f"{video_dir}/pic_test_{int(time.time())}.mp4"
            writer = imageio.get_writer(video_path, fps=10)
            print(f"视频将保存到: {video_path}")
        
        # 测试模型
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n回合 {episode+1}/{num_episodes}:")
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            max_steps = 25
            frames = []
            
            while not done and step < max_steps:
                # 渲染
                if save_video:
                    frames.append(env.render())
                else:
                    env.render()
                    time.sleep(0.2)  # 减缓显示速度方便观察
                
                # 选择动作
                actions = []
                for i, actor in enumerate(actors):
                    state_tensor = torch.FloatTensor(state[i]).to(device).unsqueeze(0)
                    with torch.no_grad():
                        mean, _ = actor(state_tensor)
                        action = mean.cpu().numpy().flatten()
                    actions.append(action)
                
                # 执行动作
                next_state, reward, done, _ = env.step(actions)
                
                # 计算并显示奖励
                step_reward = np.mean(reward)
                episode_reward += step_reward
                print(f"步骤 {step+1}: 奖励 = {step_reward:.4f}")
                
                # 更新状态
                state = next_state
                done = any(done)
                step += 1
            
            # 保存视频帧
            if save_video and frames:
                for frame in frames:
                    writer.append_data(frame)
            
            # 统计结果
            total_rewards.append(episode_reward)
            print(f"回合 {episode+1} 结束: 总奖励 = {episode_reward:.4f}, 步数 = {step}")
        
        # 关闭视频写入器
        if save_video:
            writer.close()
            print(f"视频已保存到: {video_path}")
        
        # 显示统计结果
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        min_reward = np.min(total_rewards)
        max_reward = np.max(total_rewards)
        
        print("\n" + "=" * 50)
        print(f"测试完成! {num_episodes} 个回合的统计结果:")
        print(f"平均奖励: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"最高奖励: {max_reward:.4f}")
        print(f"最低奖励: {min_reward:.4f}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 确保环境正确关闭
        try:
            env.close()
        except:
            pass

# 比较不同算法的性能可视化
def compare_algorithms(pic_result_path, masac_result_path, output_path="C:/learning/results/comparison"):
    """比较PIC和MASAC算法的性能"""
    import matplotlib.pyplot as plt
    
    print(f"比较算法性能...")
    os.makedirs(output_path, exist_ok=True)
    
    # 加载数据
    try:
        pic_rewards = np.load(f"{pic_result_path}/rewards.npy")
        pic_eval_rewards = np.load(f"{pic_result_path}/eval_rewards.npy")
        
        masac_rewards = np.load(f"{masac_result_path}/rewards.npy")
        masac_eval_rewards = np.load(f"{masac_result_path}/eval_rewards.npy")
        
        # 训练曲线比较
        plt.figure(figsize=(12, 8))
        
        # 绘制训练奖励曲线
        plt.subplot(2, 1, 1)
        # 平滑化处理
        window = 20
        if len(pic_rewards) > window:
            pic_smoothed = np.convolve(pic_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(pic_rewards)), pic_smoothed, label='PIC')
        else:
            plt.plot(pic_rewards, label='PIC')
            
        if len(masac_rewards) > window:
            masac_smoothed = np.convolve(masac_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(masac_rewards)), masac_smoothed, label='MASAC')
        else:
            plt.plot(masac_rewards, label='MASAC')
            
        plt.xlabel('训练回合')
        plt.ylabel('平均奖励')
        plt.title('训练过程奖励曲线比较')
        plt.grid(True)
        plt.legend()
        
        # 绘制评估奖励曲线
        plt.subplot(2, 1, 2)
        
        # 计算对应的训练回合数
        pic_eval_episodes = np.linspace(0, len(pic_rewards), len(pic_eval_rewards), endpoint=False).astype(int)
        masac_eval_episodes = np.linspace(0, len(masac_rewards), len(masac_eval_rewards), endpoint=False).astype(int)
        
        plt.plot(pic_eval_episodes, pic_eval_rewards, 'o-', label='PIC')
        plt.plot(masac_eval_episodes, masac_eval_rewards, 's-', label='MASAC')
        
        plt.xlabel('训练回合')
        plt.ylabel('评估奖励')
        plt.title('评估性能比较')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/algorithm_comparison.png")
        plt.close()
        
        print(f"比较结果已保存到 {output_path}/algorithm_comparison.png")
        
        # 统计最终性能
        pic_final = np.mean(pic_eval_rewards[-3:])
        masac_final = np.mean(masac_eval_rewards[-3:])
        
        print("\n最终性能比较:")
        print(f"PIC平均奖励: {pic_final:.4f}")
        print(f"MASAC平均奖励: {masac_final:.4f}")
        print(f"性能差异: {(pic_final - masac_final):.4f}")
        
    except Exception as e:
        print(f"比较过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试PIC训练模型")
    parser.add_argument("--model", type=str, default="C:/learning/models/PIC_line_n6/final_model.pt", 
                        help="模型文件路径")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--save_video", action="store_true", help="是否保存视频")
    parser.add_argument("--compare", action="store_true", help="比较PIC与MASAC算法")
    parser.add_argument("--pic_results", type=str, default="C:/learning/results/PIC_line_n6", 
                        help="PIC结果目录")
    parser.add_argument("--masac_results", type=str, default="C:/learning/results/MASAC_distributed_line_n6", 
                        help="MASAC结果目录")
    args = parser.parse_args()
    
    # 测试模型
    test_trained_model(args.model, args.episodes, args.save_video)
    
    # 比较算法
    if args.compare:
        compare_algorithms(args.pic_results, args.masac_results)