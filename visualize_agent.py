"""
遊戲畫面重現腳本 (Game Screen Reproduction Script)

此腳本載入預先訓練好的DQN模型，並在視覺化模式下運行冰球遊戲環境，
以便觀察智能體在不同實驗中的表現。本次實際使用的是 "exp_20250430_014335" 實驗
中的訓練結果，使用了 checkpoint_6500.pt 模型檔案（或最新可用的checkpoint）。

使用方法:
    python visualize_agent.py                  # 使用最新的checkpoint
    python visualize_agent.py --checkpoint checkpoint_1000.pt  # 指定checkpoint
    python visualize_agent.py --speed 0.5      # 慢動作觀看
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

# 添加父目錄到路徑以導入config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.env_wrappers import make_atari_env
from src.dqn_agent import DQNAgent
from src.device_utils import get_device

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="Visualize trained agent playing Ice Hockey")
    
    # 實驗ID和模型參數
    parser.add_argument('--exp_id', type=str, default='20250430_014335',
                        help='實驗ID (results/models/中的資料夾名稱，默認為20250430_014335)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='要載入的特定checkpoint檔案 (例如: checkpoint_1000.pt)')
    parser.add_argument('--latest', action='store_true', default=True,
                        help='若設定，則使用最新的checkpoint檔案 (默認為True)')
    
    # 環境參數
    parser.add_argument('--difficulty', type=int, default=0,
                        help='遊戲難度 (0-4，其中0最簡單)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='遊戲速度倍數 (值小於1.0時會放慢遊戲)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='要運行的遊戲回合數')
    
    return parser.parse_args()

def get_latest_checkpoint(exp_path):
    """尋找實驗目錄中最新的checkpoint檔案"""
    checkpoints = [f for f in os.listdir(exp_path) if f.startswith('checkpoint_') and f.endswith('.pt')]
    
    if not checkpoints:
        return None
    
    # 按照數字排序checkpoint檔案
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest = checkpoints[-1]
    print(f"發現{len(checkpoints)}個checkpoint檔案，最新的是: {latest}")
    return latest  # 返回最新的

def preprocess_observation(observation):
    """預處理觀察數據以符合PyTorch卷積層的輸入要求
    
    遊戲環境返回形狀為(4,84,84,1)的觀察數據，但PyTorch卷積層需要(batch,channel,height,width)格式
    """
    # 如果觀察是形狀為 (4, 84, 84, 1) 的NumPy陣列
    if isinstance(observation, np.ndarray) and observation.shape == (4, 84, 84, 1):
        # 移除最後一個維度並轉置為 (4, 84, 84)
        observation = observation.squeeze(-1)
        
    # 確保是PyTorch張量
    if not isinstance(observation, torch.Tensor):
        observation = torch.FloatTensor(observation)
        
    return observation

def load_agent(exp_id, checkpoint_file=None, use_latest=True):
    """載入預先訓練好的智能體
    
    參數:
        exp_id: 實驗ID，對應results/models/中的資料夾名
        checkpoint_file: 指定要載入的checkpoint檔案名
        use_latest: 是否使用最新的checkpoint檔案
        
    返回:
        agent: 已載入模型的DQN智能體
    """
    # 確定模型目錄
    model_dir = os.path.join('results', 'models', f'exp_{exp_id}')
    
    if not os.path.exists(model_dir):
        print(f"錯誤: 實驗目錄 {model_dir} 不存在!")
        sys.exit(1)
    
    # 確定要載入的checkpoint檔案
    if checkpoint_file is None and use_latest:
        checkpoint_file = get_latest_checkpoint(model_dir)
        print(f"使用最新的checkpoint: {checkpoint_file}")
    
    if checkpoint_file is None:
        print("錯誤: 未指定checkpoint檔案且未設置--latest!")
        sys.exit(1)
    
    checkpoint_path = os.path.join(model_dir, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: Checkpoint檔案 {checkpoint_path} 不存在!")
        sys.exit(1)
    
    # 創建環境以獲取state_shape和action_space_size
    env = make_atari_env(render_mode=None, training=False)
    
    # 檢查觀察空間的形狀
    state_shape = env.observation_space.shape
    print(f"觀察空間形狀: {state_shape}")
    
    # 對於 (4, 84, 84, 1) 的狀態，將其調整為 (4, 84, 84)
    if len(state_shape) == 4 and state_shape[-1] == 1:
        state_shape = state_shape[:-1]
    
    action_space_size = env.action_space.n
    print(f"動作空間: {env.action_space}")
    env.close()
    
    # 創建智能體
    print(f"創建DQN智能體，狀態形狀={state_shape}，動作空間大小={action_space_size}")
    agent = DQNAgent(state_shape, action_space_size, use_per=True)
    
    # 載入模型
    print(f"正在從{checkpoint_path}載入模型...")
    if not agent.load_model(checkpoint_path):
        print("錯誤: 模型載入失敗!")
        sys.exit(1)
    
    print(f"成功載入'{exp_id}'實驗的'{checkpoint_file}'模型檔案")
    
    # 設置為評估模式（禁用探索）
    agent.set_evaluation_mode(True)
    
    return agent

def visualize_gameplay(agent, difficulty=0, speed=1.0, num_episodes=3):
    """在視覺化模式下運行智能體
    
    參數:
        agent: 已訓練的DQN智能體
        difficulty: 遊戲難度級別(0-4)
        speed: 遊戲速度倍數，小於1.0時放慢
        num_episodes: 要運行的遊戲回合數
    """
    # 創建環境
    env = make_atari_env(render_mode="human", training=False, difficulty=difficulty)
    total_rewards = []
    
    print(f"\n開始視覺化遊戲畫面，難度={difficulty}，速度={speed}倍，回合數={num_episodes}")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"\n開始第{episode+1}/{num_episodes}回合")
        
        while not (done or truncated):
            # 預處理觀察數據
            processed_observation = preprocess_observation(observation)
            
            # 選擇動作
            action = agent.select_action(processed_observation, evaluate=True)
            
            # 執行動作
            observation, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # 顯示進度
            if steps % 10 == 0:
                print(f"步數: {steps}, 當前獎勵: {total_reward}", end="\r")
            
            # 控制遊戲速度
            if speed < 1.0:
                time.sleep((1.0 - speed) * 0.1)  # 降低速度的延遲
        
        total_rewards.append(total_reward)
        print(f"\n第{episode+1}回合在{steps}步後結束，總獎勵: {total_reward}")
    
    env.close()
    
    # 顯示總結
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n{num_episodes}回合的結果總結:")
    print(f"平均獎勵: {avg_reward:.2f}")
    print(f"最低獎勵: {min(total_rewards)}")
    print(f"最高獎勵: {max(total_rewards)}")

def main():
    # 解析命令行參數
    args = parse_arguments()
    
    print(f"準備重現實驗'{args.exp_id}'的遊戲畫面")
    
    # 載入智能體
    agent = load_agent(args.exp_id, args.checkpoint, args.latest)
    
    # 視覺化遊戲畫面
    visualize_gameplay(agent, args.difficulty, args.speed, args.episodes)

if __name__ == "__main__":
    main()