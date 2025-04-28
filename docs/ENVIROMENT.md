# IceHockey (ALE/IceHockey-v5)

## 📦 Environment Summary

- **Environment Name**: `ALE/IceHockey-v5`
- **Game Time Limit**: 3 minutes
- **Goal**: Score as many goals as possible by shooting the puck into the opponent's goal
- **Key Feature**: 32 different shot angles toward the opponent's goal; players can also pass the puck by bouncing it off the rink's sides

*IceHockey 是一個 Atari 環境，建議先閱讀 [Atari Environments 總覽頁面](https://gym.openai.com/envs/#atari) 以了解通用資訊。*

---

## 🎮 Action Space

- **Type**: `Discrete(18)`
- **Description**: All possible Atari 2600 actions are enabled by default, regardless of version or `full_action_space` setting

### Our Implementation

In our implementation, we utilize all 18 possible actions and select them using an `ε-greedy` strategy. The exploration rate (ε) gradually decreases from 1.0 to 0.1 during training, transitioning the agent from exploration to exploitation of learned strategies.

*在我們的實現中，我們使用 18 個可能的動作，並通過 `ε-greedy` 策略選擇動作。隨著訓練進行，探索率 (ε) 從 1.0 逐漸減少到 0.1，使智能體從初期的探索轉向後期的利用學習到的策略。*

---

## 👀 Observation Space

- **Default**: `Box((210, 160, 3), dtype=uint8)` (RGB image)
- **Alternatives**:
  - **RAM**: `Box((128,), dtype=uint8)`
  - **Grayscale Image**: `Box((250, 160), dtype=uint8)`

### Our Preprocessing

In our implementation, we apply the following preprocessing steps:
- Convert to grayscale to reduce input dimensionality
- Resize to 84×84 pixels to standardize input size
- Stack 4 consecutive frames to provide temporal information
- Normalize pixel values to [0, 1] range
- Use frame skipping technique, making a decision every 4 frames

The final processed observation space is `(4, 84, 84)`, representing 4 stacked grayscale frames of 84×84 pixels.

*在我們的實現中，我們應用了以下預處理步驟:*
*- 轉換為灰階圖像以減少輸入維度*
*- 縮放至 84×84 像素以標準化輸入大小*
*- 堆疊 4 個連續幀以提供時間序列信息*
*- 標準化像素值到 [0, 1] 範圍*
*- 使用跳幀（frame skipping）技術，每 4 幀進行一次決策*

*最終處理後的觀察空間為 `(4, 84, 84)`，表示 4 個堆疊的 84×84 灰階幀。*

---

## 🏅 Rewards

- Points are awarded by scoring goals.
- There is no upper limit to the score other than the 3-minute game timer.

### Reward Processing

In our implementation, we process rewards as follows:
- Clip rewards to {+1, 0, -1} by taking the sign of the original reward
- This processing helps stabilize training and prevents value function divergence due to large rewards

*在我們的實現中，我們對獎勵進行了以下處理:*
*- 獎勵裁剪到 {+1, 0, -1} 範圍，通過取原始獎勵的符號來實現*
*- 這種處理有助於穩定訓練過程，防止大獎勵導致的值函數發散*

---

## ⚙️ Environment Configuration

- **Import Example**:
  ```python
  import gym
  env = gym.make("ALE/IceHockey-v5")
  ```

### Wrapper Usage

We use the following environment wrappers to enhance learning:
- `NoopResetEnv`: Executes random number of no-ops on reset to increase initial state diversity
- `MaxAndSkipEnv`: Takes max of consecutive frames to deal with flickering
- `ResizeFrame`: Resizes frames and converts to grayscale
- `NormalizedFrame`: Normalizes pixel values
- `FrameStack`: Stacks consecutive frames to provide temporal information
- `ClipRewardEnv`: Clips rewards

The complete environment creation code can be found in the `make_atari_env()` function in `src/env_wrappers.py`.

*我們使用了以下環境包裝器來增強學習效果:*
*- `NoopResetEnv`: 在重置時執行隨機數量的無操作，增加初始狀態的多樣性*
*- `MaxAndSkipEnv`: 處理連續幀之間的最大值，解決閃爍問題*
*- `ResizeFrame`: 調整幀大小並轉換為灰階*
*- `NormalizedFrame`: 標準化像素值*
*- `FrameStack`: 堆疊連續幀以提供時間序列信息*
*- `ClipRewardEnv`: 裁剪獎勵*

*完整的環境創建代碼可在 `src/env_wrappers.py` 中的 `make_atari_env()` 函數中找到。*
