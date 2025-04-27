# IceHockey (ALE/IceHockey-v5)

![Ice Hockey]

IceHockey 是一個 Atari 環境，建議先閱讀 [Atari Environments 總覽頁面](https://gym.openai.com/envs/#atari) 以了解通用資訊。

---

## 📦 Environment Summary

- **Environment Name**: `ALE/IceHockey-v5`
- **Game Time Limit**: 3 分鐘
- **Goal**: 在三分鐘內儘可能多地將冰球（puck）射入對方球門得分
- **特色**: 有 32 個射門角度，可利用場地側邊傳球來創造進攻機會

---

## 🎮 Action Space

- **Type**: `Discrete(18)`
- **說明**: 預設開啟所有 Atari 2600 可用的動作（即便使用 v0, v4 或 `full_action_space=False`）

---

## 👀 Observation Space

- **預設**: `Box((210, 160, 3), dtype=uint8)`（RGB 畫面）
- **其他可選擇觀察空間**:
  - **RAM**: `Box((128,), dtype=uint8)`
  - **灰階影像**: `Box((250, 160), dtype=uint8)`

---

## 🏅 Rewards

- 每次成功將冰球射入對方球門即可得分
- 無總得分上限，但遊戲時間為 3 分鐘

---

## ⚙️ Environment Configuration

- **Import 指令**:
  ```python
  import gym
  env = gym.make("ALE/IceHockey-v5")


# IceHockey (ALE/IceHockey-v5)

![Ice Hockey](../../../_images/ice_hockey.gif)

The IceHockey environment is part of the Atari environment suite. Please refer to the [general Atari environments guide](https://gym.openai.com/envs/#atari) for common information.

---

## 📦 Environment Summary

- **Environment ID**: `ALE/IceHockey-v5`
- **Game Duration**: 3 minutes
- **Objective**: Score as many goals as possible by shooting the puck into the opponent’s goal.
- **Key Feature**: 32 different shot angles toward the opponent’s goal; players can also pass the puck by bouncing it off the rink's sides.

---

## 🎮 Action Space

- **Type**: `Discrete(18)`
- **Description**: All possible Atari 2600 actions are enabled by default, regardless of version or `full_action_space` setting.

---

## 👀 Observation Space

- **Default**: `Box((210, 160, 3), dtype=uint8)` (RGB image)
- **Alternatives**:
  - **RAM**: `Box((128,), dtype=uint8)`
  - **Grayscale Image**: `Box((250, 160), dtype=uint8)`

---

## 🏅 Rewards

- Points are awarded by scoring goals.
- There is no upper limit to the score other than the 3-minute game timer.

---

## ⚙️ Environment Configuration

- **Import Example**:
  ```python
  import gym
  env = gym.make("ALE/IceHockey-v5")
