# PER 模型開發待辦清單

## 已完成
- [x] 實作環境包裝器 (env_wrappers.py)
  - 實現了適合 Atari 遊戲的環境預處理
  - 優化了圖像處理和幀堆疊
  - 添加了訓練和評估模式的切換
- [x] 實作 SumTree 數據結構 (sumtree.py)
  - 用於優先經驗回放的高效存儲和採樣
  - 包含 `_propagate_priority_change`、`_find_priority_leaf_index`、`total_priority`、`add`、`update_priority`、`get_experience_by_priority`、`get_all_priorities` 等方法
- [x] 實作優先經驗回放記憶體 (per_memory.py)
  - 結合 SumTree 實現優先級採樣
  - 實現重要性採樣權重計算
  - 包含 `update_beta`、`_calculate_priority`、`add`、`sample`、`update_priorities`、`__len__` 等方法
- [x] 實作設備檢測工具 (device_utils.py)
  - 支援 CPU、GPU 和 Apple M 系列晶片
  - 優化不同設備上的性能
  - 包含 `get_device`、`get_gpu_memory_gb`、`get_system_info` 等方法
- [x] 實作 Q-Network 神經網絡 (q_network.py)
  - 實現卷積神經網絡架構
  - 支援不同數量的卷積層配置
  - 包含 `_get_conv_output_size`、`_initialize_weights`、`forward` 等方法
- [x] 實作 DQN 智能體 (dqn_agent.py)
  - 集成優先經驗回放
  - 實現 ε-greedy 策略和目標網絡更新
  - 包含 `select_action`、`store_transition`、`_calculate_td_error`、`optimize_model`、`save_model`、`load_model`、`set_evaluation_mode` 等方法
- [x] 實作日誌記錄工具 (logger.py)
  - 記錄訓練指標和進度
  - 實現記憶體優化的數據存儲
  - 包含 `log_episode_start`、`log_episode_end`、`log_step`、`log_per_update`、`log_epsilon`、`save_training_state`、`load_training_state`、`limit_memory_usage`、`get_training_data`、`get_training_summary`、`log_text` 等方法
- [x] 實作可視化工具 (visualization.py)
  - 提供各種訓練指標的可視化
  - 支持從 Logger 實例或保存的文件獲取數據
  - 包含 `plot_rewards`、`plot_losses`、`plot_epsilon`、`plot_per_metrics`、`plot_training_overview`、`plot_major_metrics`、`generate_all_plots` 等方法
- [x] 實作訓練腳本 (train.py)
  - 設置訓練循環
  - 保存模型和訓練數據
  - 添加評估和可視化功能
  - 已實現以下功能:
    - 初始化環境和智能體
    - 實現主訓練循環
    - 記錄訓練數據和指標
    - 定期保存模型檢查點
    - 定期進行評估
    - 處理訓練中斷和恢復
- [x] 實作恢復訓練腳本 (resume.py)
  - 從斷點恢復訓練
  - 加載先前的訓練狀態和參數
  - 已實現以下功能:
    - 加載保存的模型和訓練狀態
    - 恢復訓練計數器和參數
    - 繼續訓練過程
    - 保持一致的記錄和評估
    - 實現從檢查點恢復中斷的訓練
    - 添加模型權重加載機制
    - 添加訓練狀態恢復
    - 確保 epsilon 和 beta 參數正確恢復
    - 實現命令行參數解析以指定檢查點路徑
    - 添加恢復訓練的日誌記錄

## 潛在問題與解決方案
- **記憶體使用**: 優先經驗回放需要額外的記憶體來存儲優先級信息。config.py 中已將 MEMORY_CAPACITY 設置為 100,000，這應該對大多數系統是可行的，但可能需要進一步調整以平衡性能和記憶體使用。

- **計算效率**: SumTree 操作可能會成為瓶頸，需要高效實現。

- **超參數調整**: 優先經驗回放中的 alpha 和 beta 參數對性能有顯著影響，可能需要進行調整。

- **學習穩定性**: 使用PER可能會導致較高的方差，可能需要調整學習率或使用漸進式β更新。

- **保存大型模型**: 檢查點可能會變得很大，尤其是當訓練了很長時間後，需要確保有足夠的磁盤空間。

- **日誌數據存儲**: 長時間訓練可能產生大量日誌數據，已通過 `logger.py` 中的批量寫入和記憶體管理優化，但仍需監控磁盤空間使用。

## 下一步
1. 進行超參數調整
2. 進行完整的訓練和評估實驗

## 細節說明

### src/sumtree.py 功能
- `__init__`: 初始化SumTree數據結構，設置內存容量和優先級樹
- `_propagate_priority_change`: 將優先級變化向上傳播到樹中
- `_find_priority_leaf_index`: 查找包含與值對應的優先級的葉索引
- `total_priority`: 獲取存儲在樹中的總優先級
- `add`: 添加一個新的經驗及其優先級
- `update_priority`: 更新葉節點的優先級
- `get_experience_by_priority`: 基於優先級值獲取經驗
- `get_all_priorities`: 從葉節點獲取所有優先級

### src/per_memory.py 功能
- `__init__`: 初始化PER記憶體，設置參數如alpha、beta等
- `update_beta`: 根據當前訓練進度更新beta參數，用於重要性採樣權重的計算
- `_calculate_priority`: 根據TD誤差計算優先級，使用alpha參數控制優先級的強調程度
- `add`: 添加新的經驗轉換到記憶體，並計算初始優先級
- `sample`: 基於優先級從SumTree中採樣批次的經驗轉換，並計算重要性採樣權重
- `update_priorities`: 根據新計算的TD誤差更新經驗樣本的優先級

### src/sumtree.py 功能
- `_propagate_priority_change`: 將葉節點優先級變化向上傳播到樹根，確保總優先級正確
- `_find_priority_leaf_index`: 使用二分搜索找到優先級值對應的葉節點索引，用於採樣
- `total_priority`: 獲取所有經驗的總優先級，即樹根節點的值
- `add`: 添加新經驗及其優先級到樹中，處理循環緩衝區行為
- `update_priority`: 更新特定葉節點的優先級並傳播變化
- `get_experience_by_priority`: 基於優先級值獲取經驗樣本，用於優先級採樣
- `get_all_priorities`: 獲取所有葉節點的優先級，用於分析和可視化

### src/dqn_agent.py 功能
- `__init__`: 初始化DQN智能體，包括策略網絡和目標網絡，以及經驗回放記憶體
- `select_action`: 使用ε-貪婪策略選擇動作，平衡探索與利用
- `store_transition`: 存儲經驗轉換(狀態、動作、獎勵、下一狀態)到回放記憶體
- `_calculate_td_error`: 計算時序差分(TD)誤差，用於優先級更新
- `optimize_model`: 執行一步Q網絡優化，包括計算損失及更新優先級
- `save_model` & `load_model`: 保存和加載模型權重和訓練狀態
- `set_evaluation_mode`: 設置智能體的評估模式，禁用探索

### src/q_network.py 功能
- `__init__`: 初始化Q網絡架構，設置卷積層和全連接層
- `_get_conv_output_size`: 計算卷積層輸出後扁平化特徵的大小，用於確定全連接層的輸入維度
- `_initialize_weights`: 使用He初始化方法初始化網絡權重，以改善訓練收斂性
- `forward`: 執行網絡的前向傳播，處理卷積和全連接層的計算

### src/device_utils.py 功能
- `get_device`: 檢測並返回最佳可用計算設備(CPU/CUDA/MPS)，優先使用GPU
- `get_gpu_memory_gb`: 獲取GPU總內存(GB)，用於根據可用內存調整批次大小
- `get_system_info`: 獲取詳細系統信息，用於日誌記錄和排錯

### src/env_wrappers.py 功能
- 環境包裝器:
  - `NoopResetEnv`: 在重置時執行隨機數量的無操作，增加初始狀態的多樣性
  - `MaxAndSkipEnv`: 僅返回每n幀並對最後2幀做最大值處理，處理閃爍的問題
  - `ResizeFrame`: 將觀察幀調整為指定尺寸，降低計算需求
  - `NormalizedFrame`: 將像素值標準化到[0,1]範圍，穩定訓練
  - `FrameStack`: 將n個幀堆疊在一起，提供時間信息
  - `ClipRewardEnv`: 將獎勵裁剪到{+1,0,-1}，改善訓練穩定性
- 主要函數:
  - `make_atari_env`: 創建並配置Atari環境，應用所有必要的包裝器

### src/logger.py 功能
- `__init__`: 初始化日誌記錄器，設置文件路徑和數據結構
- `log_episode_start`: 記錄新一集的開始
- `log_episode_end`: 記錄一集的結束和總體結果
- `log_step`: 記錄訓練步驟的數據
- `log_per_update`: 記錄PER更新的數據
- `log_epsilon`: 記錄epsilon值變化
- `save_training_state`: 保存訓練狀態到文件
- `load_training_state`: 從文件加載訓練狀態
- `limit_memory_usage`: 限制記憶體使用，通過淘汰舊數據
- `get_training_data`: 獲取所有訓練數據
- `get_training_summary`: 獲取訓練的摘要統計
- `log_text`: 記錄文字日誌信息

### src/visualization.py 功能
- `__init__`: 初始化可視化工具，設置數據源和保存路徑
- `plot_rewards`: 繪製獎勵曲線
- `plot_losses`: 繪製損失曲線
- `plot_epsilon`: 繪製epsilon探索率曲線
- `plot_per_metrics`: 繪製PER相關指標
- `plot_training_overview`: 繪製訓練總覽圖
- `plot_major_metrics`: 繪製主要指標
- `generate_all_plots`: 生成所有圖表並保存