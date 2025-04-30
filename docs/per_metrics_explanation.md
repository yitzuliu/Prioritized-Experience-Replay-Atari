# Prioritized Experience Replay Metrics - Explanation

This document explains how to interpret the metrics visualized by `visualization.py` for Prioritized Experience Replay (PER), focusing on Beta values, Mean Priority, and TD Error.

*此文件解釋如何解讀由 `visualization.py` 生成的優先經驗回放 (PER) 指標圖表，特別是 Beta 值、平均優先級和 TD 誤差。*

---

## Beta Value (Beta 值)

### What is Beta?
- **Definition**: Beta is a parameter used for importance sampling to correct the sampling bias introduced by PER.
- **Range**: Increases from `BETA_START` (typically 0.4) to 1.0 over time.

### How to Interpret the Beta Chart
- **Rising Curve**: Beta should increase smoothly from its initial value to 1.0.
- **Ideal Trend**: Starts low during early training (focus on high-error samples), then gradually increases (reduce sampling bias).
- **Too Fast Convergence**: If Beta quickly approaches 1.0 early in training, consider increasing `BETA_FRAMES` or reducing `BETA_EXPONENT` to slow convergence.
- **Final Value**: Should approach or equal 1.0 by the end of training to ensure unbiased estimation.

### Key Factors
- `BETA_START`: Initial Beta value
- `BETA_FRAMES`: Total frames over which Beta increases to 1.0
- `BETA_EXPONENT`: Exponent controlling the shape of the Beta curve

---

## Mean Priority (平均優先級)

### What is Priority?
- **Definition**: The priority of each experience determines its probability of being sampled during replay.
- **Formula**: Priority = |TD Error|^α + ε, where α is the `ALPHA` parameter and ε is the `EPSILON_PER` constant.

### How to Interpret the Mean Priority Chart
- **Initial Phase**: Typically high at the start of training (large initial errors).
- **Mid-Training Trend**: Should gradually decrease and stabilize as learning progresses.
- **Spikes**: Sudden peaks indicate rare or difficult-to-learn experiences.
- **Long-Term Stability**: Persistent high values may indicate learning difficulties or the need for parameter adjustments.

### Key Factors
- `ALPHA`: Controls the non-linearity of priority, with higher values emphasizing high-error samples
- Training Stability: Stable training leads to lower mean priority over time

---

## Temporal Difference (TD) Error (時序差分誤差)

### What is TD Error?
- **Definition**: The difference between the observed reward plus discounted future estimate and the current estimate.
- **Formula**: δ = r + γ·max_a Q'(s', a) - Q(s, a), where r is the reward and γ is the discount factor.

### How to Interpret the TD Error Chart
- **High Initial Values**: Typically high at the start of training, indicating inaccurate estimates.
- **Gradual Decrease**: Should decrease over time as the Q-network learns better estimates.
- **Sudden Increases**: May indicate exploration of new areas or changes in environment dynamics.
- **Stable Level**: Should converge to a low and relatively stable level, indicating good Q-network learning.

### Key Factors
- `GAMMA`: Discount factor for future rewards
- `LEARNING_RATE`: Higher learning rates may cause larger TD error fluctuations
- Environment Stochasticity: Higher randomness in the environment leads to higher TD errors

---

## Relationships Between Metrics

- **Beta and Priority**: As Beta increases, the influence of high-priority samples is reduced, stabilizing learning.
- **Priority and TD Error**: Priority is directly derived from TD Error, so these curves often share similar shapes.
- **Overall Interpretation**: Ideally, Beta should increase while Priority and TD Error decrease, indicating effective learning.

---

## Adjustment Recommendations

If these metrics show abnormal trends, consider the following adjustments:

1. **Beta Converges Too Quickly**: Increase `BETA_FRAMES` or reduce `BETA_EXPONENT`.
2. **High Priority Persists**: Lower `LEARNING_RATE` or increase `BATCH_SIZE` to stabilize learning.
3. **TD Error Does Not Decrease**: Check `LEARNING_RATE`, `GAMMA`, or network architecture suitability.
4. **High Metric Volatility**: Increase `MEMORY_CAPACITY` to improve sample diversity.
