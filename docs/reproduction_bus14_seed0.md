# RL2Grid bus14 Reproduction：seed 0 結果與原論文對照

> 狀態（2026-11-20）：seed 0 的 14 個設定全部跑完（SAC_R 5M、LagrPPO C1/C2 各 3M 都已完成）。尚未執行的只有非 RL baseline（Idle、MILP）。
> 本文件只有 **1 個 seed**，原論文是 **10 個 runs 的平均**，所以下面的比較只能看趨勢，還不能當作 reproduction 的結論。

---

## 1. 原論文的結果（Marchesini et al., *RL2Grid*, arXiv 2503.23101）

### Table 2：拓撲動作（離散），difficulty 0，平均存活率

| Env | Diff. | Idle (I) | MILP | DQN V | DQN I | DQN R | PPO V | PPO I | PPO R | SAC V | SAC I | SAC R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bus14 | 0 | 0.18 | 0.06 | 0.07 | 0.86 | 0.74 | 0.74 | 0.99 | 0.97 | 0.17 | 0.56 | 0.16 |
| bus36-MO-v0 | 0 | 0.03 | 0.06 | 0.04 | 0.14 | 0.19 | 0.06 | 0.17 | 0.29 | 0.01 | 0.10 | 0.13 |
| bus118-MOB-v0 | 0 | 0.10 | 0.05 | 0.07 | 0.19 | 0.27 | 0.04 | 0.18 | 0.28 | 0.01 | 0.15 | 0.19 |

- 前兩欄 I、MILP 是非 RL 的 baseline：Idle 什麼都不做，MILP 是 Grid2Op 內建、以 DC 潮流近似最小化線路過載的最佳化 agent。
- RL 欄位中，**V** = vanilla；**I** = idle heuristic（電網安全時自動執行 do-nothing，只在危險時讓 agent 決策）；**R** = recovery heuristic（安全時自動重新連線並把 bus 還原成原始拓撲）。

### Table 3：Redispatch 動作（連續），平均存活率

| Env | Diff. | Idle (I) | PPO (V) | SAC (V) | TD3 (V) |
|---|---|---|---|---|---|
| bus14 | 0 | 0.00 | 0.17 | 0.01 | 0.06 |
| bus36-MO-v0 | 0 | 0.08 | 0.08 | 0.02 | 0.01 |
| bus118-MOB-v0 | 0 | 0.11 | 0.25 | 0.08 | 0.07 |

LagrPPO（約束版）在論文中只有訓練曲線（Figure 5 右、Figure G.3），沒有表格數值。

---

## 2. 本次 reproduction（bus14，seed 0）

數值 = **最後 3 次評估的平均存活率**（每次評估 10 個 episode，所以共 30 個 episode）。括號內是這 3 次評估的最小值到最大值，可以看出評估之間的波動。

### 拓撲動作（對照 Table 2 的 bus14 列）

| 設定 | 步數預算 | 論文 | **Seed 0** | 差距 | 判讀 |
|---|---|---|---|---|---|
| DQN V | 5M | 0.07 | **0.10** (0.08–0.13) | +0.03 | ✅ 一致 |
| DQN I | 500k | 0.86 | **0.97** (0.94–0.99) | +0.11 | ⬆ 比論文高 |
| DQN R | 500k | 0.74 | **0.68** (0.66–0.71) | −0.06 | ✅ 接近 |
| PPO V | 5M | 0.74 | **0.67** (0.50–0.76) | −0.07 | ✅ 接近（波動大） |
| PPO I | 500k | 0.99 | **0.90** (0.87–0.93) | −0.09 | ⬇ 略低 |
| PPO R | 500k | 0.97 | **0.80** (0.75–0.83) | −0.17 | ⬇ 偏低 |
| SAC V | 5M | 0.17 | **0.01** (0.01–0.02) | −0.16 | ⬇ 偏低（後期崩潰，見 §5） |
| SAC I | 500k | 0.56 | **0.85** (0.79–0.88) | +0.29 | ⬆ 明顯較高 |
| SAC R | 500k | 0.16 | **0.62** (0.59–0.66) | +0.46 | ⬆ 明顯較高 |
| Idle / MILP | — | 0.18 / 0.06 | 尚未執行 | — | 待補 |

### Redispatch 動作（對照 Table 3 的 bus14 列）

| 設定 | 步數預算 | 論文 | **Seed 0** | 差距 | 判讀 |
|---|---|---|---|---|---|
| PPO V | 5M | 0.17 | **0.16** (0.10–0.25) | −0.01 | ✅ 一致 |
| SAC V | 5M | 0.01 | **0.19** (0.15–0.21) | +0.18 | ⬆ 較高（訓練中最好 0.42 @ 4.6M） |
| TD3 V | 5M | 0.06 | **0.15** (0.14–0.17) | +0.09 | ⬆ 較高 |
| Idle | — | 0.00 | 尚未執行 | — | 待補 |

### 約束版（LagrPPO，topology）

| 設定 | 步數預算 | 論文 | **Seed 0 存活率** | 每 episode 平均 cost（最後 3 次） | 訓練中最好存活率 |
|---|---|---|---|---|---|
| LagrPPO LSI（C1，失敗約束，門檻 λ=50） | 3M | 只有曲線：學不起來、常超過門檻 | **0.32** (0.25–0.37) | 0.87（約 87% 的 episode 以失敗結束） | 0.51 @ 2.1M |
| LagrPPO TLO（C2，過載約束，門檻 λ=0） | 3M | 同上 | **0.50** (0.41–0.56) | 60.7（門檻 0，始終違反） | 0.79 @ 1.0M |

- 兩者都遠低於同為 topology vanilla 的 PPO V（0.67），和論文「約束版學不好」的描述一致。
- C2 的門檻是 0，只要有任何過載或斷線就算違反，所以拉格朗日乘數會一直上升；存活率在 0.07–0.79 之間大幅震盪，最後 300k 步的 cost 反而上升（45.8 → 78.4）。
- C1 的 cost 是每 episode 0 或 1，等於失敗率；它和存活率是同一件事的兩面，所以後期 cost 從 1.0 降到 0.8–0.9 時存活率也同步上升。

### 整體觀察

1. **論文的定性結論有重現**：heuristic（I、R）在三種演算法上都大幅優於 vanilla；拓撲 vanilla 只有 PPO 學得起來（約 0.7），DQN、SAC vanilla 都失敗。
2. **最接近的是 PPO V、DQN V、DQN R、PPO redispatch**：差距都在 ±0.07 以內。
3. **SAC heuristic 版和 off-policy 的 redispatch 明顯比論文好**。論文附錄 G 自己也說 SAC 和 DQN 在 bus14 的表現「低於預期」。可能的原因見 §5。
4. **PPO heuristic 版比論文低 0.09–0.17**。最可能的原因是學習率不同（見 §4.3）。

---

## 3. 訓練方式

### 3.1 程式碼版本
- Fork：`tttt1314/RL2Grid`，分支 `dev/fixed-async`（基於 upstream `emarche/RL2Grid`）。
- 合併了 upstream **PR #7** 的修正，其中最重要的是 **連續動作先 clip 到動作空間範圍內再送進環境**。原版沒有 clip，超出範圍的 redispatch 動作會被 grid2op 判定為 ambiguous，然後默默改成 do-nothing（我們量到約 99.8% 的動作是 ambiguous）。
- 自己加的**效能修改，不影響學習演算法本身**：
  - **非同步收集**（`--async-collect`）：worker 一準備好就送下一個動作，不等最慢的環境。PPO/LagrPPO 只在每個 rollout 結束時同步一次，資料仍然完全 on-policy；DQN/SAC/TD3 每次固定取 `n_envs` 筆 transition，所以訓練頻率等所有排程都不變。GAE 改成每個 worker 段落各自計算，已驗證在等長段落時和原版逐位元相同。
  - 拓撲動作表改成每個 worker 各自持有，取代原本的 `multiprocessing.Manager` 共享表（原本是單一 process 的 IPC 瓶頸）。
  - 平行評估（`--eval-workers 10`）、torch 單執行緒加上 OpenBLAS 8 執行緒、資料集預先解壓縮。
  - 可接續的 checkpoint（原子寫入加驗證）和排程 queue，用來在 GCP Spot VM 上分段執行。

### 3.2 預算與環境
- 環境：`bus14` = `l2rpn_case14_sandbox`，difficulty 0，LightSim2Grid backend，50 個平行環境，觀測值正規化（`--norm-obs True`）。
- **固定步數預算**：vanilla 和 redispatch 為 5M，heuristic（I、R）為 500k，LagrPPO 為 3M。
  - 論文的做法不同：論文是**每個 run 跑 48 小時的時間上限**，所以各演算法的總步數不一樣（附錄 G）。我們改用固定步數，讓結果和硬體速度無關。
  - heuristic 版步數少，是因為它的每一步都包含大量內部模擬，非常慢（約 50–150 SPS）。例如 seed 0 的 DQN I 在 225k–250k 步時存活率已經是 0.93–1.00，之後沒有明顯成長。
- 硬體：GCP 32 vCPU VM，同時跑 2 個 run。

### 3.3 超參數
使用 repo 的預設值（`alg/*/config.py`、`env/config.py`），只改了以下幾處：
- `--train-freq 50`（SAC）：預設的 20 不是 `n_envs=50` 的倍數，會觸發程式的檢查；50 也剛好是論文 Table F.1 的值。
- `--cost-threshold`（LagrPPO）：依論文 Table F.1 設定，LSI（C1）為 50、TLO（C2）為 0，取代 repo 預設的 15。
- `--total-timesteps`、`--eval-freq`、`--eval-episodes`：見上面的預算與 §4。

---

## 4. 評估方式

### 4.1 本次 reproduction
- **存活率（survival）**：一個 episode 中電網撐過的時間步數 ÷ 該情境的最大長度（約 8064 步，等於一個月）。1.0 代表整個月都沒有崩潰。
- **評估頻率**：vanilla/redispatch 每 50k 步、heuristic 每 25k 步、LagrPPO 每 100k 步評估一次。
- **每次評估**：**10 個完整 episode**，策略為**確定性動作**（`get_eval_action`）。評估環境使用訓練時的觀測正規化統計量（`obs_rms`）。原版預設是 3 個 episode；我們改成 10 個以降低雜訊，並用 10 個 process 平行執行（用 1 個 worker 時和原版循序評估的結果完全一致）。
- **評估情境**：評估環境和訓練環境來自**同一組 chronics**（RL2Grid 沒有獨立的測試集），每個評估 worker 使用自己的 seed（`1000 + i`），所以會抽到不同的情境順序。
- **表格數值**：每個設定取**最後 3 次評估的平均**（共 30 個 episode）。

### 4.2 原論文
- 「the average survival smoothed over 500 episodes of 10 runs per method」：也就是 10 個獨立 run，曲線經 500 個 episode 平滑；表格是「at convergence」的平均存活率，有 95% 信賴區間。
- bus36、bus118 只跑 5 個 runs。

### 4.3 和論文的主要差異（可能影響數值的因素）

| 項目 | 原論文 | 本次 | 可能的影響 |
|---|---|---|---|
| runs 數 | 10 | **1（seed 0）** | 最大的不確定來源；PPO V 單次評估就在 0.50–0.76 之間跳 |
| 訓練長度 | 48 小時時間上限 | 固定步數 | 論文的步數可能比我們多或少，視演算法而定 |
| 統計方式 | 500 episodes 平滑、收斂時的值 | 最後 3 次評估（30 episodes） | 我們的值雜訊較大 |
| 連續動作 clip | 原版沒有（PR #7 之前） | **有 clip** | 很可能是 SAC/TD3 redispatch 比論文高的主因 |
| PPO actor 學習率（拓撲） | Table F.1：3e-4 | repo 預設：**3e-5** | 可能是 PPO I/R 比論文低的原因 |
| DQN learning starts / train freq | Table F.1：20000 / 50 | repo 預設：1000 / 100 | 可能影響 DQN |
| LagrPPO 門檻 λ | Table F.1：TLO 0、LSI 50 | **已改成論文的值**：C1 `--cost-threshold 50`、C2 `--cost-threshold 0`（repo 預設 15） | 和論文一致 |
| 評估 episode 數 | 程式預設 3 | 10 | 雜訊較低 |

---

## 5. 個別結果說明

- **SAC V（拓撲）0.01，論文 0.17**：訓練中段約 1.5M 步時曾達到 0.22，之後逐漸崩潰到約 0.01。這看起來是訓練後期失穩，不是完全學不起來。需要看完整的學習曲線，以及其他 seed 是否也這樣。
- **SAC I/R、DQN I 比論文高**：論文附錄 G 承認 SAC、DQN 在 bus14 的表現「lower than expected」。我們的高數值可能來自 PR #7 的修正、評估方式不同，或只是 seed 0 運氣好，需要更多 seed 才能判斷。
- **off-policy redispatch（SAC、TD3）比論文高**：最可能的原因是 PR #7 的動作 clip。原版中幾乎所有連續動作都被當作 do-nothing，等於 agent 的動作根本沒有作用。
- **SAC V（redispatch）0.19，論文 0.01**：在 4.6M 步時曾到 0.42，最後回落到約 0.2，和 TD3（0.15）一樣比論文高，原因同上（動作 clip）。
- **LagrPPO**：非同步收集實測約 650 SPS，3M 步約 80 分鐘（同步版估計要 3 小時以上）。結果見 §2 約束版表格。
- **PPO I/R 比論文低**：除了學習率差異（3e-5 vs 3e-4），500k 的預算可能也不夠讓 PPO 收斂到 0.99。

---

## 6. 尚未完成 / 下一步

- [x] SAC_R 跑完 5M 步
- [x] LagrPPO C1、C2（非同步收集；門檻為論文的值：LSI 50、TLO 0）
- [ ] 非 RL baseline：Idle、MILP（Table 2、3 的前幾欄）
- [ ] 從完整 log 產生學習曲線，加上「訓練過程中的最好值」欄位
- [ ] 視結果決定是否跑 seed 1、2，以及是否用論文 Table F.1 的超參數重跑差距大的設定（PPO I/R、SAC V）

原始 log 在 VM 的 `~/queue/state/<job>_s0/seg_*.log`，checkpoint 在 `~/RL2Grid-fork/checkpoint/`。
