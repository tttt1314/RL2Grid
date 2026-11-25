# 研究日誌：RL2Grid（bus14）

這份日誌集中記錄 RL2Grid 專案（`bus14` reproduction 與後續 Safe RL／電網控制研究）的進度與實驗結論。完整基準數字與對照見 [reproduction_bus14_seed0.md](reproduction_bus14_seed0.md)，公式推導與研究提案見 [research_directions_rl2grid.md](research_directions_rl2grid.md)，程式碼在 `tttt1314/RL2Grid`（分支 `dev/fixed-async`），VM 原始訓練 log 封存在本機 `final_proj/vm_logs/`。

**寫法約定**
- 新的條目加在「時間軸」最後，並標明紀錄時間（`YYYY-MM-DD HH:MM UTC+8`），舊條目不改。若後續實驗推翻或修正舊推論，就新增一條含「更正對象」的新條目，並同步更新「目前結論」。
- 事前假設／預期判準與看到結果後才歸納的發現要分開，後者標為 **post-hoc**；單一 seed（`seed 0`）的觀察皆標註為初步證據，待多 seed 驗證。
- 本檔與 `RL2Grid-async/docs/research_log.md` 維持逐位元一致，修改時兩邊同步並推上 `dev/fixed-async`。

---

# 研究線一：RL2Grid bus14 基準重現與 Safe RL（LagrPPO）失效機制定位

## 研究問題

1. 在 GCP VM（`g2op-train`，`c2d-highcpu-32`）上重現 Marchesini et al.（*RL2Grid*, arXiv 2503.23101v2）的 `bus14`（`l2rpn_case14_sandbox`，difficulty 0）全部 RL 基準。
2. 定位為何論文與重現實驗中約束強化學習 `LagrPPO`（C1/LSI 失敗約束與 C2/TLO 過載約束）的存活率遠低於無約束 `PPO`，並據此設計合理且穩定的電網 CMDP 與 Lagrangian 最佳化方法（RD1＋RD2）。

- **主要評估指標：**
  - 存活率 $\text{survival} = T_f / T_{\max} \in [0, 1]$（`bus14` 最大步數約 $8064$），取**最後 3 次確定性評估的平均**（每次 10 個 episode，共 30 個 episode）。
  - 約束指標：`cost_return`（每次評估每 episode 平均累積 cost），以及每輪 `LagrRollout` 診斷量（`J_c`、更新前後 $\lambda$ 與 $\nu = \text{softplus}(\lambda)$、已完成 episode 平均長度與 cost、`kl`、`clipfrac`、`pg_r`、`surr_c`、`ent`）。

## 目前結論（2026-11-25 19:47 UTC+8 更新）

**已完成 `bus14` seed 0 全 14 組基準重現（14/14 DONE），並透過 E1／E2／E1B 三組診斷實驗（4/4 DONE，累計 18/18 DONE）精準定位 `LagrPPO C1` 低分主因：問題主要不是 O6 猜測的超參數過激，而是 O5（未做 PPO clip 與標準化的 cost surrogate 在 $\nu \approx 1.35\text{–}1.70$ 下摧毀 trust region，首輪 `kl` 達 `0.707`、E1 關掉 $\nu$ 後為 `0.049`）與 O2／O4（前期高失敗率推升 $\lambda$ 後，因 Adam 動量與僅 30 輪更新使 $\lambda$ 在後期 `J_c < 50` 時仍卡在 `1.50` 高點）的共同作用。把 $\nu$ 鎖近 0（`E1`）後，完全沿用 LagrPPO 超參數的存活率即從 `0.320 / 0.355` 跳升至 `0.579`（最好 `0.643`）。**（註：以上皆為 seed 0 單次結果，待多 seed 驗證。）

已驗證或排除的可能原因如下：

| 可能原因 | 檢驗 | 結果 |
|---|---|---|
| **O6：** C1 低分單純來自 LagrPPO 超參數（`actor_lr=3e-4`, `epochs=80`, `n_steps=2000`）太激進 | **E1**（同 C1 超參數但鎖死 $\nu \approx 0$）＋ **E1B**（ $\nu \approx 0$＋PPO 超參數） | **推翻為主因。** `E1` 達 **`0.5786`**（最好 `0.6428`），大幅勝過同超參數開約束的 `E2_C1`（`0.3552`）與原 `C1`（`0.3197`）；反倒是 3M 步用 PPO 小學習率退火到 0 的 `E1B` 僅 `0.2498`。 |
| **O5：** cost advantage 未標準化、未 clip，高 $\nu$ 時破壞 PPO trust region | **E1 vs. E2_C1 / E2_C2** 的 `LagrRollout` 逐輪 `kl` 與 `clipfrac` | **強烈支持。** `E1` 第 1 輪 `kl = 0.04863`、第 3 輪起穩於 `0.008–0.022`；`E2_C1`（ $\nu = 1.35 \to 1.70$）第 1 輪 `kl = 0.70715`（14.5×）、`clipfrac = 0.5709`、第 20 輪仍突跳 `0.106`；`E2_C2` 第 1 輪 `kl = 5.52301`，中期 `kl = 0.88–1.26` 直接伴隨存活率崩跌。 |
| **O2（原推論）：** C1 全程 `J_c < 50` 使 $\lambda$ 一路下降至 0 | 讀取舊 `final_LAGRPPO_C1.tar`＋重跑 **E2_C1** 30 輪軌跡 | **原推論部分錯誤，實際機制更嚴重。** 前 14 輪（`100k–1.4M`）`ep_len = 6.5–180.2` 使 `J_c = 1216 → 60 > 50`，把 $\lambda$ 拉到 `1.418`；第 15–30 輪 `J_c` 降至 `48 → 14–26 < 50`，但 Adam 累積動量仍把 $\lambda$ 繼續推高至第 28–29 輪 **`1.504`**（ $\nu = 1.704$），第 30 輪才微降至 `1.503`（原基準 checkpoint 亦為 `1.4806`）。 |
| **O4（原推論）：** C2 的 Adam 固定每輪線性上升 `0.05` 到 `~2.5` | 讀取舊 `final_LAGRPPO_C2.tar`＋重跑 **E2_C2** 30 輪軌跡 | **方向正確、斜率推論修正。** `J_c` 全程為正（`31279 → 53`）且 $\lambda$ 單調上升，但初期 `J_c = 31279` 把 Adam 的 `exp_avg_sq` 撐得極大（`919093`），使 30 輪結束時 $\lambda$ 僅升至 `1.355`（舊檔 `1.3626`），並非等差 `+0.05/step`；末期仍出現第 28–30 輪 `eval_cost` 暴衝至 `55.9 / 106.3 / 119.5`。 |
| **PR #7：** 連續動作未 clip 導致論文 off-policy redispatch 失效 | seed 0 `SAC_R` / `TD3_R` 與論文 Table 3 對照 | **支持。** 加上 clip 後 seed 0 `SAC_R = 0.19`、`TD3_R = 0.15`（論文原版僅 `0.01 / 0.06`）。 |
| **Post-hoc 程式碼新發現：** `OverloadReward(constrained=True)` 移除斷線懲罰 | 比對 `env/utils.py` 與 `env/reward.py` | **已於程式碼確認。** 只要 `constraints_type != 0`，`OverloadReward` 即將 `self.penalty` 設為 `0.0`（無約束 `PPO` 為 `-1.0`），導致 C1 在 reward 與 cost 兩側都完全沒有斷線懲罰。 |

**目前最強的證據（2026-11-25 19:47 UTC+8）：**
1. **E1 vs. E2_C1 同設定單因子對照：** 僅把 $\nu$ 從動態更新（`1.35 → 1.70`）鎖死為 $\text{softplus}(-20) \approx 2\times 10^{-9}$，3M 步存活率由 `0.3552`（原基準 `0.3197`）升至 **`0.5786`**（最好 `0.6428 @ 2.8M`），首輪 `kl` 由 `0.70715` 降至 `0.04863`。
2. **E2_C1 與舊 C1 checkpoint 雙重交叉驗證：** 兩次獨立跑完 3M 步的最終 `lag_mul` 分別為 `1.503` 與 `1.4806`（初始 `1.0`），共同證實「早期失敗推升 $\lambda$ ＋ 30 輪與 Adam 慣性導致後期無法退場」。
3. **E2_C2 揭示過載約束的震盪與末期違規：** 雖在 `2.3M` 步曾達 `0.8122`，但第 15–17 輪出現 `kl = 0.69–1.26` 的劇烈退化（`eval_surv` 跌至 `0.105`），最後 3 次 `eval_cost` 飆升至平均 **`93.90`**（遠高於前 200 萬步的 `23–46`）。

## 尚未解決的問題

1. **`E1`（`0.5786`）與同在 3M 步的 `PPO_T_V`（`0.7104`）之間剩餘的 `0.13` 落差來源：** 多少來自 `OverloadReward(constrained=True)` 拿掉斷線懲罰（`env/reward.py:32`），多少來自 `n_steps`（2000 vs. 400）、網路大小與 `anneal_lr` 總步數（3M vs. 5M）。
2. **多 seed 變異與泛化能力（O8／RD7）：** 目前所有數據僅來自 `seed 0`，且原版訓練與評估共用同一池 1,004 個 chronics，既有 checkpoint 亦未保存 `obs_rms`。
3. **尚未補齊的非 RL baseline 與學習曲線：** `Idle` 與 `MILP`（論文 Table 2/3）尚未執行；seed 0 全 14 組的學習曲線圖尚未從 `final_proj/vm_logs/` 繪出。
4. **RD2＋RD1 能否同時壓低過載與維持高存活：** `cost_advantage` 標準化＋clip＋`--target-kl`＋PID Lagrangian＋吸收集態 cost（RD1）實作後，在 C1／C2 上是否能同時消除 `kl` 爆炸與末期 `eval_cost` 飆升。

## 下一步候選

- **A. 先修復 O5 再接 RD1＋RD2（新主線）：** 在 `LagrPPO` 先實作「cost advantage 標準化 + PPO clip + `--target-kl`」的最小消融組，接著加上 PID Lagrangian（RD2）與合理化的 CMDP 定義（RD1：cost rate＋失敗吸收 cost＋獨立 $\gamma_c$＋episode 級 $\hat{J}_C$）。
- **B. 同步研究文件與產出 seed 0 學習曲線：** 把 E1／E2／E1B 結果與對 O2／O4／O5／O6 的更正同步回 [research_directions_rl2grid.md](research_directions_rl2grid.md)，並以 `final_proj/vm_logs/` 畫出各演算法訓練曲線。
- **C. 補齊 E3（`--chronics-split` 切分 train/test＋存 `obs_rms`）與 Idle／MILP baseline，並依資源規劃擴展多 seed。**

---

## 時間軸

### 2026-11-18 16:49 UTC+8 · 基礎設施：全演算法非同步收集、平行評估與 Spot VM 可接續 Queue
- **程式碼與文件：** `common/async_collector.py`、`env/eval.py`、`tools/queue/run_queue.py`、`tools/setup_vm.sh`、`docs/async_collect.md`（commits `89fe6bc`～`1e77f13`）
- **背景：** 原版 `gym.vector.AsyncVectorEnv` 每步等待最慢的 worker，開啟 heuristic 時 51 個程序僅約 2 個在跑（~20 SPS）；評估預設 3 個 episode 波動達 29%–97% 且單執行緒評估拖慢訓練。
- **做法：** 實作無 barrier 的 `AsyncCollector` 與分段 `compute_segment_gae`（覆蓋 PPO／DQN／SAC／TD3，等長段落與同步版誤差 `0.0`），移除拓撲動作表的 `Manager` IPC 瓶頸，加入 `--eval-workers 10`／`--eval-episodes 10` 平行評估、OpenBLAS 8 執行緒上限、資料集預解壓縮與 CRC 驗證之分段 checkpoint queue。
- **實測成效：** `PPO + Idle` 單跑由 18–21 SPS 升至 70–82 SPS（~4×）；`DQN + Idle` 由 16 升至 46 SPS、`DQN` 原始版由 418 升至 889 SPS。

### 2026-11-19 23:37 UTC+8 · LagrPPO 非同步化與門檻對齊論文 Table F.1
- **程式碼與設定：** `alg/lagr_ppo/core.py`、`tools/queue/jobs_bus14.tsv`（commits `dbfee7c`、`02c12b9`、`e4bc007`）
- **做法：** 為 `LagrPPO` 補上 `--async-collect` 與帶 cost critic 的 `compute_segment_gae`，並將 `jobs_bus14.tsv` 的約束門檻由 repo 預設的 `15` 依論文 Table F.1 改為 **LSI（C1）= `50`**、**TLO（C2）= `0`**。
- **驗證：** 單元測試確認等長段落下與同步版逐位元一致，VM 上 4096 步 smoke test 正常完成並寫出 checkpoint；在 32 vCPU VM 上 3M 步實測維持約 650–730 SPS（約 70–80 分鐘完成）。

### 2026-11-21 00:40 UTC+8 · bus14 seed 0 全 14 組基準完成與論文對照
- **報告：** [reproduction_bus14_seed0.md](reproduction_bus14_seed0.md)（commit `bd752fe`）
- **結果摘要（最後 3 次評估共 30 episodes 平均存活率）：**
  - 拓撲：`DQN V/I/R = 0.10 / 0.97 / 0.68`（論文 `0.07 / 0.86 / 0.74`）；`PPO V/I/R = 0.67 / 0.90 / 0.80`（論文 `0.74 / 0.99 / 0.97`）；`SAC V/I/R = 0.01 / 0.85 / 0.62`（論文 `0.17 / 0.56 / 0.16`，其中 `SAC V` 于 `1.5M` 曾達 `0.22` 後崩潰）。
  - Redispatch：`PPO/SAC/TD3 = 0.16 / 0.19 / 0.15`（論文 `0.17 / 0.01 / 0.06`，`SAC` 最好 `0.42 @ 4.6M`）。
  - 約束版（3M）：`LAGRPPO_C1 = 0.32`（`0.25–0.37`，最好 `0.51 @ 2.1M`，cost `0.87`）、`LAGRPPO_C2 = 0.50`（`0.41–0.56`，最好 `0.79 @ 1.0M`，cost `60.7`）。
- **結論：** 定性趨勢完全重現（heuristic 大幅勝過 vanilla；約束版大幅落後無約束 PPO）；off-policy redispatch 顯著高於論文主因歸於已併入的 PR #7 連續動作 clip。

### 2026-11-24 16:08 UTC+8 · 問題分析（O1–O8）與研究方向提案（RD1–RD8）
- **報告：** [research_directions_rl2grid.md](research_directions_rl2grid.md)（commit `ceab9f5`）
- **核心假設與提案：**
  - 整理從程式碼比對出之 8 項問題（O1 論文前後門檻矛盾且 `ConstrainedFailureGridOp` 以失敗當步計 cost=1、O2 雙邊統計量尺度與視野錯位、O3 折扣誘導提早失敗躲避過載、O4 Adam 固定步長更新 $\lambda$、O5 cost surrogate 無 clip 且未除以 $\sigma$、O6 C1 與 PPO 超參數混淆、O7 heuristic n 步轉移未用 $\gamma^{n+1}$ bootstrap、O8 中後期退化與同 pool 評估）。
  - 擬定 RD1–RD8，選定 **RD1（Well-posed Grid CMDP）＋ RD2（PID Lagrangian + clipped cost surrogate）** 為主線，並列出可低成本檢驗假設的第一批診斷實驗 **E1–E3**。

### 2026-11-25 00:19 UTC+8 · E1 / E2 / E1B 診斷實作、舊 checkpoint 稽核與排程開跑
- **程式碼與排程：** `alg/lagr_ppo/core.py`、`main.py`、`tools/queue/run_queue.py`、`tools/queue/jobs_e1_e2.tsv`（commits `1bb8bc5`、`a4765b1`）
- **既有 seed 0 checkpoint 事前稽核（未重跑前直接讀出）：**
  - `final_LAGRPPO_bus14_T_0_0__I_C1_1795173477_37035.tar`（`step=3M`, `rollout=30`）：`lag_mul = 1.4806`（ $\nu = 1.6856$），`lag_optim`: `exp_avg = 10.3308`, `exp_avg_sq = 1988.3940`。**已先打破「C1 的 $\lambda$ 應該比初始 `1.0` 低」的預期。**
  - `final_LAGRPPO_bus14_T_0_0__I_C2_1795174786_36150.tar`（`step=3M`, `rollout=30`）：`lag_mul = 1.3626`（ $\nu = 1.5905$），`lag_optim`: `exp_avg = -373.0327`, `exp_avg_sq = 919093.3125`。
- **實驗設計：**
  - **E1（`E1_LAGRPPO_C1_LAM0`）：** C1 預設超參數但固定 `--lag-lr 0 --lag-mul -20`（ $\nu \approx 2.06\times 10^{-9}$），驗證拿掉約束後是否回復高分。
  - **E2（`E2_LAGRPPO_C1`, `E2_LAGRPPO_C2`）：** 在 `LagrRollout` 記錄每輪 `J_c`、 $\lambda$、 $\nu$、訓練 episode 平均長度/cost、`kl`、`clipfrac`、`pg_r`、`surr_c`、`ent` 重跑 3M 步。
  - **E1B（`E1B_LAGRPPO_C1_PPOCFG`）：** 利用第 2 波空閒 2 slots 加測「 $\nu \approx 0$＋PPO 超參數／網路大小（`actor_lr=3e-5`, `epochs=40`, `n_steps=400`, `[256,128,64]` / `[512,256,128]`）」，3M 步。
- **VM 執行紀錄：** `00:16:40 UTC+8` 開機開跑、兩波 4 個 job 于 `02:55:41 UTC+8` 全數完成（~2h39m），`watch_shutdown.sh` 自動關機（`02:56:58 UTC+8` `guestTerminate`）。

### 2026-11-25 19:08 UTC+8 · E1 / E2 / E1B 結果分析與 O2 / O4 / O5 / O6 更正
- **狀態與原始 log：** 4/4 新實驗皆為 `DONE`（`step=3000000`），連同基準 14 組共 `18/18 DONE, 0 FAILED`，所有 `seg_*.log` 已存本機 `final_proj/vm_logs/`，VM 已確認回到 `TERMINATED`。
- **更正對象：** 2026-11-24 [research_directions_rl2grid.md](research_directions_rl2grid.md) §3 中對 **O2**（「C1 幾乎全程 `J_c < 50` 讓 $\lambda$ 往下掉」）、**O4**（「C2 的 $\lambda$ 等差 `+0.05/step` 線性升到 `2.5`」）與 **O6**（「C1 低分主要來自超參數」）的事前推論。
- **量化結果（seed 0，3M 步，最後 3 次 eval 平均）：**
  - `E1_LAGRPPO_C1_LAM0_s0`：**`0.5786`**（`0.54–0.64`，最好 `0.6428 @ 2.8M`，`cost3=0.70`，`SPS=654`）
  - `E2_LAGRPPO_C1_s0`：**`0.3552`**（`0.30–0.46`，最好 `0.4612 @ 2.9M`，`cost3=0.93`，`SPS=676`）
  - `E2_LAGRPPO_C2_s0`：**`0.5592`**（`0.49–0.62`，最好 `0.8122 @ 2.3M`，`cost3=93.90`，`SPS=711`）
  - `E1B_LAGRPPO_C1_PPOCFG_s0`：**`0.2498`**（`0.19–0.30`，最好 `0.4386 @ 2.0M`，`cost3=0.97`，`SPS=824`）
  - 基準截斷對照：`PPO_T_V_s0 @ 3M` 為 **`0.7104`**（`0.68–0.74`，最好 `0.8421 @ 2.66M`；5M 完整版 `0.6664`）。
- **關鍵發現（多數為對軌跡的 post-hoc 解析）：**
  1. **O5 確定為 C1 失敗主因，O6 非主因：** `E1` 不改超參數僅鎖 $\nu \approx 0$ 就從 `0.355` 跳升至 `0.579`；軌跡顯示 `E2_C1` 第 1–4 輪 `kl` 高達 `0.707 → 0.388 → 0.083 → 0.181`（第 1 輪 `clipfrac = 0.571`），第 20 輪仍突跳 `0.106`，反觀 `E1` 全程 `kl <= 0.049`（iter 3+ 僅 `0.008–0.022`）。
  2. **O2 實測軌跡為「前衝後鎖」而非「全程下降」：** `E2_C1` 前 14 輪（`100k–1.4M`）因失敗頻繁（`ep_len = 6.5 → 180.2`），`J_c = 1216 → 60 > 50` 推高 $\lambda$ 至 `1.418`（ $\nu = 1.635$）；後 16 輪雖 `J_c = 48 → 14–26 < 50`，受 Adam 動量與僅 30 輪更新拖累， $\lambda$ 仍上升到第 28–29 輪 **`1.504`**（ $\nu = 1.704$）才剛反轉（第 30 輪 `1.503`）。
  3. **O4 與 C2 震盪細節：** `E2_C2` 第 1 輪 `kl = 5.523`、`J_c = 31279`，巨幅梯度撐大 `exp_avg_sq`，使 $\lambda$ 僅緩慢升至 `1.355`（非 `+0.05/step`）；第 15–17 輪出現 `kl = 0.88 / 1.26 / 0.69` 崩跌（`eval_surv` 掉至 `0.105`），第 28–30 輪 `eval_cost` 更飆至 `55.9 / 106.3 / 119.5`。
  4. **Post-hoc 解析 `E1B` 偏低與 `E1` vs. `PPO_T_V` 差距：** `E1B` 在 3M 步將 `actor_lr=3e-5` 線性退火至 0（`PPO_T_V` 為 5M 退火），到 `2.6M–3.0M` 時 `kl` 已縮至 `0.0018 → 0.00001` 而提前停止學習；此外查 `env/utils.py` 與 `env/reward.py` 發現，`OverloadReward(constrained=True)` 會將斷線懲罰設為 `0.0`（無約束 PPO 為 `-1.0`），形成 `constraints_type=1` 的隱藏環境差異。
- **路線影響：** 後續進入 **RD2＋RD1** 實作時，優先單獨驗證「cost advantage 標準化＋PPO clip＋`--target-kl`」（針對 O5）的改善幅度，再疊加 PID Lagrangian 與修正後的電網 CMDP 定義。

### 2026-11-25 19:55 UTC+8 · 學習曲線繪製與研究方向文件同步更新
- **繪圖腳本與圖表：** 新增 `tools/plot_curves.py`，由本機 `final_proj/vm_logs/state/` 產出兩張 2×2 圖表並同步至 `docs/figures/` 與 `final_proj/figures/`：
  - [bus14_seed0_learning_curves.png](figures/bus14_seed0_learning_curves.png)：seed 0 全 14 組基準之評估存活率曲線（含 3-eval 移動平均與論文 Table 2/3 對照線）。
  - [e1_e2_diagnostic_curves.png](figures/e1_e2_diagnostic_curves.png)：E1／E2／E1B 存活率對照、E2-C1「前衝後鎖」dual 動態、四組 `approx_kl` 對比，以及 E2-C2 中期崩跌與末期 `eval_cost` 飆升。
- **文件同步更新與峰值更正：**
  - 更新 [research_directions_rl2grid.md](research_directions_rl2grid.md) §2（補齊全 14 組 `訓練中最好` 並嵌入基準學習曲線）、§3（補上 `OverloadReward(constrained=True)` 程式碼發現、O2／O4／O5／O6 的 E1／E2／E1B 實證更正）、§4–§6（標記 E1／E1B／E2 已完成、嵌入診斷圖、拆出下一步 **E6a** O5 消融與 **E6b** RD1＋RD2）。
  - 更新 [reproduction_bus14_seed0.md](reproduction_bus14_seed0.md) §2／§5／§6，補齊「訓練中最好」欄位並嵌入學習曲線；**更正對象：** 原註記「`SAC V` 拓撲最好為 `0.22 @ 1.5M`」，對照完整 `SAC_T_V_s0` log 後實際峰值为 **`0.3546 @ 250k`**（1.5M 步時為 `0.2517`，之後逐步崩潰至 `0.01`）。
