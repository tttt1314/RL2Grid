# 個人研究日誌（Tony）：RL2Grid

這份日誌記錄個人開發分支 `feat/tonys-work` 上的實驗進度、程式改動、觀察發現與待解問題，延續主日誌的結構與寫法約定，並與主線文件維持獨立以免多人協作或並行開發時互相覆寫衝突。

- **相關文件：** 主研究日誌見 [research_log.md](research_log.md)，基準重現表格見 [reproduction_bus14_seed0.md](reproduction_bus14_seed0.md)，理論推導與研究方向見 [research_directions_rl2grid.md](research_directions_rl2grid.md)，非同步收集說明見 [async_collect.md](async_collect.md)。
- **工作分支：** `feat/tonys-work`（自 `dev/fixed-async` @ `5c9413a` 分出）。

**寫法約定**
- **可獨立閱讀：** 不依賴未定義的外部代稱；所有演算法設定、約束定義、對照實驗與術語皆於首次提及時說明清楚，必要時再於括號補充程式參數或排程 job 名稱。
- **只追加不覆寫（Append-only）：** 新條目加在「時間軸」最後並標註紀錄時間（`YYYY-MM-DD HH:MM UTC+8`），舊條目不改。若後續實驗推翻或修正舊推論，就新增一條標明「更正對象」的新條目，並同步更新「目前結論」。
- **區分事前假設與事後發現：** 實驗前設計好的對照問題與預期判準，和看到結果後才回頭從數據或程式碼追查出的現象分開，後者明確標為 **post-hoc**；單一隨機種子（`seed 0`）的觀察皆標註為初步證據，待多 seed 驗證。

---

## 目前結論（2026-11-26 10:39 UTC+8）

目前本分支剛從 `dev/fixed-async`（commit `5c9413a`）分出，尚未執行新的個人實驗；以下為主線移交時的起點狀態摘要（詳見 [research_log.md](research_log.md)）：

1. **已完成的基準與 Batch 1（`bus14`、`seed 0`）：**
   - 原版基準（14 組）、診斷實驗（`E1 / E2 / E1B`）與 Batch 1（`E6a / E6b`）已全數完成（共 22 組）。
   - `E6a` 驗證了成本優勢標準化（`--norm-cost-adv`）、成本項悲觀 PPO 比率裁切（`--clip-cost-surr`）與 KL 早停（`--target-kl 0.02`）能徹底消除 `LagrPPO` 更新期 `approx_kl` 暴衝。
   - `E6b-C1` 在失敗約束上拿下目前最高存活率（最後 3 次平均 `0.5534`、峰值 `0.6703 @ 2.8M`），但 `E6b-C1` 與 `E6b-C2` 皆在 post-hoc 分析中發現 PID 積分飽和與統計量／吸收成本尺度偏差。
2. **分支起點 `5c9413a` 已包含的 E6c 實作（待上機驗證）：**
   - `alg/lagr_ppo/config.py` 與 `alg/lagr_ppo/core.py` 已加入 PID 防積分飽和夾限（`pid_i` 上限 `lag_max / lag_ki`）、雙變數暖機跳過更新選項 `--lag-warmup-iters`，以及無長情境跨輪取樣偏差的 `--dual-stat step_rate_1k`（每千步平均原始成本）。
   - `tools/queue/jobs_e6c.tsv` 已建立 4 組待跑實驗：`E6C_C1_PID_AW`、`E6C_C2_PID_ABS005`、`E6C_C2_PID_ABS010`、`E1_C1_LAM0_KEEPDISC`。

---

## 下一步候選

- **確認／執行 `E6c`（`tools/queue/jobs_e6c.tsv`）：** 檢驗 PID 防積分飽和、`--lag-warmup-iters 3`、`step_rate_1k`、縮小後過載吸收懲罰（`0.05` / `0.10`），以及關閉約束時保留斷線懲罰（`E1_C1_LAM0_KEEPDISC`）對存活率與雙變數動態的影響。
- **跨情境泛化對照（`tools/queue/jobs_e3_split.tsv`）：** 使用 70/15/15 互斥情境切分（`--chronics-split`）評估訓練集與未見測試集落差。
- **個人研究線記錄：** 後續於 `feat/tonys-work` 進行的程式修改、理論推導或實驗設計，直接追加至下方「時間軸」。

---

## 時間軸

### 2026-11-26 10:39 UTC+8 · 建立工作分支 `feat/tonys-work` 與個人研究日誌
- **分支起點：** 從 `dev/fixed-async`（commit `5c9413a`，`feat: add PID anti-windup, lag warmup, step_rate_1k dual stat, and E6c schedule`）建立新分支 `feat/tonys-work`。
- **建立目的：** 在 `docs/` 新增 `research_log_tony.md` 作為個人開發與實驗紀錄檔，沿用 [research_log.md](research_log.md) 的 append-only 與「區分事前假設／post-hoc 發現」格式規範，後續所有新進展皆追加於此時間軸之後。
- **起點程式碼清點（`5c9413a` 相較 `research_log.md` 最後一筆 `00:45 UTC+8` 紀錄之新增內容）：**
  - `alg/lagr_ppo/config.py`：`--dual-stat` 新增 `step_rate_1k` 選項；新增 `--lag-warmup-iters`（預設 `0`）。
  - `alg/lagr_ppo/core.py`：在非同步與同步收集路徑計算 `rollout_raw_cost_sum` 與 `rollout_steps` 以支援 `step_rate_1k = 1000.0 * rollout_raw_cost_sum / max(1.0, rollout_steps)`；PID 模式新增 `i_cap = args.lag_max / args.lag_ki`（當 `lag_ki > 0`）防止積分項無限累積；於 `iteration <= lag_warmup_iters` 期間略過 PID 與 Adam 雙變數更新。
  - `tools/queue/jobs_e6c.tsv`：排入 `E6C_C1_PID_AW`、`E6C_C2_PID_ABS005`、`E6C_C2_PID_ABS010` 與 `E1_C1_LAM0_KEEPDISC` 四組 300 萬步實驗（`bus14` 難度 0）。
