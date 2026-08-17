# Recall decomposition — local backbones (FC-SH 6k, has_pair N=74)

回應 [`docs/handoff/GX10_gemma_retrieval_check.md`](../../docs/handoff/GX10_gemma_retrieval_check.md)。
(1) 四種 memory-bank 配置的 `gt_new∈pool`(對應 paper `tab:retrieval_len`,local backbone 版);
(2) 對 faithful 配置,把 `gt_new∈top100` 拆成抽取 vs 檢索兩成因。

## (1) gt_new ∈ pool — 四種 bank 配置 × 8 local backbone

`python analysis/recall_audit_6k/pool_recall_4config_6k.py`

| 配置(bank 設置)| g3-1b | g3-4b | g3-12b | g3-27b | g2-9b | llama-8b | qwen-7b | mistral-7b |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **1 faithful**(ours/DontAsk/Vanilla, top-100)| 78 | 86 | 100 | 100 | 100 | 80 | 100 | 92 |
| **2 Mem0+FE**(faithful+destructive, top-100)| 0 | 0 | 76 | 77 | 22 | 0 | 22 | 18 |
| **3 Zep**(bi-temporal, top-10)| 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| **4 Mem0 Vanilla**(native, top-100)| — | 0 | 54 | 66 | — | — | — | — |

**四種配置裡檢索都不是瓶頸**,miss 三種來源:
- 配置 1 faithful:miss = **抽取 confound**(弱 backbone P1 沒抽到;retrieval@100|in-bank=100%,見 (2))。
- 配置 2/4 destructive:miss = **write-time 破壞性刪版**(弱端歸零 0%)——正是「write-time 不可逆」的證據。
- 配置 3 Zep:全 100%(held-fixed gpt-4o-mini graph、backbone-無關);Zep 低準確來自下游,非檢索。

## (2) faithful 配置的抽取 vs 檢索拆解

## 重現方式(零 API 成本,~7.5MB committed 資料)

```bash
conda activate "${CONDA_ENV:-repro}"
python analysis/recall_audit_6k/recall_decomposition_6k.py
```

讀取:各 backbone `extraction_cache_p1_6k.json`(bank,A 欄)+ `pools_slim_6k_no_p5.json`(top-100 slim,B 欄)
+ `sh_6k_mquake_analysis.json`(GT)+ `compute_pool_acc_crosstab.classify_pool_state`(matcher v4)。

## 結果

| backbone | cache facts | **A. gt_new∈bank**(抽取 recall)| **B. gt_new∈top100** | **C. retrieval@100 \| in-bank** |
|:--|:--:|:--:|:--:|:--:|
| gemma3-1b | 375 | 78% | 78% | **100%** |
| gemma3-4b | 381 | 86% | 86% | **100%** |
| gemma3-12b | 456 | 100% | 100% | 100% |
| gemma3-27b | 455 | 100% | 100% | 100% |
| gemma2-9b | 455 | 100% | 100% | 100% |
| llama3.1-8b | 338 | 80% | 80% | **100%** |
| qwen2.5-7b | 451 | 100% | 100% | 100% |
| mistral-7b | 452 | 92% | 92% | **100%** |

## 結論

**A 欄 == B 欄,C 欄全部 = 100%(八個 backbone 無一例外)。**

→ **檢索完全不是瓶頸**:只要 gt_new 在 bank 裡,top-100 一定 retrieve 得到(retrieval@100|in-bank = 100%)。所有 `gt_new∈top100` 的 miss **100% 都是抽取 miss**(gt_new 沒被 P1 抽進 bank),與 backbone extraction 品質相關(cache facts:llama 338→80%、gemma3-1b 375→78%),而非檢索器失效(embedder = text-embedding-3-small,與 backbone 無關)。

**論文含義**:此抽取 confound 由各 backbone 內**所有 method 共用同一 per-backbone bank** → 只影響**跨 backbone 絕對值**,不影響**同 backbone 內 method 比較**。

## (3) 全部 4 配置的 in-memory vs retrieval 拆解(檢索從不失敗)

`python analysis/recall_audit_6k/retrieval_vs_source_6k.py`

對每個配置把「gt_new∈memory(A)」與「gt_new∈pool(B)」拆開,算 `C = retrieval@K | in-memory`。各配置的 memory 來源不同:faithful = ours P1 extraction cache;dest/native = 破壞性更新後的 qdrant store facts(存於 `store_bank_6k_{dest,native}.json`)。

**結果:C = 100%,每個配置、每個 backbone(只要 gt_new 還在 memory 裡)無一例外。** 即**檢索器從不失敗**。gt_new 的 miss 只有兩種來源:
- **配置 1 faithful**:抽取沒抓到(弱 backbone P1 confound)。
- **配置 2/4 destructive**:write-time 破壞性刪除——store 從 455 塌到 **0–313**(gemma3-4b **0 筆**、gemma3-1b **5 筆**、llama **12 筆**),弱 backbone 上整個 store 幾乎全毀。
- **配置 3 Zep**:pool recall 全 100%,根本沒 miss。

→ 同時佐證兩個論點:**faithful 保留全版本的價值**、以及 **write-time commit 的不可逆傷害(弱 backbone 更致命)**。

## Bundle 內容(給跨機重現)

| 檔 | 用途 |
|:--|:--|
| `pools_slim_6k_no_p5.json`(7.5MB)| 8 backbone × 74 has_pair 的 top-100 slim:`retrieved_memories[{memory, ordinal}]` + `memories_str`。schema 與原始 dump 相容 |
| `recall_decomposition_6k.py` | 上表的 standalone 重現腳本 |
| `system_prompt.json` | 答題 system prompt(全域 1 份,原本每題重複)|
| `analysis/results/p1_caches__gemma3-{1b,4b,12b,27b}/extraction_cache_p1_6k.json` | A 欄的 bank(4 個 gemma3,本 commit 一併補上;其餘 4 個已在 repo)|

> 已丟棄(冗餘/可還原,見 handoff 討論):`resolved_pool`(文字被 memories_str 涵蓋、metadata 可由 triple_cache 還原)、per-query `system_prompt`(改全域 1 份)、`response`(= results.json output)、hash/score/created_at。
> 保留的 `ordinal` + `memories_str` 非 recall 表所需,而是為了順帶保住 freshness/serial 與 pool-collapse(gemma2 根因)分析。
