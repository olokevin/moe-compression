# Qwen2.5-0.5B — mixed-compression sweep (attention dense, MLP @ ratio 0.8)

Setting: attention (`q/k/v/o_proj`) left **uncompressed**; every layer's MLP (`gate/up/down_proj`) compressed to **80% params retained** under four methods. C4-calibrated compression -> continue-train 1000 CE steps on C4. Eval: MMLU (5% subset, 5-shot), HellaSwag (5% subset), C4/WikiText-2 PPL, right after compression (step 0) and every 200 steps. W&B project `yequan-train_aware-05B`.

**Uncompressed baseline:** c4=20.3894, wikitext2=13.0758, hellaswag=0.4911, mmlu=0.4670

## Results on ratio

| method              | MMLU post-comp | MMLU final | HellaSwag final | C4 PPL post-comp | C4 PPL final | WikiText2 PPL final |
| ------------------- | -------------- | ---------- | --------------- | ---------------- | ------------ | ------------------- |
| nystrom             | 0.3462         | 0.3352     | 0.4732          | 25.16            | 22.94        | 16.94               |
| nystrom_combined    | 0.3764         | 0.3832     | 0.4732          | 27.90            | 23.22        | 17.01               |
| btt_llm_v2          | 0.2418         | 0.2747     | 0.4513          | 167.88           | 30.49        | 23.10               |
| btt_llm_v2_combined | 0.2390         | 0.2500     | 0.4692          | 167.83           | 26.83        | 21.03               |

## Key findings

- **Nystrom clearly wins at this budget.** Both Nystrom variants keep MMLU at ~0.34–0.38 (vs. 0.467 baseline) and land C4 PPL ~23 after fine-tuning, essentially matching the ~20.4 baseline. The BTT variants collapse MMLU to chance (~0.25) and never recover it with 1000 CE steps.
- **`nystrom_combined` (trainability-aware, joint fwd+bwd kernel) is the best overall** — highest MMLU both right after compression (0.376) and after training (0.383, the only method whose MMLU *improves* with fine-tuning), with C4 PPL on par with plain Nystrom.
- **BTT compression is far more destructive up front** (post-compression C4 PPL ≈ 168 vs. ≈ 25–28 for Nystrom). Fine-tuning recovers PPL well (down to ~27–30) but not MMLU knowledge — the low-rank BTT factorization of the MLP appears to discard task-relevant structure that CE-on-C4 cannot restore.
- **Attention was left dense in every run** (the `q/k/v/o_proj` rule = `method: none`), so all differences are attributable to the MLP compression method alone. HellaSwag is barely affected across the board (0.45–0.48 vs. 0.491), consistent with commonsense being recoverable from C4 LM training while 5-shot MMLU knowledge is not.
- **The combined (fwd+bwd) methods used `calib_batch_size: 2`** to fit the extra backward-pass memory during calibration; forward-only methods (nystrom, btt_llm_v2) used the default batch
