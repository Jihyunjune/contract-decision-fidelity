# Historical v2 results: public release summary

This file summarizes the completed study. Release preparation replays evaluation and checks interfaces; it does not claim a new independent experiment or regenerate 6,800 predictions.

## Experimental population

100 sealed ContractNLI Train documents, 17 hypotheses each, four methods. The original official Test set was already observed. Of 423 Train documents, 323 were allocated to the development pool and a predetermined 24 were actually inferred during development. Prior aggregate Train statistics had been observed; 96 individually exposed IDs in stored statistics were excluded from sealed candidates. Saved history cannot prove absence of all unrecorded exposure.

The original experimental lock hash is `f5624f7c67b97152b0fba2aa1887aa616b681cab7e8537faa2f5de7fadd72fe0`. Settings, 100 IDs, frozen selection and aggregate-file hashes are retained. A new reproduction lock verifies the packaged implementation, not the original implementation's byte identity.

## Results

| Method | Strict accuracy | Macro-F1 | Flip | Error introduced | Retention | Stage wall min |
|---|---:|---:|---:|---:|---:|---:|
| Original | 67.71% | 63.59% | 0.00% | 0.00% | 100.00% | 21.60 |
| Naive | 52.47% | 49.04% | 47.00% | 28.53% | 11.42% | 15.99 |
| Contextualized | 54.82% | 51.58% | 43.41% | 25.82% | 12.66% | 16.32 |
| LongLLMLingua | 61.59% | 58.19% | 29.76% | 15.76% | 31.46% | 71.81 |

Every method produced 1,700 valid labels. Error introduced / corrected counts relative to Original were 485/226 for Naive, 439/220 for Contextualized, and 268/164 for LongLLMLingua. Original itself was wrong on 549 of 1,700 hypotheses. Reduced disagreement does not imply universal correction.

LongLLMLingua minus Naive accuracy was +9.12 pp [6.12,12.29], Macro-F1 +9.15 pp [6.29,12.07], and flip −17.24 pp [−20.82,−13.65]. LongLLMLingua minus Original accuracy was −6.12 pp [−8.24,−4.12]. These percentile intervals use 5,000 joint document-ID resamples with seed 20261019, keeping all 17 hypotheses together. Multiple contrasts and domain-specific selection limit broad significance claims.

The 99% valid-output target and improvement over Naive were met. The within-3-pp-of-Original and realized 15–30% retention targets were not met. LongLLMLingua had lower NotMentioned recall than Naive (62.27% vs 73.64%), despite higher Entailment recall (59.55% vs 33.25%). Supports were 848 Entailment, 192 Contradiction and 660 NotMentioned.

## Cost and evidence

Qwen token totals were 4,334,245 / 1,140,396 / 1,198,721 / 1,728,571. LongLLMLingua additionally processed 13,423,715 compressor forward tokens. These are not interchangeable token units. Seventeen hypothesis-specific compression runs per document explain why lower judge input volume does not produce lower total latency here. Item latency sums and stage wall times are separate fields in the JSON. External API cost was zero; electricity and hardware amortization were not measured.

Nonempty exact quote matches were 934/1,365 Original, 855/869 Naive, 923/937 Contextualized and 786/1,073 LongLLMLingua. A valid label output does not guarantee a valid quote. Conversely, substring checks do not establish semantic entailment or factual completeness. No raw evidence quotes are redistributed.

## Interpretation and v1 context

LongLLMLingua preserves more of the original judge's decisions and is more accurate than generic summaries in this experiment, while remaining worse than reading the complete document. It also receives the hypothesis and more text (31.46% vs 11.42–12.66%). Therefore this result cannot isolate the benefit of the compression algorithm under equal information budgets.

The historical v1 audit attributed 629 failed judgments to 37 failed document-method summaries across 30 unique documents (seven failed in both summary methods). Naive contributed 187 failures from 11 summaries, Contextualized 442 from 26. Final word/bullet-budget violations overlapped; these were not 629 NLI parser failures. v2 adds schema-first summaries, bounded formal retries and one-hypothesis judgment calls. Because all four v2 methods have 100% valid output on the same sealed population, the within-v2 accuracy difference is not explained by missing labels. Cross-v1/v2 gains cannot be causally assigned to a single intervention.

## Scope

This is an adaptation of [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) to [ContractNLI](https://aclanthology.org/2021.findings-emnlp.164/), not a new compression algorithm or a complete replication of the financial decision-fidelity paper. Single judge/summary models, related Qwen families, 17 fixed NDA hypotheses, prior exposure and automated evidence limitations remain. Legal textual inference is not a determination of legal enforceability.
