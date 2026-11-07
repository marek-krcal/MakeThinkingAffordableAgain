# Preference experiments

Run sequentially from fresh Qwen3-0.6B weights: shortest-correct SimPO, DPO (β=0.1), then a disjoint external-data mixture added to GSM8k. Generation uses A100; training uses L4. Preserve the frozen evaluation cohort and upstream graders.

`groups: {input: graded-groups.jsonl, source: groups, pairing: shortest, correct_ratio: 1.5}` reuses archived graded tokens. The default `source: evaluation` verifies grades against saved generations. Archived groups require `id`, `split`, `prompt_tokens` and graded `responses`; retain their provenance separately.

Pairing names: `original` retains upstream preferences; `correct-longer` uses the shortest correct positive, longer wrong negatives and correct negatives at least 1.5× longer; `all-longer` requires that 1.5× threshold for every negative. These names describe selection independently of SimPO or DPO.

The shortest correct answer wins ties by sample index. Reject wrong answers only when strictly longer; reject other correct answers when at least 1.5× longer. Each eligible pair becomes one two-response training view; no negative-negative preferences. Default pairing and four-response views remain unchanged.

Keep the original 982-update optimizer/LoRA recipe and 7,852 view exposures. Pairwise views change response exposure, which must be reported. Evaluate final step 982 as primary; step 737 is secondary. SimPO succeeds if its final point is not dominated by Qwen/737/982, saves at least 29.36% geometric-mean tokens, and loses at most 2 percentage points of four-task macro accuracy versus Qwen. These are exploratory point estimates, not significance claims. DPO uses the new pairs on success, otherwise the original preferences. For data comparisons, keep each objective’s original recipe and pairing, changing only the input groups. The priority mixture uses original SimPO (β=2, γ=1), with the same sealed questions, 982 updates and LoRA settings; evaluate 737/860/982 against GSM8k-only SimPO.

The lower-KL DPO control starts fresh on the original GSM8k groups with β=0.05 instead of 0.1. Keep the 982-update recipe unchanged and screen steps 737/860/982.

Audit external sources, licenses, revisions and train splits before sampling. Exclude evaluation questions, their source tasks and near duplicates; do not train on held-out answers. Save the mixture and exclusions before generation, then retain the same evaluation and update budget.

The expanded run generates eight answers for 2,000 audited candidates: 575 code questions (158 MBPP, 417 KodCode-Light-RL-10K), 718 IF_multi_constraints_upto5 questions and 707 ToolACE questions. After preference filtering, select 1,000 question groups by domain with seed 42 and add the unchanged 2,287 GSM8k groups; keep 982 updates and 7,852 four-response views. Never subsample using held-out scores.

The `all-longer` variant sets `wrong_ratio: 1.5`: both wrong and correct negatives must be at least 1.5× the positive token length. The `correct-longer` variant keeps `wrong_ratio: 1.0` and requires wrong negatives to be strictly longer. Run `all-longer` SimPO after DPO and before the external mixture.

New evaluations use one answer per question across the full frozen cohort. Retain steps 737, 860 (checkpoint 3.5), and 982; reserve eight-answer evaluations for later user selection. Compare against a matched one-answer Qwen baseline; preserve earlier eight-answer results separately. Training-data generation still samples eight answers per question.

Keep one rolling safety checkpoint during training; delete it after successful training. Requested evaluation steps and the final checkpoint remain. Failures preserve recovery state.

`prepare-external` replays selection from audited question JSONL artifacts. Its `sources` JSON manifest binds each artifact and audit to the raw source revision/hash, license and protected cohort hash; code sources also provide validated fixture JSONL. `protected` points to the frozen suite, and `count` defaults to 200 per domain. Curation scripts and fixture proofs accompany the input artifacts.

Instruction selection matches aggregate prompt length and constraint count; domains are then interleaved without changing source seeds. The job copies grader code into a separate training suite and removes held-out question assets. Training code labels use three original source cases, not expanded MBPP+ tests; full evaluation remains unchanged.

An explicit continuation sets `resume`, `extend_from_steps: 982` and a larger `steps` budget. It restores optimizer/RNG state, preserves the original batch sequence and partial-batch pattern, and extends the learning-rate horizon; other scientific settings must match. The DPO continuation branches from step 737 to 1473 at the same peak LR and β, separately from the original reproduction.
