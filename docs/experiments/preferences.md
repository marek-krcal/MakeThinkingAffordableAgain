# Preference experiments

Run sequentially from fresh Qwen3-0.6B weights: shortest-correct SimPO, DPO (β=0.1), then a disjoint external-data mixture added to GSM8k. Generation uses A100; training uses L4. Preserve the frozen evaluation cohort and upstream graders.

`groups: {input: graded-groups.jsonl, source: groups, pairing: shortest, correct_ratio: 1.5}` reuses archived graded tokens. The default `source: evaluation` verifies grades against saved generations. Archived groups require `id`, `split`, `prompt_tokens` and graded `responses`; retain their provenance separately.

The shortest correct answer wins ties by sample index. Reject wrong answers only when strictly longer; reject other correct answers when at least 1.5× longer. Each eligible pair becomes one two-response training view; no negative-negative preferences. Default pairing and four-response views remain unchanged.

Keep the original 982-update optimizer/LoRA recipe and 7,852 view exposures. Pairwise views change response exposure, which must be reported. Evaluate final step 982 as primary; step 737 is secondary. SimPO succeeds if its final point is not dominated by Qwen/737/982, saves at least 29.36% geometric-mean tokens, and loses at most 2 percentage points of four-task macro accuracy versus Qwen. These are exploratory point estimates, not significance claims. DPO uses the new pairs on success, otherwise the original preferences. The data mixture keeps that DPO recipe and pairing, changing only the input groups to isolate the added data.

Audit external sources, licenses, revisions and train splits before sampling. Exclude evaluation questions, their source tasks and near duplicates; do not train on held-out answers. Save the mixture and exclusions before generation, then retain the same evaluation and update budget.

`prepare-external` replays selection from audited question JSONL artifacts. Its `sources` JSON manifest binds each artifact and audit to the raw source revision/hash, license and protected cohort hash; code sources also provide validated fixture JSONL. `protected` points to the frozen suite, and `count` defaults to 200 per domain. Curation scripts and fixture proofs accompany the input artifacts.

Instruction selection matches aggregate prompt length and constraint count; domains are then interleaved without changing source seeds. The job copies grader code into a separate training suite and removes held-out question assets. Training code labels use three original source cases, not expanded MBPP+ tests; full evaluation remains unchanged.
