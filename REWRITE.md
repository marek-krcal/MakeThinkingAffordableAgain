# Configurable jobs (draft)

Upstream evaluation is unchanged at `bd52d8d54b6464478516dfa40458e73612926f2f`.
Install `requirements-tc.txt`, then run `python -m tc --name=example job.yaml`:

```yaml
command: train
groups:
  input: runs/graded-training
algorithm: simpo
```

Jobs expose their own `run(context)` implementations through `tc.jobs`: `Prepare`, `Generate`, `Train`, `Evaluate`,
`Compare`. Use `Train(groups={"input": "runs/graded-training"}).execute()` from Python;
`job.save("job.yaml")` serializes its resolved defaults. Unknown fields, wrong
YAML types and incompatible settings fail before execution. Jobs execute in the current
Python runtime, from a clean committed checkout. Colab is the supported production
environment; CPU fixtures also run in CI. Each run records its
configuration, source commit and resolved model revisions in a fresh directory.
`Job.execute()` supplies a typed ExecutionContext through `Job.execution_context()` in job.py.

Prepare selects question JSONL only. Generate answers, evaluate the generation directory,
then train with the evaluation directory in `groups.input`. PreferenceGroups.form_groups()
verifies grades against original tokens and forms preferences before model loading.
Compare takes two evaluation directories. Paths always identify explicit artifacts.
`python -m tc status JOB_UUID`, `watch` and `stop` inspect/control individual jobs.

Generation needs upstream `requirements.txt` and a GPU. Training needs Torch,
Transformers 4.57.1, PEFT 0.17.1 and Accelerate 1.10.1; `device: cpu` supports tiny
fixtures. Batch size counts views per update. Both objectives average pairs within
each view, then views within an update. DPO uses summed completion scores and a
frozen adapter-disabled reference; choose `algorithm: dpo`. `resume` names a prior
checkpoint directory and creates a new execution.

Question rows retain upstream fields plus `split` and `seed_index`; groups contain
`id`, `prompt_tokens` and responses with `sample_index`, `correct` and
`completion_tokens`. Preparation defaults to `split: train`.

Run tests with `python -m unittest discover -s tests/rewrite`.
Upload-aware retention and dataset adapters remain
unfinished. Colab/Drive helpers use `requirements-remote.txt`; their local OAuth
credentials are separate from connected-app authentication. The two-question
SimPO/DPO training passed on L4; generation, grading and comparison passed on A100.
Full lifecycle acceptance remains unfinished.
AblGSM8k reproduction is running on L4 (982 updates).

On Colab, clone this branch, install the same requirements, and run jobs on the attached GPU.
For the explicit two-question,
two-update smoke pipeline, run `PYTHONPATH=. python tests/rewrite/smoke.py /content/smoke STAGE`:
`generate` on A100, `train` on L4, then `compare` on A100. Transfer the ledger and
required artifacts at the same paths between runtimes. Keep only logs/results after smoke checks.

For long jobs, run `python -m tc.diagnostics DIAGNOSTICS_DIR python -u -m tc job.yaml`.
Use a new diagnostics directory for each process; existing directories are rejected.
It saves console output, exit codes/signals and process/GPU/memory snapshots. Copy
these logs off the VM during execution; a destroyed VM cannot report its own cause.

`tc.collector.Collector` is a serializable collection job; save its configuration
and run `python -m tc.collector collector.yaml`. It polls every 90 seconds and
waits for process exit, then fetches final logs and verifies archives before release.
Failed transfers retain the runtime for retry; health/recovery checks run every 15 minutes.
