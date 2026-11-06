# Configurable jobs (draft)

Install `requirements-tc.txt`. Jobs are serializable dataclasses exposed by `tc.jobs`.
`job.save("job.yaml")` saves defaults; `python -m tc job.yaml` runs `Prepare` with `local: true`; it records settings, source/model revisions and progress.
Other worker methods are added in later PRs.

Run tests with `python -m unittest discover -s tests/rewrite`.

Jobs execute in the current Python runtime; Colab is the supported production environment.
Use a clean committed checkout. CPU fixtures also run in CI.

Train forms preferences before model loading with `groups: {input: runs/graded-training}`.
The nested PreferenceGroups configuration selects split, count, seed and exclusions.
