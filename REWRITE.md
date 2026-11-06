# Configurable jobs (draft)

Install `requirements-tc.txt`. Jobs are serializable dataclasses exposed by `tc.jobs`.
`job.save("job.yaml")` saves defaults; `python -m tc job.yaml` records resolved settings
and source/model revisions. Worker execution is added in later PRs.

Run tests with `python -m unittest discover -s tests/rewrite`.
