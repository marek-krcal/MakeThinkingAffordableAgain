"""Public job classes and YAML configuration loader."""

from tc.config import Job, ModelJob, load_job
from tc.prepare import Prepare
from tc.generate import Generate
from tc.train import Train
from tc.evaluation import Evaluate, Compare

JOBS = {job.command: job for job in (Prepare, Generate, Train, Evaluate, Compare)}
