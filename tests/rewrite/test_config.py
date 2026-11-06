import contextlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tc.cli import parse_args
from tc.jobs import load_job, Train, Generate, Prepare, Evaluate
from tc.job import archive_source, make_record, repository_url, write_record


class ConfigTests(unittest.TestCase):
    def test_invalid_inputs(self):
        for values in ({'steps': 1, 'epochs': 1}, {'algorithm': 'dpo', 'gamma': 1},
                       {'steps': 0}, {'steps': True}, {'mod': 'foo'}, {'beta': float('nan')},
                       {'input': 'file.jsonl'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_job({'command': 'train', **values})

    def test_direct_construction_rejects_invalid_configuration(self):
        for cls, values in ((Train, {"steps": True}), (Train, {"steps": 0}),
                            (Train, {"retain_step": [0]}), (Train, {"input": "rows.jsonl"}),
                            (Generate, {"responses": 0}), (Generate, {"top_p": float("nan")}),
                            (Prepare, {"count": 0}), (Prepare, {"dataset_revision": "abc"}),
                            (Evaluate, {"workers": 0}), (Evaluate, {"sandbox": "unknown"})):
            with self.subTest(job=cls.__name__, values=values), self.assertRaises(ValueError):
                cls(**values)

    def test_yaml_and_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.yaml'
            job = Train(algorithm='dpo', epochs=2, input=['groups.jsonl'])
            job.save(path)
            loaded = parse_args(['--name=experiment', str(path)])
            self.assertEqual(loaded.name, 'experiment')
            self.assertEqual(loaded.config(), {**job.config(), 'name': 'experiment'})
            self.assertEqual(load_job(loaded.config()), loaded)
            path.write_text('command: train\nsteps: wrong\n')
            with self.assertRaises(SystemExit):
                parse_args([str(path)])

    def test_repository_url_removes_credentials(self):
        self.assertEqual(repository_url("https://user:secret@github.com/org/repo.git?token=x"),
                         "https://github.com/org/repo")
        self.assertEqual(repository_url("git@github.com:org/repo.git"), "https://github.com/org/repo")

    def test_record_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            git("remote", "add", "origin", "https://github.com/example/project.git")
            (repo / "worker.py").write_text("answer = 42\n")
            git("add", "worker.py")
            git("commit", "-qm", "base")
            commit = git("rev-parse", "HEAD")
            resolve = lambda kind, name, revision: "a" * 40
            with patch("tc.job.UPSTREAM_COMMIT", commit):
                args = Train(algorithm="dpo")
                record = make_record(args, repo, resolve)
                self.assertEqual(record["source_commit"], commit)
                self.assertEqual(record["original_argv"], [])
                self.assertEqual(record["revisions"]["reference"], record["revisions"]["model"])
                self.assertNotEqual(record["job_id"], make_record(args, repo, resolve)["job_id"])
                out = Path(tmp) / "job"
                write_record(record, out)
                self.assertEqual(yaml.safe_load((out / "config.yaml").read_text()), record)
                with self.assertRaises(FileExistsError):
                    write_record(record, out)
                (repo / "untracked.py").write_text("secret = True\n")
                with self.assertRaisesRegex(ValueError, "committed source"):
                    make_record(args, repo, resolve)
                (repo / "untracked.py").unlink()
                (repo / "worker.py").write_text("answer = 0\n")
                with self.assertRaisesRegex(ValueError, "committed source"):
                    make_record(args, repo, resolve)
                archive = Path(tmp) / "source.tar"
                archive_source(record, repo, archive)
                with tarfile.open(archive) as source:
                    self.assertEqual(source.extractfile("worker.py").read(), b"answer = 42\n")


if __name__ == "__main__":
    unittest.main()
