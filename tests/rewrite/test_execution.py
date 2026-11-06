import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


class ExecutionTests(unittest.TestCase):
    def test_prepare_cli_and_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'questions.jsonl'
            row = {'id': 'gsm:1', 'split': 'train', 'seed_index': 17,
                   'messages': [{'role': 'user', 'content': '2 + 2?'}], 'answer': '4'}
            source.write_text(json.dumps(row) + '\n')
            config = root / 'prepare.yaml'
            config.write_text(yaml.safe_dump(dict(command='prepare', name='questions', 
                                                   input=[str(source)], jobs_dir=str(root / 'jobs'))))
            command = [sys.executable, '-m', 'tc', str(config)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job = Path(result.stdout.strip())
            self.assertEqual(json.loads((job / 'status.json').read_text())['state'], 'completed')
            self.assertEqual(json.loads((job / 'outputs/questions.jsonl').read_text()), row)
            record = yaml.safe_load((job / 'config.yaml').read_text())
            self.assertEqual(job.parent, (root / 'jobs').resolve())
            self.assertEqual(job.name, record['job_id'])
            self.assertEqual(record['name'], 'questions')
            source.write_text('{}\n')
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            failed_job = Path(failed.stdout.strip())
            self.assertNotEqual(failed_job, job)
            self.assertEqual(json.loads((job / 'status.json').read_text())['state'], 'completed')
            self.assertEqual(json.loads((failed_job / 'status.json').read_text())['state'], 'failed')
            self.assertFalse((failed_job / 'artifacts.json').exists())
