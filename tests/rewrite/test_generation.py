from tc.job import ExecutionContext
import unittest
from tc.generate import pending_jobs


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [{'row': {'id': 'a'}, 'sample': i, 'seed': 9001+i} for i in range(3)]
        self.done = {'id': 'a', 'sample_index': 1, 'seed': 9002, 'token_ids': [1, 2]}

    def test_only_missing_samples(self):
        self.assertEqual(pending_jobs(self.jobs, [self.done]), [self.jobs[0], self.jobs[2]])

    def test_reject_incompatible_records(self):
        for records in ([self.done, self.done], [dict(self.done, seed=0)],
                        [dict(self.done, id='other')], [dict(self.done, token_ids=None)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                pending_jobs(self.jobs, records)


class DurableResumeTests(unittest.TestCase):
    def test_only_unfinished_tail_is_ignored(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from tc.generate import completed_records
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'records.jsonl'
            row = {'id': 'a', 'token_ids': [1]}
            for tail in (b'{"id":', b'{"text":"\xe2\x82', b'{"id":"b"}'):
                path.write_bytes(json.dumps(row).encode()+b'\n'+tail)
                self.assertEqual(completed_records(path), [row])
            path.write_bytes(json.dumps(row).encode()+b'\n{broken}\n')
            with self.assertRaises(json.JSONDecodeError):
                completed_records(path)

    def test_adapter_identity_and_resume_validation(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from contextlib import contextmanager
        from types import ModuleType
        from unittest.mock import Mock, patch
        import yaml
        from tc.generate import Generate, adapter_fingerprint
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = root / 'adapter'
            adapter.mkdir()
            config = adapter / 'adapter_config.json'
            weights = adapter / 'adapter_model.safetensors'
            config.write_text('{"r": 16}')
            weights.write_bytes(b'original weights')
            identity = {'adapter_sha256': adapter_fingerprint(str(adapter))}
            self.assertIsNone(adapter_fingerprint(None))
            config.write_text('{"r": 32}')
            self.assertNotEqual(adapter_fingerprint(str(adapter)), identity['adapter_sha256'])
            config.write_text('{"r": 16}')
            question = {'id': 'a', 'dataset': 'svamp'}
            source = root / 'questions.jsonl'
            source.write_text(json.dumps(question)+'\n')
            previous, current = root/'previous', root/'current'
            for p in (previous, current): (p/'outputs').mkdir(parents=True)
            job = Generate(input=[str(source)], responses=1, adapter=str(adapter), resume=str(previous))
            revisions = {k: {'id': 'test', 'revision': 'a'*40} for k in ('model','tokenizer')}
            record = {'arguments': job.config(), 'revisions': revisions}
            (previous/'config.yaml').write_text(yaml.safe_dump(record))
            (previous/'outputs/prompts.json').write_text(json.dumps([{'question':question,'token_ids':[1]}]))
            (previous/'outputs/model-identity.json').write_text(json.dumps(identity))
            response = {'id':'a','sample_index':0,'seed':9001,'token_ids':[2]}
            (previous/'outputs/records.jsonl').write_text(json.dumps(response)+'\n')
            @contextmanager
            def execution():
                yield ExecutionContext(current, record)
            fake = ModuleType('vllm'); fake.LLM = Mock(); fake.SamplingParams = Mock()
            request = ModuleType('vllm.lora.request'); request.LoRARequest = Mock()
            modules = {'vllm':fake,'vllm.lora':ModuleType('vllm.lora'),'vllm.lora.request':request}
            with patch.dict('sys.modules',modules), patch.object(job,'execution_context',execution), patch('transformers.AutoTokenizer.from_pretrained'), patch('tc.generate.upstream.build_jobs',return_value=[{'row':question,'sample':0,'seed':9001,'ids':[1]}]), patch('tc.generate.upstream.token_cap',return_value=1):
                weights.write_bytes(b'replaced weights')
                with self.assertRaisesRegex(ValueError,'adapter identity'):
                    job.execute()
                weights.write_bytes(b'original weights')
                (previous/'outputs/model-identity.json').unlink()
                with self.assertRaisesRegex(ValueError,'adapter identity'):
                    job.execute()
                (previous/'outputs/model-identity.json').write_text(json.dumps(identity))
                self.assertEqual(job.execute(), current)
                self.assertEqual(json.loads((current/'outputs/records.jsonl').read_text()), response)
                status = json.loads((current/'status.json').read_text())
                self.assertEqual((status['completed'], status['total'], status['output_tokens']), (1, 1, 1))
                fake.LLM.assert_not_called()
