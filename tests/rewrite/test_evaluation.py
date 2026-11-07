from tc.job import ExecutionContext
import json
from pathlib import Path
import tempfile
import shutil
import unittest
import yaml

from tc.jobs import load_job
from tc.preferences import PreferenceGroups
from contextlib import contextmanager
from unittest.mock import patch


def run(job, directory):
    @contextmanager
    def execution():
        (directory / "config.yaml").write_text(yaml.safe_dump({"arguments": job.config()}))
        yield ExecutionContext(directory, {})
        (directory / "status.json").write_text(json.dumps({"state": "completed"}))
    with patch.object(job, "execution_context", execution):
        return job.execute()


class SavedEvaluationTests(unittest.TestCase):
    def test_grade_join_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation, graded, comparison = [root / n for n in ('gen', 'eval', 'compare')]
            for directory in (generation, graded, comparison):
                (directory / 'outputs').mkdir(parents=True)
            args = load_job({'command': 'generate', 'responses': 2})
            config = {'arguments': vars(args), 'revisions': {'model': {'revision': 'a'*40}}}
            (generation / 'config.yaml').write_text(yaml.safe_dump(config))
            question = {'id': 'q', 'seed_index': 1, 'split': 'train', 'dataset': 'svamp',
                        'grader': 'numeric', 'answer': '4', 'messages': [{'role': 'user', 'content': '2+2?'}]}
            (generation / 'outputs/prompts.json').write_text(json.dumps([{'question': question, 'token_ids': [1, 2]}]))
            records = [{'id': 'q', 'dataset': 'svamp', 'sample_index': i, 'seed': 10001+i,
                        'text': 'Final answer: ' + str(answer), 'response': str(answer),
                        'token_ids': [3, 4], 'output_tokens': 2, 'reasoning_tokens': 0,
                        'truncated': False, 'boundary_error': False} for i, answer in enumerate((4, 5))]
            (generation / 'outputs/records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            evaluation_args = load_job({'command': 'evaluate', 'input': [str(generation)]})
            run(evaluation_args, graded)
            result = json.loads((graded / 'outputs/results.json').read_text())
            self.assertEqual(result['results']['svamp']['accuracy'], .5)
            candidate = root / 'candidate'
            shutil.copytree(graded, candidate)
            candidate_meta = dict(result, adapter='/synthetic/adapter')
            (candidate / 'outputs/results.json').write_text(json.dumps(candidate_meta))
            run(load_job({'command': 'compare', 'input': [str(graded), str(candidate)]}), comparison)
            groups = PreferenceGroups(input=str(graded)).form_groups()
            self.assertEqual([r['correct'] for r in groups[0]['responses']], [True, False])
            self.assertEqual(groups[0]['prompt_tokens'], [1, 2])
            saved = graded / 'outputs/records.jsonl'
            original_grades = saved.read_text()
            tampered = [json.loads(line) for line in original_grades.splitlines()]
            tampered[0]['token_ids'] = [999]
            saved.write_text(''.join(json.dumps(row) + '\n' for row in tampered))
            with self.assertRaisesRegex(ValueError, 'differs'):
                PreferenceGroups(input=str(graded)).form_groups()
            saved.write_text(original_grades)
            with self.assertRaisesRegex(ValueError, 'no training signal'):
                PreferenceGroups(input=str(graded), exclude=['q']).form_groups()
            records.pop()
            (generation / 'outputs/records.jsonl').write_text(json.dumps(records[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                run(evaluation_args, graded)

    def test_full_cohort_metadata_interoperates_with_upstream(self) -> None:
        """Compare a saved full cohort to canonical upstream metadata; keep subsets distinct."""
        import hashlib
        from types import SimpleNamespace
        import evaluate as upstream
        from utils.eval.compare import check
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite, generation, output, baseline = [root/name for name in ('suite','gen','eval','base')]
            for path in (suite, generation/'outputs', output/'outputs', baseline):
                path.mkdir(parents=True)
            question = dict(id='q',dataset='svamp',grader='numeric',answer='4',seed_index=0,
                            messages=[dict(role='user',content='2+2?')])
            def freeze(rows):
                (suite/'questions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                digest=hashlib.sha256((suite/'questions.jsonl').read_bytes()).hexdigest()
                (suite/'manifest.json').write_text(json.dumps(dict(cohort_sha256=digest,total=len(rows),revisions={})))
            freeze([question])
            args=load_job(dict(command='generate',responses=1,max_model_len=8192,adapter='/candidate'))
            (generation/'config.yaml').write_text(yaml.safe_dump(dict(arguments=args.config(),revisions=dict(model=dict(revision='a'*40)))))
            (generation/'outputs/prompts.json').write_text(json.dumps([dict(question=question,token_ids=[1])]))
            record=dict(id='q',dataset='svamp',sample_index=0,seed=9001,text='4',response='4',token_ids=[2],output_tokens=1,reasoning_tokens=0,truncated=False,boundary_error=False)
            (generation/'outputs/records.jsonl').write_text(json.dumps(record)+'\n')
            job=load_job(dict(command='evaluate',input=[str(generation)],questions=str(suite)))
            run(job,output)
            candidate=json.loads((output/'outputs/results.json').read_text())
            options=SimpleNamespace(model=args.model,revision='a'*40,adapter=None,benchmarks=['svamp'],samples=1,seed=9001)
            upstream.save_results(baseline,options,upstream.load_suite(suite),dict(mode='unused',why='numeric'),candidate['results'],{},dict(svamp=4096))
            canonical=json.loads((baseline/'results.json').read_text())
            self.assertEqual(check(canonical,candidate),['svamp'])
            freeze([question,dict(question,id='extra',seed_index=1)])
            run(job,output)
            subset=json.loads((output/'outputs/results.json').read_text())
            self.assertEqual(subset['selection'],'saved')
            with self.assertRaisesRegex(ValueError,'selection'):
                check(canonical,subset)
