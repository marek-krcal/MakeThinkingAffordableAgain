import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tc.external import PrepareExternal, sample_instructions
from tc.job import ExecutionContext


def save(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExternalTests(unittest.TestCase):
    def test_audit_binding_protected_ids_and_training_only_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root/'protected'
            protected.mkdir()
            heldout = dict(id='heldout', grader='bfcl', messages=[dict(role='user',content='secret test')])
            digest = save(protected/'questions.jsonl', heldout)
            save(protected/'manifest.json', dict(cohort_sha256=digest,revisions=dict(bfcl='frozen',mbpp='frozen')))
            for path in ('bfcl/bfcl_eval/data/test.json','instruction_following_eval/data/test.json','evalplus/evalplus/data/helper.py'):
                file=protected/'vendor'/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('retained only if grader code')
            row=dict(id='train',grader='bfcl',dataset='bfcl_simple_python',split='train',seed_index=1,messages=[dict(role='user',content='training question')])
            questions=root/'questions.jsonl'
            def inputs(row):
                qhash=save(questions,row)
                ahash=save(root/'audit.json',dict(source_sha256='raw',questions_sha256=qhash,protected_sha256=digest))
                save(root/'sources.json',dict(protected_sha256=digest,sources=[dict(domain='bfcl',questions='questions.jsonl',questions_sha256=qhash,revision='pin',raw_sha256='raw',license='license',audit='audit.json',audit_sha256=ahash)]))
            inputs(row)
            output=root/'run';(output/'outputs').mkdir(parents=True)
            job=PrepareExternal(sources=str(root/'sources.json'),protected=str(protected))
            job.run(ExecutionContext(output,{}))
            suite=output/'outputs/suite'
            self.assertTrue((suite/'vendor/evalplus/evalplus/data/helper.py').exists())
            self.assertFalse((suite/'vendor/bfcl/bfcl_eval/data').exists())
            self.assertFalse((suite/'vendor/instruction_following_eval/data').exists())
            self.assertEqual(json.loads((suite/'questions.jsonl').read_text())['id'],'train')
            questions.write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                job.run(ExecutionContext(output,{}))
            inputs(dict(row,id='heldout'))
            with self.assertRaisesRegex(ValueError,'protected evaluation identities'):
                job.run(ExecutionContext(output,{}))
            inputs(row)
            (protected/'questions.jsonl').write_text('{}')
            with self.assertRaisesRegex(ValueError,'protected evaluation cohort changed'):
                job.run(ExecutionContext(output,{}))

    def test_fallback_strata_do_not_reuse_questions(self):
        rows=[dict(id=str(i),original_prompt='word '*40,instruction_id_list=['a','b','c']) for i in range(20)]
        protected=[dict(grader='ifeval',original_prompt='word '*n,instruction_id_list=['a','b','c']) for n in (10,40)]
        chosen=sample_instructions(rows,protected,10,42)
        self.assertEqual(len({r['id'] for r in chosen}),10)
        with self.assertRaisesRegex(ValueError,'insufficient audited'):
            sample_instructions(rows[:4],protected,10,42)
