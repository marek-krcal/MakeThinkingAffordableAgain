from tc.job import ExecutionContext
from pathlib import Path
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from tc.jobs import load_job
from tc.sampling import batches, learning_rate
from unittest.mock import patch
from contextlib import contextmanager
import json
from tc.job_utils import progress


def groups():
    return [{'id': 'one', 'prompt_tokens': [1, 2], 'responses': [
        {'sample_index': 0, 'correct': True, 'completion_tokens': [3, 4]},
        {'sample_index': 1, 'correct': False, 'completion_tokens': [5, 6, 7]}]}]


def model():
    torch.manual_seed(5)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                                       num_hidden_layers=1, num_attention_heads=2,
                                       num_key_value_heads=1, head_dim=8))
    return get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=.1,
                                         target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))


class TrainingTests(unittest.TestCase):
    def test_recipe_exposure_and_lr(self):
        schedule = batches(groups(), seed=42, batch_size=8, steps=982)
        self.assertEqual(len(schedule), 982)
        self.assertEqual(sum(map(len, schedule)), 7852)
        self.assertEqual([i+1 for i, b in enumerate(schedule) if len(b) == 6], [491, 982])
        self.assertEqual(learning_rate(49, 982, 1.), 1.)
        self.assertEqual(learning_rate(786, 982, 1.), 1.)
        self.assertLess(learning_rate(787, 982, 1.), 1.)
        self.assertEqual(learning_rate(982, 982, 1.), 0.)
        self.assertTrue(all(len(b) == 8 for b in batches(groups(), seed=42, batch_size=8, steps=500)))

    def test_both_algorithms_resume_exactly(self):
        for algorithm in ('simpo', 'dpo'):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = load_job({'command': 'train', 'algorithm': algorithm, 'steps': 4, 'batch_size': 2, 'checkpoint_every': 1, 'retain_step': [2], 'learning_rate': 0.001, 'device': 'cpu'})
                full, interrupted, resumed = [root / name for name in ('full', 'interrupted', 'resumed')]
                for path in (full, interrupted, resumed):
                    (path / 'outputs').mkdir(parents=True)
                args.groups.input = 'synthetic-evaluation'

                def run(directory, interrupt=False):
                    @contextmanager
                    def execution():
                        yield ExecutionContext(directory, {'arguments': args.config(), 'revisions': {'model': {'id': 'test', 'revision': 'a'*40}}})
                    models = []
                    def load(*a, **kw):
                        models.append(model())
                        return models[-1]
                    def update(path, **fields):
                        progress(path, **fields)
                        if interrupt and fields.get('step') == 2:
                            raise InterruptedError()
                    with patch('tc.preferences.PreferenceGroups.form_groups', return_value=groups()), patch.object(args, 'execution_context', execution), patch('transformers.AutoModelForCausalLM.from_pretrained', load), patch('peft.get_peft_model', side_effect=lambda m, c: m), patch('tc.train.progress', update):
                        args.execute()
                    return models[0]

                initial = {n: p.detach().clone() for n, p in model().named_parameters() if p.requires_grad}
                original = run(full)
                self.assertTrue(any(not torch.equal(initial[n], p) for n, p in original.named_parameters() if n in initial))
                with self.assertRaises(InterruptedError):
                    run(interrupted, interrupt=True)
                args.resume = str(interrupted / 'checkpoints/step-2')
                recovered = run(resumed)
                for (name, expected), (_, actual) in zip(original.named_parameters(), recovered.named_parameters()):
                    torch.testing.assert_close(expected, actual, atol=0, rtol=0, msg=name)
