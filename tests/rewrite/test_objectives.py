import math
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from tc.objectives import preference_loss, score_completion


class ObjectiveTests(unittest.TestCase):
    def test_simpo_value_and_gradient(self):
        sums = torch.tensor([-2., -6.], requires_grad=True)
        loss = preference_loss(sums, torch.tensor([2., 3.]), [(0, 1)], algorithm='simpo', beta=2., gamma=1.)
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-1)), places=6)
        loss.backward()
        self.assertLess(sums.grad[0], 0)
        self.assertGreater(sums.grad[1], 0)

    def test_dpo_uses_sums_and_frozen_reference(self):
        sums = torch.tensor([-2., -6.], requires_grad=True)
        reference = sums.detach().clone().requires_grad_()
        loss = preference_loss(sums, torch.tensor([2., 3.]), [(0, 1)], algorithm='dpo', beta=.1, reference=reference)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertAlmostEqual(sums.grad[0].item(), -.05, places=6)
        self.assertAlmostEqual(sums.grad[1].item(), .05, places=6)
        self.assertIsNone(reference.grad)

    def test_chunked_matches_dense_values_and_gradients(self):
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                                            num_hidden_layers=1, num_attention_heads=2,
                                            num_key_value_heads=1, head_dim=8))
        model.eval()
        prompt, completion = [1, 2, 3], [4, 5, 6, 7]
        actual = score_completion(model, prompt, completion, chunk_size=2)
        actual.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()
        tokens = torch.tensor([prompt + completion])
        logits = model(tokens[:, :-1]).logits[:, len(prompt)-1:].float()
        expected = -torch.nn.functional.cross_entropy(logits.reshape(-1, 32), torch.tensor(completion), reduction='sum')
        expected.backward()
        torch.testing.assert_close(actual, expected)
        for n, p in model.named_parameters():
            if n in grads:
                torch.testing.assert_close(grads[n], p.grad, atol=1e-5, rtol=1e-4)
