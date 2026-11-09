"""Numeric training/evaluation formatting and saved-grade regressions."""

from copy import deepcopy
import unittest

from utils.train.regrade import regrade_group
from utils.train.rewards import correctness, is_task_correct


class NumericRewards(unittest.TestCase):
    def test_evaluation_answer_formats(self):
        for reply, expected in [
            ("<think>work</think>\nFinal answer: 42", True),
            ("Final answer: 41\nFinal answer: 42", True),
            ("**Final answer:** 42", False),
            ("Final answer: $\\boxed{42}$", False),
            ("Final answer: 84/2", False),
            ("Final answer: <answer>42</answer>", False),
            ("Final answer: 42 apples", False),
            ("Final answer: 42\nThanks!", False),
        ]:
            with self.subTest(reply=reply):
                self.assertEqual(is_task_correct(reply,"42","math"),expected)
        self.assertTrue(is_task_correct("Final answer: 1,234.5","1234.5","math"))

    def test_malformed_thinking_is_wrong_with_a_correct_final_line(self):
        for reply in ("<think>Final answer: 42", "work</think>\nFinal answer: 42",
                      "<think>x</think><think>y\nFinal answer: 42",
                      "<think>x</think><think>y</think>\nFinal answer: 42"):
            with self.subTest(reply=reply):
                self.assertFalse(is_task_correct(reply,"42","math"))

    def test_batch_handles_truncation_and_chat_completions(self):
        replies=["Final answer: 42",[{"role":"assistant","content":"Final answer: 42"}]]
        self.assertEqual(correctness(replies,["42","42"],["math","math"],truncated=[True,False]),[0.,1.])

    def test_choice_extraction_is_preserved(self):
        self.assertTrue(is_task_correct("<think>x</think>\n**Final answer:** B","B","multiple_choice"))


class SavedGrades(unittest.TestCase):
    def test_only_labels_change_and_prompt_identity_is_checked(self):
        class Tokenizer:
            eos_token_id=9
            def decode(self,tokens,**kwargs):
                return {1:"**Final answer:** 42",2:"Final answer: 42",3:"<think>Final answer: 42"}[tokens[0]]
        group={"id":"gsm8k/1","dataset":"gsm8k","prompt_tokens":[7],"split":"train","responses":[
            {"sample_index":i,"completion_tokens":tokens,"correct":True,"seed":42}
            for i,tokens in enumerate(([1,9],[2,9],[3,9],[2,8]))]}
        original=deepcopy(group)
        questions={group["id"]:{"prompt_tokens":[7],"answer":"42"}}
        result=regrade_group(group,questions,Tokenizer())
        self.assertEqual([r["correct"] for r in result["responses"]],[False,True,False,False])
        self.assertEqual(group,original)
        for old,new in zip(group["responses"],result["responses"]):
            self.assertEqual({k:v for k,v in old.items() if k!="correct"},{k:v for k,v in new.items() if k!="correct"})
        external=dict(group,dataset="ifeval")
        self.assertEqual(regrade_group(external,{},Tokenizer()),external)
        with self.assertRaisesRegex(ValueError,"gold prompt differs"):
            regrade_group(group,{group["id"]:{"prompt_tokens":[8],"answer":"42"}},Tokenizer())
