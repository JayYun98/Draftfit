"""Native HF assistant spans must not supervise matching user text."""

import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from draftfit.data.parse import HFParser
from draftfit.data.preprocessing import preprocess_conversations
from draftfit.data.template import TEMPLATE_REGISTRY


class HFParserTest(unittest.TestCase):
    def setUp(self):
        backend = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "user": 1, "assistant": 2, "same": 3, "answer": 4},
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]"
        )
        self.tokenizer.chat_template = (
            "{% for m in messages %}{{ m.role }} "
            "{% if m.role == 'assistant' %}{% generation %}{{ m.content }}{% endgeneration %}"
            "{% else %}{{ m.content }}{% endif %} {% endfor %}"
        )
        self.messages = [
            {"role": "user", "content": "same answer"},
            {"role": "assistant", "content": "same answer"},
        ]
        self.parser = HFParser(self.tokenizer, TEMPLATE_REGISTRY.get("hf"))

    def test_exact_mask_and_truncation(self):
        ids, mask = self.parser.parse(self.messages, 100)
        self.assertEqual(ids.tolist(), [1, 3, 4, 2, 3, 4])
        self.assertEqual(mask.tolist(), [0, 0, 0, 0, 1, 1])
        ids, mask = self.parser.parse(self.messages, 5)
        self.assertEqual(mask.tolist(), [0, 0, 0, 0, 1])
        result = preprocess_conversations(
            self.tokenizer, [self.messages], TEMPLATE_REGISTRY.get("hf"), tools=[[]]
        )
        self.assertEqual(result["loss_mask"][0].tolist(), [[0, 0, 0, 0, 1, 1]])

    def test_missing_native_spans_and_ambiguous_inputs_fail(self):
        for options in ({"preformatted": True}, {"train_only_last_turn": True}):
            with self.assertRaisesRegex(ValueError, "structured messages"):
                self.parser.parse(self.messages, 100, **options)
        self.tokenizer.chat_template = "{{ messages[0].content }}"
        with self.assertRaisesRegex(ValueError, "generation"):
            self.parser.parse(self.messages, 100)

    def test_tool_arguments_json_is_decoded_without_mutating_input(self):
        messages = [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "answer", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {
                    "name": "lookup", "arguments": '{"query": "same"}'
                }}
            ]},
        ]
        normalized = self.parser._normalize_hf_message(messages[1])
        self.assertEqual(
            normalized["tool_calls"][0]["function"]["arguments"],
            {"query": "same"},
        )
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            '{"query": "same"}',
        )
        self.tokenizer.chat_template = (
            "{% for m in messages %}{{ m.role }} "
            "{% if m.role == 'assistant' %}{% generation %}"
            "{% for call in m.tool_calls %}"
            "{% for key, value in call.function.arguments.items() %}"
            "{{ key }} {{ value }}{% endfor %}{% endfor %}"
            "{{ m.content }}{% endgeneration %}"
            "{% else %}{{ m.content }}{% endif %} {% endfor %}"
        )
        ids, mask = self.parser.parse(messages, 100)
        self.assertEqual(ids.shape, mask.shape)
        self.assertGreater(int(mask.sum()), 0)
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            self.parser._normalize_hf_message({
                "role": "assistant", "tool_calls": [
                    {"function": {"name": "lookup", "arguments": "{"}}
                ],
            })


if __name__ == "__main__":
    unittest.main()
