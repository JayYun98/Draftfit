"""Native HF assistant spans must not supervise matching user text."""

import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from specforge.data.parse import HFParser
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


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


if __name__ == "__main__":
    unittest.main()
