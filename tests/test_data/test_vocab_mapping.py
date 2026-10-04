from collections import Counter

from draftfit.data.preprocessing import process_token_dict_to_mappings


def test_vocab_mapping_marks_selected_tokens() -> None:
    d2t, t2d = process_token_dict_to_mappings(Counter({4: 3, 1: 2}), 2, 6)
    assert d2t.tolist() == [1, 3]
    assert t2d.tolist() == [False, True, False, False, True, False]
