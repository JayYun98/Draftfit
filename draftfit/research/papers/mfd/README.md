# Multi-Future Drafting

[`method.py`](method.py) ranks distinct token prefixes by their best score
across draft modes, using probability tables supplied by the caller.

- [CPU recipe](../../../../examples/configs/research/mfd-prefix-tree.yaml)
- [Selector and CLI tests](../../../../tests/research/test_mfd_prefix_tree.py)
- [Algorithm contract and limits](../../../../docs/specs/mfd-prefix-tree.md)
- [Source attribution and upstream licenses](NOTICE.md)
