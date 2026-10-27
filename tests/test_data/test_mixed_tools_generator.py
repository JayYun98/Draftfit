import json

from datasets import Dataset

from draftfit.utils import safe_conversations_generator


def test_mixed_tool_rows_have_one_arrow_schema(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"conversations": [{"role": "assistant", "content": "hello"}]},
                {
                    "conversations": [{"role": "assistant", "content": "done"}],
                    "tools": [{"name": "lookup"}],
                },
            )
        )
        + "\n"
    )
    dataset = Dataset.from_generator(
        safe_conversations_generator,
        gen_kwargs={"file_path": path},
        keep_in_memory=True,
    )
    assert [json.loads(value) for value in dataset["tools"]] == [
        [],
        [{"name": "lookup"}],
    ]
