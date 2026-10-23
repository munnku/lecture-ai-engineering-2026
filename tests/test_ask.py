from types import SimpleNamespace

from google.genai import types

from app.ask import ask


class FakeClient:
    def __init__(self):
        self.responses = iter(["最初の回答", "続きの回答"])
        self.requests = []
        self.models = self

    def generate_content(self, *, model, contents):
        self.requests.append((model, contents))
        return SimpleNamespace(
            text=next(self.responses),
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5
            ),
        )


def test_ask_preserves_conversation_history(capsys):
    client = FakeClient()
    history: list[types.Content] = []

    assert ask("最初の質問", history, client) == "最初の回答"
    assert ask("続きの質問", history, client) == "続きの回答"

    first_contents = client.requests[0][1]
    second_contents = client.requests[1][1]
    assert [message.role for message in first_contents] == ["user"]
    assert [message.role for message in second_contents] == [
        "user",
        "model",
        "user",
    ]
    assert first_contents[0].parts[0].text == "最初の質問"
    assert second_contents[1].parts[0].text == "最初の回答"
    assert second_contents[2].parts[0].text == "続きの質問"
    assert [message.role for message in history] == ["user", "model", "user", "model"]
    assert capsys.readouterr().out.count("[tokens] input=10, output=5") == 2
