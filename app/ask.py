"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 会話履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。単発モードでは会話履歴を持たず、
--interactive モードでは質問と回答の履歴を保って続けて質問できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(
    question: str,
    history: list[types.Content] | None = None,
    client: genai.Client | None = None,
) -> str:
    if client is None:
        client = genai.Client(api_key=get_api_key())

    user_message = types.Content(
        role="user", parts=[types.Part.from_text(text=question)]
    )
    contents = [*(history or []), user_message]
    response = client.models.generate_content(model=MODEL, contents=contents)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[tokens] input={usage.prompt_token_count}, "
            f"output={usage.candidates_token_count}"
        )
    else:
        print("[tokens] usage metadata unavailable")

    if response.text is None:
        raise ValueError("モデルの応答にテキストがありません")

    if history is not None:
        history.extend(
            [
                user_message,
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=response.text)],
                ),
            ]
        )

    return response.text


def run_interactive(initial_question: str | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    history: list[types.Content] = []

    if initial_question:
        print(f"[質問] {initial_question}\n")
        try:
            print(ask(initial_question, history, client))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))

    print("\n質問を入力してください（終了: /exit または Ctrl+C）。")
    while True:
        try:
            question = input("\n[質問] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            return

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("終了します。")
            return
        if not question:
            continue

        try:
            print(ask(question, history, client))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", help="質問（省略時は既定の質問）")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="会話履歴を保って続けて質問する",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.interactive:
        run_interactive(args.question)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
