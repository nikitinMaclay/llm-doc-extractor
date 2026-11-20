"""CLI: python -m extractor путь/к/документу.txt  (или текст через stdin)."""

import argparse
import os
import sys

from extractor.llm import LLMError, make_client
from extractor.pipeline import extract


def main() -> int:
    parser = argparse.ArgumentParser(prog="extractor", description="Текст документа → JSON по схеме.")
    parser.add_argument("file", nargs="?", help="файл с текстом документа; без него текст читается из stdin")
    parser.add_argument(
        "--llm",
        choices=["openai", "replay"],
        default="openai" if os.environ.get("OPENAI_API_KEY") else "replay",
        help="openai — настоящая модель; replay — записанные ответы (по умолчанию, если нет ключа)",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            document = handle.read()
    else:
        document = sys.stdin.read()

    try:
        result = extract(document, make_client(args.llm))
    except LLMError as error:
        print(error, file=sys.stderr)
        return 2

    print(result.extraction.model_dump_json(indent=2))  # в stdout — только JSON по схеме
    for error in result.errors:
        print(error, file=sys.stderr)
    return 0 if result.ok else 2  # код 2: результат пустой, документ нужно разобрать вручную


if __name__ == "__main__":
    sys.exit(main())
