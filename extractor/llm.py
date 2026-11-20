import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from extractor.prompt import SYSTEM_PROMPT
from extractor.schema import Extraction

DEFAULT_MODEL = "gpt-6-luna"
RESPONSES_DIR = Path(__file__).parent.parent / "data" / "responses"


class LLMError(Exception):
    ...


class LLMClient(Protocol):
    def complete(self, document: str, feedback: str | None = None) -> str:
        ...


def response_path(document: str) -> Path:
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()[:16]
    return RESPONSES_DIR / f"{digest}.json"


class OpenAIClient:
    def __init__(self, model: str | None = None, record: bool = False):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.record = record

    def complete(self, document: str, feedback: str | None = None) -> str:
        from openai import OpenAIError

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<document>\n{document}\n</document>"},
        ]
        if feedback:
            messages.append(
                {"role": "user", "content": f"Твой прошлый ответ не прошёл проверку:\n{feedback}\nИсправь и верни JSON заново."}
            )
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                reasoning={"effort": "low"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "document_extraction",
                        "schema": Extraction.model_json_schema(),
                        "strict": True,
                    }
                },
            )
        except OpenAIError as error:
            raise LLMError(f"Ошибка OpenAI: {error}") from error

        raw = response.output_text
        if not raw:
            raise LLMError("Модель вернула пустой ответ (отказ или обрыв по лимиту).")
        if self.record:
            self._save(document, raw, first_attempt=feedback is None)
        return raw

    def _save(self, document: str, raw: str, first_attempt: bool) -> None:
        path = response_path(document)
        responses = [] if first_attempt else json.loads(path.read_text("utf-8"))["responses"]
        responses.append(raw)
        RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        saved = {
            "source": f"openai:{self.model}",
            "first_line": document.strip().splitlines()[0],
            "responses": responses,
        }
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ReplayClient:
    def __init__(self):
        self.attempts: dict[Path, int] = {}
        self.sources: set[str] = set()

    def complete(self, document: str, feedback: str | None = None) -> str:
        path = response_path(document)
        if not path.exists():
            raise LLMError(
                "Для этого документа нет сохранённого ответа. "
                "Задайте OPENAI_API_KEY и запустите с --llm openai."
            )
        saved = json.loads(path.read_text("utf-8"))
        self.sources.add(saved["source"])
        responses = saved["responses"]
        attempt = self.attempts.get(path, 0)
        self.attempts[path] = attempt + 1
        return responses[min(attempt, len(responses) - 1)]


def make_client(mode: str, record: bool = False) -> LLMClient:
    if mode == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("Не задана переменная окружения OPENAI_API_KEY.")
        return OpenAIClient(record=record)
    if mode == "replay":
        return ReplayClient()
    raise ValueError(f"Неизвестный режим LLM: {mode}")
