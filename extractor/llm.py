"""Вызов языковой модели. Конвейеру нужен только метод complete(document, feedback) -> str."""

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from extractor.prompt import SYSTEM_PROMPT
from extractor.schema import Extraction

DEFAULT_MODEL = "gpt-6-luna"
RECORDED_DIR = Path(__file__).parent.parent / "data" / "recorded"


class LLMError(Exception):
    """Модель недоступна или не вернула текст."""


class LLMClient(Protocol):
    def complete(self, document: str, feedback: str | None = None) -> str:
        """Вернуть сырой ответ модели. feedback — текст ошибки валидации прошлой попытки."""


def recording_path(document: str) -> Path:
    """Файл записи определяется содержимым документа: изменился текст — нужна новая запись."""
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()[:16]
    return RECORDED_DIR / f"{digest}.json"


class OpenAIClient:
    """Настоящий вызов OpenAI со Structured Outputs. С record=True сохраняет ответы для ReplayClient."""

    def __init__(self, model: str | None = None, record: bool = False):
        from openai import OpenAI  # импорт здесь: режим replay работает и без пакета openai

        self.client = OpenAI()  # ключ берётся из переменной окружения OPENAI_API_KEY
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
                # strict: модель может выдать только JSON, подходящий под схему.
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
        path = recording_path(document)
        responses = [] if first_attempt else json.loads(path.read_text("utf-8"))["responses"]
        responses.append(raw)
        RECORDED_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": self.model, "responses": responses}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ReplayClient:
    """Замена вызова LLM: отдаёт сохранённые ответы настоящей модели. Работает без ключа и сети."""

    def __init__(self):
        self.attempts: dict[Path, int] = {}  # сколько раз уже спрашивали по каждому документу

    def complete(self, document: str, feedback: str | None = None) -> str:
        path = recording_path(document)
        if not path.exists():
            raise LLMError(
                "Для этого документа нет записанного ответа модели. "
                "Задайте OPENAI_API_KEY и запустите с --llm openai."
            )
        responses = json.loads(path.read_text("utf-8"))["responses"]
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
