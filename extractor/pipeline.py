"""Конвейер: вызов модели → программная валидация → повтор при ошибке → проверки по тексту."""

from dataclasses import dataclass, field

from pydantic import ValidationError

from extractor.checks import apply_checks
from extractor.llm import LLMClient, LLMError
from extractor.schema import Extraction, empty_extraction

MAX_ATTEMPTS = 2


@dataclass
class Result:
    extraction: Extraction  # всегда валидный объект схемы, даже при неудаче
    ok: bool  # False — ответ модели принять не удалось, extraction пустой
    attempts: int
    errors: list[str] = field(default_factory=list)  # что пошло не так на каждой попытке


def extract(document: str, llm: LLMClient) -> Result:
    errors: list[str] = []
    feedback = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = llm.complete(document, feedback)
            extraction = Extraction.model_validate_json(raw)
        except LLMError as error:
            errors.append(f"попытка {attempt}: {error}")
            break  # модель недоступна — повтор с подсказкой не поможет
        except ValidationError as error:
            # Коротко и без входных данных: это уйдёт модели как подсказка для второй попытки.
            feedback = "; ".join(
                f"{'.'.join(str(part) for part in problem['loc']) or 'ответ'}: {problem['msg']}"
                for problem in error.errors()
            )
            errors.append(f"попытка {attempt}: ответ не прошёл валидацию — {feedback}")
            continue
        return Result(apply_checks(extraction, document), ok=True, attempts=attempt, errors=errors)

    # Закрытый отказ: невалидный ответ дальше не уходит, вместо него — пустой результат с причиной.
    warning = "Автоматический разбор не удался, нужен ручной разбор. " + " | ".join(errors)
    return Result(empty_extraction(warning), ok=False, attempts=len(errors), errors=errors)
