"""Мини-eval: прогоняет все документы из data/docs и сравнивает результат с эталонами из data/gold.

    python eval.py                 # записанные ответы модели, ключ не нужен
    python eval.py --llm openai    # настоящая модель (нужен OPENAI_API_KEY)
"""

import argparse
import json
import re
import sys
from pathlib import Path

from extractor.llm import LLMError, make_client
from extractor.pipeline import extract
from extractor.schema import SCALAR_FIELDS

DATA = Path(__file__).parent / "data"
FIELDS = SCALAR_FIELDS + ["line_items"]

# Четыре исхода сравнения одного поля. Галлюцинация и пропуск считаются раздельно:
# выдуманный БИН уходит в оплату, а пропущенный всего лишь отправляет документ человеку.
CORRECT = "верно"
HALLUCINATION = "галлюцинация"  # поле заполнено, хотя в эталоне null
OMISSION = "пропуск"  # null, хотя в эталоне есть значение
WRONG = "неверное значение"  # заполнено и там и там, но по-разному


def compare(field: str, predicted, expected) -> str:
    predicted_empty = predicted in (None, [])
    expected_empty = expected in (None, [])
    if predicted_empty and expected_empty:
        return CORRECT
    if expected_empty:
        return HALLUCINATION
    if predicted_empty:
        return OMISSION
    return CORRECT if _same(field, predicted, expected) else WRONG


def _same(field: str, predicted, expected) -> bool:
    if field == "line_items":
        return len(predicted) == len(expected) and all(
            _same_number(p["qty"], e["qty"])
            and _same_number(p["price"], e["price"])
            and _same_item_name(p["name"], e["name"])
            for p, e in zip(predicted, expected)
        )
    if field in ("total_amount", "vat_amount"):
        return _same_number(predicted, expected)
    if field == "supplier_name":
        return _normalize(predicted) == _normalize(expected)
    return predicted == expected


def _same_number(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < 0.005


def _normalize(text: str) -> str:
    """Без регистра, кавычек и лишних пробелов; ё = е."""
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[«»\"“”„']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _same_item_name(predicted: str, expected: str) -> bool:
    """Название позиции совпадает, если одно содержит другое («Услуги…» и «За услуги…»)."""
    a, b = _normalize(predicted), _normalize(expected)
    return a in b or b in a


def run(llm_mode: str, record: bool) -> int:
    llm = make_client(llm_mode, record=record)
    outcomes = {field: [] for field in FIELDS}  # поле → список исходов по документам
    mistakes: list[str] = []
    failed: list[str] = []
    warned_when_needed = warnings_needed = false_alarms = 0

    doc_paths = sorted((DATA / "docs").glob("*.txt"))
    for doc_path in doc_paths:
        expected = json.loads((DATA / "gold" / f"{doc_path.stem}.json").read_text("utf-8"))
        result = extract(doc_path.read_text("utf-8"), llm)
        predicted = result.extraction.model_dump(mode="json")
        if not result.ok:
            failed.append(f"{doc_path.stem}: {'; '.join(result.errors)}")

        for field in FIELDS:
            outcome = compare(field, predicted[field], expected[field])
            outcomes[field].append(outcome)
            if outcome != CORRECT:
                mistakes.append(
                    f"  {doc_path.stem} · {field} · {outcome}: "
                    f"получено {json.dumps(predicted[field], ensure_ascii=False)}, "
                    f"эталон {json.dumps(expected[field], ensure_ascii=False)}"
                )

        if expected["warnings"]:
            warnings_needed += 1
            warned_when_needed += bool(predicted["warnings"])
        else:
            false_alarms += bool(predicted["warnings"])

    total_docs = len(doc_paths)
    print(f"Документов: {total_docs} · режим LLM: {llm_mode} · не прошли валидацию: {len(failed)}")
    for line in failed:
        print(f"  {line}")

    print(f"\n{'Поле':<15}{'Точность':>14}{'Галлюцинации':>14}{'Пропуски':>10}{'Неверные':>10}")
    for field in FIELDS + ["ВСЕГО"]:
        results = sum(outcomes.values(), []) if field == "ВСЕГО" else outcomes[field]
        correct = results.count(CORRECT)
        accuracy = f"{correct}/{len(results)} ({correct / len(results):.0%})"
        print(
            f"{field:<15}{accuracy:>14}{results.count(HALLUCINATION):>14}"
            f"{results.count(OMISSION):>10}{results.count(WRONG):>10}"
        )

    print(
        f"\nПредупреждения: нужны в {warnings_needed} документах, выданы в {warned_when_needed}; "
        f"лишние — в {false_alarms} из {total_docs - warnings_needed} документов без проблем."
    )
    print("\nОшибки по полям:" if mistakes else "\nОшибок по полям нет.")
    for line in mistakes:
        print(line)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прогон тестового набора и отчёт по качеству.")
    parser.add_argument("--llm", choices=["replay", "openai"], default="replay")
    parser.add_argument("--record", action="store_true", help="сохранить ответы модели в data/recorded (с --llm openai)")
    args = parser.parse_args()
    try:
        sys.exit(run(args.llm, args.record))
    except LLMError as error:
        sys.exit(str(error))
