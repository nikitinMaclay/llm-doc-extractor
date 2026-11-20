import json

from extractor.llm import LLMError
from extractor.pipeline import extract

DOCUMENT = 'ТОО "Ромашка"\nБИН 170340021554\nСчёт № 7 от 12.11.2026\nВсего к оплате: 1 000 тенге'

VALID = {
    "doc_type": "invoice",
    "supplier_name": 'ТОО "Ромашка"',
    "supplier_bin": "170340021554",
    "doc_number": "7",
    "doc_date": "2026-11-12",
    "currency": "KZT",
    "total_amount": 1000,
    "vat_amount": None,
    "line_items": [],
    "field_confidence": {
        "doc_type": 0.9, "supplier_name": 0.9, "supplier_bin": 0.9, "doc_number": 0.9,
        "doc_date": 0.9, "currency": 0.9, "total_amount": 0.9, "vat_amount": 0.9,
    },
    "warnings": [],
}


class ScriptedLLM:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.feedbacks = []

    def complete(self, document, feedback=None):
        self.feedbacks.append(feedback)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


def test_valid_answer_is_accepted_on_first_attempt():
    result = extract(DOCUMENT, ScriptedLLM(VALID))

    assert result.ok and result.attempts == 1
    assert result.extraction.supplier_bin == "170340021554"
    assert result.extraction.warnings == []


def test_invalid_answer_is_retried_with_the_validation_error_as_feedback():
    bad = {**VALID, "supplier_bin": "17034002", "doc_date": "12.11.2026"}
    llm = ScriptedLLM(bad, VALID)

    result = extract(DOCUMENT, llm)

    assert result.ok and result.attempts == 2
    assert llm.feedbacks[0] is None
    assert "supplier_bin" in llm.feedbacks[1] and "doc_date" in llm.feedbacks[1]
    assert result.extraction.supplier_bin == "170340021554"


def test_two_invalid_answers_give_empty_result_not_garbage():
    result = extract(DOCUMENT, ScriptedLLM("Конечно! Вот ваш JSON: {...}", {**VALID, "currency": "RUB"}))

    assert not result.ok and result.attempts == 2
    empty = result.extraction
    assert empty.doc_type == "unknown"
    assert empty.supplier_bin is None and empty.total_amount is None and empty.line_items == []
    assert "ручной разбор" in empty.warnings[0]
    assert len(result.errors) == 2


def test_missing_or_extra_field_is_a_validation_error():
    missing = {key: value for key, value in VALID.items() if key != "vat_amount"}
    extra = {**VALID, "buyer_name": "ТОО Покупатель"}

    result = extract(DOCUMENT, ScriptedLLM(missing, extra))

    assert not result.ok
    assert "vat_amount" in result.errors[0] and "buyer_name" in result.errors[1]


def test_llm_failure_is_not_retried_and_gives_empty_result():
    llm = ScriptedLLM(LLMError("сеть недоступна"), VALID)

    result = extract(DOCUMENT, llm)

    assert not result.ok and result.attempts == 1
    assert len(llm.answers) == 1
    assert "сеть недоступна" in result.extraction.warnings[0]


def test_schema_sent_to_the_model_meets_strict_mode_requirements():
    from extractor.schema import Extraction

    schema = Extraction.model_json_schema()
    objects = [schema, *schema["$defs"].values()]

    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
