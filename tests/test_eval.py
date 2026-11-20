"""Метрики eval: галлюцинация и пропуск — разные исходы, и набор данных целый."""

import json

from eval import CORRECT, DATA, FIELDS, HALLUCINATION, OMISSION, WRONG, compare
from extractor.checks import apply_checks
from extractor.schema import Extraction


def test_hallucination_and_omission_are_counted_separately():
    assert compare("supplier_bin", "170340021554", None) == HALLUCINATION
    assert compare("supplier_bin", None, "170340021554") == OMISSION
    assert compare("supplier_bin", "170340021555", "170340021554") == WRONG
    assert compare("supplier_bin", None, None) == CORRECT  # правильный null — это верный ответ


def test_zero_vat_is_a_value_not_a_null():
    assert compare("vat_amount", 0, None) == HALLUCINATION
    assert compare("vat_amount", None, 0) == OMISSION


def test_supplier_name_ignores_quotes_and_case_but_not_words():
    assert compare("supplier_name", "тоо «Алматы Трейд Компани»", 'ТОО "Алматы Трейд Компани"') == CORRECT
    assert compare("supplier_name", "ТОО Алматы Трейд", 'ТОО "Алматы Трейд Компани"') == WRONG


def test_line_items_compare_count_numbers_and_names():
    expected = [{"name": "Бумага А4", "qty": 50, "price": 2100}]

    assert compare("line_items", [{"name": "бумага а4", "qty": 50.0, "price": 2100.0}], expected) == CORRECT
    assert compare("line_items", [{"name": "Бумага А4", "qty": 50, "price": 2000}], expected) == WRONG
    assert compare("line_items", expected * 2, expected) == WRONG  # лишняя позиция
    assert compare("line_items", [], expected) == OMISSION
    assert compare("line_items", expected, []) == HALLUCINATION


def test_every_document_has_a_gold_file_that_passes_the_systems_own_checks():
    """Эталон обязан сам проходить схему и проверки по тексту — иначе eval требует невозможного."""
    doc_paths = sorted((DATA / "docs").glob("*.txt"))
    assert len(doc_paths) >= 10

    for doc_path in doc_paths:
        gold = json.loads((DATA / "gold" / f"{doc_path.stem}.json").read_text("utf-8"))
        confidence = {field: 1.0 for field in FIELDS if field != "line_items"}
        extraction = Extraction.model_validate({**gold, "warnings": [], "field_confidence": confidence})

        checked = apply_checks(extraction, doc_path.read_text("utf-8"))

        for field in FIELDS:
            assert getattr(checked, field) == getattr(extraction, field), f"{doc_path.stem}: {field}"
