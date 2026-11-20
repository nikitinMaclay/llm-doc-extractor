"""Проверки по тексту: значение, которого нет в документе, не проходит, даже если схема довольна."""

from extractor.checks import LOW_CONFIDENCE, apply_checks
from extractor.schema import Extraction
from tests.test_pipeline import DOCUMENT, VALID


def checked(document: str, **overrides) -> Extraction:
    return apply_checks(Extraction.model_validate({**VALID, **overrides}), document)


def test_grounded_answer_passes_unchanged():
    result = checked(DOCUMENT)

    assert result == Extraction.model_validate(VALID)


def test_bin_absent_from_document_is_removed():
    result = checked(DOCUMENT, supplier_bin="990540003217")

    assert result.supplier_bin is None
    assert result.field_confidence.supplier_bin <= LOW_CONFIDENCE
    assert "supplier_bin" in result.warnings[0]


def test_bin_repaired_from_ocr_garbage_is_removed():
    document = DOCUMENT.replace("170340021554", "2l064090Ol80")  # в тексте буквы вместо цифр

    assert checked(document, supplier_bin="210640900180").supplier_bin is None


def test_bin_inside_a_longer_number_does_not_count():
    document = DOCUMENT.replace("БИН 170340021554", "Счёт KZ9817034002155400")

    assert checked(document).supplier_bin is None


def test_computed_vat_is_removed():
    # 16% от 1000 = 160, но в документе суммы НДС нет
    result = checked(DOCUMENT, vat_amount=160)

    assert result.vat_amount is None
    assert "vat_amount" in result.warnings[0]


def test_vat_written_in_document_is_kept_in_any_number_format():
    assert checked(DOCUMENT + "\nНДС 16%: 25 680", vat_amount=25680).vat_amount == 25680
    assert checked(DOCUMENT + "\nНДС: 1 250,50", vat_amount=1250.5).vat_amount == 1250.5
    assert checked(DOCUMENT + "\nVAT: 1,250.00", vat_amount=1250).vat_amount == 1250


def test_todays_date_instead_of_missing_date_is_removed():
    document = DOCUMENT.replace(" от 12.11.2026", "")

    result = checked(document, doc_date="2026-11-20")

    assert result.doc_date is None


def test_date_with_month_written_as_word_is_kept():
    document = DOCUMENT.replace("12.11.2026", "9 қазан 2026 ж.")

    assert str(checked(document, doc_date="2026-10-09").doc_date) == "2026-10-09"


def test_currency_not_named_in_document_is_removed():
    document = DOCUMENT.replace(" тенге", "")

    assert checked(document).currency is None
    assert checked(document + "\nОплата в USD", currency="USD").currency == "USD"


def test_invented_number_and_supplier_are_removed():
    result = checked(DOCUMENT, doc_number="INV-2026-001", supplier_name="ТОО «Алматы Трейд»")

    assert result.doc_number is None
    assert result.supplier_name is None


def test_total_written_only_in_words_is_kept_but_flagged():
    document = DOCUMENT.replace("1 000 тенге", "Одна тысяча тенге 00 тиын")

    result = checked(document)

    assert result.total_amount == 1000  # не обнуляем
    assert result.field_confidence.total_amount <= LOW_CONFIDENCE
    assert "прописью" in result.warnings[0]


def test_total_that_does_not_match_line_items_is_flagged():
    items = [{"name": "Бумага", "qty": 2, "price": 400}]  # 800, а итог 1000

    mismatch = checked(DOCUMENT, line_items=items)
    with_vat_on_top = checked(DOCUMENT + "\nНДС: 200", line_items=items, vat_amount=200)
    no_quantities = checked(DOCUMENT, line_items=[{"name": "Услуга", "qty": None, "price": 400}])

    assert "не сходится" in mismatch.warnings[0]
    assert with_vat_on_top.warnings == []  # 800 + 200 = 1000
    assert no_quantities.warnings == []  # проверить нечем — молчим
