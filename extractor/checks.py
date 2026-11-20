"""Проверки ответа модели по тексту документа — обычный код, без LLM.

Схема гарантирует форму ответа, но не то, что значение взято из документа. Здесь каждое поле
сверяется с текстом: чего в тексте нет — обнуляется (или помечается) с предупреждением.
"""

import re

from extractor.schema import Extraction

LOW_CONFIDENCE = 0.3

CURRENCY_MARKERS = {
    "KZT": r"тенге|теңге|тг\b|kzt|₸",
    "USD": r"usd|\$|долл",
    "EUR": r"eur|€|евро",
}


def apply_checks(extraction: Extraction, document: str) -> Extraction:
    """Вернуть копию результата, в которой не подтверждённые текстом значения убраны или помечены."""
    result = extraction.model_copy(deep=True)
    text = document.casefold()
    digits_text = _join_digit_groups(text)

    def reject(field: str, reason: str) -> None:
        value = getattr(result, field)
        setattr(result, field, None)
        _flag(result, field, f"{field}: значение «{value}» отклонено — {reason}.")

    if result.supplier_bin and not re.search(rf"(?<!\d){result.supplier_bin}(?!\d)", text):
        reject("supplier_bin", "таких 12 цифр подряд в документе нет")

    if result.doc_number and _squash(result.doc_number) not in _squash(text):
        reject("doc_number", "такого номера в документе нет")

    if result.doc_date:
        day, year = result.doc_date.day, result.doc_date.year
        # Месяц не проверяем: он бывает записан словом на двух языках.
        if str(year) not in text or not re.search(rf"(?<!\d)0?{day}(?!\d)", text):
            reject("doc_date", "числа и года этой даты в документе нет")

    if result.supplier_name:
        words = re.findall(r"\w{4,}", result.supplier_name.casefold())
        if not any(word in text for word in words):
            reject("supplier_name", "такого названия в документе нет")

    if result.currency and not re.search(CURRENCY_MARKERS[result.currency], text):
        reject("currency", "валюта в документе не названа")

    if result.vat_amount is not None and not _amount_in_text(result.vat_amount, digits_text):
        reject("vat_amount", "такой суммы в документе нет, НДС нельзя вычислять")

    if result.total_amount is not None and not _amount_in_text(result.total_amount, digits_text):
        # Не обнуляем: сумма может быть написана только прописью. Но доверять ей меньше.
        _flag(
            result,
            "total_amount",
            f"total_amount: суммы {result.total_amount:g} цифрами в документе нет "
            "(возможно, она написана прописью) — проверьте вручную.",
        )

    _check_arithmetic(result)
    return result


def _flag(result: Extraction, field: str, warning: str) -> None:
    result.warnings.append(warning)
    confidence = getattr(result.field_confidence, field)
    setattr(result.field_confidence, field, min(confidence, LOW_CONFIDENCE))


def _check_arithmetic(result: Extraction) -> None:
    """Если у всех позиций есть количество и цена, итог должен сходиться с их суммой."""
    items = result.line_items
    if result.total_amount is None or not items:
        return
    if any(item.qty is None or item.price is None for item in items):
        return
    items_sum = sum(item.qty * item.price for item in items)
    allowed = [items_sum, items_sum + (result.vat_amount or 0)]  # НДС в цене или сверху
    if not any(abs(result.total_amount - value) < 1 for value in allowed):
        _flag(
            result,
            "total_amount",
            f"total_amount: итог {result.total_amount:g} не сходится с суммой позиций {items_sum:g}.",
        )


def _squash(value: str) -> str:
    """Убрать пробелы и регистр: «AKT - 09» и «akt-09» считаются одним номером."""
    return re.sub(r"\s+", "", value.casefold())


def _join_digit_groups(text: str) -> str:
    """«186 180» → «186180», «1,250.00» → «1250.00»: убрать разделители тысяч между цифрами."""
    text = re.sub(r"(?<=\d)[   ](?=\d)", "", text)
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def _amount_in_text(amount: float, digits_text: str) -> bool:
    """Есть ли число в тексте в любой обычной записи: 12400.5 → «12400.5», «12400,50» и т.д."""
    if amount == int(amount):
        forms = [str(int(amount))]
    else:
        forms = [f"{amount:.2f}", f"{amount:.2f}".replace(".", ","), f"{amount:g}", f"{amount:g}".replace(".", ",")]
    return any(re.search(rf"(?<!\d){re.escape(form)}(?!\d)", digits_text) for form in forms)
