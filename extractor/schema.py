from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCALAR_FIELDS = [
    "doc_type",
    "supplier_name",
    "supplier_bin",
    "doc_number",
    "doc_date",
    "currency",
    "total_amount",
    "vat_amount",
]

Confidence = float


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LineItem(Strict):
    name: str = Field(min_length=1)
    qty: float | None
    price: float | None


class FieldConfidence(Strict):
    doc_type: Confidence = Field(ge=0, le=1)
    supplier_name: Confidence = Field(ge=0, le=1)
    supplier_bin: Confidence = Field(ge=0, le=1)
    doc_number: Confidence = Field(ge=0, le=1)
    doc_date: Confidence = Field(ge=0, le=1)
    currency: Confidence = Field(ge=0, le=1)
    total_amount: Confidence = Field(ge=0, le=1)
    vat_amount: Confidence = Field(ge=0, le=1)


class Extraction(Strict):
    doc_type: Literal["invoice", "commercial_offer", "act", "unknown"]
    supplier_name: str | None
    supplier_bin: str | None = Field(pattern=r"^\d{12}$")
    doc_number: str | None
    doc_date: date | None
    currency: Literal["KZT", "USD", "EUR"] | None
    total_amount: float | None = Field(ge=0)
    vat_amount: float | None = Field(ge=0)
    line_items: list[LineItem]
    field_confidence: FieldConfidence
    warnings: list[str]


def empty_extraction(warning: str) -> Extraction:
    return Extraction(
        doc_type="unknown",
        supplier_name=None,
        supplier_bin=None,
        doc_number=None,
        doc_date=None,
        currency=None,
        total_amount=None,
        vat_amount=None,
        line_items=[],
        field_confidence=FieldConfidence(**{field: 0.0 for field in SCALAR_FIELDS}),
        warnings=[warning],
    )
