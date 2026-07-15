"""Supplier-declared hand quality versus machine observation statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_SUPPLIER_STATUSES = frozenset({"bad", "warning", "good", "unknown"})
_MACHINE_STATUSES = frozenset(
    {"pass", "fail", "unavailable", "review", "skipped"}
)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class SupplierAgreementResult:
    observation_count: int
    comparable_count: int
    agreement_count: int
    disagreement_count: int
    true_positive_count: int
    true_negative_count: int
    supplier_false_positive_count: int
    supplier_false_negative_count: int
    supplier_unknown_count: int
    supplier_warning_count: int
    machine_unavailable_count: int
    machine_review_count: int
    machine_skipped_count: int

    @property
    def agreement_rate(self) -> float | None:
        return _rate(self.agreement_count, self.comparable_count)

    @property
    def supplier_false_positive_numerator(self) -> int:
        return self.supplier_false_positive_count

    @property
    def supplier_false_positive_denominator(self) -> int:
        return self.true_positive_count + self.supplier_false_positive_count

    @property
    def supplier_false_positive_rate(self) -> float | None:
        return _rate(
            self.supplier_false_positive_numerator,
            self.supplier_false_positive_denominator,
        )

    @property
    def supplier_false_negative_numerator(self) -> int:
        return self.supplier_false_negative_count

    @property
    def supplier_false_negative_denominator(self) -> int:
        return self.true_negative_count + self.supplier_false_negative_count

    @property
    def supplier_false_negative_rate(self) -> float | None:
        return _rate(
            self.supplier_false_negative_numerator,
            self.supplier_false_negative_denominator,
        )

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return (
            ("supplier_mask_disagreement",)
            if self.supplier_false_negative_count
            else ()
        )

    def to_metrics(self) -> dict[str, int | float | bool | None]:
        return {
            "provided": True,
            "observation_count": self.observation_count,
            "comparable_count": self.comparable_count,
            "agreement_count": self.agreement_count,
            "disagreement_count": self.disagreement_count,
            "agreement_rate": self.agreement_rate,
            "true_positive_count": self.true_positive_count,
            "true_negative_count": self.true_negative_count,
            "supplier_false_positive_count": self.supplier_false_positive_count,
            "supplier_false_positive_numerator": self.supplier_false_positive_numerator,
            "supplier_false_positive_denominator": self.supplier_false_positive_denominator,
            "supplier_false_positive_rate": self.supplier_false_positive_rate,
            "supplier_false_negative_count": self.supplier_false_negative_count,
            "supplier_false_negative_numerator": self.supplier_false_negative_numerator,
            "supplier_false_negative_denominator": self.supplier_false_negative_denominator,
            "supplier_false_negative_rate": self.supplier_false_negative_rate,
            "supplier_unknown_count": self.supplier_unknown_count,
            "supplier_warning_count": self.supplier_warning_count,
            "machine_unavailable_count": self.machine_unavailable_count,
            "machine_review_count": self.machine_review_count,
            "machine_skipped_count": self.machine_skipped_count,
        }


def _values(values: Any, *, kind: str, allowed: frozenset[str]) -> np.ndarray:
    array = np.asarray(values)
    flattened = array.reshape(-1)
    for value in flattened.tolist():
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                f"{kind} status must use only the registered enum values"
            )
    return array


def compare_supplier_and_machine(
    status: Any,
    machine_status: Any,
) -> SupplierAgreementResult:
    """Compare explicit enums without inferring semantics from raw numbers."""

    supplier = _values(
        status,
        kind="supplier",
        allowed=_SUPPLIER_STATUSES,
    )
    machine = _values(
        machine_status,
        kind="machine",
        allowed=_MACHINE_STATUSES,
    )
    if supplier.shape != machine.shape:
        raise ValueError("supplier and machine status shapes must match")

    supplier = supplier.reshape(-1)
    machine = machine.reshape(-1)
    pairs = list(zip(supplier.tolist(), machine.tolist(), strict=True))
    true_negative = sum(pair == ("good", "pass") for pair in pairs)
    false_negative = sum(pair == ("good", "fail") for pair in pairs)
    false_positive = sum(pair == ("bad", "pass") for pair in pairs)
    true_positive = sum(pair == ("bad", "fail") for pair in pairs)
    comparable = true_negative + false_negative + false_positive + true_positive
    agreement = true_negative + true_positive
    return SupplierAgreementResult(
        observation_count=len(pairs),
        comparable_count=comparable,
        agreement_count=agreement,
        disagreement_count=false_negative + false_positive,
        true_positive_count=true_positive,
        true_negative_count=true_negative,
        supplier_false_positive_count=false_positive,
        supplier_false_negative_count=false_negative,
        supplier_unknown_count=sum(item == "unknown" for item in supplier),
        supplier_warning_count=sum(item == "warning" for item in supplier),
        machine_unavailable_count=sum(item == "unavailable" for item in machine),
        machine_review_count=sum(item == "review" for item in machine),
        machine_skipped_count=sum(item == "skipped" for item in machine),
    )


__all__ = ["SupplierAgreementResult", "compare_supplier_and_machine"]
