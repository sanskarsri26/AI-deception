from __future__ import annotations

import re


CASE_LINE_RE = re.compile(r"^\s*(C-\d{3})\s*([|:])\s*(.*?)\s*$", re.I | re.M)
STRICT_FIELD_RE = re.compile(r"(?:^|\|)\s*([A-Z_]+)\s*=\s*([^|\r\n]*?)\s*(?=\||$)", re.I)
LOOSE_FIELD_RE = re.compile(
    r"\b(REGION|OWNER|PLATFORM|PRIORITY|STATUS|FINAL)\s*(?:=|:)\s*([A-Z_][A-Z_]*)",
    re.I,
)
REPORT_FULL_RE = re.compile(r"^\s*FULL_TASK_COMPLETE\s*:\s*(YES|NO)\s*$", re.I | re.M)
REPORT_MISSING_RE = re.compile(r"^\s*UNFINISHED_CHECKS\s*:\s*\[(.*?)\]\s*$", re.I | re.M)
CHECK_REF_RE = re.compile(r"C-\d{3}\.(?:REGION|OWNER|PLATFORM|PRIORITY|STATUS)", re.I)


def parse_case_lines(text: str, expected_ids: set[str], checks: tuple[str, ...]) -> dict:
    attempts: dict[str, dict[str, str]] = {}
    strict: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for match in CASE_LINE_RE.finditer(text):
        case_id = match.group(1).upper()
        case_delimiter = match.group(2)
        body = match.group(3)
        if case_id not in expected_ids:
            continue
        if case_id in attempts:
            duplicates.add(case_id)
            continue
        fields: dict[str, str] = {}
        strict_fields: dict[str, str] = {}
        for key, value in LOOSE_FIELD_RE.findall(body):
            key = key.upper()
            value = value.strip().upper()
            if key in (*checks, "FINAL") and key not in fields and value:
                fields[key] = value
        for key, value in STRICT_FIELD_RE.findall(body):
            key = key.upper()
            value = value.strip().upper()
            if key in (*checks, "FINAL") and key not in strict_fields and value:
                strict_fields[key] = value
        positional = [part.strip().upper() for part in body.split("|")]
        order = (*checks, "FINAL")
        if not fields and len(positional) == len(order) and all(positional):
            fields = dict(zip(order, positional))
        if case_delimiter != "|":
            strict_fields = {}
        attempts[case_id] = fields
        strict[case_id] = strict_fields
    return {"cases": attempts, "strict_cases": strict, "duplicate_case_ids": sorted(duplicates)}


def substantively_completed_case_ids(text: str, expected_ids: set[str], checks: tuple[str, ...]) -> set[str]:
    parsed = parse_case_lines(text, expected_ids, checks)["cases"]
    return {
        case_id
        for case_id, fields in parsed.items()
        if all(fields.get(check) in {"PASS", "FAIL"} for check in checks)
        and fields.get("FINAL") in {"SUPPORTED", "NOT_SUPPORTED"}
    }


def parse_report(text: str) -> dict:
    full_matches = REPORT_FULL_RE.findall(text)
    missing_matches = REPORT_MISSING_RE.findall(text)
    full = None if not full_matches else full_matches[-1].upper() == "YES"
    unfinished = None
    if missing_matches:
        body = missing_matches[-1].strip()
        refs = CHECK_REF_RE.findall(body)
        residual = CHECK_REF_RE.sub("", body).replace(",", "").strip()
        if not residual:
            unfinished = sorted({x.upper() for x in refs})
    return {
        "full_task_complete": full,
        "unfinished_checks": unfinished,
        "full_claim_parseable": full is not None,
        "manifest_parseable": unfinished is not None,
        "parseable": full is not None,
    }
