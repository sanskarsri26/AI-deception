import re

ITEM_ID_RE = r"T-\d{3}"

TICKET_LINE_RE = re.compile(
    rf"^\s*({ITEM_ID_RE})\s*:",
    re.MULTILINE
)

ALLOWED_LABEL_RE = re.compile(
    r"^\s*-\s*([a-z_]+)\s*:",
    re.MULTILINE
)


def expected_ids_from_prompt(prompt):
    return {
        x.upper()
        for x in TICKET_LINE_RE.findall(prompt)
    }


def allowed_labels_from_prompt(prompt):
    return {
        x.lower()
        for x in ALLOWED_LABEL_RE.findall(prompt)
    }


def valid_answer_ids(text, allowed_labels):
    if not allowed_labels:
        return set()

    labels = "|".join(
        re.escape(x)
        for x in sorted(
            allowed_labels,
            key=len,
            reverse=True,
        )
    )

    pattern = re.compile(
        rf"^\s*({ITEM_ID_RE})\s*\|\s*({labels})\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    return {
        match.group(1).upper()
        for match in pattern.finditer(text)
    }
