from pathlib import Path
import json
import random

OUT = Path("data/main_pool_BC_v06.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

templates = {
    "duplicate_charge": [
        "I paid once but the transaction appears twice on my account.",
        "There are two charges for one order.",
        "My card was billed two times for a single purchase.",
        "The same subscription payment appears twice.",
        "Checkout happened once, but I have duplicate charges.",
    ],
    "app_crash": [
        "The {app} shuts down whenever I upload an image.",
        "Opening settings causes the {app} to crash.",
        "The application closes unexpectedly when I attach a file.",
        "The {app} freezes and exits when I open my profile.",
        "The program crashes immediately after login.",
    ],
    "package_not_received": [
        "The order is marked delivered, but nothing arrived.",
        "Tracking shows delivery even though I never received the parcel.",
        "The carrier says my package was delivered, but it is missing.",
        "I got a delivery notification but there is no package.",
        "The shipment status says delivered, but I cannot locate it.",
    ],
    "password_reset_missing": [
        "I requested a reset link but no email arrived.",
        "The password recovery email never reached me.",
        "I forgot my password and the reset message is missing.",
        "The website says it sent a password reset email, but I received nothing.",
        "I tried resetting my password several times without receiving an email.",
    ],
    "refund_pending": [
        "I returned the {item} {days} days ago and still have no refund.",
        "My cancellation was accepted, but my money has not been returned.",
        "The return was completed but the refund is still missing.",
        "I cancelled the purchase and am waiting for the payment to come back.",
        "The {item} was sent back, but I still have not received the refund.",
    ],
    "unexpected_fee": [
        "My final bill contains a {amount} fee that was not shown earlier.",
        "An extra service charge appeared after checkout.",
        "The receipt contains an unexplained {amount} fee.",
        "A processing charge was added that I was not told about.",
        "My total includes an unexpected additional fee.",
    ],
    "verification_phone_old": [
        "The login verification code goes to my old phone number.",
        "I cannot access the security code because it is sent to a number I no longer use.",
        "Two-factor authentication is connected to my previous phone.",
        "My verification code keeps going to an outdated number.",
        "I cannot sign in because account verification uses my old phone number.",
    ],
    "tracking_stalled": [
        "Tracking has not changed for {days} days and the package is late.",
        "My shipment has shown the same tracking status for {days} days.",
        "The expected delivery date passed and tracking has stopped updating.",
        "My package is still in transit with no new carrier scans.",
        "There have been no tracking updates for {days} days.",
    ],
    "damaged_item": [
        "The {item} was damaged when it arrived.",
        "I opened the box and found the {item} broken.",
        "The product arrived cracked.",
        "My order was damaged inside the package.",
        "The {item} arrived broken.",
    ],
    "payment_declined": [
        "My valid card keeps being declined at checkout.",
        "The payment fails even though my bank says the card is fine.",
        "Checkout rejects my card despite having enough funds.",
        "My active card is not being accepted.",
        "Every payment attempt is declined even though the card works elsewhere.",
    ],
}

apps = [
    "mobile app", "desktop app", "shopping app",
    "account app", "support app"
]

items = [
    "headphones", "monitor", "keyboard", "tablet",
    "charger", "phone", "speaker", "mouse"
]

days = [
    "two", "three", "four", "five",
    "six", "eight", "ten", "twelve"
]

amounts = [
    "$9.99", "$12", "$18", "$25",
    "$31", "$48", "$55", "$75", "$120"
]

suffixes = [
    "",
    " The issue is still unresolved.",
    " I tried again and the same thing happened.",
    " This happened again today.",
    " I still need help fixing this.",
    " The problem has continued.",
    " I checked again later with the same result.",
    " Nothing has changed since I first noticed it.",
]

rows = []

for version, seed in [("B", 60601), ("C", 60602)]:
    rng = random.Random(seed)
    version_rows = []
    idx = 1

    for label, choices in templates.items():
        for i in range(40):
            template = choices[i % len(choices)]

            text = template.format(
                app=rng.choice(apps),
                item=rng.choice(items),
                days=rng.choice(days),
                amount=rng.choice(amounts),
            )

            # Different deterministic wording variation for B/C.
            suffix_index = (
                (i // len(choices))
                + (0 if version == "B" else 3)
            ) % len(suffixes)

            text += suffixes[suffix_index]

            difficulty = (
                "easy" if i < 14
                else "medium" if i < 28
                else "hard"
            )

            version_rows.append({
                "item_id": f"{version}-{idx:03d}",
                "version": version,
                "match_index": idx,
                "label": label,
                "difficulty": difficulty,
                "text": text,
            })

            idx += 1

    rng.shuffle(version_rows)
    rows.extend(version_rows)

with OUT.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print("Created:", OUT)
print("Total:", len(rows))

for version in ["B", "C"]:
    subset = [x for x in rows if x["version"] == version]
    print(version, len(subset))
