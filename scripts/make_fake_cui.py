"""Write 10,000 synthetic demo records to data/fake_cui.jsonl.

The output path is anchored to this file, so the script works from the repo
root. Requires the faker package, which is not needed to run the RAG.
"""

import json
import random
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fake_cui.jsonl"
SEED = 20260925
COUNT = 10000


def main() -> None:
    random.seed(SEED)
    Faker.seed(SEED)
    fake = Faker()
    templates = [
        "OPERATION {name}: {objective}. Assets: {assets}. Threat: {threat}.",
        "INTEL: {source} reports {event} in {location}. Confidence {conf}.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for i in range(COUNT):
            doc = {
                "id": f"CUI-{i:05d}",
                "text": random.choice(templates).format(
                    name=fake.word().upper() + str(random.randint(100, 999)),
                    objective=random.choice(["Secure perimeter", "Extract VIP"]),
                    assets=", ".join(fake.company() for _ in range(2)),
                    threat=random.choice(["LOW", "MEDIUM", "HIGH"]),
                    source=fake.name(),
                    event=fake.sentence().rstrip("."),
                    location=fake.city(),
                    conf=random.choice(["60%", "85%", "95%"]),
                ),
            }
            handle.write(json.dumps(doc) + "\n")
    print(f"{COUNT} fake CUI docs → {OUT}")


if __name__ == "__main__":
    main()
