import json, random
from faker import Faker
fake = Faker()

templates = [
    "OPERATION {name}: {objective}. Assets: {assets}. Threat: {threat}.",
    "INTEL: {source} reports {event} in {location}. Confidence {conf}."
]

with open("../data/fake_cui.jsonl", "w") as f:
    for i in range(10000):
        doc = {
            "id": f"CUI-{i:05d}",
            "text": random.choice(templates).format(
                name=fake.word().upper()+str(random.randint(100,999)),
                objective=random.choice(["Secure perimeter", "Extract VIP"]),
                assets=", ".join([fake.company() for _ in range(2)]),
                threat=random.choice(["LOW","MEDIUM","HIGH"]),
                source=fake.name(),
                event=fake.sentence(),
                location=fake.city(),
                conf=random.choice(["60%","85%","95%"])
            )
        }
        f.write(json.dumps(doc)+"\n")
print("10k fake CUI docs → data/fake_cui.jsonl")