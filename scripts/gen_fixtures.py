#!/usr/bin/env python3
"""One-off generator for the AI-DLP fixture datasets.

Writes JSONL with UTF-8 BYTE offsets computed from the text itself, so the
fixtures cannot drift from their annotations. Entities are located by literal
search; every span is asserted to slice back to the expected surface.
"""
import json, sys

def ex(eid, text, ents, tags, lang="en", source="custom"):
    out = []
    cursor = 0
    for etype, surface, *rest in ents:
        occ = rest[0] if rest else 0
        start = -1
        from_ = 0
        for _ in range(occ + 1):
            start = text.encode().find(surface.encode(), from_)
            assert start >= 0, f"{eid}: {surface!r} not found in {text!r}"
            from_ = start + 1
        end = start + len(surface.encode())
        assert text.encode()[start:end].decode() == surface, f"{eid}: bad span for {surface!r}"
        out.append({"type": etype, "start": start, "end": end})
    out.sort(key=lambda e: (e["start"], e["end"]))
    # reject overlaps, same rule the loader enforces
    for a, b in zip(out, out[1:]):
        assert b["start"] >= a["end"], f"{eid}: overlapping gold {a} {b}"
    d = {"id": eid, "source": source, "language": lang, "text": text,
         "entities": out, "tags": tags}
    return d

TEST = []
A = TEST.append

# --- natural language ------------------------------------------------------
A(ex("nl-001", "Please contact John Smith at john@example.com.",
     [("PERSON", "John Smith"), ("EMAIL", "john@example.com")], ["prose"]))
A(ex("nl-002", "Customer passport number is X1234567 and she was born 1985-03-12.",
     [("PASSPORT", "X1234567"), ("DATE_OF_BIRTH", "1985-03-12")], ["prose"]))
A(ex("nl-003", "Call Maria Garcia on +1-415-555-1234 before Friday.",
     [("PERSON", "Maria Garcia"), ("PHONE", "+1-415-555-1234")], ["prose"]))
A(ex("nl-004", "Her IBAN is DE89370400440532013000 and the card ends 4111 1111 1111 1111.",
     [("IBAN", "DE89370400440532013000"), ("CREDIT_CARD", "4111 1111 1111 1111")], ["prose"]))
A(ex("nl-005", "Ship it to 12 Hauptstrasse, Munich, Germany.",
     [("ADDRESS", "12 Hauptstrasse, Munich, Germany")], ["prose"]))
A(ex("nl-006", "The driver's licence number on file is D1234567890123.",
     [("DRIVERS_LICENSE", "D1234567890123")], ["prose"]))

# --- multilingual ----------------------------------------------------------
A(ex("ml-001", "Свяжитесь с Михаил Ястребцов по адресу mikhail@example.ru",
     [("PERSON", "Михаил Ястребцов"), ("EMAIL", "mikhail@example.ru")], ["prose"], "ru"))
A(ex("ml-002", "Jörg Müller erreichen Sie unter joerg.mueller@example.de.",
     [("PERSON", "Jörg Müller"), ("EMAIL", "joerg.mueller@example.de")], ["prose"], "de"))
A(ex("ml-003", "José García vive en Málaga y su correo es jose@example.es.",
     [("PERSON", "José García"), ("EMAIL", "jose@example.es")], ["prose"], "es"))
A(ex("ml-004", "François Dupont habite à Paris, tél. +33 1 42 68 53 00.",
     [("PERSON", "François Dupont"), ("PHONE", "+33 1 42 68 53 00")], ["prose"], "fr"))
A(ex("ml-005", "王小明的电子邮件是 xiaoming@example.cn。",
     [("PERSON", "王小明"), ("EMAIL", "xiaoming@example.cn")], ["prose"], "zh"))

# --- source code -----------------------------------------------------------
A(ex("code-001", 'let email = "customer@example.com";',
     [("EMAIL", "customer@example.com")], ["source_code"]))
A(ex("code-002",
     'const client = new Client({ apiKey: "sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2" });',
     [("API_KEY", "sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2")], ["source_code"]))
A(ex("code-003",
     'conn = psycopg2.connect("postgres://svcuser:Tr0ub4dor@db.internal:5432/app")',
     [("CONNECTION_STRING", "postgres://svcuser:Tr0ub4dor@db.internal:5432/app")],
     ["source_code"]))

# --- config ----------------------------------------------------------------
A(ex("cfg-001", "DATABASE_URL=postgres://admin:s3cretpw99@prod-db.internal/main",
     [("CONNECTION_STRING", "postgres://admin:s3cretpw99@prod-db.internal/main")], ["config"]))
A(ex("cfg-002", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nAWS_REGION=eu-central-1",
     [("API_KEY", "AKIAIOSFODNN7EXAMPLE")], ["config"]))
A(ex("cfg-003", "SMTP_USER=alerts@example.com\nSMTP_PASSWORD=Xk9mQp2wLz",
     [("EMAIL", "alerts@example.com"), ("PASSWORD", "Xk9mQp2wLz")], ["config"]))

# --- json ------------------------------------------------------------------
A(ex("json-001", '{"customer": "John Smith", "email": "john@example.com"}',
     [("PERSON", "John Smith"), ("EMAIL", "john@example.com")], ["json"]))
A(ex("json-002",
     '{"user": {"name": "Anna Kowalska", "phone": "+48 22 123 45 67", "iban": "PL61109010140000071219812874"}}',
     [("PERSON", "Anna Kowalska"), ("PHONE", "+48 22 123 45 67"),
      ("IBAN", "PL61109010140000071219812874")], ["json"], "pl"))

# --- yaml ------------------------------------------------------------------
A(ex("yaml-001", "customer:\n  name: John Smith\n  phone: +1-415-555-1234",
     [("PERSON", "John Smith"), ("PHONE", "+1-415-555-1234")], ["yaml"]))
A(ex("yaml-002", "secrets:\n  db_password: hunter2xyzQ\n  api_key: k9Qd2mZ4pR7wL1tY6vB3",
     [("PASSWORD", "hunter2xyzQ"), ("API_KEY", "k9Qd2mZ4pR7wL1tY6vB3")], ["yaml"]))

# --- logs ------------------------------------------------------------------
A(ex("log-001", "2026-08-08 user=john@example.com ip=203.0.113.44 login failed",
     [("EMAIL", "john@example.com"), ("IP_ADDRESS", "203.0.113.44")], ["log"]))
A(ex("log-002",
     "2026-08-08T11:02:14Z WARN auth token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk expired",
     [("ACCESS_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")],
     ["log"]))

# --- sql -------------------------------------------------------------------
A(ex("sql-001", "SELECT *\nFROM customers\nWHERE email = 'john@example.com';",
     [("EMAIL", "john@example.com")], ["sql"]))
A(ex("sql-002",
     "INSERT INTO users (name, ssn) VALUES ('Robert Chen', '123-45-6789');",
     [("PERSON", "Robert Chen"), ("NATIONAL_ID", "123-45-6789")], ["sql"]))

# --- AI/coding prompts -----------------------------------------------------
A(ex("ai-001",
     "Can you debug this config?\n\nAWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
     [("API_KEY", "AKIAIOSFODNN7EXAMPLE"),
      ("SECRET", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")], ["ai_prompt", "config"]))
A(ex("ai-002",
     "Write an email to Sarah Connor (sarah.connor@example.com) about the outage.",
     [("PERSON", "Sarah Connor"), ("EMAIL", "sarah.connor@example.com")], ["ai_prompt"]))
A(ex("ai-003",
     "Here is my ssh key, help me fix permissions:\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAB\n-----END OPENSSH PRIVATE KEY-----",
     [("PRIVATE_KEY", "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAB\n-----END OPENSSH PRIVATE KEY-----")],
     ["ai_prompt"]))
A(ex("ai-004",
     "Summarise this ticket: caller Priya Sharma, mobile 020 7946 0018, account GB29NWBK60161331926819.",
     [("PERSON", "Priya Sharma"), ("PHONE", "020 7946 0018"),
      ("IBAN", "GB29NWBK60161331926819")], ["ai_prompt"], "en"))

# --- negatives -------------------------------------------------------------
NEG = [
    ("neg-001", "Fix issue #4411.", ["prose"]),
    ("neg-002", "Version 1.23.4567 was deployed to staging.", ["prose"]),
    ("neg-003", "Request ID: 550e8400-e29b-41d4-a716-446655440000", ["log"]),
    ("neg-004", "SHA256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", ["log"]),
    ("neg-005", "Service listening on 127.0.0.1:8080 with timeout 30000ms.", ["log"]),
    ("neg-006", "const nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];", ["source_code"]),
    ("neg-007", "Build 20260808.3 finished in 1.234s.", ["log"]),
    ("neg-008", "git commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b", ["log"]),
    ("neg-009", "Acme Corporation and Globex Inc. announced a merger.", ["prose"]),
    ("neg-010", "Set MAX_RETRIES=5 and CONNECT_TIMEOUT=30 in the config.", ["config"]),
    ("neg-011", "npm install @scope/package@1.2.3 --save-dev", ["source_code"]),
    ("neg-012", "Colour tokens: #ff00aa, #00ffcc, #1a1a1a", ["source_code"]),
    ("neg-013", "Use YOUR_API_KEY as a placeholder in the docs.", ["prose"]),
    ("neg-014", "The matrix has dimensions 4111 x 1111 elements.", ["prose"]),
    ("neg-015", "Error code 0x8007000E occurred at offset 0x1F4A.", ["log"]),
    ("neg-016", "Port range 8000-9000 is reserved for internal services.", ["config"]),
    ("neg-017", "docker pull ubuntu@sha256:1e622c5f073b4f6bfad6632f2616c7f59ef256e96fe78bf6a595d1dc4376ac02", ["source_code"]),
    ("neg-018", "The meeting is at 10:20:30 on 2026-08-08.", ["prose"]),
]
for eid, text, tags in NEG:
    A(ex(eid, text, [], tags + ["negative"]))

# --- dev split (calibration) ----------------------------------------------
DEV = []
B = DEV.append
B(ex("dev-nl-001", "Reach out to Emily Watson at emily.watson@example.org.",
     [("PERSON", "Emily Watson"), ("EMAIL", "emily.watson@example.org")], ["prose"]))
B(ex("dev-nl-002", "His passport is P9988776 and he was born on 1979-11-02.",
     [("PASSPORT", "P9988776"), ("DATE_OF_BIRTH", "1979-11-02")], ["prose"]))
B(ex("dev-nl-003", "Ring Tomasz Nowak on +48 601 234 567 tomorrow.",
     [("PERSON", "Tomasz Nowak"), ("PHONE", "+48 601 234 567")], ["prose"], "pl"))
B(ex("dev-ml-001", "Ольга Смирнова написала на olga@example.ru вчера.",
     [("PERSON", "Ольга Смирнова"), ("EMAIL", "olga@example.ru")], ["prose"], "ru"))
B(ex("dev-fin-001", "Charge the card 5500 0055 5555 5559, IBAN NL91ABNA0417164300.",
     [("CREDIT_CARD", "5500 0055 5555 5559"), ("IBAN", "NL91ABNA0417164300")], ["prose"]))
B(ex("dev-cfg-001", "REDIS_URL=redis://cache:Sup3rSecret@redis.internal:6379/0",
     [("CONNECTION_STRING", "redis://cache:Sup3rSecret@redis.internal:6379/0")], ["config"]))
B(ex("dev-json-001", '{"contact": {"email": "ops@example.net", "ip": "198.51.100.7"}}',
     [("EMAIL", "ops@example.net"), ("IP_ADDRESS", "198.51.100.7")], ["json"]))
B(ex("dev-yaml-001", "app:\n  owner: Lena Fischer\n  token: ghp_" + "a1B2c3D4e5" * 3 + "aaaaaa",
     [("PERSON", "Lena Fischer"), ("ACCESS_TOKEN", "ghp_" + "a1B2c3D4e5" * 3 + "aaaaaa")],
     ["yaml"], "de"))
B(ex("dev-log-001", "2026-07-01 user=ops@example.net ip=198.51.100.7 action=export",
     [("EMAIL", "ops@example.net"), ("IP_ADDRESS", "198.51.100.7")], ["log"]))
B(ex("dev-sql-001", "UPDATE staff SET phone='+44 20 7946 0958' WHERE name='Alan Turing';",
     [("PHONE", "+44 20 7946 0958"), ("PERSON", "Alan Turing")], ["sql"]))
B(ex("dev-ai-001", "Draft a reply to Hiroshi Tanaka <hiroshi@example.jp> about billing.",
     [("PERSON", "Hiroshi Tanaka"), ("EMAIL", "hiroshi@example.jp")], ["ai_prompt"], "ja"))
DEV_NEG = [
    ("dev-neg-001", "Ticket #9912 closed after 3 retries.", ["prose"]),
    ("dev-neg-002", "Release 2.14.0-rc.3 published at 2026-07-01T09:00:00Z.", ["log"]),
    ("dev-neg-003", "Correlation id 7f3e9b21-4c5d-4a7e-9f01-2b3c4d5e6f70", ["log"]),
    ("dev-neg-004", "Bind to 0.0.0.0:3000 for local development.", ["config"]),
    ("dev-neg-005", "MD5 checksum d41d8cd98f00b204e9800998ecf8427e verified.", ["log"]),
    ("dev-neg-006", "Array indices 1111, 2222, 3333, 4444 were skipped.", ["source_code"]),
]
for eid, text, tags in DEV_NEG:
    B(ex(eid, text, [], tags + ["negative"]))

def write(path, rows, header):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} examples, "
          f"{sum(len(r['entities']) for r in rows)} entities, "
          f"{sum(1 for r in rows if not r['entities'])} negatives")

write("datasets/dlp-v1.jsonl", TEST,
      "# dlp-v1 — held-out AI-DLP test set. Offsets are UTF-8 bytes.\n"
      "# Regenerate with scripts/gen_fixtures.py; do not hand-edit offsets.\n")
write("datasets/dlp-dev.jsonl", DEV,
      "# dlp-dev — calibration/development split. Choose thresholds here, not on dlp-v1.\n"
      "# Offsets are UTF-8 bytes. Regenerate with scripts/gen_fixtures.py.\n")
