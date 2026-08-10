#!/usr/bin/env python3
"""Generate datasets/dlp-multilingual.jsonl.

Offsets are computed from the text by literal search and asserted, so the
annotations cannot drift from the strings. One-time authoring tool; not part of
the runtime.

Coverage is chosen to separate three failure modes that an aggregate score
conflates:
  * Latin-script non-English (de/fr/es/it/pt/nl/pl/tr) — the easy case; a model
    can often limp along on shared subwords.
  * Non-Latin scripts (ru/uk Cyrillic, el Greek, zh/ja CJK, ar Arabic RTL,
    hi Devanagari) — where an English tokenizer fragments into bytes.
  * Universal structured entities (email/IBAN/card) in every language, which
    the deterministic detector should get right regardless of script.
"""
import json

ROWS = []

def ex(eid, lang, text, ents, tags=("prose",)):
    out = []
    for etype, surface in ents:
        b = text.encode()
        start = b.find(surface.encode())
        assert start >= 0, f"{eid}: {surface!r} not in {text!r}"
        end = start + len(surface.encode())
        assert b[start:end].decode() == surface, f"{eid}: bad span {surface!r}"
        out.append({"type": etype, "start": start, "end": end})
    out.sort(key=lambda e: (e["start"], e["end"]))
    for a, b_ in zip(out, out[1:]):
        assert b_["start"] >= a["end"], f"{eid}: overlapping gold"
    ROWS.append({"id": eid, "source": "custom-multilingual", "language": lang,
                 "text": text, "entities": out, "tags": list(tags)})

# --- Latin script, non-English --------------------------------------------
ex("de-1", "de", "Bitte kontaktieren Sie Jörg Müller unter joerg.mueller@example.de.",
   [("PERSON", "Jörg Müller"), ("EMAIL", "joerg.mueller@example.de")])
ex("de-2", "de", "Die Rechnung geht an Annika Schröder, Telefon +49 30 901820.",
   [("PERSON", "Annika Schröder"), ("PHONE", "+49 30 901820")])
ex("de-3", "de", "IBAN DE89370400440532013000 lautet auf Klaus Weber.",
   [("IBAN", "DE89370400440532013000"), ("PERSON", "Klaus Weber")])

ex("fr-1", "fr", "Veuillez contacter François Dupont à francois.dupont@example.fr.",
   [("PERSON", "François Dupont"), ("EMAIL", "francois.dupont@example.fr")])
ex("fr-2", "fr", "Le dossier de Amélie Rousseau, téléphone +33 1 42 68 53 00, est prêt.",
   [("PERSON", "Amélie Rousseau"), ("PHONE", "+33 1 42 68 53 00")])
ex("fr-3", "fr", "Carte 4111 1111 1111 1111 au nom de Luc Bernard.",
   [("CREDIT_CARD", "4111 1111 1111 1111"), ("PERSON", "Luc Bernard")])

ex("es-1", "es", "Por favor escriba a José García: jose.garcia@example.es.",
   [("PERSON", "José García"), ("EMAIL", "jose.garcia@example.es")])
ex("es-2", "es", "La cuenta de Lucía Fernández tiene el teléfono +34 91 123 45 67.",
   [("PERSON", "Lucía Fernández"), ("PHONE", "+34 91 123 45 67")])
ex("es-3", "es", "IBAN ES9121000418450200051332 pertenece a Miguel Torres.",
   [("IBAN", "ES9121000418450200051332"), ("PERSON", "Miguel Torres")])

ex("it-1", "it", "Contattare Giulia Ricci all'indirizzo giulia.ricci@example.it.",
   [("PERSON", "Giulia Ricci"), ("EMAIL", "giulia.ricci@example.it")])
ex("it-2", "it", "Il numero di Marco Esposito è +39 06 6982 1234.",
   [("PERSON", "Marco Esposito"), ("PHONE", "+39 06 6982 1234")])

ex("pt-1", "pt", "Envie um email para João Silva: joao.silva@example.pt.",
   [("PERSON", "João Silva"), ("EMAIL", "joao.silva@example.pt")])
ex("pt-2", "pt", "A conta de Maria Conceição tem o telefone +351 21 123 4567.",
   [("PERSON", "Maria Conceição"), ("PHONE", "+351 21 123 4567")])

ex("nl-1", "nl", "Neem contact op met Sanne de Vries via sanne.devries@example.nl.",
   [("PERSON", "Sanne de Vries"), ("EMAIL", "sanne.devries@example.nl")])
ex("nl-2", "nl", "Het IBAN NL91ABNA0417164300 staat op naam van Pieter Bakker.",
   [("IBAN", "NL91ABNA0417164300"), ("PERSON", "Pieter Bakker")])

ex("pl-1", "pl", "Prosimy o kontakt z Tomasz Nowak pod adresem tomasz.nowak@example.pl.",
   [("PERSON", "Tomasz Nowak"), ("EMAIL", "tomasz.nowak@example.pl")])
ex("pl-2", "pl", "Numer telefonu Katarzyny Wiśniewskiej to +48 22 123 45 67.",
   [("PERSON", "Katarzyny Wiśniewskiej"), ("PHONE", "+48 22 123 45 67")])

ex("tr-1", "tr", "Lütfen Ayşe Yılmaz ile ayse.yilmaz@example.com.tr adresinden iletişime geçin.",
   [("PERSON", "Ayşe Yılmaz"), ("EMAIL", "ayse.yilmaz@example.com.tr")])
ex("tr-2", "tr", "Mehmet Öztürk telefon numarası +90 212 555 12 34.",
   [("PERSON", "Mehmet Öztürk"), ("PHONE", "+90 212 555 12 34")])

# --- Cyrillic --------------------------------------------------------------
ex("ru-1", "ru", "Свяжитесь с Михаил Ястребцов по адресу mikhail@example.ru.",
   [("PERSON", "Михаил Ястребцов"), ("EMAIL", "mikhail@example.ru")])
ex("ru-2", "ru", "Телефон Ольги Смирновой: +7 495 123-45-67.",
   [("PERSON", "Ольги Смирновой"), ("PHONE", "+7 495 123-45-67")])
ex("ru-3", "ru", "Карта 4111 1111 1111 1111 оформлена на Дмитрия Козлова.",
   [("CREDIT_CARD", "4111 1111 1111 1111"), ("PERSON", "Дмитрия Козлова")])
ex("uk-1", "uk", "Зверніться до Олена Ковальчук за адресою olena@example.ua.",
   [("PERSON", "Олена Ковальчук"), ("EMAIL", "olena@example.ua")])

# --- Greek -----------------------------------------------------------------
ex("el-1", "el", "Επικοινωνήστε με τον Γιώργος Παπαδόπουλος στο giorgos@example.gr.",
   [("PERSON", "Γιώργος Παπαδόπουλος"), ("EMAIL", "giorgos@example.gr")])

# --- CJK -------------------------------------------------------------------
ex("zh-1", "zh", "请联系王小明，邮箱 xiaoming@example.cn。",
   [("PERSON", "王小明"), ("EMAIL", "xiaoming@example.cn")])
ex("zh-2", "zh", "李伟的电话号码是 +86 10 6552 1234。",
   [("PERSON", "李伟"), ("PHONE", "+86 10 6552 1234")])
ex("ja-1", "ja", "田中太郎さんの連絡先は taro.tanaka@example.jp です。",
   [("PERSON", "田中太郎"), ("EMAIL", "taro.tanaka@example.jp")])
ex("ja-2", "ja", "佐藤花子の電話番号は +81 3 1234 5678 です。",
   [("PERSON", "佐藤花子"), ("PHONE", "+81 3 1234 5678")])
ex("ko-1", "ko", "김민준에게 minjun.kim@example.kr 로 연락하세요.",
   [("PERSON", "김민준"), ("EMAIL", "minjun.kim@example.kr")])

# --- Arabic / Hebrew (RTL) --------------------------------------------------
ex("ar-1", "ar", "يرجى الاتصال بـ أحمد المنصور على البريد ahmed@example.ae.",
   [("PERSON", "أحمد المنصور"), ("EMAIL", "ahmed@example.ae")])
ex("he-1", "he", "אנא צרו קשר עם דוד כהן בכתובת david@example.co.il.",
   [("PERSON", "דוד כהן"), ("EMAIL", "david@example.co.il")])

# --- Devanagari -------------------------------------------------------------
ex("hi-1", "hi", "कृपया प्रिया शर्मा से priya.sharma@example.in पर संपर्क करें।",
   [("PERSON", "प्रिया शर्मा"), ("EMAIL", "priya.sharma@example.in")])

# --- English control --------------------------------------------------------
ex("en-1", "en", "Please contact John Smith at john.smith@example.com.",
   [("PERSON", "John Smith"), ("EMAIL", "john.smith@example.com")])
ex("en-2", "en", "Sarah Connor can be reached on +1-415-555-1234.",
   [("PERSON", "Sarah Connor"), ("PHONE", "+1-415-555-1234")])
ex("en-3", "en", "IBAN GB82WEST12345698765432 belongs to Alan Turing.",
   [("IBAN", "GB82WEST12345698765432"), ("PERSON", "Alan Turing")])

# --- structured / config in non-English contexts ---------------------------
ex("de-cfg", "de", "DATENBANK=postgres://admin:s3cretpw99@db.internal/haupt",
   [("CONNECTION_STRING", "postgres://admin:s3cretpw99@db.internal/haupt")], ("config",))
ex("ru-log", "ru", "2026-08-08 пользователь=ivan@example.ru ip=203.0.113.44 вход не удался",
   [("EMAIL", "ivan@example.ru"), ("IP_ADDRESS", "203.0.113.44")], ("log",))
ex("ja-json", "ja", '{"顧客": "田中太郎", "メール": "taro@example.jp"}',
   [("PERSON", "田中太郎"), ("EMAIL", "taro@example.jp")], ("json",))

# --- negatives in several scripts ------------------------------------------
for eid, lang, text in [
    ("neg-de", "de", "Fehler #4411 wurde in Version 1.23.4567 behoben."),
    ("neg-ru", "ru", "Идентификатор запроса: 550e8400-e29b-41d4-a716-446655440000"),
    ("neg-zh", "zh", "服务监听 127.0.0.1:8080，超时 30000 毫秒。"),
    ("neg-ja", "ja", "ビルド 20260808.3 は 1.234 秒で完了しました。"),
    ("neg-ar", "ar", "رمز الخطأ 0x8007000E حدث عند الإزاحة 0x1F4A."),
]:
    ex(eid, lang, text, [], ("negative",))

with open("datasets/dlp-multilingual.jsonl", "w", encoding="utf-8") as f:
    f.write("# dlp-multilingual — per-language PII benchmark. Offsets are UTF-8 bytes.\n")
    f.write("# Regenerate with scripts/gen_multilingual.py; do not hand-edit offsets.\n")
    for r in sorted(ROWS, key=lambda x: x["id"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

import collections
c = collections.Counter(r["language"] for r in ROWS)
print(f"{len(ROWS)} docs, {sum(len(r['entities']) for r in ROWS)} entities, "
      f"{sum(1 for r in ROWS if not r['entities'])} negatives")
print("languages:", ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
