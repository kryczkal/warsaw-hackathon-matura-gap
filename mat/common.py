"""Shared prompt construction for training and inference (must stay identical)."""
import re

SYSTEM = (
    "Jesteś egzaminatorem-ekspertem i najlepszym maturzystą z historii (matura CKE, poziom rozszerzony). "
    "Rozwiązujesz jedno zadanie. Czytasz polecenie, wszystkie źródła i ilustracje. "
    "Piszesz po polsku wyłącznie ostateczną odpowiedź, bez rozumowania i wstępów. "
    "Jeśli polecenie kończy się etykietami (np. 'Rozstrzygnięcie:' i 'Uzasadnienie:'), powtórz je i uzupełnij każdą. "
    "W zadaniach zamkniętych trzymaj się dokładnie składni z pola 'Format odpowiedzi'. "
    "W zadaniach otwartych podaj konkretne fakty (nazwy, daty, pojęcia), odwołaj się do źródła i do własnej wiedzy, gdy polecenie tego wymaga. "
    "Wypracowanie: zacznij od 'Temat N.', potem pełny tekst ok. 500 słów: teza, trzy rozbudowane argumenty z faktami, kontrargument, zakończenie."
)


def kind(item):
    fmt = item.get("answer_format", "")
    if item.get("max_points", 0) >= 10 or fmt.startswith("Jeden tekst"):
        return "essay"
    if fmt.startswith("Tekst po polsku"):
        return "open"
    return "closed"


def user_text(item, context=None):
    parts = []
    if context:
        parts.append("Materiały pomocnicze (fragmenty encyklopedii, mogą być nieistotne):\n" + context.strip())
    if item.get("source_text", "").strip():
        parts.append("Źródła:\n" + item["source_text"].strip())
    parts.append("Polecenie:\n" + item["question"].strip())
    parts.append("Format odpowiedzi:\n" + item.get("answer_format", "").strip())
    return "\n\n".join(parts)


def single_topic(item, topic):
    """Essay with the topic already chosen by the harness: keep only that topic in the question."""
    head = item["question"].split("\n1. ")[0].replace("trzy tematy. Wybierz jeden z nich do opracowania.", "jeden temat.")
    return dict(item, question=head + "\n" + topic)


def baseline_text(item):
    """The plain 'untouched model' prompt: exactly the exam's own fields, nothing else."""
    text = item["question"].strip()
    if item.get("source_text", "").strip():
        text = item["source_text"].strip() + "\n\n" + text
    return text + "\n\n" + item.get("answer_format", "")


def vote_closed(answers):
    """Per-line majority vote for closed answers like '1: P\\n2: F' or 'A: 3'."""
    from collections import Counter
    keyed = {}
    order = []
    plain = Counter()
    for a in answers:
        lines = [l.strip() for l in a.strip().splitlines() if l.strip()]
        m = [re.match(r"^([0-9A-Za-z]{1,3})\s*[:.)]\s*(.+)$", l) for l in lines]
        if lines and all(m):
            for mm in m:
                k, v = mm.group(1), mm.group(2).strip()
                if k not in keyed:
                    keyed[k] = Counter(); order.append(k)
                keyed[k][v] += 1
        else:
            plain[a.strip()] += 1
    if keyed and sum(sum(c.values()) for c in keyed.values()) >= sum(plain.values()):
        return "\n".join(f"{k}: {keyed[k].most_common(1)[0][0]}" for k in order)
    return plain.most_common(1)[0][0] if plain else answers[0]
