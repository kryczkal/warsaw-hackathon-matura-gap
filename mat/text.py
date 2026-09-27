"""Text helpers shared by the Wikipedia index (build_wiki.py) and retrieval (context.py, kb.py)."""
import re

HIST = re.compile(r"(bitw|wojn|powstani|król|królow|cesarz|książ|dynast|traktat|pokój|sejm|konstytuc|rewoluc|okupac|"
                  r"zabor|rozbior|papież|biskup|reformac|imperi|państw|polityk|dyplomat|hetman|szlacht|"
                  r"średniow|starożyt|antyczn|kolonia|unii |unia |zimn|komunis|faszy|nazis|PRL|Solidarno|"
                  r"legion|kampani|oblężen|ustaw|parlament|republik|monarch|chrzest|zakon|krucjat)", re.I)
YEAR = re.compile(r"\b(1[0-9]{3}|[1-9][0-9]{2})\b|\bw\.\s|wiek|p\.n\.e\.")
TOK = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")
STOP = set("i w z na do się nie że o od po za to jak jest są był była było oraz a lub przez dla który która które tego tym też ze we co czy".split())


def stem(text):
    return [t[:6] for t in (w.lower() for w in TOK.findall(text)) if t not in STOP and len(t) > 1]


def chunks(text, n=170, maxc=6):
    words = text.split()
    out = []
    for i in range(0, len(words), n):
        out.append(" ".join(words[i:i + n]))
        if len(out) >= maxc:
            break
    return out


def q_base(it):
    """retrieval query for an item: its sources (minus image placeholders) + its question"""
    return re.sub(r"\[Obraz:[^\]]*\]", " ", it.get("source_text", "")) + " " + it["question"]


def trim(chunk, qstems, maxw):
    """keep the sentences of a chunk that overlap the query most, in original order, up to maxw words"""
    sents = re.split(r"(?<=[.!?])\s+", chunk)
    sc = [(len(set(stem(s)) & qstems), i) for i, s in enumerate(sents)]
    keep, n = set(), 0
    for s, i in sorted(sc, reverse=True):
        w = len(sents[i].split())
        if n + w > maxw and keep:
            continue
        keep.add(i); n += w
    return " ".join(sents[i] for i in sorted(keep))
