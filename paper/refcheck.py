#!/usr/bin/env python3
"""Verify .bib references against Crossref to flag possibly-hallucinated entries."""
import re, sys, time, json, html
from difflib import SequenceMatcher
import requests

BIB = sys.argv[1] if len(sys.argv) > 1 else "general.bib"
MAILTO = "gianmarco.pinton@gmail.com"
HEADERS = {"User-Agent": f"refcheck/1.0 (mailto:{MAILTO})"}

def norm(s):
    s = html.unescape(s or "")
    s = re.sub(r"\{|\}|\\[a-zA-Z]+|\\.|\$", "", s)   # strip TeX braces/macros
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

# --- crude but adequate .bib parser ---
def parse(path):
    txt = open(path, encoding="utf-8").read()
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", txt):
        etype, key = m.group(1).lower(), m.group(2).strip()
        start = m.end()
        depth, i = 1, m.start() + m.group(0).rfind("{") if False else start
        # find matching close brace from the '{' after @type
        b = txt.index("{", m.start())
        depth, i = 1, b + 1
        while i < len(txt) and depth:
            if txt[i] == "{": depth += 1
            elif txt[i] == "}": depth -= 1
            i += 1
        body = txt[b+1:i-1]
        def field(name):
            fm = re.search(name + r"\s*=\s*", body, re.I)
            if not fm: return ""
            j = fm.end()
            if txt and body[j] == "{":
                d, k = 1, j+1
                while k < len(body) and d:
                    if body[k] == "{": d += 1
                    elif body[k] == "}": d -= 1
                    k += 1
                return body[j+1:k-1].strip()
            fm2 = re.match(r"[\"{]?(.*?)[\"}]?\s*[,\n]", body[j:], re.S)
            return (fm2.group(1) if fm2 else "").strip()
        entries.append({
            "key": key, "type": etype,
            "title": field("title"), "author": field("author"),
            "year": field("year"), "doi": field("doi"),
            "journal": field("journal") or field("booktitle") or field("publisher"),
        })
    return entries

def first_author(a):
    if not a: return ""
    chunk = re.split(r"\s+and\s+", a)[0]
    return norm(chunk.split(",")[0])

def crossref_by_doi(doi):
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=HEADERS, timeout=20)
        if r.status_code != 200: return None
        return r.json()["message"]
    except Exception:
        return None

def crossref_by_title(title, author):
    try:
        params = {"query.bibliographic": norm(title), "rows": 3}
        if author: params["query.author"] = first_author(author)
        r = requests.get("https://api.crossref.org/works", headers=HEADERS, params=params, timeout=20)
        if r.status_code != 200: return []
        return r.json()["message"]["items"]
    except Exception:
        return []

def cr_title(item):
    t = item.get("title") or [""]
    return t[0] if t else ""

def cr_year(item):
    for k in ("published-print", "published-online", "issued", "created"):
        dp = item.get(k, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]: return str(dp[0][0])
    return ""

results = []
entries = parse(BIB)
print(f"Parsed {len(entries)} entries from {BIB}\n", file=sys.stderr)
for e in entries:
    verdict, score, matched, mdoi, note = "NOT_FOUND", 0.0, "", "", ""

    # 1) If a DOI is given, resolve it and compare titles.
    if e["doi"]:
        item = crossref_by_doi(e["doi"])
        if not item:
            verdict, note = "DOI_DEAD", "DOI did not resolve on Crossref"
        else:
            score, matched, mdoi = sim(e["title"], cr_title(item)), cr_title(item), e["doi"]
            verdict = "VERIFIED" if score >= 0.6 else "DOI_MISMATCH"

    # 2) Title/author search as primary check (no DOI) or fallback (bad DOI).
    if verdict in ("NOT_FOUND", "DOI_DEAD", "DOI_MISMATCH") and e["title"]:
        items = crossref_by_title(e["title"], e["author"])
        best = max(items, key=lambda it: sim(e["title"], cr_title(it)), default=None)
        if best:
            s2 = sim(e["title"], cr_title(best))
            if s2 > score:
                score, matched, mdoi = s2, cr_title(best), best.get("DOI", "")
            yr_ok = (not e["year"]) or e["year"] == cr_year(best)
            if s2 >= 0.85:
                verdict = "VERIFIED"
                if not yr_ok:
                    note = f"year bib={e['year']} crossref={cr_year(best)}"
            elif s2 >= 0.6:
                verdict = "REVIEW"
            # else: leave as NOT_FOUND / DOI_DEAD / DOI_MISMATCH

    results.append({**e, "verdict": verdict, "score": round(score, 2),
                    "matched": matched[:90], "matched_doi": mdoi, "note": note})
    time.sleep(0.2)

order = {"NOT_FOUND":0,"DOI_DEAD":1,"DOI_MISMATCH":2,"REVIEW":3,"VERIFIED":4}
results.sort(key=lambda r: (order.get(r["verdict"],9), r["score"]))
json.dump(results, open("refcheck_results.json","w"), indent=2)

from collections import Counter
c = Counter(r["verdict"] for r in results)
print("="*100)
print("SUMMARY:", dict(c))
print("="*100)
for r in results:
    flag = "" if r["verdict"]=="VERIFIED" else "  <-- CHECK"
    print(f"[{r['verdict']:13}] {r['score']:.2f} {r['key']:28} ({r['type']}){flag}")
    if r["verdict"] != "VERIFIED":
        print(f"      bib title: {r['title'][:88]}")
        if r["matched"]: print(f"      crossref : {r['matched']}  (doi:{r['matched_doi']})")
        if r["note"]: print(f"      note     : {r['note']}")
