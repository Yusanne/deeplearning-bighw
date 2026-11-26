"""Shared helpers for CiteFaith project scripts (import via sys.path.insert of scripts dir)."""
import ast
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi


def norm(s):
    return re.sub(r"[%$€£'\"\s(),.]+", "", str(s)).lower()


def toks(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def load_pages(pages_dir, doc_id):
    p = os.path.join(pages_dir, doc_id + ".json")
    if not os.path.exists(p):
        return None
    return {int(k): v for k, v in json.load(open(p, encoding="utf-8")).items()}


def parse_maybe_list(v):
    if isinstance(v, list):
        return v
    try:
        x = ast.literal_eval(v) if isinstance(v, str) else [v]
        return x if isinstance(x, list) else [x]
    except Exception:
        return [v]


def corpus_pages(pages, min_chars=50):
    return [i for i in sorted(pages) if len(pages[i].strip()) >= min_chars]


def gold_pages_by_answer(pages, answer, evidence_pages):
    """Locate gold pages by normalized answer string; fall back to dataset field."""
    g = norm(answer)
    if g:
        hits = [i for i, t in pages.items() if g in norm(t)]
        if hits:
            return hits, "answer-string"
    cand = []
    for p in evidence_pages:
        for c in (p, p - 1, p + 1):
            if 0 <= c < len(pages) and c not in cand:
                cand.append(c)
    return cand, "field-fallback"


def tfidf_order(corpus, pages, query):
    vocab, docs_t = {}, []
    for i in corpus:
        t = toks(pages[i])
        docs_t.append(t)
        for w in t:
            vocab.setdefault(w, len(vocab))
    N = len(corpus)
    df = np.zeros(len(vocab))
    for t in docs_t:
        for w in set(t):
            df[vocab[w]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1
    mat = np.zeros((N, len(vocab)), dtype=np.float32)
    for r, t in enumerate(docs_t):
        for w, c in zip(*np.unique(t, return_counts=True)):
            mat[r, vocab[w]] = c
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    mat *= idf
    q = np.zeros(len(vocab), dtype=np.float32)
    for w in set(toks(query)):
        if w in vocab:
            q[vocab[w]] = toks(query).count(w) * idf[vocab[w]]
    nq = np.linalg.norm(q) or 1.0
    den = np.linalg.norm(mat, axis=1) * nq
    den[den == 0] = 1.0
    sims = (mat @ q) / den
    order = np.argsort(-sims)
    return [corpus[i] for i in order]


def rrf_fuse(rank_lists, k0=60):
    score = {}
    for ranks in rank_lists:
        for pos, p in enumerate(ranks):
            score[p] = score.get(p, 0.0) + 1.0 / (k0 + pos + 1)
    return sorted(score, key=lambda p: -score[p])


def retrieve(pages, question):
    """BM25 + TF-IDF -> RRF order; also returns bm25 order. Dense hook: add API rank list later."""
    corpus = corpus_pages(pages)
    if not corpus:
        return [], [], []
    bm = BM25Okapi([toks(pages[i]) for i in corpus])
    bm_scores = bm.get_scores(toks(question))
    bm_order = [corpus[i] for i in np.argsort(-bm_scores)]
    tf_order = tfidf_order(corpus, pages, question)
    rrf = rrf_fuse([bm_order, tf_order])
    return bm_order, tf_order, rrf


def _standalone_re(tok):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _us_commas(d):
    try:
        return f"{int(d):,}"
    except Exception:
        return d


def literal_variants(a):
    a = str(a).strip()
    out = [a, a.rstrip("% .")]
    digits = re.sub(r"\D", "", a)
    if digits and digits != a:
        out.append(digits)
    if digits and len(digits) > 3:
        out.append(_us_commas(digits))
    if len(digits) == 1:
        out.append({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}[digits])
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def other_number(page_text, ref, exclude=()):
    try:
        g = float(str(ref).replace(",", ""))
    except Exception:
        g = None
    if g is None:
        return None
    cands = []
    for n in re.findall(r"\d[\d,]*(?:\.\d+)?", page_text):
        try:
            v = float(n.replace(",", ""))
        except Exception:
            continue
        if abs(v - g) > 1e-9 and n not in exclude:
            cands.append((abs(math_log10_ratio(v, g)), n))
    cands.sort()
    return cands[0][1] if cands else None


def math_log10_ratio(v, g):
    import math
    return abs(math.log10(max(abs(v), 1e-9) / max(abs(g), 1e-9)))


def build_swap(page_text, answer):
    """Same-page substitution (week1 v3). Returns (ok, rule, e_minus, replacement, reason)."""
    a = str(answer).strip()
    tmatch = re.fullmatch(r"(\d{1,2}):(\d{2})([ap]m)?", a.lower())
    for v in literal_variants(a):
        pat = _standalone_re(v) if len(v) <= 8 else re.compile(re.escape(v), re.IGNORECASE)
        if not pat.search(page_text):
            continue
        if tmatch and ":" in v:
            repl = f"{int(tmatch.group(1)) % 12 + 1}:{tmatch.group(2)}"
            return True, "time-conflict", pat.sub(repl, page_text), repl, ""
        if re.fullmatch(r"[\d.,]+", v):
            raw = other_number(page_text, v.replace(",", ""))
            repl = raw if raw is not None else ("7" if "." not in v else "7.5")
            return True, "numeric-conflict", pat.sub(repl, page_text), repl, ""
        pool = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}|\d{1,4}[A-Za-z]{1,4}", page_text)
                if re.sub(r"\W", "", w).lower() != re.sub(r"\W", "", v).lower()]
        repl = min(pool, key=lambda w: abs(len(w) - len(v))) if pool else v + "X"
        return True, "span-replace-all", pat.sub(repl, page_text), repl, ""

    core_tokens = re.findall(r"[A-Za-z0-9]+", a)
    e_minus, replaced, used = page_text, [], set()
    for t in sorted(set(core_tokens), key=len, reverse=True):
        if len(t) < 2:
            continue
        if len(t) >= 4 and t.isdigit():
            repl = other_number(page_text, t, used | {t}) or str(int(t) + 1)
        elif t.lower() == "pm":
            repl = "am"
        else:
            pool = [w for w in re.findall(r"[A-Za-z0-9]{2,}", page_text)
                    if w not in used and w.lower() != t.lower()]
            repl = pool[0] if pool else t + "X"
        pat2 = _standalone_re(t)
        if pat2.search(e_minus):
            e_minus = pat2.sub(repl, e_minus)
            used.add(repl)
            replaced.append((t, repl))
    if replaced:
        return True, "token-fallback", e_minus, ";".join(r[1] for r in replaced), ""
    return False, "no-substitution", page_text, "", "answer tokens not substitutable on page"


def swap_valid_proxy(page_text_gold, e_minus, answer, replacement):
    g = norm(answer)
    has = all(p.lower() in e_minus.lower() for p in str(replacement).split(";") if p)
    if len(g) <= 2:
        free = not _standalone_re(str(answer)).search(e_minus)
    else:
        free = not (g and g in norm(e_minus))
    return has and free


def benign_candidates(pages, answer, gold, swap_page, replacement, bm_order, n=2):
    """§7.4 benign pages: topic-related (BM25 top) but must not contain the answer or the swapped value."""
    g = norm(answer)
    repl_parts = [x for x in str(replacement or "").split(";") if x]
    skip = set(gold) | {swap_page - 1, swap_page, swap_page + 1}
    out = []
    for p in bm_order:
        if p in skip or len(out) >= n:
            continue
        t = pages[p]
        if g and g in norm(t):
            continue
        if any(x.lower() in t.lower() for x in repl_parts):
            continue
        out.append(p)
    return out
