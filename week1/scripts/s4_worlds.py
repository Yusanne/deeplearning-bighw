#!/usr/bin/env python
"""Week1 S4: per-doc page retrieval (BM25 + TF-IDF -> RRF), build four counterfactual
worlds per question, swap distractor construction + rule-based validity proxy,
benign pages per frozen rule, token estimates.

Usage: python -I s4_worlds.py <selected20.json> <pages_dir> <worlds_dir> <summary_out.json>
"""
import json
import math
import os
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi


def norm(s):
    return re.sub(r"[%$€£'\"\s(),.]+", "", str(s)).lower()


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def tfidf_ranks(pages_idx, pages_text, query):
    vocab = {}
    docs_t = []
    for i in pages_idx:
        t = toks(pages_text[i])
        docs_t.append(t)
        for w in t:
            if w not in vocab:
                vocab[w] = len(vocab)
    N = len(pages_idx)
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
    nq = np.linalg.norm(q)
    den = np.linalg.norm(mat, axis=1) * (nq if nq else 1)
    den[den == 0] = 1
    sims = (mat @ q) / den
    order = np.argsort(-sims)
    return [pages_idx[i] for i in order], [float(sims[i]) for i in order]


def rrf_fuse(rank_lists, k0=60):
    score = {}
    for ranks in rank_lists:
        for pos, p in enumerate(ranks):
            score[p] = score.get(p, 0.0) + 1.0 / (k0 + pos + 1)
    return sorted(score, key=lambda p: -score[p])


def _standalone_re(tok):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _us_commas(d):
    try:
        return f"{int(d):,}"
    except Exception:
        return d


def literal_variants(a):
    """Ordered textual forms the answer may take inside extracted page text."""
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


def build_swap(page_text, answer, answer_format):
    """Layered same-page substitution. Returns (ok, rule, e_minus, replacement, reason)."""
    a = str(answer).strip()

    def page_numbers():
        return re.findall(r"\d[\d,]*(?:\.\d+)?", page_text)

    def other_number(ref, exclude=()):
        try:
            g = float(str(ref).replace(",", ""))
        except Exception:
            g = None
        cands = []
        for n in page_numbers():
            try:
                v = float(n.replace(",", ""))
            except Exception:
                continue
            if g is not None and abs(v - g) > 1e-9 and n not in exclude:
                cands.append((abs(math.log10(max(abs(v), 1e-9) / max(abs(g), 1e-9))), n))
        cands.sort()
        return cands[0][1] if cands else None

    tmatch = re.fullmatch(r"(\d{1,2}):(\d{2})([ap]m)?", a.lower())

    # route 1: literal variants, all occurrences
    for v in literal_variants(a):
        pat = _standalone_re(v) if len(v) <= 8 else re.compile(re.escape(v), re.IGNORECASE)
        if not pat.search(page_text):
            continue
        if tmatch and ":" in v:
            repl = f"{int(tmatch.group(1)) % 12 + 1}:{tmatch.group(2)}"
            return True, "time-conflict", pat.sub(repl, page_text), repl, ""
        if re.fullmatch(r"[\d.,]+", v):
            raw = other_number(v.replace(",", ""))
            repl = raw if raw is not None else "7" if "." not in v else "7.5"
            return True, "numeric-conflict", pat.sub(repl, page_text), repl, ""
        pool = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}|\d{1,4}[A-Za-z]{1,4}", page_text)
                if re.sub(r"\W", "", w).lower() != re.sub(r"\W", "", v).lower()]
        repl = min(pool, key=lambda w: abs(len(w) - len(v))) if pool else v + "X"
        return True, "span-replace-all", pat.sub(repl, page_text), repl, ""

    # route 2: token-level fallback (dates/lists whose literal form differs on page)
    core_tokens = re.findall(r"[A-Za-z0-9]+", a)
    e_minus = page_text
    replaced, used = [], set()
    for t in sorted(set(core_tokens), key=len, reverse=True):
        if len(t) < 2:
            continue
        if len(t) >= 4 and t.isdigit():
            repl = other_number(t, used | {t}) or str(int(t) + 1)
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


def main(selected_path, pages_dir, worlds_dir, summary_out):
    sel = json.load(open(selected_path, encoding="utf-8"))
    rows, tok_rows, skipped = [], [], []
    for q in sel["questions"]:
        pf = os.path.join(pages_dir, q["doc_id"] + ".json")
        pages = {int(k): v for k, v in json.load(open(pf, encoding="utf-8")).items()}
        g = norm(q["answer"])
        gold_pages = [i for i, t in pages.items() if g and g in norm(t)]
        if not gold_pages:
            gold_pages = [p for p in q["evidence_pages"] if p is not None]
        corpus = [i for i in sorted(pages) if len(pages[i].strip()) >= 50]
        if not corpus:
            skipped.append({"qid": q["qid"], "doc_id": q["doc_id"],
                            "reason": "no text-extractable page (likely scanned doc)"})
            continue
        page_texts = [pages[i] for i in corpus]
        bm = BM25Okapi([toks(t) for t in page_texts])
        qtoks = toks(q["question"])
        bm_scores = bm.get_scores(qtoks)
        bm_order = [corpus[i] for i in np.argsort(-bm_scores)]
        tf_order, tf_sims = tfidf_ranks(corpus, pages, q["question"])
        fused = rrf_fuse([bm_order, tf_order])
        top5 = fused[:5]
        hit5_bm = any(p in gold_pages for p in bm_order[:5])
        hit5_rrf = any(p in gold_pages for p in top5)
        gold_rank = next((r + 1 for r, p in enumerate(fused) if p in gold_pages), None)

        ev_pages = set(gold_pages) | {p for p in q["evidence_pages"] if p is not None}
        del_pages = [p for p in top5 if p not in ev_pages]
        backfill = [p for p in fused[5:] if p not in ev_pages]
        while len(del_pages) < 3 and backfill:
            del_pages.append(backfill.pop(0))

        swap_page = gold_pages[0]
        ok, rule, e_minus, repl, reason = build_swap(pages[swap_page], q["answer"], q["answer_format"])
        proxy = {
            "swap_ok": bool(ok), "rule": rule, "replacement": repl, "reason": reason,
            "e_minus_has_replacement": all(p.lower() in e_minus.lower()
                                           for p in str(repl).split(";") if p),
            "e_minus_free_of_gold": (not _standalone_re(q["answer"]).search(e_minus))
            if len(norm(q["answer"])) <= 2 else not (g and g in norm(e_minus)),
            "keyword_overlap": len(set(toks(q["question"])) & set(toks(e_minus.lower()))),
            "short_answer_needs_review": len(norm(q["answer"])) <= 2,
        }
        proxy["valid_proxy"] = (proxy["swap_ok"] and proxy["e_minus_has_replacement"]
                                and proxy["e_minus_free_of_gold"])

        skip = set(ev_pages) | {swap_page - 1, swap_page, swap_page + 1}
        cand = [p for p in bm_order if p not in skip and not (g and g in norm(pages[p]))
                and not (repl and repl.lower() in pages[p].lower())]
        benign_pages = cand[:2]
        benign_ok = len(benign_pages) == 2

        def world_chunks(idx_list, replace=None):
            out = []
            for p in idx_list:
                t = e_minus if (replace and p == swap_page) else pages[p]
                out.append({"page": p, "chars": len(t), "text": t,
                            "modified": bool(replace and p == swap_page)})
            return out

        worlds = {
            "plus": world_chunks(top5),
            "del": world_chunks(del_pages),
            "swap": world_chunks(top5, replace=ok),
            "benign": world_chunks(list(top5) + [p for p in benign_pages if p not in top5]),
        }
        toks_est = {k: sum(c["chars"] for c in v) // 4 for k, v in worlds.items()}
        tok_rows.append(toks_est)
        row = {
            "qid": q["qid"], "doc_id": q["doc_id"], "question": q["question"],
            "answer": q["answer"], "answer_format": q["answer_format"],
            "gold_pages_textview": gold_pages, "n_corpus_pages": len(corpus),
            "hit5_bm25": hit5_bm, "hit5_rrf": hit5_rrf, "gold_rank_rrf": gold_rank,
            "del_pages": del_pages, "benign_pages": benign_pages, "benign_ok": benign_ok,
            "swap_page": swap_page, "swap": proxy,
            "worlds": worlds, "token_est_per_world": toks_est,
        }
        rows.append(row)
        with open(os.path.join(worlds_dir, q["qid"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    def mean(key):
        vals = [t[key] for t in tok_rows if key in t]
        return round(sum(vals) / max(len(vals), 1))

    summary = {
        "questions": len(rows),
        "skipped_no_text": skipped,
        "hit5_bm25": round(sum(r["hit5_bm25"] for r in rows) / max(len(rows), 1), 3),
        "hit5_rrf": round(sum(r["hit5_rrf"] for r in rows) / max(len(rows), 1), 3),
        "gold_rank_rrf_median": sorted(
            [r["gold_rank_rrf"] for r in rows if r["gold_rank_rrf"] is not None] or [0])[
            len([r for r in rows if r["gold_rank_rrf"] is not None]) // 2] if any(
            r["gold_rank_rrf"] is not None for r in rows) else None,
        "swap_proxy_valid_rate": round(sum(r["swap"]["valid_proxy"] for r in rows) / max(len(rows), 1), 3),
        "benign_ok_rate": round(sum(r["benign_ok"] for r in rows) / max(len(rows), 1), 3),
        "mean_est_tokens_per_world": {k: mean(k) for k in ("plus", "del", "swap", "benign")},
    }
    json.dump(summary, open(summary_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
