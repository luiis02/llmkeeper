"""Vector-direction pipeline for prompt injection detection.

1. Load and deduplicate all splits of DATASETS (~15.8k unique prompts).
2. Random 80/20 split (seeded).
3. Direction vector = L2-normalized embedding of each prompt.
4. 80%: keep only injections (label 1) and store their directions in a
   persistent vector DB (ChromaDB, cosine space).
5. 20%: for each prompt find the nearest injection direction (cosine distance).
6. PuLP MILP: choose a distance threshold t that
     maximizes  injections found (TP) - non-injections flagged (FP)
     subject to FP <= 5% of the 20% prompts.
"""

import logging
import os
import random
import time
from pathlib import Path

import chromadb
import numpy as np
import pulp
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("INJECTION_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DATASETS = ["deepset/prompt-injections", "S-Labs/prompt-injection-dataset"]
SEED = 42
TRAIN_RATIO = 0.8
MAX_FP_RATIO = 0.05
DB_DIR = Path(__file__).resolve().parents[1] / ".cache" / "injection_direction"
COLLECTION = "injection_directions"
LOG_FILE = Path(__file__).resolve().parent / "injection_direction.log"
EPS = 1e-6

log = logging.getLogger("injection_direction")


def load_prompts():
    """Union of all splits of all DATASETS, deduplicated by text.

    Texts that appear with conflicting labels are dropped entirely.
    """
    rows = []
    for name in DATASETS:
        log.info("loading dataset %s", name)
        ds = load_dataset(name)
        for split in ds:
            rows.extend((r["text"].strip(), r["label"]) for r in ds[split])

    labels_by_text = {}
    for text, label in rows:
        labels_by_text.setdefault(text, set()).add(label)
    conflicting = {t for t, ls in labels_by_text.items() if len(ls) > 1}

    seen, texts, labels = set(), [], []
    for text, label in rows:
        if text in seen or text in conflicting:
            continue
        seen.add(text)
        texts.append(text)
        labels.append(label)
    log.info("dedup: %d raw rows -> %d unique (%d conflicting texts dropped)",
             len(rows), len(texts), len(conflicting))
    return texts, np.array(labels)


def split_indices(n, seed=SEED, ratio=TRAIN_RATIO):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(round(ratio * n))
    return idx[:cut], idx[cut:]


GEMMA_PROMPT = "task: classification | query: "


class GemmaEncoder:
    """EmbeddingGemma-2 via plain transformers.

    sentence-transformers cannot build its multimodal processor with the
    installed transformers, so this reproduces its text pipeline: classification
    prompt, mean pooling over tokens.
    """

    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name).eval()

    def encode(self, texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False):
        torch = self._torch
        out = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        t0 = time.time()
        with torch.no_grad():
            for b, start in enumerate(range(0, len(texts), batch_size), 1):
                batch = [GEMMA_PROMPT + t for t in texts[start:start + batch_size]]
                x = self._tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                h = self._model(**x).last_hidden_state
                mask = x["attention_mask"].unsqueeze(-1).to(h.dtype)
                out.append(((h * mask).sum(1) / mask.sum(1)).float())
                if b % 25 == 0 or b == n_batches:
                    done = min(b * batch_size, len(texts))
                    elapsed = time.time() - t0
                    eta = elapsed / b * (n_batches - b)
                    log.info("encode: %d/%d texts (%.1f%%) | elapsed %.0fs | eta %.0fs",
                             done, len(texts), 100 * done / len(texts), elapsed, eta)
        return torch.cat(out).numpy()


def load_encoder():
    log.info("loading encoder %s", MODEL_NAME)
    if "embeddinggemma" in MODEL_NAME:
        return GemmaEncoder(MODEL_NAME)
    return SentenceTransformer(MODEL_NAME, device="cpu")


def directions(model, texts):
    emb = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-12, None)


def build_db(train_texts, train_labels, train_dirs):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    inj = [i for i, y in enumerate(train_labels) if y == 1]
    # Chroma caps the batch size per add() call, so insert in chunks
    for start in range(0, len(inj), 4096):
        chunk = inj[start:start + 4096]
        col.add(
            ids=[str(i) for i in chunk],
            embeddings=train_dirs[chunk].tolist(),
            documents=[train_texts[i] for i in chunk],
        )
        log.info("db: inserted %d/%d injection directions", min(start + 4096, len(inj)), len(inj))
    return col, len(inj)


def nearest_injection_distance(col, test_dirs):
    res = col.query(query_embeddings=test_dirs.tolist(), n_results=1)
    # Chroma cosine distance = 1 - cosine similarity, in [0, 2]
    return np.array([d[0] for d in res["distances"]])


def solve_threshold(dist, labels, max_fp):
    """MILP over a single threshold t. z_i = 1 iff dist_i <= t."""
    n = len(dist)
    M = float(dist.max() - dist.min()) + 1.0
    prob = pulp.LpProblem("injection_threshold", pulp.LpMaximize)
    t = pulp.LpVariable("threshold", lowBound=float(dist.min()), upBound=float(dist.max()))
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]
    # z_i = 1 => dist_i <= t ; z_i = 0 => dist_i > t
    for i in range(n):
        prob += dist[i] <= t + M * (1 - z[i])
        prob += dist[i] >= t + EPS - M * z[i]
    tp = pulp.lpSum(z[i] for i in range(n) if labels[i] == 1)
    fp = pulp.lpSum(z[i] for i in range(n) if labels[i] == 0)
    prob += fp <= max_fp
    prob += tp - fp  # maximize injections found, minimize false positives
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"PuLP status: {pulp.LpStatus[prob.status]}")
    pred = np.array([int(round(z[i].value())) for i in range(n)])
    return float(t.value()), pred


def metrics(pred, labels):
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "fp_rate": fp / max(fp + tn, 1),
    }


def setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for h in (file_handler, stream_handler):
        h.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)


def main():
    setup_logging()
    t_start = time.time()
    log.info("start | model=%s | log=%s", MODEL_NAME, LOG_FILE)

    texts, labels = load_prompts()
    log.info("prompts: %d | injections: %d | safe: %d",
             len(texts), int(labels.sum()), int((labels == 0).sum()))

    tr_idx, te_idx = split_indices(len(texts))
    tr_texts = [texts[i] for i in tr_idx]
    te_texts = [texts[i] for i in te_idx]
    tr_labels = labels[tr_idx]
    te_labels = labels[te_idx]
    log.info("split 80/20 (seed %d): train=%d test=%d", SEED, len(tr_idx), len(te_idx))

    model = load_encoder()
    log.info("encoding train set (%d)", len(tr_texts))
    tr_dirs = directions(model, tr_texts)
    log.info("encoding test set (%d)", len(te_texts))
    te_dirs = directions(model, te_texts)

    log.info("building vector DB")
    col, n_inj = build_db(tr_texts, tr_labels, tr_dirs)
    log.info("vector DB '%s' at %s: %d injection directions stored", COLLECTION, DB_DIR, n_inj)

    log.info("querying nearest injection for %d test prompts", len(te_idx))
    dist = nearest_injection_distance(col, te_dirs)
    max_fp = int(np.floor(MAX_FP_RATIO * len(te_idx)))
    log.info("max FP allowed: %d (5%% of %d test prompts)", max_fp, len(te_idx))

    log.info("solving MILP threshold (CBC)")
    threshold, pred = solve_threshold(dist, te_labels, max_fp)
    m = metrics(pred, te_labels)
    log.info("optimal threshold (cosine distance): %.4f", threshold)
    log.info("found %d/%d injections | FP %d/%d (%.1f%% of safe, %.1f%% of test)",
             m["TP"], int(te_labels.sum()), m["FP"], int((te_labels == 0).sum()),
             m["fp_rate"] * 100, m["FP"] / len(te_idx) * 100)
    log.info("recall=%.3f precision=%.3f TN=%d FN=%d", m["recall"], m["precision"], m["TN"], m["FN"])
    log.info("done in %.0fs", time.time() - t_start)


if __name__ == "__main__":
    main()
