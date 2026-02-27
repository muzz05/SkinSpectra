import re
import json
import math
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

CFG = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "data_path": "../data/ingredient_mapping.csv",
    "output_dir": "../models/nlp",
    "index_path": "../models/nlp/faiss_index.bin",
    "lookup_path": "../models/nlp/alias_lookup.json",
    "meta_path":   "../models/nlp/index_meta.json",
    "epochs": 5,
    "batch_size": 64,
    "warmup_ratio": 0.1,
    "eval_steps": 50,
    "max_seq_length": 64,
    "top_k": 5,
    "semantic_weight": 0.75,
    "fuzzy_weight": 0.25,
    "confidence_threshold": 0.55,
    "faiss_nlist": 8,
    "faiss_nprobe": 4,
}

def _clean(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

def load_and_expand(csv_path):
    df = pd.read_csv(csv_path)
    alias_map = {}
    corpus = []
    alias_columns = [
        "inci_name", "common_names", "trade_names",
        "chemical_aliases", "language_variants",
    ]
    for _, row in df.iterrows():
        inci = str(row["inci_name"]).strip()
        corpus.append(inci)
        for col in alias_columns:
            raw = str(row.get(col, ""))
            if not raw or raw == "nan":
                continue
            for alias in [p.strip() for p in raw.split("|") if p.strip()]:
                key = _clean(alias)
                if key:
                    alias_map[key] = inci
    corpus = list(dict.fromkeys(corpus))
    return df, alias_map, corpus

def build_training_pairs(df, alias_map):
    examples = []
    alias_columns = [
        "common_names", "trade_names", "chemical_aliases", "language_variants",
    ]
    for _, row in df.iterrows():
        inci = str(row["inci_name"]).strip()
        func = str(row.get("function", ""))
        all_aliases = [inci]
        for col in alias_columns:
            raw = str(row.get(col, ""))
            all_aliases.extend([p.strip() for p in raw.split("|") if p.strip() and p != "nan"])
        for alias in all_aliases:
            examples.append(InputExample(texts=[alias, inci]))
        if len(all_aliases) > 2:
            for i in range(min(len(all_aliases), 6)):
                for j in range(i + 1, min(len(all_aliases), 6)):
                    examples.append(InputExample(texts=[all_aliases[i], all_aliases[j]]))
        if func and func != "nan":
            examples.append(InputExample(texts=[f"{inci} used for {func}", inci]))
    return examples

def build_eval_set(df, alias_map, n_queries=40):
    rng = np.random.default_rng(99)
    rows = df.sample(min(n_queries, len(df)), random_state=99)
    queries_ev = {}
    corpus_ev = {}
    relevant = {}
    alias_columns = ["common_names", "trade_names", "chemical_aliases"]
    for idx, (_, row) in enumerate(rows.iterrows()):
        inci = str(row["inci_name"]).strip()
        cid = f"c{idx}"
        corpus_ev[cid] = inci
        aliases = []
        for col in alias_columns:
            raw = str(row.get(col, ""))
            aliases.extend([p.strip() for p in raw.split("|") if p.strip() and p != "nan"])
        if not aliases:
            aliases = [inci]
        qid = f"q{idx}"
        queries_ev[qid] = rng.choice(aliases)
        relevant[qid] = {cid}
    return queries_ev, corpus_ev, relevant

def train_model(examples, eval_tuple, corpus, cfg):
    model = SentenceTransformer(cfg["base_model"])
    model.max_seq_length = cfg["max_seq_length"]
    loader = DataLoader(examples, shuffle=True, batch_size=cfg["batch_size"])
    loss = losses.MultipleNegativesRankingLoss(model)
    queries_ev, corpus_ev, relevant_ev = eval_tuple
    evaluator = InformationRetrievalEvaluator(
        queries=queries_ev,
        corpus=corpus_ev,
        relevant_docs=relevant_ev,
        name="inci_eval",
        mrr_at_k=[1, 5],
        ndcg_at_k=[1, 5],
        show_progress_bar=False,
        write_csv=False,
    )
    warmup_steps = math.ceil(len(loader) * cfg["epochs"] * cfg["warmup_ratio"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=cfg["epochs"],
        warmup_steps=warmup_steps,
        evaluation_steps=cfg["eval_steps"],
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )
    return model

def build_index(model, corpus, cfg):
    embeddings = model.encode(
        corpus,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    dim = embeddings.shape[1]
    if FAISS_AVAILABLE:
        nlist  = min(cfg["faiss_nlist"], len(corpus))
        nprobe = min(cfg["faiss_nprobe"], nlist)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe
        faiss.write_index(index, cfg["index_path"])
    else:
        index = embeddings
        np.save(cfg["index_path"].replace(".bin", ".npy"), embeddings)
    with open(cfg["meta_path"], "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus, "dim": dim, "faiss": FAISS_AVAILABLE}, f, indent=2)
    return index, embeddings

class INCIMapper:
    def __init__(self, model, index, corpus, alias_map, embeddings, cfg):
        self.model      = model
        self.index      = index
        self.corpus     = corpus
        self.alias_map  = alias_map
        self.embeddings = embeddings
        self.cfg        = cfg
        self._faiss     = FAISS_AVAILABLE and not isinstance(index, np.ndarray)

    def map(self, raw_name, top_k=None):
        t0    = time.perf_counter()
        top_k = top_k or self.cfg["top_k"]
        query = raw_name.strip()
        clean = _clean(query)

        if clean in self.alias_map:
            return self._result(query, self.alias_map[clean], 1.0, "exact", [], t0)

        for key, inci in self.alias_map.items():
            if clean in key or key in clean:
                if len(clean) > 4 and abs(len(clean) - len(key)) < 6:
                    return self._result(query, inci, 0.97, "exact_partial", [], t0)

        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

        if self._faiss:
            scores, indices = self.index.search(q_emb, top_k)
            sem_scores  = scores[0].tolist()
            sem_indices = indices[0].tolist()
        else:
            sims        = (self.embeddings @ q_emb.T).squeeze()
            top_idx     = np.argsort(sims)[::-1][:top_k]
            sem_scores  = sims[top_idx].tolist()
            sem_indices = top_idx.tolist()

        fuzzy_results = rf_process.extract(
            clean, self.corpus, scorer=fuzz.token_sort_ratio, limit=top_k,
        )
        fuzzy_map = {res[0]: res[1] / 100.0 for res in fuzzy_results}

        sw = self.cfg["semantic_weight"]
        fw = self.cfg["fuzzy_weight"]
        candidates = {}
        for sc, idx in zip(sem_scores, sem_indices):
            if idx < 0 or idx >= len(self.corpus):
                continue
            name = self.corpus[idx]
            candidates[name] = sw * max(sc, 0) + fw * fuzzy_map.get(name, 0.0)
        for name, fz in fuzzy_map.items():
            if name not in candidates:
                candidates[name] = fw * fz

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return self._result(query, query, 0.0, "no_match", [], t0)

        best_inci, best_score = ranked[0]
        alternatives = [{"inci_name": n, "score": round(s, 4)} for n, s in ranked[1:top_k]]
        method = "ensemble" if best_score < 0.97 else "semantic"
        return self._result(query, best_inci, best_score, method, alternatives, t0)

    def batch_map(self, names):
        return [self.map(n) for n in names]

    @staticmethod
    def _confidence(score, threshold):
        if score >= 0.90:
            return "high"
        if score >= 0.75:
            return "medium"
        if score >= threshold:
            return "low"
        return "uncertain"

    def _result(self, query, inci, score, method, alts, t0):
        score = round(float(score), 4)
        return {
            "input"       : query,
            "inci_name"   : inci,
            "score"       : score,
            "confidence"  : self._confidence(score, self.cfg["confidence_threshold"]),
            "method"      : method,
            "alternatives": alts,
            "latency_ms"  : round((time.perf_counter() - t0) * 1000, 2),
        }

    @classmethod
    def load(cls, model_dir, cfg=CFG):
        model_dir = Path(model_dir)
        model = SentenceTransformer(str(model_dir))
        model.max_seq_length = cfg["max_seq_length"]
        with open(model_dir / "index_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        corpus = meta["corpus"]
        with open(model_dir / "alias_lookup.json", "r", encoding="utf-8") as f:
            alias_map = json.load(f)
        embeddings = None
        index_bin = str(model_dir / "faiss_index.bin")
        if meta.get("faiss") and FAISS_AVAILABLE:
            index = faiss.read_index(index_bin)
            index.nprobe = cfg["faiss_nprobe"]
        else:
            embeddings = np.load(index_bin.replace(".bin", ".npy"))
            index = embeddings
        return cls(model, index, corpus, alias_map, embeddings, cfg)

def main():
    parser = argparse.ArgumentParser(description="SkinSpectra NLP Layer Trainer")
    parser.add_argument("--data",   default=CFG["data_path"])
    parser.add_argument("--output", default=CFG["output_dir"])
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch",  type=int, default=CFG["batch_size"])
    parser.add_argument("--model",  default=CFG["base_model"])
    args = parser.parse_args()

    CFG["data_path"]  = args.data
    CFG["output_dir"] = args.output
    CFG["index_path"] = f"{args.output}/faiss_index.bin"
    CFG["lookup_path"]= f"{args.output}/alias_lookup.json"
    CFG["meta_path"]  = f"{args.output}/index_meta.json"
    CFG["epochs"]     = args.epochs
    CFG["batch_size"] = args.batch
    CFG["base_model"] = args.model

    df, alias_map, corpus = load_and_expand(CFG["data_path"])
    examples  = build_training_pairs(df, alias_map)
    eval_data = build_eval_set(df, alias_map)
    model     = train_model(examples, eval_data, corpus, CFG)
    index, _  = build_index(model, corpus, CFG)

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(CFG["lookup_path"], "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()