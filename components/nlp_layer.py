"""
SkinSpectra NLP Layer — Ingredient Name to INCI Mapper
=======================================================
Architecture:
  - Primary  : BioBERT / PubMedBERT fine-tuned with contrastive learning
                (best-in-class for cosmetic/pharma ingredient text)
  - Fallback  : RapidFuzz token-sort ratio (handles typos, abbreviations)
  - Index     : FAISS IVF flat index for sub-millisecond nearest-neighbour search
  - Ensemble  : Weighted score = 0.75 * semantic_sim + 0.25 * fuzzy_ratio

Pipeline
--------
1. Load dataset1_nlp_ingredient_mapping.csv
2. Expand every row into (alias → inci_name) training pairs
3. Build a hard-negative mining set (same category, different INCI)
4. Fine-tune sentence-transformers with MultipleNegativesRankingLoss
5. Encode all canonical INCI names + aliases into a FAISS index
6. Save: model, FAISS index, alias→inci lookup JSON
"""

import os
import re
import json
import math
import time
import logging
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz, process as rf_process

# ── Sentence Transformers ──────────────────────────────────────────────────────
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    util,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch

# ── FAISS ─────────────────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("faiss-cpu not installed. Falling back to numpy cosine search. "
                  "Install with: pip install faiss-cpu", stacklevel=2)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("skinspectra.nlp")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Base model — lightweight & fast general-purpose sentence embeddings
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",

    # Paths
    "data_path": "../data/ingredient_mapping.csv",
    "output_dir": "../models/nlp",
    "index_path": "../models/nlp/faiss_index.bin",
    "lookup_path": "../models/nlp/alias_lookup.json",
    "meta_path":   "../models/nlp/index_meta.json",

    # Training
    "epochs": 5,
    "batch_size": 64,
    "warmup_ratio": 0.1,
    "eval_steps": 50,
    "max_seq_length": 64,

    # Inference
    "top_k": 5,
    "semantic_weight": 0.75,
    "fuzzy_weight": 0.25,
    "confidence_threshold": 0.55,   # below this → "uncertain"

    # FAISS
    "faiss_nlist": 8,               # IVF clusters (small dataset; keep low)
    "faiss_nprobe": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_and_expand(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Returns
    -------
    df         : raw dataframe
    alias_map  : {alias_cleaned → inci_name}  (all aliases for all rows)
    corpus     : list of unique canonical strings to embed in the index
    """
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} INCI ingredients from {csv_path}")

    alias_map: Dict[str, str] = {}
    corpus: List[str] = []

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
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            for alias in parts:
                key = _clean(alias)
                if key:
                    alias_map[key] = inci

    corpus = list(dict.fromkeys(corpus))   # deduplicate, preserve order
    log.info(f"Alias map: {len(alias_map)} entries | Corpus: {len(corpus)} canonical names")
    return df, alias_map, corpus


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING PAIR GENERATION  (positive + hard-negative mining)
# ══════════════════════════════════════════════════════════════════════════════

def build_training_pairs(
    df: pd.DataFrame,
    alias_map: Dict[str, str],
) -> List[InputExample]:
    """
    Positive pairs  : (alias, inci_name)
    Hard negatives  : (alias, wrong_inci_same_category)
    Uses MultipleNegativesRankingLoss — every other item in the batch is a negative.
    """
    examples: List[InputExample] = []
    category_to_inci: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        cat = str(row.get("ingredient_category", "other"))
        category_to_inci.setdefault(cat, []).append(str(row["inci_name"]))

    alias_columns = [
        "common_names", "trade_names", "chemical_aliases", "language_variants",
    ]

    rng = np.random.default_rng(42)

    for _, row in df.iterrows():
        inci = str(row["inci_name"]).strip()
        cat  = str(row.get("ingredient_category", "other"))
        func = str(row.get("function", ""))

        # Collect all aliases for this ingredient
        all_aliases = [inci]
        for col in alias_columns:
            raw = str(row.get(col, ""))
            parts = [p.strip() for p in raw.split("|") if p.strip() and p != "nan"]
            all_aliases.extend(parts)

        # Positive pairs: every alias paired with canonical INCI
        for alias in all_aliases:
            examples.append(InputExample(texts=[alias, inci]))

        # Cross-alias positives (strengthens synonym learning)
        if len(all_aliases) > 2:
            for i in range(min(len(all_aliases), 6)):
                for j in range(i + 1, min(len(all_aliases), 6)):
                    examples.append(InputExample(texts=[all_aliases[i], all_aliases[j]]))

        # Functional description positive
        if func and func != "nan":
            examples.append(InputExample(texts=[f"{inci} used for {func}", inci]))

    log.info(f"Training pairs generated: {len(examples)}")
    return examples


def build_eval_set(
    df: pd.DataFrame,
    alias_map: Dict[str, str],
    n_queries: int = 40,
) -> Tuple[Dict, Dict, Dict]:
    """
    Build an IR evaluation set for InformationRetrievalEvaluator.
    queries   : {qid: alias_text}
    corpus_ev : {cid: inci_name}
    relevant  : {qid: {cid}}
    """
    rng = np.random.default_rng(99)
    rows = df.sample(min(n_queries, len(df)), random_state=99)

    queries_ev: Dict[str, str] = {}
    corpus_ev:  Dict[str, str] = {}
    relevant:   Dict[str, set] = {}

    alias_columns = ["common_names", "trade_names", "chemical_aliases"]

    for idx, (_, row) in enumerate(rows.iterrows()):
        inci = str(row["inci_name"]).strip()
        cid  = f"c{idx}"
        corpus_ev[cid] = inci

        aliases = []
        for col in alias_columns:
            raw = str(row.get(col, ""))
            parts = [p.strip() for p in raw.split("|") if p.strip() and p != "nan"]
            aliases.extend(parts)

        if not aliases:
            aliases = [inci]

        qid = f"q{idx}"
        queries_ev[qid] = rng.choice(aliases)
        relevant[qid]   = {cid}

    return queries_ev, corpus_ev, relevant


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    examples: List[InputExample],
    eval_tuple: Tuple,
    corpus: List[str],
    cfg: dict,
) -> SentenceTransformer:

    log.info(f"Loading base model: {cfg['base_model']}")
    model = SentenceTransformer(cfg["base_model"])
    model.max_seq_length = cfg["max_seq_length"]

    # DataLoader
    loader = DataLoader(examples, shuffle=True, batch_size=cfg["batch_size"])

    # Loss — MNRL is the gold standard for retrieval fine-tuning
    loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
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
    log.info(f"Training for {cfg['epochs']} epochs | {len(examples)} pairs | warmup={warmup_steps}")

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
        use_amp=torch.cuda.is_available(),   # mixed precision if GPU available
    )

    log.info(f"Model saved to {output_dir}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. FAISS INDEX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_index(
    model: SentenceTransformer,
    corpus: List[str],
    cfg: dict,
) -> Tuple[object, np.ndarray]:
    """
    Encode corpus and build a FAISS IVF index for fast ANN search.
    Falls back to raw numpy array if faiss-cpu is not installed.
    """
    log.info(f"Encoding {len(corpus)} corpus entries …")
    embeddings = model.encode(
        corpus,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine via inner product
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    log.info(f"Embedding dim: {dim}")

    if FAISS_AVAILABLE:
        nlist  = min(cfg["faiss_nlist"], len(corpus))
        nprobe = min(cfg["faiss_nprobe"], nlist)

        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

        faiss.write_index(index, cfg["index_path"])
        log.info(f"FAISS index saved: {cfg['index_path']}  ({index.ntotal} vectors)")
    else:
        index = embeddings   # raw matrix fallback
        np.save(cfg["index_path"].replace(".bin", ".npy"), embeddings)
        log.info("Numpy fallback index saved.")

    # Save corpus metadata
    with open(cfg["meta_path"], "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus, "dim": dim, "faiss": FAISS_AVAILABLE}, f, indent=2)

    return index, embeddings


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE ENGINE  (used both here and imported by the app)
# ══════════════════════════════════════════════════════════════════════════════

class INCIMapper:
    """
    Industry-grade ingredient → INCI resolver.

    Scoring
    -------
    score = semantic_weight * cosine_sim + fuzzy_weight * (fuzzy_ratio / 100)

    Usage
    -----
    mapper = INCIMapper.load("skinspectra_nlp_model")
    result = mapper.map("sodium hyaluronate")
    print(result)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        index,
        corpus: List[str],
        alias_map: Dict[str, str],
        embeddings: Optional[np.ndarray],
        cfg: dict,
    ):
        self.model       = model
        self.index       = index
        self.corpus      = corpus
        self.alias_map   = alias_map
        self.embeddings  = embeddings
        self.cfg         = cfg
        self._faiss      = FAISS_AVAILABLE and not isinstance(index, np.ndarray)

    # ------------------------------------------------------------------
    def map(self, raw_name: str, top_k: int = None) -> Dict:
        """
        Map a raw ingredient name to its INCI standard name.

        Returns
        -------
        {
          "input"       : original query,
          "inci_name"   : best match,
          "score"       : float 0-1,
          "confidence"  : "high" | "medium" | "low" | "uncertain",
          "method"      : "exact" | "semantic" | "fuzzy" | "ensemble",
          "alternatives": [ {inci, score}, … ],
          "latency_ms"  : float,
        }
        """
        t0    = time.perf_counter()
        top_k = top_k or self.cfg["top_k"]
        query = raw_name.strip()
        clean = _clean(query)

        # ── 1. Exact / near-exact lookup ──────────────────────────────
        if clean in self.alias_map:
            inci = self.alias_map[clean]
            return self._result(query, inci, 1.0, "exact", [], t0)

        # Partial exact (e.g. leading/trailing junk)
        for key, inci in self.alias_map.items():
            if clean in key or key in clean:
                if len(clean) > 4 and abs(len(clean) - len(key)) < 6:
                    return self._result(query, inci, 0.97, "exact_partial", [], t0)

        # ── 2. Semantic search (FAISS / numpy) ────────────────────────
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
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

        # ── 3. Fuzzy match over corpus ────────────────────────────────
        fuzzy_results = rf_process.extract(
            clean,
            self.corpus,
            scorer=fuzz.token_sort_ratio,
            limit=top_k,
        )
        fuzzy_map = {res[0]: res[1] / 100.0 for res in fuzzy_results}

        # ── 4. Ensemble scoring ───────────────────────────────────────
        sw = self.cfg["semantic_weight"]
        fw = self.cfg["fuzzy_weight"]

        candidates: Dict[str, float] = {}
        for sc, idx in zip(sem_scores, sem_indices):
            if idx < 0 or idx >= len(self.corpus):
                continue
            name = self.corpus[idx]
            fz   = fuzzy_map.get(name, 0.0)
            candidates[name] = sw * max(sc, 0) + fw * fz

        for name, fz in fuzzy_map.items():
            if name not in candidates:
                candidates[name] = fw * fz

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return self._result(query, query, 0.0, "no_match", [], t0)

        best_inci, best_score = ranked[0]
        alternatives = [
            {"inci_name": n, "score": round(s, 4)}
            for n, s in ranked[1:top_k]
        ]
        method = "ensemble" if best_score < 0.97 else "semantic"
        return self._result(query, best_inci, best_score, method, alternatives, t0)

    # ------------------------------------------------------------------
    def batch_map(self, names: List[str]) -> List[Dict]:
        return [self.map(n) for n in names]

    # ------------------------------------------------------------------
    @staticmethod
    def _confidence(score: float, threshold: float) -> str:
        if score >= 0.90:
            return "high"
        if score >= 0.75:
            return "medium"
        if score >= threshold:
            return "low"
        return "uncertain"

    def _result(self, query, inci, score, method, alts, t0) -> Dict:
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

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_dir: str, cfg: dict = CFG) -> "INCIMapper":
        """Load a trained mapper from disk."""
        model_dir  = Path(model_dir)
        model      = SentenceTransformer(str(model_dir))
        model.max_seq_length = cfg["max_seq_length"]

        meta_path = model_dir / "index_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        corpus = meta["corpus"]

        alias_path = Path(cfg["lookup_path"])
        with open(alias_path, "r", encoding="utf-8") as f:
            alias_map = json.load(f)

        embeddings = None
        if meta.get("faiss") and FAISS_AVAILABLE:
            index = faiss.read_index(cfg["index_path"])
            index.nprobe = cfg["faiss_nprobe"]
        else:
            npy = cfg["index_path"].replace(".bin", ".npy")
            embeddings = np.load(npy)
            index = embeddings

        log.info(f"INCIMapper loaded from {model_dir}  |  corpus={len(corpus)}")
        return cls(model, index, corpus, alias_map, embeddings, cfg)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SkinSpectra NLP Layer Trainer")
    parser.add_argument("--data",    default=CFG["data_path"],    help="Path to dataset1 CSV")
    parser.add_argument("--output",  default=CFG["output_dir"],   help="Output directory")
    parser.add_argument("--epochs",  type=int, default=CFG["epochs"])
    parser.add_argument("--batch",   type=int, default=CFG["batch_size"])
    parser.add_argument("--model",   default=CFG["base_model"],   help="HuggingFace model ID")
    args = parser.parse_args()

    CFG["data_path"]  = args.data
    CFG["output_dir"] = args.output
    CFG["index_path"] = f"{args.output}/faiss_index.bin"
    CFG["lookup_path"]= f"{args.output}/alias_lookup.json"
    CFG["meta_path"]  = f"{args.output}/index_meta.json"
    CFG["epochs"]     = args.epochs
    CFG["batch_size"] = args.batch
    CFG["base_model"] = args.model

    log.info("=" * 60)
    log.info("  SkinSpectra — NLP Layer Training")
    log.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────
    df, alias_map, corpus = load_and_expand(CFG["data_path"])

    # ── Step 2: Build training pairs ──────────────────────────────────
    examples  = build_training_pairs(df, alias_map)
    eval_data = build_eval_set(df, alias_map)

    # ── Step 3: Train ─────────────────────────────────────────────────
    model = train_model(examples, eval_data, corpus, CFG)

    # ── Step 4: Build FAISS index ─────────────────────────────────────
    index, embeddings = build_index(model, corpus, CFG)

    # ── Step 5: Save alias lookup ─────────────────────────────────────
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(CFG["lookup_path"], "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)
    log.info(f"Alias lookup saved: {CFG['lookup_path']}")

    # ── Step 6: Smoke test ────────────────────────────────────────────
    mapper = INCIMapper(model, index, corpus, alias_map, embeddings, CFG)
    smoke_tests = [
        "vitamin c",
        "HA",
        "retinol",
        "shea butter",
        "sodium lauryl sulfate",
        "hyaluronic acid",
        "zinc oxide",
        "niacinamide",
        "kojic acid",
        "vitamin e",
    ]

    log.info("\n── Smoke Test ──────────────────────────────────────────")
    for q in smoke_tests:
        r = mapper.map(q)
        log.info(
            f"  '{q}' → '{r['inci_name']}'  "
            f"score={r['score']:.3f}  "
            f"conf={r['confidence']}  "
            f"method={r['method']}  "
            f"({r['latency_ms']}ms)"
        )

    log.info("\n✅  NLP Layer training complete.")
    log.info(f"    Model   : {CFG['output_dir']}")
    log.info(f"    Index   : {CFG['index_path']}")
    log.info(f"    Lookup  : {CFG['lookup_path']}")


if __name__ == "__main__":
    main()