#!/usr/bin/env python3
"""
Train a Bible verse match classifier on APB data and score candidate matches.

Usage (from thesis/ root):

    python scripts/run_bible_match_classifier.py \
        --train-csv data/apb-matches-for-model-training.csv.gz \
        --matches-path results/app/eulogies/matches.parquet \
        --output-path results/app/eulogies/matches_scored.parquet \
        --threshold 0.01

Dependencies:
    pip install lightgbm sentence-transformers rapidfuzz pyarrow
"""

import argparse
import os
import re
from collections import Counter
from math import log

import numpy as np
import pandas as pd
from numpy.linalg import norm
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import token_sort_ratio
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# -----------------------------
# Text preprocessing & helpers
# -----------------------------

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def simple_tokenize(s: str):
    return re.findall(r"[a-z0-9']+", s.lower())


# -----------------------------
# Feature builder class
# -----------------------------

class BibleMatchClassifier:
    def __init__(
        self,
        sem_model_name: str = "all-MiniLM-L12-v2",
        random_state: int = 42,
    ):
        self.sem_model_name = sem_model_name
        self.sem_model = SentenceTransformer(sem_model_name)
        self.random_state = random_state

        self.tfidf_char = None
        self.tfidf_word = None

        # BM25 stats
        self.doc_freq = None
        self.N_docs = None
        self.avg_doc_len = None

        self.model = None  # LightGBM model

    # ----- BM25 -----

    def _build_bm25_stats(self, verses):
        verses_tokens = [simple_tokenize(t) for t in verses]
        doc_freq = Counter()
        for toks in verses_tokens:
            doc_freq.update(set(toks))
        N_docs = len(verses_tokens)
        avg_doc_len = np.mean([len(toks) for toks in verses_tokens]) if N_docs > 0 else 0.0
        self.doc_freq = doc_freq
        self.N_docs = N_docs
        self.avg_doc_len = avg_doc_len

    def _bm25_idf(self, term):
        df = self.doc_freq.get(term, 0)
        return log((self.N_docs - df + 0.5) / (df + 0.5)) if self.N_docs > 0 else 0.0

    def _bm25_score(self, query, doc, k1=1.5, b=0.75):
        if self.doc_freq is None:
            return 0.0
        q_tokens = simple_tokenize(query)
        d_tokens = simple_tokenize(doc)
        if not d_tokens:
            return 0.0
        doc_len = len(d_tokens)
        d_counts = Counter(d_tokens)
        score = 0.0
        for t in set(q_tokens):
            if t not in d_counts:
                continue
            idf = self._bm25_idf(t)
            tf = d_counts[t]
            denom = tf + k1 * (1 - b + b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * (tf * (k1 + 1) / denom)
        return score

    # ----- TF-IDF -----

    def _fit_tfidf(self, snippets, verses):
        corpus = pd.concat([snippets, verses], axis=0).tolist()

        self.tfidf_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_char.fit(corpus)
        self.tfidf_word.fit(corpus)

    def _tfidf_cosine(self, snippets, verses):
        from scipy.sparse import csr_matrix

        s_char = self.tfidf_char.transform(snippets)
        v_char = self.tfidf_char.transform(verses)
        s_word = self.tfidf_word.transform(snippets)
        v_word = self.tfidf_word.transform(verses)

        def safe_cosine(a: csr_matrix, b: csr_matrix):
            # sparse .sum(axis=1) → numpy.matrix, so convert to flat arrays
            num = a.multiply(b).sum(axis=1)
            num = np.asarray(num).ravel()

            s1 = a.multiply(a).sum(axis=1)
            s2 = b.multiply(b).sum(axis=1)
            s1 = np.asarray(s1).ravel()
            s2 = np.asarray(s2).ravel()

            denom = np.sqrt(s1) * np.sqrt(s2)
            denom = np.where(denom == 0, 1e-8, denom)

            return num / denom

        cos_char = safe_cosine(s_char, v_char)
        cos_word = safe_cosine(s_word, v_word)
        return cos_char, cos_word


    # ----- Lexical overlap -----

    def _lexical_overlap_features(self, snippets, verses):
        feats = {
            "tok_jaccard": [],
            "tok_contain_snip_in_verse": [],
            "tok_contain_verse_in_snip": [],
            "len_ratio": [],
            "char_lev_norm": [],
            "fuzz_token_sort_ratio": [],
        }

        for s, v in zip(snippets, verses):
            tok_s = simple_tokenize(s)
            tok_v = simple_tokenize(v)
            set_s = set(tok_s)
            set_v = set(tok_v)

            inter = len(set_s & set_v)
            union = len(set_s | set_v) or 1
            feats["tok_jaccard"].append(inter / union)

            len_s = len(set_s) or 1
            len_v = len(set_v) or 1
            feats["tok_contain_snip_in_verse"].append(inter / len_s)
            feats["tok_contain_verse_in_snip"].append(inter / len_v)

            ls = len(s)
            lv = len(v)
            if max(ls, lv) == 0:
                feats["len_ratio"].append(1.0)
            else:
                feats["len_ratio"].append(min(ls, lv) / max(ls, lv))

            if ls + lv == 0:
                feats["char_lev_norm"].append(1.0)
            else:
                dist = Levenshtein.distance(s, v)
                feats["char_lev_norm"].append(1 - dist / max(ls, lv))

            feats["fuzz_token_sort_ratio"].append(token_sort_ratio(s, v) / 100.0)

        for k in feats:
            feats[k] = np.array(feats[k], dtype=np.float32)
        return feats

    # ----- Semantic features -----

    def _semantic_pair_features(self, snippets, verses, batch_size=64):
        s_emb = self.sem_model.encode(
            list(snippets), batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True
        )
        v_emb = self.sem_model.encode(
            list(verses), batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True
        )

        dot = (s_emb * v_emb).sum(axis=1)
        denom = np.clip(norm(s_emb, axis=1) * norm(v_emb, axis=1), 1e-8, None)
        cos_sim = dot / denom

        abs_diff = np.abs(s_emb - v_emb)
        prod = s_emb * v_emb

        feat_abs_diff_mean = abs_diff.mean(axis=1)
        feat_prod_mean = prod.mean(axis=1)

        return cos_sim.astype(np.float32), feat_abs_diff_mean.astype(np.float32), feat_prod_mean.astype(np.float32)

    # ----- Feature stacking -----

    def _stack_features(self, cos_char, cos_word, lex, bm25, sem_cos, sem_diff_mean, sem_prod_mean):
        cols = [
            cos_char,
            cos_word,
            lex["tok_jaccard"],
            lex["tok_contain_snip_in_verse"],
            lex["tok_contain_verse_in_snip"],
            lex["len_ratio"],
            lex["char_lev_norm"],
            lex["fuzz_token_sort_ratio"],
            bm25,
            sem_cos,
            sem_diff_mean,
            sem_prod_mean,
        ]
        return np.vstack(cols).T.astype(np.float32)

    # ----- Public API -----

    def fit(self, df_train: pd.DataFrame):
        """
        df_train must have columns: 'snippet_norm', 'verse_norm', 'match' (0/1).
        """
        df = df_train.copy()

        # Optional: downsample negatives to ~1:4 ratio for training
        pos = df[df["match"] == 1]
        neg = df[df["match"] == 0]
        if len(pos) > 0 and len(neg) > 0:
            neg_sampled = neg.sample(
                n=min(len(neg), 4 * len(pos)),
                random_state=self.random_state
            )
            df_bal = pd.concat([pos, neg_sampled], axis=0).sample(
                frac=1.0,
                random_state=self.random_state
            )
        else:
            df_bal = df

        df_bal = df_bal.reset_index(drop=True)

        snippets = df_bal["snippet_norm"].astype(str)
        verses = df_bal["verse_norm"].astype(str)
        y = df_bal["match"].astype(int).values

        # Fit TF-IDF & BM25 stats on training-only verses/snippets
        self._fit_tfidf(snippets, verses)
        self._build_bm25_stats(verses)

        # Build features for training
        cos_char, cos_word = self._tfidf_cosine(snippets, verses)
        lex = self._lexical_overlap_features(snippets, verses)
        bm25 = np.array(
            [self._bm25_score(s, v) for s, v in zip(snippets, verses)],
            dtype=np.float32
        )
        sem_cos, sem_diff_mean, sem_prod_mean = self._semantic_pair_features(snippets, verses)

        X = self._stack_features(
            cos_char, cos_word, lex, bm25,
            sem_cos, sem_diff_mean, sem_prod_mean
        )

        # Simple train/valid split for a bit of regularization
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "force_row_wise": True,
            "verbosity": -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"]
        )

        # Optional: print quick metrics on validation set
        y_val_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        # Use threshold 0.01 for a quick sense of behavior
        y_val_pred = (y_val_proba >= 0.01).astype(int)
        prec = precision_score(y_val, y_val_pred)
        rec = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        print(f"[Validation @ threshold=0.01] precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")

    def predict_proba(self, snippets, verses):
        """
        Return probability of match (class 1) for each pair.
        snippets, verses: iterable of strings (aligned).
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        snippets = pd.Series(list(snippets)).astype(str)
        verses = pd.Series(list(verses)).astype(str)

        cos_char, cos_word = self._tfidf_cosine(snippets, verses)
        lex = self._lexical_overlap_features(snippets, verses)
        bm25 = np.array(
            [self._bm25_score(s, v) for s, v in zip(snippets, verses)],
            dtype=np.float32
        )
        sem_cos, sem_diff_mean, sem_prod_mean = self._semantic_pair_features(snippets, verses)

        X = self._stack_features(
            cos_char, cos_word, lex, bm25,
            sem_cos, sem_diff_mean, sem_prod_mean
        )

        proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        return proba


# -----------------------------
# Main script logic
# -----------------------------

def load_training_data(train_csv: str) -> pd.DataFrame:
    """
    Expect APB training file with columns:
        - 'match' (0/1)
        - 'most_unusual_phrase' (snippet)
        - 'verse' (verse text)
    Adjust here if your schema differs.
    """
    df = pd.read_csv(train_csv)
    if "match" not in df.columns:
        raise ValueError("Training CSV must have a 'match' column.")

    # Try to infer snippet/verse columns
    if "most_unusual_phrase" in df.columns:
        snippet_col = "most_unusual_phrase"
    elif "snippet" in df.columns:
        snippet_col = "snippet"
    else:
        raise ValueError("Could not find 'most_unusual_phrase' or 'snippet' column in training CSV.")

    if "verse" in df.columns:
        verse_col = "verse"
    elif "verse_text" in df.columns:
        verse_col = "verse_text"
    else:
        raise ValueError("Could not find 'verse' or 'verse_text' column in training CSV.")

    df["snippet_norm"] = df[snippet_col].apply(normalize_text)
    df["verse_norm"] = df[verse_col].apply(normalize_text)
    df["match"] = df["match"].fillna(0).astype(int)

    # Drop empty rows
    df = df[(df["snippet_norm"] != "") & (df["verse_norm"] != "")]
    df = df.reset_index(drop=True)
    return df[["snippet_norm", "verse_norm", "match"]]


def load_matches(matches_path: str) -> pd.DataFrame:
    """
    Load candidate matches from parquet or CSV.

    For your eulogies file, expected columns (from preview):
        - 'snippet_norm'
        - 'verse_text'
    """
    ext = os.path.splitext(matches_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(matches_path)
    else:
        df = pd.read_csv(matches_path)

    if "snippet_norm" not in df.columns:
        raise ValueError("matches file must have a 'snippet_norm' column.")
    if "verse_text" not in df.columns:
        raise ValueError("matches file must have a 'verse_text' column.")

    # Re-normalize snippet_norm just in case; normalize verse_text
    df["snippet_norm"] = df["snippet_norm"].apply(normalize_text)
    df["verse_norm"] = df["verse_text"].apply(normalize_text)

    return df


def save_matches(df: pd.DataFrame, output_path: str):
    ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if ext == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    print(f"Saved scored matches to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Bible match classifier and score candidate matches.")
    parser.add_argument("--train-csv", required=True, help="Path to APB training CSV (e.g., data/apb-matches-for-model-training.csv.gz).")
    parser.add_argument("--matches-path", required=True, help="Path to candidate matches file (parquet or CSV).")
    parser.add_argument("--output-path", required=True, help="Path to save scored matches (parquet or CSV).")
    parser.add_argument("--threshold", type=float, default=0.01, help="Decision threshold for match_label (default: 0.01).")
    parser.add_argument("--sem-model-name", type=str, default="all-MiniLM-L12-v2", help="SentenceTransformer model name.")
    args = parser.parse_args()

    print("Loading training data from:", args.train_csv)
    df_train = load_training_data(args.train_csv)
    print(f"Training rows: {len(df_train)}, positives: {df_train['match'].sum()}")

    clf = BibleMatchClassifier(sem_model_name=args.sem_model_name)
    clf.fit(df_train)

    print("Loading matches from:", args.matches_path)
    df_matches = load_matches(args.matches_path)
    print(f"Candidate rows: {len(df_matches)}")

    print("Scoring candidate pairs...")
    proba = clf.predict_proba(df_matches["snippet_norm"], df_matches["verse_norm"])
    df_matches["match_proba"] = proba
    df_matches["match_label"] = (proba >= args.threshold).astype(int)

    # Helpful diagnostic: show basic distribution
    print("Score summary:")
    print(df_matches["match_proba"].describe())
    print(f"Predicted positives at threshold {args.threshold}: {df_matches['match_label'].sum()}")

    save_matches(df_matches, args.output_path)


if __name__ == "__main__":
    main()
