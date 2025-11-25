#!/usr/bin/env python3
# scripts/bible_match_model.py

# base code for classifier, not a script

import re
from collections import Counter
from math import log

import numpy as np
import pandas as pd
from numpy.linalg import norm
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import token_sort_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib


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


class BibleMatchClassifier:
    """
    LightGBM + lexical + BM25 + semantic embedding classifier
    for (snippet, verse) → probability of match.
    """

    def __init__(
        self,
        sem_model_name: str | None = "all-MiniLM-L12-v2",
        random_state: int = 42,
    ):
        self.sem_model_name = sem_model_name
        self.sem_model = None
        if sem_model_name is not None:
            from sentence_transformers import SentenceTransformer
            self.sem_model = SentenceTransformer(sem_model_name)

        self.random_state = random_state

        self.tfidf_char = None
        self.tfidf_word = None

        # BM25 stats
        self.doc_freq = None
        self.N_docs = None
        self.avg_doc_len = None

        self.model: lgb.Booster | None = None

    # ----- BM25 -----

    def _build_bm25_stats(self, verses: pd.Series):
        verses_tokens = [simple_tokenize(t) for t in verses]
        doc_freq = Counter()
        for toks in verses_tokens:
            doc_freq.update(set(toks))
        self.N_docs = len(verses_tokens)
        self.doc_freq = doc_freq
        self.avg_doc_len = (
            np.mean([len(toks) for toks in verses_tokens]) if self.N_docs > 0 else 0.0
        )

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

    def _fit_tfidf(self, snippets: pd.Series, verses: pd.Series):
        corpus = pd.concat([snippets, verses], axis=0).tolist()
        self.tfidf_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
        )
        self.tfidf_char.fit(corpus)
        self.tfidf_word.fit(corpus)

    def _tfidf_cosine(self, snippets: pd.Series, verses: pd.Series):
        from scipy.sparse import csr_matrix

        s_char = self.tfidf_char.transform(snippets)
        v_char = self.tfidf_char.transform(verses)
        s_word = self.tfidf_word.transform(snippets)
        v_word = self.tfidf_word.transform(verses)

        def safe_cosine(a: csr_matrix, b: csr_matrix):
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
            feats["len_ratio"].append(1.0 if max(ls, lv) == 0 else min(ls, lv) / max(ls, lv))

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
        if self.sem_model is None:
            n = len(snippets)
            return (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            )

        s_emb = self.sem_model.encode(
            list(snippets),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        v_emb = self.sem_model.encode(
            list(verses),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        dot = (s_emb * v_emb).sum(axis=1)
        denom = np.clip(norm(s_emb, axis=1) * norm(v_emb, axis=1), 1e-8, None)
        cos_sim = dot / denom

        abs_diff = np.abs(s_emb - v_emb)
        prod = s_emb * v_emb

        feat_abs_diff_mean = abs_diff.mean(axis=1)
        feat_prod_mean = prod.mean(axis=1)

        return (
            cos_sim.astype(np.float32),
            feat_abs_diff_mean.astype(np.float32),
            feat_prod_mean.astype(np.float32),
        )

    # ----- Stack features -----

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

    def fit(self, df_train: pd.DataFrame, sample_weights: np.ndarray | None = None):
        """
        df_train must have: snippet_norm, verse_norm, match (0/1)
        """
        df = df_train.copy()

        pos = df[df["match"] == 1]
        neg = df[df["match"] == 0]
        if len(pos) > 0 and len(neg) > 0:
            neg_sampled = neg.sample(
                n=min(len(neg), 4 * len(pos)),
                random_state=self.random_state,
            )
            df_bal = pd.concat([pos, neg_sampled], axis=0).sample(
                frac=1.0, random_state=self.random_state
            )
        else:
            df_bal = df

        df_bal = df_bal.reset_index(drop=True)
        snippets = df_bal["snippet_norm"].astype(str)
        verses = df_bal["verse_norm"].astype(str)
        y = df_bal["match"].astype(int).values

        self._fit_tfidf(snippets, verses)
        self._build_bm25_stats(verses)

        cos_char, cos_word = self._tfidf_cosine(snippets, verses)
        lex = self._lexical_overlap_features(snippets, verses)
        bm25 = np.array(
            [self._bm25_score(s, v) for s, v in zip(snippets, verses)],
            dtype=np.float32,
        )
        sem_cos, sem_diff_mean, sem_prod_mean = self._semantic_pair_features(snippets, verses)

        X = self._stack_features(
            cos_char, cos_word, lex, bm25,
            sem_cos, sem_diff_mean, sem_prod_mean,
        )

        strat = y if len(np.unique(y)) > 1 else None
        if sample_weights is not None:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                X,
                y,
                sample_weights,
                test_size=0.2,
                random_state=self.random_state,
                stratify=strat,
            )
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=strat,
            )
            w_tr = w_val = None

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)

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

        # Quick sanity metric at threshold 0.01
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_tr_proba = self.model.predict(X_tr, num_iteration=self.model.best_iteration)
        y_val_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)

        def report(name, y_true, proba):
            preds = (proba >= 0.5).astype(int)
            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            print(f"[{name}] accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

        report("Train @0.50", y_tr, y_tr_proba)
        report("Valid @0.50", y_val, y_val_proba)

    def predict_proba(self, snippets, verses):
        if self.model is None:
            raise RuntimeError("Model not fitted; call fit() or load().")

        snippets = pd.Series(list(snippets)).astype(str)
        verses = pd.Series(list(verses)).astype(str)

        cos_char, cos_word = self._tfidf_cosine(snippets, verses)
        lex = self._lexical_overlap_features(snippets, verses)
        bm25 = np.array(
            [self._bm25_score(s, v) for s, v in zip(snippets, verses)],
            dtype=np.float32,
        )
        sem_cos, sem_diff_mean, sem_prod_mean = self._semantic_pair_features(snippets, verses)

        X = self._stack_features(
            cos_char, cos_word, lex, bm25,
            sem_cos, sem_diff_mean, sem_prod_mean,
        )
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    # ----- Save / load -----

    def save(self, path: str):
        """
        Save model + featurization state (not the semantic model weights).
        """
        state = {
            "sem_model_name": self.sem_model_name,
            "tfidf_char": self.tfidf_char,
            "tfidf_word": self.tfidf_word,
            "doc_freq": self.doc_freq,
            "N_docs": self.N_docs,
            "avg_doc_len": self.avg_doc_len,
            "lgbm_model_str": self.model.model_to_string(),
        }
        joblib.dump(state, path)
        print(f"[ok] saved classifier state -> {path}")

    @classmethod
    def load(cls, path: str) -> "BibleMatchClassifier":
        state = joblib.load(path)
        sem_name = state.get("sem_model_name", "all-MiniLM-L12-v2")
        obj = cls(sem_model_name=sem_name)

        obj.tfidf_char = state["tfidf_char"]
        obj.tfidf_word = state["tfidf_word"]
        obj.doc_freq = state["doc_freq"]
        obj.N_docs = state["N_docs"]
        obj.avg_doc_len = state["avg_doc_len"]
        obj.model = lgb.Booster(model_str=state["lgbm_model_str"])

        print(f"[ok] loaded classifier from {path} (sem_model_name={sem_name})")
        return obj
