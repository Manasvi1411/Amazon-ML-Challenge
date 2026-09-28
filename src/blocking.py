"""Multi-strategy blocking and candidate generation module for Entity Resolution."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

class BlockingEngine:
    """High-recall candidate generation engine using multi-view indexing and sparse cosine similarity."""

    def __init__(self, top_k_per_s1: int = 40, min_sim_threshold: float = 0.22):
        self.top_k_per_s1 = top_k_per_s1
        self.min_sim_threshold = min_sim_threshold

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate candidate pairs between S1 and {S2, S3}.
        Returns a dict mapping:
            s1_id -> {candidate_id: max_blocking_score}
        """
        # Combine S2 and S3 for unified retrieval
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)
        
        # Ensure all S1 entities have an entry even if empty
        candidates_per_s1: Dict[str, Dict[str, float]] = {
            s1_id: {} for s1_id in df_s1["entity_id"]
        }

        # Partition by normalized country (with UNKNOWN as wildcard)
        all_countries = set(df_s1["norm_country"]).union(set(df_target["norm_country"]))

        for country in all_countries:
            if country == "UNKNOWN":
                sub_s1 = df_s1
                sub_target = df_target
            else:
                sub_s1 = df_s1[df_s1["norm_country"] == country]
                # Include target records matching this country or UNKNOWN
                sub_target = df_target[df_target["norm_country"].isin([country, "UNKNOWN"])]

            if sub_s1.empty or sub_target.empty:
                continue

            self._retrieve_country_block(sub_s1, sub_target, candidates_per_s1)

        # Enforce top-K limit per S1 entity
        final_candidates: Dict[str, Dict[str, float]] = {}
        for s1_id, cand_dict in candidates_per_s1.items():
            if not cand_dict:
                final_candidates[s1_id] = {}
            else:
                # Sort by score descending and take top-K
                sorted_cands = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)[
                    : self.top_k_per_s1
                ]
                final_candidates[s1_id] = dict(sorted_cands)

        return final_candidates

    def _retrieve_country_block(
        self,
        sub_s1: pd.DataFrame,
        sub_target: pd.DataFrame,
        candidates_map: Dict[str, Dict[str, float]],
    ) -> None:
        """Run multi-index candidate retrieval within a country partition."""
        s1_ids = sub_s1["entity_id"].values
        target_ids = sub_target["entity_id"].values

        # 1. Character 3-4 gram TF-IDF on stripped_name (captures typos, transliterations, prefixes)
        try:
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 4),
                min_df=1,
                sublinear_tf=True,
            )
            corpus = list(sub_s1["stripped_name"].fillna("")) + list(
                sub_target["stripped_name"].fillna("")
            )
            char_vec.fit(corpus)
            s1_char = char_vec.transform(sub_s1["stripped_name"].fillna(""))
            target_char = char_vec.transform(sub_target["stripped_name"].fillna(""))
            sim_char = s1_char.dot(target_char.T)
            self._accumulate_sparse_sims(s1_ids, target_ids, sim_char, candidates_map, weight=1.0)
        except Exception:
            pass

        # 2. Word n-gram TF-IDF on clean_name
        try:
            word_vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            corpus_word = list(sub_s1["clean_name"].fillna("")) + list(
                sub_target["clean_name"].fillna("")
            )
            word_vec.fit(corpus_word)
            s1_word = word_vec.transform(sub_s1["clean_name"].fillna(""))
            target_word = word_vec.transform(sub_target["clean_name"].fillna(""))
            sim_word = s1_word.dot(target_word.T)
            self._accumulate_sparse_sims(s1_ids, target_ids, sim_word, candidates_map, weight=1.0)
        except Exception:
            pass

        # 3. Combined text (name + address) TF-IDF
        try:
            comb_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            )
            corpus_comb = list(sub_s1["combined_text"].fillna("")) + list(
                sub_target["combined_text"].fillna("")
            )
            comb_vec.fit(corpus_comb)
            s1_comb = comb_vec.transform(sub_s1["combined_text"].fillna(""))
            target_comb = comb_vec.transform(sub_target["combined_text"].fillna(""))
            sim_comb = s1_comb.dot(target_comb.T)
            self._accumulate_sparse_sims(s1_ids, target_ids, sim_comb, candidates_map, weight=1.1)
        except Exception:
            pass

        # 4. Inverted index on significant name tokens (handles word transpositions)
        target_token_index = defaultdict(list)
        for idx, (t_id, s_name) in enumerate(zip(target_ids, sub_target["stripped_name"])):
            tokens = [t for t in str(s_name).split() if len(t) >= 3]
            for tok in tokens:
                target_token_index[tok].append(t_id)

        for s1_id, s1_name in zip(s1_ids, sub_s1["stripped_name"]):
            s1_tokens = [t for t in str(s1_name).split() if len(t) >= 3]
            if len(s1_tokens) >= 2:
                # Count shared significant tokens
                cand_counts = defaultdict(int)
                for tok in s1_tokens:
                    for t_id in target_token_index.get(tok, []):
                        cand_counts[t_id] += 1
                for t_id, cnt in cand_counts.items():
                    if cnt >= 2:  # Shared at least 2 significant words
                        score = float(cnt) / max(len(s1_tokens), 1)
                        if score > candidates_map[s1_id].get(t_id, 0.0):
                            candidates_map[s1_id][t_id] = score

    def _accumulate_sparse_sims(
        self,
        s1_ids: np.ndarray,
        target_ids: np.ndarray,
        sim_matrix: csr_matrix,
        candidates_map: Dict[str, Dict[str, float]],
        weight: float = 1.0,
    ) -> None:
        """Extract high scoring candidate pairs from a sparse similarity matrix."""
        sim_matrix = sim_matrix.tocsr()
        num_rows = sim_matrix.shape[0]

        for i in range(num_rows):
            s1_id = s1_ids[i]
            row_start = sim_matrix.indptr[i]
            row_end = sim_matrix.indptr[i + 1]

            if row_start == row_end:
                continue

            col_indices = sim_matrix.indices[row_start:row_end]
            data = sim_matrix.data[row_start:row_end] * weight

            # Filter by minimum similarity threshold
            mask = data >= self.min_sim_threshold
            valid_cols = col_indices[mask]
            valid_scores = data[mask]

            for col_idx, score in zip(valid_cols, valid_scores):
                t_id = target_ids[col_idx]
                prev = candidates_map[s1_id].get(t_id, 0.0)
                if score > prev:
                    candidates_map[s1_id][t_id] = float(score)
