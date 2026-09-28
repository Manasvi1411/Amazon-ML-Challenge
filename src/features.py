"""Pairwise feature extraction module using RapidFuzz and domain-specific heuristics."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

def compute_jaccard(tokens1: set, tokens2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0

def compute_number_similarity(nums1: set, nums2: set) -> Tuple[float, int]:
    """
    Compare numeric tokens (house numbers, postal codes, PIN codes).
    Returns (jaccard_similarity, overlap_count).
    """
    if not nums1 and not nums2:
        return 0.5, 0  # Neutral if neither address has numbers
    if not nums1 or not nums2:
        return 0.2, 0  # Low penalty if one has numbers and one doesn't
    
    inter = nums1.intersection(nums2)
    union = nums1.union(nums2)
    jaccard = len(inter) / len(union) if union else 0.0
    return jaccard, len(inter)

def extract_pairwise_features(
    s1_row: pd.Series,
    cand_row: pd.Series,
    s1_id: str,
    cand_id: str,
    blocking_score: float = 0.0,
) -> Dict[str, float]:
    """Extract a rich dictionary of similarity features for an (S1, S2/3) candidate pair."""
    # 1. Names
    s1_name = str(s1_row["clean_name"])
    cand_name = str(cand_row["clean_name"])
    s1_strip = str(s1_row["stripped_name"])
    cand_strip = str(cand_row["stripped_name"])

    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_jw = distance.JaroWinkler.similarity(s1_name, cand_name)

    strip_ratio = fuzz.ratio(s1_strip, cand_strip) / 100.0
    strip_token_sort = fuzz.token_sort_ratio(s1_strip, cand_strip) / 100.0
    strip_jw = distance.JaroWinkler.similarity(s1_strip, cand_strip)

    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    strip_exact = 1.0 if s1_strip and s1_strip == cand_strip else 0.0
    
    prefix_3 = 1.0 if s1_strip[:3] == cand_strip[:3] and len(s1_strip) >= 3 else 0.0
    prefix_5 = 1.0 if s1_strip[:5] == cand_strip[:5] and len(s1_strip) >= 5 else 0.0

    len1 = len(s1_name)
    len2 = len(cand_name)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = min(len1, len2) / max(len1, len2, 1)

    # Legal suffix match
    suff1 = str(s1_row.get("legal_suffix", ""))
    suff2 = str(cand_row.get("legal_suffix", ""))
    if suff1 and suff2:
        suffix_match = 1.0 if suff1 == suff2 else 0.0
    elif not suff1 and not suff2:
        suffix_match = 0.5
    else:
        suffix_match = 0.3

    # 2. Addresses
    s1_addr = str(s1_row["clean_address"])
    cand_addr = str(cand_row["clean_address"])

    addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    addr_jw = distance.JaroWinkler.similarity(s1_addr, cand_addr)

    # Word Jaccard
    s1_words = set(s1_addr.split())
    cand_words = set(cand_addr.split())
    addr_jaccard = compute_jaccard(s1_words, cand_words)

    # Number / postal code matching
    num_jaccard, num_overlap_cnt = compute_number_similarity(
        s1_row["address_numbers"], cand_row["address_numbers"]
    )

    # Landmark
    landmark_flag = float(bool(s1_row.get("has_landmark", 0)) and bool(cand_row.get("has_landmark", 0)))

    # 3. Combined & Contextual
    comb_token_set = (
        fuzz.token_set_ratio(s1_row["combined_text"], cand_row["combined_text"]) / 100.0
    )
    country_match = 1.0 if s1_row["norm_country"] == cand_row["norm_country"] else 0.0
    source_is_s3 = 1.0 if str(cand_id).startswith("S3-") else 0.0

    return {
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jw": name_jw,
        "strip_ratio": strip_ratio,
        "strip_token_sort": strip_token_sort,
        "strip_jw": strip_jw,
        "name_exact": name_exact,
        "strip_exact": strip_exact,
        "prefix_3": prefix_3,
        "prefix_5": prefix_5,
        "name_len_diff": float(name_len_diff),
        "name_len_ratio": name_len_ratio,
        "suffix_match": suffix_match,
        "addr_ratio": addr_ratio,
        "addr_partial": addr_partial,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jw": addr_jw,
        "addr_jaccard": addr_jaccard,
        "num_jaccard": num_jaccard,
        "num_overlap_cnt": float(num_overlap_cnt),
        "landmark_flag": landmark_flag,
        "comb_token_set": comb_token_set,
        "blocking_score": float(blocking_score),
        "country_match": country_match,
        "source_is_s3": source_is_s3,
    }

def build_feature_dataframe(
    candidates_dict: Dict[str, Dict[str, float]],
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Build feature matrix for all (s1_id, cand_id) candidate pairs.
    Returns (DataFrame of features, list of (s1_id, cand_id) pairs).
    """
    s1_lookup = df_s1.set_index("entity_id")
    target_all = pd.concat([df_s2, df_s3], ignore_index=True)
    target_lookup = target_all.set_index("entity_id")

    pair_list: List[Tuple[str, str]] = []
    feature_rows: List[Dict[str, float]] = []

    for s1_id, cands in candidates_dict.items():
        if s1_id not in s1_lookup.index:
            continue
        s1_row = s1_lookup.loc[s1_id]
        if isinstance(s1_row, pd.DataFrame):
            s1_row = s1_row.iloc[0]

        for cand_id, score in cands.items():
            if cand_id not in target_lookup.index:
                continue
            cand_row = target_lookup.loc[cand_id]
            if isinstance(cand_row, pd.DataFrame):
                cand_row = cand_row.iloc[0]

            feats = extract_pairwise_features(
                s1_row, cand_row, s1_id=str(s1_id), cand_id=str(cand_id), blocking_score=score
            )
            pair_list.append((s1_id, cand_id))
            feature_rows.append(feats)

    if not feature_rows:
        return pd.DataFrame(), []

    return pd.DataFrame(feature_rows), pair_list
