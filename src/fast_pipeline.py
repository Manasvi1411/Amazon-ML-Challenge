"""High-performance, memory-bounded Entity Resolution Pipeline for massive commercial datasets."""

import csv
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

COMMON_GEO_WORDS = {
    'near', 'opposite', 'behind', 'road', 'street', 'avenue', 'lane', 'drive', 'floor', 'suite',
    'rue', 'boulevard', 'place', 'allee', 'chemin', 'cours', 'null', 'bordeaux', 'paris', 'lyon',
    'marseille', 'lille', 'nantes', 'toulouse', 'nice', 'strasbourg', 'montpellier', 'rennes',
    'calais', 'dunkerque', 'roubaix', 'tourcoing', 'merignac', 'pessac', 'gironde', 'nord',
    'nouvelle', 'aquitaine', 'hauts', 'france', 'loire', 'atlantique', 'delhi', 'mumbai',
    'bangalore', 'chennai', 'hyderabad', 'kolkata', 'pune', 'ahmedabad', 'jaipur', 'surat',
    'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna',
    'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot',
    'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar', 'navi', 'allahabad', 'ranchi',
    'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur', 'madurai', 'raipur',
    'kota', 'guwahati', 'chandigarh', 'solapur', 'hubli', 'dharwad', 'bareilly', 'moradabad',
    'mysore', 'gurgaon', 'aligarh', 'jalandhar', 'tiruchirappalli', 'bhubaneswar', 'salem',
    'warangal', 'mira', 'bhayandar', 'jalgaon', 'guntur', 'thiruvananthapuram', 'bhiwandi',
    'saharanpur', 'amravati', 'noida', 'jamshedpur', 'bhilai', 'cuttack', 'firozabad', 'kochi',
    'nellore', 'bhavnagar', 'dehradun', 'durgapur', 'asansol', 'rourkela', 'nanded', 'kolhapur',
    'ajmer', 'akola', 'gulbarga', 'jamnagar', 'ujjain', 'loni', 'siliguri', 'jhansi', 'ulhasnagar',
    'jammu', 'sangli', 'mangalore', 'erode', 'belgaum', 'kurnool', 'ambattur', 'nagapattinam',
    'karnataka', 'maharashtra', 'tamil', 'nadu', 'uttar', 'pradesh', 'gujarat', 'rajasthan',
    'west', 'bengal', 'madhya', 'andhra', 'bihar', 'punjab', 'haryana', 'kerala', 'odisha',
    'telangana', 'assam', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'himachal', 'tripura',
    'meghalaya', 'manipur', 'nagaland', 'goa', 'arunachal', 'mizoram', 'sikkim'
}

def squash_name(name: str) -> str:
    """Normalize and squash name removing legal suffixes, domains, spaces, punctuation."""
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r'https?://|www\.', '', n)
    n = re.sub(r'\.(com|org|net|co|in|fr|io|gov|edu)\b', '', n)
    n = re.sub(r'\b(inc|corp|corporation|llc|llp|ltd|limited|pvt|private|co|company|holdings|group|enterprises|services|solutions|technologies|industries|sarl|sas|sasu|sa|eurl|sci|snc|gmbh|fils)\b', '', n)
    n = re.sub(r'[^a-zA-Z0-9]', '', n)
    return n

def get_name_tokens(name: str) -> List[str]:
    """Extract significant name tokens."""
    if not name:
        return []
    n = re.sub(r'\b(inc|corp|corporation|llc|llp|ltd|limited|pvt|private|co|company|holdings|group|enterprises|services|solutions|technologies|industries|sarl|sas|sasu|sa|eurl|sci|snc|gmbh|fils)\b', '', name, flags=re.I)
    tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', n) if len(t) >= 3]
    return tokens

def get_addr_numbers(addr: str) -> Set[str]:
    """Extract numeric/alphanumeric house & building tokens."""
    if not addr:
        return set()
    return set(re.findall(r'[a-zA-Z0-9]*\d+[a-zA-Z0-9]*', addr.lower()))

def get_street_tokens(addr: str) -> Set[str]:
    """Extract significant street name tokens (excluding cities, states, generic stopwords)."""
    if not addr:
        return set()
    tokens = [t.lower() for t in re.findall(r'[a-zA-Z]{3,}', addr) if t.lower() not in COMMON_GEO_WORDS]
    return set(tokens)

def get_street_part(addr: str) -> str:
    """Extract the primary street line (usually the part containing street numbers)."""
    if not addr:
        return ""
    parts = [p.strip() for p in addr.split(',') if p.strip()]
    for p in parts:
        if any(c.isdigit() for c in p):
            return p
    return parts[0] if parts else addr

def compute_pair_decision(s1_name: str, s1_addr: str, t_name: str, t_addr: str) -> Tuple[float, bool]:
    """
    Evaluate similarity between an S1 entity and a target candidate.
    Returns (candidate_score, is_match).
    """
    sq1 = squash_name(s1_name)
    sq2 = squash_name(t_name)

    # 1. Strong Name Match
    if sq1 and sq2:
        if sq1 == sq2:
            return 1.0, True
        if len(sq1) >= 6 and len(sq2) >= 6 and (sq1.startswith(sq2) or sq2.startswith(sq1)):
            return 0.95, True
        sq_ratio = fuzz.ratio(sq1, sq2)
        if sq_ratio >= 88:
            return sq_ratio / 100.0, True
        if sq_ratio >= 75:
            cand_score = sq_ratio / 100.0
        else:
            cand_score = 0.0
    else:
        cand_score = 0.0

    n_tsr = fuzz.token_sort_ratio(s1_name.lower(), t_name.lower())
    if n_tsr >= 85:
        return n_tsr / 100.0, True
    if n_tsr >= 70:
        cand_score = max(cand_score, n_tsr / 100.0)

    # 2. Moderate Name + Consistent Address
    if cand_score >= 0.70:
        a_tsr = fuzz.token_set_ratio(s1_addr.lower(), t_addr.lower())
        if a_tsr >= 60:
            return cand_score, True

    # 3. Strong Address Match (requires matching number AND matching street name!)
    s1_nums = get_addr_numbers(s1_addr)
    t_nums = get_addr_numbers(t_addr)
    num_inter = len(s1_nums.intersection(t_nums))

    if num_inter >= 1:
        st1 = get_street_part(s1_addr).lower()
        st2 = get_street_part(t_addr).lower()
        st_ratio = fuzz.ratio(st1, st2)
        st_tsr = fuzz.token_sort_ratio(st1, st2)
        if st_ratio >= 70 or st_tsr >= 72:
            return max(st_ratio, st_tsr) / 100.0, True

    if cand_score >= 0.60:
        return cand_score, False

    return 0.0, False

def run_country_resolution(
    country_label: str,
    s1_rows: List[Tuple[str, str, str]], # [(id, name, addr)]
    s2_file: Path,
    s3_file: Path,
    top_k_candidates: int = 15,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Run high-precision candidate blocking and matching for a single country partition.
    Returns (matches_map, candidates_map).
    """
    t0 = time.time()
    num_s1 = len(s1_rows)
    print(f"\n[{country_label}] Indexing {num_s1} Source 1 records...")

    s1_records: Dict[int, Tuple[str, str, str]] = {}
    s1_by_exact_sq: Dict[str, List[int]] = defaultdict(list)
    s1_by_prefix_sq: Dict[str, List[int]] = defaultdict(list)
    s1_by_token_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_by_addr_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, (s1_id, name, addr) in enumerate(s1_rows):
        s1_records[idx] = (s1_id, name, addr)

        # 1. Squashed name keys
        sq = squash_name(name)
        if sq:
            s1_by_exact_sq[sq].append(idx)
            if len(sq) >= 6:
                s1_by_prefix_sq[sq[:6]].append(idx)

        # 2. Token pair keys
        toks = get_name_tokens(name)
        if len(toks) >= 2:
            pair = tuple(sorted(toks[:2]))
            s1_by_token_pair[pair].append(idx)

        # 3. Address keys (street number + street token)
        nums = get_addr_numbers(addr)
        streets = get_street_tokens(addr)
        for num in list(nums)[:2]:
            for st in list(streets)[:2]:
                s1_by_addr_key[(num, st)].append(idx)

    # Prune unselective postings to guarantee high speed and precision
    MAX_POSTINGS = 25
    s1_by_prefix_sq = {k: v for k, v in s1_by_prefix_sq.items() if len(v) <= MAX_POSTINGS}
    s1_by_token_pair = {k: v for k, v in s1_by_token_pair.items() if len(v) <= MAX_POSTINGS}
    s1_by_addr_key = {k: v for k, v in s1_by_addr_key.items() if len(v) <= MAX_POSTINGS}

    print(f"[{country_label}] Index built in {time.time() - t0:.2f}s. Streaming target sources...")

    candidates_per_s1 = defaultdict(list)
    matches_per_s1 = defaultdict(list)

    total_targets_scanned = 0
    t1 = time.time()

    for s_file, s_name in [(s2_file, "Source 2"), (s3_file, "Source 3")]:
        file_targets = 0
        with open(s_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if len(row) < 4 or row[3] != country_label:
                    continue
                file_targets += 1
                tid, tname, taddr = row[0], row[1], row[2]

                # Fast candidate retrieval
                cand_idxs = set()

                sq = squash_name(tname)
                if sq in s1_by_exact_sq:
                    cand_idxs.update(s1_by_exact_sq[sq])
                if len(sq) >= 6 and sq[:6] in s1_by_prefix_sq:
                    cand_idxs.update(s1_by_prefix_sq[sq[:6]])

                toks = get_name_tokens(tname)
                if len(toks) >= 2:
                    pair = tuple(sorted(toks[:2]))
                    if pair in s1_by_token_pair:
                        cand_idxs.update(s1_by_token_pair[pair])

                nums = get_addr_numbers(taddr)
                streets = get_street_tokens(taddr)
                for num in list(nums)[:2]:
                    for st in list(streets)[:2]:
                        if (num, st) in s1_by_addr_key:
                            cand_idxs.update(s1_by_addr_key[(num, st)])

                if not cand_idxs:
                    continue

                for idx in cand_idxs:
                    s1_id, s1_name, s1_addr = s1_records[idx]
                    score, is_match = compute_pair_decision(s1_name, s1_addr, tname, taddr)
                    if score >= 0.60:
                        candidates_per_s1[s1_id].append((tid, score))
                        if is_match and score >= 0.70:
                            matches_per_s1[s1_id].append((tid, score))

        total_targets_scanned += file_targets
        print(f"[{country_label}] Scanned {file_targets} {s_name} records ({time.time() - t1:.1f}s elapsed)")

    # Post-process country results
    final_matches: Dict[str, List[str]] = {}
    final_candidates: Dict[str, List[str]] = {}

    for s1_id, _, _ in s1_rows:
        # Candidates
        cands = candidates_per_s1.get(s1_id, [])
        cands.sort(key=lambda x: x[1], reverse=True)
        cand_ids = [c[0] for c in cands[:top_k_candidates]]

        # Matches
        matches = matches_per_s1.get(s1_id, [])
        matches.sort(key=lambda x: x[1], reverse=True)
        match_ids = [m[0] for m in matches]

        # Ensure subset: all matches must appear in candidates
        for mid in match_ids:
            if mid not in cand_ids:
                cand_ids.append(mid)

        final_matches[s1_id] = match_ids
        final_candidates[s1_id] = cand_ids

    total_matches = sum(len(m) for m in final_matches.values())
    total_cands = sum(len(c) for c in final_candidates.values())
    print(f"[{country_label}] Complete! Found {total_matches} matches across {total_cands} candidates in {time.time() - t0:.1f}s.")

    return final_matches, final_candidates
