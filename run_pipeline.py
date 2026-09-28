#!/usr/bin/env python3
"""
Production Entity Resolution Pipeline for Business Entity Resolution Challenge.
Enhanced with high-precision state conflict guarding, Unicode NFKD normalization,
robust address digit extraction, and singleton false-merge protection.
"""

import argparse
import csv
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Normalization & Utility Functions
# ---------------------------------------------------------------------------

def unidecode_str(text: str) -> str:
    """Strip accents and diacritics using Unicode NFKD normalization."""
    if not text:
        return ""
    return "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

# ONLY actual legal corporate structures (do not strip business words like technologies, services, holdings)
LEGAL_WORDS = r'\b(inc|incorporated|corp|corporation|llc|llp|ltd|limited|pvt|private|sarl|sas|sasu|sa|eurl|sci|snc|gmbh|fils|pllc|pc|lp|m/s|ms)\b'

GENERIC_SINGLE_WORDS = {
    'consulting', 'enterprises', 'international', 'services', 'solutions',
    'technologies', 'management', 'properties', 'holdings', 'associates',
    'investments', 'marketing', 'ventures', 'industries', 'logistics',
    'trading', 'global', 'creative', 'hotel', 'products', 'commercial'
}

CITY_PREFIXES = {
    'kolkat', 'bangal', 'chenna', 'mumbai', 'hydera', 'puneci', 'delhic',
    'jaipur', 'ahmeda', 'boston', 'chicag', 'dallas', 'housto', 'austin',
    'seattl', 'denver', 'miamic', 'parisc', 'lyonci', 'marsei', 'bordeau', 'nantes'
}

US_STATES = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
    'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
    'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
    'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
    'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
    'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
    'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
}
US_CODES = set(US_STATES.values())

IN_STATES = {
    'andhra pradesh': 'AP', 'arunachal pradesh': 'AR', 'assam': 'AS', 'bihar': 'BR',
    'chhattisgarh': 'CG', 'goa': 'GA', 'gujarat': 'GJ', 'haryana': 'HR',
    'himachal pradesh': 'HP', 'jharkhand': 'JH', 'karnataka': 'KA', 'kerala': 'KL',
    'madhya pradesh': 'MP', 'maharashtra': 'MH', 'manipur': 'MN', 'meghalaya': 'ML',
    'mizoram': 'MZ', 'nagaland': 'NL', 'odisha': 'OD', 'orissa': 'OD', 'punjab': 'PB',
    'rajasthan': 'RJ', 'sikkim': 'SK', 'tamil nadu': 'TN', 'telangana': 'TS',
    'tripura': 'TR', 'uttar pradesh': 'UP', 'uttarakhand': 'UK', 'west bengal': 'WB',
    'delhi': 'DL', 'chandigarh': 'CH', 'puducherry': 'PY', 'pondicherry': 'PY',
    'jammu': 'JK', 'kashmir': 'JK', 'ladakh': 'LA'
}
IN_CODES = set(IN_STATES.values())

def get_us_state(addr: str) -> Optional[str]:
    if not addr:
        return None
    a = addr.lower()
    for name, code in US_STATES.items():
        if re.search(r'\b' + name + r'\b', a):
            return code
    tokens = re.findall(r'\b[A-Z]{2}\b', addr)
    for tok in reversed(tokens):
        if tok in US_CODES:
            return tok
    return None

def get_in_state(addr: str) -> Optional[str]:
    if not addr:
        return None
    a = addr.lower()
    for name, code in IN_STATES.items():
        if re.search(r'\b' + name + r'\b', a):
            return code
    tokens = re.findall(r'\b[A-Z]{2}\b', addr)
    for tok in reversed(tokens):
        if tok in IN_CODES:
            return tok
    return None

def get_fr_dept(addr: str) -> Optional[str]:
    if not addr:
        return None
    postcodes = re.findall(r'\b\d{5}\b', addr)
    if postcodes:
        return postcodes[0][:2]
    return None

def get_region_code(addr: str, country: str) -> Optional[str]:
    if country == 'US':
        return get_us_state(addr)
    elif country == 'India':
        return get_in_state(addr)
    elif country == 'France':
        return get_fr_dept(addr)
    return None

def clean_name_tokens(name: str) -> List[List[str]]:
    """Clean and tokenize business name, supporting d/b/a / trading-as splits."""
    if not name:
        return []
    n = unidecode_str(name).lower()
    n = re.sub(r'https?://|www\.', '', n)
    n = re.sub(r'\.(com|org|net|co|in|fr|io|gov|edu)\b', '', n)
    parts = re.split(r'\b(d/?b/?a|t/?a|a/?k/?a|trading as|doing business as)\b', n)
    results = []
    for p in parts:
        p_clean = re.sub(LEGAL_WORDS, ' ', p)
        p_clean = re.sub(r'[^a-z0-9]', ' ', p_clean)
        toks = [t for t in p_clean.split() if len(t) >= 2]
        if toks:
            results.append(toks)
    return results

def get_squashed(toks: List[str]) -> str:
    return "".join(toks)

def extract_addr_digits(addr: str) -> Set[str]:
    """Extract numeric building, house, and suite numbers stripping leading zeros."""
    if not addr:
        return set()
    nums = re.findall(r'\d+', addr)
    res = set()
    for n in nums:
        try:
            res.add(str(int(n)))
        except ValueError:
            pass
    return res

# ---------------------------------------------------------------------------
# High-Precision Pair Decision Logic
# ---------------------------------------------------------------------------

def compute_pair_decision(
    s1_name: str,
    s1_addr: str,
    t_name: str,
    t_addr: str,
    country: str,
) -> Tuple[float, bool]:
    """
    Evaluates similarity between an S1 entity and a target candidate with
    strict state and number conflict verification to protect macro F0.5.
    Returns (candidate_score, is_match).
    """
    # 1. State / Regional Conflict Guard
    s1_reg = get_region_code(s1_addr, country)
    t_reg = get_region_code(t_addr, country)
    if s1_reg and t_reg and s1_reg != t_reg:
        return 0.0, False

    s1_tok_lists = clean_name_tokens(s1_name)
    t_tok_lists = clean_name_tokens(t_name)

    # 2. Exact Name Matching Check
    name_exact = False
    longest_sq = 0
    for s1_toks in s1_tok_lists:
        sq1 = get_squashed(s1_toks)
        longest_sq = max(longest_sq, len(sq1))
        for t_toks in t_tok_lists:
            sq2 = get_squashed(t_toks)
            if sq1 and sq2 and sq1 == sq2:
                name_exact = True
                break
        if name_exact:
            break

    # 3. Fuzzy Name Similarities
    s1_clean = re.sub(r'[^a-z0-9]', ' ', unidecode_str(s1_name).lower())
    t_clean = re.sub(r'[^a-z0-9]', ' ', unidecode_str(t_name).lower())
    tsr = fuzz.token_sort_ratio(s1_clean, t_clean)

    # 4. Address Numbers & Overlap
    s1_nums = extract_addr_digits(s1_addr)
    t_nums = extract_addr_digits(t_addr)
    has_num_conflict = False
    num_overlap = False
    if s1_nums and t_nums:
        if len(s1_nums.intersection(t_nums)) == 0:
            has_num_conflict = True
        else:
            num_overlap = True

    s1_a_clean = unidecode_str(s1_addr).lower()
    t_a_clean = unidecode_str(t_addr).lower()
    addr_ratio = fuzz.token_set_ratio(s1_a_clean, t_a_clean) if s1_a_clean and t_a_clean else 0

    # 5. Decision Rules (Precision-Optimized)
    is_match = False
    score = max(tsr / 100.0, 0.50 if name_exact else 0.0)

    # Condition A: Exact canonical name match
    if name_exact:
        if not has_num_conflict:
            if not t_a_clean or not s1_a_clean:
                # Protect singletons: never match empty address if name is short or generic
                sq_name = s1_tok_lists[0][0] if len(s1_tok_lists[0]) == 1 else ""
                if longest_sq >= 8 and sq_name not in GENERIC_SINGLE_WORDS:
                    is_match = True
                    score = 0.95
            else:
                if addr_ratio >= 35 or num_overlap:
                    is_match = True
                    score = 0.98

    # Condition B: High token sort ratio (word order permutations)
    elif tsr >= 85 and not has_num_conflict:
        if num_overlap or addr_ratio >= 45:
            is_match = True
            score = max(score, 0.90)

    # Condition C: Strong address match with significant shared business words
    elif num_overlap and addr_ratio >= 65 and not has_num_conflict:
        if tsr >= 55:
            is_match = True
            score = max(score, 0.85)

    return score, is_match

# ---------------------------------------------------------------------------
# Country Partition Engine
# ---------------------------------------------------------------------------

def run_country_partition(
    country_label: str,
    s1_rows: List[Tuple[str, str, str]],
    test_dir: Path,
    top_k_candidates: int = 15,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Run candidate blocking and matching for a single country partition.
    Returns (matches_map, candidates_map).
    """
    t0 = time.time()
    num_s1 = len(s1_rows)
    print(f"\n[{country_label}] Indexing {num_s1:,} Source 1 records...")

    s1_records: Dict[int, Tuple[str, str, str]] = {}
    s1_by_exact_sq: Dict[str, List[int]] = defaultdict(list)
    s1_by_prefix_sq: Dict[str, List[int]] = defaultdict(list)
    s1_by_token_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, (s1_id, name, addr) in enumerate(s1_rows):
        s1_records[idx] = (s1_id, name, addr)
        tok_lists = clean_name_tokens(name)
        for toks in tok_lists:
            sq = get_squashed(toks)
            if sq:
                s1_by_exact_sq[sq].append(idx)
                if len(sq) >= 6:
                    s1_by_prefix_sq[sq[:6]].append(idx)
            if len(toks) >= 2:
                pair = tuple(sorted(toks[:2]))
                s1_by_token_pair[pair].append(idx)

    # Prune unselective postings and top city prefixes to prevent runaway false candidates
    MAX_P = 25
    s1_by_prefix_sq = {
        k: v for k, v in s1_by_prefix_sq.items()
        if k not in CITY_PREFIXES and len(v) <= MAX_P
    }
    s1_by_token_pair = {k: v for k, v in s1_by_token_pair.items() if len(v) <= MAX_P}

    print(f"[{country_label}] Index built in {time.time() - t0:.2f}s. Streaming target sources...")

    candidates_per_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    matches_per_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    t1 = time.time()

    for s_name, s_file in [("Source 2", test_dir / "test_source2.tsv"), ("Source 3", test_dir / "test_source3.tsv")]:
        file_targets = 0
        with open(s_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if len(row) < 4 or row[3] != country_label:
                    continue
                file_targets += 1
                tid, tname, taddr = row[0], row[1], row[2]

                cand_idxs = set()
                t_tok_lists = clean_name_tokens(tname)
                for toks in t_tok_lists:
                    sq = get_squashed(toks)
                    if sq in s1_by_exact_sq:
                        cand_idxs.update(s1_by_exact_sq[sq])
                    if len(sq) >= 6 and sq[:6] in s1_by_prefix_sq:
                        cand_idxs.update(s1_by_prefix_sq[sq[:6]])
                    if len(toks) >= 2:
                        pair = tuple(sorted(toks[:2]))
                        if pair in s1_by_token_pair:
                            cand_idxs.update(s1_by_token_pair[pair])

                if not cand_idxs:
                    continue

                for idx in cand_idxs:
                    s1_id, s1_name, s1_addr = s1_records[idx]
                    score, is_match = compute_pair_decision(
                        s1_name, s1_addr, tname, taddr, country_label
                    )
                    if score >= 0.55:
                        candidates_per_s1[s1_id].append((tid, score))
                        if is_match:
                            matches_per_s1[s1_id].append((tid, score))

        print(f"[{country_label}] Scanned {file_targets:,} {s_name} records ({time.time() - t1:.1f}s elapsed)")

    # Post-process results for this country
    final_matches: Dict[str, List[str]] = {}
    final_candidates: Dict[str, List[str]] = {}

    for s1_id, _, _ in s1_rows:
        # Candidates (top-k sorted by score)
        cands = candidates_per_s1.get(s1_id, [])
        cands.sort(key=lambda x: x[1], reverse=True)
        cand_ids = [c[0] for c in cands[:top_k_candidates]]

        # Matches (sorted by score)
        matches = matches_per_s1.get(s1_id, [])
        matches.sort(key=lambda x: x[1], reverse=True)
        match_ids = [m[0] for m in matches]

        # Enforce subset rule strictly: every match MUST be in candidate_pairs
        for mid in match_ids:
            if mid not in cand_ids:
                cand_ids.append(mid)

        final_matches[s1_id] = match_ids
        final_candidates[s1_id] = cand_ids

    total_matches = sum(len(m) for m in final_matches.values())
    total_cands = sum(len(c) for c in final_candidates.values())
    singletons = sum(1 for m in final_matches.values() if not m)
    print(f"[{country_label}] Completed in {time.time() - t0:.1f}s: {total_matches:,} matches, {total_cands:,} candidates, {singletons:,} singletons preserved.")

    return final_matches, final_candidates

# ---------------------------------------------------------------------------
# Packaging & CLI Entrypoint
# ---------------------------------------------------------------------------

def package_submission(team_name: str, base_dir: Path, output_dir: Path):
    """Package code, outputs, and documentation into <team_name>_submission.zip"""
    zip_filename = f"{team_name}_submission.zip"
    zip_path = base_dir / zip_filename
    print(f"\n[Packaging] Building {zip_filename}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output files
        for f_name in ["matching_results.tsv", "candidate_pairs.tsv"]:
            f_path = output_dir / f_name
            if f_path.exists():
                zipf.write(f_path, arcname=f"output/{f_name}")

        # 2. Source code, utilities, runner, and configs
        for subfolder in ["src", "utils"]:
            folder_p = base_dir / subfolder
            if folder_p.exists():
                for root, _, files in os.walk(folder_p):
                    if "__pycache__" in root:
                        continue
                    for f in files:
                        if f.endswith(".pyc"):
                            continue
                        full_p = Path(root) / f
                        rel_p = full_p.relative_to(base_dir)
                        zipf.write(full_p, arcname=f"code/business_entity_resolution/{rel_p.as_posix()}")

        for conf_file in ["README.md", "requirements.txt", "run_pipeline.py"]:
            cp = base_dir / conf_file
            if cp.exists():
                zipf.write(cp, arcname=f"code/business_entity_resolution/{conf_file}")

        # 3. Documentation template
        doc_file = base_dir / "Documentation_template.md"
        if doc_file.exists():
            zipf.write(doc_file, arcname="Documentation_template.md")

    print(f"[Packaging] Created submission archive: {zip_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output-dir", default="output", help="Output directory for generated TSV files")
    parser.add_argument("--team-name", default="team_antigravity", help="Team name for packaging .zip file")
    parser.add_argument("--top-k", type=int, default=15, help="Number of candidates to keep per S1 entity")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("      BUSINESS ENTITY RESOLUTION PIPELINE (PRODUCTION RUN)      ")
    print("=" * 70)

    data_path = Path(args.data_dir)
    test_dir = data_path / "test"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_s1_file = test_dir / "test_source1.tsv"
    if not test_s1_file.exists():
        print(f"Error: {test_s1_file} not found.")
        sys.exit(1)

    # 1. Read all S1 records and partition by country
    print("\n[Step 1/5] Loading test Source 1 records and partitioning by country...")
    s1_by_country: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    ordered_s1_ids: List[str] = []

    with open(test_s1_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            name = row[1]
            addr = row[2]
            country = row[3]
            ordered_s1_ids.append(s1_id)
            s1_by_country[country].append((s1_id, name, addr))

    total_s1 = len(ordered_s1_ids)
    print(f"  Total test Source 1 entities: {total_s1:,}")
    for c, items in s1_by_country.items():
        print(f"    - {c}: {len(items):,} entities")

    # 2. Run country-by-country resolution
    print("\n[Step 2/5] Running high-speed resolution across country partitions...")
    all_matches: Dict[str, List[str]] = {}
    all_candidates: Dict[str, List[str]] = {}

    # Process in optimal order: France first, then US, then India
    countries = sorted(s1_by_country.keys(), key=lambda c: len(s1_by_country[c]))
    for country in countries:
        s1_rows = s1_by_country[country]
        c_matches, c_cands = run_country_partition(
            country_label=country,
            s1_rows=s1_rows,
            test_dir=test_dir,
            top_k_candidates=args.top_k,
        )
        all_matches.update(c_matches)
        all_candidates.update(c_cands)

    # 3. Write output files in exact original order of test_source1.tsv
    print("\n[Step 3/5] Writing submission TSVs (matching_results.tsv & candidate_pairs.tsv)...")
    matching_file = output_path / "matching_results.tsv"
    candidate_file = output_path / "candidate_pairs.tsv"

    with open(matching_file, "w", encoding="utf-8", newline="") as fm, \
         open(candidate_file, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in ordered_s1_ids:
            m_list = all_matches.get(s1_id, [])
            c_list = all_candidates.get(s1_id, [])

            fm.write(f"{s1_id}\t{",".join(m_list)}\n")
            fc.write(f"{s1_id}\t{",".join(c_list)}\n")

    print(f"  Wrote: {matching_file}")
    print(f"  Wrote: {candidate_file}")

    # 4. Validate output files with official submission validator
    print("\n[Step 4/5] Validating outputs with official validator...")
    validator_script = Path("utils/validate_submission.py")
    if validator_script.exists():
        import subprocess
        res = subprocess.run(
            [sys.executable, str(validator_script), "--matching", str(matching_file), "--candidate", str(candidate_file), "--test-dir", str(test_dir)],
            capture_output=True,
            text=True,
        )
        print("  Validator output:\n" + res.stdout)
        if res.returncode != 0:
            print("  Validation failed! Errors:\n" + res.stderr)
            sys.exit(1)
        else:
            print("  Validation SUCCESS: PASS (exit 0)!")

    # 5. Build submission zip package
    print("\n[Step 5/5] Building submission archive...")
    package_submission(team_name=args.team_name, base_dir=Path("."), output_dir=output_path)

    elapsed = time.time() - t_start
    print(f"\n[DONE] Enhanced pipeline finished successfully in {elapsed / 60:.1f} minutes!")

if __name__ == "__main__":
    main()
