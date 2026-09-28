"""Post-processing and TSV output generation ensuring strict compliance with competition constraints."""

from pathlib import Path
from typing import Dict, List, Set, Union
import pandas as pd

def generate_output_tsvs(
    all_s1_ids: List[str],
    candidates_dict: Dict[str, Union[Dict[str, float], List[str]]],
    matches_dict: Dict[str, Union[List[Tuple[str, float]], List[str], Set[str]]],
    valid_target_ids: Set[str],
    output_dir: Union[str, Path] = "output",
) -> Tuple[Path, Path]:
    """
    Format and write matching_results.tsv and candidate_pairs.tsv ensuring:
    1. Tab-separated (.tsv), no quoting.
    2. Exactly one row per Source 1 entity (matching the exact test entity sequence).
    3. Empty list for singletons.
    4. No duplicate IDs within any ID list.
    5. No self-matches (only valid S2- / S3- IDs present in the test set).
    6. Strict subset guarantee: every ID in matching_results.tsv appears in candidate_pairs.tsv.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matching_rows = []
    candidate_rows = []

    for s1_id in all_s1_ids:
        # --- 1. Process candidate IDs ---
        raw_cands = candidates_dict.get(s1_id, [])
        if isinstance(raw_cands, dict):
            cand_id_list = list(raw_cands.keys())
        elif isinstance(raw_cands, (list, tuple, set)):
            cand_id_list = list(raw_cands)
        else:
            cand_id_list = []

        # Filter valid target IDs and deduplicate preserving order
        clean_cands: List[str] = []
        seen_cands: Set[str] = set()
        for cid in cand_id_list:
            cid_str = str(cid).strip()
            if cid_str in valid_target_ids and cid_str not in seen_cands:
                seen_cands.add(cid_str)
                clean_cands.append(cid_str)

        # --- 2. Process matched IDs ---
        raw_matches = matches_dict.get(s1_id, [])
        if isinstance(raw_matches, (list, tuple)):
            if raw_matches and isinstance(raw_matches[0], (tuple, list)):
                match_id_list = [item[0] for item in raw_matches]
            else:
                match_id_list = list(raw_matches)
        elif isinstance(raw_matches, set):
            match_id_list = list(raw_matches)
        else:
            match_id_list = []

        # Filter valid matches and deduplicate
        clean_matches: List[str] = []
        seen_matches: Set[str] = set()
        for mid in match_id_list:
            mid_str = str(mid).strip()
            if mid_str in valid_target_ids and mid_str not in seen_matches:
                seen_matches.add(mid_str)
                clean_matches.append(mid_str)

        # SUBSET ENFORCEMENT: All matches must be included in candidates!
        for mid in clean_matches:
            if mid not in seen_cands:
                clean_cands.append(mid)
                seen_cands.add(mid)

        # Format rows
        matched_str = ",".join(clean_matches)
        cand_str = ",".join(clean_cands)

        matching_rows.append(f"{s1_id}\t{matched_str}\n")
        candidate_rows.append(f"{s1_id}\t{cand_str}\n")

    matching_file = output_path / "matching_results.tsv"
    candidate_file = output_path / "candidate_pairs.tsv"

    with open(matching_file, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(matching_rows)

    with open(candidate_file, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.writelines(candidate_rows)

    return matching_file, candidate_file
