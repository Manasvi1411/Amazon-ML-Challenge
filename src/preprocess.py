"""Data preprocessing and normalization module for Business Entity Resolution."""

import re
import unicodedata
from typing import Dict, List, Optional, Set
import pandas as pd

# Open-country mapping dictionary to canonical 2-letter codes
COUNTRY_ALIASES: Dict[str, str] = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "in": "IN",
    "ind": "IN",
    "india": "IN",
    "bharat": "IN",
    "fr": "FR",
    "fra": "FR",
    "france": "FR",
    "uk": "GB",
    "gbr": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "ca": "CA",
    "can": "CA",
    "canada": "CA",
    "de": "DE",
    "deu": "DE",
    "germany": "DE",
    "deutschland": "DE",
}

# Common legal suffixes across US, India, France, and international jurisdictions
LEGAL_SUFFIXES: List[str] = [
    # Multi-word suffixes first
    r"\bprivate limited\b",
    r"\bpvt ltd\b",
    r"\bpvt limited\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bpvt\b",
    r"\bprivate\b",
    r"\bsarl\b",
    r"\bsas\b",
    r"\bsasu\b",
    r"\bsa\b",
    r"\beurl\b",
    r"\bsci\b",
    r"\bsnc\b",
    r"\bgmbh\b",
    r"\bholding\b",
    r"\bholdings\b",
    r"\bgroup\b",
    r"\benterprises\b",
    r"\bservices\b",
    r"\bsolutions\b",
    r"\btechnologies\b",
    r"\bindustries\b",
]

# Address token standardizations
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bsq\b": "square",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bno\b": "number",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bw/\b": "with",
    r"\bw\b": "west",
    r"\be\b": "east",
    r"\bn\b": "north",
    r"\bs\b": "south",
}

def remove_accents(text: str) -> str:
    """Normalize unicode and remove accents/diacritics."""
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", str(text))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_country(country_str: Optional[str]) -> str:
    """Normalize country string to open-set canonical label."""
    if not country_str or pd.isna(country_str):
        return "UNKNOWN"
    norm = remove_accents(str(country_str)).strip().lower()
    norm = re.sub(r"[^\w\s]", "", norm).strip()
    return COUNTRY_ALIASES.get(norm, norm.upper())

def clean_text(text: Optional[str]) -> str:
    """Basic clean text: accent removal, lowercasing, and space normalization."""
    if not text or pd.isna(text):
        return ""
    text = remove_accents(str(text)).lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_business_name(name: Optional[str]) -> Dict[str, str]:
    """Clean business name, extract legal suffix, produce stripped & token-sorted variants."""
    raw_cleaned = clean_text(name)
    if not raw_cleaned:
        return {
            "clean_name": "",
            "stripped_name": "",
            "sorted_name": "",
            "legal_suffix": "",
        }

    # Extract & remove legal suffixes from stripped name
    stripped = raw_cleaned
    matched_suffix = ""
    for suffix_pat in LEGAL_SUFFIXES:
        match = re.search(suffix_pat, stripped)
        if match:
            matched_suffix = match.group(0)
            stripped = re.sub(suffix_pat, " ", stripped)

    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        stripped = raw_cleaned  # fallback if all tokens were suffixes

    # Sorted tokens to handle word transpositions
    tokens = [t for t in stripped.split() if len(t) > 1]
    sorted_name = " ".join(sorted(tokens)) if tokens else stripped

    return {
        "clean_name": raw_cleaned,
        "stripped_name": stripped,
        "sorted_name": sorted_name,
        "legal_suffix": matched_suffix,
    }

def normalize_address(address: Optional[str]) -> Dict[str, any]:
    """Clean address, expand abbreviations, extract numbers/PIN codes."""
    raw_cleaned = clean_text(address)
    if not raw_cleaned:
        return {
            "clean_address": "",
            "address_numbers": set(),
            "has_landmark": 0,
        }

    # Expand common abbreviations
    standardized = raw_cleaned
    for pat, repl in ADDRESS_ABBREVIATIONS.items():
        standardized = re.sub(pat, repl, standardized)
    standardized = re.sub(r"\s+", " ", standardized).strip()

    # Extract all digit sequences (PIN codes, house numbers, street numbers)
    numbers = set(re.findall(r"\b\d+\b", standardized))

    # Landmark cues
    landmark_match = 1 if re.search(r"\b(near|opposite|opp|behind|beside|next to|adjacent)\b", standardized) else 0

    return {
        "clean_address": standardized,
        "address_numbers": numbers,
        "has_landmark": landmark_match,
    }

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a source dataframe and add standardized features."""
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    # Country
    df["norm_country"] = df["country"].apply(normalize_country)

    # Name normalization
    name_dicts = df["business_name"].apply(normalize_business_name).tolist()
    name_df = pd.DataFrame(name_dicts, index=df.index)
    df["clean_name"] = name_df["clean_name"]
    df["stripped_name"] = name_df["stripped_name"]
    df["sorted_name"] = name_df["sorted_name"]
    df["legal_suffix"] = name_df["legal_suffix"]

    # Address normalization
    addr_dicts = df["business_address"].apply(normalize_address).tolist()
    addr_df = pd.DataFrame(addr_dicts, index=df.index)
    df["clean_address"] = addr_df["clean_address"]
    df["address_numbers"] = addr_df["address_numbers"]
    df["has_landmark"] = addr_df["has_landmark"]

    # Combined text for TF-IDF / Embedding
    df["combined_text"] = df["stripped_name"] + " " + df["clean_address"]

    return df
