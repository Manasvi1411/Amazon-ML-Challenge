# Amazon ML Challenge: Business Entity Resolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Passed Validation](https://img.shields.io/badge/Competition%20Validation-PASS-brightgreen.svg)]()

An end-to-end, high-precision Machine Learning pipeline engineered for large-scale multi-source commercial entity linking under the **Macro-Averaged $F_{0.5}$** metric.

The system scales across **1.73 million test entities** against a candidate pool of **~10 million records** with bounded memory usage (< 400 MB RAM) and zero external API lookups.

---

## Problem Overview

In commercial entity resolution, business records for the same real-world organization arrive from independent, noisy data sources:
- **Source 1 (`S1-`)**: Reference source with deduplicated entity records.
- **Source 2 (`S2-`) & Source 3 (`S3-`)**: Unaligned sources with noisy, incomplete, transliterated, and varied entity records.

### Challenge Constraints:
1. **Macro-Averaged $F_{0.5}$ Metric**:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Precision is weighted **$2\times$ as heavily as Recall**. False merges on singletons (records with no match) receive a score of **$0.0$** (instead of $1.0$).
2. **Open-Set Country Generalization**:
   - Training Set: **US**, **India**
   - Test Set: **US**, **India**, **France** (unseen jurisdiction)
3. **Fair-Play Rule**: Strictly zero external lookups (no external APIs, geocoders, or online databases).
4. **Subset Constraint**: Every final predicted match must be a strict subset of the candidate pairs ($\text{matches} \subseteq \text{candidates}$).

---

## Architecture & Key Innovations

```
                                  [Test Source 1 Records (1.73M)]
                                                │
                                     [Country Partitioning]
                           ┌────────────────────┼────────────────────┐
                           ▼                    ▼                    ▼
                      [France]                 [US]               [India]
                           │                    │                    │
                ┌──────────┴──────────┐         │                    │
                ▼                     ▼         │                    │
     [Inverted Index Blocking] [Candidate Pool] │                    │
                │                     │         │                    │
                └──────────┬──────────┘         │                    │
                           ▼                    ▼                    ▼
            [Multi-Tier Feature Extraction & Conflict Resolution]
            - Regional / State Conflict Filtering (US 50 states, India 28 states, FR depts)
            - Unicode NFKD Accent Stripping (e.g., Ó -> O)
            - Number & Leading Zero Normalization (e.g., 038 -> 38)
            - True Legal Form Stripping (preserves business terms: tech, services)
            - Singleton Protection (guards against empty-address false merges)
            - Street-Anchored Multi-Script Matching (Devanagari, Telugu, Tamil, Odia)
                           │
                           ▼
            [Decision Engine & Metric Calibration]
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [matching_results.tsv]       [candidate_pairs.tsv]
  (Leaderboard Evaluation)     (Blocking Candidate Pool)
            │                             │
            └──────────────┬──────────────┘
                           ▼
              [Official Rule Validator]
                           ▼
             [Final Submission Archive]
```

### 1. State & Regional Conflict Filtering
- Prevents cross-jurisdiction false merges:
  - **United States**: Resolves both 2-letter postal codes (`NC`, `IA`, `TX`) and full state names across all 50 states + DC.
  - **India**: Resolves 28 states and 8 union territories (`Telangana`, `Gujarat`, `Maharashtra`, `Delhi`, etc.).
  - **France**: Extracts 2-digit department codes from 5-digit postal codes (`75` for Paris, `13` for Marseille, `69` for Lyon).
- If two candidate records have differing regional markers, the merge is rejected.

### 2. Singleton Guarding (Precision Protection)
- When a target record has a blank address (`Addr: []`), the engine **never** matches on generic prefixes or single common words (e.g. *Cornerstone*, *Hotel*, *Creative*, *Global*).
- An empty-address candidate is only accepted if the normalized name is an exact match and has $\ge 8$ non-generic characters.

### 3. Normalization & Formatting Resilience
- **Unicode NFKD Normalization**: Accented letters (`Ó` $\rightarrow$ `O`, `é` $\rightarrow$ `e`) are flattened so transliterated variations match correctly.
- **Leading Zeros**: Addresses with leading zeros (e.g., `038 R Street` vs `38 R Street`) normalize to the identical integer token `38`.
- **DBA & Trade Names**: Automatically splits `d/b/a`, `t/a`, `a/k/a`, and `trading as` into independent indexable aliases.

### 4. Street-Anchored Multi-Script Matching
- In India, business names in Source 2 or 3 frequently appear in regional scripts (Hindi, Telugu, Tamil, Odia) where Latin string similarity is 0%.
- When house/unit numbers and street tokens match strongly in the same locality, the engine links the real-world business entity even with zero string overlap in the corporate name.

---

## Results & Validation

Validation on the full test dataset of **1,732,544 Source 1 entities** and **9,969,589 candidate records**:

| Partition | S1 Entities | Candidates Scanned | Matches Generated | Singletons Preserved |
| :--- | :--- | :--- | :--- | :--- |
| **France (FR)** | 259,452 | 1,434,993 | 1,886,641 | 13,375 |
| **United States (US)** | 663,106 | 3,817,031 | 2,070,368 | 54,145 |
| **India (IN)** | 809,986 | 4,717,565 | 3,556,643 | 40,343 |
| **Total** | **1,732,544** | **9,969,589** | **7,513,652** | **107,863** |

### Validator Status (`utils/validate_submission.py`):
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (107863 empty, 1624681 non-empty).
  candidate_pairs.tsv: 1732544 rows (17175 empty, 1715369 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## Repository Structure

```
Amazon-ML-Challenge/
├── Documentation_template.md    # Official methodology and error analysis write-up
├── README.md                    # Project overview & execution guide
├── requirements.txt             # Pinned open-source dependencies
├── run_pipeline.py              # Main unified streaming orchestrator
├── output/
│   └── .gitkeep                 # Generated submission TSVs (gitignored due to file size)
├── src/
│   ├── __init__.py
│   ├── blocking.py              # Multi-strategy TF-IDF sparse blocking
│   ├── fast_pipeline.py         # Streaming inverted-index candidate generator
│   ├── features.py              # 25+ pairwise RapidFuzz similarity features
│   ├── model.py                 # LightGBM classifier & macro F_0.5 threshold calibration
│   ├── postprocess.py           # TSV formatter & constraint verification
│   └── preprocess.py            # Text normalization & open-set country mapping
└── utils/
    ├── generate_synthetic_data.py # Sample data generator for quick testing
    └── validate_submission.py     # Official standalone competition validator
```

---

## Getting Started

### 1. Prerequisites & Environment
- Python 3.9+
- Recommended: 8 GB+ RAM, multi-core CPU

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Manasvi1411/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge
pip install -r requirements.txt
```

Pinned dependencies:
- `pandas>=2.2.0`
- `numpy>=1.26.0`
- `scipy>=1.12.0`
- `scikit-learn>=1.4.0`
- `lightgbm>=4.3.0`
- `rapidfuzz>=3.6.0`

---

## Execution

### Running the End-to-End Pipeline
To run candidate blocking, matching, TSV generation, validation, and submission packaging:

```bash
python run_pipeline.py --team-name <your_team_name>
```

#### Optional CLI Arguments:
- `--team-name <name>`: Prefix for `<team_name>_submission.zip` (default: `team_antigravity`).
- `--top-k <int>`: Max candidate pairs to retain per Source 1 entity (default: `15`).
- `--data-dir <path>`: Dataset directory containing `train/` and `test/` (default: `dataset`).
- `--output-dir <path>`: Output folder for generated TSVs (default: `output`).

### Validating the Submission
Verify compliance against all challenge formatting and logical rules:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Deliverables Generated

1. `output/matching_results.tsv`: Final high-precision entity matches (used for leaderboard evaluation).
2. `output/candidate_pairs.tsv`: Candidate blocking pairs satisfying the subset rule.
3. `<team_name>_submission.zip`: Self-contained archive with code, outputs, and documentation.

---

## License
This project is licensed under the MIT License. All external libraries used comply strictly with MIT / Apache 2.0 open-source licensing.
