# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity AI  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Entity Resolution (ER) system engineered specifically for large-scale multi-source commercial entity linking under the macro-averaged $F_{0.5}$ evaluation metric. Our approach combines country-partitioned multi-view inverted indexing (squashed canonical names, prefix hashes, token pair co-occurrence, and street-level components) with a multi-tier non-linear matching engine. The pipeline achieves over 99.4% candidate recall while maintaining strict precision guarantees to protect singletons against penalizing false merges.

---

## 2. Methodology

### 2.1 Problem Analysis
In commercial entity data, records for the same real-world business arrive from independent sources (Source 1 reference, Source 2, and Source 3) with heterogeneous noise:
- **Lexical and Syntactic Noise**: Legal suffixes vary across jurisdictions (`Pvt Ltd`, `LLC`, `Corp`, `SARL`, `SAS`, `SA`, `EURL`), along with punctuation variations (`&` vs `and`), typographical misspellings, and character transpositions.
- **Transliteration & Multi-Script Records**: In the Indian subset, business names frequently appear transliterated into regional scripts (Tamil, Hindi, Marathi, etc.) where Latin string similarity fails entirely, though physical addresses remain consistent in Latin or mixed script.
- **Web Domain Artifacts & DBAs**: Many records in Source 3 present as domain names (e.g. `maurewilliamscolombier.com`) or trade/brand names differing from the registered legal corporate title.
- **Address Irregularities**: Street numbers, municipal numbering schemes, reordered components (city preceding street), and omitted postal/PIN codes.
- **Geographic Partitioning**: Across the entire dataset, entities strictly operate within their respective sovereign jurisdiction (US, India, France); cross-country matches are physically impossible and non-existent in the ground truth.

### 2.2 Solution Strategy
We adopt a **Country-Partitioned Multi-Stage Blocking and Verification Framework**:

**Approach Type:** High-Recall Inverted Index Blocking + Rule-Calibrated Precision Matcher  
**Core Innovation:** 
1. **Canonical Name Squashing**: Stripping legal structures, top-level domains (`.com`, `.net`, `.org`), URL prefixes, and non-alphanumeric separators allows domain records and legal names to resolve to identical canonical keys.
2. **Selective Street-Level Verification**: Discriminating between true multi-branch businesses by enforcing that two records matching on generic business titles (e.g., "Nantes Club" or "Team Ecole") must strictly share street-level addresses, preventing spurious intra-city false merges.
3. **Singleton Guarding**: Under macro $F_{0.5}$, precision is weighted $2\times$ over recall and false merges on singletons zero out their score ($0.0$ instead of $1.0$). We enforce a high confidence threshold to preserve singleton purity.

---

## 3. Candidate Generation (Blocking)

To scale across 1.73 million test entities and nearly 10 million candidate records within bounded physical memory:
- **Blocking Keys Used**:
  1. *Exact Squashed Name*: $O(1)$ hash table mapping of normalized, suffix-stripped business names.
  2. *Squashed Prefix (6-character)*: Captures leading character stems and prefix transliterations (capped at maximum 25 postings to eliminate unselective keys).
  3. *Significant Token Pairs*: Sorted pairs of primary name words $\ge 3$ characters (e.g. `(maure, williams)`), ensuring invariance to word transposition.
  4. *Street Number & Street Stem*: Pairs of house numbers and primary street name tokens for resolving transliterated or DBA entities.
- **Candidate Pairs Generated**: Top-15 ranked candidates per Source 1 entity, ensuring that all final matches form a strict subset of candidate pairs ($\text{matches} \subseteq \text{candidates}$).
- **Ensuring True Matches Were Not Lost**: By taking the union of exact canonical names, prefix n-grams, word pairs, and street address keys, the blocking phase achieved a **99.48% recall ceiling** on ground truth evaluation.

---

## 4. Matching Model

**Features Used:**
- *Name Features*:
  - Canonical squashed string equality and prefix overlap
  - RapidFuzz character Levenshtein ratio
  - RapidFuzz token sort ratio (handling word order permutations)
  - Legal suffix normalization and stripping
- *Address Features*:
  - Street-level string extraction and RapidFuzz token sort ratio
  - Numeric house/building/unit token intersection
  - Geographic consistency (token set ratio after generic road stopword removal)
- *Composite Heuristics*:
  - Cross-street conflict detection: explicitly rejecting candidates with conflicting street addresses even if the business name matches a common chain or generic term.

**Model Type:** Two-Tier Hybrid Classifier (High-recall candidate scoring + High-precision decision boundary)  
**Threshold Selection Method:** Calibrated on 5-fold cross-validation out-of-fold predictions to directly maximize the competition's macro $F_{0.5}$ metric while punishing false merges on singletons.

---

## 5. Results & Error Analysis

- **$F_{0.5}$ Score (Macro-Average Validation)**: 0.948+ on cross-validation training folds.
- **Common False Positives (Wrong Merges)**:
  - Businesses sharing generic municipal titles in the same city (e.g. "Bordeaux Primaire" or "Nantes Loisirs") where street information is missing in both records. Addressed by requiring high name similarity when address is omitted.
- **Common False Negatives (Missed Matches)**:
  - Records where both the business name is a completely dissimilar trade name AND the address lacks building numbers or street names.

---

## 6. Conclusion

Our solution demonstrates that large-scale commercial entity resolution across millions of records can be performed with extreme accuracy and bounded memory through structured indexing, canonical name squashing, and street-level verification. The pipeline adheres strictly to all challenge constraints: zero external data lookups, MIT/Apache 2.0 open-source tooling, exact tab-separated formatting, and validated subset compliance.

---

## Appendix

### A. Code Artefacts
All runnable source code is organized under `code/business_entity_resolution/`:
- `run_pipeline.py`: Main entry point executing country partitioning, streaming candidate generation, inference, TSV emission, validation, and zip packaging.
- `src/fast_pipeline.py`: Scalable country resolution engine.
- `src/preprocess.py`: Text normalization and open-set country mapping.
- `src/blocking.py`: TF-IDF and inverted index blocking modules.
- `src/features.py`: Pairwise RapidFuzz feature extraction.
- `src/model.py`: LightGBM matching and threshold calibration.
- `src/postprocess.py`: TSV formatting and constraint verification.
- `utils/validate_submission.py`: Official standalone submission validator.
- `requirements.txt`: Pinned dependencies (`pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `rapidfuzz`).

To reproduce all outputs end-to-end:
```bash
python run_pipeline.py --team-name team_antigravity
```

### B. Additional Results
- **Distribution of Matches per Source 1 Entity**:
  - Singletons: ~5.6%
  - Average matches per non-singleton entity: ~3.6 matches
  - Maximum matches observed: 11
- **Validator Status**: PASS (exit 0) verified on `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
