---
title: Image Dataset Analyzer
---

# Image Dataset Analyzer

Image dataset analyzer with visual, structural, and privacy checks.

---

## Method behavior (granular)

| Method | What it does | How it behaves (step-by-step) | Output |
|--------|-------------|-------------------------------|--------|
| `guess_columns_meta` | Suggests metadata roles | • Regex scoring + uniqueness<br>• Infers path, label, split, groups | Dict |
| `read_zip_images` | Loads images | • Reads ZIP<br>• Samples if large<br>• Extracts features, EXIF, hashes | DataFrame |
| `image_features` | Extracts features | • Resolution, brightness, color stats<br>• Blur via Laplacian<br>• Entropy | Dict |
| `exif_flags` | Extracts metadata | • Camera, timestamp, GPS presence | Dict |
| `perceptual_hash` | Computes fingerprint | • Generates phash for similarity | String |
| `_exact_duplicates` | Detects duplicates | • Uses SHA-256 | Dict |
| `_near_duplicates_phash` | Detects near duplicates | • Hamming distance on phash<br>• Bucketed comparisons | Dict |
| `assess_quality` | Evaluates quality | • Corrupt images<br>• Low resolution<br>• Blur<br>• Duplicates<br>• Label stats | Dict |
| `assess_reliability` | Evaluates stability | • Slice-based feature drift<br>• KS statistics | Dict |
| `assess_robustness` | Detects anomalies | • MAD-based outliers<br>• Rare category-label patterns | Dict |
| `assess_fairness` | Evaluates disparities | • Group representation<br>• Outcome disparity | Dict |
| `assess_security` | Detects risks | • EXIF GPS<br>• PII in paths<br>• Dataset hash | Dict |
| `assess_all` | Runs pipeline | • Aggregates all checks | Dict |
| `verdict_panel` | Final decision | • Flags corruption, drift, PII | Verdict |
| `build_recommendations` | Suggests actions | • Cleaning, deduplication, masking | List |

---

[← Back to home](./index.html)