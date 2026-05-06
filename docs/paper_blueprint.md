# ASTRID Scientific Paper Blueprint

## Working Title

ASTRID: A Multimodal Dataset Reliability Assessment Framework for Industrial AI

## One-Sentence Thesis

ASTRID operationalizes dataset reliability as a measurable, auditable, and locally executable assessment across tabular, time-series, and image datasets, detecting quality, security, reliability, robustness, fairness, and transparency risks before model training.

## Recommended Paper Positioning

The strongest paper is not a software announcement. It should be framed as a methods-and-evaluation paper about dataset readiness for industrial AI.

Recommended angle:

> We propose a modality-aware framework for pre-training dataset reliability assessment, instantiate it in an open-source tool, and evaluate its ability to expose known dataset failure modes and produce actionable remediation evidence.

This lets the paper compete on scientific contribution rather than interface polish.

## Target Venue Families

- Data-centric AI / ML systems venues: dataset governance, data validation, MLOps, trustworthy AI.
- Applied AI and software engineering venues: industrial AI quality assurance, empirical software engineering for ML.
- Domain-focused journals: trustworthy AI, AI engineering, industrial informatics, data quality.

For a top-tier submission, choose one primary audience before writing:

- ML systems audience: emphasize pipeline integration, automation, runtime, reproducibility, and comparison with validation baselines.
- Trustworthy AI audience: emphasize risk dimensions, standards alignment, auditability, fairness, privacy, and transparency.
- Data quality audience: emphasize formal metric taxonomy, scoring validity, and modality coverage.

## Core Contributions

1. A modality-aware taxonomy of dataset reliability risks for industrial AI covering tabular, temporal, and image datasets.
2. A composite health-score formulation that maps heterogeneous reliability evidence into interpretable dimension-level and global scores.
3. A local-first implementation that supports confidential industrial datasets without external data transfer.
4. A set of analyzers for quality, security, reliability, robustness, fairness, and transparency signals.
5. An empirical evaluation on real and injected-fault datasets showing detection accuracy, runtime behavior, and usefulness for remediation.

## What ASTRID Currently Implements

ASTRID includes four Streamlit modules:

- Tabular analyzer: CSV, Parquet, and Excel-oriented reliability checks.
- Time-series analyzer: temporal profiling, cadence irregularity, gaps, duplicate timestamps, and slice-based drift.
- Image analyzer: ZIP-based image scanning with metadata, annotation, duplicate, EXIF, fairness, and transparency checks.
- Experimental drift module: simulation of active learning and adaptation policies under post-drift supervision.

Shared capabilities include:

- Configurable score weights for quality, security, reliability, robustness, and fairness.
- HTML and JSON report export.
- Heuristic PII detection.
- KS-based distribution drift detection.
- Duplicate and split-leakage detection.
- Group disparity checks.
- Recommendation generation.
- Local execution for privacy-sensitive datasets.

## Formal Method Section

### Dataset Reliability Model

Define a dataset as:

`D = (X, Y, M, S, T, G)`

where:

- `X` is the observed feature space.
- `Y` is the optional label space.
- `M` is metadata.
- `S` is split assignment.
- `T` is temporal order or timestamp.
- `G` is group or operational context.

ASTRID estimates reliability evidence across dimensions:

`R(D) = {Q(D), Sec(D), Rel(D), Rob(D), Fair(D), Trans(D)}`

where the dimensions correspond to:

- `Q`: completeness, duplicates, schema consistency, leakage, annotation quality.
- `Sec`: PII, EXIF privacy, suspicious samples, source concentration.
- `Rel`: drift, time-axis validity, provenance, annotator agreement.
- `Rob`: anomaly rates, rare categories, low-coverage operating conditions.
- `Fair`: group representation, missingness disparities, label parity gaps.
- `Trans`: datasheet completeness and traceability, currently emphasized in the image analyzer.

### Composite Score

For each dimension `d`, normalize metric evidence into a bounded score:

`s_d(D) in [0, 1]`

Then compute:

`H(D) = sum_d w_d s_d(D)`

where `w_d >= 0` and `sum_d w_d = 1`.

ASTRID reports:

- Global health score: `100 * H(D)`.
- Letter grade.
- Dimension contribution breakdown.
- Ranked recommendations.

### Important Scientific Claim

The health score should be presented as a decision-support index, not a proof of dataset safety. A top-tier paper should explicitly state that ASTRID surfaces measurable risk signals that support human-led dataset governance.

## Empirical Evaluation Plan

### Research Questions

RQ1. How accurately does ASTRID detect injected dataset reliability failures across modalities?

RQ2. How does ASTRID compare with common baseline validation tools and ad hoc quality checks?

RQ3. How sensitive is the composite health score to threshold and weight choices?

RQ4. How does ASTRID scale with dataset size and modality?

RQ5. Do ASTRID recommendations help users identify and prioritize remediation actions?

### Evaluation Datasets

Use a mix of public and synthetic/injected-fault datasets:

- Tabular: UCI/OpenML-style classification and regression datasets.
- Time-series: industrial sensor, predictive maintenance, or energy consumption datasets.
- Image: small public image datasets with labels and metadata; include controlled duplicates, corrupt files, EXIF metadata, and label noise.

For industrial relevance, include at least one realistic private or semi-public case study if possible. If the data cannot be shared, publish the injection protocol and anonymized metric outputs.

### Fault Injection Protocol

Create controlled corruptions:

- Missingness: MCAR, MAR by group, and block missingness.
- Duplicates: exact duplicates and cross-split duplicates.
- Drift: mean shift, variance shift, covariate shift, temporal shift.
- Security: synthetic PII-like strings in text fields and file paths.
- Fairness: group imbalance, group-dependent missingness, label disparity.
- Image quality: low resolution, blur, corrupt files, duplicate images, GPS EXIF.
- Annotation quality: inconsistent schemas, invalid bounding boxes, annotator disagreement.

### Metrics

Report:

- Detection precision, recall, F1 for injected failures.
- Score monotonicity as fault severity increases.
- Runtime and memory behavior.
- Recommendation coverage: percentage of injected faults matched by a relevant recommendation.
- Sensitivity of health score to thresholds and weights.
- False positive analysis on clean datasets.

### Baselines

Use baselines appropriate to the venue:

- Generic data validation libraries.
- Manual pandas/sklearn quality scripts.
- Modality-specific checks such as duplicate detection, drift tests, and image metadata scanners.
- No-audit baseline for downstream model degradation experiments.

The paper should be careful: baselines may not cover all ASTRID dimensions. Compare by failure category, not only by global score.

## Suggested Experiments

### Experiment 1: Controlled Fault Detection

Inject known faults at increasing severity levels and measure whether ASTRID flags them. This is the core validity experiment.

Expected figure:

- Line plot: fault severity vs. health score.
- Bar chart: detection F1 by fault type and modality.

### Experiment 2: Downstream Model Risk

Train simple models on clean vs. corrupted datasets. Show that ASTRID score degradation correlates with downstream performance degradation, calibration degradation, or fairness gaps.

Expected figure:

- Scatter plot: ASTRID health score vs. model performance or fairness metric.

### Experiment 3: Runtime and Scalability

Measure analysis time as rows, columns, image count, or time-series length increase.

Expected table:

- Dataset size, modality, checks enabled, runtime, peak memory.

### Experiment 4: Ablation of Dimensions

Remove one dimension at a time and measure lost detection coverage.

Expected table:

- Full ASTRID vs. without security, without fairness, without drift, without robustness.

### Experiment 5: Drift Simulation Module

Use the experimental drift module as a separate case study or appendix. Position it carefully: it is a simulation environment for evaluating supervision policies after drift, not the same as the static dataset audit.

Expected figure:

- Cost vs. accuracy under Static, SAL, ADWIN-SAL, and Symbiosis-Edge.

## Paper Structure

1. Introduction
2. Background and Related Work
3. Dataset Reliability Taxonomy
4. ASTRID Framework
5. Implementation
6. Experimental Design
7. Results
8. Discussion
9. Threats to Validity
10. Conclusion

## Draft Abstract

Industrial AI systems are often evaluated after model training, while the datasets that determine their reliability are audited late, manually, or not at all. This paper presents ASTRID, a local-first framework for multimodal dataset reliability assessment across tabular, time-series, and image data. ASTRID operationalizes dataset readiness through six evidence dimensions: quality, security, reliability, robustness, fairness, and transparency. For each modality, it extracts measurable risk signals including missingness, duplicate and split leakage, temporal drift, cadence irregularity, image corruption, annotation consistency, privacy-sensitive metadata, group disparities, and documentation completeness. These signals are aggregated into interpretable dimension scores, a global health index, and actionable remediation recommendations. We evaluate ASTRID using controlled fault injection and real-world datasets, measuring detection performance, score sensitivity, runtime behavior, and correlation with downstream model degradation. The results show that modality-aware pre-training assessment can expose reliability failures that would otherwise propagate into model development workflows, supporting reproducible dataset governance for industrial AI. ASTRID is released as an open-source tool designed for local execution on confidential datasets.

## Top-Tier Risk Areas

The current repository is promising, but the paper will need stronger evidence before it can be top tier:

- The scoring model needs a precise mathematical definition and justification.
- The thresholds need either empirical calibration or a clear rationale.
- The evaluation must include controlled ground truth faults.
- Related work must be positioned carefully to avoid overclaiming.
- The UI should be secondary; scientific novelty should be taxonomy, measurement, and validation.
- Limitations must be explicit: heuristic PII, approximate fairness proxies, incomplete metadata, and modality-specific assumptions.

## Immediate Next Steps

1. Choose the venue family and target paper length.
2. Decide whether the paper should emphasize dataset governance, ML systems, or data quality.
3. Build a benchmark script that generates injected-fault datasets and runs ASTRID analyzers without the UI.
4. Create a `paper/` directory with LaTeX source, figures, tables, and experiment scripts.
5. Run the first controlled tabular experiment and use it as the seed result for the paper.

