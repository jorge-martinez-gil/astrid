# ASTRID

<p align="center">
  <img src="logo.png" width="40%" alt="ASTRID Logo">
</p>

<p align="center">
  <strong>Advanced Software Tools for Reliable Industrial Datasets</strong>
</p>

<p align="center">
  <em>Open-source tooling to assess and improve the reliability of datasets used in industrial AI systems.</em>
</p>

---

## Overview

ASTRID delivers open-source software components and metrics for evaluating dataset reliability in industrial AI. The project focuses on practical, production-ready checks that support safer, fairer, and more dependable AI across manufacturing and related industrial domains.

## Motivation

Industrial AI systems depend heavily on data quality, yet issues such as bias, hidden identifiers, label leakage, missing values, and unstable sensor streams often go undetected until late deployment stages — when fixing them is costly or unsafe.

ASTRID helps engineers and researchers answer critical questions before problems reach production:

- Can this dataset be trusted for training and validation?
- Does data quality degrade model performance in safety-relevant cases?
- Are there hidden risks that standard pipelines fail to detect?

## Scope

ASTRID targets datasets commonly found in industrial environments:

- **Time-series data** — sensor streams, machine logs, and telemetry
- **Industrial imagery** — inspection, quality control, and process monitoring
- **Synthetic and simulated data** — digital twins and augmented datasets

The tools are built for real production-oriented settings, not only research labs.

## Core Features

- **Reliability checks** tailored to industrial AI tasks
- **Bias, imbalance, and leakage detection** across modalities
- **Multi-modal support** for time-series, images, and synthetic data
- **Behavior-linked metrics** that connect data issues to model performance under noise or incompleteness
- **Transparent algorithms** with user-configurable parameters
- **Resource-efficient design** suitable for operational environments

## Project Structure

Planned components include:

| Component | Description |
|---|---|
| Analysis library | Modular Python library for dataset checks and metrics |
| Validation pipelines | End-to-end dataset validation workflows |
| Benchmark datasets | Curated examples for testing and demonstration |
| Documentation | Usage guides, API reference, and tutorials |
| Reference experiments | Reproducible experiments illustrating typical use cases |

The repository will grow incrementally as components mature.

## Standards and Regulation

ASTRID aligns its metrics and checks with current and emerging European requirements, including:

- Trustworthy AI principles
- Data protection constraints
- Industrial safety expectations

The project aims to provide early technical references that support compliance efforts in practice.

## Project Status

> 🚧 **Early stage** — this repository marks the starting point of the ASTRID project.

Initial releases will include:

- [ ] Core reliability metrics
- [ ] Example datasets
- [ ] Reference experiments

Development is ongoing, and contributions and feedback are welcome as the project evolves.

## License

All software developed in ASTRID is released under an open-source license. Specific license details will be added once the initial modules are published.

## Acknowledgements

ASTRID is developed within the scope of European research and innovation activities focused on trustworthy AI for industry. Further details on funding and collaborations will be added as the project progresses.

