# References and code provenance

The reconstruction backbone and numerical monitoring modules originate from the existing AERIS project. Their module names are retained; the release adds a portable workflow rather than a different detector.

- ALFA: A. Keipour, M. Mousaei, and S. Scherer, *ALFA: A Dataset for UAV Fault and Anomaly Detection*, IJRR, 2021. [Paper](https://arxiv.org/abs/1907.06268), [data/tools](https://www.cs.cmu.edu/~mmousaei/alfa-dataset/).
- RflyMAD: X. Le et al., *RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment*, IJRR, 2025. [Paper](https://arxiv.org/abs/2311.11340), [provider](https://rfly-openha.github.io/documents/4_resources/dataset.html).
- VUS: J. Paparrizos et al., *Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection*, PVLDB 15(11), 2022. The compact metric implementation follows the official `vus==0.0.5` definition and retains its source attribution in `src/eval/vus_metrics.py`.
- The reconstruction model contains Fourier-period processing inspired by TimesNet. This is a component of the existing Base, not a claim that the proposed monitor is an official TimesNet reproduction. See the comments in `src/models/modules/spectral.py` and the companion manuscript.

No third-party dataset, experimental result archive, official-baseline repository, proprietary library, or access credential is bundled. Dataset licenses/conditions remain those of their providers. The project retains its existing MIT license.
