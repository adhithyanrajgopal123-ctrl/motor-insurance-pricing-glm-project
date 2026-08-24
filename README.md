
# Motor Insurance Pricing — Frequency-Severity GLM vs XGBoost

## Overview

This project develops and validates a **motor insurance pricing framework** using traditional actuarial Generalised Linear Models (GLMs) and a machine-learning benchmark based on XGBoost.

The analysis uses the **French Motor Third-Party Liability (freMTPL2) dataset**, a large real-world anonymised motor insurance dataset sourced through OpenML. The project follows a complete actuarial pricing workflow, beginning with data preparation and exploratory analysis and progressing through claims frequency and severity modelling, pure-premium estimation, machine-learning benchmarking and formal out-of-sample validation.

The central objective is not simply to determine which model has the highest predictive accuracy, but to investigate **where a traditional actuarial pricing model performs well, where it breaks down, and where machine learning can provide additional pricing value**.

---

## Key Objectives

* Analyse motor insurance policy and claims experience.
* Identify important rating factors affecting claim frequency and severity.
* Develop a **Poisson frequency GLM** using an exposure offset.
* Develop a **Gamma severity GLM** for policies with positive claims.
* Develop a **Tweedie GLM** as a joint pure-premium model.
* Benchmark the actuarial models against **XGBoost gradient boosting**.
* Evaluate model discrimination using the **Gini coefficient**.
* Assess model calibration using exposure-weighted lift charts.
* Perform formal **Actual-vs-Expected (A/E) validation** on an untouched holdout dataset.
* Investigate statistically significant segment-level miscalibration.
* Convert GLM coefficients into actuarial rating relativities.
* Translate model results into practical pricing implications.

---

## Dataset

The project uses the **French Motor Third-Party Liability (freMTPL2) dataset**.

Two datasets were used:

* `freMTPL2freq` — policy-level exposure and claim counts
* `freMTPL2sev` — individual claim amounts

The datasets were sourced from **OpenML** (dataset IDs 41214 and 41215), which provides a public mirror of the CASdatasets French Motor Third-Party Liability data. The severity records were aggregated to policy level and merged with the frequency data using policy ID. Policies without matched claims were assigned zero claim amount.

The dataset contains approximately **676,789 anonymised policies** and is widely used in actuarial pricing research.

---

## Data Preparation

Several data-quality and modelling decisions were applied before model development:

* Policies with exposure outside the valid range `(0, 1]` were removed.
* Claim counts were capped at 4 per policy to reduce the influence of extreme count outliers.
* Claim amounts were capped at €200,000 for modelling stability.
* BonusMalus banding was corrected to ensure that the lower boundary value of 50 was correctly included.
* `Area` was excluded because exploratory analysis showed that it provided redundant information relative to `Density`.
* `log(Density)` was retained as a continuous risk proxy.

The claim amount cap is treated as a modelling decision rather than a production recommendation. In practice, large claims would require separate tail or reinsurance modelling.

---

## Modelling Framework

The data was divided into:

* **70% training data**
* **30% holdout test data**

A fixed random seed was used to support reproducibility. All models were fitted on the training data and evaluated out-of-sample on the untouched test set.

### 1. Frequency-Severity GLM

The primary actuarial pricing framework consists of two components.

#### Claim Frequency

A **Poisson GLM with a log link and exposure offset** was fitted to model the expected number of claims.

#### Claim Severity

A **Gamma GLM with a log link** was fitted to policies with positive claims.

The frequency and severity predictions are combined to obtain an expected pure premium.

This two-part frequency-severity structure was retained as the primary actuarial approach because the joint Tweedie model did not materially improve severity prediction and diluted the strong frequency signal.

---

### 2. Tweedie GLM

A Tweedie GLM was fitted directly to pure premium as a single joint model.

The model used a variance power of **1.5** and provided a useful comparison with the two-part frequency-severity framework.

---

### 3. XGBoost

**XGBoost gradient-boosted trees** were developed as a non-linear machine-learning benchmark.

The model used:

* Poisson objective
* Exposure incorporated through the base margin
* The same core feature set as the actuarial models

This provides a comparison between a transparent, additive actuarial model and a flexible tree-based model capable of capturing non-linear relationships and interactions.

---

## Exploratory Data Analysis

The exploratory analysis examined claim frequency across the available rating factors.

### BonusMalus

BonusMalus emerged as the strongest risk driver. Claim frequency increased sharply with increasing BonusMalus score.

### Driver Age

Raw univariate analysis showed substantially higher claim frequency among younger drivers. However, this relationship changed after controlling for BonusMalus in the multivariate GLM.

This led to one of the key analytical findings of the project: a **genuine confounding effect** between driver age and claims history.

### Vehicle Age

Claim frequency was highest for the newest vehicles and declined with increasing vehicle age. This counter-intuitive relationship remained statistically significant in the GLM.

---

## Key Actuarial Finding — Driver Age Confounding

One of the most significant findings was the reversal of the Driver Age effect after BonusMalus was included in the frequency GLM.

The DrivAge coefficient became positive and highly significant after controlling for BonusMalus, despite the negative raw univariate relationship.

Variance Inflation Factors were low across the predictors, indicating that the result was not caused by unstable multicollinearity.

The interpretation is genuine confounding: younger drivers have higher raw risk but also tend to have lower BonusMalus because they have less claims history. Once BonusMalus captures much of the claims-history effect, the remaining conditional effect of age changes direction.

---

## Severity Modelling

The Gamma severity model found that most of the available rating factors had little explanatory power for claim size.

BonusMalus was the main significant predictor, while variables such as:

* Vehicle Power
* Vehicle Age
* Driver Age
* Density

did not show meaningful effects on severity.

This suggests that the available variables explain **how often a policyholder claims** considerably better than **how large a claim will be once a claim occurs**.

---

## Model Performance

### Gini Discrimination

XGBoost achieved stronger risk discrimination than the Poisson GLM:

| Model       |  Gini |
| ----------- | ----: |
| Poisson GLM | 0.252 |
| XGBoost     | 0.324 |

The XGBoost advantage was particularly driven by its ability to separate the highest-risk segment.

---

## Actual-vs-Expected Validation

Both models were evaluated on the untouched 30% holdout set using actuarial **Actual-vs-Expected (A/E)** analysis.

### Overall Calibration

| Model   | Actual Claims | Expected Claims |   A/E |
| ------- | ------------: | --------------: | ----: |
| GLM     |        10,846 |        10,777.7 | 1.006 |
| XGBoost |        10,846 |        11,404.6 | 0.951 |

The GLM was extremely well calibrated in aggregate, with an A/E ratio close to 1.00. XGBoost slightly overpredicted aggregate claims.

### Young Driver Segment

The aggregate results concealed an important segment-level issue.

For drivers aged 17–22:

| Model   |  A/E | 95% CI         |     p-value |
| ------- | ---: | -------------- | ----------: |
| GLM     | 1.38 | [1.228, 1.549] | 1.48 × 10⁻⁷ |
| XGBoost | 1.12 | [0.993, 1.252] |       0.060 |

The GLM significantly underpredicted risk for the youngest drivers, while the XGBoost difference was not statistically significant at the 5% level.

This demonstrates why **aggregate model performance alone is not sufficient for actuarial pricing validation**.

---

## Business Translation — Rating Relativities

GLM coefficients were converted into multiplicative rating relativities using the exponential transformation of the coefficients.

Selected results include:

| Rating Factor                                     | Relativity |
| ------------------------------------------------- | ---------: |
| Vehicle Age — per year                            |      0.962 |
| Driver Age — per year, controlling for BonusMalus |      1.006 |
| BonusMalus — per point                            |      1.023 |
| log(Density)                                      |      1.043 |
| Vehicle Power — per unit                          |      1.015 |

These relativities provide an actuarially interpretable way of translating model output into rating factors.

The project also demonstrates that rating factors compound multiplicatively. A combined high-risk profile can produce a frequency relativity exceeding **10 times the baseline**.

---

## Main Conclusion

The analysis does not treat GLM and machine learning as simple competing alternatives.

The results show that:

* **GLM provides strong aggregate calibration and transparency.**
* **XGBoost provides stronger risk discrimination.**
* XGBoost performs particularly well in identifying and calibrating the highest-risk segments.
* GLM can exhibit material segment-level miscalibration even when aggregate A/E is close to 1.
* The strongest evidence for additional ML value occurs in segments where the linear additive structure of the GLM cannot adequately capture interactions.

The resulting recommendation is a **hybrid pricing architecture**:

> **Use the GLM as the transparent actuarial rating engine, supplemented by ML-driven calibration checks and risk flags for segments requiring additional investigation or manual loadings.**

This approach preserves the interpretability and practical usability of a traditional actuarial rating model while using machine learning to identify potential areas of mispricing.

---

## Limitations and Further Work

The project identifies several areas for future development:

* Additional variables could improve claim severity modelling.
* Richer claim-level information and regional repair-cost measures could improve severity prediction.
* Large claims should ideally be modelled separately using an appropriate tail model or Extreme Value Theory rather than being capped.
* XGBoost could be further optimised using cross-validated hyperparameter tuning.
* A formal Tweedie AIC/BIC comparison could be conducted using specialised packages.
* The proposed GLM + ML flagging approach could be operationalised through defined override rules and ongoing model monitoring.

---

## Tools Used

* **R** — data preparation, exploratory analysis, GLM modelling, XGBoost, validation and visualisation
* **Excel** — supporting data and model outputs
* **Microsoft Word** — technical report preparation
* **Git/GitHub** — version control and project documentation
* **OpenML / CASdatasets** — source of the freMTPL2 motor insurance data

---

## Repository Structure

```text
motor-insurance-pricing-glm-project/
│
├── inputs/
│   └── Input datasets and supporting files
│
├── scripts/
│   └── R scripts for data preparation and modelling
│
├── reports/
│   ├── Motor_Insurance_Pricing_Report.docx
│   └── model_results.xlsx
│
├── .gitignore
├── .gitattributes
└── README.md
```

Additional output folders and pricing tools may be added as the project develops.

---

## Project Status

**Status: Completed modelling and validation**

The current project contains the complete modelling workflow from data preparation and exploratory analysis through actuarial GLM development, XGBoost benchmarking, out-of-sample validation and business interpretation.

Future work will focus on extending the feature set, improving severity modelling, tuning the machine-learning model and developing a more operational pricing implementation.

---

## Disclaimer

This project is an independent actuarial modelling exercise conducted for educational and analytical purposes.

The results are based on the freMTPL2 dataset and should not be interpreted as the pricing methodology or commercial practice of any particular insurer.

