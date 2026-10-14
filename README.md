# 🍽️ Food Waste Analysis and Prediction Using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeepaddanki1-spec/DM-Project/blob/main/Food_Waste_Analysis_and_Prediction.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/sandeepaddanki1-spec/DM-Project)
[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.0+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-EB662B?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Documentation PDF](https://img.shields.io/badge/Documentation-PDF-red?logo=adobe-acrobat-reader&logoColor=white)](docs/Food_Waste_Analysis_and_Prediction_Documentation.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning system developed for **Restaurant Food Wastage Prediction and Management**. The platform implements regression for precise quantity estimation ($kg$), multi-class classification for operational risk stratification (Low / Medium / High), unsupervised clustering for shift behavioral pattern discovery, and an intelligent prescriptive recommendation engine to mitigate commercial kitchen food wastage by up to 35–40%.

---

## 📌 Table of Contents
- [Project Overview & Motivation](#-project-overview--motivation)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Model Benchmarks & Empirical Results](#-model-benchmarks--empirical-results)
  - [1. Regression Models (Continuous Waste in kg)](#1-regression-models-continuous-waste-in-kg)
  - [2. Classification Models (3-Tier Operational Risk)](#2-classification-models-3-tier-operational-risk)
  - [3. Unsupervised Clustering (Operational Patterns)](#3-unsupervised-clustering-operational-patterns)
- [Model Interpretability & Feature Importance](#-model-interpretability--feature-importance)
- [Prescriptive Kitchen Recommendation Engine](#-prescriptive-kitchen-recommendation-engine)
- [Repository Structure](#-repository-structure)
- [Quick Start & Execution](#-quick-start--execution)
  - [Running on Google Colab](#1-running-on-google-colab-recommended)
  - [Running Locally](#2-running-locally)
- [Project Contributors](#-project-contributors)
- [License](#-license)

---

## 🎯 Project Overview & Motivation

Commercial food service operations and restaurant kitchens represent a significant contributor to global food waste, incurring substantial financial losses, supply chain inefficiencies, and environmental degradation. 

This project tackles food wastage through a multi-faceted machine learning pipeline:
1. **Accurate Continuous Forecasting**: Accurately predict wasted food in kilograms per service shift using non-linear ensemble regressors.
2. **Operational Risk Stratification**: Classify upcoming dining shifts into **Low**, **Medium**, or **High Waste** tiers using balanced tertile-derived thresholds to trigger proactive culinary safeguards.
3. **Behavioral Shift Clustering**: Uncover latent operating conditions (e.g., event surges vs. standard seating) via unsupervised clustering algorithms.
4. **Data-Driven Prescriptive Advisory**: Generate dynamic, context-aware operational recommendations for head chefs and restaurant management to reduce surplus preparation.

---

## 🏗️ System Architecture & Workflow

The platform follows a rigorous, production-ready machine learning lifecycle designed for seamless execution on Kaggle, Google Colab, and local environments:

![System Architecture Workflow](figures/fig_proposed_workflow.png)

1. **Ingestion & Imputation**: Automated ingestion of messy operational CSV logs, imputing missing numerical values via medians and categoricals via most-frequent strategy.
2. **Feature Engineering**: Dynamic mapping of chef experience hierarchy, guest capacity ratios, weather-condition indicators, and preparation lead-time indices.
3. **Benchmarking & Validation**: 5-fold cross-validation across multiple diverse model families.
4. **Champion Selection**: Selection of best-performing models based on RMSE, MAE, R², and Macro F1 scores.
5. **Prescriptive Engine**: Contextual recommendations tailored to risk tier and shift characteristics.

---

## 📊 Dataset & Feature Engineering

The training pipeline ingests raw restaurant shift data featuring complex interactions between dining room volume, kitchen staffing, and environmental factors:

| Feature Name | Type | Description |
|---|---|---|
| `guest_count` | Numerical | Total dining guests served during the service shift |
| `prep_time_minutes` | Numerical | Lead time allocated for prep station setup |
| `menu_items_count` | Numerical | Number of active unique dishes prepared |
| `pricing_category` | Categorical | Meal price tier (Budget, Standard, Premium) |
| `chef_experience` | Categorical | Kitchen lead seniority level (`beginner`, `intermediate`, `expert`) |
| `special_event` | Binary / Categorical | Presence of banquets, holiday catering, or private bookings |
| `weather_condition` | Categorical | Prevailing weather conditions (Sunny, Rainy, Cold, etc.) |
| `food_waste_kg` | Target (Continuous) | Actual recorded surplus kitchen waste in kilograms |

### Target Distribution
The empirical distribution of recorded food waste demonstrates a unimodal profile with a mean of **41.14 kg** and a median of **41.00 kg**:

![Target Waste Distribution](figures/fig_target_distribution.png)

---

## 🔬 Model Benchmarks & Empirical Results

### 1. Regression Models (Continuous Waste in kg)

All regression models were benchmarked using identical 5-fold cross-validation splits. The **Gradient Boosting Regressor** emerged as the decisive champion:

| Model | MAE ($kg$) | RMSE ($kg$) | $R^2$ Score | Status |
|:---|:---:|:---:|:---:|:---:|
| 🏆 **Gradient Boosting Regressor** | **4.878** | **8.340** | **0.903** | **Champion (Selected)** |
| 🌲 **Random Forest Regressor** | 5.241 | 9.149 | 0.886 | Compared |
| ⚡ **XGBoost Regressor** | 5.018 | 9.120 | 0.885 | Compared |
| 📈 **Linear Regression** | 5.618 | 11.069 | 0.824 | Baseline |

#### Comparative Performance & Fit
The comparison chart below highlights the superior variance explanation ($R^2 > 0.90$) and lowest error metrics achieved by Gradient Boosting:

![Regression Model Comparison](figures/fig_regression_comparison.png)

The predicted vs. actual plot confirms tight alignment along the 45-degree diagonal across both low and high-volume extremes:

![Predicted vs Actual Plot](figures/fig_predicted_vs_actual.png)

---

### 2. Classification Models (3-Tier Operational Risk)

To assist kitchen staff with simple, actionable shift risk warnings, target waste values were partitioned into balanced operational tertiles:
- **Low Waste**: $\le 35.64\text{ kg}$
- **Medium Waste**: $35.64\text{ kg} - 46.65\text{ kg}$
- **High Waste**: $> 46.65\text{ kg}$

| Model | Accuracy | Precision | Recall | Macro F1 | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Logistic Regression** | **74.4%** | **74.8%** | **74.4%** | **0.745** | **Champion (Selected)** |
| 🌳 **Extra Trees Classifier** | 75.0% | 74.7% | 75.0% | 0.744 | Compared |
| 🌲 **Random Forest Classifier** | 73.7% | 74.2% | 73.7% | 0.737 | Compared |
| ⚡ **XGBoost Classifier** | 73.7% | 73.8% | 73.6% | 0.735 | Compared |

*Logistic Regression with standard scaling was selected as the champion classifier for its robust cross-validated macro-F1 score (0.745), minimal risk of overfitting, and instantaneous inference latency.*

---

### 3. Unsupervised Clustering (Operational Patterns)

Unsupervised clustering was applied to discover latent operational dynamics without target contamination:
- **K-Means Clustering ($K=2$)**: Silhouette Score = **0.352**
- **Agglomerative Hierarchical Clustering ($K=2$)**: Silhouette Score = **0.341**
- **Cluster Agreement**: Adjusted Rand Index (**ARI = 0.97**), demonstrating near-perfect agreement between partition structures.
- **Key Insight**: 100% of cluster partitioning aligned with banquet / special event loading, isolating high-intensity banqueting shifts from regular daily dining operations.

---

## 🔍 Model Interpretability & Feature Importance

Inspection of the champion Gradient Boosting Regressor reveals the primary operational drivers governing commercial food waste:

![Feature Importances](figures/fig_feature_importances.png)

1. **Guest Count & Dining Volume**: Serves as the dominant volume multiplier.
2. **Pricing Category**: Higher pricing tiers correspond to complex multi-course preparations with elevated plate garnishes.
3. **Chef Seniority & Experience**: Intermediate and beginner shifts experience higher prep trimming losses compared to expert culinary leads.
4. **Prep Time Lead**: Long prep times correlate with batch over-preparation during peak shifts.

---

## 💡 Prescriptive Kitchen Recommendation Engine

The built-in `recommend_actions()` function translates raw model predictions into actionable culinary protocols:

```python
from food_waste_features import recommend_actions

# Example shift profile
shift_input = {
    "guest_count": 185,
    "chef_experience": "beginner",
    "special_event": 1,
    "pricing_category": "Premium"
}

guidance = recommend_actions(predicted_kg=58.4, risk_tier="High Waste", meta=shift_input)
print(guidance)
```

**Automated Action Items Triggered:**
- 🚨 **High Waste Alert**: Implement pre-shift portion-control briefing and restrict batch prep to 80% baseline.
- 👨‍🍳 **Culinary Supervision**: Pair beginner station chefs with senior kitchen supervisors on trimming stations.
- 🎉 **Banqueting Buffers**: Transition buffet dessert and cold stations to cook-to-order replenishments.
- 📦 **Surplus Redistribution**: Pre-schedule food bank recovery or staff meal reallocation 2 hours prior to service conclusion.

---

## 📁 Repository Structure

```text
Food-Waste-Analysis-and-Prediction-Using-Machine-Learning/
├── dataset/
│   ├── train.csv                                  # Training dataset with target food_waste_kg
│   ├── test.csv                                   # Evaluation dataset
│   └── submission.csv                             # Sample submission benchmark format
├── docs/
│   └── Food_Waste_Analysis_and_Prediction_Documentation.pdf # Full 19-page academic project documentation
├── figures/
│   ├── fig_proposed_workflow.png                  # System architecture diagram
│   ├── fig_target_distribution.png                # Food waste histogram and density
│   ├── fig_regression_comparison.png              # Multi-model regression benchmark chart
│   ├── fig_predicted_vs_actual.png                # Gradient Boosting goodness-of-fit
│   └── fig_feature_importances.png                # Feature importance rankings
├── Food_Waste_Analysis_and_Prediction.ipynb       # Primary Colab-optimized research notebook
├── dm-capstone-project.ipynb                      # Kaggle capstone evaluation notebook
├── food_waste_features.py                         # Standalone modular preprocessing & cleaning functions
├── requirements.txt                               # Pinned library dependencies
├── .gitignore                                     # Standard Python / Jupyter exclusion rules
└── README.md                                      # Comprehensive project documentation
```

---

## 🚀 Quick Start & Execution

### 1. Running on Google Colab (Recommended)
You can launch and run the entire pipeline in Google Colab with a single click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeepaddanki1-spec/DM-Project/blob/main/Food_Waste_Analysis_and_Prediction.ipynb)

*The notebook automatically clones the repository and loads the local `dataset/` files with zero manual upload required.*

---

### 2. Running Locally

#### Step 1: Clone the repository
```bash
git clone https://github.com/sandeepaddanki1-spec/DM-Project.git
cd Food-Waste-Analysis-and-Prediction-Using-Machine-Learning
```

#### Step 2: Create a virtual environment & install requirements
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook Food_Waste_Analysis_and_Prediction.ipynb
```

---

## 👥 Project Contributors

This project was developed by:

| Name | Roll Number | GitHub Profile |
|:---|:---:|:---:|
| **A. Sandeep** | `24501A0502` | [@sandeepaddanki1-spec](https://github.com/sandeepaddanki1-spec) |
| **G. Bhavyasree** | `24501A0558` | [@gogabhavyasree](https://github.com/gogabhavyasree) |
| **P. Dimple Hasini** | `24501A0549` | — |
| **A. Kalpana** | `24501A0509` | — |

**Academic Year:** 2026–2027

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free for academic, personal, and commercial research.
