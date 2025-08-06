# Machine Failure Prediction Using Sensor Data

## Project Overview
This project is focused on building machine learning models to **predict machine failures** in advance using real-time **sensor data**. Predictive maintenance enables industries to proactively fix issues before they lead to costly breakdowns.

## Objective
- Analyze sensor data to identify patterns associated with machine failure.
- Build and evaluate multiple machine learning models.
- Identify key sensor features affecting failures.
- Deploy a model that can be integrated into real-time monitoring systems.

---

## Dataset Overview

| Column Name   | Description                                              |
|---------------|----------------------------------------------------------|
| `footfall`    | Number of objects/people passing the machine             |
| `tempMode`    | Machine temperature mode                                 |
| `AQ`          | Air Quality Index                                        |
| `USS`         | Ultrasonic Sensor data                                   |
| `CS`          | Current sensor readings                                  |
| `VOC`         | Volatile Organic Compounds near machine                  |
| `RP`          | Rotational position / RPM                                |
| `IP`          | Input pressure                                           |
| `Temperature` | Operating temperature                                    |
| `fail`        | Target variable (1 = failure, 0 = no failure)            |

- **Rows:** 944
- **Features:** 9
- **Target:** Binary (`fail`)

---

## Exploratory Data Analysis

- **Balanced target**: Sufficient samples of both failure and non-failure cases.
- **Strong predictors**: `VOC`, `AQ`, and `USS` showed high correlation with failure.
- **Visualizations**: Correlation heatmap, pairplot, distribution plots, and feature importance plots.

---

## Machine Learning Models

| Model                | Type           | Notes                            |
|---------------------|----------------|----------------------------------|
| Logistic Regression | Classification | Good baseline model              |
| Decision Tree        | Classification | Best performing model            |
| Linear Regression    | Regression     | Used for R², MAE, and MSE checks |

---

## Model Evaluation

**Classification Reports** provided for Logistic Regression, Decision Tree, and rounded Linear Regression.

### Linear Regression Metrics:
- **R² Score**: ~0.35  
- **MAE**: ~0.27  
- **MSE**: ~0.11  

### Best Model: Decision Tree Classifier
- High interpretability
- Captures non-linear relationships
- Visual feature importance

---

## Top 5 Features (by importance):

| Feature      | Importance |
|--------------|------------|
| `VOC`        | ★★★★★       |
| `AQ`         | ★★★★☆       |
| `USS`        | ★★★☆☆       |
| `Temperature`| ★★★☆☆       |
| `IP`         | ★★★☆☆       |

---

## Tools & Libraries Used
- Python, Google Colab
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `joblib` (for model saving)

---

## Model Export

The final decision tree model was saved using `joblib`:

```python
import joblib
joblib.dump(tree_model, 'machine_failure_model.pkl')
