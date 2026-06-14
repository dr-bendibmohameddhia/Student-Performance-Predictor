# 🎓 Student Performance Predictor

> **Startup-level AI application** for predicting student success and identifying at-risk students — built with modern ML, explainable AI, and a stunning glassmorphism UI.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![SHAP](https://img.shields.io/badge/SHAP-0.44-purple)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## 🚀 Overview

Student Performance Predictor is a production-ready AI system that:

- **Predicts** whether a student will pass or fail using 4 ML models
- **Identifies at-risk students** early for timely intervention
- **Explains predictions** using SHAP (SHapley Additive exPlanations)
- **Visualizes** deep analytics through an interactive dashboard
- **Compares** model performance with transparent metrics

---

## 🏗️ Architecture

```
student_performance_predictor/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-service orchestration
│
├── data/
│   ├── raw/                        # Original dataset
│   │   └── student_performance.csv
│   ├── processed/                  # Engineered features
│   │   └── student_processed.csv
│   └── models/                     # Trained model artifacts
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       ├── lightgbm.pkl
│       ├── scaler.pkl
│       ├── imputer.pkl
│       ├── feature_cols.pkl
│       └── results.json
│
├── src/
│   ├── data/
│   │   └── loader.py               # Data ingestion & validation
│   ├── features/
│   │   └── engineer.py             # Feature engineering pipeline
│   ├── models/
│   │   └── predictor.py            # Model loading & inference
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP value computation
│   ├── visualization/
│   │   └── charts.py               # Plotly chart factory
│   └── utils/
│       └── __init__.py
│
├── tests/                          # Unit tests
├── docs/                           # Documentation
└── .streamlit/
    └── config.toml                 # Streamlit theme config
```

---

## ✨ Features

### 🏠 Dashboard
- Live KPI cards: students, pass rate, at-risk count, model accuracy
- Grade distribution charts, study time analysis
- Risk scatter plots and absence impact analysis
- At-risk student table with risk scoring

### 🔮 Prediction Engine
- Interactive form for entering student details
- Real-time predictions from all 4 models
- Confidence bars and probability scores
- Risk badge (Low / Medium / High) with actionable recommendations

### 📊 Analytics
- Grade distributions and box plots
- Feature correlation heatmap
- Demographic breakdowns (gender, address, age)
- G1 vs G2 scatter analysis

### 🧠 Explainability (SHAP)
- Global feature importance bar chart
- SHAP value distributions per feature
- Model-specific explanations (RF, XGB, LGBM)

### ⚖️ Model Comparison
- Side-by-side metrics: Accuracy, AUC, F1
- Grouped bar chart comparison
- Gauge charts for best model
- Selection rationale

---

## 🧠 ML Pipeline

### Dataset
- **2,000 students**, **34 raw features**, **42 engineered features**
- Based on UCI Student Performance dataset structure
- Features: demographics, parental education, study habits, alcohol use, grades

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `parent_edu_avg` | Average of mother + father education |
| `alcohol_total` | Workday + weekend alcohol sum |
| `support_score` | School + family support combined |
| `academic_history` | Average of G1 + G2 grades |
| `grade_improvement` | G2 − G1 trend |
| `study_efficiency` | Study time / (go_out + 1) |
| `at_risk_score` | Composite risk indicator |
| `absence_category` | Binned absence level (0–3) |

### Models Trained
| Model | Accuracy | AUC | F1 |
|-------|----------|-----|----|
| Logistic Regression | ~95.8% | ~98.3% | ~97.5% |
| Random Forest | ~95.0% | ~97.4% | ~97.0% |
| XGBoost | ~93.5% | ~97.6% | ~96.1% |
| LightGBM | ~94.0% | ~97.6% | ~96.4% |

### Explainability
- **SHAP TreeExplainer** for tree-based models
- **SHAP LinearExplainer** for Logistic Regression
- Global + local explanations

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-org/student-performance-predictor
cd student-performance-predictor
pip install -r requirements.txt
```

### 2. Generate Data & Train Models
```bash
python scripts/generate_data.py
python scripts/train_models.py
```

### 3. Run Dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501**

---

## 🐳 Docker Deployment

```bash
# Build & run
docker-compose up --build

# Production
docker build -t student-predictor .
docker run -p 8501:8501 student-predictor
```

---

## ☁️ Cloud Deployment

### Streamlit Community Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo → Deploy

### Heroku
```bash
heroku create student-predictor
heroku stack:set container
git push heroku main
```

### AWS ECS / GCP Cloud Run
Use the provided `Dockerfile` with your cloud container registry.

---

## 🔬 Design Principles

- **Clean Architecture**: Domain logic isolated from UI
- **SOLID Principles**: Each module has a single responsibility
- **Modular Code**: Feature engineering, models, and charts are independent
- **Production Ready**: Error handling, caching, type hints throughout

---

## 📄 License

MIT License — built for educational and research purposes.

---

## 📸 Screenshots

### Dashboard
![Dashboard](student_performance_predictor\assets\Dashboard.png)

### Predict Student
![Predict Student](student_performance_predictor\assets\predict student.png)

### Analytics       
![Analytics](student_performance_predictor\assets\analytics.png)

### Explainability
![Explainability](student_performance_predictor\assets\explainability.png)

### Model Comparaison
![Model comparaison](student_performance_predictor\assets\Model Comparaison.png)

Developed by BENDIB Mohamed Dhia
