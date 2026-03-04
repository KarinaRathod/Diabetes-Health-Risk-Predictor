

### 1. `requirements.txt`

This file lists all the Python libraries required to run your application.

```text
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.14.0
scikit-learn>=1.2.0

```

---

### 2. `.gitignore`

This file tells Git which files and folders to ignore when tracking your project. It prevents you from accidentally committing virtual environments, temporary cache files, or the large CSV datasets.

```text
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Virtual Environments
venv/
env/
.env
.venv/

# Streamlit specific
.streamlit/secrets.toml

# IDE / Editor folders
.vscode/
.idea/
*.swp
*.swo

# Data files (Ignores the large datasets to prevent GitHub upload errors)
diabetes_binary_health_indicators_BRFSS2021.csv
diabetes_binary_5050split_health_indicators_BRFSS2021.csv
diabetes_012_health_indicators_BRFSS2021.csv
*.csv
*.xlsx

# OS generated files
.DS_Store
.DS_Store?
ehthumbs.db
Icon?
Thumbs.db

```

---

### 3. `README.md`

This is the documentation for your project, explaining what it is, how it works, and how to set it up.

```markdown
# 🩺 Diabetes Health Risk Predictor

An AI-powered Streamlit web application that predicts the risk of diabetes based on lifestyle and health factors using the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2021 dataset.



[Image of random forest algorithm diagram]


The engine uses a Machine Learning pipeline powered by a **Random Forest Classifier** to analyze patient vitals, demographics, and lifestyle choices, offering a real-time probability of diabetes risk.

## 🌟 Features

* **🔍 Prediction Engine:** An interactive sidebar to input 21 different health indicators (e.g., BMI, High Blood Pressure, Age, General Health). The model instantly outputs a risk assessment and a visual probability gauge.
* **📊 Data Insights:** Explores the underlying population statistics from the training data, visualizing relationships between factors like BMI, Blood Pressure, and Diabetes prevalence.
* **⚙️ Model Performance:** Displays real-time model telemetry, including accuracy, a confusion matrix, and a Feature Importance chart showing which health factors carry the most weight in the AI's decision-making.

## 📁 Project Structure

```text
diabetes-risk-predictor/
├── app.py                                                      # Main Streamlit application code
├── diabetes_binary_5050split_health_indicators_BRFSS2021.csv   # Balanced training dataset (Required)
├── requirements.txt                                            # Python dependencies
├── .gitignore                                                  # Git ignore rules
└── README.md                                                   # Project documentation

```

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone [https://github.com/yourusername/diabetes-risk-predictor.git](https://github.com/yourusername/diabetes-risk-predictor.git)
cd diabetes-risk-predictor

```

**2. Create a virtual environment (Recommended)**

```bash
python -m venv venv
# On Mac/Linux:
source venv/bin/activate  
# On Windows:
venv\Scripts\activate

```

**3. Install dependencies**

```bash
pip install -r requirements.txt

```

**4. Add your dataset**
Ensure that the `diabetes_binary_5050split_health_indicators_BRFSS2021.csv` file is placed in the root directory of the project (the exact same folder as `app.py`).

**5. Run the application**

```bash
streamlit run app.py

```

## 🛠️ Technologies Used

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `train_test_split`)
* **Data Visualization:** Plotly Express & Graph Objects

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It relies on historical statistical data and is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult with a qualified healthcare provider regarding your health or medical conditions.

```

```