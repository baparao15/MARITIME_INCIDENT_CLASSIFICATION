# Maritime Incident Severity Classification System

A machine learning-based system for automatically classifying maritime incidents by severity level using natural language processing and ensemble learning techniques.

## 🎯 Project Overview

This project implements an intelligent classification system that analyzes maritime incident descriptions and automatically categorizes them into five severity levels:

- **Level 1 (Critical)**: Fatalities, vessel loss, explosions, major oil spills
- **Level 2 (Serious)**: Serious injuries, major damage, evacuations, significant fires
- **Level 3 (Moderate)**: Collisions, groundings, engine failures, moderate injuries
- **Level 4 (Minor)**: Minor injuries, slight damage, first aid cases
- **Level 5 (Administrative)**: Regulatory non-compliance, training exercises, inspections

## 🚀 Features

- **Automated Severity Classification**: Uses machine learning to classify incidents based on text descriptions
- **Multi-Model Comparison**: Evaluates Random Forest, Gradient Boosting, SVM, and Logistic Regression
- **Web Interface**: Flask-based web application for real-time predictions
- **Data Preprocessing**: Comprehensive data cleaning and feature engineering pipeline
- **SMOTE Implementation**: Handles class imbalance using custom SMOTE implementation
- **Performance Visualization**: Detailed model comparison charts and metrics

## 📊 Dataset

The system uses maritime incident data with the following features:
- Latitude/Longitude coordinates
- Occurrence location
- Weather conditions
- Wind force and sea state
- Visibility and natural light
- Short and detailed incident descriptions
- Main event classification

## 🛠️ Technology Stack

- **Python 3.x**
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Web Framework**: Flask
- **NLP**: TF-IDF Vectorization

## 📁 Project Structure

```
MARITIME_INCIDENT_CLASSIFICATION/
├── main.ipynb                          # Main Jupyter notebook with ML pipeline
├── app.py                              # Flask web application
├── data.csv                            # Raw maritime incident data
├── classified_dataset.csv              # Dataset with severity labels
├── classified_dataset_final.csv        # Final processed dataset
├── maritime_model.pkl                  # Trained model (generated)
├── tfidf_vectorizer.pkl               # TF-IDF vectorizer (generated)
├── templates/
│   └── index.html                      # Web interface template
└── static/
    └── (CSS/JS files)
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/baparao15/MARITIME_INCIDENT_CLASSIFICATION
cd MARITIME_INCIDENT_CLASSIFICATION
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib flask joblib xgboost
```

3. Ensure the dataset file `data.csv` is in the project directory

## 💻 Usage

### Running the Jupyter Notebook

1. Open the notebook:
```bash
jupyter notebook main.ipynb
```

2. Run cells sequentially to:
   - Load and clean data
   - Classify incidents by severity
   - Train multiple ML models
   - Compare model performance
   - Generate visualizations

### Running the Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter an incident description to get real-time severity classification

## 📈 Model Performance

The system compares multiple machine learning algorithms:

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Gradient Boosting | 0.627 | 0.568 | 0.585 | 0.627 |
| Random Forest | 0.615 | 0.557 | 0.569 | 0.615 |
| SVM | 0.542 | 0.414 | 0.405 | 0.542 |
| Logistic Regression | 0.504 | 0.389 | 0.364 | 0.504 |

*Note: Performance metrics may vary based on data preprocessing and hyperparameters*

## 🔍 Key Components

### 1. Data Preprocessing
- Removal of unnecessary columns
- Handling missing values
- Text cleaning and normalization

### 2. Severity Classification
- Keyword-based initial classification
- 370+ severity-specific keywords across 5 categories
- Machine learning refinement for unclassified incidents

### 3. Feature Engineering
- TF-IDF vectorization of text descriptions
- N-gram features (unigrams and bigrams)
- Categorical encoding of location and weather data

### 4. Model Training
- Train/test split (80/20)
- Cross-validation for hyperparameter tuning
- Ensemble methods for improved accuracy

### 5. Class Imbalance Handling
- Custom SMOTE implementation
- Oversampling minority classes
- Balanced training data generation

## 📝 API Endpoints

### Web Application

- `GET /` - Main interface
- `POST /predict` - Classify incident severity
  ```json
  {
    "text": "Vessel experienced engine failure and required towing"
  }
  ```
- `GET /health` - Health check endpoint

## 🎓 Educational Value

This project demonstrates:
- End-to-end machine learning pipeline development
- Text classification using NLP techniques
- Handling imbalanced datasets
- Model comparison and selection
- Web application deployment
- Real-world application of ML in maritime safety

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

