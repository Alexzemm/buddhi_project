# Titanic Survival Prediction Project

This project predicts Titanic passenger survival using machine learning, with both batch and real-time (Kafka) prediction, interactive filtering, and a Streamlit UI.

## Features & Deliverables

- **Cleaned and preprocessed Titanic dataset** (missing values handled, features engineered)
- **Exploratory Data Analysis (EDA)** with visualizations and insights (see Jupyter notebook)
- **Machine learning models** (Logistic Regression, Random Forest, etc.) with evaluation metrics
- **Streamlit UI** for batch and real-time prediction, with filtering and sorting
- **Real-time prediction system using Kafka** (producer/consumer)
- **All code, data, and notebooks** in this repository

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Start Kafka using Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Run the main pipeline to train and save the model**
   ```bash
   python titanic_ml/main.py
   ```

5. **Start the Kafka producer (for real-time streaming)**
   ```bash
   python titanic_ml/kafka_producer.py
   ```

6. **Start the Streamlit app**
   ```bash
   streamlit run titanic_ml/streamlit_app.py
   ```

7. **Use the UI**
   - Explore batch predictions, filter and sort by class, age, gender, etc.
   - Switch to the real-time explorer to see live predictions as data streams in via Kafka.

## Repository Structure

- `titanic/` — Raw data files
- `titanic_ml/` — ML code, Streamlit app, Kafka scripts, cleaned data
- `notebooks/` — EDA and analysis notebooks
- `results.csv` — Model predictions
- `best_model.joblib` — Saved model

## Demo Video

[Insert your video link here]

## Credits

Developed by [Your Name].

---

For any questions, please contact [your.email@example.com].
