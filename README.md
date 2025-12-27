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

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. (Recommended) Run everything with Docker Compose
```bash
docker-compose up --build
```
This will start Zookeeper, Kafka, and the Streamlit app in containers. The app will be available at [http://localhost:8501](http://localhost:8501).

To stream data in real-time, open a new terminal and run the Kafka producer inside the app container:
```bash
docker-compose exec app python titanic_ml/kafka_producer.py
```

### 3. (Alternative) Run locally without Docker
1. Install dependencies:
   ```bash
   pip install -r titanic_ml/requirements.txt
   ```
2. Start Kafka and Zookeeper (e.g., with Docker Compose):
   ```bash
   docker-compose up -d kafka zookeeper
   ```
3. Run the main pipeline:
   ```bash
   python titanic_ml/main.py
   ```
4. Start the Streamlit app:
   ```bash
   streamlit run titanic_ml/streamlit_app.py
   ```
5. In a separate terminal, run the Kafka producer:
   ```bash
   python titanic_ml/kafka_producer.py
   ```

### 4. Using the UI
- Explore batch predictions, filter and sort by class, age, gender, etc.
- Switch to the real-time explorer to see live predictions as data streams in via Kafka.


## Repository Structure

- `titanic/` — Raw data files
- `titanic_ml/` — ML code, Streamlit app, Kafka scripts, cleaned data, model
- `notebooks/` — EDA and analysis notebooks
- `results.csv` — Model predictions (output)
- `best_model.joblib` — Saved model
- `Dockerfile`, `docker-compose.yml` — Containerization and orchestration


## Demo Video

[Insert your video link here]


## Credits

Developed by [Your Name].

---

For any questions, please contact [your.email@example.com].
