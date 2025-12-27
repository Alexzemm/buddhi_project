import json
import pandas as pd
from kafka import KafkaConsumer
import joblib

# Load the trained model
model = joblib.load('titanic_ml/best_model.joblib')

consumer = KafkaConsumer(
    'titanic_passengers',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

drop_cols = ['Name', 'Cabin', 'Ticket', 'Survived']

for message in consumer:
    data = message.value
    df = pd.DataFrame([data])
    # Drop columns not used in training (adjust as needed)
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    # Predict
    prob = model.predict_proba(X)[:, 1][0]
    print(f"Passenger: {data} | Survival Probability: {prob:.2f}")
