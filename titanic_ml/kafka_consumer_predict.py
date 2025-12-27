import json
import pandas as pd
from kafka import KafkaConsumer
import joblib

# Load the trained model
model = joblib.load('best_model.joblib')

consumer = KafkaConsumer(
    'titanic_passengers',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

drop_cols = ['Name', 'Cabin', 'Ticket', 'Survived']

for message in consumer:
    msg = message.value
    # Expecting {'normalized': ..., 'original': ...}
    norm_data = msg['normalized']
    orig_data = msg['original']
    df = pd.DataFrame([norm_data])
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    prob = model.predict_proba(X)[:, 1][0]
    # Print original data with prediction
    print(f"Passenger: {orig_data} | Survival Probability: {prob:.2f}")
