import json
import pandas as pd
from kafka import KafkaProducer

# Load test data (or any data you want to stream)
df = pd.read_csv('titanic_ml/titanic_cleaned_test.csv')

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for _, row in df.iterrows():
    data = row.to_dict()
    producer.send('titanic_passengers', value=data)
    print(f"Sent: {data}")

producer.flush()
producer.close()
