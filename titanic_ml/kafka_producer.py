
import json
import pandas as pd
from kafka import KafkaProducer

# Load normalized and original test data
df_norm = pd.read_csv('titanic_ml/titanic_cleaned_test.csv')
df_orig = pd.read_csv('titanic/test.csv')

# Ensure both dataframes align by index (assume same order)
if len(df_norm) != len(df_orig):
    raise ValueError("Normalized and original test data row counts do not match!")

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for idx, row in df_norm.iterrows():
    norm_data = row.to_dict()
    orig_data = df_orig.iloc[idx].to_dict()
    # Send both normalized and original data in one message
    message = {
        'normalized': norm_data,
        'original': orig_data
    }
    producer.send('titanic_passengers', value=message)
    print(f"Sent: {message}")

producer.flush()
producer.close()
