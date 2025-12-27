
import streamlit as st
import pandas as pd
import json
from kafka import KafkaConsumer
import joblib
import threading
import time

# Load original test set for imputation values (for display)
_orig_df = pd.read_csv('../titanic/test.csv')
# Compute fill values from original data
_median_age = _orig_df['Age'].median()
_median_fare = _orig_df['Fare'].median()
_mode_embarked = _orig_df['Embarked'].mode()[0] if not _orig_df['Embarked'].mode().empty else 'Unknown'
# Cabin: always use 'Unknown' for missing
_mode_ticket = _orig_df['Ticket'].mode()[0] if 'Ticket' in _orig_df.columns and not _orig_df['Ticket'].mode().empty else 'Unknown'

st.set_page_config(page_title="Real-Time Kafka Explorer", layout="wide")

# Shared list to store predictions
data_buffer = []

# Function to run Kafka consumer in a thread
def run_kafka_consumer(buffer, stop_event):
    print("[DEBUG] Kafka consumer thread started")
    model = joblib.load('best_model.joblib')
    consumer = KafkaConsumer(
        'titanic_passengers',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest'
    )
    drop_cols = ['Name', 'Cabin', 'Ticket', 'Survived']
    for message in consumer:
        if stop_event.is_set():
            break
        msg = message.value
        # Expecting {'normalized': ..., 'original': ...}
        norm_data = msg['normalized']
        orig_data = msg['original']
        df = pd.DataFrame([norm_data])
        X = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
        prob = model.predict_proba(X)[:, 1][0]
        # Add prediction to original data
        orig_data_disp = orig_data.copy()
        # Fill missing values using cleaned set statistics
        if orig_data_disp.get('Age') in [None, '', 'Unknown'] or (isinstance(orig_data_disp.get('Age'), float) and pd.isna(orig_data_disp.get('Age'))):
            orig_data_disp['Age'] = _median_age
        if orig_data_disp.get('Fare') in [None, '', 'Unknown'] or (isinstance(orig_data_disp.get('Fare'), float) and pd.isna(orig_data_disp.get('Fare'))):
            orig_data_disp['Fare'] = _median_fare
        if orig_data_disp.get('Embarked') in [None, '', 'Unknown'] or (isinstance(orig_data_disp.get('Embarked'), float) and pd.isna(orig_data_disp.get('Embarked'))):
            orig_data_disp['Embarked'] = _mode_embarked
        if orig_data_disp.get('Cabin') in [None, '', 'Unknown'] or (isinstance(orig_data_disp.get('Cabin'), float) and pd.isna(orig_data_disp.get('Cabin'))):
            orig_data_disp['Cabin'] = 'Unknown'
        if orig_data_disp.get('Ticket') in [None, '', 'Unknown'] or (isinstance(orig_data_disp.get('Ticket'), float) and pd.isna(orig_data_disp.get('Ticket'))):
            orig_data_disp['Ticket'] = _mode_ticket
        # For any other missing values, fallback to empty string
        for k, v in orig_data_disp.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                orig_data_disp[k] = ''
        orig_data_disp['Survival_Prob'] = prob
        buffer.append(orig_data_disp)
        print(f"[DEBUG] Buffer length after append: {len(buffer)}")
    consumer.close()

def run():
    st.title("Real-Time Titanic Survival Prediction (Kafka)")
    st.write("Live predictions as passenger data arrives via Kafka.")
    
    if 'kafka_thread' not in st.session_state:
        st.session_state.kafka_thread = None
        st.session_state.stop_event = threading.Event()
        st.session_state.data_buffer = []

    if st.button("Start Listening to Kafka"):
        if st.session_state.kafka_thread is None or not st.session_state.kafka_thread.is_alive():
            st.session_state.stop_event.clear()
            st.session_state.data_buffer = []
            st.session_state.kafka_thread = threading.Thread(
                target=run_kafka_consumer,
                args=(st.session_state.data_buffer, st.session_state.stop_event),
                daemon=True
            )
            st.session_state.kafka_thread.start()
            st.success("Started listening to Kafka topic 'titanic_passengers'.")
        else:
            st.info("Already listening to Kafka.")

    if st.button("Stop Listening"):
        if st.session_state.kafka_thread and st.session_state.kafka_thread.is_alive():
            st.session_state.stop_event.set()
            st.session_state.kafka_thread.join(timeout=2)
            st.success("Stopped listening to Kafka.")

    st.write("---")
    st.subheader("Live Predictions (Original Data)")
    refresh = st.button("Refresh Table")
    st.write(f"[DEBUG] Buffer length in session: {len(st.session_state.get('data_buffer', []))}")
    if refresh:
        if st.session_state.get('data_buffer'):
            df = pd.DataFrame(st.session_state.data_buffer)
            # Show selected columns for clarity
            display_cols = ['PassengerId','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survival_Prob']
            display_cols = [col for col in display_cols if col in df.columns]
            if 'Survival_Prob' in df.columns:
                df = df.copy()
                df['Survival_Prob (%)'] = (df['Survival_Prob'] * 100).round(1)
                display_cols = [col if col != 'Survival_Prob' else 'Survival_Prob (%)' for col in display_cols]
            st.write(f"Filtered Results ({len(df)} passengers):")
            st.dataframe(df[display_cols], use_container_width=True)
        else:
            st.info("No predictions received yet. Start the producer to send data.")
