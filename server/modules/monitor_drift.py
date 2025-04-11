# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import time
# from pathlib import Path
# from faker import Faker
# from tensorflow.keras.models import load_model
# from train_autoencoder import train_autoencoder

# # Configs
# fake = Faker()
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_PATH = BASE_DIR / "modules/fraud_autoencoder_model.h5"
# PREPROCESSOR_PATH = BASE_DIR / "modules/fraud_preprocessor.pkl"
# THRESHOLD_PATH = BASE_DIR / "modules/fraud_threshold.txt"
# DATA_PATH = BASE_DIR / "non_fraud_transactions.csv"

# st.set_page_config(page_title="Drift Monitoring Dashboard", layout="wide")
# st.title("📡 Real-Time Drift & Model Monitoring")

# placeholder_chart = st.empty()
# placeholder_status = st.empty()

# history = []

# @st.cache_data(show_spinner=False)
# def load_assets():
#     model = load_model(MODEL_PATH, compile=False)
#     preprocessor = joblib.load(PREPROCESSOR_PATH)
#     with open(THRESHOLD_PATH, "r") as f:
#         threshold = float(f.read().strip())
#     return model, preprocessor, threshold

# model, preprocessor, threshold = load_assets()

# def send_email_alert(drift_score, fraud_rate):
#     import smtplib, ssl, os
#     from email.message import EmailMessage
#     from dotenv import load_dotenv

#     load_dotenv()
#     email_sender = os.getenv("EMAIL_USER")
#     email_password = os.getenv("EMAIL_PASS")
#     email_receiver = os.getenv("TO_EMAIL")

#     subject = "🚨 Model Drift Alert!"
#     body = (
#         f"Drift MSE: {drift_score:.4f}\n"
#         f"Detected Fraud Rate: {fraud_rate:.2%}\n\n"
#         "Model performance has degraded. Consider retraining."
#     )

#     msg = EmailMessage()
#     msg["From"] = email_sender
#     msg["To"] = email_receiver
#     msg["Subject"] = subject
#     msg.set_content(body)

#     context = ssl.create_default_context()
#     with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
#         smtp.login(email_sender, email_password)
#         smtp.send_message(msg)

#     print("📧 Email alert sent.")


# def generate_anomalous_data(existing_df, num_rows=300):
#     anomalies = []
#     customer_ids = existing_df["customer_id"].unique().tolist()

#     for _ in range(num_rows):
#         row = {
#             "transaction_id": fake.uuid4(),
#             "timestamp": fake.date_time_this_year().isoformat(),
#             "amount": np.random.normal(5000, 2000),  # Instead of huge anomalies
#             "transaction_type": fake.random_element(["card", "upi", "wallet", "pos"]),
#             "merchant": fake.company(),
#             "location": fake.country(),
#             "is_foreign": fake.boolean(chance_of_getting_true=10),  # mostly domestic
#             "is_high_risk_country": fake.boolean(chance_of_getting_true=5),
#             "opening_balance": np.random.normal(40000, 10000),
#             "closing_balance": np.random.normal(30000, 8000),
#             "account_id": fake.uuid4(),
#             "account_type_txn": fake.random_element(["savings", "current"]),
#             "account_type_acct": fake.random_element(["savings", "current"]),
#             "account_number": fake.random_number(digits=10),
#             "balance": np.random.normal(20000, 5000),
#             "created_at_acct": fake.date_this_decade(),
#             "customer_id": fake.random_element(customer_ids),
#             "name": fake.name(),
#             "email": fake.email(),
#             "phone": fake.phone_number(),
#             "address": fake.address(),
#             "dob": fake.date_of_birth().isoformat(),
#             "created_at_cust": fake.date_this_decade(),
#             "past_txn_count": np.random.randint(5, 20),
#             "past_avg_amount": np.random.normal(4500, 1500),
#             "past_common_merchant": fake.company(),
#             "past_common_location": fake.country(),
#             "agg_txn_count": np.random.randint(5, 15),
#             "agg_avg_amount": np.random.normal(4800, 1600),
#             "agg_std_amount": np.random.normal(1200, 400),
#             "agg_max_amount": np.random.normal(10000, 3000),
#             "agg_unique_merchants": np.random.randint(1, 3),
#             "agg_unique_locations": np.random.randint(1, 3),
#         }
#         anomalies.append(row)

#     return pd.DataFrame(anomalies)


# # Main Loop
# st.sidebar.title("⚙️ Controls")
# run_monitoring = st.sidebar.toggle("Run Drift Monitoring", value=False)

# df_old = pd.read_csv(DATA_PATH)

# if run_monitoring:
#     with st.spinner("Monitoring data drift..."):
#         for i in range(100):
#             df_new = generate_anomalous_data(df_old)
#             X = df_new.copy()
#             X = X.fillna("missing")
#             for col in X.select_dtypes(include=["object", "category"]).columns:
#                 X[col] = X[col].astype(str)

#             X_processed = preprocessor.transform(X)
#             recon = model.predict(X_processed)
#             mse = np.mean(np.square(X_processed - recon), axis=1)
#             predicted_fraud = (mse > threshold).astype(int)

#             drift_score = mse.mean()
#             fraud_rate = predicted_fraud.mean()
#             history.append({"Step": i, "Drift MSE": drift_score, "Detected Fraud Rate": fraud_rate})

#             status = f"\n**Drift MSE:** `{drift_score:.4f}`\n**Fraud Rate:** `{fraud_rate*100:.2f}%`"
#             placeholder_status.markdown(status)

#             if fraud_rate < 0.05:
#                 st.warning("⚠️ Model performance dropped. Sending alert & retraining...")

#                 # 📨 Send email alert
#                 send_email_alert(drift_score, fraud_rate)

#                 # 🧠 Retrain
#                 df_combined = pd.concat([df_old, df_new], ignore_index=True)
#                 df_combined.to_csv(DATA_PATH, index=False)
#                 train_autoencoder()
#                 model, preprocessor, threshold = load_assets()
#                 st.success("✅ Model retrained and alert sent!")


#             df_hist = pd.DataFrame(history)
#             placeholder_chart.line_chart(df_hist.set_index("Step"))
#             time.sleep(2)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from faker import Faker
from tensorflow.keras.models import load_model
from train_autoencoder import train_autoencoder

import smtplib, ssl, os
from email.message import EmailMessage
from dotenv import load_dotenv

# Configs
fake = Faker()
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modules/fraud_autoencoder_model.h5"
PREPROCESSOR_PATH = BASE_DIR / "modules/fraud_preprocessor.pkl"
THRESHOLD_PATH = BASE_DIR / "modules/fraud_threshold.txt"
DATA_PATH = BASE_DIR / "non_fraud_transactions.csv"

st.set_page_config(page_title="Drift Monitoring Dashboard", layout="wide")
st.title("\U0001F4E1 Real-Time Drift & Model Monitoring")

placeholder_chart = st.empty()
placeholder_status = st.empty()

history = []
bad_streak = 0

@st.cache_data(show_spinner=False)
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())
    return model, preprocessor, threshold

model, preprocessor, threshold = load_assets()

def send_email_alert(drift_score, fraud_rate):
    load_dotenv()
    email_sender = os.getenv("EMAIL_USER")
    email_password = os.getenv("EMAIL_PASS")
    email_receiver = os.getenv("TO_EMAIL")

    subject = "\U0001F6A8 Model Drift Alert!"
    body = (
        f"Drift MSE: {drift_score:.4f}\n"
        f"Detected Fraud Rate: {fraud_rate:.2%}\n\n"
        "Model performance has degraded. Consider retraining."
    )

    msg = EmailMessage()
    msg["From"] = email_sender
    msg["To"] = email_receiver
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.send_message(msg)

    print("\U0001F4E7 Email alert sent.")

def generate_anomalous_data(existing_df, num_rows=300, drift_intensity=0.0):
    anomalies = []
    customer_ids = existing_df["customer_id"].unique().tolist()

    for _ in range(num_rows):
        row = {
            "transaction_id": fake.uuid4(),
            "timestamp": fake.date_time_this_year().isoformat(),
            "amount": np.random.normal(5000 + drift_intensity * 10000, 2000 + drift_intensity * 1000),
            "transaction_type": fake.random_element(["card", "upi", "wallet", "pos"]),
            "merchant": fake.company(),
            "location": fake.country(),
            "is_foreign": fake.boolean(chance_of_getting_true=10 + int(drift_intensity * 40)),
            "is_high_risk_country": fake.boolean(chance_of_getting_true=5 + int(drift_intensity * 40)),
            "opening_balance": np.random.normal(40000 + drift_intensity * 20000, 10000),
            "closing_balance": np.random.normal(30000 - drift_intensity * 15000, 8000),
            "account_id": fake.uuid4(),
            "account_type_txn": fake.random_element(["savings", "current"]),
            "account_type_acct": fake.random_element(["savings", "current"]),
            "account_number": fake.random_number(digits=10),
            "balance": np.random.normal(20000 - drift_intensity * 10000, 5000),
            "created_at_acct": fake.date_this_decade(),
            "customer_id": fake.random_element(customer_ids),
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.address(),
            "dob": fake.date_of_birth().isoformat(),
            "created_at_cust": fake.date_this_decade(),
            "past_txn_count": np.random.randint(5, 20),
            "past_avg_amount": np.random.normal(4500 + drift_intensity * 3000, 1500),
            "past_common_merchant": fake.company(),
            "past_common_location": fake.country(),
            "agg_txn_count": np.random.randint(5, 15),
            "agg_avg_amount": np.random.normal(4800 + drift_intensity * 4000, 1600),
            "agg_std_amount": np.random.normal(1200 + drift_intensity * 2000, 400),
            "agg_max_amount": np.random.normal(10000 + drift_intensity * 30000, 3000),
            "agg_unique_merchants": np.random.randint(1, 3 + int(drift_intensity * 3)),
            "agg_unique_locations": np.random.randint(1, 3 + int(drift_intensity * 3)),
        }
        anomalies.append(row)

    return pd.DataFrame(anomalies)

# Sidebar Controls
st.sidebar.title("⚙️ Controls")
run_monitoring = st.sidebar.toggle("Run Drift Monitoring", value=False)
drift_override = st.sidebar.slider("Drift Intensity (Override)", 0.0, 1.0, step=0.05, value=0.0)
df_old = pd.read_csv(DATA_PATH)

drift_intensity = drift_override
if run_monitoring:
    with st.spinner("Monitoring data drift..."):
        for i in range(100):
            drift_intensity = min(1.0, drift_intensity + 0.01)
            df_new = generate_anomalous_data(df_old, drift_intensity=drift_intensity)

            X = df_new.copy()
            X = X.fillna("missing")
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = X[col].astype(str)

            X_processed = preprocessor.transform(X)
            recon = model.predict(X_processed)
            mse = np.mean(np.square(X_processed - recon), axis=1)
            predicted_fraud = (mse > threshold).astype(int)

            drift_score = mse.mean()
            fraud_rate = predicted_fraud.mean()
            history.append({"Step": i, "Drift MSE": drift_score, "Detected Fraud Rate": fraud_rate})

            status = f"\n**Drift MSE:** `{drift_score:.4f}`\n**Fraud Rate:** `{fraud_rate*100:.2f}%`"
            placeholder_status.markdown(status)

            if fraud_rate < 0.05:
                bad_streak += 1
            else:
                bad_streak = 0

            if bad_streak >= 3 or drift_score > 0.02:
                st.warning("⚠️ Model performance dropped. Alerting & Retraining...")
                send_email_alert(drift_score, fraud_rate)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.to_csv(DATA_PATH, index=False)
                train_autoencoder()
                model, preprocessor, threshold = load_assets()
                bad_streak = 0
                st.success("✅ Model retrained and alert sent!")

            df_hist = pd.DataFrame(history)
            placeholder_chart.line_chart(df_hist.set_index("Step"))
            time.sleep(2)
