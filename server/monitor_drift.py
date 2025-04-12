import subprocess
import time
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv
import os
import signal

# Constants
BASE_DIR = Path(__file__).resolve().parent
FRAUD_OUTPUT = BASE_DIR / "modules/fraud_cases_for_llm.csv"
RETRAIN_THRESHOLD = 0.30
CHECK_INTERVAL_SECONDS = 300
CONSUMER_LOG = BASE_DIR / "logs" / "consumer.log"

# Load .env
load_dotenv(BASE_DIR / ".env")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Ensure log folder exists
os.makedirs(BASE_DIR / "logs", exist_ok=True)

# Track background process for cleanup
consumer_process = None

def send_alert_email(fraud_ratio):
    msg = MIMEText(f"""
⚠️ ALERT: {fraud_ratio*100:.2f}% of the transactions were flagged as fraud by the model.
This may indicate data drift or model degradation.

🧠 The model will be retrained immediately.
""")
    msg["Subject"] = f"🚨 Model Alert: {fraud_ratio*100:.1f}% Fraud Detected"
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"📧 Alert sent to {TO_EMAIL}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def run_pipeline_step(label, command):
    print(f"\n🔄 {label}...")
    subprocess.run(command, shell=True, check=True)

def start_consumer_background():
    global consumer_process
    log_file = open(CONSUMER_LOG, "a")
    print("🧪 Starting Kafka consumer in background...")
    consumer_process = subprocess.Popen(
        ["python3", "modules/consumer.py"],
        stdout=log_file,
        stderr=log_file
    )

def stop_consumer_background():
    global consumer_process
    if consumer_process and consumer_process.poll() is None:
        print("🛑 Stopping previous Kafka consumer...")
        consumer_process.send_signal(signal.SIGTERM)
        consumer_process.wait()

def run_monitoring_cycle():
    print("\n⏱️ Starting new pipeline run...\n")

    try:
        # Step 0: Reset environment
        stop_consumer_background()

        # Step 1: Start fresh data stream
        run_pipeline_step("Generating data", "python3 modules/datagen.py")
        start_consumer_background()

        # Step 2: Process and predict
        run_pipeline_step("Merging parquet files", "python3 modules/combine.py")
        run_pipeline_step("Enriching with historical features", "python3 modules/transformation.py")
        run_pipeline_step("Applying rule-based fraud detection", "python3 modules/rule_based_fraud_detection.py denormalized_transactions/denoised_enriched_transactions.csv")
        run_pipeline_step("Generating account-level history", "python3 modules/history.py denormalized_transactions")
        run_pipeline_step("Running autoencoder fraud detection", "python3 modules/model.py")

        # Step 3: Evaluate model
        if not FRAUD_OUTPUT.exists():
            print("⚠️ No fraud prediction output found.")
            return

        df = pd.read_csv(FRAUD_OUTPUT)
        if df.empty:
            print("✅ No frauds detected. Model is OK.")
            return

        total_records = pd.read_csv(BASE_DIR / "modules/non_fraud_transactions.csv").shape[0]
        fraud_ratio = len(df) / total_records

        print(f"📈 Total evaluated: {total_records}")
        print(f"🚨 Predicted frauds: {len(df)}")
        print(f"📊 Fraud prediction rate: {fraud_ratio:.2%}")

        if fraud_ratio > RETRAIN_THRESHOLD:
            print("🚨 High fraud detection ratio! Sending alert and retraining model...")
            send_alert_email(fraud_ratio)
            run_pipeline_step("Retraining autoencoder model", "python3 modules/train_autoencoder.py")

    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline step failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    while True:
        start_time = time.time()
        run_monitoring_cycle()
        elapsed = time.time() - start_time
        wait_time = max(0, CHECK_INTERVAL_SECONDS - elapsed)
        print(f"\n🕒 Waiting {wait_time:.1f} seconds before next run...\n")
        time.sleep(wait_time)

if __name__ == "__main__":
    main()
