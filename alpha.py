# filename: weekly_retrain_scan.py

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
import datetime
import time
import requests
import os
import sys

warnings.filterwarnings('ignore')

# (All class definitions like DataManager, TechnicalAnalyzer, AIManager, etc., remain the same as the previous script)
# ...
# NOTE: For brevity, the class definitions are omitted here, but they should be included
# in your actual `weekly_retrain_scan.py` file exactly as they were in the previous script.
# Please copy all the class definitions from the script in the previous response into this file.
# ...

# ==============================================================================
# COMBINED WORKFLOW FOR WEEKLY RETRAINING AND SCANNING
# ==============================================================================

def weekly_retrain_and_scan_workflow():
    """
    WORKFLOW: WEEKLY MODEL RETRAINING & SCANNING
    This function retrains the model on the latest data and then runs a scan.
    """
    print("="*60)
    print("Starting Weekly Retraining and Scanning Workflow")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # --- Part 1: Retrain the Model ---
    print("\n--- STAGE 1: MODEL TRAINING ---\n")
    data_manager = DataManager()
    technical_analyzer = TechnicalAnalyzer()
    ai_manager = AIManager() # Initializes a new model manager

    # Get a diverse list of stocks for training
    training_symbols = get_sp500_symbols()[:100] # Using first 100 S&P 500 stocks

    # Fetch a long period of historical data for robust training
    stock_data_dict = data_manager.get_stock_data(training_symbols, period="5y")

    # Calculate technical indicators for all stocks
    data_with_indicators = {
        symbol: technical_analyzer.calculate_indicators(data)
        for symbol, data in stock_data_dict.items() if len(data) > 100
    }

    # Train and save the generalized model
    ai_manager.train_generalized_model(data_with_indicators, epochs=20)
    print("\nModel training complete. The model has been updated and saved.")

    # --- Part 2: Scan with the Newly Trained Model ---
    print("\n--- STAGE 2: STOCK SCANNING ---\n")

    # The AIManager instance now holds the newly trained model, so we can proceed to scan.
    # No need to load it from disk, as it's already in memory.
    scanner = StockScanner(ai_manager)

    # Define the list of stocks to scan (e.g., all S&P 500)
    stock_list_to_scan = get_sp500_symbols()
    trade_ideas = scanner.scan_stocks(stock_list_to_scan)

    print("\n" + "="*60 + "\nSCAN COMPLETE\n" + "="*60)

    if not trade_ideas.empty:
        top_5_ideas = trade_ideas.head(5)
        print("\nTop 5 Trade Ideas:\n" + top_5_ideas.to_string())

        # --- Part 3: Send Telegram Notification ---
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            print("\n" + "="*50 + "\nSending results to Telegram...\n" + "="*50)
            try:
                notifier = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID)
                message = notifier.format_trade_ideas(top_5_ideas)
                notifier.send_notification(message)
            except Exception as e:
                print(f"✗ An unexpected error occurred during notification: {e}")
        else:
            print("\n" + "="*50 + "\nTelegram credentials not set. Skipping notification.\n" + "="*50)
    else:
        print("\nNo trade ideas met the criteria today.")


if __name__ == '__main__':
    # This script now has a single, primary purpose.
    weekly_retrain_and_scan_workflow()
