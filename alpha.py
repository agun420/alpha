# filename: run_scanner.py

# ==============================================================================
# Refined Short-Term Stock Scanner (1-10 Day Holding Period)
#
# Key Improvements:
# 1.  **Generalized AI Model:** Instead of training a model for each stock
#     on-the-fly, this script trains a single, more robust model on a large
#     dataset of multiple stocks over a longer time period (5 years).
# 2.  **Model Persistence:** The trained AI model is saved to a file. For daily
#     scanning, the script loads the pre-trained model, which is significantly
#     faster and more consistent.
# 3.  **Enhanced Features:** More technical indicators have been added to provide
#     the AI model with a richer dataset for making predictions.
# 4.  **Structured Workflow:** The script is now organized into two main parts:
#     a one-time training process and the daily scanning process.
# 5.  **Telegram Notifications:** Added a feature to send the top trade ideas
#     directly to a Telegram chat.
# ==============================================================================

# ==============================================================================
# IMPORTS
# ==============================================================================
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

# ==============================================================================
# DATA ACQUISITION AND PREPARATION
# ==============================================================================

class DataManager:
    """Handles data acquisition, cleaning, and preprocessing."""

    def __init__(self):
        # Scaler for feature normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_stock_data(self, symbols: list, period: str = "6mo") -> dict:
        """
        Fetch and clean stock data for multiple symbols.
        """
        stock_data = {}
        print(f"Fetching data for {len(symbols)} symbols for period '{period}'...")
        for i, symbol in enumerate(symbols):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)

                if not data.empty:
                    data = self._clean_data(data)
                    stock_data[symbol] = data
                    print(f"({i+1}/{len(symbols)}) ✓ {symbol}: {len(data)} records")
                else:
                    print(f"({i+1}/{len(symbols)}) ✗ {symbol}: No data available")
            except Exception as e:
                print(f"({i+1}/{len(symbols)}) ✗ {symbol}: Error - {str(e)}")
            time.sleep(0.1)  # Delay to avoid rate-limiting
        return stock_data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and ensures correct data types for a stock DataFrame.
        """
        data = data.dropna()
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data.dropna()

# ==============================================================================
# TECHNICAL INDICATORS AND STRATEGY IMPLEMENTATION
# ==============================================================================

class TechnicalAnalyzer:
    """Implements technical indicators and trading strategies."""

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a comprehensive set of technical indicators.
        """
        df = data.copy()
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        # ADX
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        # Volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        # Additional Momentum Indicators
        df['WilliamsR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)

        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicator rules.
        """
        df = data.copy()
        df['momentum_signal'] = 0
        df['reversal_signal'] = 0
        df['breakout_signal'] = 0
        
        df.loc[(df['EMA_12'] > df['EMA_26']) & (df['RSI'] > 55) & (df['MACD'] > df['MACD_signal']) & (df['ADX'] > 25), 'momentum_signal'] = 1
        df.loc[(df['RSI'] < 30) & (df['Close'] < df['BB_lower']), 'reversal_signal'] = 1
        df['volume_avg_20'] = df['Volume'].rolling(window=20).mean()
        df.loc[(df['Close'] > df['BB_upper']) & (df['Volume'] > df['volume_avg_20'] * 1.5), 'breakout_signal'] = 1

        df['composite_signal'] = (df['momentum_signal'] * 0.4) + (df['reversal_signal'] * 0.3) + (df['breakout_signal'] * 0.3)
        return df

    def get_catalyst(self, latest_data: pd.Series) -> str:
        """Identifies the primary technical catalyst for a signal."""
        signals = {
            'Momentum': latest_data.get('momentum_signal', 0),
            'Reversal': latest_data.get('reversal_signal', 0),
            'Breakout': latest_data.get('breakout_signal', 0),
        }
        strongest_signal = max(signals, key=lambda k: abs(signals[k]))
        return "None" if signals[strongest_signal] == 0 else strongest_signal

# ==============================================================================
# AI/ML MODEL MANAGEMENT
# ==============================================================================
class AIManager:
    """Handles the training, saving, loading, and prediction of a generalized LSTM model."""

    def __init__(self, sequence_length: int = 60, model_path: str = 'generalized_stock_model.h5'):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'ADX', 'WilliamsR', 'CMF']

    def _prepare_lstm_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> tuple:
        """Prepares sequences of data for LSTM training or prediction."""
        target_col = 'Close'
        available_features = [f for f in self.features if f in data.columns]
        if len(available_features) != len(self.features):
            raise ValueError(f"Missing features. Required: {self.features}, Found: {available_features}")

        feature_data = data[available_features].values
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(feature_data)
        else:
            scaled_data = self.scaler.transform(feature_data)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(1 if data[target_col].iloc[i] > data[target_col].iloc[i-1] else 0)
        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple):
        """Builds the LSTM model architecture."""
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(units=100, return_sequences=False),
            Dropout(0.3),
            Dense(units=50, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_generalized_model(self, all_stocks_data: dict, epochs: int = 20):
        """Trains a single model on data from multiple stocks."""
        print("Preparing data for generalized model training...")
        all_X, all_y = [], []
        for symbol, data in all_stocks_data.items():
            try:
                fit_scaler_on_this_data = len(all_X) == 0
                X, y = self._prepare_lstm_data(data, fit_scaler=fit_scaler_on_this_data)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                print(f"Processed {symbol} for training.")
            except Exception as e:
                print(f"Could not process {symbol} for training: {e}")

        if not all_X:
            print("No data available for training. Exiting.")
            return

        X_combined, y_combined = np.concatenate(all_X), np.concatenate(all_y)
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

        print(f"\nTraining generalized model on {len(X_train)} samples, testing on {len(X_test)} samples.")
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_accuracy', save_best_only=True)
        ]
        
        self.model.fit(
            X_train, y_train, batch_size=64, epochs=epochs,
            validation_data=(X_test, y_test), callbacks=callbacks, verbose=1
        )
        print(f"\nTraining complete. Best model saved to {self.model_path}")

    def load_trained_model(self):
        """Loads the pre-trained model from disk."""
        if os.path.exists(self.model_path):
            print(f"Loading pre-trained model from {self.model_path}...")
            self.model = load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")

    def predict_direction(self, data: pd.DataFrame) -> float:
        """Predicts the price direction for the next day."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        feature_data = data[self.features].values
        scaled_data = self.scaler.transform(feature_data)
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features))
        prediction = self.model.predict(last_sequence, verbose=0)
        return prediction[0][0]

# ==============================================================================
# RISK AND NOTIFICATION
# ==============================================================================

class RiskManager:
    """Handles risk calculations."""
    def calculate_atr_stop_loss(self, close_price: float, atr: float, multiplier: float = 2.0) -> float:
        return close_price - (atr * multiplier)

    def calculate_target_price(self, entry_price: float, stop_loss_price: float, risk_reward_ratio: float = 1.5) -> float:
        risk_per_share = entry_price - stop_loss_price
        return entry_price + (risk_per_share * risk_reward_ratio) if risk_per_share > 0 else entry_price

class TelegramNotifier:
    """Handles sending notifications to a Telegram bot."""
    def __init__(self, token: str, chat_id: str):
        if not token or not chat_id:
            raise ValueError("Telegram Bot Token and Chat ID must be provided.")
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def format_trade_ideas(self, trade_ideas_df: pd.DataFrame) -> str:
        """Formats the trade ideas DataFrame into a Markdown string for Telegram."""
        if trade_ideas_df.empty:
            return "No new trade ideas found today."
        
        message = f"*📈 Top {len(trade_ideas_df)} Short-Term Trade Ideas*\n"
        message += f"_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        for _, row in trade_ideas_df.iterrows():
            signal_emoji = "🟢" if row['Signal'] == 'Buy' else "🔴"
            message += f"*{signal_emoji} {row['Symbol']} ({row['Signal']})*\n"
            message += f"  - *Catalyst:* {row['Catalyst']}\n"
            message += f"  - *Entry:* ${row['Close']:.2f}\n"
            message += f"  - *Target:* ${row['Target Price']:.2f}\n"
            message += f"  - *Stop Loss:* ${row['Stop Loss']:.2f}\n"
            message += f"  - *Score:* {row['Final Score']:.2f}\n\n"
        return message

    def send_notification(self, message: str):
        """Sends a message to the configured Telegram chat."""
        params = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'Markdown'}
        try:
            response = requests.post(self.base_url, data=params)
            response.raise_for_status()
            print("✓ Telegram notification sent successfully.")
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to send Telegram notification: {e}")

# ==============================================================================
# STOCK SCANNER AND FILTERING
# ==============================================================================

class StockScanner:
    """Main scanner class that combines all components."""
    def __init__(self, ai_manager: AIManager):
        self.data_manager = DataManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_manager = ai_manager
        self.risk_manager = RiskManager()

    def scan_stocks(self, symbols: list) -> pd.DataFrame:
        """Scans stocks using technical signals and the pre-trained AI model."""
        scan_results = []
        stock_data_dict = self.data_manager.get_stock_data(symbols, period="1y")

        for symbol, data in stock_data_dict.items():
            print(f"\nAnalyzing {symbol}...")
            if len(data) < self.ai_manager.sequence_length + 30:
                print(f"✗ {symbol}: Insufficient data.")
                continue

            data_with_indicators = self.technical_analyzer.calculate_indicators(data)
            data_with_signals = self.technical_analyzer.generate_signals(data_with_indicators)
            if data_with_signals.empty:
                continue

            latest_data = data_with_signals.iloc[-1]
            composite_signal = latest_data['composite_signal']
            
            try:
                lstm_confidence = self.ai_manager.predict_direction(data_with_signals)
                print(f"✓ {symbol}: AI Confidence (Up): {lstm_confidence:.2f}")
            except Exception as e:
                print(f"✗ {symbol}: AI prediction failed - {e}")
                lstm_confidence = 0.5 # Neutral

            lstm_score = (lstm_confidence - 0.5) * 2
            final_score = (composite_signal * 0.6) + (lstm_score * 0.4)

            if abs(final_score) > 0.25:
                stop_loss = self.risk_manager.calculate_atr_stop_loss(latest_data['Close'], latest_data['ATR'])
                target_price = self.risk_manager.calculate_target_price(latest_data['Close'], stop_loss)
                scan_results.append({
                    'Symbol': symbol, 'Close': latest_data['Close'],
                    'Signal': 'Buy' if final_score > 0 else 'Sell',
                    'Catalyst': self.technical_analyzer.get_catalyst(latest_data), 
                    'Final Score': final_score, 'Target Price': target_price, 
                    'Stop Loss': stop_loss,
                })
                print(f"✓ {symbol}: Potential trade idea found.")
        
        if not scan_results:
            return pd.DataFrame()
        return pd.DataFrame(scan_results).sort_values(by='Final Score', ascending=False)

# ==============================================================================
# UTILITY AND WORKFLOWS
# ==============================================================================

def get_sp500_symbols():
    """Fetches the list of S&P 500 symbols from Wikipedia."""
    print("Fetching S&P 500 symbols...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        symbols = [s.replace('.', '-') for s in table[0]['Symbol'].tolist()]
        print(f"✓ Found {len(symbols)} S&P 500 symbols.")
        return symbols
    except Exception as e:
        print(f"✗ Could not fetch S&P 500 symbols: {e}. Falling back to a default list.")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ']

def train_model_workflow():
    """WORKFLOW 1: ONE-TIME MODEL TRAINING"""
    print("="*60 + "\nStarting AI Model Training Workflow\n" + "="*60)
    data_manager = DataManager()
    technical_analyzer = TechnicalAnalyzer()
    ai_manager = AIManager()
    
    training_symbols = get_sp500_symbols()[:100] # Use first 100 for a faster example
    stock_data_dict = data_manager.get_stock_data(training_symbols, period="5y")
    
    data_with_indicators = {
        symbol: technical_analyzer.calculate_indicators(data)
        for symbol, data in stock_data_dict.items() if len(data) > 100
    }
    
    ai_manager.train_generalized_model(data_with_indicators, epochs=20)
    print("\nModel training complete.")

def scanning_workflow():
    """WORKFLOW 2: DAILY STOCK SCANNING"""
    print("="*60 + f"\nStarting Daily Stock Scanning Workflow\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*60)
    
    ai_manager = AIManager()
    try:
        ai_manager.load_trained_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the 'train' workflow first to create the model file.")
        return

    scanner = StockScanner(ai_manager)
    stock_list = get_sp500_symbols()
    trade_ideas = scanner.scan_stocks(stock_list)

    print("\n" + "="*60 + "\nSCAN COMPLETE\n" + "="*60)
    if not trade_ideas.empty:
        top_5_ideas = trade_ideas.head(5)
        print("\nTop 5 Trade Ideas:\n" + top_5_ideas.to_string())
        
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
    # Default to 'scan' if no argument is provided
    action = sys.argv[1] if len(sys.argv) > 1 else 'scan'

    if action == 'train':
        train_model_workflow()
    elif action == 'scan':
        scanning_workflow()
    else:
        print(f"Error: Invalid action '{action}'. Use 'train' or 'scan'.")
        sys.exit(1)
