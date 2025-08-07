# train_xgboost.py
#
# Deskripsi:
# Skrip ini melatih model XGBoost untuk setiap simbol berdasarkan data latih
# yang dihasilkan oleh generate_training_data.py.

import json
import logging
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRAINING_DATA_PATH = 'trade_feedback.json'
MODEL_OUTPUT_PATH = 'xgboost_model_{symbol}.json'

def load_training_data(filepath):
    """Memuat data latih dari file JSON."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error saat memuat data latih: {e}")
        return []

def train_model_for_symbol(symbol, trades):
    """Melatih model XGBoost untuk simbol tertentu."""
    if not trades:
        logging.warning(f"Tidak ada data untuk simbol {symbol}. Melewati pelatihan.")
        return None

    # Siapkan data
    X = np.array([trade['gng_input_features_on_signal'] for trade in trades])
    y = np.array([trade['result'] for trade in trades])
    
    # Validasi jumlah data
    if len(X) < 10:
        logging.warning(f"Data untuk {symbol} terlalu sedikit ({len(X)}). Minimal 10 trade diperlukan.")
        return None

    # Split data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih model XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', verbose=False)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model untuk {symbol} - Akurasi pada data test: {accuracy:.2f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Simpan model
    model.save_model(MODEL_OUTPUT_PATH.format(symbol=symbol))
    logging.info(f"Model untuk {symbol} disimpan di {MODEL_OUTPUT_PATH.format(symbol=symbol)}")
    
    return model

def main():
    """Fungsi utama untuk melatih model XGBoost untuk semua simbol."""
    training_data = load_training_data(TRAINING_DATA_PATH)
    if not training_data:
        logging.error("Tidak ada data latih yang valid. Keluar.")
        return

    # Pisahkan data per simbol
    trades_by_symbol = {}
    for trade in training_data:
        symbol = trade['symbol']
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)
    
    # Latih model untuk setiap simbol
    for symbol, trades in trades_by_symbol.items():
        logging.info(f"Melatih model untuk simbol {symbol} ({len(trades)} trade)...")
        train_model_for_symbol(symbol, trades)

    logging.info("Pelatihan model selesai untuk semua simbol.")

if __name__ == '__main__':
    main()