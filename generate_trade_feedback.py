import json
import requests
import random
import time

def generate_feedback():
    """
    Reads the existing trade feedback file, simulates trade outcomes (win/loss),
    and sends the feedback to the running Flask server.
    """
    try:
        with open("trade_feedback.json", 'r') as f:
            trades = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("❌ 'trade_feedback.json' tidak ditemukan atau kosong. Jalankan bot utama untuk menghasilkan sinyal terlebih dahulu.")
        return

    server_url = "http://127.0.0.1:5000/api/feedback_trade"

    # Filter for trades that need feedback (i.e., those without a final 'result' like 'win' or 'loss')
    pending_feedback_trades = [
        trade for trade in trades
        if trade.get('result') not in ['win', 'loss'] and 'signal_id' in trade
    ]

    if not pending_feedback_trades:
        print("✅ Tidak ada trade yang memerlukan feedback. Semua sudah memiliki hasil (win/loss).")
        return

    print(f"🔍 Ditemukan {len(pending_feedback_trades)} trade yang memerlukan feedback.")

    for trade in pending_feedback_trades:
        signal_id = trade.get('signal_id')

        # Simulate a trade outcome
        simulated_result = random.choice(['win', 'loss'])

        feedback_payload = {
            "signal_id": signal_id,
            "result": simulated_result,
            "pnl": random.uniform(5.0, 100.0) if simulated_result == 'win' else random.uniform(-50.0, -5.0)
        }

        print(f"  -> Mengirim feedback untuk signal_id {signal_id[:8]}... : Hasil = {simulated_result.upper()}")

        try:
            response = requests.post(server_url, json=feedback_payload, timeout=10)
            if response.status_code == 200:
                print(f"     ✅ Berhasil dikirim!")
            else:
                print(f"     ❌ Gagal mengirim. Status: {response.status_code}, Pesan: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"     ❌ Error koneksi ke server: {e}")
            print("     Pastikan server Flask (app_rev4.py) sedang berjalan.")
            return # Stop if server is not running

        time.sleep(0.1) # Small delay between requests

    print("\n✅ Proses pembuatan feedback simulasi selesai.")
    print("File 'trade_feedback.json' sekarang seharusnya telah diperbarui dengan hasil 'win'/'loss'.")
    print("Anda sekarang dapat menjalankan 'analyze_performance.py' untuk menganalisis hasilnya.")

if __name__ == '__main__':
    generate_feedback()
