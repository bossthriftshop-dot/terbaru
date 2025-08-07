import json
import pandas as pd
from collections import Counter

def analyze_trade_feedback(filepath="trade_feedback.json"):
    """
    Menganalisis file trade_feedback.json untuk memberikan wawasan tentang kinerja strategi.

    Karena data saat ini memiliki keterbatasan (tidak semua trade memiliki hasil 'win'/'loss'
    dan data 'score_components' hilang), analisis ini akan fokus pada apa yang tersedia
    dan memberikan rekomendasi untuk perbaikan pencatatan data.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"❌ Error: File '{filepath}' tidak ditemukan atau formatnya tidak valid.")
        return

    if not data:
        print("🟡 Info: File feedback kosong, tidak ada yang bisa dianalisis.")
        return

    df = pd.DataFrame(data)

    print("--- ANALISIS KINERJA TRADING (Berdasarkan Data yang Tersedia) ---")
    print("\n1. Ringkasan Umum")
    print("="*20)
    print(f"Total Sinyal Tercatat: {len(df)}")

    # --- Analisis Win Rate (dengan data yang ada) ---
    # Filter hanya untuk trade yang memiliki hasil 'win' atau 'loss' yang jelas
    result_df = df[df['result'].isin(['win', 'loss'])].copy()

    if not result_df.empty:
        total_trades_with_result = len(result_df)
        wins = len(result_df[result_df['result'] == 'win'])
        losses = len(result_df[result_df['result'] == 'loss'])
        win_rate = (wins / total_trades_with_result) * 100 if total_trades_with_result > 0 else 0

        print(f"Trade dengan Hasil (Win/Loss): {total_trades_with_result}")
        print(f"  - Menang (Win): {wins}")
        print(f"  - Kalah (Loss): {losses}")
        print(f"  - Win Rate Terhitung: {win_rate:.2f}%")

        if losses == 0 and wins > 0:
            print("\n⚠️  PERINGATAN: Tidak ada trade yang tercatat sebagai 'loss'.")
            print("   Win rate yang ditampilkan kemungkinan besar tidak akurat. Proses pencatatan hasil")
            print("   trading yang kalah (loss) tampaknya tidak berjalan dengan benar.")

    else:
        print("Tidak ditemukan trade dengan hasil 'win' atau 'loss' yang jelas.")
        print("Analisis win rate tidak dapat dilakukan.")

    # --- Analisis Komponen Sinyal (jika ada) ---
    print("\n2. Analisis Komponen Sinyal (Penyebab SL)")
    print("="*20)

    losing_trades = result_df[result_df['result'] == 'loss']
    if not losing_trades.empty and 'score_components' in losing_trades.columns:
        # Filter baris di mana 'score_components' adalah list dan tidak kosong
        losing_trades = losing_trades[losing_trades['score_components'].apply(lambda x: isinstance(x, list) and x)]

        if not losing_trades.empty:
            all_loss_components = [comp for sublist in losing_trades['score_components'] for comp in sublist]
            component_counts = Counter(all_loss_components)

            print("Komponen sinyal yang paling sering muncul pada trade yang KALAH:")
            # Urutkan berdasarkan jumlah kemunculan, dari yang terbanyak
            for component, count in component_counts.most_common():
                print(f"  - {component}: {count} kali")
        else:
            print("Tidak ada data 'score_components' yang valid pada trade yang kalah.")
            print("Ini adalah data krusial untuk menganalisis penyebab stoploss.")
    else:
        print("Tidak ada data 'loss' atau kolom 'score_components' tidak ada.")
        print("Analisis komponen penyebab stoploss tidak dapat dilakukan.")


    # --- Rekomendasi ---
    print("\n3. REKOMENDASI UNTUK PENGEMBANGAN")
    print("="*20)
    print("Untuk dapat menganalisis penyebab Stoploss secara akurat, langkah-langkah berikut sangat penting:")
    print("1. **Perbaiki Pencatatan Hasil Trade**: Pastikan setiap trade yang ditutup (baik SL maupun TP)")
    print("   memanggil endpoint `/api/feedback_trade` dengan hasil yang benar ('win' atau 'loss').")
    print("2. **Sertakan Konteks Sinyal**: Saat sinyal dikirim ke server, pastikan seluruh konteksnya,")
    print("   terutama `profile_name` dan `score_components`, ikut disimpan.")
    print("3. **Jalankan Ulang Sistem**: Setelah pencatatan diperbaiki, kumpulkan data trading baru.")
    print("\nDengan data yang lengkap, kita bisa menjawab pertanyaan seperti:")
    print(" - Sinyal mana ('BULLISH_BOS', 'FVG_BEARISH', dll.) yang paling sering menyebabkan kerugian?")
    print(" - Apakah profil 'scalping' lebih berisiko daripada 'intraday'?")
    print(" - Apakah sinyal dengan skor (confidence) rendah lebih sering rugi?")
    print("\nAnalisis ini akan menjadi dasar untuk menyesuaikan bobot (`weights`) dan ambang batas (`confidence_threshold`)")
    print("agar strategi menjadi lebih profitabel.")


if __name__ == '__main__':
    analyze_trade_feedback()
