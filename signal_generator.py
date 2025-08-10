# signal_generator.py (VERSI FINAL YANG SUDAH DIPERBAIKI)

from __future__ import annotations

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import MetaTrader5 as mt5

from data_fetching import get_candlestick_data
from technical_indicators import (
    detect_structure,
    detect_order_blocks_multi,
    detect_fvg_multi,
    detect_eqh_eql,
    detect_liquidity_sweep,
    detect_liquidity_grab,
    calculate_optimal_trade_entry,
    detect_engulfing,
    detect_pinbar,
    detect_continuation_patterns,
    detect_change_of_character,
)
from gng_model import (
    get_gng_input_features_full,
    get_gng_context,
)

def get_open_positions_per_tf(symbol: str, tf: str, mt5_path: str) -> int:
    if not mt5.initialize(path=mt5_path): return 99
    positions = mt5.positions_get(symbol=symbol)
    mt5.shutdown()
    return len(positions) if positions is not None else 0

def get_active_orders(symbol: str, mt5_path: str) -> List[float]:
    if not mt5.initialize(path=mt5_path): return []
    active_prices: List[float] = []
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions: active_prices.append(pos.price_open)
        orders = mt5.orders_get(symbol=symbol)
        if orders:
            for order in orders: active_prices.append(order.price_open)
    except Exception as e:
        logging.error(f"Error saat mengambil order/posisi aktif: {e}")
    finally:
        mt5.shutdown()
    return active_prices

def is_far_enough(entry_price: float, existing_prices: List[float], point_value: float, min_distance_pips: float) -> bool:
    min_distance_points = min_distance_pips * point_value
    for price in existing_prices:
        if abs(entry_price - price) < min_distance_points:
            logging.warning(f"Sinyal DITOLAK: Entry {entry_price:.3f} terlalu dekat dengan order aktif di {price:.3f}.")
            return False
    return True

def build_signal_format(symbol: str, entry_price: float, direction: str, sl: float, tp1: float, tp2: float, tp3: float, order_type: str) -> dict:
    signal = {"Symbol": symbol}
    # Disesuaikan dengan case-sensitivity dari MQL dan tambahkan kunci yang hilang
    order_keys = [
        "BuyEntry", "BuySL", "BuyTP", "BuyTP2", "BuyTP3", "SellEntry", "SellSL", "SellTP", "SellTP2", "SellTP3",
        "BuyStop", "BuyStopSL", "BuyStopTP", "BuyStopTP2", "BuyStopTP3", "SellStop", "SellStopSL", "SellStopTP", "SellStopTP2", "SellStopTP3",
        "Buylimit", "BuylimitSL", "BuylimitTP", "BuylimitTP2", "BuylimitTP3", "Selllimit", "SelllimitSL", "SellLimitTP", "SellLimitTP2", "SellLimitTP3",
        "DeleteLimit/Stop"
    ]
    for key in order_keys:
        signal[key] = ""
    order_type_upper = order_type.upper()
    if order_type_upper == 'BUY':
        signal.update({"BuyEntry": str(entry_price), "BuySL": str(sl), "BuyTP": str(tp1), "BuyTP2": str(tp2), "BuyTP3": str(tp3)})
    elif order_type_upper == 'SELL':
        signal.update({"SellEntry": str(entry_price), "SellSL": str(sl), "SellTP": str(tp1), "SellTP2": str(tp2), "SellTP3": str(tp3)})
    elif order_type_upper == 'BUY_LIMIT':
        signal.update({"Buylimit": str(entry_price), "BuylimitSL": str(sl), "BuylimitTP": str(tp1), "BuylimitTP2": str(tp2), "BuylimitTP3": str(tp3)})
    elif order_type_upper == 'SELL_LIMIT':
        signal.update({"Selllimit": str(entry_price), "SelllimitSL": str(sl), "SellLimitTP": str(tp1), "SellLimitTP2": str(tp2), "SellLimitTP3": str(tp3)})
    elif order_type_upper == 'BUY_STOP':
        signal.update({"BuyStop": str(entry_price), "BuyStopSL": str(sl), "BuyStopTP": str(tp1), "BuyStopTP2": str(tp2), "BuyStopTP3": str(tp3)})
    elif order_type_upper == 'SELL_STOP':
        signal.update({"SellStop": str(entry_price), "SellStopSL": str(sl), "SellStopTP": str(tp1), "SellStopTP2": str(tp2), "SellStopTP3": str(tp3)})
    return signal

def make_signal_id(signal_json: Dict[str, str]) -> str:
    return str(abs(hash(json.dumps(signal_json, sort_keys=True))))

def analyze_tf_opportunity(
    symbol: str,
    tf: str,
    mt5_path: str,
    gng_model,
    gng_feature_stats: Dict[str, Dict[str, Any]],
    confidence_threshold: float,
    min_distance_pips_per_tf: Dict[str, float],
    weights: Dict[str, float],
    htf_bias: str = 'NEUTRAL',
    htf_config: Dict[str, Any] = None
) -> Optional[Dict[str, Any]]:
    if htf_config is None:
        htf_config = {}
        
    df = get_candlestick_data(symbol, tf, 200, mt5_path)
    if df is None or len(df) < 50:
        logging.warning(f"Data TF {tf} tidak cukup untuk analisis.")
        return None
        
    # --- Analisis Komponen ---
    current_price = df['close'].iloc[-1]
    structure_str, swing_points = detect_structure(df)
    atr = df['high'].sub(df['low']).rolling(14).mean().iloc[-1]
    order_blocks = detect_order_blocks_multi(df, structure_filter=structure_str)
    fvg_zones = detect_fvg_multi(df)
    liquidity_sweep = detect_liquidity_sweep(df)
    liquidity_grabs = detect_liquidity_grab(df)
    choch = detect_change_of_character(df)
    patterns = detect_engulfing(df) + detect_pinbar(df) + detect_continuation_patterns(df)
    
    # --- Kalkulasi Skor ---
    score = 0.0
    score_components = {} # <-- DITAMBAHKAN DI SINI
    info_list: List[str] = []
    logging.info(f"[Arshy | {tf}] --- Memulai Analisis Konfluensi ---")
    
    # Skor Struktur
    structure_score = 0
    if "BULLISH_BOS" in structure_str: structure_score += weights.get("BULLISH_BOS", 3.0)
    if "BEARISH_BOS" in structure_str: structure_score += weights.get("BEARISH_BOS", -3.0)
    if "HH" in structure_str: structure_score += weights.get("HH", 1.0)
    if "LL" in structure_str: structure_score += weights.get("LL", -1.0)
    if "HL" in structure_str: structure_score += weights.get("HL", 1.0)
    if "LH" in structure_str: structure_score += weights.get("LH", -1.0)
    score_components['structure'] = structure_score
    score += structure_score
    logging.info(f"[Arshy | {tf}] Analisis Struktur: Teridentifikasi '{structure_str}' (Skor: {structure_score:+.2f})")

    if choch:
        choch_score = weights.get(choch['type'], 0.0)
        score_components['choch'] = choch_score
        score += choch_score
        logging.info(f"[Arshy | {tf}] Perubahan Karakter (CHoCH): {choch['type']} @ {choch['price']:.3f} terdeteksi (Skor: {choch_score:+.2f})")

    # Skor Zona (FVG, OB) & Event (LS)
    if fvg_zones:
        nearest_fvg = fvg_zones[0]
        fvg_score = (weights.get('FVG_BULLISH', 3.0) if 'BULLISH' in nearest_fvg['type'] else weights.get('FVG_BEARISH', -3.0)) * nearest_fvg['strength']
        score_components['fvg'] = fvg_score
        score += fvg_score
        logging.info(f"[Arshy | {tf}] Zona Inefisiensi (FVG): {nearest_fvg['type']} @ {nearest_fvg['start']:.3f}-{nearest_fvg['end']:.3f} terdeteksi (Skor: {fvg_score:+.2f})")
    if liquidity_sweep:
        ls_event = liquidity_sweep[-1]
        ls_score = weights.get(ls_event.get('type'))
        if ls_score:
            score_components['liquidity_sweep'] = ls_score
            score += ls_score
            logging.info(f"[Arshy | {tf}] Perburuan Likuiditas (Sweep): {ls_event.get('type')} @ {ls_event.get('swept_level'):.3f} terdeteksi (Skor: {ls_score:+.2f})")
    if liquidity_grabs:
        grab = liquidity_grabs[-1] # Ambil yang terbaru
        grab_score = weights.get(grab.get('type'), 0.0) * grab.get('strength', 1.0)
        if grab_score != 0:
            score_components['liquidity_grab'] = grab_score
            score += grab_score
            logging.info(f"[Arshy | {tf}] Perburuan Likuiditas (Grab): {grab.get('type')} pada level {grab.get('swept_level'):.4f} terdeteksi (Skor: {grab_score:+.2f})")
    if order_blocks:
        nearest_ob = order_blocks[0]
        ob_score = (weights.get('BULLISH_OB', 1.0) if 'BULLISH' in nearest_ob['type'] else weights.get('BEARISH_OB', -1.0)) * nearest_ob['strength']
        score_components['order_block'] = ob_score
        score += ob_score
        logging.info(f"[Arshy | {tf}] Zona Order Block: {nearest_ob['type']} @ {nearest_ob['low']:.3f}-{nearest_ob['high']:.3f} terdeteksi (Skor: {ob_score:+.2f})")

    # Skor Pola Minor
    pattern_score = sum(weights.get(p.get('type'), 0) for p in patterns)
    if pattern_score != 0:
        bullish_patterns = [p.get('type') for p in patterns if weights.get(p.get('type'), 0) > 0]
        bearish_patterns = [p.get('type') for p in patterns if weights.get(p.get('type'), 0) < 0]
        bull_summary = ", ".join([f"{count}x {name}" for name, count in Counter(bullish_patterns).items()])
        bear_summary = ", ".join([f"{count}x {name}" for name, count in Counter(bearish_patterns).items()])
        summary_parts = [s for s in [f"Bullish: [{bull_summary}]" if bull_summary else "", f"Bearish: [{bear_summary}]" if bear_summary else ""] if s]
        logging.info(f"[Arshy | {tf}] Konfluensi Pola Minor: {' | '.join(summary_parts)} (Skor Total: {pattern_score:+.2f})")
    score_components['patterns'] = pattern_score
    score += pattern_score
    logging.info(f"[Arshy | {tf}] Kalkulasi Skor Awal: {score:.2f}")

    # --- Penerapan HTF Bias ---
    if htf_config.get('enabled', False) and htf_bias != 'NEUTRAL':
        direction_pre_bias = "BUY" if score > 0 else "SELL"
        bias_score = htf_config.get('bias_influence_score', 2.5)
        penalty_score = htf_config.get('penalty_score', -5.0)
        if (htf_bias == 'BULLISH' and direction_pre_bias == 'BUY') or (htf_bias == 'BEARISH' and direction_pre_bias == 'SELL'):
            score_components['htf_bias'] = bias_score if direction_pre_bias == 'BUY' else -bias_score
            score += bias_score if direction_pre_bias == 'BUY' else -bias_score
            logging.info(f"[Arshy | {tf}] Validasi Tren HTF: Sinyal {direction_pre_bias} didukung. Skor disesuaikan {bias_score:+.2f}")
        else:
            score_components['htf_bias'] = penalty_score
            score += penalty_score
            logging.warning(f"[Arshy | {tf}] Validasi Tren HTF: Sinyal {direction_pre_bias} berlawanan dengan bias {htf_bias}. Penalti {penalty_score:.2f} diterapkan.")
    logging.info(f"[Arshy | {tf}] Skor Akhir Terkalkulasi: {score:.2f}")

    # --- Penentuan Arah & Tipe Order Berdasarkan Setup ---
    direction = "WAIT"
    order_type = None
    entry_price_chosen = current_price

    if score >= confidence_threshold:
        direction = "BUY"
        # 1. Cek Skenario Reversal (CHoCH + Sweep/Grab)
        if choch and choch.get('type') == 'BULLISH_CHoCH' and (liquidity_sweep or liquidity_grabs):
            order_type, entry_price_chosen = "BUY", current_price
            info_list.append(f"REVERSAL BUY based on BULLISH_CHoCH and Liquidity event.")
            logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Reversal (CHoCH + Liquidity). Menggunakan MARKET BUY.")
        # 2. Cek Skenario Continuation - Pullback (LIMIT)
        elif "BULLISH" in structure_str:
            bullish_poi = sorted([z for z in fvg_zones if 'BULLISH' in z['type'] and z['start'] < current_price] +
                                   [z for z in order_blocks if 'BULLISH' in z['type'] and z['high'] < current_price],
                                   key=lambda z: z['distance'])
            if bullish_poi:
                best_zone = bullish_poi[0]
                swing_start, swing_end = best_zone.get('end', best_zone.get('low')), best_zone.get('start', best_zone.get('high'))
                ote_price = calculate_optimal_trade_entry(swing_start, swing_end, direction).get('mid')
                if ote_price and ote_price < current_price:
                    order_type, entry_price_chosen = "BUY_LIMIT", ote_price
                    info_list.append(f"CONTINUATION BUY_LIMIT based on {best_zone['type']} OTE")
                    logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Pullback ke {best_zone['type']}. Menggunakan BUY_LIMIT.")

            # 3. Cek Skenario Continuation - Breakout (STOP)
            if not order_type and "BULLISH_BOS" in structure_str and swing_points.get('last_high'):
                order_type, entry_price_chosen = "BUY_STOP", swing_points['last_high'] + (atr * 0.1)
                info_list.append(f"CONTINUATION BUY_STOP based on BULLISH_BOS")
                logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Konfirmasi Breakout. Menggunakan BUY_STOP.")

        # 4. Default ke Market Order jika momentum kuat
        if not order_type:
            order_type, entry_price_chosen = "BUY", current_price
            info_list.append("BUY market order based on strong bullish score.")
            logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Momentum Kuat. Menggunakan MARKET BUY.")

    elif score <= -confidence_threshold:
        direction = "SELL"
        # 1. Cek Skenario Reversal (CHoCH + Sweep/Grab)
        if choch and choch.get('type') == 'BEARISH_CHoCH' and (liquidity_sweep or liquidity_grabs):
            order_type, entry_price_chosen = "SELL", current_price
            info_list.append(f"REVERSAL SELL based on BEARISH_CHoCH and Liquidity event.")
            logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Reversal (CHoCH + Liquidity). Menggunakan MARKET SELL.")
        # 2. Cek Skenario Continuation - Pullback (LIMIT)
        elif "BEARISH" in structure_str:
            bearish_poi = sorted([z for z in fvg_zones if 'BEARISH' in z['type'] and z['start'] > current_price] +
                                   [z for z in order_blocks if 'BEARISH' in z['type'] and z['low'] > current_price],
                                   key=lambda z: z['distance'])
            if bearish_poi:
                best_zone = bearish_poi[0]
                swing_start, swing_end = best_zone.get('start', best_zone.get('high')), best_zone.get('end', best_zone.get('low'))
                ote_price = calculate_optimal_trade_entry(swing_start, swing_end, direction).get('mid')
                if ote_price and ote_price > current_price:
                    order_type, entry_price_chosen = "SELL_LIMIT", ote_price
                    info_list.append(f"CONTINUATION SELL_LIMIT based on {best_zone['type']} OTE")
                    logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Pullback ke {best_zone['type']}. Menggunakan SELL_LIMIT.")

            # 3. Cek Skenario Continuation - Breakout (STOP)
            if not order_type and "BEARISH_BOS" in structure_str and swing_points.get('last_low'):
                order_type, entry_price_chosen = "SELL_STOP", swing_points['last_low'] - (atr * 0.1)
                info_list.append(f"CONTINUATION SELL_STOP based on BEARISH_BOS")
                logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Konfirmasi Breakout. Menggunakan SELL_STOP.")

        # 4. Default ke Market Order
        if not order_type:
            order_type, entry_price_chosen = "SELL", current_price
            info_list.append("SELL market order based on strong bearish score.")
            logging.info(f"[Arshy | {tf}] Setup Teridentifikasi: Momentum Kuat. Menggunakan MARKET SELL.")

    # --- Validasi Akhir & Penentuan SL/TP ---
    if direction == "WAIT":
        return None

    sl, tp1, tp2, tp3 = 0.0, 0.0, 0.0, 0.0
    risk = atr * 1.5
    if direction == "BUY":
        sl = entry_price_chosen - risk
        tp1 = entry_price_chosen + (risk * 1.5)
        tp2 = entry_price_chosen + (risk * 2.0)
        tp3 = entry_price_chosen + (risk * 3.0)
    elif direction == "SELL":
        sl = entry_price_chosen + risk
        tp1 = entry_price_chosen - (risk * 1.5)
        tp2 = entry_price_chosen - (risk * 2.0)
        tp3 = entry_price_chosen - (risk * 3.0)

    invalidation_point = sl
    risk_reward_ratio = (tp3 - entry_price_chosen) / (entry_price_chosen - sl) if direction == "BUY" else (entry_price_chosen - tp3) / (sl - entry_price_chosen)

    logging.info(f"[Arshy | {tf}] --- Analisis Selesai --- | Rekomendasi: {direction} | Tipe: {order_type} | Skor Keyakinan: {score:.2f}")
    return {
        "signal": direction, "order_type": order_type, "entry_price_chosen": entry_price_chosen,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "score": score, "info": "; ".join(info_list),
        "score_components": score_components,
        "invalidation_point": invalidation_point,
        "risk_reward_ratio": risk_reward_ratio,
        "features": get_gng_input_features_full(df, gng_feature_stats, tf) if gng_model else None, 
        "tf": tf, "symbol": symbol,
    }