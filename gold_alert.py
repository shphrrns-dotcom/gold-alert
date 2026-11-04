"""
gold_alert.py - ระบบแจ้งเตือนทอง XAU/USD เข้า Discord
- H1 บอกทิศทางเทรนด์ (EMA 50/200)
- M15 หาจุดเข้า: FVG / Order Block / แนวรับ-แนวต้าน + Stochastic + แท่งยืนยัน
- แจ้งเตือนเฉพาะตอนเงื่อนไขครบ (#setup-alert) และสรุปตลาดทุกชั่วโมง (#hourly-update)
- ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน
"""
import os
import sys
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

# ================= ค่าที่ปรับได้ =================
SYMBOL = "XAU/USD"
EMA_FAST, EMA_SLOW = 50, 200        # เทรนด์ H1
STOCH_OVERSOLD = 25                 # ช่วง oversold (ฝั่ง Buy)
STOCH_OVERBOUGHT = 75               # ช่วง overbought (ฝั่ง Sell)
LOOKBACK = 80                       # มองย้อนหลังกี่แท่ง M15 เพื่อหา FVG/OB
COOLDOWN_MIN = 60                   # ไม่ส่งสัญญาณทิศเดิมซ้ำภายในกี่นาที
STATE_FILE = "state.json"
# =================================================

TD_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
HOOK_SETUP = os.environ.get("DISCORD_WEBHOOK_SETUP", "")
HOOK_HOURLY = os.environ.get("DISCORD_WEBHOOK_HOURLY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")
FORCE_TEST = os.environ.get("FORCE_HOURLY") == "1"   # รันมือ = ทดสอบ
UA = {"User-Agent": "gold-alert-bot/1.0"}


# ---------------- ดึงข้อมูลราคา ----------------
def fetch_twelvedata(interval, size=320):
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": SYMBOL, "interval": interval, "outputsize": size,
                "timezone": "UTC", "order": "ASC", "apikey": TD_KEY},
        timeout=30,
    )
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError("TwelveData: " + str(j.get("message", j))[:200])
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["datetime", "open", "high", "low", "close"]]


def fetch_yahoo(interval, period):
    import yfinance as yf
    df = yf.download("GC=F", interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "datetime", "Open": "open",
                            "High": "high", "Low": "low", "Close": "close"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[["datetime", "open", "high", "low", "close"]]


def closed_only(df, minutes):
    """ตัดแท่งที่ยังไม่ปิดออก"""
    now = pd.Timestamp.now(tz="UTC")
    df = df.sort_values("datetime")
    return df[df["datetime"] + pd.Timedelta(minutes=minutes) <= now].reset_index(drop=True)


def get_candles(td_interval, yf_interval, yf_period, minutes):
    try:
        df, src = fetch_twelvedata(td_interval), "Twelve Data XAU/USD"
    except Exception as e:
        print("Twelve Data ใช้ไม่ได้ -> สลับไป Yahoo (GC=F):", e)
        df, src = fetch_yahoo(yf_interval, yf_period), "Yahoo GC=F (ฟิวเจอร์ส)"
    return closed_only(df, minutes), src


# ---------------- อินดิเคเตอร์ ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def stoch(df, k=14, d=3, smooth=3):
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll)
    kk = raw.rolling(smooth).mean()
    return kk, kk.rolling(d).mean()


def h1_trend(h1):
    c = h1["close"]
    e_fast, e_slow = ema(c, EMA_FAST).iloc[-1], ema(c, EMA_SLOW).iloc[-1]
    last = c.iloc[-1]
    if last > e_fast > e_slow:
        t = "up"
    elif last < e_fast < e_slow:
        t = "down"
    else:
        t = "range"
    return t, round(float(e_fast), 2), round(float(e_slow), 2)


def swing_points(h, l, w=5, start=0):
    highs, lows = [], []
    for i in range(max(w, start), len(h) - w):
        if h[i] == h[i - w:i + w + 1].max():
            highs.append((i, float(h[i])))
        if l[i] == l[i - w:i + w + 1].min():
            lows.append((i, float(l[i])))
    return highs, lows


# ---------------- หาโซน FVG / OB / SR ----------------
def detect_zones(m):
    o, h, l, c = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(m)
    a = atr(m).to_numpy()
    zones = []
    start = max(6, n - LOOKBACK)

    # FVG (ช่องว่างระหว่างแท่งที่ 1 กับแท่งที่ 3)
    for i in range(start, n - 1):
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= 0.15 * a[i]:
            lo, hi = h[i - 2], l[i]
            if not (c[i + 1:] < lo).any():
                zones.append(dict(type="FVG", side="buy", lo=float(lo), hi=float(hi)))
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= 0.15 * a[i]:
            lo, hi = h[i], l[i - 2]
            if not (c[i + 1:] > hi).any():
                zones.append(dict(type="FVG", side="sell", lo=float(lo), hi=float(hi)))

    # Order Block (แท่งสวนทางสุดท้ายก่อนแรงเร่งที่ทะลุ high/low ล่าสุด)
    seen = set()
    for i in range(start, n - 1):
        if abs(c[i] - o[i]) < 1.5 * a[i]:
            continue
        if c[i] > o[i] and c[i] > h[i - 5:i].max():
            for k in range(i - 1, i - 5, -1):
                if c[k] < o[k]:
                    if (k, "buy") not in seen:
                        seen.add((k, "buy"))
                        if not (c[i + 1:] < l[k]).any():
                            zones.append(dict(type="OB", side="buy", lo=float(l[k]), hi=float(h[k])))
                    break
        elif c[i] < o[i] and c[i] < l[i - 5:i].min():
            for k in range(i - 1, i - 5, -1):
                if c[k] > o[k]:
                    if (k, "sell") not in seen:
                        seen.add((k, "sell"))
                        if not (c[i + 1:] > h[k]).any():
                            zones.append(dict(type="OB", side="sell", lo=float(l[k]), hi=float(h[k])))
                    break

    # แนวรับ-แนวต้าน (swing ที่แตะซ้ำ >= 2 ครั้ง)
    highs, lows = swing_points(h, l, 5, n - 200)
    levels = sorted([v for _, v in highs] + [v for _, v in lows])
    clusters = []
    for v in levels:
        if clusters and v - clusters[-1][-1] <= 0.5 * a[-1]:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    for cl in clusters:
        if len(cl) < 2:
            continue
        lv, half = float(np.mean(cl)), 0.2 * a[-1]
        side = "buy" if c[-2] > lv else "sell"
        zones.append(dict(type="SR", side=side, lo=lv - half, hi=lv + half))
    return zones


def touches(z, row):
    if z["side"] == "buy":
        return row["low"] <= z["hi"] and row["close"] >= z["lo"]
    return row["high"] >= z["lo"] and row["close"] <= z["hi"]


# ---------------- ตรวจสัญญาณ ----------------
def find_signal(m, trend):
    if trend not in ("up", "down"):
        return None
    side = "buy" if trend == "up" else "sell"
    kk, dd = stoch(m)
    k, d = kk.to_numpy(), dd.to_numpy()
    if np.isnan(k[-3:]).any() or np.isnan(d[-1]):
        return None
    row = m.iloc[-1]
    if side == "buy":
        stoch_ok = k[-3:].min() < STOCH_OVERSOLD and k[-1] > d[-1] and k[-1] > k[-2]
        candle_ok = row["close"] > row["open"]
    else:
        stoch_ok = k[-3:].max() > STOCH_OVERBOUGHT and k[-1] < d[-1] and k[-1] < k[-2]
        candle_ok = row["close"] < row["open"]
    zs = [z for z in detect_zones(m) if z["side"] == side and touches(z, row)]
    if not zs:
        return dict(ok=False, side=side, zones=[], stoch_ok=bool(stoch_ok), candle_ok=bool(candle_ok))
    a = float(atr(m).iloc[-1])
    lo, hi = min(z["lo"] for z in zs), max(z["hi"] for z in zs)
    return dict(
        ok=bool(stoch_ok and candle_ok), side=side,
        zones=sorted({z["type"] for z in zs}), zone_lo=round(lo, 2), zone_hi=round(hi, 2),
        stoch_k=round(float(k[-1]), 1), stoch_d=round(float(d[-1]), 1),
        stoch_ok=bool(stoch_ok), candle_ok=bool(candle_ok),
        close=round(float(row["close"]), 2), atr=round(a, 2),
        invalid=round(lo - 0.5 * a, 2) if side == "buy" else round(hi + 0.5 * a, 2),
        candle_time=str(row["datetime"]),
    )


def build_facts(h1, m, trend, e_fast, e_slow):
    o, h, l, c = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
    price = float(c[-1])
    a = float(atr(m).iloc[-1])
    kk, dd = stoch(m)
    highs, lows = swing_points(h, l, 5, len(m) - 150)
    sh = highs[-1][1] if highs else None
    sl = lows[-1][1] if lows else None
    if sh is not None and price > sh:
        structure = "ราคาทะลุ high ล่าสุดขึ้นไป (BOS ฝั่งขึ้น)"
    elif sl is not None and price < sl:
        structure = "ราคาหลุด low ล่าสุดลงมา (BOS ฝั่งลง)"
    else:
        structure = "ราคายังอยู่ในกรอบระหว่าง low และ high ล่าสุด"
    zones = detect_zones(m)

    def nearest(side):
        cand = [z for z in zones if z["side"] == side and
                ((z["hi"] <= price + 0.2 * a) if side == "buy" else (z["lo"] >= price - 0.2 * a))]
        cand.sort(key=lambda z: abs(price - (z["lo"] + z["hi"]) / 2))
        return [dict(type=z["type"], lo=round(z["lo"], 2), hi=round(z["hi"], 2)) for z in cand[:2]]

    bias = {"up": "BUY", "down": "SELL", "range": "WAIT"}[trend]
    if (trend == "up" and sl is not None and price < sl) or (trend == "down" and sh is not None and price > sh):
        bias = "WAIT (ราคาสวนเทรนด์ H1 ตอนนี้)"
    return dict(
        symbol="XAUUSD", price=round(price, 2), atr_m15=round(a, 2),
        h1_trend={"up": "ขาขึ้น", "down": "ขาลง", "range": "ไซด์เวย์/ไม่ชัด"}[trend],
        h1_ema50=e_fast, h1_ema200=e_slow,
        last_swing_high=round(sh, 2) if sh else None,
        last_swing_low=round(sl, 2) if sl else None,
        structure=structure, buy_zones=nearest("buy"), sell_zones=nearest("sell"),
        stoch_k=round(float(kk.iloc[-1]), 1), stoch_d=round(float(dd.iloc[-1]), 1),
        bias=bias,
    )


# ---------------- Gemini ----------------
def model_candidates():
    names = [GEMINI_MODEL] if GEMINI_MODEL else []
    names += ["gemini-flash-latest", "gemini-flash-lite-latest"]
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         headers={"x-goog-api-key": GEMINI_KEY}, params={"pageSize": 200}, timeout=30)
        bad = ("live", "tts", "image", "audio", "embed", "translate", "robotics", "computer",
               "gemma", "vision", "preview", "omni", "exp")
        found = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        found = sorted([n for n in found if "flash" in n and not any(b in n for b in bad)], reverse=True)
        names += found[:2]
    except Exception:
        pass
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


_MODELS = None


def gemini(prompt):
    global _MODELS
    if not GEMINI_KEY:
        return None
    if not _MODELS:
        _MODELS = model_candidates()
    for model in _MODELS:
        for attempt in range(3):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}},
                    timeout=60,
                )
                if r.status_code == 200:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if text:
                        return text
                    break
                print("Gemini", model, r.status_code, r.text[:120].replace("\n", " "))
                if r.status_code in (500, 503) and attempt < 2:
                    time.sleep(8 * (attempt + 1))   # โมเดลคนใช้เยอะ ลองใหม่
                    continue
                break
            except Exception as e:
                print("Gemini error", model, str(e)[:120])
                break
    return None


RULES = ("กติกา: ใช้เฉพาะตัวเลขในข้อมูลที่ให้ ห้ามแต่งตัวเลขเพิ่ม ห้ามฟันธงว่าราคาจะขึ้นหรือลง "
         "ห้ามแนะนำขนาดไม้หรือจำนวนเงิน เขียนภาษาไทยกระชับ")


# ---------------- Discord ----------------
def discord(hook, title, desc, color):
    if not hook:
        print("ไม่มี webhook:", title)
        return
    payload = {"embeds": [{
        "title": title, "description": desc[:4000], "color": color,
        "footer": {"text": "ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน"},
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }]}
    r = requests.post(hook, json=payload, headers=UA, timeout=30)
    print("Discord", title, r.status_code)


def fmt_zone(z):
    return f"{z['type']} {z['lo']:,.2f}–{z['hi']:,.2f}"


def send_signal(sig, facts_trend, src):
    side_th = "LONG 🟢" if sig["side"] == "buy" else "SHORT 🔴"
    prompt = (f"คุณเป็นผู้ช่วยสรุปสัญญาณเทรดทอง ข้อมูลสัญญาณ: {json.dumps(sig, ensure_ascii=False)} "
              f"เทรนด์ H1: {facts_trend}\n"
              f"อธิบาย 3-4 บรรทัดว่าทำไมเงื่อนไขจึงครบ (เทรนด์ โซน stochastic แท่งยืนยัน) "
              f"และบอกว่าถ้าราคาไปถึงจุดยกเลิกสัญญาณ แปลว่าอะไร\n{RULES}")
    ai = gemini(prompt) or "(สรุปด้วย AI ไม่พร้อมใช้งานในรอบนี้ ดูเงื่อนไขด้านบนได้เลย)"
    desc = (
        f"**XAUUSD | {side_th}** · Timeframe M15 (ทิศทางจาก H1: {facts_trend})\n\n"
        f"✅ **เงื่อนไขครบ**\n"
        f"1. เทรนด์ H1 ตรงทิศ ({facts_trend})\n"
        f"2. ราคาแตะโซน: {' + '.join(sig['zones'])} ({sig['zone_lo']:,.2f}–{sig['zone_hi']:,.2f})\n"
        f"3. Stochastic K={sig['stoch_k']} D={sig['stoch_d']} กลับตัวจากโซน "
        f"{'oversold' if sig['side'] == 'buy' else 'overbought'}\n"
        f"4. แท่งยืนยัน: {'เขียว' if sig['side'] == 'buy' else 'แดง'}\n\n"
        f"💰 ราคาปิดล่าสุด: {sig['close']:,.2f}\n"
        f"⛔ จุดยกเลิกสัญญาณ: {'ต่ำกว่า' if sig['side'] == 'buy' else 'สูงกว่า'} {sig['invalid']:,.2f}\n\n"
        f"🤖 **AI:** {ai}\n\n"
        f"_ที่มาราคา: {src}_"
    )
    discord(HOOK_SETUP, f"สัญญาณ {sig['side'].upper()} · XAUUSD", desc,
            0x2ECC71 if sig["side"] == "buy" else 0xE74C3C)


def send_hourly(facts, src):
    prompt = (f"คุณเป็นผู้ช่วยวิเคราะห์ตลาดทอง XAUUSD M15 ข้อมูล: {json.dumps(facts, ensure_ascii=False)}\n"
              "เขียนสรุปเป็น 5 หัวข้อ: 1) โครงสร้างตลาด 2) โซน Buy 3) โซน Sell "
              "4) Confirmation (เทรนด์ H1 และ Stochastic) 5) Bias\n"
              "หัวข้อละ 1-2 บรรทัด ถ้าไม่มีโซนให้บอกว่าไม่มีโซนใกล้ราคา\n" + RULES)
    ai = gemini(prompt)
    if not ai:
        bz = ", ".join(fmt_zone(z) for z in facts["buy_zones"]) or "ไม่มีโซนใกล้ราคา"
        sz = ", ".join(fmt_zone(z) for z in facts["sell_zones"]) or "ไม่มีโซนใกล้ราคา"
        ai = (f"1. โครงสร้าง: {facts['structure']}\n2. โซน Buy: {bz}\n3. โซน Sell: {sz}\n"
              f"4. Confirmation: H1 {facts['h1_trend']}, Stoch K={facts['stoch_k']} D={facts['stoch_d']}\n"
              f"5. Bias: {facts['bias']}")
    desc = f"**XAUUSD / M15** · ราคาปิดล่าสุด {facts['price']:,.2f}\n\n{ai}\n\n_ที่มาราคา: {src}_"
    discord(HOOK_HOURLY, "สรุปตลาดทองรายชั่วโมง", desc, 0xF1C40F)


# ---------------- state ----------------
def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)


# ---------------- main ----------------
def main():
    now = dt.datetime.now(dt.timezone.utc)
    state = load_state()
    h1, src_h1 = get_candles("1h", "60m", "1mo", 60)
    m15, src = get_candles("15min", "15m", "5d", 15)
    if len(h1) < EMA_SLOW + 5 or len(m15) < 60:
        raise RuntimeError(f"ข้อมูลไม่พอ H1={len(h1)} M15={len(m15)}")

    age = (pd.Timestamp.now(tz="UTC") - m15["datetime"].iloc[-1]).total_seconds()
    if age > 3 * 3600 and not FORCE_TEST:
        print("ตลาดน่าจะปิดอยู่ (ไม่มีแท่งใหม่) ข้ามรอบนี้")
        save_state(state)
        return

    trend, e_fast, e_slow = h1_trend(h1)
    print(f"H1 trend={trend} ema{EMA_FAST}={e_fast} ema{EMA_SLOW}={e_slow} | M15 close={m15['close'].iloc[-1]}")
    trend_th = {"up": "ขาขึ้น", "down": "ขาลง", "range": "ไซด์เวย์"}[trend]

    # 1) สัญญาณ
    sig = find_signal(m15, trend)
    if sig:
        print("ตรวจสัญญาณ:", {k: sig[k] for k in ("side", "zones", "stoch_ok", "candle_ok", "ok")})
    if sig and sig["ok"]:
        last = state.get("last_signal", {})
        recent = False
        if last.get("side") == sig["side"] and last.get("ts"):
            ts = dt.datetime.fromisoformat(last["ts"])
            recent = (now - ts).total_seconds() < COOLDOWN_MIN * 60
        if last.get("candle") == sig["candle_time"] or recent:
            print("ข้าม: ส่งสัญญาณทิศนี้ไปแล้วเมื่อไม่นานมานี้")
        else:
            send_signal(sig, trend_th, src)
            state["last_signal"] = {"side": sig["side"], "ts": now.isoformat(), "candle": sig["candle_time"]}

    # 2) สรุปรายชั่วโมง (รอบแรกของแต่ละชั่วโมง)
    hour_key = now.strftime("%Y-%m-%dT%H")
    if FORCE_TEST or state.get("hourly") != hour_key:
        facts = build_facts(h1, m15, trend, e_fast, e_slow)
        send_hourly(facts, src)
        state["hourly"] = hour_key

    if FORCE_TEST:
        discord(HOOK_SETUP, "✅ ทดสอบระบบ", "ห้อง setup-alert เชื่อมต่อสำเร็จ จะมีข้อความที่นี่เมื่อเงื่อนไขสัญญาณครบ", 0x3498DB)
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
