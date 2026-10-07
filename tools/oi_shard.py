#!/usr/bin/env python3
"""Jedna paczka monet: transakcje F+S + oi_z. Dane ciagnie z data.binance.vision.

Dlaczego na GitHubie: lokalnie ten pomiar zajal 49 minut na jednym rdzeniu
maszyny, na ktorej stoja cztery instancje handlowe — i nie wypisywal postepu,
wiec nie bylo wiadomo, czy zyje. Tu idzie 20 paczek naraz i kazda gada, co robi.

Warunki doktryny sa przepisane WEKTOROWO, ale nie z pamieci: wersja wektorowa
zostala sprawdzona wobec kodu silnika (`tools/doktryna_wektor.py` na VPS,
4 800 porownan, 0 rozjazdow). "ATR" doktryny to ROZSTEP 14 h, nie klasyczny ATR.

Nie liczymy tu zwrotow wzgledem rynku — do tego potrzeba wszystkich monet, wiec
robi to dopiero krok zbiorczy. Stad zapisujemy ceny i czasy, nie gotowy zysk.
"""
import argparse, io, os, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

BAZA = "https://data.binance.vision/data/futures/um"
TRZYM_H, OSTROZ, BAZA_H = 24, 0.0005, 30 * 24
MIN_WAR, PROG_ATR = 2, 0.40     # jak w .env EPV/DEV


def _zip(sesja, url, kol):
    for p in range(3):
        try:
            r = sesja.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    d = pd.read_csv(f, header=None)
            if str(d.iloc[0, 0]).replace(".", "").isdigit() is False:
                d = d.iloc[1:]
            d.columns = kol[:d.shape[1]]
            return d
        except Exception:
            if p == 2:
                return None
            time.sleep(2 * (p + 1))


def klines(sesja, sym, od, do):
    """Swiece 1 h z paczek MIESIECZNYCH — 45 plikow zamiast 1 400 dziennych."""
    kol = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "qav", "trades", "tbb", "tbq", "ign"]
    mies = pd.period_range(od, do, freq="M").astype(str)
    cz = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for d in ex.map(lambda m: _zip(sesja, f"{BAZA}/monthly/klines/{sym}/1h/{sym}-1h-{m}.zip", kol), mies):
            if d is not None and len(d):
                cz.append(d)
    if not cz:
        return None
    o = pd.concat(cz, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "open_time"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o["t"] = pd.to_datetime(o.open_time, unit="ms", errors="coerce")
    if o.t.isna().mean() > 0.5:
        o["t"] = pd.to_datetime(o.open_time, unit="us", errors="coerce")
    return o.dropna(subset=["t"]).drop_duplicates("t").sort_values("t").set_index("t")


def metrics(sesja, sym, od, do):
    """Otwarte pozycje co 5 min — paczki DZIENNE, bo miesiecznych Binance nie daje."""
    kol = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
           "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
           "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    dni = [d.strftime("%Y-%m-%d") for d in pd.date_range(od, do, freq="D")]
    cz = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for d in ex.map(lambda x: _zip(sesja, f"{BAZA}/daily/metrics/{sym}/{sym}-metrics-{x}.zip", kol), dni):
            if d is not None and len(d):
                cz.append(d[["create_time", "sum_open_interest"]])
    if not cz:
        return None
    m = pd.concat(cz, ignore_index=True)
    m["create_time"] = pd.to_datetime(m.create_time, errors="coerce")
    m["sum_open_interest"] = pd.to_numeric(m.sum_open_interest, errors="coerce")
    return m.dropna().set_index("create_time").sum_open_interest.sort_index()


def warunki(c, h, l, v):
    """Sprawdzone wobec kodu silnika — patrz naglowek pliku."""
    C, H, L, V = (pd.Series(x) for x in (c, h, l, v))
    hm, lm = (lambda k: H.rolling(k).max()), (lambda k: L.rolling(k).min())
    d = C.diff()
    zy, st = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    rsi = np.where(st < 1e-12, 100.0, 100 - 100 / (1 + zy / st.replace(0, np.nan)))
    sa = (((hm(9) + lm(9)) / 2).shift(26) + ((hm(26) + lm(26)) / 2).shift(26)) / 2
    sb = ((hm(52) + lm(52)) / 2).shift(26)
    m20, s20 = C.rolling(20).mean(), C.rolling(20).std(ddof=1)
    bbw = np.where(m20 > 0, 4 * s20 / m20 * 100, np.nan)
    W = {
        "rsi": rsi < 40,
        "kumo": (C < np.minimum(sa, sb)).to_numpy(),
        "tk": (((hm(9) + lm(9)) / 2) <= ((hm(26) + lm(26)) / 2)).to_numpy(),
        "vol": (((V - V.rolling(168).mean()) / V.rolling(168).std(ddof=1).replace(0, np.nan)) > 1.0).to_numpy(),
        "wst": (pd.Series(bbw) > pd.Series(bbw).rolling(720, min_periods=1).median()).to_numpy() & np.isfinite(bbw),
    }
    mn24 = L.rolling(24).min().shift(1)
    sw = ((L < mn24) & (C > mn24)).to_numpy()
    zm = np.zeros(len(c), bool)
    for k in (1, 2, 3):
        zm[k:] |= sw[:-k]
    W = {k: np.nan_to_num(x.astype(float), nan=0.0).astype(bool) for k, x in W.items()}
    W["zm"] = zm
    F = W["wst"] & ((W["rsi"].astype(int) + W["kumo"] + W["tk"] + W["vol"]) >= MIN_WAR)
    return F, W["zm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monety", required=True)
    ap.add_argument("--od", default="2023-01-01")
    ap.add_argument("--do", default=(pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    ap.add_argument("--jedno-zlecenie", default="1",
                    help="1 = jedno zlecenie na monete naraz (jak silnik), 0 = jak oryginalny "
                         "pomiar glebokosc_lacznie.py. Bez tej reguly jest 6x wiecej transakcji: "
                         "wersja do pytania CZY JEST INFORMACJA. Z regula — do pytania ILE ZAROBIMY.")
    ap.add_argument("--glebokosc", type=float, default=2.0,
                    help="limit N*rozstep14 pod cena; 2,0 to EPV/DEV, 1,5 daje 2x wiecej wypelnien")
    ap.add_argument("--wyjscie", default="wynik")
    a = ap.parse_args()
    os.makedirs(a.wyjscie, exist_ok=True)
    GLEB = a.glebokosc
    JEDNO = a.jedno_zlecenie == "1"
    print(f"glebokosc limitu: {GLEB} x rozstep 14 h | jedno zlecenie na monete: {JEDNO}", flush=True)
    s = requests.Session(); s.headers["User-Agent"] = "HAI/1.0"
    tx, zam = [], {}
    for sym in [x.strip().upper() for x in a.monety.split(",") if x.strip()]:
        t0 = time.time()
        sb = sym if sym.endswith("USDT") else f"{sym}USDT"
        o = klines(s, sb, a.od, a.do)
        if o is None or len(o) < 800:
            print(f"  {sym}: brak swiec", flush=True); continue
        oi = metrics(s, sb, a.od, a.do)
        if oi is None or len(oi) < 2000:
            print(f"  {sym}: brak OI", flush=True); continue
        c, h, l, v = (o[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
        atr = (o.high.rolling(14).max() - o.low.rolling(14).min()).to_numpy(float)
        oih = oi.groupby(oi.index.floor("h")).last().reindex(o.index)
        z = ((oih - oih.rolling(BAZA_H, min_periods=BAZA_H // 3).mean())
             / oih.rolling(BAZA_H, min_periods=BAZA_H // 3).std()).to_numpy(float)
        F, zm = warunki(c, h, l, v)
        S = F & zm
        zam[sym] = o.close
        wolne, n = 0, 0
        for i in np.flatnonzero(S):
            if JEDNO and i < wolne:
                continue
            if i + 2 * TRZYM_H >= len(o) or not np.isfinite(atr[i]) or not np.isfinite(z[i]):
                continue
            lim = c[i] - GLEB * atr[i]
            k = next((j for j in range(i + 1, i + 1 + TRZYM_H) if l[j] <= lim * (1 - OSTROZ)), None)
            if k is None:
                wolne = i + TRZYM_H; continue
            wolne = k + TRZYM_H
            if k + TRZYM_H >= len(o):
                continue
            tx.append({"coin": sym, "t_syg": o.index[i], "t_we": o.index[k],
                       "t_wy": o.index[k + TRZYM_H], "wejscie": lim,
                       "wyjscie": c[k + TRZYM_H], "oiz": z[i],
                       "atr_pct": atr[i] / c[i] * 100})
            n += 1
        print(f"  {sym}: {len(o):,} swiec, {n} transakcji, {time.time()-t0:.0f} s", flush=True)
    if tx:
        pd.DataFrame(tx).to_parquet(f"{a.wyjscie}/tx.parquet", index=False)
    if zam:
        pd.DataFrame(zam).to_parquet(f"{a.wyjscie}/zamkniecia.parquet")
    print(f"\nRAZEM {len(tx)} transakcji, {len(zam)} monet", flush=True)


if __name__ == "__main__":
    main()
