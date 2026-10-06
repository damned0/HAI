#!/usr/bin/env python3
"""bookDepth z data.binance.vision -> SREDNIE GODZINOWE (2026-10-06).

Po co: surowe bookDepth to zdjecie ksiegi co ~30 s, czyli 2 880 zdjec na dobe x
12 poziomow = ~34 tys. wierszy na monete na dobe. Na naszym dysku rok dla 134
monet zajal 22 GB, a do niczego, co robimy, nie potrzebujemy rozdzielczosci
30-sekundowej. Srednia godzinowa to 24 x 12 = 288 wierszy na dobe, czyli
120 razy mniej. Archiwum siega 2023-01-01, wiec na GitHubie bierzemy CALE trzy
lata i wracamy z plikiem rzedu pol giga zamiast 54 GB.

Dlaczego na GitHubie, a nie u siebie: 135 tys. plikow po pol megabajta to 54 GB
ruchu i kilka godzin. Runner i tak to wyrzuca po zakonczeniu, a do nas wraca
tylko wynik.

Wyjscie: {SYM}.parquet z kolumnami
    godzina (UTC, pelna godzina), percentage, notional, depth, n
gdzie notional/depth to SREDNIE ze zdjec w tej godzinie, a n to liczba zdjec
(trzymamy ja, zeby bylo widac godziny z dziurami w danych).
"""
import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BAZA = "https://data.binance.vision/data/futures/um/daily/bookDepth"
START = "2023-01-01"      # wczesniej archiwum nie istnieje (2022-12-01 = 404)
KOLUMNY = ["timestamp", "percentage", "depth", "notional"]


def dzien(sesja, sym, d):
    """Jeden dzien -> DataFrame srednich godzinowych albo None."""
    url = f"{BAZA}/{sym}/{sym}-bookDepth-{d}.zip"
    for proba in range(3):
        try:
            r = sesja.get(url, timeout=120)
            if r.status_code == 404:
                return None          # moneta jeszcze nie istniala tego dnia
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    o = pd.read_csv(f)
            break
        except Exception:
            if proba == 2:
                print(f"    {sym} {d}: nie udalo sie", flush=True)
                return None
            time.sleep(2 * (proba + 1))
    if o.empty:
        return None
    o.columns = [c.strip() for c in o.columns]
    brak = [c for c in KOLUMNY if c not in o.columns]
    if brak:
        print(f"    {sym} {d}: brak kolumn {brak}", flush=True)
        return None
    o["timestamp"] = pd.to_datetime(o.timestamp)
    o["godzina"] = o.timestamp.dt.floor("h")
    g = (o.groupby(["godzina", "percentage"])
           .agg(notional=("notional", "mean"), depth=("depth", "mean"),
                n=("notional", "size"))
           .reset_index())
    return g


def moneta(sym, od, do, watki, wyj):
    dni = [d.strftime("%Y-%m-%d") for d in pd.date_range(od, do, freq="D")]
    sesja = requests.Session()
    sesja.headers["User-Agent"] = "HAI/1.0"
    czesci = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=watki) as ex:
        for g in ex.map(lambda d: dzien(sesja, sym, d), dni):
            if g is not None and len(g):
                czesci.append(g)
    if not czesci:
        print(f"  {sym}: ZERO dni", flush=True)
        return 0
    o = pd.concat(czesci, ignore_index=True).sort_values(["godzina", "percentage"])
    o["notional"] = o.notional.astype("float32")
    o["depth"] = o.depth.astype("float32")
    o["n"] = o.n.astype("int16")
    p = Path(wyj) / f"{sym.replace('USDT','')}.parquet"
    o.to_parquet(p, compression="zstd", index=False)
    print(f"  {sym}: {len(o):,} wierszy, {len(czesci)} dni, "
          f"{p.stat().st_size/2**20:.1f} MB, {time.time()-t0:.0f} s", flush=True)
    return len(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monety", required=True, help="lista po przecinku, bez USDT")
    ap.add_argument("--od", default=START)
    ap.add_argument("--do", default=(pd.Timestamp.utcnow() - pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    ap.add_argument("--watki", type=int, default=12)
    ap.add_argument("--wyjscie", default="wynik")
    a = ap.parse_args()
    Path(a.wyjscie).mkdir(parents=True, exist_ok=True)
    syms = [s.strip().upper() for s in a.monety.split(",") if s.strip()]
    print(f"monet: {len(syms)}, zakres {a.od} .. {a.do}, watkow {a.watki}", flush=True)
    razem = 0
    for s in syms:
        razem += moneta(s if s.endswith("USDT") else f"{s}USDT", a.od, a.do, a.watki, a.wyjscie)
    print(f"\nRAZEM {razem:,} wierszy w {a.wyjscie}", flush=True)


if __name__ == "__main__":
    main()
