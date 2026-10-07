#!/usr/bin/env python3
"""Jedna paczka monet: PELNA tablica godzinowa, nie tylko sygnaly.

PYTANIE: ile z przewagi F+S zostaje ponad sam trigger "BTC spadl w dobie"?

Zmierzone 07.10: 88% transakcji doktryny powstaje przy BTC 24 h ponizej -2%,
a przy BTC ponizej -6% srednia skacze z ~+2,7 na ~+10,4. Trigger robi wiecej
roboty niz wszystkie filtry razem. Opisywalismy go jako tlo.

Zeby to rozstrzygnac, nie wystarczy wiedziec, co doktryna WYBRALA. Trzeba
wiedziec, co by sie stalo, gdyby TE SAMO zlecenie zlozyc na LOSOWEJ monecie
w TEJ SAMEJ godzinie. Dlatego ten plik liczy wynik zlecenia dla KAZDEJ monety
i KAZDEJ godziny — niezaleznie od tego, czy doktryna cokolwiek powiedziala.

Trzy rzeczy, ktore trzeba rozdzielic, bo do tej pory lezaly razem:
  1. TRIGGER   — sama godzina paniki
  2. WYKONANIE — limit 2,0 rozstepu pod cena (zmierzone 28.09: przewaga limitu
                 to LEPSZA CENA WEJSCIA, nie przewidywanie)
  3. WYBOR     — ze doktryna wskazala akurat te monete

Kontrola dopasowana (losowa moneta, ta sama godzina, ten sam limit) rozdziela
2 od 3. Wejscie rynkowe rozdziela 1 od 2.

Warunki doktryny przepisane z `oi_shard.py`, ktory byl sprawdzony wobec kodu
silnika (4 800 porownan, 0 rozjazdow). "ATR" to ROZSTEP 14 h, nie klasyczny ATR.
"""
import argparse, os, time
import numpy as np, pandas as pd, requests

from oi_shard import klines, warunki, TRZYM_H, OSTROZ, MIN_WAR


def wyniki_zlecen(c, h, l, atr, gleb):
    """Dla KAZDEJ godziny i: co zrobi zlecenie zlozone wlasnie teraz.

    Limit = close[i] - gleb*rozstep14[i], wazny TRZYM_H godzin. Po wypelnieniu
    trzymanie TRZYM_H godzin na zegar. Zwraca cene wejscia, cene wyjscia i
    godzine wypelnienia (NaN gdy nie wypelnil).

    Wektorowo sie nie da sensownie — wypelnienie to "pierwsze j takie, ze...".
    Petla po i z wewnetrznym skanem 24 h: 33 tys. godzin x 24 = 800 tys. krokow
    na monete. Lokalnie to minuty razy 98 monet, tutaj kazda paczka robi swoje.
    """
    n = len(c)
    we = np.full(n, np.nan); wy = np.full(n, np.nan); kiedy = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(atr[i]) or i + 2 * TRZYM_H >= n:
            continue
        lim = c[i] - gleb * atr[i]
        if not (lim > 0):
            continue
        prog = lim * (1 - OSTROZ)
        for j in range(i + 1, i + 1 + TRZYM_H):
            if l[j] <= prog:
                we[i] = lim; wy[i] = c[j + TRZYM_H]; kiedy[i] = j - i
                break
    return we, wy, kiedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monety", required=True)
    ap.add_argument("--od", default="2023-01-01")
    ap.add_argument("--do", default=(pd.Timestamp.utcnow().tz_localize(None)
                                     - pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    ap.add_argument("--glebokosc", type=float, default=2.0)
    ap.add_argument("--wyjscie", default="wynik")
    a = ap.parse_args()
    os.makedirs(a.wyjscie, exist_ok=True)
    print(f"glebokosc {a.glebokosc} | okres {a.od}..{a.do}", flush=True)
    s = requests.Session(); s.headers["User-Agent"] = "HAI/1.0"
    czesci = []
    for sym in [x.strip().upper() for x in a.monety.split(",") if x.strip()]:
        t0 = time.time()
        sb = sym if sym.endswith("USDT") else f"{sym}USDT"
        o = klines(s, sb, a.od, a.do)
        if o is None or len(o) < 800:
            print(f"  {sym}: brak swiec", flush=True); continue
        c, h, l, v = (o[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
        atr = (o.high.rolling(14).max() - o.low.rolling(14).min()).to_numpy(float)
        F, zm, nwar = warunki(c, h, l, v)
        we, wy, kiedy = wyniki_zlecen(c, h, l, atr, a.glebokosc)
        # wejscie rynkowe w tej samej godzinie — do rozdzielenia triggera od limitu
        ryn_wy = np.r_[c[TRZYM_H:], np.full(TRZYM_H, np.nan)]
        czesci.append(pd.DataFrame({
            "coin": sym, "t": o.index,
            "close": c, "atr_pct": atr / c * 100,
            "F": F, "zm": zm, "n_war": nwar.astype(np.int8),
            "we": we, "wy": wy, "godz_do_wypelnienia": kiedy,
            "ryn_wy": ryn_wy,
        }))
        print(f"  {sym}: {len(o):,} swiec, wypelnien {np.isfinite(we).sum():,}, "
              f"F+S {(F & zm).sum():,}, {time.time()-t0:.0f} s", flush=True)
    if czesci:
        d = pd.concat(czesci, ignore_index=True)
        # float32 wystarcza na ceny wzgledne, a plik schodzi o polowe
        for k in ("close", "atr_pct", "we", "wy", "ryn_wy"):
            d[k] = d[k].astype(np.float32)
        d["godz_do_wypelnienia"] = d.godz_do_wypelnienia.astype(np.float32)
        d.to_parquet(f"{a.wyjscie}/godziny.parquet", index=False)
        print(f"\nRAZEM {len(d):,} wierszy, {d.coin.nunique()} monet", flush=True)


if __name__ == "__main__":
    main()
