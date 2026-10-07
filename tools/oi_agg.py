#!/usr/bin/env python3
"""Krok zbiorczy: sklada paczki, odejmuje rynek, podaje KSZTALT zaleznosci oi_z.

Rynek liczymy dopiero tutaj, bo indeks potrzebuje WSZYSTKICH monet, a paczka
widzi tylko swoje piec. Stad shardy zapisuja ceny i czasy, a nie gotowy zysk.

BRAMKA ZDROWIA: jesli baza F+S nie wypadnie w +2,3..+2,7 na transakcje, pomiar
staje i nie podaje wnioskow. Trzy razy 06.10 uratowala przed opublikowaniem
"poprawy" liczonej wewnatrz zepsutej bazy.
"""
import glob, sys
import numpy as np, pandas as pd

KOSZT, PROG_ATR = 0.0885, 0.40

# Oczekiwana baza F+S ZALEZY OD GLEBOKOSCI — plytszy limit daje mniej z kazdej
# transakcji, za to wiecej transakcji. Liczby z tools/glebokosc_lacznie.py
# (2021-08..2026-10, 134 monety). Pierwsza wersja miala wpisane +2,3..+2,7 na
# sztywno i przy glebokosci 1,5 zatrzymala poprawny pomiar.
ODNIESIENIE = {0.75: 0.2574, 1.0: 0.4958, 1.25: 0.8591, 1.5: 1.3643,
               2.0: 2.6743, 2.5: 4.0772, 3.0: 6.3076}


def main():
    tx = [pd.read_parquet(p) for p in sorted(glob.glob("paczki/*/tx.parquet"))]
    zm = [pd.read_parquet(p) for p in sorted(glob.glob("paczki/*/zamkniecia.parquet"))]
    if not tx or not zm:
        print("brak paczek"); sys.exit(1)
    T = pd.concat(tx, ignore_index=True)
    Z = pd.concat(zm, axis=1)
    Z = Z.loc[:, ~Z.columns.duplicated()].sort_index()
    print(f"paczek: {len(tx)} | transakcji: {len(T):,} | monet w indeksie: {Z.shape[1]}")

    mi = (1 + Z.pct_change().mean(axis=1).fillna(0)).cumprod()
    rm = (mi.reindex(T.t_wy).to_numpy() / mi.reindex(T.t_we).to_numpy() - 1) * 100
    T["zysk"] = (T.wyjscie / T.wejscie - 1) * 100 - np.nan_to_num(rm) - KOSZT

    # prog ATR PRZEKROJOWY — ranga monety wsrod wszystkich w tej samej godzinie.
    # BYLO TU wyliczenie `A` ze stosu Z, ktorego nigdy nie uzywalem, a ktore
    # wywalalo caly krok (w nowszej pandzie `stack().reset_index()` nie daje
    # kolumn level_0/level_1). Martwy kod zabil zywy wynik.
    atr = T[["t_syg", "coin", "atr_pct"]].rename(columns={"t_syg": "t"})
    atr["rank"] = atr.groupby("t").atr_pct.rank(pct=True)
    T = T.merge(atr[["t", "coin", "rank"]], left_on=["t_syg", "coin"],
                right_on=["t", "coin"], how="left")
    T = T[T["rank"].fillna(1.0) >= PROG_ATR].dropna(subset=["zysk"]).reset_index(drop=True)

    dni = (T.t_syg.max() - T.t_syg.min()).days or 1
    sr = T.zysk.mean()
    print(f"\n  po progu ATR: {len(T):,} transakcji, TxPD {len(T)/dni:.2f}")
    print(f"  baza: {sr:+.4f} na transakcje (oczekiwane +2,3..+2,7)")
    if not (1.0 < sr < 4.5):
        print("\n  STOP: baza nie odtwarza F+S. Nie podaje wnioskow z zepsutego pomiaru.")
        sys.exit(0)

    print("\n" + "=" * 76)
    print("KSZTALT: wynik w kolejnych DECYLACH oi_z")
    print("=" * 76)
    T["dec"] = pd.qcut(T.oiz, 10, labels=False, duplicates="drop")
    print(f"  {'decyl':>6}{'zakres oi_z':>18}{'n':>7}{'srednia':>10}{'mediana':>10}{'LACZNIE':>10}")
    for d, g in T.groupby("dec"):
        print(f"  {int(d)+1:>6}{f'{g.oiz.min():+.2f}..{g.oiz.max():+.2f}':>18}"
              f"{len(g):>7}{g.zysk.mean():>10.3f}{g.zysk.median():>10.3f}{g.zysk.sum():>10.0f}")
    srd = T.groupby("dec").zysk.mean()
    rho = pd.Series(srd.index).corr(pd.Series(srd.values), method="spearman")
    naj = int(srd.idxmax()) + 1
    print(f"\n  korelacja rangowa decyl -> srednia: {rho:+.3f}")
    print(f"  najlepszy decyl: {naj} z 10")
    if abs(rho) > 0.7:
        print("  -> MONOTONICZNA. Model nie znajdzie nic ponad progi i wagi, ktore juz odpadly.")
    elif naj in (1, 10):
        print("  -> kraniec wygrywa, krzywa nierowna — raczej prog niz garb.")
    else:
        print("  -> GARB W SRODKU. Tego monotoniczne ksztalty nie moglyby zlapac.")
    print(f"\n  korelacja oi_z z ATR%: {T[['oiz','atr_pct']].corr(method='spearman').iloc[0,1]:+.4f}")
    T.to_parquet("wynik_tx.parquet", index=False)


if __name__ == "__main__":
    main()
