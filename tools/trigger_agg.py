#!/usr/bin/env python3
"""Krok zbiorczy: rozdzielic TRIGGER od WYKONANIA od WYBORU.

Kontrola jest tasowana WEWNATRZ GODZINY, nie globalnie. Powod zmierzony 29.09:
globalne tasowanie dalo 126 falszywych wzorcow na 162. Jesli kontrola bije
dane, blad jest w tescie.

Oczekiwanie przy losowej monecie liczymy jako srednia CALEJ puli wypelnionych
monet w tej godzinie — to jest dokladnie wartosc oczekiwana losowania, bez
symulacji. Losowanie sluzy tylko do rozrzutu.
"""
import argparse, glob, sys
import numpy as np, pandas as pd

KOSZT = 0.0885


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wejscie", default="paczki")
    ap.add_argument("--lista98", default="")
    ap.add_argument("--losowan", type=int, default=400)
    a = ap.parse_args()
    pl = sorted(glob.glob(f"{a.wejscie}/**/godziny.parquet", recursive=True))
    if not pl:
        sys.exit("brak paczek")
    d = pd.concat([pd.read_parquet(p) for p in pl], ignore_index=True)
    d["t"] = pd.to_datetime(d.t)
    print(f"wczytane: {len(d):,} wierszy, {d.coin.nunique()} monet, "
          f"{d.t.min():%Y-%m-%d} .. {d.t.max():%Y-%m-%d}\n", flush=True)

    # --- trigger: co robil BTC w dobie przed ta godzina ---
    btc = d[d.coin.str.startswith("BTC")].drop_duplicates("t").set_index("t").close.sort_index()
    if len(btc) < 1000:
        sys.exit("BTC musi byc w paczkach — bez niego nie ma triggera")
    btc24 = (btc / btc.shift(24) - 1) * 100
    d["btc24"] = d.t.map(btc24).astype(np.float32)

    if a.lista98:
        gra = {x.strip().upper() for x in open(a.lista98).read().split(",") if x.strip()}
        d = d[d.coin.isin(gra) | d.coin.str.replace("USDT", "").isin(gra)]
        print(f"zawezone do listy ktora GRAMY: {d.coin.nunique()} monet\n", flush=True)

    d["FS"] = d.F & d.zm
    W = d[np.isfinite(d.we) & np.isfinite(d.wy) & np.isfinite(d.btc24)].copy()
    W["zysk"] = (W.wy / W.we - 1) * 100 - KOSZT
    W["zysk_ryn"] = (W.ryn_wy / W.close - 1) * 100 - KOSZT

    # --- BRAMKA ZDROWIA: czy odtwarzamy znany wynik F+S ---
    fs = W[W.FS]
    print("=" * 82)
    print(f"BRAMKA ZDROWIA — F+S bez reguly jednego zlecenia: {len(fs):,} tx, "
          f"srednia {fs.zysk.mean():+.3f}")
    print("  (oczekiwane: dodatnie i wyraznie powyzej calej puli; przy regule "
          "jednego zlecenia bywa ~+4,9 na 98 monetach)")
    if not (len(fs) > 500 and fs.zysk.mean() > 0.5):
        sys.exit("\nSTOP: F+S nie odtwarza znanego wyniku. Nie podaje wnioskow "
                 "z zepsutego pomiaru.")
    print("=" * 82 + "\n", flush=True)

    rng = np.random.default_rng(0)
    kody, _ = pd.factorize(W.t)
    W["hid"] = kody

    def porownaj(M, nazwa):
        """W obrebie tych samych godzin: wybor doktryny kontra losowa moneta."""
        P = W[M]
        if len(P) < 200:
            print(f"  {nazwa:<34} za malo ({len(P)})"); return
        godz = P.hid.unique()
        pula = W[W.hid.isin(godz)]
        wyb = pula[pula.FS]
        if len(wyb) < 100:
            print(f"  {nazwa:<34} za malo wyborow F+S ({len(wyb)})"); return
        # oczekiwanie losowej monety = srednia puli w tych samych godzinach,
        # ale wazona tak, by kazda godzina liczyla sie tyle razy, ile F+S wybral
        licz = wyb.groupby("hid").size()
        srg = pula.groupby("hid").zysk.mean()
        los_oczek = float((srg.reindex(licz.index) * licz).sum() / licz.sum())
        srg_ryn = pula.groupby("hid").zysk_ryn.mean()
        ryn_oczek = float((srg_ryn.reindex(licz.index) * licz).sum() / licz.sum())
        # rozrzut: tasowanie WEWNATRZ godziny, ta sama liczba wyborow
        z = pula.zysk.to_numpy(); hid = pula.hid.to_numpy()
        porz = np.argsort(hid, kind="stable"); z_s = z[porz]; h_s = hid[porz]
        gran = np.r_[0, np.flatnonzero(np.diff(h_s)) + 1, len(h_s)]
        ile = licz.reindex(pd.Index(h_s[gran[:-1]])).fillna(0).to_numpy().astype(int)
        pr = []
        for _ in range(a.losowan):
            sk = []
            for b in range(len(ile)):
                if ile[b] == 0:
                    continue
                seg = z_s[gran[b]:gran[b + 1]]
                sk.append(rng.choice(seg, size=ile[b], replace=True))
            pr.append(np.concatenate(sk).mean() if sk else np.nan)
        pr = np.array(pr)
        akt = wyb.zysk.mean()
        print(f"  {nazwa:<34}{len(wyb):>7}{akt:>9.3f}{los_oczek:>10.3f}"
              f"{akt - los_oczek:>+9.3f}{(pr < akt).mean() * 100:>7.0f}%{ryn_oczek:>10.3f}")

    print("=" * 96)
    print("1. CO NAPRAWDE DAJE PRZEWAGE — w obrebie TYCH SAMYCH godzin")
    print("=" * 96)
    print(f"  {'rezim BTC 24 h':<34}{'tx F+S':>7}{'F+S':>9}{'losowa':>10}"
          f"{'wybor':>9}{'bije':>7}{'rynkowo':>10}")
    for nm, m in (("wszystko", np.ones(len(W), bool)),
                  ("BTC > 0%", W.btc24 >= 0),
                  ("BTC 0..-2%", (W.btc24 < 0) & (W.btc24 >= -2)),
                  ("BTC -2..-6%", (W.btc24 < -2) & (W.btc24 >= -6)),
                  ("BTC < -6%", W.btc24 < -6)):
        porownaj(np.asarray(m), nm)
    print("\n  'losowa' = srednia CALEJ puli wypelnionych monet w tych samych godzinach.")
    print("  'wybor'  = ile dokłada sam fakt, ze doktryna wskazala te monete.")
    print("  'rynkowo'= ta sama pula, ale wejscie po cenie rynkowej zamiast limitem.")
    print("             Roznica losowa-rynkowo to przewaga SAMEGO WYKONANIA.")
    print("             (rynkowo liczone tym samym kosztem 0,0885 — w rzeczywistosci")
    print("              taker kosztuje wiecej, wiec to gorna granica)\n", flush=True)

    print("=" * 96)
    print("2. POTWIERDZENIE PROGU -6% NA PELNYM WSZECHSWIECIE")
    print("=" * 96)
    print(f"  {'BTC 24 h':<16}{'tx':>8}{'srednia':>10}{'trafnosc':>10}{'I pol.':>10}{'II pol.':>10}")
    pol = fs.t.median()
    for nm, m in (("> 0%", fs.btc24 >= 0), ("0..-2%", (fs.btc24 < 0) & (fs.btc24 >= -2)),
                  ("-2..-4%", (fs.btc24 < -2) & (fs.btc24 >= -4)),
                  ("-4..-6%", (fs.btc24 < -4) & (fs.btc24 >= -6)),
                  ("-6..-9%", (fs.btc24 < -6) & (fs.btc24 >= -9)), ("< -9%", fs.btc24 < -9)):
        g = fs[m]
        if len(g) < 40:
            print(f"  {nm:<16}{len(g):>8}   za malo"); continue
        print(f"  {nm:<16}{len(g):>8}{g.zysk.mean():>+10.3f}{(g.zysk > 0).mean() * 100:>9.0f}%"
              f"{g[g.t < pol].zysk.mean():>+10.3f}{g[g.t >= pol].zysk.mean():>+10.3f}")

    print("\n" + "=" * 96)
    print("3. SIZING DWUSTOPNIOWY PRZY TEJ SAMEJ SREDNIEJ EKSPOZYCJI")
    print("=" * 96)
    z = fs.zysk.to_numpy()
    for nm, w in (("plaski", np.ones(len(fs))),
                  ("<-6% pelny, reszta pol", np.where(fs.btc24 < -6, 1.0, 0.5)),
                  ("<-5% pelny, reszta pol", np.where(fs.btc24 < -5, 1.0, 0.5))):
        print(f"  {nm:<30}{np.sum(z * w) / np.sum(w):>+9.3f}   max rozmiar "
              f"{(w / w.mean()).max():.2f}x")
    w6 = np.where(fs.btc24 < -6, 1.0, 0.5)
    los = np.array([np.sum(z * rng.permutation(w6)) / np.sum(w6) for _ in range(a.losowan)])
    print(f"  kontrola (wagi potasowane): {los.mean():+.3f} "
          f"[5-95%: {np.percentile(los, 5):+.3f}..{np.percentile(los, 95):+.3f}] — "
          f"bije w {(los < np.sum(z * w6) / np.sum(w6)).mean() * 100:.0f}%")
    print("\n  po latach:", flush=True)
    for y, g in fs.groupby(fs.t.dt.year):
        mm = (fs.t.dt.year == y).to_numpy()
        print(f"    {y}  n={mm.sum():>6}  plaski {z[mm].mean():>+7.3f} -> "
              f"dwustopniowy {np.sum(z[mm] * w6[mm]) / np.sum(w6[mm]):>+7.3f}")


if __name__ == "__main__":
    main()
