"""
MLE NUMERIC PENTRU RETELE SP SI PS
====================================

Date citite din fisierele CSV reale (cifre pseudoaleatoare 0-9, cate una per linie).

Parametri experiment:
  n = 33   (volum esantion, ca in lucrarea de referinta [1])
  R = 1000 (repetitii — pentru statistici robuste ale distributiei K̂)
  K = 9    (valoarea adevarata a parametrului, de estimat)

De ce n=33 si nu mai mare?
  Cu n mare, L(K) → 0 numeric (underflow) pentru orice K ≠ K̂,
  chiar si in log-spatiu comparativ. Cu n=33, curba log L(K)
  ramane bine definita si maximul este clar identificabil.
  n=33 este totodata volumul utilizat in lucrarea [1].

Rezultate raportate per PRNG × retea × schema cenzurare:
  - Media K̂
  - Dispersia (Var)
  - MSE = (1/R) Σ(K̂ᵣ − 9)²
  - % estimari exacte (K̂ = 9)
  - Histograma distributiei K̂
"""

import math
import os

# ── Caile catre fisierele CSV ─────────────────────────────────────────────────
BASE    = "/Users/astafivalentina/PycharmProjects/SP_PS_Reliability_Simulation/"
FIGURES = BASE

FILES = {
    "Java ThreadLocalRandom": [
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data1.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data2.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data3.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data4.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data5.csv"),
    ],
    "Python secrets": [
        os.path.join(BASE, "data", "python_secrets", "secrets_data1.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data2.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data3.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data4.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data5.csv"),
    ],
    "Digits of π": [
        os.path.join(BASE, "data", "pi", "pi_digits_part.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part1.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part2.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part3.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part4.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part5.csv"),
    ],
}

# ── Parametri ─────────────────────────────────────────────────────────────────
K_TRUE = 9
n = 33       # volumul esantionului (ca in [1])
R = 1000     # numarul de repetitii

# Scheme de cenzurare (ca in [1]):
#   C3: 3 intervale  [0,4), [4,7), [7,+inf)
#   C5: 5 intervale  [0,5), [5,7), [7,8), [8,9), [9,+inf)
INTERVALE_C3 = [(0, 4), (4, 7), (7, float('inf'))]
INTERVALE_C5 = [(0, 5), (5, 7), (7, 8), (8, 9), (9, float('inf'))]


# ════════════════════════════════════════════════════════════════════════════
# FORMULE MATEMATICE SP si PS
# ════════════════════════════════════════════════════════════════════════════

def cdf_sp(k, K, N=3, M=2):
    """F_SP(k) = 1 - [1 - ((k+1)/(K+1))^N]^M"""
    if k < 0: return 0.0
    if k >= K: return 1.0
    return 1.0 - (1.0 - ((k + 1) / (K + 1)) ** N) ** M

def cdf_ps(k, K, N=2, M=3):
    """F_PS(k) = [1 - ((K-k)/(K+1))^N]^M"""
    if k < 0: return 0.0
    if k >= K: return 1.0
    return (1.0 - ((K - k) / (K + 1)) ** N) ** M

def pmf_sp(k, K, N=3, M=2):
    """P(U=k) = F_SP(k) - F_SP(k-1)"""
    if k < 0 or k > K: return 0.0
    return cdf_sp(k, K, N, M) - cdf_sp(k - 1, K, N, M)

def pmf_ps(k, K, N=2, M=3):
    """P(V=k) = F_PS(k) - F_PS(k-1)"""
    if k < 0 or k > K: return 0.0
    return cdf_ps(k, K, N, M) - cdf_ps(k - 1, K, N, M)


# ════════════════════════════════════════════════════════════════════════════
# SIMULARE PRIN METODA INVERSA A CDF
# ════════════════════════════════════════════════════════════════════════════

def uniform_la_durata(u, K, tip):
    """
    Transforma u ∈ [0,1) in durata de viata k prin F^{-1}(u).
    Gasim cel mai mic k astfel incat F(k) >= u.
    """
    cdf = cdf_sp if tip == 'sp' else cdf_ps
    for k in range(K + 1):
        if cdf(k, K) >= u:
            return k
    return K


# ════════════════════════════════════════════════════════════════════════════
# MLE NUMERIC
# ════════════════════════════════════════════════════════════════════════════

def mle_date_complete(date, tip, K_max=30):
    """
    Maximizam log L(K) = Σ log P(U=uᵢ; K) peste K ∈ {1,...,K_max}.

    De ce numeric (nu analitic)?
    Pentru distributia uniforma clasica: K̂ = max(date).
    Pentru MinMax/MaxMin (SP/PS): PMF e o functie complicata de K,
    nu exista formula inchisa => cautare exhaustiva pe K intreg.

    De ce log L si nu L direct?
    L(K) = ∏ P(uᵢ; K) → 0 numeric rapid cu n crescand.
    log L(K) = Σ log P(uᵢ; K) ramane bine definit pentru n=33.
    """
    pmf = pmf_sp if tip == 'sp' else pmf_ps
    best_K, best_ll = None, -math.inf

    for K in range(1, K_max + 1):
        ll, valid = 0.0, True
        for x in date:
            p = pmf(x, K)
            if p <= 0:
                valid = False
                break
            ll += math.log(p)
        if valid and ll > best_ll:
            best_ll, best_K = ll, K

    return best_K


def mle_date_cenzurate(date, intervale, tip, K_max=30):
    """
    Date cenzurate: observatorul vede doar intervalul in care cade
    fiecare observatie, nu valoarea exacta.

    log L_cenz(K) = Σⱼ nⱼ · log P(U ∈ Iⱼ; K)
    unde nⱼ = numarul de observatii in intervalul j.

    P(U ∈ [a,b)) = F(b-1) - F(a-1)   (discret)
    P(U ∈ [b,+∞)) = 1 - F(b-1)
    """
    cdf = cdf_sp if tip == 'sp' else cdf_ps

    # Calculam frecventele per interval
    frecvente = []
    for (a, b) in intervale:
        if b == float('inf'):
            nj = sum(1 for x in date if x >= a)
        else:
            nj = sum(1 for x in date if a <= x < b)
        frecvente.append(nj)

    best_K, best_ll = None, -math.inf

    for K in range(1, K_max + 1):
        ll, valid = 0.0, True
        for j, (a, b) in enumerate(intervale):
            if b == float('inf'):
                prob = 1.0 - (cdf(a - 1, K) if a > 0 else 0.0)
            else:
                prob = cdf(b - 1, K) - (cdf(a - 1, K) if a > 0 else 0.0)
            if prob <= 0:
                valid = False
                break
            if frecvente[j] > 0:
                ll += frecvente[j] * math.log(prob)
        if valid and ll > best_ll:
            best_ll, best_K = ll, K

    return best_K


# ════════════════════════════════════════════════════════════════════════════
# CITIRE CSV si CONVERSIE IN VALORI UNIFORME
# ════════════════════════════════════════════════════════════════════════════

def incarca_cifre_csv(fisiere):
    """
    Citeste cifrele (0-9, cate una per linie) din lista de fisiere CSV.
    Concateneaza toate fisierele intr-un flux continuu.
    """
    toate = []
    for fpath in fisiere:
        with open(fpath, 'r') as f:
            for linie in f:
                l = linie.strip()
                if l and l.isdigit():
                    toate.append(int(l))
    return toate


def extrage_seturi_uniforme(cifre, n, R):
    """
    Din fluxul de cifre, extrage R seturi de n numere uniforme in [0,1).

    Metoda: grupe de 9 cifre consecutive → numar 9 cifre → / 10^9
    Exemplu: [1,4,1,5,9,2,6,5,3] → 141592653 / 10^9 = 0.141592653

    Necesar: R × n × 9 cifre (verificam disponibilitatea).
    """
    necesar = R * n * 9
    if len(cifre) < necesar:
        raise ValueError(
            f"Insuficiente cifre: {len(cifre):,} disponibile, "
            f"{necesar:,} necesare pentru R={R}, n={n}"
        )

    seturi = []
    for r in range(R):
        start = r * n * 9
        uniforme = []
        for i in range(n):
            grup = cifre[start + i * 9 : start + i * 9 + 9]
            val = int(''.join(str(c) for c in grup)) / 1_000_000_000
            uniforme.append(val)
        seturi.append(uniforme)

    return seturi


# ════════════════════════════════════════════════════════════════════════════
# STATISTICI
# ════════════════════════════════════════════════════════════════════════════

def calculeaza_statistici(khat_list, K_true=9):
    """
    Calculeaza statisticile distributiei K̂ pe R repetitii:
      - medie K̂
      - dispersie (Var)
      - MSE = Var + Bias²
      - % estimari exacte
      - histograma valorilor K̂
    """
    R = len(khat_list)
    medie = sum(khat_list) / R
    var   = sum((k - medie) ** 2 for k in khat_list) / R
    mse   = sum((k - K_true) ** 2 for k in khat_list) / R
    exacte = sum(1 for k in khat_list if k == K_true) / R * 100

    hist = {}
    for k in khat_list:
        hist[k] = hist.get(k, 0) + 1

    return medie, var, mse, exacte, hist


def afiseaza_histograma(hist, R):
    for k in sorted(hist):
        bar = '█' * int(hist[k] / R * 50)
        print(f"    K̂={k:>2}: {hist[k]:>5} ({hist[k]/R*100:5.1f}%) {bar}")


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print(f"MLE DIN CSV — K adevarat={K_TRUE}, n={n}, R={R} repetitii")
print("=" * 70)

rezultate = {}

for nume_gen, fisiere in FILES.items():
    print(f"\n{'─'*60}")
    print(f"GENERATOR: {nume_gen}")
    print(f"{'─'*60}")

    # 1. Incarcam toate cifrele din CSV-uri
    cifre = incarca_cifre_csv(fisiere)
    seturi_per_fisier = len(cifre) // (n * 9)
    print(f"  Cifre totale: {len(cifre):,} | Seturi n={n} disponibile: {seturi_per_fisier:,}")

    # 2. Extragem R seturi de n valori uniforme
    seturi = extrage_seturi_uniforme(cifre, n, R)
    rezultate[nume_gen] = {}

    for tip, label in [('sp', 'SP (N=3, M=2)'), ('ps', 'PS (N=2, M=3)')]:
        Khat_complet, Khat_C3, Khat_C5 = [], [], []

        for uniforme in seturi:
            # 3. Simulam duratele de viata prin metoda inversa
            date = [uniform_la_durata(u, K_TRUE, tip) for u in uniforme]

            # 4. MLE numeric pentru fiecare schema
            Khat_complet.append(mle_date_complete(date, tip))
            Khat_C3.append(mle_date_cenzurate(date, INTERVALE_C3, tip))
            Khat_C5.append(mle_date_cenzurate(date, INTERVALE_C5, tip))

        # 5. Statistici
        st_c  = calculeaza_statistici(Khat_complet)
        st_c3 = calculeaza_statistici(Khat_C3)
        st_c5 = calculeaza_statistici(Khat_C5)

        rezultate[nume_gen][tip] = {
            'complet': st_c,
            'C3':      st_c3,
            'C5':      st_c5,
        }

        print(f"\n  {label}:")
        for schema, st in [
            ('Date complete', st_c),
            ('Cenz. C3',      st_c3),
            ('Cenz. C5',      st_c5),
        ]:
            m, v, mse, pct, hist = st
            print(f"  [{schema}]  K̂={m:.3f}  Var={v:.3f}  MSE={mse:.3f}  exact={pct:.1f}%")
            afiseaza_histograma(hist, R)


# ════════════════════════════════════════════════════════════════════════════
# TABEL SINTETIC PENTRU ARTICOL
# ════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 82)
print(f"TABEL 2 — Media K̂, Var, MSE, %exacte  (R={R}, n={n}, K={K_TRUE})")
print(f"{'PRNG':<25} {'Ret.':<5} "
      f"{'Date complete':^22} {'Cenz. C3':^22} {'Cenz. C5':^22}")
print(f"{'':25} {'':5} "
      f"{'K̂  Var  MSE  %ex':^22} {'K̂  Var  MSE  %ex':^22} {'K̂  Var  MSE  %ex':^22}")
print("-" * 82)

for gen in ["Java ThreadLocalRandom", "Python secrets", "Cifrele lui pi"]:
    for tip, label in [('sp', 'SP'), ('ps', 'PS')]:
        d = rezultate[gen][tip]
        row = []
        for schema in ['complet', 'C3', 'C5']:
            m, v, mse, pct, _ = d[schema]
            row.append(f"{m:.2f} {v:.2f} {mse:.2f} {pct:.0f}%")
        print(f"{gen:<25} {label:<5} "
              f"{row[0]:^22} {row[1]:^22} {row[2]:^22}")

print("\nLegenda: K̂=media estimarilor, Var=dispersie, MSE=eroare medie patratica, %ex=% estimari exacte K̂=9")