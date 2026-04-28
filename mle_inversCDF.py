"""
Simulare INVERSA CDF → salveaza rezultate in CSV
Output: results_inversa.csv
"""
import math, csv

K_TRUE = 9
n, R   = 33, 1000

C3 = [(0,4), (4,7), (7, float('inf'))]
C5 = [(0,5), (5,7), (7,8), (8,9), (9, float('inf'))]

FISIERE = {
    'Java ThreadLocalRandom': '/Users/astafivalentina/PycharmProjects/SP_PS_Reliability_Simulation/data/java_threadlocal/threadlocal_data1.csv',
    'Python secrets':          '/Users/astafivalentina/PycharmProjects/SP_PS_Reliability_Simulation/data/python_secrets/secrets_data1.csv',
    'Cifrele lui pi':          '/Users/astafivalentina/PycharmProjects/SP_PS_Reliability_Simulation/data/pi/pi_digits_part.csv',
}
OUTPUT_CSV = 'results_inversa.csv'

# ─── FORMULE ─────────────────────────────────────────────────────────────────

def cdf_sp(k, K=9, N=3, M=2):
    if k < 0: return 0.0
    if k >= K: return 1.0
    return 1.0 - (1.0 - ((k+1)/(K+1))**N)**M

def cdf_ps(k, K=9, N=2, M=3):
    if k < 0: return 0.0
    if k >= K: return 1.0
    return (1.0 - ((K-k)/(K+1))**N)**M

def pmf_sp(k, K, N=3, M=2):
    if k < 0 or k > K: return 0.0
    return cdf_sp(k,K,N,M) - cdf_sp(k-1,K,N,M)

def pmf_ps(k, K, N=2, M=3):
    if k < 0 or k > K: return 0.0
    return cdf_ps(k,K,N,M) - cdf_ps(k-1,K,N,M)

def sim_inversa(cifre, start, n, tip):
    cdf = cdf_sp if tip=='sp' else cdf_ps
    date = []
    for i in range(n):
        s = start + i*9
        u = int(''.join(str(c) for c in cifre[s:s+9])) / 1_000_000_000
        for k in range(10):
            if cdf(k) >= u:
                date.append(k); break
    return date

def mle_complet(date, tip, K_max=30):
    pmf = pmf_sp if tip=='sp' else pmf_ps
    best_K, best_ll = None, -math.inf
    for K in range(1, K_max+1):
        ll, ok = 0.0, True
        for x in date:
            p = pmf(x, K)
            if p <= 0: ok=False; break
            ll += math.log(p)
        if ok and ll > best_ll:
            best_ll, best_K = ll, K
    return best_K

def mle_cenzurat(date, intervale, tip, K_max=30):
    cdf = cdf_sp if tip=='sp' else cdf_ps
    freq = [sum(1 for x in date if (x>=a if b==float('inf') else a<=x<b))
            for (a,b) in intervale]
    best_K, best_ll = None, -math.inf
    for K in range(1, K_max+1):
        ll, ok = 0.0, True
        for j,(a,b) in enumerate(intervale):
            p = (1.0-(cdf(a-1,K) if a>0 else 0.0)) if b==float('inf') \
                else cdf(b-1,K)-(cdf(a-1,K) if a>0 else 0.0)
            if p<=0: ok=False; break
            if freq[j]>0: ll += freq[j]*math.log(p)
        if ok and ll>best_ll:
            best_ll, best_K = ll, K
    return best_K

def stat(lst):
    R_ = len(lst)
    m   = sum(lst)/R_
    v   = sum((k-m)**2 for k in lst)/R_
    mse = sum((k-K_TRUE)**2 for k in lst)/R_
    pct = sum(1 for k in lst if k==K_TRUE)/R_*100
    return round(m,4), round(v,4), round(mse,4), round(pct,2)

def citeste(filepath, necesar):
    c = []
    with open(filepath) as f:
        for line in f:
            l = line.strip()
            if l and l.isdigit(): c.append(int(l))
            if len(c) >= necesar: break
    return c

# ─── SIMULARE + SALVARE CSV ──────────────────────────────────────────────────

per_rep = n * 9   # 297

rows = []

print(f"Simulare INVERSA CDF — R={R}, n={n}")
print(f"Salvare in: {OUTPUT_CSV}")
print("─" * 50)

for gen, filepath in FISIERE.items():
    print(f"  {gen}...", end="", flush=True)
    cifre = citeste(filepath, R * per_rep)

    for tip, lab in [('sp','SP'), ('ps','PS')]:
        Kc, Kc3, Kc5 = [], [], []
        for r in range(R):
            date = sim_inversa(cifre, r * per_rep, n, tip)
            Kc.append(mle_complet(date, tip))
            Kc3.append(mle_cenzurat(date, C3, tip))
            Kc5.append(mle_cenzurat(date, C5, tip))

        mc, vc, msec, pctc   = stat(Kc)
        mc3,vc3,msec3,pctc3  = stat(Kc3)
        mc5,vc5,msec5,pctc5  = stat(Kc5)

        rows.append({
            'metoda':     'inversa_cdf',
            'prng':       gen,
            'retea':      lab,
            'c_khat':  mc,  'c_var':  vc,  'c_mse':  msec,  'c_pct':  pctc,
            'c3_khat': mc3, 'c3_var': vc3, 'c3_mse': msec3, 'c3_pct': pctc3,
            'c5_khat': mc5, 'c5_var': vc5, 'c5_mse': msec5, 'c5_pct': pctc5,
        })
    print(" OK")

fieldnames = ['metoda','prng','retea',
              'c_khat','c_var','c_mse','c_pct',
              'c3_khat','c3_var','c3_mse','c3_pct',
              'c5_khat','c5_var','c5_mse','c5_pct']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n Salvat {len(rows)} randuri in '{OUTPUT_CSV}'")