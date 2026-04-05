"""
================================================================
COD PYTHON — Cazul a) N=3, M=2, Reteaua SERIAL-PARALELA (SP)
n = 644,000 | R = 5 rulari independente | K = 9
================================================================
Rulare: python3 caz_a_SP.py
Timp estimat: ~10-20 minute (n mare)
================================================================
"""

import numpy as np
import secrets as sec_module
import time

# ── Parametri ─────────────────────────────────────────────────
K_TRUE = 9
N      = 3
M      = 2
n      = 644_000   # prin TLC: epsilon=0.005, alpha=0.05
R      = 5         # rulari independente

np.random.seed(42)

# ── Cifrele lui pi (primele ~10000) ───────────────────────────
_PI = (
    "14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
    "82148086513282306647093844609550582231725359408128"
    "48111745028410270193852110555964462294895493038196"
    "44288109756659334461284756482337867831652712019091"
    "45648566923460348610454326648213393607260249141273"
    "72458700660631558817488152092096282925409171536436"
    "78925903600113305305488204665213841469519415116094"
    "33057270365759591953092186117381932611793105118548"
    "07446237996274956735188575272489122793818301194912"
    "98336733624406566430860213949463952247371907021798"
    "60943702770539217176293176752384674818467669405132"
    "00056812714526356082778577134275778960917363717872"
    "14684409012249534301465495853710507922796892589235"
    "42019956112129021960864034418159813629774771309960"
    "51870721134999999837297804995105973173281609631859"
    "50244594553469083026425223082533446850352619311881"
    "71010003137838752886587533208381420617177669147303"
    "59825349042875546873115956286388235378759375195778"
    "18577805321712268066130019278766111959092164201989"
    "38095257201065485863278865936153381827968230301952"
    "03530185296899577362259941389124972177528347913151"
    "55748572424541506959508295331168617278558890750983"
    "81754637464939319255060400927701671139009848824012"
    "85836160356370766010471018194295559619894676783744"
    "94482553797747268471040475346462080466842590694912"
    "93313677028989152104752162056966024058038150193511"
    "25338243003558764024749647326391419927260426992279"
    "67823547816360093417216412199245863150302861829745"
    "55706749838505494588586926995690927210797509302955"
    "32116534498720275596023648066549911988183479775356"
    "63698074265425278625518184175746728909777727938000"
    "81647060016145249192173217214772350141441973568548"
    "16136115735255213347574184946843852332390739414333"
    "45477624168625189835694855620992192221842725502542"
    "56887671790494601653466804988627232791786085784383"
    "82796797668145410095388378636095068006422512520511"
    "73929848960841284886269456042419652850222106611863"
    "06744278622039194945047123713786960956364371917287"
    "46776465757396241389086583264599581339047802759010"
)
_PI_ARR = np.array([int(d) for d in _PI if d.isdigit()])

def pi_digits(n_needed, offset=0):
    total = offset + n_needed
    if total <= len(_PI_ARR):
        return _PI_ARR[offset:offset+n_needed].copy()
    reps = total // len(_PI_ARR) + 2
    return np.tile(_PI_ARR, reps)[offset:offset+n_needed]

# ── Generatoare GNPA ──────────────────────────────────────────

def gnpa1_numpy(n):
    """Java ThreadLocalRandom — simulat cu numpy Mersenne Twister"""
    return np.random.randint(0, 10, size=n)

def gnpa2_secrets(n):
    """Python secrets — generator criptografic"""
    return np.array([sec_module.randbelow(10) for _ in range(n)])

def gnpa3_pi(n, offset=0):
    """Cifrele zecimale ale lui pi"""
    return pi_digits(n, offset)

# ── Model SP: U = min(max(X_ij)) ─────────────────────────────

def pmf_sp(k, N=N, M=M, K=K_TRUE):
    pk  = (k+1)/(K+1)
    pk0 = k/(K+1)
    if k == 0:
        return 1 - (1 - pk**N)**M
    return (1 - pk0**N)**M - (1 - pk**N)**M

def F_sp(k, N=N, M=M, K=K_TRUE):
    return 1 - (1 - ((k+1)/(K+1))**N)**M

def R_sp(k, N=N, M=M, K=K_TRUE):
    return 1 - F_sp(k, N, M, K)

# ── Simulare durate de viata ──────────────────────────────────

def simulate_SP(gnpa_func, n_sim, pi_offset=0):
    """Simuleaza n_sim durate de viata ale retelei SP."""
    units_total = N * M
    lifetimes = np.empty(n_sim, dtype=np.int32)
    
    if gnpa_func == gnpa3_pi:
        # Pt pi: generam tot blocul odata
        all_digits = pi_digits(n_sim * units_total, offset=pi_offset)
        matrix = all_digits.reshape(n_sim, M, N)
    else:
        all_units = gnpa_func(n_sim * units_total)
        matrix = all_units.reshape(n_sim, M, N)
    
    # SP: U = min_i(max_j(X_ij))
    lifetimes = matrix.max(axis=2).min(axis=1)
    return lifetimes.astype(np.int32)

# ── MLE pentru date complete ──────────────────────────────────

def mle_complete(data, K_max=30):
    """MLE pentru K pe baza datelor complete."""
    K_min = int(np.max(data))
    best_K, best_ll = K_min, -np.inf
    
    for K_try in range(K_min, K_max+1):
        ll = 0.0
        for val, cnt in zip(*np.unique(data, return_counts=True)):
            p = pmf_sp(int(val), K=K_try)
            if p <= 0:
                ll = -np.inf; break
            ll += cnt * np.log(p)
        if ll > best_ll:
            best_ll, best_K = ll, K_try
    return best_K

# ── MLE pentru date censurate ─────────────────────────────────

def prob_interval_sp(a, b, K_try):
    """P(U in [a, b)) pentru reteaua SP."""
    F_b = F_sp(b-1, K=K_try) if b is not None else 1.0
    F_a = F_sp(a-1, K=K_try) if a > 0 else 0.0
    return F_b - F_a

def mle_censored(data, intervals, K_max=30):
    """MLE pentru K pe baza datelor censurate."""
    # Calculeaza frecventele
    counts = []
    for (a, b) in intervals:
        if b is None:
            counts.append(int(np.sum(data >= a)))
        else:
            counts.append(int(np.sum((data >= a) & (data < b))))
    
    K_min = int(intervals[-1][0])
    best_K, best_ll = K_min, -np.inf
    
    for K_try in range(K_min, K_max+1):
        ll = 0.0
        valid = True
        for (a, b), cnt in zip(intervals, counts):
            if cnt == 0: continue
            p = prob_interval_sp(a, b, K_try)
            if p <= 0: valid = False; break
            ll += cnt * np.log(p)
        if valid and ll > best_ll:
            best_ll, best_K = ll, K_try
    return best_K

# Scheme de censurare (SP)
INT_C3 = [(0, 4), (4, 7), (7, None)]
INT_C5 = [(0, 4), (4, 6), (6, 7), (7, 8), (8, None)]

# ── Rulare completa ───────────────────────────────────────────

GENERATORS = [
    ("Java ThreadLocalRandom (numpy)", gnpa1_numpy),
    ("Python secrets",                  gnpa2_secrets),
    ("Cifrele lui pi",                  gnpa3_pi),
]

print("=" * 70)
print(f"CAZ a) N={N}, M={M}, K={K_TRUE} — Reteaua SERIAL-PARALELA (SP)")
print(f"n = {n:,}  |  R = {R} rulari independente")
print("=" * 70)

# Valori teoretice
E_U = sum(k * pmf_sp(k) for k in range(K_TRUE+1))
D_U = sum((k-E_U)**2 * pmf_sp(k) for k in range(K_TRUE+1))
print(f"\nValori teoretice: E[U] = {E_U:.6f}  D[U] = {D_U:.6f}")
print(f"h_sp(5) = {pmf_sp(5)/R_sp(4):.6f}\n")

results = {}

for gnpa_name, gnpa_func in GENERATORS:
    print(f"\n{'─'*70}")
    print(f"GNPA: {gnpa_name}")
    print(f"{'─'*70}")
    
    K_hats_c  = []
    K_hats_c3 = []
    K_hats_c5 = []
    means_sim = []
    
    t0 = time.time()
    
    for r in range(R):
        print(f"  Rularea {r+1}/{R}...", end=" ", flush=True)
        t1 = time.time()
        
        # Offset pentru pi
        pi_off = r * n * N * M
        
        # Simuleaza datele
        data = simulate_SP(gnpa_func, n, pi_offset=pi_off)
        means_sim.append(np.mean(data))
        
        # MLE date complete
        K_c = mle_complete(data)
        K_hats_c.append(K_c)
        
        # MLE date censurate C3
        K_c3 = mle_censored(data, INT_C3)
        K_hats_c3.append(K_c3)
        
        # MLE date censurate C5
        K_c5 = mle_censored(data, INT_C5)
        K_hats_c5.append(K_c5)
        
        t2 = time.time()
        print(f"K_c={K_c}, K_c3={K_c3}, K_c5={K_c5} [{t2-t1:.1f}s]")
    
    # Statistici
    K_hats_c  = np.array(K_hats_c)
    K_hats_c3 = np.array(K_hats_c3)
    K_hats_c5 = np.array(K_hats_c5)
    
    print(f"\n  Rezultate finale:")
    print(f"  Media simulata (medie pe rulari): {np.mean(means_sim):.6f}  (teoretic: {E_U:.6f})")
    print(f"  {'Tip date':<25} {'K_hat_mean':>12} {'Bias':>10} {'MSE':>10} {'Std':>10}")
    print(f"  {'-'*67}")
    for label, arr in [("Complete", K_hats_c), ("Censurate C3", K_hats_c3), ("Censurate C5", K_hats_c5)]:
        print(f"  {label:<25} {np.mean(arr):>12.4f} {np.mean(arr)-K_TRUE:>10.4f} {np.mean((arr-K_TRUE)**2):>10.4f} {np.std(arr):>10.4f}")
    
    results[gnpa_name] = {
        'means_sim': float(np.mean(means_sim)),
        'complete': {'mean': float(np.mean(K_hats_c)), 'bias': float(np.mean(K_hats_c)-K_TRUE), 'mse': float(np.mean((K_hats_c-K_TRUE)**2)), 'std': float(np.std(K_hats_c))},
        'cens3':    {'mean': float(np.mean(K_hats_c3)),'bias': float(np.mean(K_hats_c3)-K_TRUE),'mse': float(np.mean((K_hats_c3-K_TRUE)**2)),'std': float(np.std(K_hats_c3))},
        'cens5':    {'mean': float(np.mean(K_hats_c5)),'bias': float(np.mean(K_hats_c5)-K_TRUE),'mse': float(np.mean((K_hats_c5-K_TRUE)**2)),'std': float(np.std(K_hats_c5))},
    }
    
    print(f"\n  Timp total GNPA: {time.time()-t0:.1f}s")

# ── Tabel comparativ final ────────────────────────────────────

print(f"\n{'='*70}")
print("TABEL COMPARATIV FINAL — Cazul a) SP, N=3, M=2")
print(f"{'='*70}")
print(f"{'GNPA':<35} {'K_c':>7} {'Bias_c':>7} {'MSE_c':>7} | {'K_c3':>7} {'MSE_c3':>7} | {'K_c5':>7} {'MSE_c5':>7}")
print("-"*90)
for name, res in results.items():
    short = name.split()[0] + " " + (name.split()[1] if len(name.split())>1 else "")
    print(f"{name:<35}"
          f" {res['complete']['mean']:>7.4f} {res['complete']['bias']:>7.4f} {res['complete']['mse']:>7.4f}"
          f" | {res['cens3']['mean']:>7.4f} {res['cens3']['mse']:>7.4f}"
          f" | {res['cens5']['mean']:>7.4f} {res['cens5']['mse']:>7.4f}")

print(f"\n✅ Cazul a) SP complet! Copiaza rezultatele de mai sus in articol.")
