"""
Citeste results_structurala.csv si results_inversa.csv

"""
import csv

K_TRUE = 9
FILE_S = 'results_structurala.csv'
FILE_I = 'results_inversa.csv'

# ─── CITIRE CSV ──────────────────────────────────────────────────────────────

def citeste_csv(filepath):
    """Returneaza dict: (prng, retea) -> row"""
    data = {}
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['prng'], row['retea'])
            data[key] = {k: float(v) if k not in ('metoda','prng','retea')
                         else v for k, v in row.items()}
    return data

# ─── AFISARE TABEL INDIVIDUAL ────────────────────────────────────────────────

def print_tabel(data, titlu, R=1000, n=33):
    print(f"\n{'='*82}")
    print(f"TABEL — {titlu}  (R={R}, n={n}, K={K_TRUE})")
    print(f"{'PRNG':<25} {'Ret.':<5} "
          f"{'Date complete':^22} {'Cenz. C3':^22} {'Cenz. C5':^22}")
    print(f"{'':25} {'':5} "
          f"{'K̂  Var  MSE  %ex':^22} {'K̂  Var  MSE  %ex':^22} {'K̂  Var  MSE  %ex':^22}")
    print("-" * 82)
    for gen in ['Java ThreadLocalRandom', 'Python secrets', 'Cifrele lui pi']:
        for lab in ['SP', 'PS']:
            row = data[(gen, lab)]
            def fmt(prefix):
                m   = row[f'{prefix}_khat']
                v   = row[f'{prefix}_var']
                mse = row[f'{prefix}_mse']
                pct = row[f'{prefix}_pct']
                return f"{m:.2f} {v:.2f} {mse:.2f} {pct:.0f}%"
            print(f"{gen:<25} {lab:<5} "
                  f"{fmt('c'):^22} {fmt('c3'):^22} {fmt('c5'):^22}")

# ─── TABEL COMPARATIV ────────────────────────────────────────────────────────

def print_comparatie(ds, di):
    print(f"\n{'='*95}")
    print("COMPARATIE DIRECTA: Structurala vs Inversa CDF")
    print(f"{'='*95}")

    for schema, label in [('c','Date complete'), ('c3','Cenz. C3'), ('c5','Cenz. C5')]:
        print(f"\n  ── {label} ──")
        print(f"  {'PRNG':<25} {'Ret.':<5} "
              f"{'Struct. MSE':>12} {'Inv. MSE':>10} {'ΔMSE':>8} "
              f"{'Struct.%':>10} {'Inv.%':>8} {'Δ%':>6} {'Castig.':>12}")
        print("  " + "-"*90)

        for gen in ['Java ThreadLocalRandom', 'Python secrets', 'Cifrele lui pi']:
            for lab in ['SP', 'PS']:
                rs = ds[(gen, lab)]
                ri = di[(gen, lab)]

                mse_s = rs[f'{schema}_mse']
                mse_i = ri[f'{schema}_mse']
                pct_s = rs[f'{schema}_pct']
                pct_i = ri[f'{schema}_pct']

                delta_mse = mse_i - mse_s   # pozitiv = structurala mai buna
                delta_pct = pct_i - pct_s   # pozitiv = inversa mai buna

                if mse_s < mse_i:
                    castig = 'Structurala ◄'
                elif mse_i < mse_s:
                    castig = 'Inversa CDF ◄'
                else:
                    castig = 'Egale'

                print(f"  {gen:<25} {lab:<5} "
                      f"{mse_s:>12.4f} {mse_i:>10.4f} "
                      f"{delta_mse:>+8.4f} "
                      f"{pct_s:>10.1f}% {pct_i:>7.1f}% "
                      f"{delta_pct:>+5.1f}% {castig:>12}")

# ─── SUMAR FINAL ─────────────────────────────────────────────────────────────

def print_sumar(ds, di):
    print(f"\n{'='*60}")
    print("SUMAR: Care metoda e mai buna?")
    print(f"{'='*60}")

    contoare = {'Structurala': 0, 'Inversa CDF': 0, 'Egale': 0}

    for schema in ['c', 'c3', 'c5']:
        for gen in ['Java ThreadLocalRandom', 'Python secrets', 'Cifrele lui pi']:
            for lab in ['SP', 'PS']:
                mse_s = ds[(gen,lab)][f'{schema}_mse']
                mse_i = di[(gen,lab)][f'{schema}_mse']
                if mse_s < mse_i:   contoare['Structurala'] += 1
                elif mse_i < mse_s: contoare['Inversa CDF'] += 1
                else:               contoare['Egale'] += 1

    total = sum(contoare.values())
    for metoda, cnt in contoare.items():
        bar = '█' * int(cnt/total*30)
        print(f"  {metoda:<15}: {cnt:>2}/{total} ({cnt/total*100:.0f}%) {bar}")

    print(f"\n  Concluzie: ", end="")
    if contoare['Structurala'] > contoare['Inversa CDF']:
        print("Metoda STRUCTURALA are MSE mai mic in mai multe cazuri.")
    elif contoare['Inversa CDF'] > contoare['Structurala']:
        print("Metoda INVERSA CDF are MSE mai mic in mai multe cazuri.")
    else:
        print("Metodele sunt echivalente.")

# ─── MAIN ────────────────────────────────────────────────────────────────────

print("Citesc rezultatele din CSV...")
try:
    ds = citeste_csv(FILE_S)
    di = citeste_csv(FILE_I)
    print(f"   {FILE_S}: {len(ds)} randuri")
    print(f"   {FILE_I}: {len(di)} randuri")
except FileNotFoundError as e:
    print(f"\n Fisier negasit: {e}")
    print("Asigura-te ca ai rulat mai intai:")
    print("  1. save_results_structurala.py")
    print("  2. save_results_inversa.py")
    exit(1)

print_tabel(ds, "Metoda STRUCTURALA (U=min(max), V=max(min), 6 cifre/obs)")
print_tabel(di, "Metoda INVERSA CDF (9 cifre/obs → u uniform → F⁻¹(u))")
print_comparatie(ds, di)
print_sumar(ds, di)