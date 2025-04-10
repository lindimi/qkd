import math, random, csv

def H2(x):
    if x <= 0 or x >= 1:
        return 0.0
    return - (x*math.log2(x) + (1-x)*math.log2(1-x))

Lbc = 20
fiber_loss = 0.2
det_eff = 0.8
eta = (10 ** (-fiber_loss * (Lbc/2) / 10)) * det_eff

def compute_key_rate(Y0, ed, mu):
    P_single = mu * math.exp(-mu)
    P11_Z = (P_single ** 2)
    Y11_Z = 0.5 * eta * eta
    E_mu = ed
    P0 = math.exp(-mu)
    Q_single = P11_Z * Y11_Z
    Q_one_dark = 2 * (P_single * P0 * Y0)
    Q_mu = Q_single + Q_one_dark
    return P11_Z * Y11_Z * (1 - H2(ed)) - Q_mu * 1.16 * H2(E_mu)

def optimize_parameters(Y0, ed, N):
    mu = 0.5
    best_mu, best_rate = mu, compute_key_rate(Y0, ed, mu)
    for m in [x/100 for x in range(5, 81, 5)]:
        R = compute_key_rate(Y0, ed, m)
        if R > best_rate:
            best_mu, best_rate = m, R
    m_start = max(0.05, best_mu - 0.05)
    m_end   = min(0.8, best_mu + 0.05)
    for m in [x/1000 for x in range(int(m_start*1000), int(m_end*1000)+1, 5)]:
        R = compute_key_rate(Y0, ed, m)
        if R > best_rate:
            best_mu, best_rate = m, R
    if best_mu >= 0.4:
        nu = 0.2
    else:
        nu = max(0.05, 0.5 * best_mu)
    omega = 0.02
    if omega >= nu:
        omega = 0.5 * nu
    if N >= 5e12:
        decoy_fraction = 0.10
    elif N >= 1e12:
        decoy_fraction = 0.15
    else:
        decoy_fraction = 0.20
    p_mu = 1 - decoy_fraction
    p_nu = 0.05 * decoy_fraction
    p_omega = decoy_fraction - p_nu
    return best_mu, nu, omega, p_mu, p_nu, p_omega, best_rate

random.seed(0)
dataset = []
for _ in range(1000):
    Y0 = 10 ** random.uniform(math.log10(1e-8), math.log10(1e-5))
    ed = random.uniform(0.005, 0.02)
    N = int(10 ** random.uniform(math.log10(1e10), math.log10(1e14)))
    mu, nu, omega, p_mu, p_nu, p_omega, key_rate = optimize_parameters(Y0, ed, N)
    dataset.append([Lbc, Y0, ed, N, mu, nu, omega, p_mu, p_nu, p_omega, key_rate])

with open('mdi_qkd_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Lbc_km","Y0","ed","N","mu","nu","omega","p_mu","p_nu","p_omega","key_rate"])
    writer.writerows(dataset)

print("Dataset of", len(dataset), "samples saved to mdi_qkd_dataset.csv")