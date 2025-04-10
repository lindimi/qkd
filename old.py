import math
import random
import pandas as pd

# Constants for the channel and setup
alpha_db_per_km = 0.2          # fiber loss in dB/km
eta_detector = 0.1            # detector efficiency (10% in this example)
f_error_correction = 1.1      # error correction inefficiency
security_failure = 1e-10      # target failure probability for finite-key

def binary_entropy(p):
    return -p*math.log2(p) - (1-p)*math.log2(1-p) if 0 < p < 1 else 0.0

def compute_key_rate(L, Y0, e_d, N, params):

    mu = params['mu']
    nu = params['nu']
    p_mu = params['p_mu']
    p_nu = params['p_nu']
    p0 = max(0.0, 1 - p_mu - p_nu)
    
    transmittance = eta_detector * (10 ** (-alpha_db_per_km * L / 10.0))
    
    (detection probabilities) for signal, decoy, and vacuum
    Q_mu = 1 - (1 - Y0) * math.exp(-mu * transmittance)
    Q_nu = 1 - (1 - Y0) * math.exp(-nu * transmittance)
    Q0  = 1 - (1 - Y0) * math.exp(-0 * transmittance) 
    
    E_mu = 0.0
    E_nu = 0.0
    if Q_mu > 0:
        E_mu = (0.5 * Y0 + e_d * (1 - math.exp(-mu * transmittance))) / Q_mu
    if Q_nu > 0:
        E_nu = (0.5 * Y0 + e_d * (1 - math.exp(-nu * transmittance))) / Q_nu
    E0 = 0.5 
    
    numerator = mu * math.exp(mu) * Q_nu - nu * math.exp(nu) * Q_mu
    denominator = mu - nu
    if abs(denominator) < 1e-6:
        Y1 = 0.0
    else:
        Y1 = numerator / denominator
        if Y1 < 0:
            Y1 = 0.0

    e1 = 0.0
    if Y1 > 1e-12:
        e1 = (E_nu * Q_nu * math.exp(nu) - E_mu * Q_mu * math.exp(mu)) / ((math.exp(nu) - math.exp(mu)) * Y1)
        e1 = min(max(e1, 0.0), 0.5)

    Q1 = Y1 * math.exp(-mu) * mu
    R_infinite = Q1 * (1 - binary_entropy(e1)) - Q_mu * f_error_correction * binary_entropy(E_mu)
    if R_infinite < 0:
        return 0.0
    penalty = 6 * math.sqrt(math.log(1/security_failure) / (2 * N))
    R = max(0.0, R_infinite - penalty)
    return R

def random_initial_params():
    mu0 = random.random() 
    nu0 = random.random()  
    if nu0 > mu0:
        mu0, nu0 = nu0, mu0 
    p_mu0 = random.random()
    p_nu0 = random.random() * (1 - p_mu0)
    return {'mu': mu0, 'nu': nu0, 'p_mu': p_mu0, 'p_nu': p_nu0}

def local_search_optimize(L, Y0, e_d, N, init_params):
    # Initialize with given starting parameters
    current_params = init_params.copy()
    best_rate = compute_key_rate(L, Y0, e_d, N, current_params)
    # Initial step sizes for each parameter (choose reasonably small fractions of range)
    step = {'mu': 0.05, 'nu': 0.05, 'p_mu': 0.05, 'p_nu': 0.05}
    improved = True
    # Iteratively descend along each coordinate
    while improved:
        improved = False
        for param in ['mu', 'nu', 'p_mu', 'p_nu']:
            current_value = current_params[param]
            for direction in [+1, -1]:  # try both increasing and decreasing
                new_value = current_value + direction * step[param]
                # Bound the new value within [0,1]
                if param in ['mu','nu']:
                    new_value = max(0.0, min(1.0, new_value))
                else:  # probability
                    new_value = max(0.0, min(1.0, new_value))
                # If adjusting a probability, ensure p_mu + p_nu <= 1
                new_params = current_params.copy()
                new_params[param] = new_value
                if param in ['p_mu','p_nu']:
                    # enforce the leftover probability for vacuum >= 0
                    p_mu_val = new_params['p_mu']
                    p_nu_val = new_params['p_nu']
                    if p_mu_val + p_nu_val > 1.0:
                        # skip this adjustment if it violates the probability sum constraint
                        continue
                # Compute key rate for the tweaked parameter set
                new_rate = compute_key_rate(L, Y0, e_d, N, new_params)
                if new_rate > best_rate:
                    # Improvement found: update current best parameters and rate
                    current_params = new_params
                    best_rate = new_rate
                    improved = True
                    # Reset loop to re-check all parameters from beginning
            # end for direction
        # end for each parameter
        if not improved:
            # No improvement found in this iteration, refine step sizes and try again
            # (We reduce step sizes to search finer adjustments)
            for param in step:
                step[param] *= 0.5
            # If step sizes become very small, break the loop
            if step['mu'] < 1e-4:  # threshold for convergence
                break
            improved = True  # continue with smaller steps
    return current_params, best_rate

def find_top_parameters(L, Y0, e_d, N, trials=1000):
    results = []
    for _ in range(trials):
        params0 = random_initial_params()
        best_params, best_rate = local_search_optimize(L, Y0, e_d, N, params0)
        results.append({
            'mu': best_params['mu'],
            'nu': best_params['nu'],
            'p_mu': best_params['p_mu'],
            'p_nu': best_params['p_nu'],
            'key_rate': best_rate
        })
    results.sort(key=lambda x: x['key_rate'], reverse=True)
    return results

L, Y0, e_d, N = 50.0, 1e-5, 0.02, 1e9   # example: 50 km, dark count 1e-5, misalignment 2%, 1e9 signals
all_results = find_top_parameters(L, Y0, e_d, N, trials=200)
top_1000 = all_results[:1000]  # top 1000 sets
print(f"Best key rate found: {top_1000[0]['key_rate']:.2e} at parameters {top_1000[0]}")
print(f"Number of unique parameter sets collected: {len(all_results)}")

df = pd.DataFrame(top_1000)

print(df.head(10))
df.to_json("top1000_params.json", orient="records", indent=2)
