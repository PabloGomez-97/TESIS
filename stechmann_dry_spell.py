import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Parameters (consistent with notebook)
E_star = 0.4          # mm/h (drift in dry state)
D0_var = 8.0          # D0^2 (mm^2/h)
D0 = np.sqrt(D0_var)
qc = 65.0
q0 = 63.0
b = qc - q0          # distance to threshold

# Simulation settings
N = 12000            # number of independent trajectories
dt_minutes = 0.1
dt = dt_minutes / 60.0   # hours
max_steps = int(2e6)     # safety cap
max_time = 2000.0        # max time in hours (stop condition)
max_steps = int(max_time / dt)

rng = np.random.default_rng(0)


def simulate_batch(N, dt, E_star, D0, q0, qc, floor_mode=None, max_steps=1000000):
    """Vectorized simulation of first-passage times for dq = E_star dt + D0 dW.
    floor_mode: None -> allow negatives; 'clip'/'reflect'/'stay' implement treatments when q<0.
    Returns array of hitting times (hours) for those trajectories that hit within max_steps.
    """
    q = np.full(N, q0, dtype=float)
    alive = np.ones(N, dtype=bool)
    hit_times = np.full(N, np.nan)
    dt_sqrt = np.sqrt(dt)

    for step in range(max_steps):
        if not alive.any():
            break
        n_alive = alive.sum()
        dW = rng.normal(loc=0.0, scale=dt_sqrt, size=n_alive)
        idx = np.nonzero(alive)[0]
        q[idx] += E_star * dt + D0 * dW

        if floor_mode is not None:
            if floor_mode == 'clip':
                q[idx] = np.maximum(q[idx], 0.0)
            elif floor_mode == 'reflect':
                # reflect negatives: q -> |q|
                neg = q[idx] < 0.0
                if neg.any():
                    q[idx[neg]] = -q[idx[neg]]
            elif floor_mode == 'stay':
                # any negative reverts to previous value -> approximate by not applying update where q would be negative
                # For vectorized code we need previous values; we approximate by stepping per-index
                # Simpler: do per-index correction
                for i in idx:
                    if q[i] < 0.0:
                        # revert the last increment
                        q[i] -= (E_star * dt)
                        # remove the stochastic increment by re-drawing until non-negative or keep previous
                        q[i] = q0 if True else q[i]
                # Note: 'stay' is approximated as reverting to q0 (coarse)
            else:
                raise ValueError('Unknown floor_mode')

        # check hits
        hit_mask = alive & (q >= qc)
        if hit_mask.any():
            hit_indices = np.nonzero(hit_mask)[0]
            hit_times[hit_indices] = (step + 1) * dt
            alive[hit_indices] = False

    return hit_times[~np.isnan(hit_times)]


# Analytical PDF (eq.6 form):
# f(t) = b / sqrt(2*pi*D0^2 * t^3) * exp(E_star*b/D0^2) * exp(-b^2/(2*D0^2 * t)) * exp(-E_star^2 * t/(2*D0^2))
from numpy import exp

def analytic_pdf(t, b, E_star, D0):
    pref = b / np.sqrt(2.0 * np.pi * (D0**2) * (t**3))
    return pref * np.exp(E_star * b / (D0**2)) * np.exp(-b**2 / (2.0 * (D0**2) * t)) * np.exp(- (E_star**2) * t / (2.0 * (D0**2)))


if __name__ == '__main__':
    print('Parameters: E_star=', E_star, 'D0=', D0, 'b=', b)

    # Variant A: allow negative q (no floor)
    print('Simulating variant A: allow negative q...')
    times_A = simulate_batch(N, dt, E_star, D0, q0, qc, floor_mode=None, max_steps=max_steps)
    print('Collected', len(times_A), 'first-passage events (variant A)')

    # Variant B: reflect at 0
    print('Simulating variant B: reflect q at 0...')
    times_B = simulate_batch(N, dt, E_star, D0, q0, qc, floor_mode='reflect', max_steps=max_steps)
    print('Collected', len(times_B), 'first-passage events (variant B)')

    # Prepare plotting
    plt.figure(figsize=(8,5))

    # log-binning
    all_times = [times_A, times_B]
    labels = ['allow negative', 'reflect at 0']
    colors = ['C0', 'C1']

    t_min = min(np.nanmin(t) for t in all_times)
    t_max = max(np.nanmax(t) for t in all_times)
    bins = np.logspace(np.log10(t_min*0.9), np.log10(t_max*1.1), 40)

    for times, label, c in zip(all_times, labels, colors):
        counts, edges = np.histogram(times, bins=bins)
        widths = np.diff(edges)
        centers = np.sqrt(edges[:-1] * edges[1:])
        pdf = counts / (len(times) * widths)
        plt.loglog(centers, pdf, 'o', label=f'data ({label})', color=c)

    # Analytical curve
    t_fit = np.logspace(np.log10(t_min*0.9), np.log10(t_max*1.1), 400)
    p_fit = analytic_pdf(t_fit, b, E_star, D0)
    plt.loglog(t_fit, p_fit, '-', color='k', label='analytic eq.6')

    plt.xlabel('t (h)')
    plt.ylabel('p(t)')
    plt.legend()
    plt.title('Dry spell duration distribution: numeric vs analytic')
    plt.grid(True, which='both', ls=':')
    plt.savefig('dry_spell_comparison.png', dpi=150)
    print('Saved figure dry_spell_comparison.png')

    # Basic stats
    for times, label in zip(all_times, labels):
        print(f"{label}: mean={np.mean(times):.3f} h, median={np.median(times):.3f} h, n={len(times)}")

    # Save durations
    np.savez_compressed('dry_spell_times_A_B.npz', times_A=times_A, times_B=times_B)
    print('Saved durations to dry_spell_times_A_B.npz')
