# Methodology

The derivations behind every technique used in this project. Notation follows
the code: `S0` spot, `K` strike, `r` risk-free rate, `q` dividend yield,
`sigma` volatility, `T` maturity in years, `N` paths.

---

## 1. Model and risk-neutral valuation

Under the risk-neutral measure $\mathbb{Q}$, the underlying follows geometric
Brownian motion:

$$dS_t = (r - q)\,S_t\,dt + \sigma S_t\,dW_t$$

This has a known solution, which matters a great deal for the implementation:

$$S_t = S_0 \exp\!\left[\left(r - q - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right]$$

The price of a claim with payoff $g(\cdot)$ is its discounted expectation:

$$V = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[g(S)\right]$$

Monte Carlo estimates that expectation by sampling. Everything below is about
doing so with the least noise per unit of computation.

---

## 2. The estimator and its error

Drawing $N$ independent paths gives

$$\hat{V}_N = \frac{1}{N}\sum_{i=1}^{N} e^{-rT} g\!\left(S^{(i)}\right)$$

which is unbiased. By the central limit theorem,

$$\sqrt{N}\left(\hat{V}_N - V\right) \xrightarrow{d} \mathcal{N}\!\left(0, \sigma_g^2\right)$$

so the standard error is estimated by $s/\sqrt{N}$ with $s$ the sample standard
deviation of the discounted payoffs, and the 95% interval is
$\hat{V}_N \pm 1.96\,s/\sqrt{N}$.

**This is the number most implementations omit.** Without it, a price quoted to
four decimal places says nothing about whether any of those decimals are real.
`PriceEstimate` in `mcopt/engine.py` therefore carries `std_error` alongside
`value`, and `sigmas_from()` reports how many standard errors a benchmark sits
away — under 2 means indistinguishable from noise.

The $1/\sqrt{N}$ rate is the method's defining limitation: one additional
decimal digit of accuracy costs **one hundred times** the computation.
`convergence.py` recovers this slope empirically rather than quoting it; the
measured value is −0.4886.

---

## 3. Why European payoffs need no path grid

Because the GBM solution above is exact, the terminal value can be drawn in a
single step:

$$S_T = S_0 \exp\!\left[\left(r - q - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \qquad Z \sim \mathcal{N}(0,1)$$

There is no Euler discretisation and therefore no discretisation bias — this is
not an approximation to a finer grid, it is the exact distribution. A European
payoff depends only on $S_T$, so simulating 252 intermediate steps performs 252
times the random-number generation for no gain.

The path grid is built only where the payoff genuinely requires it: Asian
options (which average over fixings) and barrier options (which monitor for a
breach).

---

## 4. Antithetic variates

For each draw $Z$, also use $-Z$, and average within the pair:

$$Y_i = \tfrac{1}{2}\left[g(Z_i) + g(-Z_i)\right]$$

Writing $\rho = \operatorname{Corr}\!\left(g(Z), g(-Z)\right)$,

$$\operatorname{Var}(Y) = \tfrac{1}{2}\operatorname{Var}(g)\,(1 + \rho)$$

With $N = 2m$ total draws arranged in $m$ pairs, the estimator variance is
$\operatorname{Var}(g)(1+\rho)/N$ against $\operatorname{Var}(g)/N$ for plain
Monte Carlo. The technique helps precisely when $\rho < 0$, which is guaranteed
whenever $g$ is monotone in $Z$ — true for vanilla calls and puts.

### The trap

The standard error **must** be computed across the $m$ pair averages, not the
$N$ raw samples. Taking $s/\sqrt{N}$ over all draws estimates
$\operatorname{Var}(g)/N$, silently discarding the $(1+\rho)$ factor that is the
entire point of the method — and so misstates the error.

`_summarise()` in `mcopt/engine.py` collapses pairs before computing the error,
which is why the reported reduction (1.4x here) is real rather than an artefact.

---

## 5. Control variates

Let $X$ be the discounted payoff, and $W$ a correlated quantity whose
expectation $\mu_W$ is known exactly. For any constant $c$,

$$Y = X - c\left(W - \mu_W\right)$$

satisfies $\mathbb{E}[Y] = \mathbb{E}[X]$, so the estimator stays unbiased.
Its variance,

$$\operatorname{Var}(Y) = \operatorname{Var}(X) - 2c\operatorname{Cov}(X,W) + c^2\operatorname{Var}(W)$$

is minimised at

$$c^{*} = \frac{\operatorname{Cov}(X,W)}{\operatorname{Var}(W)}, \qquad \operatorname{Var}(Y^{*}) = \operatorname{Var}(X)\left(1 - \rho_{XW}^{2}\right)$$

The reduction depends entirely on the correlation. At $\rho = 0.9$ the variance
falls by 81%; at $\rho = 0.999$ it falls by 99.8%.

| Payoff | Control $W$ | Known mean $\mu_W$ |
|---|---|---|
| European | $S_T$ | $S_0 e^{(r-q)T}$ |
| Arithmetic Asian | discounted **geometric** Asian payoff | closed form, §6 |

### Honest caveat

$c^{*}$ is estimated from the same sample used for the price, which introduces
an $O(1/N)$ bias. This is standard practice and negligible at the path counts
used here, but it is an approximation and is flagged in the code rather than
hidden. A pilot run to estimate $c^{*}$ independently would remove it.

---

## 6. The geometric Asian closed form

This is the most useful result in the project, because it turns an
unsolvable problem into a nearly-solved one.

The **arithmetic** average of lognormal variables is not lognormal and has no
closed form — which is exactly why Monte Carlo is needed. The **geometric**
average is lognormal, because the log of a geometric mean is an arithmetic mean
of normals.

With equally spaced fixings $t_i = iT/n$ for $i = 1,\dots,n$ and
$G = \left(\prod_i S_{t_i}\right)^{1/n}$:

$$\ln G = \ln S_0 + \left(r - q - \tfrac{1}{2}\sigma^2\right)\bar{t} + \frac{\sigma}{n}\sum_{i=1}^{n} W_{t_i}, \qquad \bar{t} = \frac{1}{n}\sum_i t_i$$

Since $\operatorname{Cov}(W_{t_i}, W_{t_j}) = \min(t_i, t_j)$, we have
$\ln G \sim \mathcal{N}(\mu, s^2)$ with

$$\mu = \ln S_0 + \left(r - q - \tfrac{1}{2}\sigma^2\right)\bar{t}, \qquad s^2 = \frac{\sigma^2}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\min(t_i, t_j)$$

Applying the standard lognormal expectation gives a Black-Scholes-shaped result:

$$V_{\text{geo}} = e^{-rT}\left[e^{\mu + s^2/2}\,\Phi(d_1) - K\,\Phi(d_2)\right], \qquad d_1 = \frac{\mu - \ln K + s^2}{s}, \quad d_2 = d_1 - s$$

The double sum is computed directly via `np.minimum.outer` rather than through a
closed-form simplification, because the simplification is easy to get subtly
wrong and the cost is negligible.

Two consequences used elsewhere:

- **AM ≥ GM pointwise**, so the arithmetic Asian is always worth more than the
  geometric one. This is asserted as a test.
- The two payoffs are correlated at roughly 0.999, making $V_{\text{geo}}$ an
  outstanding control variate — measured error reduction **34.5x**, equivalent
  to about 1,200x the path count.

---

## 7. Common random numbers for Greeks

A central-difference delta is

$$\hat{\Delta} = \frac{\hat{V}(S_0 + h) - \hat{V}(S_0 - h)}{2h}$$

With **independent** samples for the two valuations, the numerator has variance
$2\operatorname{Var}(\hat{V})$, so

$$\operatorname{Var}(\hat{\Delta}) = \frac{\operatorname{Var}(\hat{V})}{2h^2}$$

which diverges as $h \to 0$. The bias-variance trade-off is brutal: a small $h$
gives a good finite-difference approximation and an unusable estimate.

Reusing the **same** normal draws for both valuations makes the numerator a
difference of two highly correlated quantities. For payoffs that are Lipschitz
in $S_0$ the difference is $O(h)$, so $\operatorname{Var}(\hat{\Delta})$ is
$O(1)$ rather than $O(h^{-2})$.

In `mc_greeks_fd()` this is implemented by passing the identical seed into every
bumped valuation. The measured agreement with analytic Greeks is $3.7\times10^{-4}$
on delta and $6.9\times10^{-6}$ on gamma.

---

## 8. Discrete monitoring bias in barrier options

The simulation checks the barrier only at grid points, so it can miss a breach
that occurred between two of them. Since

$$\max_{i} S_{t_i} \le \max_{t \in [0,T]} S_t$$

a discretely monitored knock-out survives at least as often as a continuously
monitored one, and is therefore **systematically overpriced**. Increasing the
monitoring frequency reduces the bias at rate $O(1/\sqrt{m})$ but never removes
it.

The Broadie–Glasserman–Kou continuity correction addresses this by shifting the
barrier,

$$H_{\text{adj}} = H \exp\!\left(\pm 0.5826\,\sigma\sqrt{T/m}\right)$$

with a positive sign for up barriers and negative for down barriers, where
$0.5826 \approx -\zeta(1/2)/\sqrt{2\pi}$. **It is not implemented here** — the
bias is documented instead, which is the honest option when the correction has
not been validated.

---

## References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer.
- Kemna, A. & Vorst, A. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, 14(1), 113–129.
- Broadie, M., Glasserman, P. & Kou, S. (1997). A continuity correction for discrete barrier options. *Mathematical Finance*, 7(4), 325–349.
- Boyle, P., Broadie, M. & Glasserman, P. (1997). Monte Carlo methods for security pricing. *Journal of Economic Dynamics and Control*, 21(8–9), 1267–1321.
