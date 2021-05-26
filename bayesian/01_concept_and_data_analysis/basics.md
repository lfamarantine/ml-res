
#### Probabilities in Python & R

`from scipy.stats import binom, norm, t, uniform, expon, beta, gamma, chisquare` \
`import numpy as np`

*Example*

Assuming X∼Binomial(n=3,p=0.2), we can calculate probabilities as follows in R/Python:


| Event | R | Python |
| :----- | :---- | :------ |
| P(X=0) | `dbinom(x = 0, size = 3, prob = 0.2)` | `binom.pmf(k=0, n=3, p=0.2)` |
| P(X≤2) | `pbinom(q = seq(from = 0, to = 2, by = 1), size = 3, prob = 0.2)` | `binom.cdf(k=np.arange(0, 3, step=1), n=3, p=0.2)` |


*Common Probability Distributions in Python & R*

| Distribution | Tool | PDF | CDF | Generating pseudo-random samples |
| ------------ | ---- | :---: | :---: | :--------------------------------: |
| Binomial | R | `dbinom` | `pbinom` | `rbinom` |
|  | Python | `binom.pmf` | `binom.cdf` | `binom.rvs` |
| Normal | R | `dnorm` | `pnorm` | `rnorm` |
|  | Python | `norm.pdf` | `norm.cdf` | `norm.rvs` |
| Student t | R | `dt` | `pt` | `rt` |
|  | Python | `t.pdf` | `t.cdf` | `t.rvs` |
| Uniform | R | `dunif` | `punif` | `runif` |
|  | Python | `uniform.pdf` | `uniform.cdf` | `uniform.rvs` |
| Exponential | R | `dexp` | `pexp` | `rexp` |
|  | Python | `expon.pdf` | `expon.cdf` | `expon.rvs` |
| Beta | R | `dbeta` | `pbeta` | `rbeta` |
|  | Python | `beta.pdf` | `beta.cdf` | `beta.rvs` |
| Gamma | R | `dgamma` | `pgamma` | `rgamma` |
|  | Python | `gamma.pdf` | `gamma.cdf` | `gamma.rvs` |
| Chi-Square | R | `dchisq` | `pchisq` | `rchisq` |
|  | Python | `chisquare.pdf` | `chisquare.cdf` | `chisquare.rvs` |


