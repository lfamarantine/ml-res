
**Probabilities in Python & R**

`from scipy.stats import binom`

| Distribution | Example | R |  Python | 
| :------------ | :--------- |:---| :----|
|  Binomial(n, p) | X∼Binomial(3,0.2), P(X=0) |`dbinom(x = 0, size = 3, prob = 0.2)  `| `binom.pmf(k=0, n=3, p=0.2)` |
|  | X∼Binomial(3,0.2), P(X≤2) | `pbinom(q = seq(from = 0, to = 2, by = 1), size = 3, prob = 0.2)`| `binom.cdf(k=np.arange(0, 3, step=1), n=3, p=0.2)` |



