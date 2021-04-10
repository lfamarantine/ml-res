import numpy as np
from scipy import stats

def linfit(x, y):
  xh = np.sum(x)/len(x)
  yh = np.sum(y)/len(y)
  m = np.sum((x - xh) * y) / np.sum((x - xh)**2)
  c = yh - m * xh
  return [m, c]

x = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
y = np.array([0.1, 0.25, 0.55, 0.75, 0.85])

re_1 = linfit(x=x, y=y)
re_2 = stats.linregress(x=x, y=y)


