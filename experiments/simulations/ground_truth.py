import numpy as np
import matplotlib.pyplot as plt
from math import lgamma, exp

def g(x1, x2): return x1 * x2**2

def micro_shares(m):
    # u={1,2}, x0=(0,0), x1=(1,1)
    g_t = np.array([[g(p1/m, p2/m) for p2 in range(m+1)]
                     for p1 in range(m+1)])
    r_t = np.zeros((m+1, m+1))
    for p1 in range(m+1):
        for p2 in range(m+1):
            r_t[p1,p2] = (g_t[p1,p2] - g_t[0,p2]
                          - g_t[p1,0] + g_t[0,0])
    n = 2*m
    def lc(n,k):
        if k<0 or k>n: return float('-inf')
        return lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
    logC = [lc(m,j) for j in range(m+1)]
    shares = []
    for i_idx in range(2):
        acc = 0.0
        for p1 in range(m+1):
            for p2 in range(m+1):
                p = [p1,p2]
                if p[i_idx] >= m: continue
                ps = p1+p2
                lw = lgamma(ps+1)+lgamma(n-ps)-lgamma(n+1)
                lm = logC[p1]+logC[p2]+np.log(m-p[i_idx])
                pn = list(p); pn[i_idx]+=1
                delta = r_t[tuple(pn)] - r_t[tuple(p)]
                acc += exp(lw+lm)*delta
        shares.append(acc)
    s = sum(shares)
    if abs(s)>1e-12:
        shares = [x*r_t[m,m]/s for x in shares]
    return shares

M_VALUES = [1,2,3,4,5,6,8,10,15,20,30,50]
s1 = [micro_shares(m)[0] for m in M_VALUES]
s2 = [micro_shares(m)[1] for m in M_VALUES]

fig, ax = plt.subplots(figsize=(4.5, 2.8))
ax.plot(M_VALUES, s1, marker='o', markersize=4,
        color='#1f77b4', linewidth=1.8, label=r'$x_1$ (micro-game)')
ax.plot(M_VALUES, s2, marker='s', markersize=4,
        color='#2ca02c', linewidth=1.8, label=r'$x_2$ (micro-game)')
ax.axhline(1/3, color='#1f77b4', linestyle='--',
           linewidth=1.2, label=r'Ground truth $x_1=1/3$')
ax.axhline(2/3, color='#2ca02c', linestyle='--',
           linewidth=1.2, label=r'Ground truth $x_2=2/3$')
ax.axhline(0.5, color='#d62728', linestyle=':',
           linewidth=1.5, label='Equal-split (constant bias)')
ax.set_xlabel('Grid resolution $m$', fontsize=9)
ax.set_ylabel('Within-pot share', fontsize=9)
ax.set_title(r'Convergence to ground truth: $g=x_1x_2^2+x_3$,'
             r' $u=\{1,2\}$', fontsize=9)
ax.legend(fontsize=7.5, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('convergence_groundtruth.pdf', dpi=200, bbox_inches='tight')
print("Saved convergence_groundtruth.pdf")
plt.show()