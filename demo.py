import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(2); n = 120
x = rng.uniform(0, 10, n)
y = 2 + 1.5*x + rng.normal(0, 0.4 + 0.25*x, n)   # heteroscedastic noise
y[5] += 12                                        # one outlier
X = np.column_stack([np.ones(n), x])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
fit = X @ beta; resid = y - fit
H = X @ np.linalg.inv(X.T @ X) @ X.T; lev = np.diag(H)
s = resid.std(ddof=2); std_resid = resid / (s * np.sqrt(1 - lev))
theo = np.sort(rng.standard_normal(n))            # Monte-Carlo normal quantiles for QQ
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0,0].scatter(fit, resid, s=12); ax[0,0].axhline(0, c="k", ls="--"); ax[0,0].set_title("Residuals vs fitted"); ax[0,0].set_xlabel("fitted"); ax[0,0].set_ylabel("residual")
ax[0,1].scatter(theo, np.sort(std_resid), s=12); ax[0,1].plot([-3,3],[-3,3],"k--"); ax[0,1].set_title("Normal QQ of residuals"); ax[0,1].set_xlabel("theoretical"); ax[0,1].set_ylabel("observed")
ax[1,0].scatter(fit, np.sqrt(np.abs(std_resid)), s=12); ax[1,0].set_title("Scale-location (heteroscedasticity)"); ax[1,0].set_xlabel("fitted"); ax[1,0].set_ylabel("sqrt|std resid|")
ax[1,1].scatter(lev, std_resid, s=12); ax[1,1].axhline(0, c="k", ls="--"); ax[1,1].set_title("Residuals vs leverage"); ax[1,1].set_xlabel("leverage"); ax[1,1].set_ylabel("std residual")
fig.suptitle("Linear regression diagnostics (demo data)"); fig.tight_layout(); fig.savefig("figures/demo.png", dpi=140)
open("results/summary.csv","w").write(f"intercept,{beta[0]:.3f}\nslope,{beta[1]:.3f}\nmax_leverage,{lev.max():.3f}\n")
print(f"beta={beta.round(2)}"); print("ok")
