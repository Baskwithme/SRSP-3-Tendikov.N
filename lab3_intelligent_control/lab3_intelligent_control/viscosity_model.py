"""Прогноз вязкости ЛКМ: линейная регрессия и случайный лес (синтетические данные)."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mu(T, n, t, C):
    return 900*np.exp(-0.035*(T-25))*(C/50)**2*(300/n)**0.25*(1+0.5*np.exp(-t/8))

rng = np.random.default_rng(42)
N = 300
df = pd.DataFrame({"T": rng.uniform(20, 50, N), "n": rng.uniform(100, 500, N),
                   "t": rng.uniform(2, 20, N), "C": rng.uniform(40, 60, N)})
df["mu"] = mu(df["T"], df["n"], df["t"], df["C"])*np.exp(0.04*rng.standard_normal(N))
df.to_csv("dataset.csv", index=False)
print(df.describe().round(1)); print(df.corr().round(2)["mu"])

X, y = df[["T", "n", "t", "C"]], df["mu"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
lr = LinearRegression().fit(np.c_[Xtr["T"], np.log(Xtr["n"]), Xtr["t"], np.log(Xtr["C"])], np.log(ytr))
p_lr = np.exp(lr.predict(np.c_[Xte["T"], np.log(Xte["n"]), Xte["t"], np.log(Xte["C"])]))
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=1).fit(Xtr, ytr)
p_rf = rf.predict(Xte)
for name, p in [("Linear regression", p_lr), ("Random forest", p_rf)]:
    print(f"{name}: MAE={mean_absolute_error(yte, p):.1f} RMSE={mean_squared_error(yte, p)**0.5:.1f} "
          f"R2={r2_score(yte, p):.3f} MAPE={np.mean(np.abs(p-yte)/yte)*100:.1f}%")
print(pd.DataFrame({"fact": yte.values[:8], "LR": p_lr[:8], "RF": p_rf[:8]}).round(0))
