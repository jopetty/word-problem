"""Will's plotting script for log-depth transformers experiments."""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

PYTHIA = r"Group: pythia-([0-9]+[mb]) - eval_n@0.05__MAX"
DEPTH = r"Group: sfirah-d([0-9]+) - eval_n@0.05__MAX"
WIDTH = r"Group: sfirah-w([0-9]+) - eval_n@0.05__MAX"

PYTHIA_DEPTHS = {"14m": 6, "31m": 6, "70m": 6, "160m": 12}
PYTHIA_WIDTHS = {"14m": 128, "31m": 256, "70m": 512, "160m": 768}


def get_data(df, pattern) -> tuple[list[str], list[int]]:
    xs = []
    ys = []
    for key in df.keys():
        match = re.match(pattern, key)
        if match:
            xs.append(match.group(1))
            ys.append(df[key].max())
    return xs, ys


class LogRegression:
    def __init__(self):
        self.reg = LinearRegression()

    def fit(self, xs, ys):
        self.reg.fit(np.log2(xs), ys)

    def predict(self, xs):
        return self.reg.predict(np.log2(xs))
    
    @property
    def slope(self):
        return self.reg.coef_.item()
    
    @property
    def intercept(self):
        return self.reg.intercept_


df = pd.read_csv("data/log-depth-clean.csv")

pythia_sizes, pythia_ns = get_data(df, PYTHIA)
pythia_ns = np.array(pythia_ns).reshape(-1, 1)
breakpoint()

pythia_depths = [PYTHIA_DEPTHS[size] for size in pythia_sizes]
pythia_depths = np.array(pythia_depths)

pythia_widths = [PYTHIA_WIDTHS[size] for size in pythia_sizes]
pythia_widths = np.array(pythia_widths)

depths, depth_ns = get_data(df, DEPTH)
depths = np.array(depths)
depth_ns = np.array(depth_ns).reshape(-1, 1)

widths, width_ns = get_data(df, WIDTH)

reg = LogRegression()
X = np.concatenate([depth_ns, pythia_ns])
y = np.concatenate([depths, pythia_depths])
reg.fit(X, y)
print(f"depth ~ {reg.slope:.2f} * log2(n) + {reg.intercept:.2f}")
print("depth r2:", r2_score(reg.predict(X), y))

plt.figure()
depths = [int(depth) for depth in depths]
plt.scatter(pythia_depths, pythia_ns, marker=".", label="pythia")
plt.scatter(depths, depth_ns, marker=".", label="random")
domain = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(reg.predict(domain), domain, label="fit")
plt.yscale("log")
plt.xticks(list(sorted(depths)))
plt.xlabel("depth")
plt.ylabel("context length")
plt.legend()
plt.savefig("plots/depth.png")

plt.figure()
widths = [int(width) for width in widths]
plt.scatter(pythia_widths, pythia_ns, marker=".", label="pythia")
plt.scatter(widths, width_ns, marker=".", label="random")
plt.xscale("log")
plt.xlabel("width")
plt.ylabel("context length")
plt.legend()
plt.savefig("plots/width.png")
