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

PYTHIA_DEPTHS = {"14m": 6, "31m": 6, "70m": 6, "160m": 12, "410m": 24, "1b": 16, "1.4b": 24}
PYTHIA_WIDTHS = {"14m": 128, "31m": 256, "70m": 512, "160m": 768, "410m": 1024, "1b": 2048, "1.4b": 2048}

PYTHIA_COLOR = "blue"
RANDOM_COLOR = "orange"


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
        return self.reg.fit(np.log2(xs), ys)

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
pythia_ns = np.array(pythia_ns)

pythia_depths = [PYTHIA_DEPTHS[size] for size in pythia_sizes]
pythia_depths = np.array(pythia_depths)

pythia_widths = [PYTHIA_WIDTHS[size] for size in pythia_sizes]
pythia_widths = np.array(pythia_widths)

depths, depth_ns = get_data(df, DEPTH)
depths = np.array(depths).astype(np.int64)
depth_ns = np.array(depth_ns)

widths, width_ns = get_data(df, WIDTH)
widths = np.array(widths).astype(np.int64)
width_ns = np.array(width_ns)

def regress(X, y) -> LogRegression:
    reg = LogRegression()
    reg.fit(X, y)
    r2 = r2_score(reg.predict(X), y)
    return reg, r2


print ("=== DEPTH REGRESSION ===")
# regress(np.concatenate([depth_ns, pythia_ns]).reshape(-1, 1), np.concatenate([depths, pythia_depths]))
depth_reg, depth_r2 = regress(depth_ns.reshape(-1, 1), depths)
print(f"depth ~ {depth_reg.slope:.2f} * log2(n) + {depth_reg.intercept:.2f}")
print("depth r2:", depth_r2)

plt.figure()
plt.scatter(pythia_depths, pythia_ns, marker=".", label="pythia", color=PYTHIA_COLOR)
plt.scatter(depths, depth_ns, marker=".", label="random", color=RANDOM_COLOR)
domain = np.linspace(depth_ns.min(), depth_ns.max(), 100).reshape(-1, 1)
plt.plot(depth_reg.predict(domain), domain, label=f"fit (r2={depth_r2:.2f})", color=RANDOM_COLOR)
plt.yscale("log")
plt.xticks(list(sorted(depths)))
plt.xlabel("depth")
plt.ylabel("context length")
plt.legend()
plt.savefig("plots/depth.png")

print ("=== WIDTH REGRESSION ===")
width_reg, width_r2 = regress(widths.reshape(-1, 1), width_ns)
print(f"n ~ {width_reg.slope:.2f} * log2(width) + {width_reg.intercept:.2f}")
print("depth r2:", width_r2)

plt.figure()
plt.scatter(pythia_widths, pythia_ns, marker=".", label="pythia", color=PYTHIA_COLOR)
plt.scatter(widths, width_ns, marker=".", label="random", color=RANDOM_COLOR)
domain = np.linspace(widths.min(), widths.max(), 100).reshape(-1, 1)
plt.plot(domain, width_reg.predict(domain), label=f"fit (r2={width_r2:.2f})", color=RANDOM_COLOR)
plt.xscale("log")
plt.xlabel("width")
plt.ylabel("context length")
plt.legend()
plt.savefig("plots/width.png")
