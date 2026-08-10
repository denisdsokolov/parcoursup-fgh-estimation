import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

# ====================================================
# Elsevier-style fonts
# ====================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# ---- Z-axis formatter: integer percentages ----
def percent_formatter_int(x, pos):
    return f"{x:.0f}%"


# ====================================================
# Select and load an annual 20×18 matrix
# ====================================================
parser = argparse.ArgumentParser(description="Render residency histograms for one Parcoursup year.")
parser.add_argument("--year", choices=("22", "23", "24"), default="22")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()

source_dir = Path(__file__).resolve().parent
args.output_dir.mkdir(parents=True, exist_ok=True)
hist3d_data = np.loadtxt(source_dir / f"hist3d_res_matrix_{args.year}.csv",
                         delimiter=",", dtype=int)
hist3d_data = np.delete(hist3d_data, 0, axis=0)  # drop first row

n_i, n_j = hist3d_data.shape
assert n_i == 20, "Expected 20 bins in the 'share of resident wishes' dimension."

total_students = hist3d_data.sum()
hist_pct = 100 * hist3d_data / total_students  # each cell = % of students
dz = hist_pct.ravel()

# ====================================================
# Axis coordinates
# ====================================================
wish_indices = np.arange(1, n_j + 1)

share_l = np.arange(0, 20) * 5
share_r = np.arange(1, 21) * 5
share_mids = (share_l + share_r) / 200.0   # 0.025, 0.075, ...

xpos, ypos = np.meshgrid(wish_indices, share_mids)
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

dx = 0.8
dy = (1 / n_i) * 0.8

# ====================================================
# COLOR MAP WITH EXPONENTIAL FADE TO BLACK AT 3%
# ====================================================
cutoff = 4.0
dzmax = dz.max()

if dzmax <= 0:
    dzmax = 1.0

thr = min(cutoff / dzmax, 1.0)

n = 256
n_grad = max(2, int(thr * n))

t = np.linspace(0.0, 1.0, n_grad)
alpha = 2.0

v = np.exp(-alpha * t)
v = (v - np.exp(-alpha)) / (1 - np.exp(-alpha))   # scale to [1 → 0]

gradient_part = v[:, None]
black_part = np.zeros((n - n_grad, 1))

rgba = np.vstack([
    np.hstack([gradient_part]*3 + [np.ones_like(gradient_part)]),
    np.hstack([black_part]*3     + [np.ones_like(black_part)])
])

custom_cmap = LinearSegmentedColormap.from_list("exp_black_cutoff", rgba)
norm = plt.Normalize(vmin=dz.min(), vmax=dzmax)

mappable = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
mappable.set_array([])

colors = custom_cmap(norm(dz))

# ====================================================
# Views
# ====================================================
angles = [
    (40, 50,  "front"),
    (40, 220, "back"),
    (40, 130, "left"),
    (40, -40, "right"),
    (90, 45,  "top"),
    (10, 40,  "lowangle"),
]

# ====================================================
# PLOT + EXPORT
# ====================================================
for elev, azim, name in angles:

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(
        xpos, ypos, zpos,
        dx, dy, dz,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        shade=False,
    )

    ax.view_init(elev=elev, azim=azim)

    # ----------------------------
    # X axis
    # ----------------------------
    ax.set_xlabel("Number of wishes")
    ax.set_xticks(wish_indices)
    xticklabels = [str(k) for k in range(1, n_j)]
    xticklabels.append(f"{n_j}+")
    ax.set_xticklabels(xticklabels)

    # ----------------------------
    # Y axis
    # ----------------------------
    ytick_positions = share_mids
    ytick_labels = [f"{L}–{R}%" for L, R in zip(share_l, share_r)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel("Share of resident wishes")

    # ----------------------------
    # Z axis
    # ----------------------------
    ax.set_zlabel("")

    if name == "top":
        # No z axis in top view
        ax.set_zticks([])
        ax.zaxis.set_ticklabels([])
        try:
            ax.w_zaxis.line.set_lw(0.)
            ax.zaxis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass
        # smaller x,y ticks
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(5)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(5)

    else:
        ax.zaxis.set_major_formatter(FuncFormatter(percent_formatter_int))
        # smaller x,y,z ticks
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(5)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(5)
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(8)

    # ====================================================
    # COLORBAR (legend)
    # ====================================================
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Share of candidates in bin (%)")

    # Integer ticks: 0,1,2,...,max
    ticks = np.arange(0, int(np.floor(dzmax)) + 1)

    labels = []
    for t in ticks:
        if abs(t - cutoff) < 1e-6:   # t == 3.0
            labels.append("4% (cutoff)")
        else:
            labels.append(f"{t:.0f}%")

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    # Ensure visibility
    cbar.ax.tick_params(labelsize=10, colors="black")

    filename = args.output_dir / f"hist3d_{name}_{args.year}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

    print("Saved:", filename)

# ====================================================
# 2D MARGINALS
# ====================================================

# -----------------------------
# 1) Marginal over share of resident wishes
# -----------------------------
row_sums_pct = hist_pct.sum(axis=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(share_mids, row_sums_pct, width=(1/n_i)*0.8,
       color="0.6", edgecolor="black")

ax.set_xlabel("Share of resident wishes")
ax.set_ylabel("Share of candidates (%)")
#ax.set_title("Marginal distribution by share of resident wishes")

# X ticks
ax.set_xticks(share_mids)
ax.set_xticklabels([f"{L}–{R}%" for L, R in zip(share_l, share_r)],
                   rotation=45, ha="right")

# Y ticks
ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter_int))

# ---- ADD DASHED GRIDLINES ----
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(8)

plt.tight_layout()
share_output = args.output_dir / f"hist_marginal_share_residents_{args.year}.pdf"
plt.savefig(share_output, format="pdf", bbox_inches="tight")
plt.close()
print("Saved 2D:", share_output)


# -----------------------------
# 2) Marginal over number of wishes
# -----------------------------
col_sums_pct = hist_pct.sum(axis=0)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(wish_indices, col_sums_pct, width=0.8,
       color="0.6", edgecolor="black")

ax.set_xlabel("Number of wishes")
ax.set_ylabel("Share of candidates (%)")
#ax.set_title("Marginal distribution by number of wishes")

ax.set_xticks(wish_indices)
xticklabels = [str(k) for k in range(1, n_j)] + [f"{n_j}+"]
ax.set_xticklabels(xticklabels)

ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter_int))

# ---- ADD DASHED GRIDLINES ----
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(8)

plt.tight_layout()
wish_output = args.output_dir / f"hist_marginal_num_wishes_{args.year}.pdf"
plt.savefig(wish_output, format="pdf", bbox_inches="tight")
plt.close()
print("Saved 2D:", wish_output)
