import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

categories = [
    "role_play",
    "unknown",
    "indirect",
    "system_prompt_leak",
    "jailbreak",
    "direct",
]
f1_scores = [0.9644, 0.9962, 1.0000, 1.0000, 1.0000, 1.0000]
counts = [57, 1566, 1, 12, 26, 92]
colors = ["#1A6B3C" if f >= 0.98 else "#C0622B" for f in f1_scores]

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#F4F2EE")

bars = ax.barh(categories, f1_scores, color=colors, height=0.55, zorder=2)

ax.set_xlim(0.955, 1.008)
ax.set_xlabel("Macro F1", fontsize=11, color="#555250", labelpad=10)
ax.set_title(
    "Prompt Injection Guard v1  —  Per-Category F1\n"
    "DeBERTa-v3-base  ·  Test set n=1,754  ·  Macro F1: 0.9957  (95% CI: 0.9924–0.9982)",
    fontsize=11,
    color="#111111",
    pad=16,
    loc="left",
)

ax.tick_params(colors="#777470", labelsize=10)
ax.xaxis.label.set_color("#555250")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax.grid(axis="x", color="#E4E1D9", linewidth=0.8, zorder=1)

for bar, score, count, cat in zip(bars, f1_scores, counts, categories):
    label = f"{score:.4f}   n={count:,}"
    if cat == "role_play":
        label += "   ← lowest, needs more data"
    ax.text(
        score + 0.0005,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center",
        ha="left",
        fontsize=9.5,
        color="#C0622B" if score < 0.98 else "#1A6B3C",
    )

green_patch = mpatches.Patch(color="#1A6B3C", label="F1 ≥ 0.98")
orange_patch = mpatches.Patch(color="#C0622B", label="F1 < 0.98")
ax.legend(
    handles=[green_patch, orange_patch],
    loc="upper right",
    bbox_to_anchor=(1.0, 1.15),
    fontsize=9.5,
    frameon=True,
    framealpha=1,
    edgecolor="#E4E1D9",
)

fig.text(
    0.98,
    0.02,
    "Overall Macro F1: 0.9957",
    ha="right",
    va="bottom",
    fontsize=10,
    color="#111111",
    fontweight="semibold",
)

plt.tight_layout(pad=1.5)
plt.savefig(
    "assets/v1_per_category_f1.png", dpi=150, bbox_inches="tight", facecolor="#FFFFFF"
)
print("Saved to assets/v1_per_category_f1.png")
