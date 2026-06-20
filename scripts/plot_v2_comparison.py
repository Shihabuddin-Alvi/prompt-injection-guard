"""Generate v1 vs v2 macro F1 comparison chart for LinkedIn Post 2."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Standard test set (n=1,754) — saturated, near-identical
test_categories = ["Macro F1\n(split_test, n=1,754)"]
v1_test = [0.9957]
v2_test = [0.9932]

# Adversarial slice (n=651, injection-heavy) — where v2's targeting shows up
adv_categories = ["Benign recall", "Injection recall", "Benign F1"]
v2_adv = [1.00, 0.99, 0.85]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 2]})

# Left panel: v1 vs v2 on saturated standard test
ax1 = axes[0]
x = [0, 1]
bars1 = ax1.bar(x, [v1_test[0], v2_test[0]], color=["#4a5568", "#2b6cb0"], width=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(["v1", "v2"])
ax1.set_ylim(0.98, 1.0)
ax1.set_title("Standard test set\n(saturated, no signal)", fontsize=10)
ax1.set_ylabel("Macro F1")
for bar, val in zip(bars1, [v1_test[0], v2_test[0]]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.0005,
        f"{val:.4f}",
        ha="center",
        fontsize=9,
    )

# Right panel: v2 performance on adversarial slice
ax2 = axes[1]
y = range(len(adv_categories))
bars2 = ax2.barh(y, v2_adv, color="#2b6cb0", height=0.5)
ax2.set_yticks(y)
ax2.set_yticklabels(adv_categories)
ax2.set_xlim(0, 1.05)
ax2.set_title("v2 on adversarial slice\n(n=651, injection-heavy)", fontsize=10)
ax2.set_xlabel("Score")
for bar, val in zip(bars2, v2_adv):
    ax2.text(
        val + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}",
        va="center",
        fontsize=9,
    )

plt.suptitle("Prompt Injection Guard: v1 vs v2", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/v1_vs_v2_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to assets/v1_vs_v2_comparison.png")
