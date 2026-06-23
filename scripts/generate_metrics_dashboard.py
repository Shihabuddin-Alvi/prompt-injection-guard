import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(14, 8))
fig.suptitle(
    "Prompt Injection Guard — Model & Serving Metrics",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# ── Panel 1: v1 vs v2 Macro F1 ──────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
models = [
    "v1\n(standard test)",
    "v2\n(standard test)",
    "v1\n(adversarial)",
    "v2\n(adversarial)",
]
f1_scores = [0.9957, 0.9932, 0.94, 0.92]
colors = ["#2196F3", "#4CAF50", "#2196F3", "#4CAF50"]
bars = ax1.bar(models, f1_scores, color=colors, width=0.5)
ax1.set_ylim(0.88, 1.01)
ax1.set_ylabel("Macro F1")
ax1.set_title("v1 vs v2 Macro F1", fontweight="bold")
ax1.axhline(y=0.85, color="red", linestyle="--", linewidth=1, label="Gate threshold")
for bar, score in zip(bars, f1_scores):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
v1_patch = mpatches.Patch(color="#2196F3", label="v1")
v2_patch = mpatches.Patch(color="#4CAF50", label="v2")
ax1.legend(handles=[v1_patch, v2_patch], fontsize=8)

# ── Panel 2: Latency ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
backends = ["PyTorch\n(CPU)", "ONNX int8\n(CPU)"]
single_req = [116, 66]
p99 = [None, 91]
x = np.arange(len(backends))
ax2.bar(x - 0.2, single_req, width=0.35, label="Single req (warm)", color="#FF9800")
ax2.text(1 + 0.2, 91 + 1, "91ms", ha="center", va="bottom", fontsize=8)
ax2.axhline(y=100, color="red", linestyle="--", linewidth=1, label="100ms gate")
ax2.set_xticks(x)
ax2.set_xticklabels(backends)
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Inference Latency", fontweight="bold")
ax2.legend(fontsize=8)
for i, v in enumerate(single_req):
    ax2.text(i - 0.2, v + 1, f"{v}ms", ha="center", va="bottom", fontsize=8)
ax2.text(1 + 0.2, 92, "91ms", ha="center", va="bottom", fontsize=8)

# ── Panel 3: v2 Per-Class Metrics ────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
classes = ["Benign", "Injection"]
precision = [0.99, 0.99]
recall = [1.00, 0.9904]
f1 = [1.00, 0.99]
x = np.arange(len(classes))
ax3.bar(x - 0.25, precision, width=0.25, label="Precision", color="#9C27B0")
ax3.bar(x, recall, width=0.25, label="Recall", color="#00BCD4")
ax3.bar(x + 0.25, f1, width=0.25, label="F1", color="#8BC34A")
ax3.set_ylim(0.97, 1.01)
ax3.set_xticks(x)
ax3.set_xticklabels(classes)
ax3.set_ylabel("Score")
ax3.set_title("v2 Per-Class Metrics\n(standard test, n=1754)", fontweight="bold")
ax3.legend(fontsize=8)

# ── Panel 4: Training Loss Curve ─────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
epochs = [1, 2, 3]
train_loss = [0.098, 0.018, 0.0093]
val_loss = [0.0648, 0.0378, 0.0446]
ax4.plot(epochs, train_loss, "o-", color="#2196F3", label="Train loss")
ax4.plot(epochs, val_loss, "s--", color="#F44336", label="Val loss")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.set_title("v2 Training Loss Curve", fontweight="bold")
ax4.set_xticks(epochs)
ax4.legend(fontsize=8)

# ── Panel 5: Adversarial Slice Detail ────────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
metrics = [
    "Benign\nPrecision",
    "Benign\nRecall",
    "Injection\nPrecision",
    "Injection\nRecall",
]
v1_scores = [0.83, 0.95, 1.00, 0.99]
v2_scores = [0.74, 1.00, 1.00, 0.99]
x = np.arange(len(metrics))
ax5.bar(x - 0.2, v1_scores, width=0.35, label="v1", color="#2196F3")
ax5.bar(x + 0.2, v2_scores, width=0.35, label="v2", color="#4CAF50")
ax5.set_ylim(0.65, 1.05)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics, fontsize=8)
ax5.set_ylabel("Score")
ax5.set_title("Adversarial Slice\n(n=651)", fontweight="bold")
ax5.legend(fontsize=8)

# ── Panel 6: Summary Stats ────────────────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
summary = [
    ["Metric", "Value"],
    ["v1 Macro F1", "0.9957"],
    ["v2 Macro F1", "0.9932"],
    ["Training examples", "9,870"],
    ["Synthetic examples", "1,687"],
    ["ONNX p99 latency", "91ms"],
    ["ONNX speedup vs PyTorch", "43%"],
    ["Test set size", "1,754"],
]
table = ax6.table(
    cellText=summary[1:], colLabels=summary[0], loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)
ax6.set_title("Summary", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("assets/metrics_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved assets/metrics_dashboard.png")
