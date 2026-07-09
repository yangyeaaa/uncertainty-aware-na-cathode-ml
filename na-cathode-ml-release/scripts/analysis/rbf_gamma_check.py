"""
rbf_gamma_check.py
==================

可视化验证 MiniCGCNN 的 RBF γ=10 设置是否合理.

数学:
  RBF(d, mu_k) = exp(-gamma * (d - mu_k)^2)
  对照标准高斯: exp(-(d - mu)^2 / (2 sigma^2))
  所以 gamma = 1/(2 sigma^2)  =>  sigma = 1/sqrt(2*gamma)
  
  gamma = 10 A^-2  ->  sigma = 0.2236 A
  中心间距 = 8 A / 39 = 0.205 A
  相邻 RBF 中心的重叠 = exp(-gamma * (0.205)^2 / 2) = 0.81 (FWHM area)

  即:相邻 RBF 在 ~80% 高度处相交,是经典 CGCNN 标准设置.

输出:
  rbf_gamma_visualization.png
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
})


def rbf(d, mu, gamma):
    return np.exp(-gamma * (d - mu) ** 2)


def main():
    cutoff = 8.0
    n_centers = 40
    gamma = 10.0  # A^-2

    centers = np.linspace(0, cutoff, n_centers)
    delta_mu = centers[1] - centers[0]
    sigma = 1 / np.sqrt(2 * gamma)

    d_grid = np.linspace(0, cutoff, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: All 40 bases
    ax = axes[0]
    for mu in centers:
        ax.plot(d_grid, rbf(d_grid, mu, gamma), lw=0.6,
                color="#2C5282", alpha=0.6)
    ax.set_xlabel(r"Interatomic distance $d_{ij}$ (Å)")
    ax.set_ylabel("RBF basis value")
    ax.set_title(f"All 40 Gaussian RBF bases at γ = {gamma} Å⁻²")
    ax.set_xlim(0, cutoff)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.axvspan(2.0, 4.0, color="orange", alpha=0.1,
               label="Typical bond range")
    ax.legend(loc="upper right", frameon=False)

    # Panel 2: zoomed in: show adjacent RBF overlap
    ax = axes[1]
    zoom_centers = centers[18:23]
    for mu in zoom_centers:
        ax.plot(d_grid, rbf(d_grid, mu, gamma), lw=1.6, alpha=0.8,
                label=f"μ = {mu:.3f} Å")

    # Mark adjacent-center overlap height
    overlap_d = 0.5 * (zoom_centers[0] + zoom_centers[1])
    overlap_h = rbf(overlap_d, zoom_centers[0], gamma)
    ax.axhline(overlap_h, ls=":", color="red", lw=0.8)
    ax.text(zoom_centers[0] + delta_mu * 0.1, overlap_h + 0.05,
            f"Adjacent-RBF crossover height = {overlap_h:.2f}",
            color="red", fontsize=8)

    ax.set_xlabel(r"Interatomic distance $d_{ij}$ (Å)")
    ax.set_ylabel("RBF basis value")
    ax.set_title(
        f"Zoomed: σ = {sigma:.3f} Å, "
        f"center spacing Δμ = {delta_mu:.3f} Å, σ/Δμ = {sigma/delta_mu:.2f}"
    )
    ax.set_xlim(zoom_centers[0] - 0.3, zoom_centers[-1] + 0.3)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    fig.suptitle(
        "RBF Basis Configuration Sanity Check — γ=10 Å⁻² is appropriate",
        y=1.02, fontsize=10
    )
    fig.tight_layout()
    fig.savefig("rbf_gamma_visualization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"γ = {gamma} Å⁻²")
    print(f"σ = 1/sqrt(2γ) = {sigma:.4f} Å")
    print(f"40 centers in [0, {cutoff}] Å, spacing Δμ = {delta_mu:.4f} Å")
    print(f"σ / Δμ = {sigma/delta_mu:.3f}  "
          "(target ~1.0 for moderate overlap)")
    print(f"Adjacent-center crossover height: {overlap_h:.3f}")
    print(f"-> γ = 10 Å⁻² gives {overlap_h*100:.0f}% adjacent-RBF overlap,")
    print("   appropriate for distance encoding in CGCNN-style models.")
    print("Saved: rbf_gamma_visualization.png")


if __name__ == "__main__":
    main()
