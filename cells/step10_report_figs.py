# %% [markdown]
# ---
# ## Appendix B: Page-width figures
# The figures above are sized for the notebook. This section redraws the main ones at page width with larger fonts.
# No new computation is done here. Every value comes from the objects created in Steps 1–8 and Appendix A.

# %%
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
      "legend.fontsize": 7, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3}
C_NN, C_D, C_36, C_OLS = "#c0392b", "#2e8b57", "#2c6fb0", "#7f7f7f"

with plt.rc_context(RC):
    # R1: target distribution + loss curves
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.15))
    bins = np.linspace(-1.5, 3.5, 81)
    for s in SPLITS:
        v = sample.loc[sample.split == s, TARGET]
        ax[0].hist(v.clip(-1.5, 3.5), bins=bins, density=True, histtype="step", lw=1.2, color=SPLIT_COLORS[s],
                   label=f"{s}: mean {v.mean():.2f}")
    ax[0].axvline(1, color="k", ls="--", lw=0.7); ax[0].set_xlabel("realised future beta (display clipped to [-1.5, 3.5])")
    ax[0].set_ylabel("density"); ax[0].set_title("(a) Distribution of beta_fwd_12m by split")
    ax[0].set_ylim(0, 1.65); ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    cfg_cols = ["#1f77b4", "#ff7f0e", C_NN, "#2ca02c"]
    for (k, v), cc in zip(runs.items(), cfg_cols):
        h = v["hist"]; sel = k == BEST_CFG
        short = k.replace("1 layer (32)", "22-32-1").replace("2 layers (32-16)", "22-32-16-1")
        ax[1].plot(h.epoch, h.val_mse, color=cc, lw=1.7 if sel else 0.9, label=short + (" (selected)" if sel else ""))
        ax[1].plot(h.epoch, h.train_mse, color=cc, lw=1.2 if sel else 0.8, ls=":")
    ax[1].plot([], [], color="grey", lw=1, label="solid: validation MSE")
    ax[1].plot([], [], color="grey", lw=1, ls=":", label="dotted: training MSE")
    ax[1].axvline(runs[BEST_CFG]["best_epoch"], color="k", ls="--", lw=0.7)
    dmse = bench.loc[("validation", "beta_daily_12m"), "RMSE"] ** 2
    ax[1].axhline(dmse, color=C_D, ls="-.", lw=0.9)
    ax[1].text(ax[1].get_xlim()[1], dmse - 0.002, "validation MSE of beta_daily_12m ", color=C_D, fontsize=6, ha="right", va="top")
    ax[1].set_ylim(0.13, 0.32); ax[1].set_yticks([0.14, 0.16, 0.18, 0.20, 0.22, 0.24]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MSE")
    ax[1].set_title("(b) Training and validation loss, four MLP configurations"); ax[1].legend(frameon=False, fontsize=5.6, ncol=2, loc="upper center")
    plt.tight_layout(); plt.savefig("figures/report_fig1.png", bbox_inches="tight"); plt.savefig("figures/report_fig1.pdf", bbox_inches="tight"); plt.show()

    # R2: scatter + time series
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1, 1.55]})
    p_ = preds_test["MLP (neural net)"]
    ax[0].hexbin(np.clip(p_, -1, 3.5), np.clip(y_test, -1, 3.5), gridsize=55, bins="log", cmap="viridis", mincnt=1)
    ax[0].plot([-1, 3.5], [-1, 3.5], "r--", lw=0.8, label="45 degree line")
    b1, b0 = np.polyfit(p_, y_test, 1)
    ax[0].plot([-1, 3.5], [b0 - b1, b0 + 3.5 * b1], "k-", lw=1.4, label=f"OLS fit, slope {b1:.2f}")
    ax[0].set_xlabel("MLP predicted beta"); ax[0].set_ylabel("realised beta_fwd_12m")
    ax[0].set_title("(a) Predicted vs realised beta, test"); ax[0].legend(frameon=True, loc="upper left", fontsize=6.5)
    ax[1].plot(ts5.index, ts5.realised, color="k", lw=1.6, label="realised")
    ax[1].plot(ts5.index, ts5.MLP, color=C_NN, lw=1.3, label="MLP")
    ax[1].plot(ts5.index, ts5.beta_daily_12m, color=C_D, lw=1, ls="--", label="beta_daily_12m")
    ax[1].plot(ts5.index, ts5.beta_36m, color=C_36, lw=1, ls=":", label="beta_36m")
    ax[1].axhline(1, color="grey", lw=0.6); ax[1].set_ylabel("cross-sectional mean beta")
    ax[1].set_title("(b) Monthly cross-sectional average, test"); ax[1].legend(frameon=False, ncol=4, loc="upper left")
    ax[1].set_ylim(0.85, 1.45)
    plt.tight_layout(); plt.savefig("figures/report_fig2.png", bbox_inches="tight"); plt.savefig("figures/report_fig2.pdf", bbox_inches="tight"); plt.show()

    # R3: RMSE by industry + quintiles
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.45, 1]})
    order = ind_err.drop(index="Unknown", errors="ignore").sort_values("RMSE MLP (neural net)").index
    x = np.arange(len(order)); w = 0.27
    for j, (k, c, lab) in enumerate([("beta_36m", C_36, "beta_36m"), ("beta_daily_12m", C_D, "beta_daily_12m"),
                                     ("MLP (neural net)", C_NN, "MLP")]):
        ax[0].bar(x + (j - 1) * w, ind_err.loc[order, f"RMSE {k}"], w, color=c, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=35, fontsize=6.8, ha="right"); ax[0].set_ylabel("test RMSE")
    ax[0].set_title("(a) Test RMSE by FF10 industry"); ax[0].legend(frameon=False, ncol=3, loc="upper left"); ax[0].set_ylim(0, 1.65)
    xq = np.arange(1, 6)
    for j, (col, c, lab) in enumerate([("beta_36m", C_36, "sorted on beta_36m"), ("beta_daily_12m", C_D, "sorted on beta_daily_12m"),
                                       ("beta_NN", C_NN, "sorted on MLP beta")]):
        ax[1].bar(xq + (j - 1) * w, comp[col]["avg realised beta"], w, color=c, label=lab)
    ax[1].set_xticks(xq); ax[1].set_xticklabels([f"Q{i}" for i in xq]); ax[1].set_ylabel("avg realised beta")
    ax[1].set_ylim(0, 2.6)
    ax2 = ax[1].twinx(); ax2.plot(xq, 100 * qt.loc[[1, 2, 3, 4, 5], "avg_next_month_return"], "ko-", ms=3, lw=1,
                                  label="next-month return, MLP sort (right axis)")
    ax2.set_ylabel("% per month"); ax2.set_ylim(0, 3.4); ax2.grid(False)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper left")
    ax[1].set_title("(b) Quintile sorts on predicted beta, test")
    plt.tight_layout(); plt.savefig("figures/report_fig3.png", bbox_inches="tight"); plt.savefig("figures/report_fig3.pdf", bbox_inches="tight"); plt.show()

    # R4: permutation importance + multi-output
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1, 1.2]})
    top = var_tab.head(6).iloc[::-1]
    labels = [f"group: {g}" for g in imp_tab.index[::-1]] + list(top.index)
    vals = list(imp_tab.mean_RMSE_increase[::-1]) + list(top.mean_RMSE_increase)
    errs = list(imp_tab.sd_RMSE_increase[::-1]) + list(top.sd)
    gc = {"historical risk": C_NN, "returns & size": C_36, "industry": C_D}
    cols = [gc[g] for g in imp_tab.index[::-1]] + [gc[g] for g in top.group]
    ypos = [7, 8, 9] + list(range(6))          # groups on top, separated from the single variables
    ax[0].barh(ypos, vals, xerr=errs, color=cols, capsize=2)
    ax[0].set_yticks(ypos); ax[0].set_yticklabels(labels, fontsize=6.8)
    ax[0].axhline(6.5, color="grey", lw=0.6)
    ax[0].set_xlabel(f"increase in test RMSE (base {base_rmse:.3f})")
    ax[0].set_title("(a) Permutation importance")
    o = per_ind.reindex(IND_NAMES)
    xi = np.arange(10)
    ax[1].bar(xi, o[SEED], color=[C_D if v < 0 else C_NN for v in o[SEED]], label="seed 42 (main)")
    ax[1].scatter(xi, o[1], color="k", s=9, marker="o", label="seed 1", zorder=3)
    ax[1].scatter(xi, o[2], color="k", s=11, marker="x", label="seed 2", zorder=3)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(xi); ax[1].set_xticklabels(IND_NAMES, fontsize=6.8, rotation=35, ha="right")
    ax[1].set_ylabel("test MSE change (%)")
    ax[1].set_title("(b) Multi-output vs separate: test MSE change")
    ax[1].set_ylim(-13, 10)
    ax[1].legend(frameon=False, fontsize=6, loc="upper right", ncol=3)
    plt.tight_layout(); plt.savefig("figures/report_fig4.png", bbox_inches="tight"); plt.savefig("figures/report_fig4.pdf", bbox_inches="tight"); plt.show()
