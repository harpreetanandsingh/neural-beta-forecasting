# %% [markdown]
# ## Step 8: Multi-Output Industry-Beta Extension
# For each firm-month, the target is now the vector of ten future **industry betas**,
# $\beta^R_{i,t+12}=[\beta^R_{i,1,t+12},\dots,\beta^R_{i,10,t+12}]$. Each element is a *separate univariate* daily-return slope of the firm's return on one
# FF10 value-weighted industry portfolio over months $t+1..t+12$. They are neither joint multifactor coefficients nor the industries' market betas.
#
# **Sample:** `eligible_multi == True` with the same chronological splits. **Inputs:** the same 12 numeric predictors plus industry dummies. Preprocessing
# is refit on the multi-output training sample, again on training data only. **Architecture and optimiser:** the market model selected in Step 4
# (22 → 32 → 16, ReLU, dropout 0.05, Adam, the selected learning rate, batch 512, at most 100 epochs, early stopping with patience 10, best checkpoint restored).
# There is no separate hyperparameter search for any industry.
# - **Separate-output benchmark ("sep")**: ten single-output MLPs, one per industry target.
# - **Multi-output model ("sim")**: one MLP whose final layer has ten outputs, trained on the **mean MSE across the ten unscaled targets**.

# %%
IND_NAMES = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]  # FF10 labels, same order as IND_SUFFIX
multi = df.loc[df.eligible_multi].copy()
assert multi[IND_TARGETS].notna().all().all()
print("eligible_multi rows by split:", multi.split.value_counts().reindex(SPLITS).to_dict())
print("Identical to the eligible_core sample:", multi.index.equals(sample.index))

mtr, mva, mte = (multi[multi.split == s] for s in SPLITS)
prep_m = BetaPreprocessor(FEATURES_NUM, FEATURES_CAT, no_clip=MARKET_LEVEL).fit(mtr)    # training rows only
Xm_tr, Xm_va, Xm_te = prep_m.transform(mtr), prep_m.transform(mva), prep_m.transform(mte)
Ym_tr, Ym_va, Ym_te = (d[IND_TARGETS].to_numpy(np.float32) for d in (mtr, mva, mte))
assert prep_m.feature_names_ == FEATURE_NAMES
print(f"X shapes {Xm_tr.shape} / {Xm_va.shape} / {Xm_te.shape}; Y shapes {Ym_tr.shape} / {Ym_va.shape} / {Ym_te.shape}")
print(f"Selected market-model settings reused: hidden {SEL['hidden']}, lr {SEL['lr']}, dropout {DROPOUT}, batch {BATCH}, patience {PATIENCE}")

# %%
def fit_sep_and_sim(seed):
    """Ten single-output models + one ten-output model with identical settings. Returns test predictions and training info."""
    sep_pred, sep_info = np.zeros_like(Ym_te), []
    for k, tname in enumerate(IND_TARGETS):
        mdl, _, inf = train_mlp(Xm_tr, Ym_tr[:, k], Xm_va, Ym_va[:, k], SEL["hidden"], SEL["lr"], seed=seed)
        sep_pred[:, k] = predict(mdl, Xm_te).ravel()
        sep_info.append({"target": IND_NAMES[k], **{c: inf[c] for c in ["best_epoch", "epochs_run", "best_val_mse"]}})
    sim_model, sim_hist, sim_inf = train_mlp(Xm_tr, Ym_tr, Xm_va, Ym_va, SEL["hidden"], SEL["lr"], seed=seed)
    return sep_pred, pd.DataFrame(sep_info).set_index("target"), predict(sim_model, Xm_te), sim_hist, sim_inf, sim_model

t0 = time.time()
sep_pred, sep_info, sim_pred, sim_hist, sim_info, sim_model = fit_sep_and_sim(SEED)
print(f"Trained 10 separate models + 1 multi-output model in {time.time() - t0:.0f}s")
print(f"Multi-output model: best epoch {sim_info['best_epoch']} of {sim_info['epochs_run']}, "
      f"validation mean MSE {sim_info['best_val_mse']:.5f}; "
      f"separate models: mean validation MSE {sep_info.best_val_mse.mean():.5f}")
sep_info

# %%
def multi_scores(Y, P_sep, P_sim):
    mse_sep, mse_sim = ((P_sep - Y) ** 2).mean(0), ((P_sim - Y) ** 2).mean(0)
    t = pd.DataFrame({"test MSE sep": mse_sep, "test MSE sim": mse_sim,
                      "test RMSE sep": np.sqrt(mse_sep), "test RMSE sim": np.sqrt(mse_sim)}, index=IND_NAMES)
    t["MSE change sim vs sep %"] = 100 * (t["test MSE sim"] / t["test MSE sep"] - 1)
    t["sim better"] = t["test MSE sim"] < t["test MSE sep"]
    return t

m8 = multi_scores(Ym_te, sep_pred, sim_pred)
mean_sep, mean_sim = m8["test MSE sep"].mean(), m8["test MSE sim"].mean()
summary8 = pd.Series({"mean MSE across industries (sep)": mean_sep, "mean MSE across industries (sim)": mean_sim,
                      "pooled RMSE = sqrt(mean MSE) (sep)": np.sqrt(mean_sep),
                      "pooled RMSE = sqrt(mean MSE) (sim)": np.sqrt(mean_sim),
                      "change in mean MSE, sim vs sep (%)": 100 * (mean_sim / mean_sep - 1),
                      "industries where sim improves on sep": f"{int(m8['sim better'].sum())} of 10"}, name="test")
m8.to_csv("results/step8_per_industry.csv"); summary8.to_csv("results/step8_summary.csv")
print(f"Identical test rows for both approaches: {len(Ym_te):,}")
display(m8.round(4))
summary8.to_frame()

# %%
# Reliability: monthly pooled loss differential (sep - sim, mean over firms and the 10 targets), Newey-West t with 12 lags
dl = pd.Series(((sep_pred - Ym_te) ** 2 - (sim_pred - Ym_te) ** 2).mean(1)).groupby(mte.date.to_numpy()).mean()
print(f"Mean monthly reduction in pooled MSE from sim: {dl.mean():+.5f} (NW t = {nw_t(dl.to_numpy()):.2f}); "
      f"sim better in {(dl > 0).mean():.0%} of the {len(dl)} test months")
per_ind_t = {IND_NAMES[k]: nw_t(pd.Series((sep_pred[:, k] - Ym_te[:, k]) ** 2 - (sim_pred[:, k] - Ym_te[:, k]) ** 2)
                                 .groupby(mte.date.to_numpy()).mean().to_numpy()) for k in range(10)}
m8["NW t (sep - sim)"] = pd.Series(per_ind_t)
m8.to_csv("results/step8_per_industry.csv")
summary8["pooled NW t (sep - sim)"] = nw_t(dl.to_numpy()); summary8["share of months sim better"] = (dl > 0).mean()
summary8.to_csv("results/step8_summary.csv")
m8[["MSE change sim vs sep %", "NW t (sep - sim)"]].round(2).T

# %%
# Context: how each industry target relates to the market beta and to the other industry betas (training sample)
corr_tr = pd.DataFrame(Ym_tr, columns=IND_NAMES).assign(MARKET=mtr[TARGET].to_numpy()).corr()
ctx = pd.DataFrame({"corr with market beta": corr_tr["MARKET"].drop("MARKET"),
                    "avg corr with other 9 industry betas": (corr_tr.loc[IND_NAMES, IND_NAMES].sum() - 1) / 9,
                    "target sd (test)": Ym_te.std(0), "MSE change sim vs sep %": m8["MSE change sim vs sep %"]})
print("Correlation across industries between 'avg corr with other betas' and the sim-vs-sep MSE change:",
      round(ctx["avg corr with other 9 industry betas"].corr(ctx["MSE change sim vs sep %"]), 3))
ctx.sort_values("MSE change sim vs sep %").round(3)

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1.5, 1.1, 1]})
x = np.arange(10); w = 0.38
ax[0].bar(x - w / 2, m8["test RMSE sep"], w, color="#1f77b4", label="separate (10 single-output MLPs)")
ax[0].bar(x + w / 2, m8["test RMSE sim"], w, color="#d62728", label="multi-output (one 10-output MLP)")
ax[0].set_xticks(x); ax[0].set_xticklabels(IND_NAMES); ax[0].set_ylabel("test RMSE")
ax[0].set_title("Test RMSE per industry-beta target"); ax[0].legend(fontsize=8)
o = m8["MSE change sim vs sep %"].sort_values()
ax[1].barh(o.index, o.values, color=["#2ca02c" if v < 0 else "#d62728" for v in o.values])
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlabel("test MSE change, multi vs separate (%)  (negative = multi better)")
ax[1].set_title(f"Pooled: {summary8['change in mean MSE, sim vs sep (%)']:+.2f}%; multi better in {int(m8['sim better'].sum())}/10")
ax[2].plot(sim_hist.epoch, sim_hist.train_mse, label="train mean MSE (10 outputs)")
ax[2].plot(sim_hist.epoch, sim_hist.val_mse, label="validation mean MSE")
ax[2].axvline(sim_info["best_epoch"], color="k", ls="--", lw=1); ax[2].legend(fontsize=8)
ax[2].set_xlabel("epoch"); ax[2].set_title("Multi-output model loss curves")
plt.tight_layout(); savefig("step8_multi_output"); plt.show()

# %% [markdown]
# ### 8.1 Discussion
# **Does joint prediction help? Yes, modestly.**
# - On identical test rows, the multi-output model lowers the mean MSE across the ten industry betas by **3.5%** (pooled RMSE 0.422 → 0.415).
# - It improves on the separate models for **9 of 10 industries**.
# - The pooled gain is statistically reliable: it is positive in 85% of test months, with Newey–West t = 3.8.
# - The size is similar to what Cramer, Drobetz & Otto (2026) report (gains of up to 4.8% value-weighted and 6.3% equal-weighted), in a much smaller setting.
#
# **Caveats.**
# - In the *validation* period, the separate models were slightly better (mean validation MSE 0.149 vs 0.152), so the advantage is not visible in every period.
# - The seed check in **Appendix A** shows that the pooled gain appears for all three seeds (−3.5%, −2.9%, −0.3%), but its size, and especially the
#   *per-industry* ranking, varies considerably from run to run. Conclusions about individual industries should be read cautiously.
#
# **Which industries benefit most and least.**
# - With seed 42, the largest gains are in **Durables (−11.6%)**, HiTec (−8.8%), Energy (−8.1%) and Utilities (−5.8%).
# - **Manufacturing is the one loser (+4.6%)**, and NoDur is unchanged.
# - Across seeds, Durables has the largest average gain (−5.8%). NoDur, Durbl, Telcm and Hlth improve under all three seeds. Manufacturing averages about 0.
#
# **Possible reasons.**
# 1. *A shared common factor.* The ten targets are highly correlated (0.59–0.94) because each largely reflects the firm's general market sensitivity.
#    The shared hidden layers learn this common structure from ten targets at once. Each target is a noisy one-year regression slope, so pooling them
#    averages out label noise and acts as a regulariser (the classic multi-task learning argument, and the paper's "cross-sectional interdependencies").
#    Seven of the ten separate models reach their best validation loss after only 1–2 epochs, so the single-target problems are noisy and overfit quickly.
# 2. *Noisier, smaller-industry targets gain most.* The Durables portfolio (autos and appliances) is small and idiosyncratic, so a firm's slope on it
#    is estimated noisily. Borrowing strength from the other nine targets helps most there.
# 3. *Targets that already look like the market gain least.* The Manufacturing beta is almost the market beta (correlation 0.96), and a separate model
#    already captures that common factor well. In the shared model, the 16 hidden units must also serve nine other outputs, which can slightly compromise its fit.
# 4. The pattern is **not** simply "more correlated → more gain". Distinctive industries such as Energy and Utilities also benefit, and across the ten industries
#    the correlation between co-movement and gain is weak (0.39, with only 10 points). With seed variation this large, finer explanations are not supported.
#
# Limits: the paper uses a richer predictor set (firm characteristics and macro variables), log returns, beta-change targets, rolling re-estimation and portfolio
# applications. This classroom version only tests whether sharing hidden layers helps with the same inputs.
