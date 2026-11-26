# %% [markdown]
# ## Step 5: Out-of-Sample Evaluation (test period 2018–2022)
# The selected MLP (fixed in Step 4) is applied **once** to the untouched test sample. Every forecast is scored on the same
# 170,650 eligible rows. The $R^2_{OOS}$ denominator is the monthly benchmark selected on validation in Step 2 (`beta_36m`):
# $$R^2_{OOS}=1-\frac{\sum_{\mathcal T}(\beta^R-\widehat\beta^{NN})^2}{\sum_{\mathcal T}(\beta^R-\widehat\beta^{36m})^2}.$$
# As an **extra reference** (not part of the main model set), we also fit an ordinary least-squares regression on the same 22 inputs,
# using the training data only. The comparison MLP vs OLS shows how much of the gain comes from non-linearity rather than from simply combining the inputs.

# %%
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X_train, y_train)          # training data only; OLS has no hyperparameters to tune

preds_test = {"MLP (neural net)": predict(nn_model, X_test).ravel(),
              "OLS on same inputs (ref)": ols.predict(X_test),
              "beta_12m": test_df.beta_12m.to_numpy(), "beta_24m": test_df.beta_24m.to_numpy(),
              "beta_36m": test_df.beta_36m.to_numpy(), "beta_daily_12m": test_df.beta_daily_12m.to_numpy(),
              "constant 1.0 (ref)": np.ones(len(test_df))}
preds_val = {"MLP (neural net)": predict(nn_model, X_val).ravel(), "OLS on same inputs (ref)": ols.predict(X_val)}
assert all(len(v) == len(y_test) and np.isfinite(v).all() for v in preds_test.values())

bench_pred = preds_test[BEST_BENCH]
sse_bench = ((y_test - bench_pred) ** 2).sum()
sse_daily = ((y_test - preds_test["beta_daily_12m"]) ** 2).sum()
res = []
for k, p_ in preds_test.items():
    sse = ((y_test - p_) ** 2).sum()
    res.append({"forecast": k, **metrics(y_test, p_),
                f"R2_OOS vs {BEST_BENCH} (main)": 1 - sse / sse_bench,
                "R2_OOS vs beta_daily_12m (suppl.)": 1 - sse / sse_daily})
test_tab = pd.DataFrame(res).set_index("forecast")
test_tab.to_csv("results/step5_test_metrics.csv")
print(f"Test sample: {len(y_test):,} firm-months, {test_df.date.nunique()} months, identical for all forecasts. "
      f"R2_OOS denominator = {BEST_BENCH} (chosen on validation in Step 2).")
test_tab

# %%
# Validation vs test for the fitted models (consistency check; validation was used for selection)
vt_tab = pd.DataFrame({f"{s} {m}": {k: metrics(y, p_[k])[m] for k in preds_val}
                       for s, y, p_ in [("validation", y_val, preds_val), ("test", y_test, preds_test)]
                       for m in ["RMSE", "MAE", "Corr"]})
vt_tab

# %% [markdown]
# ### 5.1 Is the improvement statistically reliable?
# Rows in the same month share market shocks, and consecutive months have overlapping 12-month target windows, so the errors are not independent.
# We therefore use a Diebold–Mariano-style test on **monthly** loss differentials:
# $d_t=\overline{e^2_{bench}}-\overline{e^2_{MLP}}$ (the cross-sectional mean in month $t$), with a Newey–West standard error using 12 lags.

# %%
def dm_test(y, p_model, p_bench, dates, loss="se", lags=12):
    e_m, e_b = (y - p_model), (y - p_bench)
    l = (e_b ** 2 - e_m ** 2) if loss == "se" else (np.abs(e_b) - np.abs(e_m))
    d = pd.Series(l).groupby(np.asarray(dates)).mean().to_numpy()
    T, dbar = len(d), d.mean()
    u = d - dbar
    lrv = u @ u / T + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T for k in range(1, lags + 1))
    return dbar, dbar / np.sqrt(lrv / T), (d > 0).mean()

dm = []
for b in ["beta_36m", "beta_daily_12m", "OLS on same inputs (ref)", "constant 1.0 (ref)"]:
    for loss in ["se", "ae"]:
        dbar, t, share = dm_test(y_test, preds_test["MLP (neural net)"], preds_test[b], test_df.date, loss)
        dm.append({"MLP vs": b, "loss": "squared error" if loss == "se" else "absolute error",
                   "mean monthly loss reduction": dbar, "NW t-stat": t, "share of months MLP better": share})
dm_tab = pd.DataFrame(dm).set_index(["MLP vs", "loss"])
dm_tab.to_csv("results/step5_dm_tests.csv")
dm_tab

# %% [markdown]
# ### 5.2 Predicted vs realised beta

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, k in zip(axes, ["MLP (neural net)", "beta_daily_12m", "beta_36m"]):
    p_ = preds_test[k]
    hb = ax.hexbin(np.clip(p_, -1.5, 4), np.clip(y_test, -1.5, 4), gridsize=70, bins="log", cmap="viridis", mincnt=1)
    ax.plot([-1.5, 4], [-1.5, 4], "r--", lw=1, label="45° line")
    b1, b0 = np.polyfit(p_, y_test, 1)
    ax.plot([-1.5, 4], [b0 - 1.5 * b1, b0 + 4 * b1], "k-", lw=1.5, label=f"fit: slope {b1:.2f}")
    m = test_tab.loc[k]
    ax.set_title(f"{k}\nRMSE {m.RMSE:.3f} | MAE {m.MAE:.3f} | corr {m.Corr:.3f}", fontsize=10)
    ax.set_xlabel("predicted beta"); ax.set_ylabel("realised beta_fwd_12m")
    ax.set_xlim(-1.5, 4); ax.set_ylim(-1.5, 4); ax.legend(loc="upper left", fontsize=8)
plt.colorbar(hb, ax=axes, fraction=0.015, label="log10(count)")
plt.suptitle("Test sample 2018–2022: predicted vs realised beta (values clipped to [-1.5, 4] for display only)", y=1.02)
savefig("step5_scatter"); plt.show()

# %% [markdown]
# ### 5.3 Cross-sectional average predicted and realised beta over time

# %%
ts5 = pd.DataFrame({"date": test_df.date.to_numpy(), "realised": y_test,
                    "MLP": preds_test["MLP (neural net)"], "beta_daily_12m": preds_test["beta_daily_12m"],
                    "beta_36m": preds_test["beta_36m"]}).groupby("date").mean()
fig, ax = plt.subplots(1, 2, figsize=(16, 4.3), gridspec_kw={"width_ratios": [2, 1]})
ax[0].plot(ts5.index, ts5.realised, color="k", lw=2.2, label="realised beta_fwd_12m")
ax[0].plot(ts5.index, ts5.MLP, color="#d62728", lw=1.8, label="MLP prediction")
ax[0].plot(ts5.index, ts5.beta_daily_12m, color="#2ca02c", lw=1.2, ls="--", label="beta_daily_12m")
ax[0].plot(ts5.index, ts5.beta_36m, color="#1f77b4", lw=1.2, ls=":", label="beta_36m")
ax[0].axhline(1, color="grey", lw=0.8)
ax[0].set_title("Cross-sectional average beta by forecast month (equal-weighted, test sample)")
ax[0].legend(fontsize=8, ncol=2); ax[0].set_ylabel("average beta")
# month-by-month cross-sectional RMSE
cs = pd.DataFrame({"date": test_df.date.to_numpy(), **{k: (preds_test[k] - y_test) ** 2
                   for k in ["MLP (neural net)", "beta_daily_12m", "beta_36m"]}}).groupby("date").mean() ** 0.5
for k, c in [("MLP (neural net)", "#d62728"), ("beta_daily_12m", "#2ca02c"), ("beta_36m", "#1f77b4")]:
    ax[1].plot(cs.index, cs[k], color=c, label=k)
ax[1].set_title("Cross-sectional RMSE by forecast month"); ax[1].legend(fontsize=8)
plt.tight_layout(); savefig("step5_timeseries"); plt.show()
yr5 = ts5.assign(year=ts5.index.year).groupby("year").mean()
yr5["MLP bias"] = yr5.MLP - yr5.realised; yr5["daily bias"] = yr5.beta_daily_12m - yr5.realised
yr5["36m bias"] = yr5.beta_36m - yr5.realised
print("Average level and bias by forecast year (means of the monthly cross-sectional averages)")
yr5.round(3)

# %% [markdown]
# ### 5.4 Error by industry and by firm size

# %%
err = pd.DataFrame({"industry": test_df.industry.to_numpy(), "size_q": pd.qcut(test_df.log_market_cap, 5, labels=[
    "Q1 small", "Q2", "Q3", "Q4", "Q5 large"]).to_numpy(), "y": y_test,
    **{k: preds_test[k] for k in ["MLP (neural net)", "OLS on same inputs (ref)", "beta_daily_12m", "beta_36m"]}})
def grp_rmse(g, key):
    out = g.groupby(key, observed=True).apply(lambda d: pd.Series(
        {"N": len(d), **{f"RMSE {k}": np.sqrt(((d[k] - d.y) ** 2).mean()) for k in
                         ["MLP (neural net)", "OLS on same inputs (ref)", "beta_daily_12m", "beta_36m"]},
         "MAE MLP": (d["MLP (neural net)"] - d.y).abs().mean(), "MAE daily": (d["beta_daily_12m"] - d.y).abs().mean(),
         "mean realised": d.y.mean(), "MLP bias": (d["MLP (neural net)"] - d.y).mean()}))
    out["MLP RMSE gain vs daily %"] = 100 * (1 - out["RMSE MLP (neural net)"] / out["RMSE beta_daily_12m"])
    out["MLP RMSE gain vs 36m %"] = 100 * (1 - out["RMSE MLP (neural net)"] / out["RMSE beta_36m"])
    return out
ind_err = grp_rmse(err, "industry").sort_values("RMSE MLP (neural net)")
ind_err.to_csv("results/step5_error_by_industry.csv")
ind_err

# %%
size_err = grp_rmse(err, "size_q")
size_err.to_csv("results/step5_error_by_size.csv")
size_err

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
order = ind_err.drop(index="Unknown", errors="ignore").index
w = 0.27; x = np.arange(len(order))
for j, (k, c) in enumerate([("beta_36m", "#1f77b4"), ("beta_daily_12m", "#2ca02c"), ("MLP (neural net)", "#d62728")]):
    ax[0].bar(x + (j - 1) * w, ind_err.loc[order, f"RMSE {k}"], w, color=c, label=k)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{i}\n(n={int(ind_err.loc[i, 'N']):,})" for i in order], fontsize=8)
ax[0].set_ylabel("test RMSE"); ax[0].set_title("Test RMSE by industry (FF10)"); ax[0].legend(fontsize=8)
xs = np.arange(5)
for j, (k, c) in enumerate([("beta_36m", "#1f77b4"), ("beta_daily_12m", "#2ca02c"), ("MLP (neural net)", "#d62728")]):
    ax[1].bar(xs + (j - 1) * w, size_err[f"RMSE {k}"], w, color=c, label=k)
ax[1].set_xticks(xs); ax[1].set_xticklabels(size_err.index); ax[1].set_title("Test RMSE by size quintile")
plt.tight_layout(); savefig("step5_error_groups"); plt.show()

# %% [markdown]
# ### 5.5 Discussion: where the neural model improves and where it does not
# **Headline (test 2018–2022, 170,650 identical rows).**
# - The MLP has RMSE 0.443, MAE 0.316 and correlation 0.595. **$R^2_{OOS}$ relative to `beta_36m` is 0.760**: the MLP's squared error is 76% below the benchmark's.
# - It improves on *every* benchmark on every metric, and the ranking from validation holds on test.
# - The MLP's loss is lower than both `beta_36m` and `beta_daily_12m` in **all 60 test months** (Newey–West t = 11.3 and 9.4).
#
# **Gains relative to the monthly rolling betas and to the daily-beta persistence forecast are two different things.**
# - *Against the monthly betas (12/24/36m) the gain is large (R² 0.76), but it is an easy bar.* A constant forecast of 1.0 already achieves R² 0.64 against `beta_36m`.
#   Most of this gain comes from replacing noisy, upward-biased monthly estimates (bias +0.20) with a precise, well-centred forecast (bias −0.01).
#   Better *measurement* (daily data) and *shrinkage* do the work.
# - *Against the daily-beta persistence forecast, which is also an input to the model, the gain is the genuinely informative one.*
#   - MSE falls by **30%** (supplementary R² vs daily = 0.30; RMSE 0.443 vs 0.530; MAE 0.316 vs 0.368).
#   - The daily beta is too dispersed: realised beta rises only 0.51 for each unit of daily beta. The MLP pulls extreme values back towards the mean
#     (fit slope 0.76, predictions stay between about 0 and 3).
#   - It also conditions that shrinkage on size, volatility, past returns and industry.
# - *Non-linearity adds a smaller but significant part.* An OLS regression on the same inputs already captures most of the gain (R² vs daily 0.26).
#   The MLP improves on OLS further (RMSE 0.443 vs 0.455; t = 3.9; better in 73% of months). The gain is largest among **large firms**, where OLS is
#   *worse* than the daily beta but the MLP is 12% better, and among small firms.
#
# **Where it improves most.**
# - **By industry:** the MLP beats both benchmarks in every industry.
#   - The largest gains over the daily beta are in Durables (27%), HiTec (20%), Health (19%) and Energy (17%). These are volatile, heterogeneous industries
#     where the daily beta is noisiest and shrinkage pays off most.
#   - Errors remain highest in Energy (0.63) and Health (0.50), whose betas move with commodity prices and biotech-specific news.
# - **By size:** the gain over the daily beta is largest for small firms (Q1: 20%) and falls to about 11–12% for mid-to-large firms.
#   This is consistent with the thin-trading correction that size provides.
#
# **Where it does not improve, or has limits.**
# - **Small firms are under-predicted** (bias −0.18 in Q1).
#   - Small test firms have higher realised betas (0.79) than the training-era pattern implies.
#   - This is the non-stationarity found in Step 4: the strong thin-trading discount learned from 1998–2002 is too strong for 2018–22.
# - **Market-wide shifts in beta are not anticipated.**
#   - The average MLP forecast tracks the trailing daily beta with a lag. It under-predicts the COVID-era jump in realised betas in 2019–20 (bias −0.08)
#     and over-predicts in 2021 (+0.11), when trailing windows still contained the crash.
#   - The gain is in **ranking firms within each month**, not in timing the average level. That is expected given the few market-level inputs.
# - **Utilities:** the linear model is slightly better than the MLP (0.364 vs 0.375). With a low and stable beta, the extra flexibility buys nothing.
# - **Noise in the realised target itself** caps achievable accuracy. Even the best model explains only about 35% of the cross-sectional variance
#   (correlation 0.595). Most of the remaining error is probably measurement noise in 12-month realised betas rather than forecastable signal.
