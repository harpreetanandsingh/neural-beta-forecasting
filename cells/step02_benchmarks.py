# %% [markdown]
# ## Step 2: Classical Rolling-Beta Benchmarks
# Before any neural network, we use the supplied historical estimates directly as forecasts of `beta_fwd_12m`
# ("next year's beta = the beta we measured up to today"):
# - `beta_12m`, `beta_24m`, `beta_36m`: rolling OLS betas estimated on 12, 24 and 36 monthly returns;
# - `beta_daily_12m`: trailing 12-month daily-return beta. It is a **persistence benchmark**, the same statistic as the target, one year earlier.
#
# Metrics: RMSE, MAE and the Pearson correlation between forecast and realised beta. We also report the mean
# forecast error (bias). The best **monthly** benchmark is selected on **validation RMSE only** and then kept fixed.
# For context we add two naive references that are *not* candidates for selection: a constant forecast of 1
# (the average market beta) and the constant training-sample mean.

# %%
def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = yhat - y
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
            "Corr": np.corrcoef(y, yhat)[0, 1] if np.std(yhat) > 0 else np.nan,
            "Bias (mean pred - real)": e.mean(), "N": len(y)}

MONTHLY_BENCH = ["beta_12m", "beta_24m", "beta_36m"]
DAILY_BENCH = "beta_daily_12m"
val = sample[sample.split == "validation"]
test = sample[sample.split == "test"]
train_mean = sample.loc[sample.split == "train", TARGET].mean()

# All benchmarks are scored on identical rows (eligible_core has no missing benchmark values)
assert val[MONTHLY_BENCH + [DAILY_BENCH]].notna().all().all() and test[MONTHLY_BENCH + [DAILY_BENCH]].notna().all().all()

rows = []
for name, part in [("validation", val), ("test", test)]:
    for b in MONTHLY_BENCH + [DAILY_BENCH]:
        rows.append({"split": name, "forecast": b, **metrics(part[TARGET], part[b])})
    rows.append({"split": name, "forecast": "const 1.0 (ref)", **metrics(part[TARGET], np.ones(len(part)))})
    rows.append({"split": name, "forecast": f"train mean {train_mean:.3f} (ref)",
                 **metrics(part[TARGET], np.full(len(part), train_mean))})
bench = pd.DataFrame(rows).set_index(["split", "forecast"])
bench.to_csv("results/step2_benchmarks.csv")
bench

# %%
# Select the monthly benchmark on validation RMSE only, then freeze the choice
val_rmse = bench.loc["validation"].loc[MONTHLY_BENCH, "RMSE"]
BEST_BENCH = val_rmse.idxmin()
print("Validation RMSE of monthly benchmarks:", val_rmse.round(4).to_dict())
print(f"==> Selected monthly benchmark (fixed for Step 5): {BEST_BENCH}")

# %%
# Do the rankings persist from validation to test?
rank = pd.DataFrame({f"{s} {m}": bench.loc[s].loc[MONTHLY_BENCH + [DAILY_BENCH], m].rank(ascending=(m != "Corr")).astype(int)
                     for s in ["validation", "test"] for m in ["RMSE", "MAE", "Corr"]})
print("Rank (1 = best) among the four historical-beta forecasts")
rank

# %% [markdown]
# ### 2.1 Responsiveness vs noise
# A shorter window reacts faster to a change in a firm's true beta, but it averages fewer observations, so it is noisier.
# We measure both sides on the validation + test rows:
# - **Dispersion.** The cross-sectional standard deviation of the forecast compared with that of the realised beta. Too much dispersion signals noise.
# - **Month-to-month movement.** The average |change| in a firm's forecast from one month to the next (consecutive months only).
# - **Slope of the realised on the forecast beta.** Regress realised on forecast. A slope of 1 means well calibrated; a slope below 1 means
#   the forecast is too extreme and should be shrunk towards the mean (errors-in-variables attenuation).

# %%
vt = sample[sample.split.isin(["validation", "test"])].sort_values(["permno", "date"])
consec = (vt.groupby("permno").date.diff() <= pd.Timedelta(days=31))
resp = []
for b in MONTHLY_BENCH + [DAILY_BENCH]:
    d = vt.groupby("permno")[b].diff().abs()[consec]
    slope = np.polyfit(vt[b], vt[TARGET], 1)[0]
    resp.append({"forecast": b, "sd of forecast": vt[b].std(), "sd of realised beta": vt[TARGET].std(),
                 "mean |monthly change|": d.mean(), "median |monthly change|": d.median(),
                 "slope realised-on-forecast": slope, "1st-99th pct range": vt[b].quantile(.99) - vt[b].quantile(.01)})
resp_tab = pd.DataFrame(resp).set_index("forecast")
resp_tab.to_csv("results/step2_responsiveness.csv")
resp_tab

# %%
# Error by forecast year (validation + test), showing stability over time
yr = []
for (y_, s), g in vt.groupby(["year", "split"]):
    for b in MONTHLY_BENCH + [DAILY_BENCH]:
        yr.append({"year": y_, "split": s, "forecast": b, "RMSE": np.sqrt(((g[b] - g[TARGET]) ** 2).mean()),
                   "MAE": (g[b] - g[TARGET]).abs().mean()})
yr = pd.DataFrame(yr)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
cols_b = {"beta_12m": "#d62728", "beta_24m": "#ff7f0e", "beta_36m": "#1f77b4", "beta_daily_12m": "#2ca02c"}
for b, c in cols_b.items():
    g = yr[yr.forecast == b]
    ax[0].plot(g.year.astype(str), g.MAE, marker="o", color=c, label=b)
ax[0].set_title("MAE by forecast year (validation 2013-16, test 2018-22)")
ax[0].legend(fontsize=8, ncol=2, loc="upper center"); ax[0].set_ylim(0.25, 1.2)
m = bench.xs("test")[["RMSE", "MAE"]].iloc[:4]
m.plot.bar(ax=ax[1], color=["#7f7f7f", "#bcbd22"], rot=0); ax[1].set_title("Test-set RMSE and MAE")
ax[1].axhline(bench.loc[("test", "const 1.0 (ref)"), "RMSE"], color="k", ls="--", lw=1, label="RMSE of constant 1.0")
ax[1].legend(fontsize=8)
ax[1].tick_params(axis="x", labelsize=8)
ax[2].scatter(resp_tab["mean |monthly change|"], bench.xs("validation").loc[resp_tab.index, "RMSE"],
              s=80, c=[cols_b[b] for b in resp_tab.index])
for b in resp_tab.index:
    ax[2].annotate(b, (resp_tab.loc[b, "mean |monthly change|"], bench.xs("validation").loc[b, "RMSE"]),
                   xytext=(5, 5), textcoords="offset points", fontsize=9)
ax[2].set_xlabel("mean |month-to-month change| in forecast (responsiveness)")
ax[2].set_ylabel("validation RMSE (inaccuracy)"); ax[2].set_title("Responsiveness vs accuracy")
plt.tight_layout(); savefig("step2_benchmarks"); plt.show()

# %%
# How much of the squared error comes from the few extreme labels? (diagnostic only; scoring is never clipped)
ext_t = test[TARGET].abs() > 4
for b in MONTHLY_BENCH + [DAILY_BENCH]:
    se = (test[b] - test[TARGET]) ** 2
    print(f"{b:>15}: test RMSE {np.sqrt(se.mean()):.4f}; share of SSE from |beta_fwd|>4 rows "
          f"({ext_t.sum()} rows, {ext_t.mean():.2%}): {se[ext_t].sum()/se.sum():.1%}; "
          f"RMSE without them {np.sqrt(se[~ext_t].mean()):.4f}")

# %% [markdown]
# ### 2.2 Answers
# **Which window is best on validation, and does the ranking persist?**
# `beta_36m` has the lowest validation RMSE (0.787, against 0.967 for 24m and 1.417 for 12m), so it is the fixed benchmark
# for Step 5. The ranking 36m < 24m < 12m is identical on test (0.905 / 1.074 / 1.439) and holds for MAE and correlation too.
# The daily persistence beta beats all three by a wide margin (test RMSE 0.530, correlation 0.51).
#
# **Are shorter windows more responsive but noisier?** Yes, very clearly:
# - `beta_12m` changes by 0.31 per month on average, against 0.09 for `beta_36m`, so it reacts faster.
# - Its cross-sectional dispersion (sd 1.46) is 2.6 times that of realised beta (0.55).
# - The slope of realised on forecast beta is only 0.09, against 0.24 for 36m and 0.58 for the daily beta. Most of
#   the variation in the 12-month estimate is estimation noise, which averages out as the window lengthens.
# - With 12 monthly observations the extra responsiveness buys nothing. The added noise swamps any gain from tracking
#   a changing beta faster. Its year-by-year error is also the least stable (see 2018 and 2021).
# - In my earlier CAPM beta study I found the same pattern in-sample: the 12-month beta changed about 3 times as much from year to year as the
#   36-month beta (mean |change| 0.71 vs 0.25), and I concluded that neither window was simply better. Scoring the windows against a *realised
#   future* beta settles that trade-off: the longer window is clearly more accurate.
#
# **How difficult is the problem?** Hard.
# - Every monthly rolling beta is *worse than the naive forecast β = 1* (test RMSE 0.54): the raw estimates are too noisy
#   and, in the test years, biased upwards (+0.20 for 36m).
# - Even the best historical estimate, the daily beta, cuts RMSE only about 2% below the constant 1.0 on test (0.530 vs 0.541).
#   It reaches a correlation of 0.51 with the realised beta, so roughly three quarters of the cross-sectional
#   variance is unexplained.
# - The extreme labels are not the reason: they account for only 1–4% of the squared error.
# - The main reasons are (i) estimation noise and mean reversion in historical betas and (ii) noise in the realised
#   target itself. This points to what a better forecast needs: shrink noisy estimates, lean on the precise
#   daily beta, and adjust the level using characteristics such as size.
