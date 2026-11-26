# %% [markdown]
# ## Step 6: Economic Interpretation of the Neural Beta Forecasts (Quintile Sorts)
# In each test month, firms are sorted into five equal-sized portfolios by their **predicted** neural beta. Ties are broken
# deterministically: rows are ordered by `permno` first, and `rank(method="first")` then gives tied predictions consecutive ranks in `permno` order.
# For each month and quintile we compute equal-weighted averages of predicted beta, realised future beta (`beta_fwd_12m`)
# and the subsequent **one-month** return (`return_fwd_1m`), then average those monthly portfolio statistics over the 60 test months.
# Missing next-month returns are excluded **only** from the return averages (not filled with 0), and their count is reported.
# These are descriptive sorts, not an implementable trading strategy.
#
# For comparison, the same sort is repeated with `beta_36m` (the classical rolling beta from my earlier CAPM study) and `beta_daily_12m`.

# %%
def nw_t(x, lags=12):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; T = len(x); u = x - x.mean()
    lrv = u @ u / T + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T for k in range(1, lags + 1))
    return x.mean() / np.sqrt(lrv / T)

def quintile_sort(frame, sort_col):
    d = frame.sort_values(["date", "permno"]).copy()                       # permno order breaks ties
    d["rank"] = d.groupby("date")[sort_col].rank(method="first")
    d["n_month"] = d.groupby("date")["rank"].transform("size")
    d["Q"] = np.ceil(d["rank"] * 5 / d["n_month"]).astype(int)             # 1 = lowest predicted beta
    monthly = d.groupby(["date", "Q"]).agg(pred=(sort_col, "mean"), realised=(TARGET, "mean"),
                                           ret_1m=("return_fwd_1m", "mean"),   # NaNs skipped, not filled
                                           n=("permno", "size"), n_ret_missing=("return_fwd_1m", lambda s: s.isna().sum()))
    return d, monthly

q_frame = test_df[["permno", "date", TARGET, "return_fwd_1m", "beta_36m", "beta_daily_12m"]].copy()
q_frame["beta_NN"] = preds_test["MLP (neural net)"]
q_rows, q_monthly = q_frame, {}
for col in ["beta_NN", "beta_36m", "beta_daily_12m"]:
    q_rows_col, q_monthly[col] = quintile_sort(q_frame, col)
    if col == "beta_NN":
        q_rows = q_rows_col
assert q_rows.groupby(["date", "Q"]).size().groupby("date").agg(lambda s: s.max() - s.min()).max() <= 1  # equal sizes

m = q_monthly["beta_NN"]
qt = m.groupby("Q").agg(avg_predicted_beta=("pred", "mean"), avg_realised_beta=("realised", "mean"),
                        avg_next_month_return=("ret_1m", "mean"), avg_firms_per_month=("n", "mean"),
                        total_missing_returns=("n_ret_missing", "sum"))
qt["return NW t-stat"] = [nw_t(m.xs(q, level="Q").ret_1m) for q in qt.index]
spread = m.ret_1m.unstack()[5] - m.ret_1m.unstack()[1]
bspread = m.realised.unstack()[5] - m.realised.unstack()[1]
qt.loc["Q5 - Q1"] = [qt.avg_predicted_beta[5] - qt.avg_predicted_beta[1], bspread.mean(), spread.mean(),
                     np.nan, np.nan, nw_t(spread)]
qt.to_csv("results/step6_quintiles_nn.csv")
print(f"Test months: {m.index.get_level_values('date').nunique()}; missing return_fwd_1m excluded from return "
      f"averages only: {int(m.n_ret_missing.sum())} of {int(m.n.sum()):,} firm-months")
qt

# %%
# Monotonicity checks
real = m.realised.unstack(); pred_ = m.pred.unstack(); ret_ = m.ret_1m.unstack()
mono_avg = bool(np.all(np.diff(qt.loc[[1, 2, 3, 4, 5], "avg_realised_beta"]) > 0))
mono_month = (real.diff(axis=1).iloc[:, 1:] > 0).all(axis=1)
print(f"Average realised beta increases monotonically Q1->Q5: {mono_avg}")
print(f"Share of test months in which realised beta is monotone across all five quintiles: {mono_month.mean():.1%}")
print(f"Q5-Q1 realised-beta spread: mean {bspread.mean():.3f}, min {bspread.min():.3f}, positive in {(bspread > 0).mean():.0%} of months")
print(f"Q5-Q1 next-month return spread: {spread.mean():.4f} per month (NW t = {nw_t(spread):.2f}); "
      f"positive in {(spread > 0).mean():.0%} of months")
print(f"Realised/predicted ratio by quintile: {(qt.avg_realised_beta / qt.avg_predicted_beta).loc[[1,2,3,4,5]].round(3).to_dict()}")

# %%
# The same sort with the classical and daily historical betas
comp = {}
for col in ["beta_NN", "beta_36m", "beta_daily_12m"]:
    mm = q_monthly[col]
    comp[col] = pd.DataFrame({"avg sort beta": mm.pred.groupby("Q").mean(), "avg realised beta": mm.realised.groupby("Q").mean(),
                              "avg next-month return": mm.ret_1m.groupby("Q").mean()})
comp_tab = pd.concat(comp, axis=1)
summ = pd.DataFrame({col: {
    "realised-beta spread Q5-Q1": q_monthly[col].realised.unstack().pipe(lambda u: (u[5] - u[1]).mean()),
    "sort-beta spread Q5-Q1": q_monthly[col].pred.unstack().pipe(lambda u: (u[5] - u[1]).mean()),
    "realised / sorted spread": q_monthly[col].realised.unstack().pipe(lambda u: (u[5] - u[1]).mean())
                                / q_monthly[col].pred.unstack().pipe(lambda u: (u[5] - u[1]).mean()),
    "% months realised beta monotone": 100 * (q_monthly[col].realised.unstack().diff(axis=1).iloc[:, 1:] > 0).all(axis=1).mean(),
    "within-quintile sd of realised beta (avg)": q_frame.assign(Q=quintile_sort(q_frame, col)[0]["Q"])
                                                     .groupby(["date", "Q"])[TARGET].std().mean(),
    "return spread Q5-Q1 (per month)": q_monthly[col].ret_1m.unstack().pipe(lambda u: (u[5] - u[1]).mean()),
    "return spread NW t": nw_t(q_monthly[col].ret_1m.unstack().pipe(lambda u: u[5] - u[1]))} for col in comp})
comp_tab.to_csv("results/step6_quintiles_comparison.csv"); summ.to_csv("results/step6_sort_summary.csv")
display(comp_tab.round(4))
summ.round(4)

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
x = np.arange(1, 6); w = 0.27
for j, (col, c) in enumerate([("beta_36m", "#1f77b4"), ("beta_daily_12m", "#2ca02c"), ("beta_NN", "#d62728")]):
    ax[0].bar(x + (j - 1) * w, comp[col]["avg realised beta"], w, color=c, label=f"sorted on {col}")
ax[0].plot(x, qt.loc[[1, 2, 3, 4, 5], "avg_predicted_beta"], "k--o", ms=4, label="avg predicted NN beta")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"Q{i}" for i in x]); ax[0].axhline(1, color="grey", lw=0.8)
ax[0].set_title("Average realised future beta by sort quintile"); ax[0].legend(fontsize=8)
ax[1].bar(x, 100 * qt.loc[[1, 2, 3, 4, 5], "avg_next_month_return"], color="#d62728", alpha=0.8)
for xi, t in zip(x, qt.loc[[1, 2, 3, 4, 5], "return NW t-stat"]):
    ax[1].annotate(f"t={t:.1f}", (xi, 0), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"Q{i}" for i in x]); ax[1].set_ylabel("% per month")
ax[1].set_title("Average next-month return by NN-beta quintile")
ax[2].scatter(qt.loc[[1, 2, 3, 4, 5], "avg_realised_beta"], 100 * qt.loc[[1, 2, 3, 4, 5], "avg_next_month_return"],
              s=70, color="#d62728")
for q in x:
    ax[2].annotate(f"Q{q}", (qt.loc[q, "avg_realised_beta"], 100 * qt.loc[q, "avg_next_month_return"]),
                   xytext=(5, 4), textcoords="offset points")
b_, a_ = np.polyfit(qt.loc[[1, 2, 3, 4, 5], "avg_realised_beta"], 100 * qt.loc[[1, 2, 3, 4, 5], "avg_next_month_return"], 1)
xx = np.linspace(qt.avg_realised_beta.loc[[1, 5]].min(), qt.avg_realised_beta.loc[[1, 5]].max(), 10)
ax[2].plot(xx, a_ + b_ * xx, "k--", lw=1, label=f"slope {b_:.2f}% per unit beta")
ax[2].set_xlabel("average realised beta"); ax[2].set_ylabel("average next-month return (%)")
ax[2].set_title("Empirical security market line (test period)"); ax[2].legend(fontsize=8)
plt.tight_layout(); savefig("step6_quintiles"); plt.show()

# %%
# Return spread over time (cumulative Q5 - Q1, descriptive only)
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(spread.index, (1 + spread).cumprod() - 1, color="#d62728", label="cumulative Q5 - Q1 (NN beta), equal-weighted")
ax.plot(ret_.index, (1 + ret_[5]).cumprod() - 1, color="#ff7f0e", lw=1, ls="--", label="Q5 (high beta)")
ax.plot(ret_.index, (1 + ret_[1]).cumprod() - 1, color="#1f77b4", lw=1, ls="--", label="Q1 (low beta)")
ax.axhline(0, color="grey", lw=0.8); ax.legend(fontsize=8); ax.set_title("Cumulative next-month returns by portfolio formation month")
plt.tight_layout(); savefig("step6_cumulative"); plt.show()

# %%
# Who is in each NN-beta quintile? (characteristics at formation, averaged over months)
chars = q_rows.merge(test_df[["permno", "date", "log_market_cap", "idio_vol_daily_12m", "industry", "beta_daily_12m"]]
                     .rename(columns={"beta_daily_12m": "bd"}), on=["permno", "date"])
char_tab = chars.groupby("Q").agg(median_mcap_musd=("log_market_cap", lambda s: np.exp(s.median()) / 1e6),
                                  mean_idio_vol_daily=("idio_vol_daily_12m", "mean"),
                                  mean_daily_beta=("bd", "mean"))
ind_mix = pd.crosstab(chars.Q, chars.industry, normalize="index")
char_tab["top industries"] = [", ".join(f"{i} {v:.0%}" for i, v in ind_mix.loc[q].nlargest(3).items()) for q in char_tab.index]
ret_by_year = ret_.assign(year=ret_.index.year).groupby("year").mean()
char_tab

# %%
print("Average next-month return (per month) by NN quintile and formation year")
(100 * ret_by_year).round(2)

# %% [markdown]
# ### 6.1 Discussion
# **Risk separation: strong, and monotone in every month.**
# - Average realised future beta rises monotonically from Q1 to Q5 (0.63 → 0.91 → 1.06 → 1.22 → 1.53). The ordering is monotone in **all 60 test months**,
#   and the Q5–Q1 spread is never below 0.56.
# - The neural forecast separates future risk better than the historical estimates:
#   - realised-beta spread 0.90, against 0.82 for sorting on the daily beta and 0.62 on `beta_36m`;
#   - `beta_36m` is monotone in only 92% of months;
#   - the within-quintile dispersion of realised beta is lowest for the NN sort (0.42, against 0.43 and 0.47).
# - `beta_36m` sorts are dominated by estimation noise. Its sort-beta spread is 2.32, but only 27% of it shows up in realised beta. For the NN the figure is 78%.
# - **Calibration:** in Q3–Q5, realised betas are within about 6% of the predictions. The lowest quintile is under-predicted (0.46 predicted vs 0.63 realised).
#   It is dominated by micro-caps (median size \$66m against \$1.3–1.8bn in the other quintiles), whose low trailing daily betas partly reflect thin trading.
#   This is the same small-firm bias as in Step 5.
#
# **Relationship with subsequent returns: flat or even negative.**
# - Next-month returns do not rise with beta: 1.56% (Q1), 0.70%, 0.91%, 0.87%, 1.07% (Q5) per month. The Q5–Q1 spread is −0.48% per month
#   (Newey–West t = −1.1, positive in only 43% of months).
# - None of the quintile means or the spread is statistically significant. Five years (60 months, including the COVID crash and rebound) is far too short
#   to estimate expected returns precisely.
# - The high Q1 average comes almost entirely from **2020**, when micro-caps rebounded (6.8% per month). It reflects size, illiquidity and market-timing effects,
#   not a reward for low beta.
# - Economically, the CAPM predicts that higher beta should earn higher expected returns (a positive security-market-line slope, about the market premium).
#   The test period shows the well-known **flat security market line** (Black, Jensen & Scholes 1972; Fama & French 1992; Frazzini & Pedersen 2014,
#   "betting against beta"). Better beta *forecasts* do not create a positive beta–return relation that the data do not contain.
#
# **Comparison with my earlier CAPM beta study.**
# - There I sorted the largest ten firms per industry on 12-month historical betas (quintile averages 0.00 to 2.12). Next-year excess returns
#   were not monotonic, with a 5–1 spread of only 0.13 pp per month (0.11 value-weighted), which I read as the beta anomaly.
# - Step 2 shows that a 12-month beta passes only about 9% of its deviation into next year's beta, so much of that 0.00 to 2.12 spread was
#   estimation noise, and the true risk gap between those portfolios was far smaller than it appeared.
# - Here the MLP sort produces a genuine realised-beta gap (0.90), and returns are *still* flat. The beta anomaly is therefore not just an
#   artifact of noisy beta estimates. The samples differ (largest firms and excess returns in 1.2; all eligible firms and raw returns here),
#   so the comparison is qualitative.
# - The earlier study estimated betas from historical rolling windows. Here those historical estimates are the weakest forecasts: `beta_36m` quintiles
#   separate future beta far less well, because most of their spread is estimation noise.
# - The neural forecast (and even the trailing daily beta) produces much cleaner risk-sorted portfolios. For risk management and portfolio construction,
#   e.g. building a low-beta or beta-neutral portfolio, the forward-looking beta is the better input.
# - On the pricing side, the conclusion is the same as in the classical analysis: beta, however well measured, does not explain the cross-section of
#   average returns in this sample.
# - Caveats: these are equal-weighted descriptive sorts on an ex-post-selected sample (no delisting returns, and survival over the next 12 months
#   is required), not an implementable trading strategy.
