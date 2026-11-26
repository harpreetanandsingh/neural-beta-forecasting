# %% [markdown]
# ## Step 1: The Prediction Problem and the Modelling Sample
#
# **Prediction problem.** At the close of month $t$, we use information dated at or before $t$ to forecast
# the firm's realised market beta over months $t+1$ to $t+12$:
# $$\beta^R_{i,t+12}=\frac{\mathrm{Cov}_{d\in D(t)}(R_{i,d},R_{m,d})}{\mathrm{Var}_{d\in D(t)}(R_{m,d})},\qquad
# \widehat\beta^{NN}_{i,t+12}=f_\theta(X_{i,t}).$$
# The target (`beta_fwd_12m`) uses **future** daily returns, and every input must be known at $t$.
#
# **Modelling sample.** Rows with `eligible_core == True`, all of which fall in the train (1996–2011),
# validation (2013–2016) or test (2018–2022) split. The purge years 2012 and 2017, and the 2023 forecast-only
# rows, are never used for fitting or evaluation. The split is chronological; no row is assigned at random.

# %%
sample = df.loc[df.eligible_core].copy()
assert set(sample.split.unique()) == set(SPLITS), "eligible_core must only contain train/validation/test"
assert not sample.year.isin([2012, 2017, 2023]).any(), "purge / forecast-only years must be excluded"
assert (sample.split.map({"train": 0, "validation": 1, "test": 2}).groupby(sample.year).nunique() == 1).all()
expected = {"train": 589_239, "validation": 138_516, "test": 170_650}
assert sample.split.value_counts().to_dict() == expected, "counts differ from the expected sample"
print("Modelling sample OK:", {s: f"{n:,}" for s, n in expected.items()})

# %% [markdown]
# ### 1.1 Inputs, targets and excluded columns
# Only the 12 numeric predictors and the `industry` category may enter the model. Every other column is either
# an identifier, a future outcome, a target-audit field or a sample-control flag.

# %%
FEATURES_NUM = GROUPS["risk"] + GROUPS["returns_and_size"]
FEATURES_CAT = ["industry"]
EXCLUDED = [c for c in df.columns if c not in FEATURES_NUM + FEATURES_CAT]
roles = defs.set_index("variable")[["role", "group"]]
inputs_tab = roles.loc[FEATURES_NUM + FEATURES_CAT].assign(used_as="MODEL INPUT")
excl_tab = roles.reindex(EXCLUDED).assign(used_as="excluded")
pd.concat([inputs_tab, excl_tab])

# %%
# Look-ahead guard: none of the forbidden column types may be among the inputs
forbidden = defs.loc[defs.role.str.contains("NEVER|identifier|sample control|audit|auxiliary|categorical source"), "variable"]
assert not set(forbidden) & set(FEATURES_NUM + FEATURES_CAT)
print(f"{len(FEATURES_NUM)} numeric + {len(FEATURES_CAT)} categorical inputs; "
      f"{len(forbidden)} forbidden columns, none of which are inputs.")

# %% [markdown]
# ### 1.2 Verifying that predictors are dated at or before $t$ and the target uses future returns
# Three checks:
# 1. **Timing fields.** `source_date ≤ date`, and the target window starts in month $t+1$ and ends at the end of $t+12$ (done in Step 0.3).
# 2. **Recomputing the predictors.** For a random sample of 500 firms, we rebuild `beta_12m`, `beta_24m`, `beta_36m`,
#    `vol_12m`, `market_vol_12m`, `mom_12_2` and `ret_3m` from the monthly `ret` and `market_ret` of months
#    $\le t$ only. If the rebuilt values match the supplied values, the supplied predictors cannot contain
#    any information from after $t$.
# 3. **The target is future information.** `beta_fwd_12m`$(t)$ = `beta_daily_12m`$(t+12)$ (Step 0.3), and the
#    target is far less correlated with the *contemporaneous* daily beta than with the one 12 months ahead.

# %%
rng = np.random.default_rng(SEED)
firms = rng.choice(df.permno.unique(), 500, replace=False)
recomp = []
for p, g in df[df.permno.isin(firms)].groupby("permno"):
    g = g.set_index(g.date.dt.to_period("M")).sort_index()
    g = g.reindex(pd.period_range(g.index.min(), g.index.max(), freq="M"))  # gaps -> NaN -> window invalid
    r, m = g["ret"], g["market_ret"]
    out = pd.DataFrame(index=g.index)
    for w in (12, 24, 36):  # OLS slope with intercept over t-w+1..t (needs w consecutive complete months)
        out[f"beta_{w}m"] = r.rolling(w).cov(m) / m.rolling(w).var()
    out["vol_12m"] = r.rolling(12).std()
    out["market_vol_12m"] = m.rolling(12).std()
    out["mom_12_2"] = np.expm1(np.log1p(r).shift(1).rolling(11).sum())   # t-11..t-1, skips month t
    out["ret_3m"] = np.expm1(np.log1p(r).rolling(3).sum())              # t-2..t
    out = out.add_suffix("_re").join(g[[c for c in out.columns]]).dropna(how="all")
    recomp.append(out)
recomp = pd.concat(recomp)
ver = []
for c in ["beta_12m", "beta_24m", "beta_36m", "vol_12m", "market_vol_12m", "mom_12_2", "ret_3m"]:
    both = recomp[[c, c + "_re"]].dropna()
    ver.append({"predictor": c, "rows compared": len(both),
                "max |supplied - recomputed|": (both[c] - both[c + "_re"]).abs().max(),
                "share matching (tol 1e-4)": np.isclose(both[c], both[c + "_re"], atol=1e-4).mean()})
verify_tab = pd.DataFrame(ver).set_index("predictor")
verify_tab

# %%
# Target vs the same firm's daily beta at t (known) and at t+12 (future), eligible rows
lead = df[["permno", "date", "beta_daily_12m"]].copy()
lead["date"] -= pd.offsets.MonthEnd(12)
tmp = sample[["permno", "date", TARGET, "beta_daily_12m"]].merge(
    lead.rename(columns={"beta_daily_12m": "beta_daily_12m_at_t+12"}), on=["permno", "date"], how="left")
print("corr(target, beta_daily_12m at t)      =", round(tmp[TARGET].corr(tmp["beta_daily_12m"]), 4))
print("corr(target, beta_daily_12m at t + 12) =", round(tmp[TARGET].corr(tmp["beta_daily_12m_at_t+12"]), 4),
      "  <- identical series: the target is the future daily beta")

# %% [markdown]
# ### 1.3 Firms and firm-month observations by split

# %%
sample_tab = sample.groupby("split").agg(
    first_month=("date", "min"), last_month=("date", "max"), months=("date", "nunique"),
    firm_months=("permno", "size"), firms=("permno", "nunique")).reindex(SPLITS)
sample_tab["firms_per_month_avg"] = (sample.groupby(["split", "date"]).size().groupby("split").mean()).round(0)
sample_tab["share_of_sample"] = sample_tab.firm_months / len(sample)
sample_tab.loc["total"] = [sample.date.min(), sample.date.max(), sample.date.nunique(), len(sample),
                           sample.permno.nunique(), np.nan, 1.0]
sample_tab.to_csv("results/step1_sample_by_split.csv")
sample_tab

# %%
# The split is by time, so the same firm can appear in several splits (at different dates). This is fine,
# because no future information about any firm enters an earlier split's inputs or labels.
fs = {s: set(sample.loc[sample.split == s, "permno"]) for s in SPLITS}
print(f"Firms in train & validation: {len(fs['train'] & fs['validation']):,}; "
      f"train & test: {len(fs['train'] & fs['test']):,}; test firms never seen in training: "
      f"{len(fs['test'] - fs['train']):,} ({len(fs['test'] - fs['train'])/len(fs['test']):.1%})")

# %% [markdown]
# ### 1.4 Distribution of the target

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
bins = np.linspace(-1.5, 3.5, 101)
for s in SPLITS:
    v = sample.loc[sample.split == s, TARGET]
    ax[0].hist(v.clip(-1.5, 3.5), bins=bins, density=True, histtype="step", lw=1.6, color=SPLIT_COLORS[s],
               label=f"{s}: mean {v.mean():.2f}, sd {v.std():.2f}, skew {v.skew():.2f}")
ax[0].axvline(1, color="k", ls="--", lw=1)
ax[0].set_xlabel("beta_fwd_12m (values beyond [-1.5, 3.5] piled at the edges for display)")
ax[0].set_ylabel("density"); ax[0].legend(fontsize=8.5); ax[0].set_title("Future 12-month realised market beta")
ax[1].boxplot([sample.loc[sample.split == s, TARGET] for s in SPLITS], labels=SPLITS, whis=(0.5, 99.5),
              flierprops=dict(marker=".", ms=2, alpha=0.3))
ax[1].axhline(1, color="k", ls="--", lw=1)
ax[1].set_title("Full range (whiskers 0.5th-99.5th pct, dots = extremes)")
plt.tight_layout(); savefig("step1_target_distribution"); plt.show()

# %%
tail = pd.DataFrame({s: {
    "n": (v := sample.loc[sample.split == s, TARGET]).size,
    "mean": v.mean(), "median": v.median(), "skew": v.skew(), "excess kurtosis": v.kurt(),
    "share < 0": (v < 0).mean(), "share > 2": (v > 2).mean(), "share > 3": (v > 3).mean(),
    "count |beta| > 5": int((v.abs() > 5).sum()), "min": v.min(), "max": v.max()} for s in SPLITS})
tail.to_csv("results/step1_target_stats.csv")
tail

# %%
# Who are the extreme observations? Firm size of |beta| > 4 rows compared with the rest of the sample
ext = sample[TARGET].abs() > 4
print(f"Rows with |beta_fwd_12m| > 4: {ext.sum()} ({ext.mean():.3%})")
print(f"Median market cap: extreme rows ${np.exp(sample.loc[ext, 'log_market_cap'].median())/1e6:,.0f}m "
      f"vs all rows ${np.exp(sample.log_market_cap.median())/1e6:,.0f}m")
print(f"Median trailing idiosyncratic daily vol: extreme {sample.loc[ext, 'idio_vol_daily_12m'].median():.3f} "
      f"vs all {sample.idio_vol_daily_12m.median():.3f}")
sample.loc[ext].groupby("split").size().reindex(SPLITS, fill_value=0).to_frame("n_extreme")

# %% [markdown]
# **Discussion of the target distribution.**
# - Future beta is centred slightly below 1 in training (mean 0.88) and near 1 in the validation and test years (1.07).
#   Median beta rises in step with firm size, which grows from a median of about \$280m to about \$1bn.
# - The distribution is **moderately right-skewed** (skew 0.2–0.5). High-beta firms can reach 2.5–3.5, while negative
#   betas are uncommon (5.5% of training rows, 1.2% of test rows).
# - The tails are **fat, especially in the test period** (excess kurtosis 7.3, range −10.9 to +9.3). The extreme
#   values are rare (212 rows with |beta| > 4, 0.02% of the sample). They come disproportionately from the test years
#   (96 of the 212) and from smaller, very volatile firms (median cap \$225m vs \$404m; idiosyncratic daily volatility
#   2.5 times the sample median). The most extreme cases are micro-caps worth about \$10m. For such stocks, a few very large daily
#   moves dominate a 12-month OLS slope, so these labels mostly reflect estimation noise rather than true economic exposure.
# - Consequences:
#   - squared-error metrics (MSE, RMSE, R²_OOS) will be sensitive to a few rows, so MAE is also reported;
#   - by design, **targets are not clipped** for training or scoring.
