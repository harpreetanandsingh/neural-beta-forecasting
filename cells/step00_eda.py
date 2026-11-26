# %% [markdown]
# # Forecasting Market and Industry Betas Using Neural Networks
#
# **Question:** can a neural network use historical risk, return, size and industry information
# to forecast a firm's realised market beta over the next 12 months more accurately than a simple
# historical rolling-beta estimate?
#
# ## Step 0: Exploratory Data Analysis (EDA)
# Before modelling, we examine the panel's structure, the sample-selection flags, missing values,
# the target and predictor distributions, and the relationships between the historical betas and
# the future (target) beta. The EDA only describes the data. No modelling decision is fitted on
# validation or test data.

# %%
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

def savefig(name):
    plt.savefig(f"figures/{name}.png", bbox_inches="tight")

SPLITS = ["train", "validation", "test"]
SPLIT_COLORS = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

# %% [markdown]
# ### 0.1 Load the panel and the variable dictionary

# %%
df = pd.read_csv("data/neural_beta_panel_1996_2023.csv.gz",
                 parse_dates=["date", "target_start", "target_end", "source_date"])
defs = pd.read_csv("data/variable_definitions.csv")
print(f"Panel shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
print(f"Date range: {df.date.min().date()} to {df.date.max().date()}; "
      f"unique firms (permno): {df.permno.nunique():,}; unique months: {df.date.nunique()}")
print("Duplicate (permno, date) keys:", df.duplicated(["permno", "date"]).sum())

# %%
# The variable dictionary, grouped by role
defs.groupby("role")["variable"].apply(lambda s: ", ".join(s)).to_frame("variables")

# %%
NUM = defs.loc[defs.role == "numeric_predictor", "variable"].tolist()
CAT = defs.loc[defs.role == "categorical_predictor", "variable"].tolist()
IND_SUFFIX = ["nodur", "durbl", "manuf", "enrgy", "hitec", "telcm", "shops", "hlth", "utils", "other"]
IND_TARGETS = [f"beta_fwd_12m_{s}" for s in IND_SUFFIX]
TARGET = "beta_fwd_12m"
GROUPS = defs.loc[defs.role == "numeric_predictor"].groupby("group")["variable"].apply(list).to_dict()
print("Numeric predictors (12):", NUM)
print("Categorical predictor:", CAT)
print("Predictor groups:", GROUPS)

# %% [markdown]
# ### 0.2 Sample structure: splits, years and eligibility flags
# The panel keeps warm-up rows, the purge years (2012, 2017) and the 2023 forecast-only rows.
# Modelling uses only `eligible_core` rows in train/validation/test (and `eligible_multi` for Step 8).

# %%
split_tab = (df.groupby("split")
               .agg(rows=("permno", "size"), firms=("permno", "nunique"),
                    first_month=("date", "min"), last_month=("date", "max"),
                    eligible_core=("eligible_core", "sum"), eligible_multi=("eligible_multi", "sum"))
               .reindex(["train", "purge", "validation", "test", "forecast_only"]))
split_tab["core_share"] = split_tab.eligible_core / split_tab.rows
split_tab

# %%
core = df[df.eligible_core].copy()
core_tab = core.groupby("split").agg(firm_months=("permno", "size"), firms=("permno", "nunique"),
                                     months=("date", "nunique")).reindex(SPLITS)
core_tab["avg_firms_per_month"] = core_tab.firm_months / core_tab.months
print("Eligible-core sample by split (expected 589,239 / 138,516 / 170,650):")
core_tab

# %%
# Why are rows ineligible? Each reason is counted separately (a row can fail several)
base = df[df.split.isin(SPLITS)]
reasons = pd.DataFrame({
    "target beta_fwd_12m missing": base[TARGET].isna(),
    "beta_12m missing (warm-up)": base.beta_12m.isna(),
    "beta_24m missing (warm-up)": base.beta_24m.isna(),
    "beta_36m missing (warm-up)": base.beta_36m.isna(),
    "market cap missing/nonpositive": base.log_market_cap.isna(),
})
print(f"Train/validation/test rows: {len(base):,}; eligible_core: {base.eligible_core.sum():,}")
reasons.mean().to_frame("share of train/val/test rows").round(4)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
per_year = df.groupby(["year", "split"]).size().unstack(fill_value=0)
per_year_core = core.groupby("year").size()
ax[0].bar(per_year.index, per_year.sum(1), color="lightgrey", label="all rows")
ax[0].bar(per_year_core.index, per_year_core, color="#1f77b4", label="eligible_core rows")
for y in (2012, 2017):
    ax[0].axvspan(y - 0.5, y + 0.5, color="red", alpha=0.12)
ax[0].set_title("Firm-months per year (red = purge years)")
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k")); ax[0].legend()
firms_m = core.groupby(["date", "split"]).permno.nunique().unstack()
for s in SPLITS:
    ax[1].plot(firms_m.index, firms_m[s], color=SPLIT_COLORS[s], label=s)
ax[1].set_title("Eligible firms per month by split"); ax[1].legend()
plt.tight_layout(); savefig("eda_sample_over_time"); plt.show()

# %% [markdown]
# ### 0.3 Timing and leakage audit
# Predictors are dated at or before month-end $t$ (`date`). The target window starts at $t+1$
# and ends at $t+12$. The purge years must stop train and validation labels from overlapping the
# next split.

# %%
chk = {
    "source_date <= date (all rows)": (df.source_date <= df.date).all(),
    "target_start > date (core rows)": (core.target_start > core.date).all(),
    "target_start = first day of month t+1": (core.target_start == core.date + pd.offsets.Day(1)).all(),
    "target_end = month-end of t+12": (core.target_end == core.date + pd.offsets.MonthEnd(12)).all(),
}
for s in SPLITS:
    chk[f"{s}: last target_end"] = core.loc[core.split == s, "target_end"].max().date()
    chk[f"{s}: first date"] = core.loc[core.split == s, "date"].min().date()
pd.Series(chk, name="check").to_frame()

# %%
# Consistency check on the target definition. beta_fwd_12m at t uses daily returns in t+1..t+12,
# so it should equal the trailing daily beta (beta_daily_12m) recorded 12 months later.
# This confirms the target is future information. It is legitimate for the row at t+12 but must never be an input at t.
lead = df[["permno", "date", "beta_daily_12m"]].copy()
lead["date"] = lead["date"] - pd.offsets.MonthEnd(12)
m = df[["permno", "date", TARGET]].merge(lead, on=["permno", "date"], suffixes=("", "_at_t12")).dropna()
print(f"Matched rows: {len(m):,}; share where beta_fwd_12m(t) == beta_daily_12m(t+12): "
      f"{np.isclose(m[TARGET], m['beta_daily_12m']).mean():.4f}")

# %% [markdown]
# ### 0.4 Missing values
# Missing values in the full panel come mostly from warm-up periods. The rolling statistics need
# 12, 24 or 36 consecutive months of history.

# %%
cols = NUM + CAT + [TARGET] + IND_TARGETS + ["return_fwd_1m", "return_fwd_12m", "siccd"]
miss = pd.DataFrame({"full_panel": df[cols].isna().mean()})
for s in SPLITS:
    miss[f"core_{s}"] = core.loc[core.split == s, cols].isna().mean()
print("Share missing. Every predictor is complete in the eligible-core sample (all zeros below).")
miss.round(4)

# %% [markdown]
# ### 0.5 The target: future 12-month realised market beta

# %%
y = core[TARGET]
def dist_stats(s):
    return pd.Series({"n": s.count(), "mean": s.mean(), "std": s.std(), "min": s.min(),
                      "p0.1": s.quantile(.001), "p1": s.quantile(.01), "p25": s.quantile(.25),
                      "median": s.median(), "p75": s.quantile(.75), "p99": s.quantile(.99),
                      "p99.9": s.quantile(.999), "max": s.max(), "skew": s.skew(), "kurtosis": s.kurt(),
                      "share<0": (s < 0).mean(), "share>3": (s > 3).mean()})
target_stats = pd.DataFrame({s: dist_stats(core.loc[core.split == s, TARGET]) for s in SPLITS})
target_stats["all_core"] = dist_stats(y)
target_stats

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
bins = np.linspace(-1.5, 4, 111)
for s in SPLITS:
    ax[0].hist(core.loc[core.split == s, TARGET].clip(-1.5, 4), bins=bins, density=True,
               histtype="step", lw=1.5, color=SPLIT_COLORS[s], label=s)
ax[0].axvline(1, color="k", ls="--", lw=1); ax[0].legend()
ax[0].set_title("Distribution of beta_fwd_12m (clipped at [-1.5, 4] for display only)")
ax[1].hist(y, bins=200, color="grey", log=True)
ax[1].set_title("Full range, log-scaled counts (tails)")
from scipy import stats
qs = np.linspace(0.001, 0.999, 400)
ax[2].plot(stats.norm.ppf(qs, y.mean(), y.std()), np.quantile(y, qs), ".", ms=3)
lim = [y.quantile(.001), y.quantile(.999)]; ax[2].plot(lim, lim, "k--", lw=1)
ax[2].set_title("QQ plot vs normal"); ax[2].set_xlabel("normal quantile"); ax[2].set_ylabel("empirical quantile")
plt.tight_layout(); savefig("eda_target_distribution"); plt.show()

# %%
# The target's cross-sectional distribution over time
ts = core.groupby("date")[TARGET].agg(mean="mean", median="median",
                                      p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9),
                                      std="std")
ts = ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="M"))  # leave purge years blank
fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(ts.index, ts.p10, ts.p90, alpha=0.2, label="10th-90th percentile")
ax.plot(ts.index, ts["mean"], label="mean"); ax.plot(ts.index, ts["median"], label="median")
ax.axhline(1, color="k", ls="--", lw=1)
for s in SPLITS:
    d = core.loc[core.split == s, "date"]; ax.axvspan(d.min(), d.max(), color=SPLIT_COLORS[s], alpha=0.06)
ax.set_title("Cross-section of beta_fwd_12m by forecast month (shading = train / validation / test)")
ax.legend(loc="upper left"); plt.tight_layout(); savefig("eda_target_over_time"); plt.show()
print("Target cross-sectional mean by split:", core.groupby("split")[TARGET].mean().reindex(SPLITS).round(3).to_dict())

# %% [markdown]
# ### 0.6 Predictor distributions
# The summary statistics show which predictors are heavy-tailed. This motivates winsorising the
# predictors in Step 3, with thresholds fitted on the training sample only.

# %%
tr = core[core.split == "train"]
pred_stats = tr[NUM].describe(percentiles=[.001, .01, .5, .99, .999]).T
pred_stats["skew"] = tr[NUM].skew(); pred_stats["kurtosis"] = tr[NUM].kurt()
pred_stats["missing"] = tr[NUM].isna().mean()
pred_stats["group"] = [next(g for g, v in GROUPS.items() if c in v) for c in NUM]
print("Numeric predictors, TRAINING eligible-core sample")
pred_stats

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, c in zip(axes.ravel(), NUM):
    s = tr[c].dropna(); lo, hi = s.quantile([.005, .995])
    ax.hist(s.clip(lo, hi), bins=80, color="#1f77b4", alpha=0.8)
    ax.set_title(f"{c}\nskew={s.skew():.2f}, kurt={s.kurt():.1f}", fontsize=9)
    ax.tick_params(labelsize=8)
plt.suptitle("Predictor distributions, training sample (clipped at 0.5/99.5 pct for display only)", y=1.0)
plt.tight_layout(); savefig("eda_predictor_hists"); plt.show()

# %%
# Do predictor distributions shift across splits? Medians by split
core.groupby("split")[NUM].median().reindex(SPLITS).T

# %% [markdown]
# ### 0.7 Industry composition
# `industry` is the Fama-French 10 mapping of the firm's historical SIC code, plus `Unknown` where
# the SIC code is missing or invalid.

# %%
ind = (core.groupby(["industry", "split"]).size().unstack().reindex(columns=SPLITS))
ind_share = ind / ind.sum()
ind_beta = core.groupby("industry")[TARGET].agg(["mean", "median", "std"])
ind_summary = pd.concat([ind, ind_share.add_suffix("_share"), ind_beta.add_prefix("beta_fwd_")], axis=1)
ind_summary.sort_values("train", ascending=False)

# %%
print("Rows labelled 'Unknown' by split:", core.loc[core.industry == "Unknown"].split.value_counts().to_dict())
print("-> 'Unknown' never appears in training, so the one-hot encoder (fit on train) must ignore unseen categories.")

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
order = ind_beta.sort_values("median").index
data = [core.loc[core.industry == i, TARGET].values for i in order]
ax[0].boxplot(data, labels=order, showfliers=False, vert=True)
ax[0].axhline(1, color="k", ls="--", lw=1); ax[0].tick_params(axis="x", rotation=45)
ax[0].set_title("beta_fwd_12m by industry (whiskers 1.5 IQR, outliers hidden)")
ind_share.loc[order].plot.bar(ax=ax[1], color=[SPLIT_COLORS[s] for s in SPLITS])
ax[1].set_title("Industry share of firm-months by split"); ax[1].tick_params(axis="x", rotation=45)
ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout(); savefig("eda_industry"); plt.show()

# %% [markdown]
# ### 0.8 How informative are historical betas about future beta?
# If beta were perfectly persistent, the points would lie on the 45° line. Scatter below
# 45° shows mean reversion, where extreme historical betas are followed by less extreme betas.

# %%
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, c in zip(axes, ["beta_12m", "beta_24m", "beta_36m", "beta_daily_12m"]):
    m = tr[[c, TARGET]].dropna()
    ax.hexbin(m[c].clip(-2, 5), m[TARGET].clip(-2, 5), gridsize=70, bins="log", cmap="viridis", mincnt=1)
    ax.plot([-2, 5], [-2, 5], "r--", lw=1)
    b = np.polyfit(m[c].clip(m[c].quantile(.005), m[c].quantile(.995)), m[TARGET], 1)
    xs = np.array([-2, 5]); ax.plot(xs, b[1] + b[0] * xs, "k-", lw=1.5)
    ax.set_title(f"{c}\ncorr={m[c].corr(m[TARGET]):.3f}, OLS slope={b[0]:.2f}", fontsize=9)
    ax.set_xlabel(c); ax.set_ylabel(TARGET); ax.set_xlim(-2, 5); ax.set_ylim(-2, 5)
plt.suptitle("Training sample: historical beta vs future realised beta (red = 45° line, black = fitted line)", y=1.02)
plt.tight_layout(); savefig("eda_persistence"); plt.show()

# %%
# Mean reversion in bins: average future beta within deciles of each historical beta (training)
rows = {}
for c in ["beta_12m", "beta_36m", "beta_daily_12m"]:
    m = tr[[c, TARGET]].dropna()
    dec = pd.qcut(m[c], 10, labels=False)
    rows[c] = pd.DataFrame({"hist_mean": m.groupby(dec)[c].mean(), "fwd_mean": m.groupby(dec)[TARGET].mean()})
pd.concat(rows, axis=1).rename_axis("decile")

# %%
# Size and the level of daily vs monthly betas (all eligible-core rows, by size quintile)
q = pd.qcut(core.log_market_cap, 5, labels=[f"Q{i} {'(small)' if i==1 else '(large)' if i==5 else ''}" for i in range(1, 6)])
core.groupby(q)[["beta_12m", "beta_36m", "beta_daily_12m", TARGET]].mean().round(3)

# %%
# Average target by forecast year, showing time variation and the level shift between train and later splits
core.groupby("year")[TARGET].agg(["mean", "median", "std"]).round(3).T

# %% [markdown]
# ### 0.9 Correlations: predictors with the target, and among predictors

# %%
corr_cols = NUM + [TARGET]
pear = tr[corr_cols].corr()[TARGET].drop(TARGET)
spear = tr[corr_cols].corr(method="spearman")[TARGET].drop(TARGET)
corr_tab = pd.DataFrame({"pearson": pear, "spearman": spear}).sort_values("spearman", key=abs, ascending=False)
print("Correlation of each predictor with beta_fwd_12m (training sample)")
corr_tab

# %%
C = tr[corr_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(C))); ax.set_yticklabels(C.columns, fontsize=8)
for i in range(len(C)):
    for j in range(len(C)):
        ax.text(j, i, f"{C.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(C.iat[i, j]) > 0.6 else "black")
plt.colorbar(im, fraction=0.046); ax.set_title("Spearman correlation matrix (training sample)"); ax.grid(False)
plt.tight_layout(); savefig("eda_corr_matrix"); plt.show()

# %%
# Size and volatility relationship with future beta: mean target by decile (training)
out = {}
for c in ["log_market_cap", "vol_12m", "idio_vol_daily_12m", "mom_12_2"]:
    m = tr[[c, TARGET]].dropna(); dec = pd.qcut(m[c], 10, labels=False, duplicates="drop") + 1
    out[c] = m.groupby(dec)[TARGET].mean()
fig, ax = plt.subplots(figsize=(8, 4))
for c, s in out.items():
    ax.plot(s.index, s.values, marker="o", label=c)
ax.axhline(1, color="k", ls="--", lw=1); ax.set_xticks(range(1, 11))
ax.set_xlabel("predictor decile (1 = lowest)"); ax.set_ylabel("mean beta_fwd_12m")
ax.set_title("Future beta across predictor deciles (training)"); ax.legend(fontsize=8)
plt.tight_layout(); savefig("eda_deciles"); plt.show()
pd.DataFrame(out).rename_axis("decile")

# %% [markdown]
# ### 0.10 The ten industry-beta targets (Step 8 sample)

# %%
multi = df[df.eligible_multi]
print("eligible_multi firm-months by split:", multi.split.value_counts().reindex(SPLITS).to_dict())
print("eligible_core rows without full industry targets:", len(core) - len(multi))
ind_t = multi[IND_TARGETS + [TARGET]]
ind_t.describe(percentiles=[.01, .5, .99]).T.assign(skew=ind_t.skew())

# %%
Ct = ind_t.rename(columns=lambda c: c.replace("beta_fwd_12m_", "").replace("beta_fwd_12m", "MARKET")).corr()
fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(Ct, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(Ct))); ax.set_xticklabels(Ct.columns, rotation=90)
ax.set_yticks(range(len(Ct))); ax.set_yticklabels(Ct.columns)
for i in range(len(Ct)):
    for j in range(len(Ct)):
        ax.text(j, i, f"{Ct.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(Ct.iat[i, j]) > 0.6 else "black")
plt.colorbar(im, fraction=0.046); ax.grid(False)
ax.set_title("Correlation among the 10 future industry betas and the market beta")
plt.tight_layout(); savefig("eda_industry_target_corr"); plt.show()

# %% [markdown]
# ### 0.11 Outcome used in Step 6: next-month return

# %%
r1 = core.groupby("split").return_fwd_1m.agg(n="size", missing=lambda s: s.isna().sum(),
                                             mean="mean", median="median", std="std").reindex(SPLITS)
r1["missing_share"] = r1.missing / r1.n
r1
