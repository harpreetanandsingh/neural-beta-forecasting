# %% [markdown]
# ## Step 3: Predictor Selection and Leakage-Free Preprocessing
#
# ### 3.1 Predictor groups and their economic rationale
# | Group | Variables | Why it should help forecast future beta |
# |---|---|---|
# | **Historical risk** | `beta_12m`, `beta_24m`, `beta_36m`, `beta_daily_12m`, `vol_12m`, `idio_vol_daily_12m`, `market_vol_12m` | Beta is persistent, so past betas are the direct starting point. Having several windows and frequencies lets the model *combine* and *shrink* them: the daily beta is precise and the monthly betas are robust to thin trading. Total and idiosyncratic volatility show how noisy the historical estimates are, so how much they should be shrunk. They also relate to leverage and distress. Market volatility captures the market regime, since betas behave differently in turbulent markets. |
# | **Returns & size** | `ret`, `market_ret`, `mom_12_2`, `ret_3m`, `log_market_cap` | Size is linked to beta through business maturity and diversification, and it corrects the thin-trading bias that pushes small firms' daily betas towards 0 (Step 0.8). Past returns change a firm's leverage: after a price fall, equity is a smaller share of the firm, so equity beta rises. They also mark growth or momentum stocks, whose risk profile is changing. The market return describes the current market state. |
# | **Industry** | one-hot `industry` (FF10) | Industries differ in the cyclicality of their cash flows and in operating leverage. Utilities and staples have low betas; energy and tech have high betas (Step 0.7). Industry is also a stable anchor to shrink noisy firm-level estimates towards, like a Bayesian prior. |
#
# ### 3.2 Preprocessing rules (all fitted on the **training** split only)
# 1. **Missing values:** each numeric predictor is filled with its training median, and industry is filled with "Unknown". The eligible sample has
#    no missing predictors (Step 0.4), so the imputer is kept only as a safeguard. We show below that it changes 0 values.
# 2. **Winsorisation (predictors only):** firm-level predictors are clipped at their **training 1st and 99th percentiles**. This limits the
#    influence of extreme values such as `beta_12m` = 36 or `ret_3m` = +4,140% on the standardised inputs and on the gradients.
#    `market_ret` and `market_vol_12m` are **not** clipped: they are the same for every firm in a month, have no outliers
#    (kurtosis < 1), and clipping them would only remove information about crisis months. **Targets and benchmark forecasts are never clipped.**
# 3. **Standardisation:** subtract the training mean and divide by the training standard deviation (computed after clipping).
# 4. **Industry:** one-hot encoding fitted on the training categories. `Unknown` never occurs in training, so it maps to all zeros
#    (`handle_unknown="ignore"`), meaning "no industry information".
# 5. **Never inputs:** identifiers, future betas and returns, target dates, observation counts, coverage, split labels and eligibility flags.

# %%
from sklearn.preprocessing import OneHotEncoder

MARKET_LEVEL = ["market_ret", "market_vol_12m"]   # identical across firms within a month -> not clipped

class BetaPreprocessor:
    """Impute -> clip (1st/99th pct) -> standardise numeric predictors; one-hot encode industry.
    Every statistic is estimated in .fit() on the training rows only; .transform() only applies them."""
    def __init__(self, num_cols, cat_cols, clip_q=(0.01, 0.99), no_clip=()):
        self.num_cols, self.cat_cols, self.clip_q, self.no_clip = list(num_cols), list(cat_cols), clip_q, set(no_clip)

    def fit(self, train_df):
        X = train_df[self.num_cols]
        self.median_ = X.median()
        Xf = X.fillna(self.median_)
        self.lo_ = Xf.quantile(self.clip_q[0]); self.hi_ = Xf.quantile(self.clip_q[1])
        for c in self.no_clip:
            self.lo_[c], self.hi_[c] = -np.inf, np.inf
        Xc = Xf.clip(self.lo_, self.hi_, axis=1)
        self.mean_, self.std_ = Xc.mean(), Xc.std(ddof=0)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.ohe_.fit(train_df[self.cat_cols].fillna("Unknown"))
        self.feature_names_ = self.num_cols + list(self.ohe_.get_feature_names_out(self.cat_cols))
        return self

    def transform(self, d):
        X = d[self.num_cols]
        self.last_imputed_ = int(X.isna().sum().sum())
        Xc = X.fillna(self.median_).clip(self.lo_, self.hi_, axis=1)
        self.last_clip_share_ = ((X < self.lo_) | (X > self.hi_)).mean()
        Z = ((Xc - self.mean_) / self.std_).to_numpy(np.float32)
        return np.hstack([Z, self.ohe_.transform(d[self.cat_cols].fillna("Unknown"))]).astype(np.float32)

train_df = sample[sample.split == "train"]
val_df, test_df = sample[sample.split == "validation"], sample[sample.split == "test"]

prep = BetaPreprocessor(FEATURES_NUM, FEATURES_CAT, no_clip=MARKET_LEVEL).fit(train_df)
X_train = prep.transform(train_df); imp_tr, clip_tr = prep.last_imputed_, prep.last_clip_share_
X_val = prep.transform(val_df);     imp_va, clip_va = prep.last_imputed_, prep.last_clip_share_
X_test = prep.transform(test_df);   imp_te, clip_te = prep.last_imputed_, prep.last_clip_share_
y_train, y_val, y_test = (d[TARGET].to_numpy(np.float32) for d in (train_df, val_df, test_df))
FEATURE_NAMES = prep.feature_names_
P = X_train.shape[1]
print(f"Input dimension p = {P}  (12 numeric + {P - 12} industry dummies)")
print("Industry categories learned from training:", list(prep.ohe_.categories_[0]))
print(f"Shapes: train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
print(f"Values imputed: train {imp_tr}, validation {imp_va}, test {imp_te}")

# %%
# Fitted preprocessing parameters (all estimated on the training split) and the share of rows clipped in each split
params = pd.DataFrame({"group": [next(g for g, v in GROUPS.items() if c in v) for c in FEATURES_NUM],
                       "train median (imputation)": prep.median_, "clip low (train p1)": prep.lo_,
                       "clip high (train p99)": prep.hi_, "train mean (after clip)": prep.mean_,
                       "train sd (after clip)": prep.std_,
                       "clipped % train": 100 * clip_tr, "clipped % validation": 100 * clip_va,
                       "clipped % test": 100 * clip_te}, index=FEATURES_NUM)
params.to_csv("results/step3_preprocessing_params.csv")
params

# %%
# Sanity checks on the transformed data
Ztr = pd.DataFrame(X_train[:, :12], columns=FEATURES_NUM)
chk3 = pd.DataFrame({"train mean": Ztr.mean(), "train sd": Ztr.std(ddof=0),
                     "val mean": X_val[:, :12].mean(0), "test mean": X_test[:, :12].mean(0),
                     "raw kurtosis (train)": train_df[FEATURES_NUM].kurt().values,
                     "kurtosis after clip (train)": Ztr.kurt().values})
assert np.allclose(chk3["train mean"], 0, atol=1e-4) and np.allclose(chk3["train sd"], 1, atol=1e-3)
assert np.isfinite(X_train).all() and np.isfinite(X_val).all() and np.isfinite(X_test).all()
chk3

# %%
# Industry dummies: each row has exactly one 1, except 'Unknown' rows (all zeros)
D_te = X_test[:, 12:]
print("Row sums of industry dummies (test):", pd.Series(D_te.sum(1)).value_counts().to_dict(),
      "| Unknown rows in test:", int((test_df.industry == "Unknown").sum()))
assert ((D_te.sum(1) == 0) == (test_df.industry == "Unknown").to_numpy()).all()

# Leakage guard: the model inputs are exactly the 12 numeric predictors + industry dummies
assert FEATURE_NAMES[:12] == FEATURES_NUM and all(f.startswith("industry_") for f in FEATURE_NAMES[12:])
assert not set(FEATURE_NAMES) & set(forbidden)

# Column indices for each predictor group (used by the permutation importance in Step 7)
GROUP_IDX = {"historical risk": [FEATURE_NAMES.index(c) for c in GROUPS["risk"]],
             "returns & size": [FEATURE_NAMES.index(c) for c in GROUPS["returns_and_size"]],
             "industry": list(range(12, P))}
print({g: [FEATURE_NAMES[i] for i in idx] for g, idx in GROUP_IDX.items()})

# %% [markdown]
# **Observations on preprocessing.**
# - p = 22: 12 standardised numeric predictors plus 10 industry dummies. `Unknown` gets no column; its 149 test and 5 validation rows are all zeros.
# - The imputer changed 0 values in every split, because the eligibility filter already requires complete histories.
# - Clipping affects about 2% of training rows per firm-level variable by construction (1% in each tail). It cuts kurtosis sharply,
#   e.g. `ret_3m` from 616 to 3.5 and `ret` from 152 to 2.7, without discarding any rows.
# - Validation and test rows are clipped at the *training* thresholds, so their clipped shares differ. The largest difference is
#   `log_market_cap` in test, clipped in 3.9% of rows, almost all at the top: firms above the training p99 of about \$60bn. Mega-caps grew
#   after 2011. Those firms' future beta averages 0.96, so treating them as "\$60bn firms" loses little information.
# - The shifts in standardised means reflect the regime differences described in Step 0:
#   - validation `market_vol_12m` is 0.9 sd below the training mean, because 2013–16 was calm;
#   - validation and test firms are larger (+0.4 to +0.5 sd) and have higher trailing daily betas (+0.3 sd).
#   The network must extrapolate somewhat, which is a realistic feature of forecasting over time.
