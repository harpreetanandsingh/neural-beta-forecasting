# %% [markdown]
# ## Step 7: Predictor Importance (Group Permutation on the Test Data)
# 1. Compute the fitted MLP's test RMSE on the unaltered test inputs.
# 2. Shuffle **one predictor group** across the test observations, using the **same row permutation for every column in the group**
#    (including all 10 industry dummies, so every row still has a valid one-hot industry). The other groups are left unchanged.
# 3. Recompute the RMSE with the *same* fitted model. There is no refitting and nothing is tuned on these results.
# 4. Repeat with 5 different permutations and report the mean and standard deviation of the RMSE increase.
#
# Shuffling breaks the link between that group and the target while keeping the group's distribution (and the correlations *within* the group) intact.
# A large RMSE increase means the model relies on that group. This measures how much the *fitted model* uses the information,
# which is not a causal effect of the underlying characteristic.

# %%
N_REP = 5
base_pred = predict(nn_model, X_test).ravel()
base_rmse = np.sqrt(np.mean((base_pred - y_test) ** 2)); base_mae = np.mean(np.abs(base_pred - y_test))
print(f"Original test RMSE {base_rmse:.4f} (MAE {base_mae:.4f})")

def permute_cols(X, idx, rng):
    Xp = X.copy()
    perm = rng.permutation(len(X))                      # ONE permutation shared by all columns in the group
    Xp[:, idx] = X[perm][:, idx]
    return Xp

imp_rows = []
for g, idx in GROUP_IDX.items():
    rng = np.random.default_rng(SEED)
    for r in range(N_REP):
        pp = predict(nn_model, permute_cols(X_test, idx, rng)).ravel()
        imp_rows.append({"group": g, "rep": r, "RMSE": np.sqrt(np.mean((pp - y_test) ** 2)),
                         "MAE": np.mean(np.abs(pp - y_test))})
imp = pd.DataFrame(imp_rows)
imp["dRMSE"] = imp.RMSE - base_rmse; imp["dMAE"] = imp.MAE - base_mae
imp_tab = imp.groupby("group").agg(n_columns=("group", lambda s: len(GROUP_IDX[s.iloc[0]])),
                                   mean_RMSE_permuted=("RMSE", "mean"), mean_RMSE_increase=("dRMSE", "mean"),
                                   sd_RMSE_increase=("dRMSE", "std"), min_increase=("dRMSE", "min"),
                                   max_increase=("dRMSE", "max"), mean_MAE_increase=("dMAE", "mean"))
imp_tab["% increase in RMSE"] = 100 * imp_tab.mean_RMSE_increase / base_rmse
imp_tab = imp_tab.sort_values("mean_RMSE_increase", ascending=False)
imp_tab.to_csv("results/step7_group_importance.csv")
imp_tab

# %%
# Supplementary: permuting each numeric predictor on its own (industry block kept as one unit), same procedure.
# Shows which variables inside each group drive its importance.
var_rows = []
singles = {c: [FEATURE_NAMES.index(c)] for c in FEATURES_NUM} | {"industry (all dummies)": GROUP_IDX["industry"]}
for v, idx in singles.items():
    rng = np.random.default_rng(SEED)
    d_ = []
    for r in range(N_REP):
        pp = predict(nn_model, permute_cols(X_test, idx, rng)).ravel()
        d_.append(np.sqrt(np.mean((pp - y_test) ** 2)) - base_rmse)
    grp = "industry" if v.startswith("industry") else ("historical risk" if v in GROUPS["risk"] else "returns & size")
    var_rows.append({"variable": v, "group": grp, "mean_RMSE_increase": np.mean(d_), "sd": np.std(d_, ddof=1)})
var_tab = pd.DataFrame(var_rows).set_index("variable").sort_values("mean_RMSE_increase", ascending=False)
var_tab["% increase"] = 100 * var_tab.mean_RMSE_increase / base_rmse
var_tab.to_csv("results/step7_variable_importance.csv")
print("Sum of single-variable increases within historical risk:",
      round(var_tab.loc[var_tab.group == "historical risk", "mean_RMSE_increase"].sum(), 4),
      "vs group permutation:", round(imp_tab.loc["historical risk", "mean_RMSE_increase"], 4))
var_tab

# %%
gcol = {"historical risk": "#d62728", "returns & size": "#1f77b4", "industry": "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].barh(imp_tab.index[::-1], imp_tab.mean_RMSE_increase[::-1], xerr=imp_tab.sd_RMSE_increase[::-1],
           color=[gcol[g] for g in imp_tab.index[::-1]], capsize=4)
for i, (g, r) in enumerate(imp_tab.iloc[::-1].iterrows()):
    ax[0].annotate(f"+{r.mean_RMSE_increase:.3f} ({r['% increase in RMSE']:.0f}%)", (r.mean_RMSE_increase, i),
                   xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
ax[0].set_xlabel(f"increase in test RMSE (base {base_rmse:.3f}); mean ± sd over {N_REP} permutations")
ax[0].set_title("Group permutation importance"); ax[0].set_xlim(0, imp_tab.mean_RMSE_increase.max() * 1.4)
vt_ = var_tab.iloc[::-1]
ax[1].barh(vt_.index, vt_.mean_RMSE_increase, xerr=vt_.sd, color=[gcol[g] for g in vt_.group], capsize=3)
ax[1].set_xlabel("increase in test RMSE"); ax[1].set_title("Supplementary: single-variable permutation importance")
ax[1].tick_params(axis="y", labelsize=8.5)
plt.tight_layout(); savefig("step7_importance"); plt.show()

# %% [markdown]
# ### 7.1 Discussion
# **Ranking of groups** (test RMSE 0.443 without any shuffling):
# - **Historical risk:** +0.153 (+35%).
# - **Returns & size:** +0.075 (+17%).
# - **Industry:** +0.004 (+1%).
#
# The ranking is identical for MAE. It is extremely stable across permutations (sd ≤ 0.001), so the differences are not random-draw noise.
#
# **Economic plausibility.**
# - **Historical risk matters most, and almost all of it is `beta_daily_12m`** (+0.122 on its own). This is exactly what economics and Steps 0–2 lead us to expect:
#   beta is persistent, and the trailing daily beta is the most precise estimate of the same quantity we are forecasting. Idiosyncratic volatility is next in the group (+0.010).
#   It tells the network how noisy that estimate is, and so how much to shrink it. The monthly betas add little once the daily beta is known
#   (`beta_36m` +0.008; 12m/24m about +0.001). They are noisy and largely redundant.
# - **Returns & size is driven by `log_market_cap`** (+0.066, the second most important variable overall). Size is how the model corrects the
#   thin-trading bias of small firms' daily betas and captures the hump-shaped size–beta relation (Step 0.8). This is also where the MLP beats a linear model (Step 5).
#   Recent returns (`ret_3m`, `mom_12_2`) add a little, in line with a leverage effect: after a price fall a firm's equity beta tends to rise.
#   The market-wide variables (`market_ret`, `market_vol_12m`) contribute almost nothing. They are the same for every firm in a month, so they cannot help rank firms,
#   and the model does not use them to time the market level (Step 5.3).
# - **Industry adds little *once firm-level history is known*.** A firm's industry exposure is already reflected in its own trailing daily beta, and "Other" (the least
#   informative category) makes up 45% of test rows. This does **not** mean industry is irrelevant to beta (Step 0.7 shows large industry differences). It means the
#   information is redundant given the other inputs.
#
# **Caveats: permutation importance is not causal.**
# - It measures how much *this fitted model* relies on a group, not the effect of changing a firm's characteristics.
# - Correlated predictors share credit. The single-variable importances within historical risk add up to about the group total (0.148 vs 0.153), but a variable can look
#   unimportant simply because a correlated variable carries the same information. The monthly betas are an example.
# - Shuffling creates combinations that don't occur in reality (e.g. a micro-cap with a mega-cap's volatility), so the network is evaluated partly outside its training distribution.
# - The results describe the 2018–22 test period, and they were not used to tune or refit anything.
