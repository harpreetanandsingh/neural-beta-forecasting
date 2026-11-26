# %% [markdown]
# ---
# ## Appendix A: Seed Robustness of the Multi-Output Comparison
# The differences between separate and multi-output models in Step 8 are small. The Step 4 seed check showed that run-to-run noise from weight initialisation
# and minibatch order is about 1% of MSE. To see whether the Step 8 conclusion is robust, the complete Step 8 procedure (10 separate models + 1 multi-output model,
# identical settings) is repeated with seeds 1 and 2 and combined with the main run (seed 42). This is a robustness check only. The main results remain those of Step 8,
# and nothing is tuned on these test results.

# %%
seed_res = {SEED: m8[["test MSE sep", "test MSE sim"]]}
for sd in (1, 2):
    sp, _, smp, _, _, _ = fit_sep_and_sim(sd)
    seed_res[sd] = multi_scores(Ym_te, sp, smp)[["test MSE sep", "test MSE sim"]]
    print(f"seed {sd} done")
sr = pd.concat(seed_res, names=["seed", "industry"])
sr["change %"] = 100 * (sr["test MSE sim"] / sr["test MSE sep"] - 1)

pooled = sr.groupby("seed")[["test MSE sep", "test MSE sim"]].mean()
pooled["pooled RMSE sep"] = np.sqrt(pooled["test MSE sep"]); pooled["pooled RMSE sim"] = np.sqrt(pooled["test MSE sim"])
pooled["change in mean MSE %"] = 100 * (pooled["test MSE sim"] / pooled["test MSE sep"] - 1)
pooled["industries sim better"] = sr.groupby("seed")["change %"].apply(lambda s: int((s < 0).sum()))
pooled.to_csv("results/appendixA_seed_pooled.csv")
print("Pooled results by seed")
display(pooled.round(5))

per_ind = sr["change %"].unstack("seed")
per_ind["mean change %"] = per_ind.mean(1); per_ind["sd"] = per_ind[[SEED, 1, 2]].std(1)
per_ind["seeds with sim better"] = (per_ind[[SEED, 1, 2]] < 0).sum(1)
# Seed-averaged forecasts' MSE (averaging sep and sim over their 3 seeds reduces run-to-run noise)
per_ind = per_ind.reindex(IND_NAMES)
per_ind.to_csv("results/appendixA_seed_per_industry.csv")
print("Per-industry MSE change (multi vs separate, %) by seed")
per_ind.round(2)

# %%
fig, ax = plt.subplots(figsize=(11, 4))
for j, sd in enumerate([SEED, 1, 2]):
    ax.bar(np.arange(10) + (j - 1) * 0.27, per_ind[sd], 0.27, label=f"seed {sd}")
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(10)); ax.set_xticklabels(IND_NAMES)
ax.set_ylabel("test MSE change, multi vs separate (%)"); ax.legend(fontsize=8)
ax.set_title("Appendix A: multi-output vs separate models across three seeds (negative = multi-output better)")
plt.tight_layout(); savefig("appendixA_seeds"); plt.show()

# %% [markdown]
# **Appendix A conclusion.**
# - The multi-output model has a lower pooled test MSE than the separate models for **all three seeds** (−3.5%, −2.9%, −0.3%).
#   It wins in 9, 9 and 6 of 10 industries, so the direction of the Step 8 result is robust.
# - The size varies a lot, and seed 42 happens to show the largest gain. A fair summary is "about 2% lower pooled MSE on average, and never worse".
# - Per-industry differences swing by up to about ±6 percentage points across seeds. Only NoDur, Durbl, Telcm and Hlth improve under every seed.
# - So the industry-level ranking in Step 8 should be read as indicative. The pooled comparison is the reliable one.
# - Averaging several seeds (an ensemble) would reduce this noise for both approaches. The paper also averages over many estimations.
