# %% [markdown]
# ## Step 4: Neural Market-Beta Model (PyTorch MLP)
#
# **Model.** A feed-forward multilayer perceptron maps the $p=22$ preprocessed inputs to one forecast of future beta:
# $p \rightarrow 32 \rightarrow 16 \rightarrow 1$, with ReLU hidden activations, a linear output, and dropout of 0.05 after each hidden layer.
#
# **Training.**
# - Loss and optimiser: MSE loss with Adam.
# - Minibatches: 512 rows, reshuffled every epoch. Shuffling happens only *within* the training period.
# - Epochs and early stopping: at most 100 epochs. Validation MSE is recorded after every epoch. Training stops after 10 epochs without improvement,
#   and the weights from the **best validation epoch** are restored.
# - Seeds are fixed for reproducibility.
#
# **Model selection (validation only).** A deliberately small grid of four configurations:
# {one hidden layer (32), two hidden layers (32→16)} × learning rate {1e-3, 3e-4}. The configuration with the lowest
# validation MSE is selected. The test set is not touched in this step.

# %%
import copy, time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, RandomSampler

torch.manual_seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() // 2))
DEVICE = torch.device("cpu")    # a small MLP trains fastest (and deterministically) on the CPU
BATCH, MAX_EPOCHS, PATIENCE, DROPOUT = 512, 100, 10, 0.05
print("torch", torch.__version__, "| device", DEVICE, "| threads", torch.get_num_threads())

def make_mlp(p, hidden, out_dim=1, dropout=DROPOUT):
    layers, d = [], p
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))          # linear output layer
    return nn.Sequential(*layers)

@torch.no_grad()
def predict(model, X, batch=65536):
    model.eval()
    X = torch.as_tensor(X)
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)]).numpy()

def train_mlp(Xtr, Ytr, Xva, Yva, hidden, lr, seed=SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE,
              batch=BATCH, verbose=False):
    """Adam + MSE; early stopping on validation MSE; returns the model restored to its best validation epoch.
    Y can be (n,) or (n, k). For k > 1 the loss is the mean MSE across the k outputs."""
    torch.manual_seed(seed); np.random.seed(seed)
    Ytr, Yva = np.asarray(Ytr, np.float32).reshape(len(Ytr), -1), np.asarray(Yva, np.float32).reshape(len(Yva), -1)
    model = make_mlp(Xtr.shape[1], hidden, Ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(Ytr))
    sampler = BatchSampler(RandomSampler(ds, generator=torch.Generator().manual_seed(seed)), batch, drop_last=False)
    loader = DataLoader(ds, sampler=sampler, batch_size=None)   # fetches whole minibatches at once (fast)
    best, best_state, best_epoch, wait, hist = np.inf, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train(); run, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward(); opt.step()
            run += loss.item() * len(xb); n += len(xb)
        tr_mse = float(((predict(model, Xtr) - Ytr) ** 2).mean())    # full training MSE, dropout off
        va_mse = float(((predict(model, Xva) - Yva) ** 2).mean())
        hist.append({"epoch": epoch, "train_loss_batches": run / n, "train_mse": tr_mse, "val_mse": va_mse})
        if va_mse < best - 1e-6:
            best, best_epoch, wait, best_state = va_mse, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train {tr_mse:.5f}  val {va_mse:.5f}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)                 # restore best validation checkpoint
    return model, pd.DataFrame(hist), {"best_epoch": best_epoch, "epochs_run": epoch, "best_val_mse": best,
                                       "train_mse_at_best": hist[best_epoch - 1]["train_mse"],
                                       "seconds": time.time() - t0}

# %%
GRID = [{"name": "1 layer (32), lr 1e-3", "hidden": [32], "lr": 1e-3},
        {"name": "1 layer (32), lr 3e-4", "hidden": [32], "lr": 3e-4},
        {"name": "2 layers (32-16), lr 1e-3", "hidden": [32, 16], "lr": 1e-3},
        {"name": "2 layers (32-16), lr 3e-4", "hidden": [32, 16], "lr": 3e-4}]
runs = {}
for cfg in GRID:
    model, hist, info = train_mlp(X_train, y_train, X_val, y_val, cfg["hidden"], cfg["lr"])
    runs[cfg["name"]] = {"cfg": cfg, "model": model, "hist": hist, **info}
    print(f"{cfg['name']:<28} best epoch {info['best_epoch']:3d} / run {info['epochs_run']:3d} | "
          f"val MSE {info['best_val_mse']:.5f} (RMSE {np.sqrt(info['best_val_mse']):.4f}) | "
          f"train MSE {info['train_mse_at_best']:.5f} | {info['seconds']:.0f}s")

# %%
grid_tab = pd.DataFrame([{"config": k, "hidden layers": str(v["cfg"]["hidden"]), "learning rate": v["cfg"]["lr"],
                          "parameters": sum(p.numel() for p in v["model"].parameters()),
                          "best epoch": v["best_epoch"], "epochs run": v["epochs_run"],
                          "train MSE @best": v["train_mse_at_best"], "val MSE @best": v["best_val_mse"],
                          "val RMSE @best": np.sqrt(v["best_val_mse"])} for k, v in runs.items()]).set_index("config")
BEST_CFG = grid_tab["val MSE @best"].idxmin()
nn_model = runs[BEST_CFG]["model"]
SEL = runs[BEST_CFG]["cfg"]
grid_tab.to_csv("results/step4_grid.csv")
print(f"==> Selected on validation MSE: {BEST_CFG}")
print(f"    For reference, validation RMSE of beta_daily_12m = {bench.loc[('validation', 'beta_daily_12m'), 'RMSE']:.4f}, "
      f"{BEST_BENCH} = {bench.loc[('validation', BEST_BENCH), 'RMSE']:.4f}")
grid_tab

# %%
# Robustness of the selection to the random seed (validation data only; test data is not used).
# Each configuration is retrained with two further seeds. This checks whether the differences between configurations exceed
# the run-to-run noise from weight initialisation and minibatch order. The selection rule itself (seed 42) is unchanged.
seed_rows = []
for cfg in GRID:
    for sd in (1, 2):
        _, _, inf_ = train_mlp(X_train, y_train, X_val, y_val, cfg["hidden"], cfg["lr"], seed=sd)
        seed_rows.append({"config": cfg["name"], "seed": sd, "val MSE": inf_["best_val_mse"]})
    seed_rows.append({"config": cfg["name"], "seed": SEED, "val MSE": runs[cfg["name"]]["best_val_mse"]})
seed_tab = pd.DataFrame(seed_rows).groupby("config")["val MSE"].agg(["mean", "std", "min", "max"]).reindex(grid_tab.index)
seed_tab.to_csv("results/step4_seed_robustness.csv")
print("Validation MSE across 3 seeds (42, 1, 2)")
seed_tab

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, (k, v) in zip(axes, runs.items()):
    h = v["hist"]
    ax.plot(h.epoch, h.train_mse, label="training MSE (eval mode)", color="#1f77b4")
    ax.plot(h.epoch, h.train_loss_batches, label="training loss (minibatches, dropout on)", color="#1f77b4", ls=":", alpha=0.7)
    ax.plot(h.epoch, h.val_mse, label="validation MSE", color="#ff7f0e")
    ax.axvline(v["best_epoch"], color="k", ls="--", lw=1)
    ax.axhline(bench.loc[("validation", "beta_daily_12m"), "RMSE"] ** 2, color="#2ca02c", lw=1, ls="-.",
               label="validation MSE of beta_daily_12m")
    ax.set_title(("[SELECTED] " if k == BEST_CFG else "") + k, fontsize=10, fontweight="bold" if k == BEST_CFG else None)
    ax.set_xlabel("epoch")
axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=7.5, loc="upper right")
axes[0].set_ylim(0.13, 0.26)
plt.suptitle("Training and validation loss curves (dashed line = best validation epoch, restored)", y=1.02)
plt.tight_layout(); savefig("step4_loss_curves"); plt.show()
pd.concat({k: v["hist"] for k, v in runs.items()}).to_csv("results/step4_loss_history.csv")

# %%
# Overfitting diagnostics for the selected model
h = runs[BEST_CFG]["hist"]; be = runs[BEST_CFG]["best_epoch"]
print(f"Selected: {BEST_CFG}; architecture: p={P} -> {' -> '.join(map(str, SEL['hidden']))} -> 1, ReLU, dropout {DROPOUT}, "
      f"Adam lr {SEL['lr']}, batch {BATCH}, early-stopping patience {PATIENCE}")
print(nn_model)
print(f"Best epoch {be}: train MSE {h.train_mse[be-1]:.5f}, validation MSE {h.val_mse[be-1]:.5f} "
      f"(gap {h.val_mse[be-1] - h.train_mse[be-1]:+.5f})")
print(f"Last epoch {len(h)}: train MSE {h.train_mse.iloc[-1]:.5f}, validation MSE {h.val_mse.iloc[-1]:.5f}")
print(f"Train MSE improvement from best epoch to last: {h.train_mse[be-1] - h.train_mse.iloc[-1]:+.5f}; "
      f"validation change: {h.val_mse.iloc[-1] - h.val_mse[be-1]:+.5f}")

# %%
# Is the train/validation gap overfitting, or a difference between the periods? Compare with benchmarks that have NO fitted parameters.
gap = pd.DataFrame({
    "train MSE": [((predict(nn_model, X_train).ravel() - y_train) ** 2).mean(),
                  ((train_df.beta_daily_12m - train_df[TARGET]) ** 2).mean(), ((1 - train_df[TARGET]) ** 2).mean()],
    "validation MSE": [runs[BEST_CFG]["best_val_mse"],
                       ((val_df.beta_daily_12m - val_df[TARGET]) ** 2).mean(), ((1 - val_df[TARGET]) ** 2).mean()]},
    index=["selected MLP", "beta_daily_12m (no fitting)", "constant 1.0 (no fitting)"])
gap["val / train"] = gap["validation MSE"] / gap["train MSE"]
gap["R2 train (1 - MSE/var)"] = 1 - gap["train MSE"] / train_df[TARGET].var(ddof=0)
gap["R2 validation (1 - MSE/var)"] = 1 - gap["validation MSE"] / val_df[TARGET].var(ddof=0)
gap

# %% [markdown]
# **Selected model and overfitting assessment.**
# - **Selected:** two hidden layers, 22 → 32 → 16 → 1 (1,281 parameters). ReLU, dropout 0.05, Adam with learning rate 1e-3, batch size 512.
#   Best validation epoch 8 (restored); training stopped at epoch 18. Validation RMSE is 0.424, against 0.487 for `beta_daily_12m` and 0.787 for `beta_36m`.
# - **Robustness:** the four configurations differ by at most 0.003 in validation MSE, which is the same size as the run-to-run noise across
#   seeds (sd 0.001–0.002). The choice matters little. The two-layer model with learning rate 1e-3 also has the best *average* across three seeds, so the selection is not a lucky draw.
# - **Overfitting: mild within training, plus some loss of generalisation to a new period.**
#   - *Within-sample overfitting is small and handled by early stopping.* After epoch 8 training MSE falls only another 0.002, while validation
#     MSE drifts up by 0.003. This is the classic onset of overfitting, and restoring the best checkpoint removes it. The model is tiny
#     (1,281 parameters for 589k rows) and learns most of what it can in 1–2 epochs, so it cannot memorise individual observations.
#   - *The gap between periods is real.* Validation MSE is 24% above training MSE (0.180 vs 0.145). The parameter-free daily-beta forecast does *not* show
#     this: its validation MSE is 3% *below* its training MSE. So part of the MLP's in-sample fit comes from relationships specific to 1998–2011
#     (e.g. the thin-trading pattern among the many small firms of that era) that carry over only partly to 2013–16.
#     The MLP's advantage over the daily beta shrinks from 41% lower MSE in training to 24% lower in validation.
#   - Conclusion: the out-of-sample gain is substantial but smaller than the in-sample fit suggests. This is why selection and evaluation must use later,
#     unseen periods. Stronger regularisation might narrow the gap, but the grid above shows it is not driven by model capacity:
#     the one-layer models have the same validation error.
