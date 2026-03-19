# Mathematics Reference

Every mathematical operation in the system, explained from first principles. Version 2.5.1.

---

## Table of Contents

1. [Synthetic Data Generation](#1-synthetic-data-generation)
2. [Pre-Game Rolling Average Enrichment](#2-pre-game-rolling-average-enrichment)
3. [Feature Scaling — StandardScaler](#3-feature-scaling--standardscaler)
4. [Train-Test Split](#4-train-test-split)
5. [Model Mathematics](#5-model-mathematics)
   - [Gradient Boosting](#51-gradient-boosting)
   - [Random Forest](#52-random-forest)
   - [Extra Trees](#53-extra-trees)
   - [SVM with RBF Kernel](#54-svm-with-rbf-kernel)
   - [Neural Network (MLP)](#55-neural-network-mlp)
   - [XGBoost](#56-xgboost)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Cross-Validation](#7-cross-validation)
8. [Feature Importances](#8-feature-importances)
9. [Team Season Averages](#9-team-season-averages)
10. [Prediction Confidence](#10-prediction-confidence)
11. [Promote Threshold](#11-promote-threshold)
12. [Adaptive Tree Depth](#12-adaptive-tree-depth)

---

## 1. Synthetic Data Generation

Used only by `_generate_synthetic()` as a fallback when ESPN is unavailable.

### Strength Score

Each team is assigned a scalar strength score from a weighted sum of their features:

$$S_{home} = hfg \cdot 100 + hrb \cdot 0.5 + ha \cdot 0.8 - ht \cdot 0.6 + hst \cdot 0.4 + hbl \cdot 0.3 + 3$$

$$S_{away} = afg \cdot 100 + arb \cdot 0.5 + aa \cdot 0.8 - at \cdot 0.6 + ast \cdot 0.4 + abl \cdot 0.3$$

The **+3 home court advantage** is added to $S_{home}$ before comparison. This reflects the well-documented empirical advantage of playing at home in college basketball.

Note that this formula uses efficiency metrics only — it does not include raw points scored. This mirrors the real enrichment pipeline, where score-derived features are excluded from the feature vector.

### Probabilistic Outcome

The outcome is not deterministic — 15% noise is added to prevent perfect separability:

$$\text{outcome} = \begin{cases} 1 & \text{if } S_{home} > S_{away} \text{ and } U(0,1) > 0.15 \\ 0 & \text{if } S_{home} > S_{away} \text{ and } U(0,1) \leq 0.15 \\ 0 & \text{if } S_{home} \leq S_{away} \text{ and } U(0,1) > 0.15 \\ 1 & \text{if } S_{home} \leq S_{away} \text{ and } U(0,1) \leq 0.15 \end{cases}$$

Where $U(0,1)$ is a uniform random draw. This gives an 85% chance the stronger team wins, mirroring realistic sports outcomes.

Synthetic records are passed through `enrich_with_pregame_averages()` before being stored, so training on synthetic data uses the same rolling-average features as training on real data.

---

## 2. Pre-Game Rolling Average Enrichment

This is the core mathematical operation added in v2.5. It converts raw in-game statistics into genuine pre-game predictors.

### The Problem with In-Game Statistics

Let $x_{i,j}$ be the value of feature $j$ for game $i$. When this represents an in-game statistic (e.g. actual field goal percentage during game $i$), the correlation with outcome $y_i$ is:

$$\rho(x_j, y) = \frac{\text{Cov}(x_j, y)}{\sigma_{x_j} \sigma_y}$$

For in-game FG%, this correlation is approximately +0.81. This is nearly as high as using the score itself, because shooting well causes winning. The feature is circular, not predictive.

For pre-game rolling averages of the same statistic, the same correlation is approximately +0.15. This is genuine predictive signal.

### Rolling Average Computation

For a given team $t$ and statistic $s$, let $H_t = [h_1, h_2, \ldots, h_k]$ be the ordered sequence of the team's per-game values for statistic $s$ in their prior $k$ games (in chronological order). The pre-game rolling average used as the feature for their next game is:

$$\bar{x}_{t,s} = \frac{1}{\min(k, W)} \sum_{i = \max(1, k-W+1)}^{k} h_i$$

Where $W$ is the configured window size (default 10). Only the most recent $W$ games contribute.

### Assist-to-Turnover Ratio

For `ast_to_tov`, the rolling average is not computed by averaging the per-game ratio. Instead:

$$\text{ast\_to\_tov} = \frac{\bar{x}_{t,\text{assists}}}{\bar{x}_{t,\text{turnovers}}}$$

This is the ratio of the rolling averages, not the rolling average of the ratios. The former is more numerically stable — a single game with very few turnovers inflates the per-game ratio, but has limited effect on the average count.

### Cold-Start Exclusion

A game is excluded from the training set (not from the history) if either team has fewer than $m$ prior games, where $m$ is `data.pregame_min_games` (default 1). With $m = 1$, only the very first game of each team's season is excluded:

$$\text{include game } i \iff |H_{home}| \geq m \text{ and } |H_{away}| \geq m$$

At the time of processing game $i$, the history $H_t$ contains only games prior to game $i$. A game is never its own pre-game feature.

---

## 3. Feature Scaling — StandardScaler

Applied as the first step of every model Pipeline. The scaler is fit only on the training set and applied to both train and test sets — this prevents data leakage from test statistics affecting the scale.

### Z-Score Normalization

For each feature $j$ across all training samples:

$$\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$$

$$\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}$$

Each value is then transformed:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

After scaling, every feature has mean 0 and standard deviation 1. This is critical for SVM and MLP, which are sensitive to feature magnitude. For tree-based models (GBM, RF, Extra Trees, XGBoost), scaling has no effect on results since trees split on rank order. It is applied uniformly across all models for consistency.

---

## 4. Train-Test Split

```
n_train = floor(0.8 x n)
n_test  = n - n_train
```

With 2900 enriched games, approximately: `n_train = 2320`, `n_test = 580`.

### Stratification

The split preserves the class ratio in both sets. With real ESPN multi-season data, the home win rate is approximately 69%. Without stratification, a random split might produce a test set with 62% or 76% home wins by chance, making evaluation unstable. Stratification guarantees both sets mirror the full dataset's class distribution.

---

## 5. Model Mathematics

All models receive the scaled feature matrix $Z \in \mathbb{R}^{n \times 14}$ and binary labels $y \in \{0, 1\}^n$.

---

### 5.1 Gradient Boosting

Builds an additive ensemble of $M$ shallow decision trees sequentially. Each tree corrects the residual errors of the previous ensemble.

**Initialization:**

$$F_0(x) = \log\frac{p}{1-p}, \quad p = \frac{1}{n}\sum y_i$$

**Boosting iteration** for $m = 1, 2, \ldots, M$:

1. Compute pseudo-residuals (negative gradient of log-loss):

$$r_{im} = y_i - \sigma(F_{m-1}(x_i))$$

Where $\sigma$ is the sigmoid function.

2. Fit a regression tree $h_m(x)$ to the pseudo-residuals.

3. Update the ensemble:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $\eta = 0.05$ is the learning rate.

**Final prediction:**

$$\hat{p}(x) = \sigma(F_M(x)) = \frac{1}{1 + e^{-F_M(x)}}$$

Config: `n_estimators=300`, `learning_rate=0.05`, `max_depth=4` (further capped by adaptive depth), `subsample=0.8`, `min_samples_split=5`, `min_samples_leaf=2`.

`subsample=0.8` — stochastic gradient boosting. Each tree is fit on a random 80% of samples, reducing variance.

---

### 5.2 Random Forest

Builds $T$ decision trees independently, each on a bootstrap sample with random feature subsets.

**Bootstrap:** Each tree uses $n$ samples drawn with replacement. Approximately 63.2% of unique training examples per tree.

**Node splitting:** At each node, a random subset of $\sqrt{14} \approx 4$ features is considered. The best split maximises the reduction in Gini impurity:

$$\text{Gini}(S) = 1 - \sum_{k=0}^{1} p_k^2$$

$$\Delta\text{Gini} = \text{Gini}(S) - \frac{|S_L|}{|S|}\text{Gini}(S_L) - \frac{|S_R|}{|S|}\text{Gini}(S_R)$$

**Probability estimate:**

$$\hat{p}(x) = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[h_t(x) = 1]$$

Config: `n_estimators=300`, `max_depth=10` (adaptive ceiling applies), `min_samples_split=5`, `min_samples_leaf=2`.

---

### 5.3 Extra Trees (Extremely Randomized Trees)

Identical to Random Forest with one key difference: split thresholds are randomly selected rather than optimized.

For each candidate feature $j$ at a node, a random threshold $\theta$ is drawn uniformly:

$$\theta_j \sim U(\min_j, \max_j)$$

The best (feature, threshold) pair among the random candidates is chosen by Gini reduction. This extra randomisation reduces variance further at the cost of slightly higher bias. On smaller datasets this often outperforms Random Forest.

Config: `n_estimators=300`, `max_depth=10` (adaptive ceiling applies), `min_samples_split=5`, `min_samples_leaf=2`.

---

### 5.4 SVM with RBF Kernel

Finds the maximum-margin hyperplane in a high-dimensional space induced by the RBF kernel.

**RBF Kernel:**

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

With `gamma="scale"`: $\gamma = \frac{1}{n\_features \cdot \text{Var}(X)}$

**Optimization (soft-margin SVM):**

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

Subject to: $y_i(w \cdot \phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$

$C = 2.0$ (raised from 1.0 in v2.5.1). Higher $C$ fits training data more tightly. With clean pre-game features (no leakage), the model can afford less slack.

**Probability calibration (Platt scaling):**

$$P(y=1 | f(x)) = \frac{1}{1 + e^{Af(x) + B}}$$

Where $A$ and $B$ are fit by an additional cross-validation pass.

Config: `kernel="rbf"`, `C=2.0`, `gamma="scale"`, `probability=True`.

---

### 5.5 Neural Network (MLP)

Feedforward network with three hidden layers.

**Architecture:** $14 \xrightarrow{} 128 \xrightarrow{} 64 \xrightarrow{} 32 \xrightarrow{} 1$

Architecture restored to [128, 64, 32] in v2.5.1 after being incorrectly shrunk to [64, 32] in v2.4.

**Forward pass:**

$$a^{(l)} = \text{ReLU}(W^{(l)} a^{(l-1)} + b^{(l)})$$

$$\text{ReLU}(z) = \max(0, z)$$

**Loss function (cross-entropy):**

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Backpropagation:** Gradients flow backward via chain rule. Weights updated by Adam optimizer.

**Early stopping:** Training halts when validation loss on `validation_fraction` (15%) of training data does not improve for 10 consecutive epochs.

Config: `hidden_layer_sizes=[128, 64, 32]`, `activation="relu"`, `max_iter=500`, `early_stopping=True`, `validation_fraction=0.15`.

---

### 5.6 XGBoost

Optimised gradient boosting with a regularised objective.

**Objective:**

$$\mathcal{L}^{(m)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(m-1)} + f_m(x_i)) + \Omega(f_m)$$

**Regularisation:**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Where $T$ is the number of leaves and $w_j$ are leaf weights. This explicit regularisation is the key difference from vanilla gradient boosting.

**Optimal leaf weights** (derived analytically):

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Where $g_i$ and $h_i$ are first and second derivatives of the loss.

Config: `n_estimators=300`, `learning_rate=0.05`, `max_depth=5` (adaptive ceiling applies), `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`.

`min_child_weight=3` is XGBoost's equivalent of `min_samples_leaf` — requires at least 3 samples per leaf node.

---

## 6. Evaluation Metrics

All metrics computed on the held-out test set.

Let TP, TN, FP, FN be counts relative to the positive class (Home Win = 1).

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Misleading with ~69% home wins. A model predicting "Home Win" for every game achieves 69% accuracy with AUC = 0.5.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all games predicted as Home Win, what fraction actually were?

### Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual Home Wins, what fraction did we correctly identify?

### F1-Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

The harmonic mean penalises imbalance between Precision and Recall.

### ROC-AUC (Primary Selection Metric)

The ROC curve plots TPR (Recall) against FPR at every classification threshold $\tau \in [0, 1]$:

$$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}, \quad \text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

The Area Under this Curve measures the probability that the model assigns a higher probability to a randomly chosen Home Win than to a randomly chosen Away Win:

$$\text{AUC} = P(\hat{p}(x^+) > \hat{p}(x^-))$$

AUC = 1.0 is perfect ranking. AUC = 0.5 is no better than random.

**Why AUC is used for model selection:** It is threshold-independent and robust to class imbalance. A model outputting $\hat{p} = 0.69$ for every game (reflecting the base rate) achieves 69% accuracy but AUC = 0.5. It is correctly penalised.

**Expected AUC range after v2.5 leakage fix:** 0.60 – 0.75. In v2.4, AUC was 0.95 – 0.97 because features contained in-game statistics correlated ~0.81 with outcome. After removing that circular information, AUC reflects genuine pre-game predictive power. The reduction is correct and expected.

---

## 7. Cross-Validation

5-fold cross-validation run on the full dataset after the main train-test evaluation.

**Procedure:**

1. Shuffle the full dataset and split into 5 equal folds $F_1, F_2, F_3, F_4, F_5$
2. For each fold $k \in \{1, \ldots, 5\}$: train on the other 4 folds, evaluate AUC on $F_k$
3. Report:

$$\overline{\text{AUC}} = \frac{1}{5}\sum_{k=1}^{5} \text{AUC}_k$$

$$\sigma_{\text{AUC}} = \sqrt{\frac{1}{5}\sum_{k=1}^{5}(\text{AUC}_k - \overline{\text{AUC}})^2}$$

A low $\sigma_{\text{AUC}}$ (below 0.02) indicates the model performs consistently across different data partitions. A high $\sigma_{\text{AUC}}$ (above 0.05) signals instability — the model is sensitive to which games land in the test fold.

---

## 8. Feature Importances

### Tree-Based Models (Gradient Boosting, Random Forest, Extra Trees, XGBoost)

Mean Decrease in Impurity (MDI): for each feature $j$, the total weighted Gini reduction across all splits on that feature, averaged over all trees:

$$I_j = \frac{1}{T}\sum_{t=1}^{T} \sum_{\text{nodes } v \text{ splitting on } j} \frac{n_v}{n} \cdot \Delta\text{Gini}(v)$$

Normalised so importances sum to 1:

$$\tilde{I}_j = \frac{I_j}{\sum_{k=1}^{14} I_k}$$

### SVM (RBF kernel)

No `feature_importances_` is available. `get_feature_importances()` returns None. The Features tab does not show an importance chart for SVM. This is expected — the kernel trick maps to a high-dimensional space where individual feature contributions are not directly interpretable.

### MLP

Similarly returns None. Per-neuron weights do not translate cleanly to per-feature importances without a separate analysis (such as SHAP). This is a known limitation listed in the remaining work section.

---

## 9. Team Season Averages

Used to auto-fill the prediction form. For each team $t$ and feature $f$, the season average is computed over all games in which team $t$ appears:

$$\bar{x}_{t,f} = \frac{1}{|G_t|} \sum_{g \in G_t} x_{g,f,t}$$

Since game records now store pre-game rolling averages as feature values, this is an average of averages — a stable estimate of the team's recent form over the entire dataset.

When a rolling window $W$ is specified, only the most recent $W$ games per team are included (sorted by `game_date` descending).

The mirroring logic ensures consistent averaging regardless of home/away assignment. When team $t$ appeared as the away team in game $g$, their `away_fg_pct` value is stored under the `home_fg_pct` key in the accumulator, so every team ends up with a single set of `home_*` stats representing their own performance.

---

## 10. Prediction Confidence

For models with `predict_proba`:

$$\text{confidence} = \max(\hat{p}(y=0 | x),\ \hat{p}(y=1 | x)) = \max(\hat{p},\ 1 - \hat{p})$$

Always in $[0.5, 1.0]$. A confidence of 0.5 is maximum uncertainty (50/50). A confidence of 0.75 means the model assigns 75% probability to its predicted class. Displayed as a percentage in the dashboard.

---

## 11. Promote Threshold

When the auto-learn scheduler retrains, the new best model only replaces the current active model if:

$$\text{AUC}_{new} \geq \text{AUC}_{current} + \delta$$

Where $\delta = 0.002$ (`promote_threshold` in config).

**Why a threshold rather than strict improvement:** AUC estimates have variance. Two training runs on slightly different data samples (6 hours apart) can produce AUC differences of ±0.003 to ±0.005 purely from sampling noise. The threshold filters these out and only promotes when there is a non-trivial, meaningful gain.

**Effect:** The active model's AUC is monotonically non-decreasing over time. The model can only improve or stay the same, never regress.

---

## 12. Adaptive Tree Depth

Tree model `max_depth` is capped at runtime to prevent overfitting on smaller datasets.

**Rule of thumb:** each leaf needs approximately $10 \times p$ samples to split reliably, where $p$ is the number of features. At depth $D$ there are up to $2^D$ leaves. Solving for the maximum safe depth:

$$\text{max\_safe} = \left\lfloor \log_2\left(\frac{n}{10 \cdot p}\right) \right\rfloor$$

Clamped to a minimum of 3:

$$\text{depth} = \max\left(3,\ \min\left(\text{base\_depth},\ \text{max\_safe}\right)\right)$$

At the current dataset size:
- $n = 2300$ (training set, 80% of 2900 enriched games)
- $p = 14$ features
- $\text{max\_safe} = \lfloor \log_2(2300 / 140) \rfloor = \lfloor \log_2(16.4) \rfloor = 4$

So regardless of whether the config specifies `max_depth: 4` (Gradient Boosting, XGBoost) or `max_depth: 10` (Random Forest, Extra Trees), all tree models run at depth 4 at the current dataset size. As more data is collected, the ceiling rises automatically:

| n_samples | max_safe |
|-----------|----------|
| 2300 | 4 |
| 5000 | 5 |
| 10000 | 6 |
| 20000 | 7 |

---

*All implementations use scikit-learn 1.8.0 and XGBoost 2.x conventions.*
