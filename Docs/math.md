# 📐 Mathematics Reference

> *Every mathematical operation in main.py, explained from first principles*

---

## Table of Contents

1. [Synthetic Data Generation](#1-synthetic-data-generation)
2. [Feature Scaling — StandardScaler](#2-feature-scaling--standardscaler)
3. [Train-Test Split](#3-train-test-split)
4. [Model Mathematics](#4-model-mathematics)
   - [Gradient Boosting](#41-gradient-boosting)
   - [Random Forest](#42-random-forest)
   - [Extra Trees](#43-extra-trees)
   - [SVM with RBF Kernel](#44-svm-with-rbf-kernel)
   - [Neural Network (MLP)](#45-neural-network-mlp)
   - [XGBoost](#46-xgboost)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Cross-Validation](#6-cross-validation)
7. [Feature Importances](#7-feature-importances)
8. [Team Season Averages](#8-team-season-averages)
9. [Prediction Confidence](#9-prediction-confidence)
10. [Promote Threshold](#10-promote-threshold)

---

## 1. Synthetic Data Generation

Used only by `_generate_synthetic()` as a fallback when ESPN is unavailable.

### Strength Score

Each team is assigned a scalar strength score from a weighted sum of their features:

$$S_{home} = 0.3 \cdot ppg + 80 \cdot fg_{pct} + 40 \cdot p3_{pct} + 0.5 \cdot reb + 0.8 \cdot ast - 0.6 \cdot tov + 0.4 \cdot stl + 3$$

$$S_{away} = 0.3 \cdot ppg + 80 \cdot fg_{pct} + 40 \cdot p3_{pct} + 0.5 \cdot reb + 0.8 \cdot ast - 0.6 \cdot tov + 0.4 \cdot stl$$

The **+3 home court advantage** is added to $S_{home}$ before comparison. This reflects the well-documented empirical advantage of playing at home in college basketball.

### Probabilistic Outcome

The outcome is not deterministic — 15% noise is added to prevent perfect separability:

$$\text{outcome} = \begin{cases} 1 & \text{if } S_{home} > S_{away} \text{ and } U(0,1) > 0.15 \\ 0 & \text{if } S_{home} > S_{away} \text{ and } U(0,1) \leq 0.15 \\ 0 & \text{if } S_{home} \leq S_{away} \text{ and } U(0,1) > 0.15 \\ 1 & \text{if } S_{home} \leq S_{away} \text{ and } U(0,1) \leq 0.15 \end{cases}$$

Where $U(0,1)$ is a uniform random draw. This gives an 85% chance the stronger team wins, mirroring realistic sports outcomes.

---

## 2. Feature Scaling — StandardScaler

Applied as the first step of every model Pipeline. The scaler is **fit only on the training set** and then applied to both train and test sets — this prevents data leakage.

### Z-Score Normalization

For each feature $j$ across all training samples:

$$\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$$

$$\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}$$

Each value is then transformed:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

After scaling, every feature has mean 0 and standard deviation 1. This is critical for SVM and MLP, which are sensitive to feature magnitude. For tree-based models (GBM, RF, Extra Trees, XGBoost) it has no effect on the result — trees split on rank order, not absolute value — but it doesn't hurt either.

---

## 3. Train-Test Split

```
n_train = floor(0.8 × n)
n_test  = n - n_train
```

With 500 games: `n_train = 400`, `n_test = 100`.

### Stratification

The split preserves the class ratio in both sets. If the full dataset has home win rate $p$:

$$p = \frac{\sum_{i=1}^{n} y_i}{n}$$

Then both the training and test sets are guaranteed to have approximately $p$ proportion of home wins. With real ESPN data, $p \approx 0.738$ — without stratification, a random split could produce a test set with 65% or 85% home wins by chance, making evaluation unreliable.

---

## 4. Model Mathematics

All models receive the scaled feature matrix $Z \in \mathbb{R}^{n \times 14}$ and binary labels $y \in \{0, 1\}^n$.

---

### 4.1 Gradient Boosting

Builds an additive ensemble of $M$ shallow decision trees sequentially. Each tree corrects the residual errors of the previous ensemble.

**Initialization:**

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

For log-loss (binary classification), this is the log-odds of the training mean:

$$F_0 = \log\frac{p}{1-p}, \quad p = \frac{1}{n}\sum y_i$$

**Boosting iteration** — for $m = 1, 2, \ldots, M$:

1. Compute pseudo-residuals (negative gradient of loss):

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

For log-loss: $r_{im} = y_i - \sigma(F_{m-1}(x_i))$, where $\sigma$ is the sigmoid function.

2. Fit a regression tree $h_m(x)$ to the pseudo-residuals.

3. Update the ensemble:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $\eta = 0.05$ is the learning rate (set in config). Smaller $\eta$ requires more trees but generalizes better.

**Final prediction** — probability of home win:

$$\hat{p}(x) = \sigma(F_M(x)) = \frac{1}{1 + e^{-F_M(x)}}$$

**Config parameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`

`subsample=0.8` means each tree is fit on a random 80% of training samples — this is *stochastic* gradient boosting, which reduces variance.

---

### 4.2 Random Forest

Builds $T$ decision trees independently, each on a bootstrap sample with random feature subsets.

**Bootstrap sampling:** For each tree $t$, draw $n$ samples with replacement from the training set. On average, each bootstrap sample contains ~63.2% of unique training examples (the rest are out-of-bag).

**Node splitting:** At each node, a random subset of $\sqrt{14} \approx 4$ features is considered. The best split maximizes the reduction in **Gini impurity**:

$$\text{Gini}(S) = 1 - \sum_{k=0}^{1} p_k^2$$

Where $p_k$ is the fraction of class $k$ samples in set $S$.

**Split criterion:**

$$\Delta\text{Gini} = \text{Gini}(S) - \frac{|S_L|}{|S|}\text{Gini}(S_L) - \frac{|S_R|}{|S|}\text{Gini}(S_R)$$

The split that maximizes $\Delta\text{Gini}$ is chosen.

**Final prediction** — majority vote across all $T$ trees:

$$\hat{y} = \text{mode}\{h_t(x)\}_{t=1}^{T}$$

**Probability estimate** — fraction of trees voting for class 1:

$$\hat{p}(x) = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[h_t(x) = 1]$$

**Config parameters:** `n_estimators=200`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`

---

### 4.3 Extra Trees (Extremely Randomized Trees)

Identical to Random Forest with one key difference: split thresholds are **randomly selected** rather than optimized.

For each candidate feature $j$ at a node, a random threshold $\theta$ is drawn uniformly from the feature's range in the current node's samples:

$$\theta_j \sim U(\min_j, \max_j)$$

The best (feature, threshold) pair among the random candidates is chosen by Gini reduction. This extra randomization reduces variance further at the cost of slightly higher bias. On smaller datasets like ours (~500 games), this often outperforms Random Forest.

**Config parameters:** `n_estimators=200`, `max_depth=12`, `min_samples_split=5`

---

### 4.4 SVM with RBF Kernel

Finds the maximum-margin hyperplane separating the two classes in a high-dimensional feature space induced by the RBF kernel.

**RBF Kernel:**

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

With `gamma="scale"`: $\gamma = \frac{1}{n\_features \cdot \text{Var}(X)}$

The kernel measures similarity — two games with similar statistics get a value close to 1; dissimilar games get a value close to 0.

**Optimization problem** (soft-margin SVM):

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

Subject to:

$$y_i(w \cdot \phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

$C = 1.0$ (config) controls the bias-variance tradeoff — higher $C$ fits training data more tightly (lower bias, higher variance).

**Prediction:** The decision function is:

$$f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, x) + b\right)$$

Where the sum runs only over support vectors (the training examples nearest the margin boundary).

**Probability calibration:** With `probability=True`, Platt scaling fits a logistic regression on the SVM decision values to produce probabilities. This is what `predict_proba` returns.

$$P(y=1 | f(x)) = \frac{1}{1 + e^{Af(x) + B}}$$

Where $A$ and $B$ are fit by an additional cross-validation pass.

**Config parameters:** `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `probability=True`

---

### 4.5 Neural Network (MLP)

A feedforward network with three hidden layers.

**Architecture:** $14 \xrightarrow{} 128 \xrightarrow{} 64 \xrightarrow{} 32 \xrightarrow{} 1$

**Forward pass** — for each layer $l$:

$$a^{(l)} = \text{ReLU}(W^{(l)} a^{(l-1)} + b^{(l)})$$

$$\text{ReLU}(z) = \max(0, z)$$

The output layer uses softmax (for two classes) to produce probabilities.

**Loss function** — cross-entropy:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Backpropagation** — gradients flow backward through the network via chain rule:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial W^{(l)}}$$

Weights are updated by Adam optimizer (scikit-learn default):

$$W \leftarrow W - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$$

Where $\hat{m}$ and $\hat{v}$ are bias-corrected first and second moment estimates of the gradient.

**Early stopping:** Training halts when validation loss (on 10% of training data) does not improve for 10 consecutive epochs. This prevents overfitting without needing a fixed epoch count.

**Config parameters:** `hidden_layer_sizes=[128, 64, 32]`, `activation="relu"`, `max_iter=500`, `early_stopping=True`, `validation_fraction=0.1`

---

### 4.6 XGBoost

An optimized implementation of gradient boosting with a regularized objective.

**Objective function:**

$$\mathcal{L}^{(m)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(m-1)} + f_m(x_i)) + \Omega(f_m)$$

**Regularization term:**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Where $T$ is the number of leaves, $w_j$ are leaf weights, $\gamma$ penalizes tree complexity, and $\lambda$ is L2 regularization on leaf weights. This is the key difference from vanilla gradient boosting — XGBoost explicitly regularizes the tree structure.

**Second-order Taylor approximation** of the loss at each boosting step allows XGBoost to find optimal leaf weights analytically rather than fitting to residuals:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Where $g_i = \partial L / \partial \hat{y}_i$ and $h_i = \partial^2 L / \partial \hat{y}_i^2$ are first and second derivatives of the loss.

**Config parameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`

`colsample_bytree=0.8` randomly selects 80% of features for each tree — similar to Random Forest's feature subsampling, reducing correlation between trees.

---

## 5. Evaluation Metrics

All metrics are computed on the held-out test set ($n_{test} \approx 100$ games).

Let:
- TP = True Positives (predicted Home Win, actual Home Win)
- TN = True Negatives (predicted Away Win, actual Away Win)
- FP = False Positives (predicted Home Win, actual Away Win)
- FN = False Negatives (predicted Away Win, actual Home Win)

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Fraction of all predictions that were correct. Misleading when classes are imbalanced — with ~73% home wins, always predicting "Home Win" gives 73% accuracy for free.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all games we predicted as Home Win, what fraction actually were? Measures how trustworthy a positive prediction is.

### Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual Home Wins, what fraction did we correctly identify? Measures how many wins we caught.

### F1-Score

Harmonic mean of Precision and Recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

The harmonic mean penalizes imbalance between Precision and Recall — a model with 100% Precision and 0% Recall gets F1 = 0, not 50%.

### ROC-AUC (Primary Selection Metric)

The Receiver Operating Characteristic curve plots True Positive Rate (Recall) against False Positive Rate at every possible classification threshold $\tau \in [0, 1]$:

$$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}, \quad \text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

The **Area Under this Curve** (AUC) measures the probability that the model ranks a randomly chosen Home Win higher than a randomly chosen Away Win:

$$\text{AUC} = P(\hat{p}(x^+) > \hat{p}(x^-))$$

Where $x^+$ is a random Home Win game and $x^-$ is a random Away Win game.

AUC = 1.0 means perfect ranking. AUC = 0.5 means no better than random.

**Why ROC-AUC is used for model selection:** It is threshold-independent and robust to class imbalance. With ~73% home wins, a model that outputs $\hat{p} = 0.73$ for every game achieves 73% accuracy but AUC = 0.5 — it gets correctly penalized.

---

## 6. Cross-Validation

5-fold cross-validation is run on the full dataset after the main train-test evaluation.

**Procedure:**

1. Shuffle the full dataset and split into 5 equal folds $F_1, F_2, F_3, F_4, F_5$
2. For each fold $k \in \{1, \ldots, 5\}$:
   - Train on $\bigcup_{j \neq k} F_j$ (80% of data)
   - Evaluate ROC-AUC on $F_k$ (20% of data)
3. Report mean and standard deviation:

$$\overline{\text{AUC}} = \frac{1}{5}\sum_{k=1}^{5} \text{AUC}_k$$

$$\sigma_{\text{AUC}} = \sqrt{\frac{1}{5}\sum_{k=1}^{5}(\text{AUC}_k - \overline{\text{AUC}})^2}$$

A low $\sigma_{\text{AUC}}$ indicates the model performs consistently across different data partitions — it hasn't memorized the specific train-test split. A high $\sigma_{\text{AUC}}$ signals instability (the model is sensitive to which games end up in the test set).

---

## 7. Feature Importances

### Tree-Based Models (Gradient Boosting, Random Forest, Extra Trees, XGBoost)

**Mean Decrease in Impurity (MDI):**

For each feature $j$, importance is the total weighted Gini reduction across all splits on that feature, averaged over all trees:

$$I_j = \frac{1}{T}\sum_{t=1}^{T} \sum_{\text{nodes } v \text{ splitting on } j} \frac{n_v}{n} \cdot \Delta\text{Gini}(v)$$

Where $n_v$ is the number of samples reaching node $v$.

Features are then normalized so they sum to 1:

$$\tilde{I}_j = \frac{I_j}{\sum_{k=1}^{14} I_k}$$

### SVM (coef_ — linear kernel only)

SVM with RBF kernel has no `feature_importances_`. The code checks `hasattr(clf, "coef_")` — this only holds for linear SVMs. For the RBF kernel used here, `get_feature_importances()` returns `None` and the Features tab shows no importance chart for SVM.

### MLP

Similarly has no `feature_importances_`. The code returns `None` for MLP. This is why the importance selector in the dashboard only shows tree-based models.

---

## 8. Team Season Averages

Used to auto-fill the prediction form. For each team $t$ and feature $f$:

$$\bar{x}_{t,f} = \frac{1}{|G_t|} \sum_{g \in G_t} x_{g,f,t}$$

Where $G_t$ is the set of all games involving team $t$, and $x_{g,f,t}$ is team $t$'s value for feature $f$ in game $g$.

The mirroring logic ensures consistent averaging regardless of home/away assignment:

- When team $t$ was **home** in game $g$: their `home_ppg` value is `game["home_ppg"]`
- When team $t$ was **away** in game $g$: their `home_ppg` value is `game["away_ppg"]` (mirrored)

This produces a single season-average stat per feature per team that represents their typical performance, used to fill the "home team stats" and "away team stats" panels.

**Win count:**

$$W_t = \sum_{g \in G_t} \mathbf{1}[\text{team } t \text{ won game } g]$$

---

## 9. Prediction Confidence

For models with `predict_proba` (all except vanilla SVM without calibration — though SVM here uses `probability=True` so it also has it):

$$\text{confidence} = \max(\hat{p}(y=0 | x),\ \hat{p}(y=1 | x))$$

Since $\hat{p}(y=0|x) + \hat{p}(y=1|x) = 1$:

$$\text{confidence} = \max(\hat{p},\ 1 - \hat{p})$$

This is always in $[0.5, 1.0]$. A confidence of 0.5 means the model is maximally uncertain (50/50). A confidence of 0.95 means the model assigns 95% probability to its predicted class.

---

## 10. Promote Threshold

When the auto-learn scheduler or a manual trigger retrains the models, the new best model only replaces the current active model if:

$$\text{AUC}_{new} \geq \text{AUC}_{current} + \delta$$

Where $\delta = 0.002$ (configurable as `promote_threshold` in `config.yaml`).

**Why a threshold rather than strict improvement ($>$):** AUC estimates have variance — two training runs on slightly different data samples can produce AUC differences of ±0.005 purely from sampling noise, not genuine model improvement. The threshold filters out these noise-level fluctuations and only promotes when there is a meaningful, non-trivial gain.

**Effect:** The active model's AUC is monotonically non-decreasing over time (ignoring the edge case where a smaller dataset produces a coincidentally higher AUC). The model can only get better or stay the same — never regress.

---

*All implementations use scikit-learn 1.8.0 and XGBoost 2.x conventions. Default parameters not listed here match their scikit-learn defaults.*
