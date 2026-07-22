




# Training-Aware Compression

## Two-Sided Whitening for Linear Layers

The core idea is to compress weight matrices $W$ via SVD while accounting for both input and output curvature. The optimization problem is:

$$
\hat{W} = \arg\min_{\hat{W}} \left\| C_g^{1/2}(W - \hat{W})C_x^{1/2} \right\|_F^2,
$$

where:

- $C_x = \sum_t \mathbf{x}_t \mathbf{x}_t^{\top}$ is the input activation covariance,
- $C_g = \sum_t \mathbf{g}_t \mathbf{g}_t^{\top}$ is the backpropagated gradient covariance.

**Solution:** Compute the SVD of the two-sided whitened matrix $M = C_g^{1/2} W C_x^{1/2}$, then truncate to rank $k$.

**Interpretation:** This two-sided whitening is a Hessian estimate for the layer weights:

$$
H_W = \mathbb{E}\left[\mathbf{x}\mathbf{x}^{\top} \otimes \mathbf{g}_y\mathbf{g}_y^{\top}\right] \approx \mathbb{E}[\mathbf{x}\mathbf{x}^{\top}] \otimes \mathbb{E}[\mathbf{g}_y\mathbf{g}_y^{\top}].
$$

Truncation **discards the least loss-sensitive subspace**. The surviving high-curvature directions are exactly those that training keeps actively updating.

---

## Closed-Form Loss-Aware Decomposition

For a weight decomposition $W \to \hat{W} = BA$, the second-order Taylor expansion of the task-loss change is:

$$
\Delta L(w) \approx (\mathbf{w} - \hat{\mathbf{w}}) H_w (\mathbf{w} - \hat{\mathbf{w}})^{\top}.
$$

Using the K-FAC approximation $H_w = C_x \otimes C_g$ with Cholesky factors $C_x = L_x L_x^{\top}$ and $C_g = L_g L_g^{\top}$:

$$
\Delta L(W) = \mathrm{vec}(\Delta W) (C_x \otimes C_g) \mathrm{vec}(\Delta W)^{\top} = \left\| L_g^{\top} \Delta W L_x \right\|_F^2,
$$

where $\Delta W = W - \hat{W}$.

**Optimal Solution:** Truncated SVD of the two-sided whitened matrix $\tilde{W} = L_g^{\top} W L_x$ **minimizes the loss change** $\Delta L$. This gives a closed-form optimal decomposition.

---

## Training-Aware MLP Compression via Nyström

For gated MLPs, compression applies two-sided whitening to the **hidden-neuron transport core**, not a single weight matrix.

### Hidden Core Formulation

Given an MLP:

$$
u = XW_u, \quad g = XW_g, \quad z = u \odot \phi(g), \quad y = zW_d,
$$

the hidden core is the transport operator from hidden activations to outputs. Compression constrains this through a selection matrix $S$.

### Forward and Backward Statistics

**Forward covariance** of hidden activations:

$$
C_f = Z^{\top}Z, \quad Z = z.
$$

**Backward covariance** of signals flowing into $W_u$ and $W_g$:

$$
\delta u = \delta z \odot \phi(g), \quad \delta g = \delta z \odot u \odot \phi'(g),
$$

$$
C_b = B_u^{\top}B_u + B_g^{\top}B_g, \quad B_u = \delta u, \quad B_g = \delta g.
$$

### Joint Forward+Backward Kernel

The trainability-aware two-sided whitening for the hidden core is captured by the **joint kernel**:

$$
K_{\text{joint}} = \bar{C}_f^{1/2} \bar{C}_b \bar{C}_f^{1/2} + \lambda I,
$$

where normalization prevents scale imbalance:

$$
\bar{C}_f = \frac{C_f}{\operatorname{tr}(C_f)}, \quad \bar{C}_b = \frac{C_b}{\operatorname{tr}(C_b)}.
$$

This kernel captures hidden directions important to **both** forward propagation and backward gradient flow into the upstream factors.

### Selection and Reconstruction

Hidden neurons are selected by Nyström approximation of $K_{\text{joint}}$:

$$
\text{score}_i = \operatorname{diag}\left((K_{\text{joint}} + \lambda I)^{-1} K_{\text{joint}}\right),
$$

The down-projection is reconstructed via:

$$
\hat{W}_D = (S^{\top}K_{\text{joint}}S)^{+} S^{\top}K_{\text{joint}} W_D,
$$

where $S$ is the column-selection matrix corresponding to the top-scoring neurons.

**When backward signal is isotropic** ($C_b \propto I$), the joint kernel reduces to $K_{\text{joint}} \propto C_f$, recovering the forward-only Nyström selection.
