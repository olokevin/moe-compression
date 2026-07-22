# Trainability-Aware Two-Sided Whitening for MoDeGPT Type-I (MLP) Compression

This section documents a trainability-aware extension of MoDeGPT's Type-I MLP compression using the same two-sided whitening idea used earlier for linear-layer SVD compression. The key difference is that, in the MoDeGPT MLP case, the compressed object is not a single weight matrix, but the **hidden-neuron transport core** inside the MLP. MoDeGPT's original Type-I module compresses the MLP by constraining the up/gate side with a column-selection matrix and solving the down matrix in closed form from the hidden activation correlation kernel.

---

## 1. MoDeGPT Type-I MLP formulation

For a gated MLP, write

$$
u = X W_u,\qquad g = X W_g,\qquad z = u \odot \phi(g),\qquad y = z W_d,
$$

where

- $X \in \mathbb{R}^{N \times d_h}$ is the calibration input batch,
- $W_u, W_g \in \mathbb{R}^{d_h \times d_{\text{int}}}$,
- $W_d \in \mathbb{R}^{d_{\text{int}} \times d_h}$,
- $z \in \mathbb{R}^{N \times d_{\text{int}}}$ is the hidden activation after gating.

MoDeGPT treats this as a Type-I module and compresses it by selecting a reduced hidden dimension through a column-selection matrix $S$, then reconstructing the down matrix in closed form using a Nyström-style kernel on the hidden activations.

---

## 2. Core viewpoint

The hidden-neuron dimension is $d_{\text{int}}$. The original MLP can be viewed as

$$
z \xrightarrow{\,I\,} z \xrightarrow{\,W_d\,} y,
$$

where the hidden-space transport operator is simply the identity

$$
W_{\text{core}} = I_{d_{\text{int}}}.
$$

Compression replaces this full hidden transport by a rank-$k$ core $P$, induced by a low-dimensional hidden subspace. In the unconstrained setting, $P$ is a rank-$k$ matrix in $\mathbb{R}^{d_{\text{int}} \times d_{\text{int}}}$. Under MoDeGPT's structured parameterization, $P$ is realized through hidden selection and reconstruction.

Thus, the object to be preserved is not $W_d$ alone, nor $W_u$ or $W_g$ alone, but the hidden transport between forward activations and backward gradients.

---

## 3. Forward-side statistics

The forward side is straightforward. Define the hidden activation matrix

$$
Z := z = \sigma_s(X W_U) \in \mathbb{R}^{N \times d_{\text{int}}},
$$

where $W_U$ denotes the concatenated up/gate matrix in MoDeGPT's notation. Then the forward hidden covariance is

$$
C_f := Z^\top Z.
$$

This is exactly the Type-I hidden activation kernel used by MoDeGPT for forward-only compression.

If the compressed hidden core is $P$, then the forward hidden-state reconstruction loss is

$$
\mathcal{L}_f(P)
=
\|Z - ZP\|_F^2
=
\|(I-P) C_f^{1/2}\|_F^2.
$$

So $C_f$ plays the role of the forward-side whitening/statistics matrix.

---

## 4. Backward-side statistics

The backward objective is not mainly about $W_d$, because $W_d$ always directly receives gradient from the output side. The real concern is whether enough gradient information passes through the compressed hidden core so that the upstream matrices $W_u$ and $W_g$ still receive informative gradients.

Let $\delta y \in \mathbb{R}^{N \times d_h}$ denote the upstream gradient at the MLP output. Then the hidden gradient is

$$
\delta z = \delta y\, W_d^\top.
$$

For the gated branches,

$$
\delta u = \delta z \odot \phi(g),
\qquad
\delta g = \delta z \odot u \odot \phi'(g).
$$

These are precisely the signals that generate gradients for the up and gate matrices:

$$
\nabla_{W_u}\ell = X^\top \delta u,
\qquad
\nabla_{W_g}\ell = X^\top \delta g.
$$

Therefore, to preserve trainability of $W_u$ and $W_g$, we should preserve the hidden-space backward signals $\delta u$ and $\delta g$, not merely $\delta y$.

Define

$$
B_u := \delta u \in \mathbb{R}^{N \times d_{\text{int}}},
\qquad
B_g := \delta g \in \mathbb{R}^{N \times d_{\text{int}}}.
$$

Then the backward preservation loss is

$$
\mathcal{L}_b(P)
=
\|B_u - B_u P\|_F^2 + \|B_g - B_g P\|_F^2.
$$

Equivalently,

$$
\mathcal{L}_b(P)
=
\|(I-P) C_b^{1/2}\|_F^2,
$$

where

$$
C_b := B_u^\top B_u + B_g^\top B_g.
$$

Thus $C_b$ is the hidden-space covariance of the backpropagated signals that actually reach $W_u$ and $W_g$.

---

## 5. Two-sided whitening surrogate in MoDeGPT MLP

Following the two-sided whitening idea for linear-layer SVD compression, the MoDeGPT hidden-core analogue is

$$
\widetilde{\mathcal{L}}(P)
=
\|C_b^{1/2}(I-P)C_f^{1/2}\|_F^2.
$$

This is the direct Type-I MLP analogue of the linear-layer surrogate

$$
\|C_g^{1/2}(W-\hat W)C_x^{1/2}\|_F^2,
$$

except that here the compressed operator is the hidden core identity $I$ rather than a weight matrix. The earlier derivation shows that in the linear case, the optimal surrogate solution comes from truncated SVD of the two-sided whitened matrix.

Here, the corresponding MoDeGPT hidden-core matrix is

$$
\boxed{
M_{\text{MLP}} = C_b^{1/2} I C_f^{1/2} = C_b^{1/2} C_f^{1/2}.
}
$$

Interpretation:

- $C_f^{1/2}$ emphasizes hidden directions important for the forward pass,
- $C_b^{1/2}$ emphasizes hidden directions important for backpropagation into $W_u$ and $W_g$,
- $M_{\text{MLP}}$ captures hidden directions important to both.

---

## 6. Optimal unconstrained rank-$k$ hidden core

If $P$ is allowed to be any rank-$k$ matrix, then the surrogate problem is

$$
\min_{\operatorname{rank}(P)\le k}
\|C_b^{1/2}(I-P)C_f^{1/2}\|_F^2.
$$

Let

$$
Q := C_b^{1/2} P C_f^{1/2}.
$$

Then the problem is equivalent to

$$
\min_{\operatorname{rank}(Q)\le k}
\|M_{\text{MLP}} - Q\|_F^2.
$$

By Eckart--Young--Mirsky, if

$$
M_{\text{MLP}} = U \Sigma V^\top,
$$

then the optimal rank-$k$ approximation is

$$
Q_k^\star = U_k \Sigma_k V_k^\top,
$$

and therefore the optimal unconstrained hidden core is

$$
\boxed{
P_k^\star
=
C_b^{-1/2}\, U_k \Sigma_k V_k^\top\, C_f^{-1/2}.
}
$$

The corresponding minimal surrogate loss is

$$
\boxed{
\widetilde{\mathcal{L}}_k^\star
=
\sum_{i>k} \sigma_i^2\!\left(C_b^{1/2} C_f^{1/2}\right).
}
$$

This is exactly the same algebraic pattern as the two-sided-whitened SVD derivation for a linear layer, with $W$ replaced by the hidden-core identity.

---

## 7. Equivalent PSD formulation

Because

$$
M_{\text{MLP}} = C_b^{1/2} C_f^{1/2}
$$

is generally non-symmetric, it is often more convenient to work with the equivalent PSD matrix

$$
\boxed{
K_{\text{joint}} = C_f^{1/2} C_b\, C_f^{1/2}.
}
$$

The singular values of $M_{\text{MLP}}$ are the square roots of the eigenvalues of $K_{\text{joint}}$. Therefore the dominant joint forward/backward hidden directions are exactly the leading eigenspace of $K_{\text{joint}}$.

This is especially useful because MoDeGPT's Type-I compression is naturally phrased in terms of a PSD kernel and Nyström approximation.

So the trainability-aware Type-I hidden kernel becomes

$$
\boxed{
K_{\text{joint}} = C_f^{1/2} C_b\, C_f^{1/2},
\qquad
C_f = Z^\top Z,
\qquad
C_b = B_u^\top B_u + B_g^\top B_g.
}
$$

---

## 8. Relation to MoDeGPT's structured parameterization

MoDeGPT does not optimize an arbitrary rank-$k$ hidden core. Instead, it constrains the hidden compression through selection:

$$
\hat W_U = W_U S,
\qquad
\hat W_D \in \mathbb{R}^{k \times d_h},
$$

where $S$ is a $k$-column selection matrix. Under the original forward-only objective, the optimal $\hat W_D$ is

$$
\hat W_D^\star
=
(S^\top C_f S)^\dagger S^\top C_f W_D.
$$

The trainability-aware extension is therefore to replace the forward-only kernel $C_f$ by the joint kernel $K_{\text{joint}}$:

$$
\boxed{
\hat W_D^\star
=
(S^\top K_{\text{joint}} S)^\dagger S^\top K_{\text{joint}} W_D.
}
$$

Then the hidden selection matrix $S$ should be chosen by Nyström approximation of $K_{\text{joint}}$, rather than of $C_f$.

Thus the full extension is conceptually:

1. estimate forward hidden covariance $C_f$,
2. estimate backward hidden covariance $C_b$,
3. construct joint PSD kernel $K_{\text{joint}} = C_f^{1/2} C_b C_f^{1/2}$,
4. apply MoDeGPT Type-I Nyström selection to $K_{\text{joint}}$,
5. reconstruct $W_D$ using the same closed-form formula but with $K_{\text{joint}}$.

---

## 9. Why this matches the trainability objective

This directly addresses the concern that, after compression, enough gradient must pass through the transformed hidden core so that $W_u$ and $W_g$ still receive informative updates.

- Preserving $C_f$ alone keeps hidden directions that matter for the forward pass.
- Preserving $C_b$ alone keeps hidden directions that matter for gradient transport into the upstream factors.
- Preserving the two-sided-whitened core through $C_b^{1/2} I C_f^{1/2}$ keeps hidden directions important to both.

So the trainability-aware object in MoDeGPT Type-I is the hidden core identity, weighted jointly by forward activity and backward transport.

---

## 10. Compact final formulas

For gated MLP,

$$
u = XW_u,\qquad g = XW_g,\qquad z = u \odot \phi(g),\qquad y = zW_d.
$$

Given upstream gradient $\delta y$,

$$
\delta z = \delta y\, W_d^\top,
\qquad
\delta u = \delta z \odot \phi(g),
\qquad
\delta g = \delta z \odot u \odot \phi'(g).
$$

Define

$$
C_f = Z^\top Z,
\qquad
C_b = B_u^\top B_u + B_g^\top B_g.
$$

Then the trainability-aware two-sided-whitened MoDeGPT surrogate is

$$
\boxed{
\widetilde{\mathcal{L}}(P)
=
\|C_b^{1/2}(I-P)C_f^{1/2}\|_F^2.
}
$$

The corresponding joint hidden matrix is

$$
\boxed{
M_{\text{MLP}} = C_b^{1/2}C_f^{1/2},
}
$$

and the equivalent PSD kernel is

$$
\boxed{
K_{\text{joint}} = C_f^{1/2} C_b\, C_f^{1/2}.
}
$$

If unconstrained rank-$k$ core optimization is allowed, the optimal core is

$$
\boxed{
P_k^\star
=
C_b^{-1/2} U_k \Sigma_k V_k^\top C_f^{-1/2},
\qquad
U\Sigma V^\top = \operatorname{SVD}(C_b^{1/2}C_f^{1/2}),
}
$$

with minimal surrogate loss

$$
\boxed{
\widetilde{\mathcal{L}}_k^\star
=
\sum_{i>k}\sigma_i^2(C_b^{1/2}C_f^{1/2}).
}
$$

Under MoDeGPT's structured hidden-selection form, replace $C_f$ by $K_{\text{joint}}$ in the Nyström reconstruction:

$$
\boxed{
\hat W_D^\star
=
(S^\top K_{\text{joint}} S)^\dagger S^\top K_{\text{joint}} W_D.
}
$$

---

## 11. Practical note

In implementation, it is often better to regularize and normalize:

$$
\bar C_f = \frac{C_f}{\operatorname{tr}(C_f)},
\qquad
\bar C_b = \frac{C_b}{\operatorname{tr}(C_b)},
$$

and use

$$
K_{\text{joint}}
=
\bar C_f^{1/2}\, \bar C_b\, \bar C_f^{1/2} + \lambda I.
$$

This avoids scale imbalance and numerical instability.

---

## 12. Summary

The trainability-aware extension of MoDeGPT Type-I compression is obtained by applying two-sided whitening not to a single linear weight matrix, but to the **hidden-neuron transport core** of the MLP. The forward statistics are captured by the hidden activation covariance $C_f$, and the backward statistics relevant to training $W_u$ and $W_g$ are captured by the hidden gradient covariance $C_b$. Their joint interaction is summarized by

$$
C_b^{1/2} I C_f^{1/2} = C_b^{1/2} C_f^{1/2},
$$

or equivalently by the PSD kernel

$$
K_{\text{joint}} = C_f^{1/2} C_b C_f^{1/2}.
$$

This gives the precise MoDeGPT analogue of the two-sided-whitening idea developed earlier for linear-layer SVD compression.


## Implementation

### `nystrom_combined` — trainability-aware joint kernel

Forward-only selection keeps directions that matter for the **forward** pass,
but says nothing about whether enough gradient reaches the upstream `W_u`/`W_g`
after compression. `nystrom_combined` swaps the forward kernel for a
**joint forward+backward kernel** that weights hidden directions by both
forward activity and backward transport (derivation in
`docs/plans/nystrom_combined.md`):

$$
\bar C_f = \frac{C_f}{\operatorname{tr} C_f},\quad
\bar C_b = \frac{C_b}{\operatorname{tr} C_b},\quad
K_{\text{joint}} = \bar C_f^{1/2}\,\bar C_b\,\bar C_f^{1/2} + \lambda I,
$$

where `C_f = Zᵀ Z` is the forward hidden covariance and `C_b` is the
**backward hidden-gradient covariance** — the covariance of `δz`, the gradient
flowing *into* `down_proj`. `δz` is the shared signal that drives both gated
branches (`δu = δz ⊙ φ(g)`, `δg = δz ⊙ u ⊙ φ'(g)` are per-neuron rescalings of
it), so `C_b = δzᵀ δz` is the dominant term of the plan's
`C_b = B_uᵀ B_u + B_gᵀ B_g`. Selection and reconstruction then use the same
rules as `nystrom` with `C_f → K_joint`:

$$
\text{score}_i = \operatorname{diag}\big((K_{\text{joint}}+\lambda I)^{-1} K_{\text{joint}}\big),
\qquad
\hat W_D = (S^\top K_{\text{joint}} S)^{+} S^\top K_{\text{joint}}\, W_D .
$$

(`structured/nystrom.py:nystrom_combined_compress_mlp`). When the backward
signal is isotropic (`C_b ∝ I`), `K_joint ∝ C_f` and the selected neuron set
reduces exactly to forward-only `nystrom`
(`tests/test_nystrom_combined.py`).
