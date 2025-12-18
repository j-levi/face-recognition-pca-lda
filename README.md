# Face Recognition — PCA & LDA (Mathematical Notes)

## Overview
This project demonstrates face recognition using Principal Component Analysis (PCA) for dimensionality reduction and Linear Discriminant Analysis (LDA / Fisherfaces) for supervised discrimination. The notebooks show preprocessing, projection, classification, and reconstruction.

## Data and Notation
- Let X = [x_1, x_2, ..., x_n] ∈ R^{d×n} be column-stacked vectorized face images (each x_i ∈ R^d).
- Let c be the number of classes (identities); class i has n_i samples, ∑_i n_i = n.
- μ = (1/n) ∑_{i=1}^n x_i denotes the global mean; μ_j denotes class j mean.

## Principal Component Analysis (PCA)
PCA finds orthonormal directions (principal components) that maximize variance.

1. Center data: \(\tilde X = X - μ1^T\) where 1 is an all-ones vector.
2. Covariance (empirical):
$$
C = \frac{1}{n} \tilde X \tilde X^T \in R^{d\times d}.
$$
3. Eigen-decomposition: solve
$$
C u_k = \lambda_k u_k, \quad \lambda_1 \ge \lambda_2 \ge \dots.
$$
4. Select top K eigenvectors U_K = [u_1, ..., u_K]. Project a centered image x as
$$
y = U_K^T (x - μ) \in R^K.
$$
Reconstruction (rank-K approximation):
$$
\hat x = μ + U_K y = μ + U_K U_K^T (x - μ).
$$
Notes: for d ≫ n the eigen-decomposition is computed via the smaller n×n matrix \(\tilde X^T \tilde X\) and connection between eigenvectors through \(\tilde X \tilde X^T u = \lambda u\).

## Linear Discriminant Analysis (LDA / Fisherfaces)
LDA finds projections that maximize class separability using between-class and within-class scatter matrices.

Define:
$$
S_W = \sum_{j=1}^c \sum_{x \in C_j} (x - μ_j)(x - μ_j)^T,
\\
S_B = \sum_{j=1}^c n_j (μ_j - μ)(μ_j - μ)^T.
$$
Fisher's criterion for a projection vector w:
$$
J(w) = \frac{w^T S_B w}{w^T S_W w}.
$$
Maximizing J leads to the generalized eigenvalue problem:
$$
S_B w = \lambda S_W w.
$$
In practice, when d is large and S_W is singular, we apply PCA first to reduce dimensionality to at most n-c, then solve the LDA eigenproblem in the PCA subspace. This pipeline (PCA → LDA) yields the Fisherfaces used for classification.

## Classification
After projection (using PCA, LDA, or PCA+LDA), classification is typically done via nearest-neighbor in the projection space or simple linear discriminant rules based on class means.

## Implementation Notes
- Centering, numeric stability (regularization of S_W by adding small εI), and dimensionality choices (K for PCA) are critical.
- Use SVD for robust PCA: \(\tilde X = U Σ V^T\) and choose top singular vectors.

## References
- Jolliffe, Principal Component Analysis.
- Belhumeur, Hespanha, & Kriegman, "Eigenfaces vs. Fisherfaces." 1997.

## Files
See notebooks for experiments: faces.ipynb
