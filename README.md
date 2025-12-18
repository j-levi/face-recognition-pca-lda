# Mini Project 3 — Eigenfaces using SVD
**Author:** James Levi  
**Course:** Data Mining and Machine Learning  
**Date:** October 15, 2025  

---

## Overview
This project applies **Singular Value Decomposition (SVD)** to a dataset of facial images to understand how high-dimensional data can be represented efficiently using a smaller set of basis vectors, or **eigenfaces**. The focus is on both the implementation and the mathematical reasoning behind why SVD works for dimensionality reduction and image reconstruction.

---

## Mathematical Derivation

For any matrix \( A_{n \times m} \), there exists a **Singular Value Decomposition**:

\[
A = U \Sigma V^T
\]

where  
- \( U \in \mathbb{R}^{n \times n} \) has orthonormal columns (\(U^T U = I_n\))  
- \( V \in \mathbb{R}^{m \times m} \) has orthonormal columns (\(V^T V = I_m\))  
- \( \Sigma \in \mathbb{R}^{n \times m} \) is diagonal with non-negative singular values \( \sigma_1 \ge \sigma_2 \ge \dots \ge 0 \)

---

### Relationship to Eigenvalue Decomposition

From the definitions,

\[
A^T A = V \Sigma^T \Sigma V^T
\]

and

\[
A A^T = U \Sigma \Sigma^T U^T
\]

Thus,  
- The columns of \(V\) are eigenvectors of \(A^T A\)  
- The columns of \(U\) are eigenvectors of \(A A^T\)  
- The singular values \(\sigma_i\) are the square roots of the nonzero eigenvalues of both \(A^T A\) and \(A A^T\)

---

### Rank-1 Expansion

Each matrix \(A\) can be expressed as the sum of rank-1 outer products:

\[
A = \sum_{i=1}^{r} \sigma_i u_i v_i^T
\]

where \(r = \text{rank}(A)\).  
Each term \( \sigma_i u_i v_i^T \) represents one fundamental “mode” of variation — in this case, a distinct facial pattern.

---

### Truncated SVD

By keeping only the top \(k < r\) terms, we get the **best rank-\(k\)** approximation of \(A\) in the least-squares sense:

\[
\tilde{A} = \sum_{i=1}^{k} \sigma_i u_i v_i^T
\]

This is what allows reconstruction using only a few eigenfaces. The larger the singular value, the more information that component carries.

---

## Implementation Summary

1. **Data Preparation** — The dataset `allFaces.mat` contains 38 people, each represented by multiple grayscale images. Faces were vectorized into columns of a large matrix.  
2. **Training Set** — Persons 0–35 were used for training; their average was computed and subtracted to center the data.  
3. **SVD Decomposition** — Applied `np.linalg.svd(X, full_matrices=False)` to compute eigenfaces and singular values.  
4. **Visualization** — The first 54 eigenfaces were plotted, showing distinct lighting and facial feature patterns.  
5. **Reconstruction** — Reconstructed the first image of person 36 with different truncation levels (`r = 5, 10, 100, 200, 800`).  
6. **Singular Value Analysis** — A semi-log plot revealed an exponential decay, confirming that only the first few components carry meaningful variance.  
7. **Classification** — Compared eigenface coefficients for different people (e.g., person 1 vs person 6) and showed clear separation in 2D space.

---

## Key Insights

- \(U\) and \(V\) form orthogonal bases for the image and feature spaces.  
- The mean face centers the data, allowing SVD to capture deviations that define identity.  
- The first few singular values dominate — truncation gives near-optimal compression.  
- Eigenfaces derived from SVD reveal how facial information is distributed across principal directions.  

---

## Tools Used
- Python 3.11  
- NumPy  
- SciPy (`loadmat`)  
- Matplotlib  

---

## Author Note
Written and implemented by **James Levi** as part of the FAU Data Mining and Machine Learning course. The math derivation follows my handwritten notes connecting SVD to eigenvalue decomposition and rank-\(r\) approximation theory.
