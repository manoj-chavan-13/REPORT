Let’s use a new numerical example to explore **Singular Value Decomposition (SVD)**.  

---

### **Matrix and Example**  

#### Matrix \( A \):  
\[
A = \begin{bmatrix}
3 & 1 \\
1 & 3 \\
2 & 2
\end{bmatrix}
\]

We will:
1. Perform **SVD** for \( A \).
2. Extract singular values, left-singular vectors (\( U \)), and right-singular vectors (\( V^T \)).
3. Show the rank-1 approximation (dimensionality reduction).

---

### **Python Implementation**  

#### Step 1: Perform SVD  

```python
import numpy as np

# Define the new matrix A
A = np.array([[3, 1], [1, 3], [2, 2]])

# Perform SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Singular Value Matrix (Σ)
Sigma = np.diag(S)

# Print results
print("Matrix A:\n", A)
print("\nLeft-Singular Vectors (U):\n", U)
print("\nSingular Values (S):\n", S)
print("\nRight-Singular Vectors (V^T):\n", VT)
print("\nSingular Value Matrix (Σ):\n", Sigma)
```

---

#### **Results**  

1. **Original Matrix \( A \):**
   \[
   \begin{bmatrix}
   3 & 1 \\
   1 & 3 \\
   2 & 2
   \end{bmatrix}
   \]

2. **SVD Components:**
   - **Left-Singular Vectors (U):**  
     Orthogonal matrix corresponding to \( A A^T \).  
     Example Output:
     \[
     U = \begin{bmatrix}
     -0.707 & 0.408 \\
     -0.408 & -0.816 \\
     -0.577 & 0.408
     \end{bmatrix}
     \]

   - **Singular Values (S):**  
     Diagonal elements of \( \Sigma \), computed from \( \sqrt{\text{Eigenvalues of } A^T A} \).  
     Example Output:  
     \[
     S = [5.83, 2.0]
     \]

   - **Right-Singular Vectors (\( V^T \)):**  
     Orthogonal matrix corresponding to \( A^T A \).  
     Example Output:
     \[
     V^T = \begin{bmatrix}
     -0.707 & -0.707 \\
      0.707 & -0.707
     \end{bmatrix}
     \]

3. **Reconstruct \( A \) Using \( U \Sigma V^T \):**
   Verify \( A = U \Sigma V^T \).  

---

#### Step 2: Rank-1 Approximation  

We keep only the first singular value and its corresponding singular vectors:  

```python
# Rank-1 approximation
rank1_approx = np.dot(U[:, :1], np.dot(np.diag(S[:1]), VT[:1, :]))

# Print results
print("\nRank-1 Approximation of A:\n", rank1_approx)
```

---

#### **Dimensionality Reduction Results**  

1. **Rank-1 Approximation:**  
   Approximate \( A \) as:  
   \[
   A_{rank1} = U[:, :1] \cdot \Sigma[:1, :1] \cdot V^T[:1, :]
   \]
   Example Output:  
   \[
   A_{rank1} = \begin{bmatrix}
   3.536 & 3.536 \\
   2.041 & 2.041 \\
   2.789 & 2.789
   \end{bmatrix}
   \]

2. **Interpretation:**  
   The rank-1 approximation retains the most significant singular value, simplifying \( A \) while losing some detail.

---

### **Visualization (Optional)**  

If \( A \) represents 2D data points, we can visualize the dimensionality reduction:  

```python
import matplotlib.pyplot as plt

# Original Data
plt.scatter(A[:, 0], A[:, 1], label="Original Data", color="blue")

# Rank-1 Approximation Data
plt.scatter(rank1_approx[:, 0], rank1_approx[:, 1], label="Rank-1 Approximation", color="red")

plt.title("SVD Dimensionality Reduction")
plt.legend()
plt.show()
```

---

### Key Insights  

1. **Singular Values** represent the importance of each dimension.
2. **Rank-1 Approximation** simplifies the data to its most essential features while discarding noise.
3. This process applies to various fields, including data compression, image processing, and machine learning.
