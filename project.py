import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================

def power_method(A, x0, maxit, tol):
    """Approximate the dominant eigenvalue and eigenvector of a real symmetric matrix A.

    Parameters
    ----------
    A : (n, n) ndarray
        Real symmetric matrix.
    x0 : (n,) ndarray
        Initial guess for eigenvector (nonzero).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence in relative change of eigenvalue.

    Returns
    -------
    lam : float
        Approximate dominant eigenvalue.
    v : (n,) ndarray
        Approximate unit eigenvector (||v||_2 = 1).
    iters : int
        Number of iterations performed.
    """
    A = np.asarray(A, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    # Basic checks
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n,n) array.")
    n = A.shape[0]
    if x0.size != n:
        raise ValueError("x0 must have length n.")
    if maxit <= 0:
        raise ValueError("maxit must be a positive integer.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    # Normalize initial vector
    normx = np.linalg.norm(x0)
    if normx == 0:
        raise ValueError("x0 must be nonzero.")
    v = x0 / normx

    lam_prev = None

    for it in range(1, maxit + 1):
        # Power iteration step
        w = A @ v
        normw = np.linalg.norm(w)
        if normw == 0:
            # A v = 0 => eigenvalue 0 in direction v
            return 0.0, v, it

        v = w / normw  # keep unit length

        # Rayleigh quotient for eigenvalue estimate (good for symmetric A)
        lam = float(v @ (A @ v))

        # Convergence check: relative change in eigenvalue
        if lam_prev is not None:
            denom = max(1.0, abs(lam))
            if abs(lam - lam_prev) / denom < tol:
                return lam, v, it

        lam_prev = lam

    # If not converged, return last estimate
    return lam_prev, v, maxit

def svd_compress(image, k):
    """Compute a rank-k approximation of a grayscale image using SVD.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    k : int
        Target rank (1 <= k <= min(m, n)).

    Returns
    -------
    image_k : (m, n) ndarray
        Rank-k approximation of the image.
    rel_error : float
        Relative Frobenius error ||image - image_k||_F / ||image||_F.
    compression_ratio : float
        (Number of stored parameters in image_k) / (m * n).
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array (grayscale).")

    m, n = A.shape
    r = min(m, n)

    k = int(k)
    if not (1 <= k <= r):
        raise ValueError(f"k must satisfy 1 <= k <= min(m,n) = {r}.")

    # SVD: A = U diag(s) V^T
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate to rank k
    Uk = U[:, :k]          # (m,k)
    sk = s[:k]             # (k,)
    Vk = Vt[:k, :]         # (k,n)

    # Reconstruct rank-k approximation: Uk @ diag(sk) @ Vk
    image_k = (Uk * sk) @ Vk

    # Relative Frobenius error
    denom = np.linalg.norm(A, ord="fro")
    if denom == 0:
        rel_error = 0.0 if np.linalg.norm(image_k, ord="fro") == 0 else np.inf
    else:
        rel_error = float(np.linalg.norm(A - image_k, ord="fro") / denom)

    # Compression ratio: store Uk (m*k) + sk (k) + Vk (k*n) parameters
    stored_params = m * k + k + k * n
    compression_ratio = float(stored_params / (m * n))

    return image_k, rel_error, compression_ratio


# =========================================================
# 3. SVD-based feature extraction
# =========================================================

def svd_features(image, p):
    """Extract SVD-based features from a grayscale image.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    p : int
        Number of leading singular values to use (p <= min(m, n)).

    Returns
    -------
    feat : (p + 2,) ndarray
        Feature vector consisting of:
        [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array (grayscale).")

    m, n = A.shape
    r = min(m, n)

    p = int(p)
    if p < 1 or p > r:
        raise ValueError(f"p must satisfy 1 <= p <= min(m,n) = {r}.")

    # Compute singular values (sorted descending)
    s = np.linalg.svd(A, compute_uv=False, full_matrices=False)

    # Normalize the first p singular values (sum-normalization)
    s_p = np.log(s[:p].copy() + 1e-8)
    denom = s.sum()
    if denom > 0:
        s_p /= np.linalg.norm(s_p)
    else:
        # all-zero image -> singular values are all zero
        s_p[:] = 0.0

    # Energy ratios r_0.9 and r_0.95:
    # smallest k such that cumulative energy >= alpha of total energy
    # Energy uses squared singular values (since ||A||_F^2 = sum sigma_i^2)
    energy = s**2
    total_energy = energy.sum()

    if total_energy == 0:
        r_09 = 0.0
        r_095 = 0.0
    else:
        cum = np.cumsum(energy) / total_energy
        k09 = int(np.searchsorted(cum, 0.90) + 1)   # +1 for 1-based count
        k095 = int(np.searchsorted(cum, 0.95) + 1)
        r_09 = k09 / r
        r_095 = k095 / r

    # Assemble feature vector
    feat = np.empty(p + 2, dtype=float)
    feat[:p] = s_p
    feat[p] = r_09
    feat[p + 1] = r_095

    return feat


# =========================================================
# 4. Two-class LDA: training
# =========================================================

def lda_train(X, y):
    """Train a two-class LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix (rows = samples, columns = features).
    y : (N,) ndarray
        Labels, each 0 or 1.

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction vector (not necessarily unit length).
    threshold : float
        Threshold in 1D projected space for classifying 0 vs 1.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N,d).")
    N, d = X.shape
    if y.size != N:
        raise ValueError("y must have length N.")
    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError("y must contain only labels 0 and 1.")

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes must be present to train LDA.")

    # Class means
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    # Within-class scatter: S_W = sum (x-mu_c)(x-mu_c)^T over both classes
    Z0 = X0 - mu0
    Z1 = X1 - mu1
    SW = Z0.T @ Z0 + Z1.T @ Z1  # (d,d)

    # Regularize a tiny bit for numerical stability (helps if SW is singular)
    reg = 1e-6
    SW_reg = SW + reg * np.eye(d)

    # Fisher LDA direction: w ∝ SW^{-1} (mu1 - mu0)
    b = (mu1 - mu0)
    w = np.linalg.solve(SW_reg, b)  # avoid explicit inverse

    # Project training data onto 1D line: z = X w
    z0 = X0 @ w
    z1 = X1 @ w

    # Threshold: midpoint between projected class means
    m0 = float(z0.mean())
    m1 = float(z1.mean())
    threshold = 0.5 * (m0 + m1)

    # Optional: orient w so that class 1 tends to be above the threshold
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, threshold


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================

def lda_predict(X, w, threshold):
    """Predict class labels using a trained LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    w : (d,) ndarray
        Discriminant direction (from lda_train).
    threshold : float
        Threshold (from lda_train).

    Returns
    -------
    y_pred : (N,) ndarray
        Predicted labels (0 or 1).
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    if X.ndim == 1:
        # Single sample: reshape to (1, d)
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N,d).")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch: X.shape[1] must equal len(w).")

    # Project onto discriminant direction
    scores = X @ w

    # Classify based on threshold
    y_pred = (scores >= threshold).astype(int)

    return y_pred


# =========================================================
# Simple self-test on the example data
# =========================================================

def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Sanity check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(5, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    # Build feature matrices
    def build_features(X):
        feats = []
        for img in X:
            feats.append(svd_features(img, p))
        return np.vstack(feats)

    try:
        Xf_train = build_features(X_train)
        Xf_test = build_features(X_test)
    except NotImplementedError:
        print("Implement 'svd_features' first to run this example.")
        return

    print("Feature dimension:", Xf_train.shape[1])

    try:
        w, threshold = lda_train(Xf_train, y_train)
    except NotImplementedError:
        print("Implement 'lda_train' first to run this example.")
        return

    try:
        y_pred = lda_predict(Xf_test, w, threshold)
    except NotImplementedError:
        print("Implement 'lda_predict' first to run this example.")
        return

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # This allows students to run a quick local smoke test.
    _example_run()
