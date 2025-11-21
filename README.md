# SVD + LDA Image Classification Project (Starter)

This repository is a starter kit for the final programming project.

You will implement:
- `power_method` for the dominant eigenvalue/eigenvector of a symmetric matrix.
- `svd_compress` for rank-k image approximation using SVD.
- `svd_features` to build feature vectors from image singular values.
- `lda_train` to train a two-class Linear Discriminant Analysis (LDA) classifier.
- `lda_predict` to apply the trained LDA classifier.

The autograder will import `project.py` and call these functions with specific
signatures. **Do not change the function names or signatures.**

## Files

- `project.py` — main implementation file with function stubs.
- `project_data_example.npz` — tiny synthetic dataset for local testing.
- `requirements.txt` — minimal Python dependencies.
- `.gitignore` — standard Python ignores.

## Quick start

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `project.py` and implement each function where indicated.
4. Try running the self-test:

   ```bash
   python project.py
   ```

   This will:
   - Load `project_data_example.npz` if present.
   - Extract SVD features.
   - Train an LDA classifier.
   - Print the test accuracy on the tiny synthetic dataset.

In the real assignment, the Gradescope autograder will provide a different
`project_data.npz` with more realistic images and labels.
