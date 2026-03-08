# Experimental results on Semi-NMF on toy model

## Motivation
Based on the claims on the usefulness of features extracted from MLP activations using SemiNMF by Shafran et al. I tried to test the usefulness of the approach as a potential alternative to SAE. 

## Experiment & Methodology
To test the usefulness or interpretability of features in LLMs using SemiNMF. I used a toy data procedurally created by [HERE] with ground truth features and activations. The belief is that if ground-truth features are known, we can easily test the efficacy of the method. The metric used for assessing the quality of the model is `mean max cosine similarity`(MMCS). The higher the value the more performant the method for extracting meaningful features from the input data. Subsequently, SemiNMF was trained on activations to extract learned features and MMCS was used to determine how similar it is to ground-truth features

## Experiment 1.
Considering that decomposed matrices(Z, Y) were solved for analytically, I tried to vary 3 hyperparameters namely: ridge regularizer, sparsity regularizer and rank of learned feature.
- `k_scale` (controls learned feature rank scale): `0.75`, `1.00`, `1.25`, `1.50`, `2.00`
- `closed_form_eqn_reg` (ridge regularizer): `1e-8`, `1e-7`, `1e-6`, `1e-5`, `1e-4`
- `sparsity_reg` (sparsity regularizer): `0.0`, `1e-4`, `1e-3`, `1e-2`, `1e-1`

After a grid search over all these hyperparameters, the optimal value for `k_scale, closed_form_eqn_reg and sparsity_reg` are `2.00, 1e-4, 0.1` respectively with __MMCS__ of 0.25. This value is still significantly small suggesting trying tweaking other knobs within the same model(e.g initializations) or other approaches altogether.

#### TODO
- [ ] Use maybe SVD, knn or other initializations to validate its quality.
- [ ] Try different loss functions beside frobenius norm as reconstruction loss.

