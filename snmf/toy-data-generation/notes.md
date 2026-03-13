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

After a grid search over all these hyperparameters using random initializations, the optimal value for `k_scale, closed_form_eqn_reg and sparsity_reg` are `2.00, 1e-4, 0.1` respectively with __MMCS__ of 0.25. This value is still significantly small suggesting trying tweaking other knobs within the same model(e.g initializations) or other approaches altogether.

SVD initializations give 0.253 while Knn at current default gives 1.00. Knn perfect similarity is likely a result of overfitting.

We also varied dataset sizes between 5000 - 100000 with range of similarity scores between 0.246 - 0.253. This suggest dataset size has minimal impact on similarity scores and ultimately weak improvement in the quality of learned features extracted by SNMF.

Retried SemiNMF without sparsity loss on random initialization with ridge regularizer of 1e-4, K=2G=1024 with a similarity of 0.25. Still rather small and within range of what we've gotten so far after doing hyperparameter grid search.

Used sklearn NMF api to fit NMF on 5000 activations and started by doing a sweep over alpha_h from 1e-4 -- 1, G=K=512, with highest MMCS at 0.195. Average training time roughly about 4mins.

Subsequently, I trained NMF on 100_000 activations with alpha_h of 1e-4 with K=2G leading with similarity score of 0.196 for __5hrs__. My conclusion here is that there appears not to be a significant difference in similarity score with varying dataset size.

Using sparse_nmf based on hoyer's method, similarity score of 0.2 was obtained on the toy model. This thereby shows how terrible all of these methods are at extracting features. Another disadvantage of these NMF methods besides SemiNMF is that they are computationally inefficient(i.e high time complexity). The conclusion therefore is that if we cannot get a reasonable score for a toy model with minimal data, we can be reasonably sure that they would perform poorly on real models with huge parameter size. 

## TODO:
- [X] After experimenting with SemiNMF. Retry with optimal parameters but now without sparsity loss.
- [X] Maybe try different loss functions for SemiNMF besides frobenious norm for reconstruction loss.
- [X] Run NMF and tweak hyperparameters to see how far it similarity goes.
- [X] Run SparseNMF(Hoyer's method for NMF with sparsity constraint) and tweak hyperparams to see how far similarity goes.


