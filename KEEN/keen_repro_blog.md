# Reproduction of the KEEN paper on estimation of knowledge on QA dataset

## Motivation
KEEN (Gottesman & Geva, 2024) estimates latent knowledge of a subject entity by probing internal representations rather than relying on model generated answers. The core hypothesis is that if a model “knows” a fact, then that knowledge should be recoverable from hidden representations of the subject entity before any answer tokens are produced. This reproduction applies the KEEN approach to `gpt2-xl` model and evaluates how well trained probes predict gold QA accuracy.

## Problem setup
- **Model:** `gpt2-xl`.
- **Dataset:** PopQA subject–relation questions ([link](https://huggingface.co/datasets/kokolamba/keen_popqa_gpt2xl_generations)).
- **Subjects:** 3,491 rows of subjects data for `gpt2-xl` (vs. 3,475 in the paper’s Llama-derived data).
- **Prompt template:** `This document describes {subject}` (used to extract subject representations).
- **Gold labels:** per-subject QA accuracy (fraction of questions answered correctly).
- **Probe:** logistic linear probe \(\hat{y} = \sigma(\theta^\top \mathbf{z})\), trained to maximize correlation with gold QA accuracy.
- **Training hyperparameters:** `lr=1e-5, epoch=500, wd=0.01, cosine scheduling`.
- **GPUs:** Colab's T4 and Runpod's A4 GPU.
- **Source paper:** [Gottesman & Geva (2024)](https://arxiv.org/pdf/2406.12673). 

## Methodology
First, we created completions for all subjects using the prompt template and `gpt2-xl` model. We then generated gold scores(QA accuracy) per subject entity in the data. Later on, we extract hidden representations(residual stream, mlp outputs and attention outputs and VP-k representations) at layer `33, 34, and 35` of `gpt2-xl` and then train probes on the normalized average of these representations across the 3 layers. We chose these "late layers" because [previous work](https://arxiv.org/pdf/2304.14767) shows that hidden representation in these layers are feature rich. Finally, knowledge estimation for QA subject was measured by computing the correlation between probe's predicted score and gold scores on the test set.

>__VP-k__ representations were obtained by extracting the logits of tokens that corresponds to the `top-k` absolute weights of the trained probe on full logits in model's vocab space.

>I tried a couple of hyperparameters before eventually settling for the one used throughout the experiment. I restricted the epoch to 500 to reduce computational cost. Some of the results in the paper have epoch on the order of 3000 steps.

>At "late layers", subjects such as `Elon musk` in these layers would have a hidden representation with an understanding that he is billionaire, the owner of X(formerly twitter), and one of biggest proponents of pronatalism. 

## Probes on hidden representations
We trained probes on hidden representations on subject entities such as residual stream, mlp outputs, attention outputs, and VP-{50257, 200, 100, 50, 25, 10}. It turns out VP-50257 has the highest correlation followed by residual stream. This follows the work in the paper where the strongest correlation were set to be in the range between 0.60-0.68. we were able to achieve this with significantly less epoch compared to what's used in the paper. 

The weak correlations in VP-{100, 50, 25, 10} as shown in the `Table 1` can be further improved by increasing number of epochs.

| Representations | Test Correlation |
| --- | --- |
| Residual Stream | 0.62 |
| MLP Outputs | 0.61 |
| Attention Outputs | 0.38 |
| VP-50257 | 0.64 |
| VP-200 | 0.49 |
| VP-100 | 0.33 |
| VP-50 | 0.17 |
| VP-25 | 0.22 |
| VP-10| 0.06 |

> Table 1: Average of test correlations across 3 runs for various hidden representations.

Generally, as k increase, the stronger the correlation and the more likely VP-k will correctly predict model's knowledge of a subject. However, there appears to be a slight drop at `k=50` before continuing the uptrend. This is slightly different from what's obtained in the paper which is normally smoothly increasing before plateauing at around `k=50257`.

![VP-k correlation plot](vp_k_corr.png)


For HS(residual stream)/MLP, the slopes are around 0.3 meaning that the probe’s predictions move in the right direction as true QA accuracy increases (i.e it is monotonic), but the movement is somewhat compressed. The probe tends to predict scores in a narrower band (e.g. 0.15-0.50) rather than spanning 0-1, which is why the fit is far from y=x even though correlation can still be reasonably high. However, the slope for VP-50 is near-zero (0.024) showing that there is a collapse in its predictive power. In contrast with KEEN paper's findings, residual stream and MLP have better knowledge estimation power when compared to VP-50 representation. Nevertheless, VP-50's predictive power can be improved through hyperparameter tuning.

![Scatter plot](scatter-plot-corr.png)

## Analysis of positively and negatively weighted tokens

We identify the tokens associated with the largest absolute weights for `full logits(VP-50257)` probe because they have most effect on the predicted score. Next, for VP-50, we compare the median rank of tokens with positive and negative weights in the vocabulary space for subjects with high QA accuracy (1.0) and low QA accuracy (0.0). For low QA accuracy subjects, the median rank of negative weight tokens is generally higher than that of positive weight tokens. On the contrary, for high QA accuracy subjects, the median rank of negative weight tokens is generally lower than that of positive weight tokens. Please note, the higher the rank, the lower the number. For example, 0 has the highest rank. Ultimately, based on the plot in `Figure 2`, there is still about "__>15000__" tokens that differentiates between subjects the model knows about and those it does not know much about.


![Median rank difference plot](median-rank-diff.png)

> Figure 2: Distribution of median rank difference for low and high qa accuracy subjects

## Conclusion
Residual stream and MLP probes show the strongest correlations (~0.62), while attention and VP-50 features lag. The VP-50 dip and slope collapse remain the main discrepancy. However, I believe it can be made more predictive through hyperparameter tuning.

#### BibTex

```
@article{abdulhakeem2026keen-paper-reproduction,
  title={Reproduction of the KEEN paper on estimation of knowledge on QA dataset},
  author={Abdulhakeem, Adefioye},
  year={2026},
  month={January},
  url={add-url-link}
}
```