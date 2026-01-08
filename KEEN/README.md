# KEEN Paper
This paper attempts to estimate knowledge using hidden states of known entities in the wild.

# Progress so far
- [x] Generated completions for per-subject query/questionn using `gpt2-xl`.
- [x] Generate gold scores(QA accuracy) for per-subject question.
- [x] Generate internal/hidden representations(residual stream).
- [x] Train a linear/MLP probe on the hidden states(residual stream) and then evaluate the correlation. What real-valued vectors are we correlating really? It is the correlation between gold scores and probes' predicted scores for per-subject question.
- [ ] Experiment with features beyond hidden representation. I will try attention outputs and fully-connected activations(Will need to fully understand what these 2 features are).
- [ ] Train probe on representation in the vocab space and experiment with top-k values as used in the paper

# TODO:
- [ ] Create script that essentially trains probe on representations in vocab space. `2 runs` with deterministic seed seems to create unstable results for all VP-{200, 100, 50, 25, 10} features especially at `500 epochs`. To keep things rather simple, I will be using the number of epochs from the KEEN paper. Using a script, I will be using the seed number used in the paper.
- [ ] Train probe on fully-connected activations(aka MLP output) and self-attention output and how to extract them perhaps using hooks.

# Findings
- After training probe on residual stream, I was able to achieve a correlation of 0.64 between predicted score and gold QA accuracy albeit with batch size of 1 instead of 32 used in the paper. hyperparameters used are `lr=1e-5, max_iter=500 and batch_size=1 with sigmoid and cosine scheduling with wd=0.01`.

# Questions I need answer to
- Why is correlation result better with batch_size of 1 instead of 32 when training probe on residual stream

# Paper limitation to be addressed

# My own new spins to be considered
- [ ] Will filler tokens/prompt repetition enhance feature representations(residual stream, representation in vocab space etc) and ultimately increase correlation between predicted scores and gold scores. This is motivated by a [recent work](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens) from Redwood Research that shows that these 2 methods enhances latent reasoning in recent models. I will only be testing these ideas on just gpt2-xl(for cost reasons) to validate if there is any benefit in improving gold label accuracy(QA accuracy) and subsequently if any of the aforestated feature representations can explain the improvement/deficiency in accuracy.
-

# Reference
```
@misc{gottesman2024estimatingknowledge,
      title={Estimating Knowledge in Large Language Models Without Generating a Single Token},
      author={Daniela Gottesman and Mor Geva},
      year={2024},
      eprint={2406.12673},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12673},
}
```
