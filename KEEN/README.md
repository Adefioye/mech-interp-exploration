# KEEN Paper
This paper attempts to estimate knowledge using hidden states of known entities in the wild.

# How to generate dataset for training
- [x] Generated completions for per-subject query/questionn using `gpt2-xl`.
- [ ] Generate gold scores for per-subject question.
- [ ] Generate internal/hidden representations.
- [ ] Train a linear/MLP probe on the hidden representations and then evaluate the correlation. What real-valued vectors are we correlating really? It is the correlation between gold scores and probes' predicted scores for per-subject question.
- [ ] Experiment with features beyond hidden representation. I will try attention outputs and fully-connected activations(Will need to fully understand what these 2 features are).

TODO:
- [ ] How to generate hidden representations at some layers for each per-subject query/question
- [ ] Should reuse the linear/MLP probe from paper code.


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
