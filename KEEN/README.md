# KEEN Paper
This paper attempts to estimate knowledge using hidden states of known entities in the wild.

# How to generate dataset for training
[x] Generated completions for per-subject query/questionn using `gpt2-xl`.
[ ] Generate internal/hidden representations(residual stream vectors at some specific layers)
[ ] Train a linear/MLP probe on the hidden representations and then evaluate the correlation. What real-valued vectors are we correlating really?

TODO:
- How to generate hidden representations at some layers for each per-subject query/question
- Will reuse the linear/MLP probe from paper code.


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
