# Experimental evaluation of the 3-stages of Li2 framework
Here, I will be doing my utmost to conduct experiments that attempt to validate the existence of 3-stages of a neural network in the referenced [paper](https://arxiv.org/pdf/2509.21519) by Yuandong et al.

Li2 framework is principled framework for determining how neural nework models learn generalizable features in 3-stages. Using a 2-layer neural network, the framework proposed that in the first stage, the ouput layer fully fits the training data using random features leading to a memorized circuit/feature. Secondly, each neuron in the hidden layer then independently learn features. Finally, each neuron in the hidden layer then starts repelling each other by learning missing features from stage 2 thereby producing emergent and generalizable features.

It is important to note that Li2 framework is a gradient-based approach that provides an illumination into the training dynamics of a neural network model using toy models. However, the paper claims that the framework is applicable to deeper networks.

In the light of this, I will be doing thorough evaluations of the major claims in the paper using different kind datasets and models that were not tried in the paper in order to test the robustness or universality of the claims in the paper.

# Rough ideas I would be trying out
- [ ] Reproduce the claims of the 3-stage phenomena using modular arithmetic data as used in the paper.
- [ ] Use deeper networks on modular arithmetic data to validate same claims.
- [ ] Use alternative datasets outside that used in the paper to test the phenomena.
- [ ] Use a 1 or 2-layer attention model and test the existence of same training dynamics.

# Reference
```
@misc{yuandongtian2025feature-emergence,
      title={Provably scaling laws of feature emergence from learning dynamic of grokking},
      author={Yuandong Tian},
      year={2025},
      eprint={2509.21519},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/pdf/2509.21519},
}
```