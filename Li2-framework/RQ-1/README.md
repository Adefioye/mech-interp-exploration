# Reproducing the essential claims of the Li2 framework

## RQ-1.1
### Reproducing Grokking in modular addition
- [ ] Generating training dynamics plot (training vs test accuracy curves)
- [ ] What other metrics(weight or gradient norm) can be used to determine when the network stops memorizing and starts grokking?
- [ ] What data split(train-test ratio) or regularization is sufficient to achieve both memorization and grokking?
- [ ] What is the impact of initialization and output layer setup on grokking? 
    - Does using a large hidden weight initilization help grokking without weight decay?
    - How does hidden weight initialization affect time-to-grokking?
    - We repeat the same for top-layer weights.
    - How does the output layer converge to a ridge regression solution?

### Layer-wise dynamics and feature emergence
- [ ] How to empirically detect transition from stage 1 to 2?
    - Track weight updates on each layer over training time.
    - Investigate metrics needed to learn the evolution. 
        - Could it be cosine distance between successive weight snaphots in each layer?
        - (or/ and) cosine distance between current weight and initial weight snapshot at beginning of training?
        - (or/ and) ratio of gradient norms between hidden layer and output layer?
        - (or/ and) other metrics used in the paper or some other works.
- [ ] What visualization methods can help determine that network has learnt independent and meaningful features
    - Check the hidden activation or weight vectors to see if each neuron has develop a distinct pattern(e.g fourier basis for modular addition)
    - Is it realistic to plot individual neuron activations across inputs?
    - Will performing PCA on hidden activations reveal if neurons have independent features, or is there a better way to reveal this?
    - Are hidden neurons learning redundant features? Are there interesting metrics to help investigate this?
- [ ] How to measure the structure of backpropated gradients of the features $G_f$




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