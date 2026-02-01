# Reproducing the essential claims of the Li2 framework

## RQ-1
### 1.1 Reproducing Grokking in modular addition
- [ ] Generating training dynamics plot (training vs test accuracy and loss curves)
    - How best to distinguish stage 1, 2 and 3 using the accuracy and loss curves
    - Could it be that stage 3 starts around when test accuracy has been saturated
    - Would it make sense to measure `post-grokking` length(i.e in number of epochs). Maybe defined by number of epochs beyond 99% test accuracy

- [ ] What other metrics(weight or gradient norm) can be used to determine when the network stops memorizing and starts grokking?
- [ ] What data split(train-test ratio) or regularization is sufficient to achieve both memorization and grokking?
- [ ] What is the impact of initialization and output layer setup on grokking? 
    - Does using a large hidden weight initilization help grokking without weight decay?
    - How does hidden weight initialization affect time-to-grokking?
    - We repeat the same for top-layer weights.
    - How does the output layer converge to a ridge regression solution?

### 1.2 Layer-wise dynamics and feature emergence
- [ ] How to empirically detect transition from stage 1 to 2?
    - Track weight updates on each layer over training time.
    - Investigate metrics needed to learn the evolution. 
        - Could it be cosine distance between successive weight snaphots in each layer?
        - (or/ and) cosine distance between current weight and initial weight snapshot at beginning of training?
        - (or/ and) ratio of gradient norms between hidden layer and output layer?
        - (or/ and) other metrics used in the paper or some other works.
- [ ] How to empirically detect transition from stage 2 to 3?
    - Detect stage 3 via feature evolution after saturation of test loss.
    - Sustained test accuracy rise is for stage 2.
    - How best can weights and weights norm help to determine that learning continues after stage 2(test accuracy saturation, correct?).
    - Could $G_f$ be used to differentiate the 3 stages?
- [ ] What visualization methods can help determine that network has learnt independent and meaningful features
    - Check the hidden activation or weight vectors to see if each neuron has develop a distinct pattern(e.g fourier basis for modular addition)
    - Is it realistic to plot individual neuron activations across inputs?
    - Will performing PCA on hidden activations reveal if neurons have independent features, or is there a better way to reveal this?
    - Are hidden neurons learning redundant features? Are there interesting metrics to help investigate this?
- [ ] How to measure the structure of backpropated gradients of the features $G_f$ over time and its relation to feature emergence?
    - How realistic it is to measure $G_f$ over the course of training? Does $G_f$ become more structured(~ diagonal) or random?
    - How alignment of $G_f$ with target label could help show the outset of stage 2 and independent feature learning?
    - What other metrics beyond what's used in the paper could help corroborate outset of stage 1 and 2?

### 1.3 Scaling law and sample complexity
- [ ] Can we empirically validate $O(M \log M)$ sample complexity as predicted by __Theorem 4__ of the paper?
    - What experimental approach can be used to validate this?
- [ ] How are phase/stage transitions affected by group size M, training fraction and network width K?
    - For a fixed network width, how is grokking affected by group size and training fraction and vice versa?
    - How does network width affect grokking(sudden generalization)?
    - How does hyperparameter sweeps help over the 3 parameters help here?

### 1.4 Weight decay and regularization
- [ ] How does weight decay affect learning dynamics and generalization in grokking?
    - The paper shows that weight decay is essential for generalization, we perform experiment to prove this perhaps across 2-4 layer networks?
- [ ] What is the effect of using closed-form ridge regression for the output layer instead of gradient descent?
    - Paper shows that stage 1 performs ridge regression on fixed random features? How to show this empirically?
    - How does ridge regression of output layer compares to gradient descent? How to show this empirically?
    - Is grokking altered by using ridge regression? Is there a way to combine both approaches at different regime of training and how does the hybrid approach affect grokking or generalization?
- [ ] Does the backpropagated signal $G_f$ remain negligible without regularization?
    - With or without weight decay, measure $G_f$ throughout training.

### 1.5 Feature independence and structure validation
- [ ] Do individual neurons learn independent and specialized features? same as viz in section 1.2
    - Could plotting neuron activations or weights over many examples be useful and realistic?
    - How can the plots of the neurons help show that it utilizes different portions of the input space (or frequency components), indicating learned features are independent and not copies of one another?
    - Show that learned hidden features/activations correspond to known mathematical representations(e.g fourier basis for modular addition). Ideally, this is done after training. The existence of fourier basis in the weights or activations would verify the paper's claim that hidden units converge to an __irreducible representation__ of the group rather than an arbitrary solution.
    - Is there a difference between loss landscape between memorized solution and feature-based(generalized) solution?
        - Theoretical results show that stage 1 has sharp optimum and stage 2 has flatter optima with at least one zero Hessian eigenvalue. we need to verify this.
        - Conduct pertubation experiments or compute hessians at different training stages.
        - Add small noise to weights when model is memorizing vs generalizing to see which stage's solution is more robust(flat) vs sensitive (sharp)
        - Could loss landscape be used to distinguish stage 1, 2 and 3?
    - How best to track redundancy over time? Could effective rank on $F^TF$ be useful here or something else.

### 1.6 Optimizer effects and initialization ablation
- [ ] How does initializing the output layer to zero(vs. random initialization) influence feature emergence and grokking timing?
    - Ideally, we compare random vs zero initialization on the output layer and effect on time-to-grokking(i.e number of epichs before test loss starts to improve) and test loss.
- [ ] Effect of different optimizers on speed to grokking. 
    - How best to compare different optimizers(Adam, SGD and Muon) as they each have different parameters to tweak? 
    - Check how they impact time-to-grokking(stage 2 outset) and post-grokking refinement(stage 3).
    - How different optimizers affect feature diversity?
        - using cosine similarity between neurons weights to check how aligned or diverse they are? how useful is this?
        - using effective rank of $F^TF$
        - using kernel off-diagonal/diagonal ratio. how useful is this?
    - Using a suitable feature diversity approach, we can verify that there is more diversity in stage 3 compared to previous stages.
    - How the optimizers affect 

### 1.7 Deeper networks and attention-based models
- [ ] How does delayed generalization(grokking) happen in deeper networks(3, 5 and 10-layer network)?
    - Are the 3-stages observed using train-test loss curve?
    - Observe length of memorization phase and sharpness of transition to generalization as it relates to depth of network.
    - Is there a stage-wise learning dynamics from top-outer layer to the bottom layer as observed in a 2-layer network in the paper. We have to measure weight updates in every layer to see the one with the largest update over the course of training and in each phase.
- [ ] Use gradient-based Li2 framework to understand grokking in a toy 1-layer attention model.
    - Is weight decay necessary for grokking as showed in Li2 framework?
    - Try some of the insights gleaned in paper and see if they extend to a single-layer attention model.
    - Might also add 1-MLP and draw insights using gradient-based approach to inspect 3 stages: lazy learning independent learning and feature diversity over the course of training.




> Note: M = group size



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