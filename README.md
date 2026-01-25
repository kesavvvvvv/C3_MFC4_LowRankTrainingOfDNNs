# Low Rank Training of Deep Neural Networks (DNNs)

---

## 1. Team Members

| Name | Roll Number |
|---|---|
| Nakka Saampotth Maddileti | CB.SC.U4AIE24233 |
| Nimmagadda Kesav Satya Sai | CB.SC.U4AIE24236 |
| Gnana Vikas Sai Pabbathi | CB.SC.U4AIE24238 |
| Vemula Poorna Chandra | CB.SC.U4AIE24258 |

---

## 2. Base / Reference Paper(s)

**Low-Rank Training of Deep Neural Networks**  
Paper link: https://arxiv.org/abs/2004.09031

This paper proposes training neural networks directly in a low-rank parameterized form using singular value decomposition (SVD), enabling model compression and reduced computational cost during training itself.

---

## 3. Project Outline

Modern deep neural networks are computationally expensive and memory-intensive, making deployment on edge and resource-constrained devices challenging. Although trained weight matrices often exhibit low-rank structure, conventional training does not explicitly control or exploit this property.

In this project, we implement low-rank training of neural networks using SVD-based parameterization, where each fully connected layer is represented directly by its singular vectors and singular values instead of full weight matrices. The model is trained on these SVD components without performing repeated decompositions during training, while enforcing orthogonality constraints on the singular vectors and sparsity constraints on the singular values to preserve SVD properties and encourage a low effective rank.

Experiments are conducted on the MNIST dataset using a standard DNN architecture and its SVD-modified counterpart. The performance is evaluated in terms of accuracy, loss convergence, and parameter reduction, demonstrating the trade-off between compression and predictive performance.

---

## 4. Updates

*(To be updated)*

---

## 5. Challenges / Issues Faced

- Maintaining orthogonality of singular vectors during training required careful regularization.
- Balancing sparsity and accuracy was challenging, as aggressive rank reduction led to performance degradation.
- Training stability was sensitive to the choice of regularization weights in the loss function.
- Ensuring convergence of the SVD-parameterized model took longer compared to standard DNN training.
- Hyperparameter tuning for loss components significantly impacted model behavior.

---

## 6. Future Plans

- Extend the current approach to Convolutional Neural Networks (CNNs).
- Experiment with different datasets and analyze convergence behavior across varying data distributions.
- Modify and compare different loss function combinations to improve training stability.
- Perform convergence analysis under varying sparsity and orthogonality constraints.
- Aim to achieve more stable convergence while preserving low-rank structure.
- Focus on improving overall accuracy while maintaining reduced parameter count.
