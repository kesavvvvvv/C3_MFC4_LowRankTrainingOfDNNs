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

Ran simulations changing learning rate, lambda_ortho parameters
Below is the table of the recorded accuracies at different parameters

| Lambda_ortho ↓ \ LR → | 1e-1           | 1e-2                     | 1e-3   | 1e-4   | 1e-5   | 1e-6   |
| --------------------- | -------------- | ------------------------ | ------ | ------ | ------ | ------ |
| **1e-1**              | 9.8% (Overfit) | 93.03% (highly unstable) | 93.37% | 86.35% | 64.07% | 31.85% |
| **1e-2**              | 9.8% (Overfit) | 9.8% (Overfit)           | 93.41% | 87.14% | 64.26% | 31.82% |
| **1e-3**              | 9.8% (Overfit) | 9.8% (Overfit)           | 94.86% | 87.18% | 64.27% | 31.83% |
| **1e-4**              | 9.8% (Overfit) | 93.75%                   | 94.78% | 87.18% | 64.29% | 31.83% |
| **1e-5**              | 9.8% (Overfit) | 93.30%                   | 94.80% | 87.16% | 64.29% | 31.83% |
| **1e-6**              | 9.8% (Overfit) | 92.81%                   | 94.74% | 87.81% | 64.29% | 31.83% |


For standard DNN,
| Learning Rate | Accuracy |
| ------------- | -------- |
| **1e-1**      | 97.74%   |
| **1e-2**      | 97.10%   |
| **1e-3**      | 74.17%   |
| **1e-4**      | 11.35%   |
| **1e-5**      | 11.35%   |
| **1e-6**      | 8.48%    |



---

## 5. Observations

- Standard DNN training is robust only above a critical learning-rate threshold (as expected)
- Compared to standard DNNs, SVD-based training is significantly more sensitive to learning-rate selection and operates within a much narrower range
- Strong orthogonality regularization destabilizes training at high learning rates
- Very small learning rates causes underfitting in SVD-based models
- Optimal SVD performance remains below standard DNN accuracy

---

## 6. Advice / Future Work

1.
Generate a list of random number seeds by fixing the first seed.
MNIST digits, MNIST fashion, CIFAR 10.
Compare the results for how quick the loss function drops for SVD vs full weight matrix

2.
Combine SVD pruned and standard DNN training
Print the gradient (difference of loss function). Choose a threshold from here
Until the gradient falls below the threshold follow SVD pruning
After that switch over to standard DNN (full weight matrix) training.

3.
Matlab code for CNN : Phil KIM (give citation in results section)
For the same architecture do SVD pruned layers and Bi-PIL
Compare the results.

4.
Understanding the loss function,
remove each term and show the result, loss function graph

---

