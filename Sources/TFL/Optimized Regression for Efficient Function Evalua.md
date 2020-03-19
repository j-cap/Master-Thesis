# Optimized Regression for Efficient Function Evaluation

Author: Arora, Garcia, Gupta
Type: Paper
URL: https://ieeexplore.ieee.org/document/6203580

[TensorFlow Lattice](https://www.tensorflow.org/lattice/overview)

---

---

## 1. Introduction

Focus here is on problems in which one is given a training set of n sample input-output pairs. The goal is to produce a function that is both faitful to these samples and efficient to evaluate. 

The common two-step solution to this problem is:

1. Estimate an intermediate function from the given data
2. Evaluate the intermediate function on a regular lattice and store it. Given a new input x, the estimated function $f(x)$ is evaluated by interpolating the $2^d$ surrounding lattices nodes. 

This approach is suboptimal in terms of accuracy because the effect of interpolation is not considered when estimating the intermediate function. Therefore, *lattice regression* is proposed. 

This paper differes from preliminary work (e.g. [Lattice Regression](https://www.notion.so/chancellorjcap/Lattice-Regression-fefd9679d2774018ae8b5d50e4b1339b)) in that a new, simpler regularizer is introduced and related work is discussion in more depth. Also, new experimental results are shown.

## 2. Lattice Regression

For an introduction to lattice regression, please look at the original [Lattice Regression paper](https://www.notion.so/chancellorjcap/Lattice-Regression-fefd9679d2774018ae8b5d50e4b1339b) from 2009. 

### 2.B Lattice Regression Regularizer

A regularizer is added to ensure a unique solution and to promote *smootheness*. But how exactly should one measure the "smootheness" of a function that is to be interpolated from a lattice? This is highly application-dependent. Therefore, also the choice of the regularizer is application-dependent. In previous work, the graph Laplacian as first-order measure of smoothness was used as regularizer. Here a second-order difference regularizer called *graph Hessian* is introduced. 

**Graph Hessian**

To avoid penalizing the estimation of linear functions (as done by the graph Laplacian), we prefer to regularize by penalizing the second-order difference in each dimension of the lattice, summed over the d dimensions

$$\sum_{k=1}^d \sum_{at_h, a_i, a_j \ adjacent \ in \ \dim. k} \big( (b_h - b_i) - (b_i - b_j\big)^2 = b^T \mathbf K_H b$$

where $\mathbf K_H$ is a $m \times m$ matrix. $\mathbf K_H$ is not positive definite, so a touch of identity is added to build $\mathbf{\tilde K_H} = \mathbf K_H + 10^{-6} \mathbf I$. Using this regularizer, the problem has a closed-form solution (given in the paper). 

## 3. Related Regression Methods

Lattice regression belongs to tow major families: 

- Structural risk minimizations
- Splines

It is a structural risk minimization method since it selects a function that tradesoff minimization of an empirical risk term with a regularizer. 

### 3.A Relation to Splines

Common definition of spline is that a spline is a *piecewise polynomial function*. Lattice regression produces in fact piecewiese polynomial functions because linear interpolation applied to the vertices of any parallelotope produces a polynomial function. 

A more convenient representations of splines is in term of basis functions. In this, a spline is a linear combination of some piecewise-polynomial base functions $k : \R^d \times R^d → R$, repeated and centered at each knot (i.e. lattice node), such that

$$\tilde f(x) = \sum_{j=1}^m b_j k(x, a_j)$$

where $b_j$ is the weight given to the j-th basis function centered at the knot (lattice node) $a_j$. 

There is a lot of literature on splines, e.g. on linear b-splines, smoothing splines, etc. When the knots area fixed in a rectangular lattice, the solution to the smoothing spline objective is known as *tensor splines.* 

To summarize, the proposed lattice regression is a spline method that:

1. uses a fixed rectangular grid of knots independent of the training samples
2. uses a d-linear interpolation function
3. uses a discrete regularizer (e.g. graph Hessian)

### 3.B Comparison with Higher Order Basis Functions

The larger support of higher order basis functions (e.g. cubic splines) requires an far larger amount of lattice points to calculate an estimated function value (in case of cubic splines $4^d$ points are needed). Also, low-level speed ups from implementations (e.g. bit shifting) are also not possible for higher order basis functions. 

In general, the accuracy differences between different interpolation functions will depend on the true function being approximated, with smoother functions being better approximated with the higher order basis functions. 

## 4. Applications and Experiments

For a detailed info, please look in the paper.

## 5. Conclusion

Lattice values that minimize a regularized post-interpolation training erro can be determind in closed form. Also large performance gains over the state-of-the-art were shown for two applications (color management and omnidirectional super-resolution). 

---