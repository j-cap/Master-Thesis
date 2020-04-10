# EQL - Extrapolation and Learning Equations

Author: Lampert, Martius
Tags: Source
Type: Paper
1---

## Introduction

in general: *quality of a model is* measured by its ability to generalize from training data to prev. unseen data from the same destribution. In regression, this boils down to interpolation if the training data is sufficiently dense, but it could happen that future data lies outside of the training domain. In such cases the regression model should continue to make good predictions. This is called *extrapolation generalization.* 

## Regression and Extrapolation

Consider a multivariate regression problem with a training set. The main interesst here lies on extrapolation. The general task is to learn a function $\psi: \R^n → \R^m$ that approximates the true functional relation as well as possible in the squared loss sense, i.e. achieives a minimal expected error $\mathbb E \left \| \psi(x) - \phi(x) \right \| ^2$.

If training and test data are sampled from the same distribution, this is an interpolation problem. In the extrapolation setting the training data is assumed to cover only a limited range of the data domain. 

## Learning a Network for Function Extrapolation

Here, a multi-layered feed-forward network with specially designed computational units is proposed. Each layer consists of a linear mapping followed by non-linear transformations. These transformations are either unitary (identity, sin, cos, sigmoid) or binary (multiplication) units. each This architecture is called *Equation Learner (EQL)*. 

### Discussion of the Architecture

The proposed architecture differes in two main aspects from a standard ANN: 

- the existance of multiplication units: first introduced as product-units [Durbin and Rumelhart, 1989] and Pi-Sigma-unit [Shin and Gosh, 1991]
- possibility of sine and cosine as nonlinearities

This architecture is a superclass of artificial neural networks (ANN), as it also includes the sigmoid nonlinearity. 

### Network Training

A Lasso-like objective (linear combination of $L_2$ loss and $L_1$ regularization, is used with stochastic gradient descent with mini-batches and Adam. The $L_1$ regularization encourages networks with sparse connections, but there's a side-effect: During the optimization, the weights hardly ever change their sign → might get stuck. Therefore, a hybrid optimizaion strategy was employed: At the beginning, no regularization is used, s.t. parameters can vary freely and reach reasonable starting points. Afterwards, regularization was switched on. Finally, for the last steps of the training, the $L_1$ regularization is again disabled, but the same $L_0$ norm of the weights is enforced. This is achieved by keeping all weights that are close to zero at zero.

### Model Selection for Extrapolation

Standard techniques for model selection as cross-validation or evaluation on a hold-out set will not be optimal for this purpose, since they rely on the interpolation quality. Therefore, Occams razor principle is used: *The simplest formula is most likely the right one*. The number of hidden units is used as a proxy for the complexity of the formula. This argumentation is only correct if the model explains the data well, i.e. has a low validation error ⇒ dual objective to minimize, which is solved by ranking instances w.r.t. validation error and sparisty and select the on with the smallest $L_2$ norm (in rank-space). 

### Related Work

- *Black box process* of identifying suitable real-valued functions from a hypotheses set (e.g. RKHS, GPR, SVR, ANN of suitable expressive power): the goal here is to find a prediction fcuntion that leads to a small expected error on future data, not necessarily to gain insights into the mechanism of how the output values derive from the input
- *System identification:* == learning a true, functional dependenace from observing a physical system. Typically, the functional form is known and only the parameters have to be identified.
- *Causal learning:* identify causal relations between multiple observables
- *Domain adaption and Covariate shift:* Extrapolation in the data domain implies that the data distribution at prediction time will differ from the data distribution at training time [[Ben-David et a., 2010](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf)]
- *Symbolic regression*: typically with evolutionary computation. The problem here is the exponential increase in computational complexity for large expressions and high-dim. systems.

## Experimental Evaluation

The ability of EQL to learn physically inspired models with good extrapolation quality is shown here in the following examples:

- Pendulum: good extrapolation
- Double pendulum kinematics: good extrapolation
- Robitic arms: good extrapolation
- Learning complex formula: good extrapolation
- X-Ray trainsition energies: good extrapolation
- Cart-pendulum system: poor extrapolation, because problem includes divisions, which are not modeled by this type of network