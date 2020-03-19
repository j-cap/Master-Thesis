# EQL+ - Learning Equations for Extrapolation and Control

[martius-lab/EQL](https://github.com/martius-lab/EQL)

---

---

## Introduction

In machine learning, models are judged by their ability to predict correctly for unseen data. In natural sciences, one searches for interpretable models that provie a deeper understanding of the system of interest. One tries to identify the true underlying functional relationship behind the data. 

In previous work ([EQL - Extrapolation and Learning Equations](https://www.notion.so/chancellorjcap/EQL-Extrapolation-and-Learning-Equations-f64adca30a324bd1ab298cbdc2b6f53a)), two significatn shortcommings are present: 

- EQL is not able to represent divisions
- The model selection procedure is unreliable in identifying the true functional relation out of multiple candidates

These shortcommings are overcome with the new, improved network EQL+

## Identifying equation with a Network

see [EQL - Extrapolation and Learning Equations](https://www.notion.so/chancellorjcap/EQL-Extrapolation-and-Learning-Equations-f64adca30a324bd1ab298cbdc2b6f53a)

### Introducing Division Units

Introducting division is a nontrivial step, because any division $\frac{a}{b}$ creates a pole at $b → 0$ with an abrupt change in convexity and diverging function values and derivatives. To overcome the divergence problem, notice one important aspect of natural quantities: they do not diverge. This implies that a single branch of the hyperboly $\frac{1}{b}$ with $b > 0$ suffices as a basis function. 

To prevent problems during optimization, a new curriculum approach for optimization is introduced, progressing from a strongly regularized version of division to the unregularized one. 

**Regularized Division**

The last layer (only here the division units are used) of EQL+ is 

$$y^{(L)} = \Big ( h_1^\theta(z_1^{(L)}, z_2^{(L)} ), ...,h_m^\theta(z_{2m}^{(L)} , z_{2m+1}^{(L)} ) \Big) $$

where $h^\theta (a,b)$ is the division-activation funtion given by

$$h^\theta (a,b) := \begin{cases} \frac{a}{b} \quad if \ b > 0 \\ 0 \quad otherwise \end{cases}$$

where $\theta \ge 0$ is a threshold. 

**Penalty Term**

To steer the network awa from negative values of the denominator, a cost term is added to the objective that penalizes "forbidden" inputs to each division unit. 

$$p^\theta(b) := max(\theta - b, 0)$$

where $\theta$ is the same threshold as before. 

**Penalty Epchs**

Used to prevent that output values on future data (extrapolation region) have a very different magnitude that the observed inputs. to enforce this, penalty epochs are injected at regular intervals (every 50 epochs) into the training process. 

During a penalty epoch N randomly sampled input data points in the expected test range (including extrapolation region) without labels are sampled and the network is trained using the cost $\mathcal L^{Penalty} = P^\theta + P^{bound}$, where the latter is given by:
 

$$P^{bound} := \sum_{i=1}^N \sum_{j=1}^n max(y_j^{(L)}(x_i) - B, 0) + max(-y_j^{(L)}(x_i) - B, 0)$$

The value B reflects the maximal desired output value. 

The treshold $\theta$ play the rolfe of avoiding overly large gradients. However, ultimately we want to learn the precise quation so we introducce a curriculum diring training in which regularization is reduced continously. $\theta$ is decreased with epoch t as $\theta(t) = \frac{1}{\sqrt{t + 1}}$ 

### Model Selection for Extrapolation

**Without extrapolation data**

The selection process is based on the validation and the sparsity of the instance. The difference to the schemce in [EQL](https://www.notion.so/chancellorjcap/EQL-Extrapolation-and-Learning-Equations-f64adca30a324bd1ab298cbdc2b6f53a) is that validation error and sparsity of the network are normalized to $[0, 1]$ w.r.t. over all instances (trained networks). The criterion for the best model is:

$$\arg\min_{\psi} [ \alpha \tilde \nu^{int}(\psi)^2 + \beta \tilde s(\psi)^2]$$

where $\psi$ stands for an instance (trained network), $\tilde \nu^{int}(\Psi)^2$ is the validation error and $\tilde s(\psi)^2$ is the sparisty of the network $\psi$. $\alpha$ and $\beta$ are empirically determinded (here both equal to 0.5). This is called $\mathbf{V^{int}-S}$-selection method.

**With few extrapolation data**

Here, an additional extrapolation-validation dataset is formed. The error on this dataset is then included in the model selection scheme. Again, the different erros and the sparisty are normalized over all instances. The criterion for the best model is:

$$\arg\min_{\psi} [ \alpha \tilde \nu^{int}(\psi)^2 + \beta \tilde s(\psi)^2 + \gamma \tilde \nu^{ex}(\psi)^2]$$

A grid search is done to determine the weighting factors $\alpha,\ \beta \ and \ \gamma$. This method is then called $\mathbf{V^{int\&ex}}$, because the grid search shows that the sparsity term loses its importance. 

Because of the strong non-convexity of the problem, the optimization process may get stuck in a local minimum. We use 10 independet runs with random initializaion conditsion. 

## Relation to prior Work

see [EQL](https://www.notion.so/chancellorjcap/EQL-Extrapolation-and-Learning-Equations-f64adca30a324bd1ab298cbdc2b6f53a)

## Experimental Evalution

Experiments are done on:

- Formulas with divisions
- Complex formulas
- Random expressions
- Cart-pendulum system

## Control using Learned Dynamics

Here, the effectiveness of the equation learning for robot controll from EQL+ is shown.