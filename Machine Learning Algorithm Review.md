# Machine Learning Algorithm Review

### Model Evaluation and Selection

$\textbf{Train/Test Split}$

Choose a development and test set to reflect data you expect to get in the future. Generally, for small dataset 70% training and 30% testing, or 60% training, 30% development and 20% testing; for large dataset 98% training, 1% development and testing.

$\bullet$ Hold-out: split the original data set into training and testing to keep the same distribution, stratified sampling. Use multiple hold-out and generate training/testing set for experiments, then average the results. Generally, 2/3 to 4/5 as training set.

$\bullet$ Cross-validation:

K-fold: divide the original dataset into K groups, each time use K-1 as training set and 1 as testing set. Averaging the results to get the final model.

Leave-One-Out: assume m records in the dataset, each time use m-1 as training set and 1 as testing set (K = m in K-fold case). Accurate but computationally expensive when play with large dataset. 

$\bullet$ Bootstrap

Based on bootstrap sampling. Suppose original dataset $D$ has $m$ samples, every time randomly select with repetation from m samples. The selected samples forms training set $D'$, the dataset $D-D'$ serves for testing set. Based on $ \lim _{m\rightarrow \infty}(1-\frac{1}{m})^m = 1/e$, approximately 36.8% of data would not be sampled in the training set and would serve for testing the model.

Usually used when the original dataset is small, and can generate multiple different training set such that would benefit ensemble learning. However, it changes the distribution of the original dataset (some data is sampled multiple times such that has higher weight than the others), which introduce bias to the model.

$\textbf{Parameter Tuning}$

Original dataset should be divided into training, validation and testing set. Validation set is used for tunning and selecting the parameters.

$\bullet$ Grid Search

$\bullet$ Random Search

$\bullet$ Stochastic Optimization

$\textbf{Performance Measure / Cost Function}$

$\bullet$ Regression 

1. Mean squared error $E(f;D) = \frac{1}{m}\sum_{i=1}^{m}(f(x_i)-y_i)^2$
2. Mean absolute error $E(f;D) = \frac{1}{m}\sum_{i=1}^{m}|f(x_i)-y_i|$

What would be the differences when training the model based on MSE and MAE? 

Mean squared loss is generally used, reduce the differences between any of the observations and predictions. Absolute error penalizes big differences between $y_i$ and $X_i^T\beta$ to a less degree than the least squares loss, so that the estimator is less affected by the outliers. Use a combination of square and absolute loss as Huber loss.

$\bullet$ Classification

1. Error rate $E(f;D) = \frac{1}{m}\sum_{i=1}^{m}I(f(x_i)\neq y_i)$
2. Accuracy $acc(f;D) = 1-E(f;D)$

Sometimes, we care more about what's the percentage in output that is accurate, which cannot be simply explained by error rate and accuracy. 

3. Confusion matrix: precision and recall

   precision $P = \frac{TP}{TP + FP}$

   recall $R = \frac{TP}{TP + FN}$

4. Break-even point: the point where precision = recall

5. F1-measure $F1 = \frac{2\times P\times R}{P+R}$

In the application, difference weight should be put to precision and recall. For example, in recommand system, we care more about whether the things that users are interested in are recommended so that precision is more important.

6. $F-\beta$ Measure $F_{\beta} = \frac{(1+\beta^2)\times P \times R}{(\beta^2 \times P)+R}$, where $\beta$ measures the relative importance

7. Macro- and micro: corresponding to multiple confusion matrix (i.e. multi-class classification has multiple binary confusion matrix)

8. Receiver Operating Characteristic (ROC)

9. Cost Matrix: $c_{ij}$ refers to the cost that mis-classify class $i$ as $j$

10. Hypothesis testing: check the hypothesis of generalized error based on the distribution of testing error. 

    Binomial test

    t-test for Leave-One-Out: compute the confidence interval of maximum error rate

    Paired t-test: compare two classifiers

$\textbf{Bias-Variance Decomposition}$
$$
\begin{aligned}
\text{Expected Prediction} \bar f(x) &= E_D[f(x;D)]\\
\text{Variance } var(x) &= E_D[(f(x;D) - \bar f(x))^2]\\
\text{Error } \epsilon^2 &= E_D[(y_D-y)^2], \epsilon \sim N(0, \sigma^2)\\
\text{Bias }bias^2(x) &= (\bar f(x) - y)^2\\
&\text{Assume the expectation of error is zero}\\
E(f;D) &= E_D[(f(x;D) - y_D)^2]\\
&= E_D[(f(x;D) -\bar f(x) + \bar f(x) - y_D)^2]\\
&= E_D[(f(x;D)-\bar f(x))^2] + E_D[(\bar f(x)-y_D)^2] + \\&E_D[2(f(x;D)-\bar f(x))(\bar f(x)-y_D)]\\
&= E_D[(f(x;D)-\bar f(x))^2] + E_D[(\bar f(x)-y_D)^2]\\
&= E_D[(f(x;D)-\bar f(x))^2] + E_D[(\bar f(x)-y+y-y_D)^2]\\
&= E_D[(f(x;D)-\bar f(x))^2] + E_D[(\bar f(x)-y)^2] + E_D[(y_D - y)^2]\\
&= bias^2(x) + var(x) + \epsilon^2
\end{aligned}
$$
$\bullet$ Clustering

###Linear Models

#### Linear Regression

$y_i = \sum_{j=1}^p x_{ij}\beta_j + \epsilon_i, \epsilon_i \sim N(0, \sigma^2)$

Matrix form $y =X^T \beta$

Least Square $\min L(\beta) =  ||y-X^T\beta||^2 \Rightarrow \frac{\part L}{\part \beta} = 2(y-X^T\beta)(-X)= 0 \Rightarrow \beta = (X^TX)^{-1}X^Ty $

Maximum Likelihood gives the same result as Least Square $\max L(\beta) = \sum_{i=1}^n \log(p(y_i|x_i, \theta))=\sum_{i=1}^n \log(\frac{1}{\sqrt{2\pi \sigma^2}}\exp(\frac{-(y_i - x_i^T \theta)^2}{2\sigma^2}))$

$E(\beta) = E((X^TX)^{-1}Xy) = E((X^TX)^{-1}X(X^T\beta + \epsilon)) = E(\beta + (X^TX)^{-1}X^T\epsilon) = \beta$The estimator of linear regression ($\beta$) is unbiased estimator.

Considering the bias-variance trade off, penalize complex model. 

$\textbf{Ridge Regression}$

Loss function $L(\beta) = ||y-X^T\beta||^2 + \lambda ||\beta||^2$

$\hat \beta_j = (1+N\lambda)^{-1} \hat \beta_j^{OLS}$

Ridge regression shriks all coefficents by a uniform factor and does not set any coefficients to 0 such that cannot select variables.

$\textbf{Lasso Regression}$

Loss function (Lagrangian form) $L(\beta) = ||y-X^T\beta||^2 + \lambda |\beta|$ 

Subgradient method $\hat \beta_j = S_{N\lambda}(\hat \beta^{OLS}) = \hat \beta^{OLS}\max(0,1-\frac{N\lambda}{|\hat \beta ^{OLS}|})$

Translates values towards zero instead of setting smaller values to zero and leaving larger ones untouched, which can be used for variable selection.

#### Generalize Linear Model

$y = g^{-1}(w^Tx+b)$

Newton method to get the weight and intercept 

#### Logistic Regression

$y_i \in \{0,1\}, y_i \sim \text{Bernoulli}(p_i), \Pr(y_i = 1) = p_i$

logit function $\text{logit}(p_i) = \log{\frac{p_i}{1-p_i}}=x_i^T \beta$, then sigmoid function $p_i=\text{sigmoid}(x_i^T \beta) = \frac{1}{1+\exp(x_i^T \beta)}$

$p_0 = \Pr(y=0|x) = \frac{1}{1+\exp(-w^Tx + b)}, p_1 = Pr(y=1|x) = \frac{\exp(-w^Tx + b)}{1+\exp(-w^Tx + b)}$

Compute the weight by maximum likelihood 
$$
\begin{aligned}
likelihood(\beta) &= \prod_{i=1}^{m} [y_ip_1 + (1-y_i)p_0] \\
log-lik(\beta) &= \sum_{i=1}^m \log(y_ip_1 + (1-y_i)p_0)\\
&= \sum_{i=1}^m \log(y_i\frac{\exp(-\beta^Tx)-1}{1+\exp(-\beta^Tx)} + \frac{1}{1+\exp(-\beta^Tx)})\\
&= \sum_{i=1}^m \log(y_i (\exp(-\beta^Tx)-1)+1)+\log(1+\exp(-\beta^Tx))\\
&= \sum_{i=1}^m -y_i\beta^Tx +\log(1+\exp(-\beta^Tx))
\end{aligned}
$$
Use gradient descent or Newton method to solve.

In Newton Method, the update rule is 

$\beta^{t+1} = \beta^t - (\frac{\part^2l(\beta)}{\part \beta \part \beta})^{-1}\frac{\part l(\beta)}{\part \beta}$

#### Support Vector Machine

$\textbf{Linear Separable}$

Linear separable problem required no misclassification, which means all cases should be correctly classified. SVM finds a boundary that maximize the margin or the gap between data points.If a point is further away from the decision boundary, there ought to be greater confidence in classifying the point.

Function margin of a hyperplane $(w,b)$ with respect  to data point $x_i, y_i$ is $\gamma_i = y_i(w^Tx_i + b)$. For positive label $y_i > 0$, we want $w^T x_i+b >>0$; for negative label $y_i < 0$, we want $w^Tx_i + b << 0$. 

The distance between hyperplane $(w,b)$ and data point $x_i, y_i$ is $\gamma =\frac{|w^Tx_i + b|}{||w||}$. Find the hyperplan that maximize the margin. For positive examples $w^Tx_i + b \ge 1$, and for negative examples $w^Tx_i + b \le -1$, the equality exists at the points that closest to the hyperplan. Then consider the closet distance between positive and negative examples to the hyperplan, the objective is to maximize the distances $\frac{2}{||w||}$.  Consequently, the objective function is as following. Solve the problem through Lagragian multiplier and dual problem.
$$
\min \frac{1}{2} ||w||^2 \text{s.t. } y_i(w^Tx_i + b) \ge 1
$$
$\textbf{Not Linear Separable}$

Introduce 'soft margin', where mis-classification is allowed, but we have to limit the number of misclassification. For the misclassification cases, we have $y_i(w^Tx_i+b) < 1$ or $y_i(w^Tx_i+b)-1<0$, the objective function can be written as following. 
$$
\min \frac{1}{2}||w||^2 + C\sum_{i=1}^m l_{0/1}(y_i(w^Tx_i+b)-1)
$$
For correct classification $l_{0/1} = 0$, for misclassification $l_{0/1} = 1$. However this function (zero one loss) is not differentiable, choose other forms of losses (Hinge loss, Exponential loss, Logistic loss). Use coordinate descent to solve the problem.

Hinge loss $\text{hinge}(y_i, x_i) = \max(0, 1-y_i(w^T x_i + b))$: if classify correctly, $y_i$ and $w^Tx_i+b$ have the same sign, $\text{hinge}(y_i, x_i ) \in [0,1]$; if classify wrongly, $y_i$ and $w^Tx_i+b$ have different signs, $\text{hinge}(y_i, x_i ) > 1$.

$\textbf{SVM Regression}$



#### Linear Discriminate Analysis (LDA)

Project all data on to linear hyperplane, make the points in the same group as close as possible, make the points in different groups as far as possible. 

For Binary Classification, suppose $X_i, \mu_i, \Sigma_i$ are the data, mean and covariance matrix of class $i \in {0,1}$. Denote the target hyperplan is $w$. Consequently, the projection on the hyperplane of data, mean and covariance would be $w^TX$, $w^T\mu_i$ and $w^T\Sigma_iw$.

The distance between projected mean is 
$$
D = ||w^T\mu_0 - w^T \mu_1||^2_2 = w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw
$$
We want to have large between group distance and small within group distance, so the objective function is $\arg_w \max J = \frac{D}{w^T\Sigma_0w + w^T \Sigma_1 w} = \frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T(\Sigma_0 + \Sigma_1)w}$.

Optimization algorithm

Denote $S_b = (\mu_0 - \mu_1)(\mu_0 - \mu_1)^T, S_w = \Sigma_0 + \Sigma_1 \Rightarrow J = \frac{w^TS_bw}{w^TS_w w}$

Assume $w^TS_ww = 1$, the problem is $\min - w^TS_bw, s.t. w^TS_ww = 1$. Introduce Lagragian Multiplier, $S_bw = \lambda S_ww$, $w = S_w^{-1}(\mu_0 -\mu_1)$.

#### Multi-class Classification

One-hot Encoder vs. Labeling Encoder

Labeling encoder applies to ordinal data, for example the average of category 3 and 1 is 2 or high school is higher than elementrary school. One-hot encoder applies to non-ordinal data and when the values that are close to each other but the corresponding target values are not close in labeling encoder.

$\bullet$ One vs. One

$O(N^2)$: for each two groups, train a binary classifier. Totally $\frac{N(N-1)}{2}$ binary classifiers vote

$\bullet$ One vs. Rest

$O(N)$: for each class, current class is positive and the rest is negative.

$\bullet$ Many vs. Many

$\textbf{Softmax Regression}$

The probability that the prediction is class $j$ is $P(y_i=j|x) =  \frac{\exp(w^Tx_i+b)}{\sum_{k=1}^K \exp(w^Tx_k+b)}$

Likelihood $\prod_{i=1}^n p_i = \prod_{i = 1}^n \frac{\exp(w^Tx_i+b)}{\sum_{k=1}^K \exp(w^Tx_k+b)}$

Log-likelihood $(w^Tx_i+b) - \log(\sum_{k=1}^K \exp(w^Tx_k + b))$

#### Problems in Classification

$\textbf{Unbalanced Classification}$

$\bullet$ Adjust the objective function: 1. add weights to small labels / penalize correct classification on the large labels; 2. use confusion matrix; 3. use F-score  

$\bullet$ Sampling techniques: 1. undersample the large labels; 2. oversample the small labels.

### Kernel Machine/Non-Parametric Regression

### Classification and Regression Tree (CART)

$\textbf{Information Entropy}$ $Ent(D) = -\sum_{i=1}^{|y|}p_k \log p_k$, where $p_k$ is the proportion of $k^{th}$ class in current sample $D$. Entropy describes the purity of current set, the lower the entropy, the higher the purity of D.

$\textbf{Information Gain}$

​	$\bullet$ Categorical Variable

​		$Gain(D,a) = Ent(D)-\sum_{v=1}^V \frac{|D^v|}{|D|}Ent(D^v)$, where $a$ is an discrete feature with $V$ possible categories. Information Gain describes the gain in purity when classify using feature $a$. Consequently, each step, we can use Information Gain to select the feature where we grow a subtree.

​	$\bullet$ Continuous Variable

​		Splitting on continuous variable usually uses binary partition. Value $t$ on variable $a$ is selected to split the set into $D^-$ and $D^+$. If $t$ is the median of $a$,

$Gain(D,a)=Ent(D)-\sum_{\lambda \in {-,+}}\frac{|D_t^{\lambda}|}{|D|}Ent(D_t^{\lambda})$ 

$\textbf{Gain Ratio}$ Gain_ratio$(D,a) = \frac{Gain(D,a)}{IV(a)}$, $IV(a) = -\sum_{v=1}^V \frac{|D^v|}{|D|}\log(\frac{|D^v|}{|D|})$, where $IV(a)$ is intrinsic value, the number of possible values of $a$ the larger the intrinsic value is. If using information gain to select the feature, it tends to choose the feature that has more subclasses, which may lead to overfitting. On the contrast, Gain Ratio tends to choose features with less subclasses. Consequently, features with Gain Ratios greater than average Gain Ratios are selected first, and the feature with the largest Information Gain is selected as the feature growing sub-tree. 

$\textbf{Gini Index}$ Gini$(D) = \sum_{i=1}^{y}\sum_{k'\neq k}p_k p_{k'} = 1-\sum_{i=1}^{y} p_k^2$, which is used in CART to select the feature that next step split on. Suppose using feature $a$ to split, the Gini Index of feature $a$ is Gini_index$(D,a) = \sum_{v=1}^V \frac{|D^v|}{|D|}$Gini$(D^v)$. Select the feature with minimum Gini Index. 

$\textbf{Pruning}$  Regularization of trees to increase the generalization ability. 

​	$\bullet$ pre-pruning

​		When growing a tree, if a spliting on a node cannot increase the generalization ability of the model, then make this node a leaf node and do not grow a subtree on it anymore. It generally grows a shallower tree such that computational expense is reduced. However, its greedy property may increase the risk of underfitting. (Why greedy? Though current splitting on the evluated feature cannot increase the generalization ability, further splitting may increase the generalization ability.)

​	$\bullet$ post-pruning

​		Growing a tree on the training dataset, check all root nodes, if the generalization ability of the tree increases by substituting the subtree of the current evaluated root node by leaf node, then use a leaf node rather than a subtree. Greater computational expense but less risk of underfitting. 

$\textbf{Missing Data}$ Assign weight for the samples on current feature without missing data. $\tilde D$ is the subset of $D$ that does not have missing data on feature $a$. $\rho = \frac{\sum_{x\in \tilde D }w_x}{\sum_{x\in D}w_x}$

#### Ada Boosting



####Random Forest

####Extreme Gradient Boosting

###  Bayesian 

### Neural Network

$\textbf{Initialization}$

$\bullet$ Zero initialization: Symmetry-breaking problem: all the neurons of all layers performs the same calculation and gives the same output. Complexity would be the same as a single neuron.

$\bullet$ Random initialization: Solves Symmetry-breaking problem.

$\textbf{Activation Function}$ 

Why activation function? When nonlinear activation function is used, it can be proved that any function can be estimated by the combination of the neurons.

ReLu $Relu(x) = \max(0,x)$, not differentiable at 0. There is no backpropagation errors, speed is fast. Biological plausibility, sparse activation, better gradient propagation, efficient computation, scale-invariant.

Sigmoid $sigmoid(x) = \frac{1}{1+\exp(-x)}$, project input space to [0,1], but the derivative has a short range. When more layers (deeper) in the NN, the more info is compressed and lost at each layer.

tanh $tanh(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}$, project input space to [-1,1]

Leaky Relu $Leaky-Relu(x) = max(0.001x, x)$. There may be Dying ReLu problem: some neurons essentially die for all inputs and remain inactive and no gradients flow. 

(reference: https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e)

$\textbf{Hidden Layer}$ $S_l = w_lh_{l-1}$, linear combination of the last layer output, $l$ refers to $l^{th}$ layer. $w_l = (w_{l,1}, w_{l,2},..., w{l,k})$, $k$ refers to $k^{th}$ neuron.

$\textbf{Forward Propagation}$ $S_l = w_lh_{l-1}, h_l = f(S_l) = Relu(S_l)$
$$
\begin{aligned}
h_{l,k} &= f(s_{l,k}) = Relu(s_{l,k})\\
&= max(0, s_{l,k}) \\
&= I(s_{l,k} \ge 0) \times s_{l,k} = \delta_{l,k}s_{l,k}\\
\Rightarrow s_l &= w_lh_{l-1}\\
&= w_l\delta_{l-1}s_{l-1}\\
&...=\\
&= w_lw_{l-1}...w_1\delta_{l-1}\delta_{l-2}...\delta_1 x
\end{aligned}
$$
$\textbf{Optimization Algorithms}$

$\bullet$ Gradient Descent, Mini-Batch Gradient Descent, Stochastic Gradient Descent

$w_l = w_l - \alpha \frac{\part L}{\part w_l}, b_l= b_l - \alpha \frac{\part L}{\part b_l} $

1. Batch Gradient Descent: use all training set in a single epoch 
2. Mini-Batch Gradient Descent: use subset of all training set in a single epoch (fastest learning)
3. Stochastic Gradient Descent: use a single training example in a single epoch

$\bullet$ Gradient Descent with Momentum

$w_l = w_l - \alpha v_{dw},  b_l  = b_l - \alpha v_{db}$

$v_{dw} = \beta v_{dw}+ (1-\beta) dw, v_{db} = \beta v_{db} + (1-\beta) db$

Exponentially weighted average 

$\bullet$ RMSprop

$w_l = w_l - \alpha \frac{dw}{\sqrt{s_{dw}+\epsilon}},  b_l  = b_l - \alpha \frac{db}{\sqrt{s_{db}+\epsilon}}$

$s_{dw} = \beta_2 s_{dw}+ (1-\beta_2) dw^2, s_{db} = \beta_2 s_{db} + (1-\beta_2) db^2$

$\bullet$ Adam

$w_l = w_l - \alpha \frac{v_{dw}^{corrected}}{\sqrt{s_{dw}^{corrected}+\epsilon}},  b_l  = b_l - \alpha \frac{v_{db}^{corrected}}{\sqrt{s_{db}^{corrected}+\epsilon}}$

$v_{dw} = \beta_1 v_{dw}+ (1-\beta) dw, v_{db} = \beta_1 v_{db} + (1-\beta) db$

$v_{dw}^{corrected} = v_{dw}/(1-\beta_1), v_{db}^{corrected} = v_{db}/(1-\beta_1)$

$s_{dw} = \beta_2 s_{dw}+ (1-\beta_2) dw^2, s_{db} = \beta_2 s_{db} + (1-\beta_2) db^2$

$s_{dw}^{corrected} = s_{dw}/(1-\beta_2), s_{db}^{corrected} = s_{db}/(1-\beta_2)$

$\textbf{Backward Propagation}$

$\hat y = s_l, L(\theta) = \sum_{i=1}^nL(y_i,s_{l,i})$, where $L(y_i, s_{l,i})$ can be either loss functions for both classification and regression. 

e.g. Mean Square Loss $L(y, s) = \frac{1}{2}|y-s|^2, \frac{\part L}{\part s} = -(y-s)$

For layer $l$ and $l-1$
$$
\begin{aligned}
\frac{\part h_l}{\part h_{l-1}^T} &= \frac{\part f(s_l)}{\part h_{l-1}^T}\\
&= \frac{\part f(s_l)}{\part s_l} \frac{\part s_l}{\part h_{l-1}^T}\\
&= f'(s_l) \frac{\part w_l h_{l-1}^T}{\part h_{l-1}^T}\\
&= f'(s_l)w_l\\
\text{similarly, }\frac{\part h_l}{\part w_{l}^T} &= f'(s_l)h_{l-1}^T
\end{aligned}
$$
$\frac{\part L}{\part h_{l-1}^T} = \frac{\part L}{\part h_l^T} \delta_l w_l, \frac{\part L}{\part w_l} = \delta_l \frac{\part L}{\part h_l}h_{l-1}^T$

Gradient descent parameter update $w _{l,t+1} = w _{l,t}-\eta \frac{\part L}{\part w_{l,t}}$

Gradient descent with momentum $v_{t+1} = \gamma v_t + \eta_t \frac{\part L}{\part w_{l,t}}, w_{l,t+1} = w_{l,t} - v_{t+1}$

Use losses of each mini-batch to do backward propagation.

$\textbf{Dropout}$

Dropout is a regularization, where each layer is assigned a keep probability. A mask matrix is generated by comparing the randomly generated number with keep-probability. Dropout is performed on forward and backward propagation. After one round of calculation, the weights of kept neurons are updated.

```python
# For example, illustrate with the 3rd layer, keep_probability = 0.8
keep_prob = 0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
z4 = np.multiply(w4, a3) + b4 # The size of a3 is reduced by 20%
# When doing prediction, no dropout is performed. 
```

Intuition: cannot rely on all features, so have to spread out weights. Dropout would reduce the dependences between neurons. Dropout can be regarding as taking the average of subsets of neurons in one layer such that better generalization can be acquired. Rescale the neuron by multiplying $p$, which is the keep-probability.

For a network with $n$ neurons, there are $2^n$ possible sub-networks. Dropout can randomly sample over all $2^n$ possibilities, then can be viewed as a way to learn ensemble of $2^n$ models.

(Reference: Improving neural networks by preventing co-adaptation of feature detectors)

$\textbf{Vanishing / Exploding Gradients}$

In backward propagation 
$$
\begin{aligned}
\frac{\part L}{\part h_{l-1}^T} &= \frac{\part L}{\part h_l^T} \delta_l w_l\\ 
\frac{\part L}{\part w_l} &= \delta_l \frac{\part L}{\part h_l}h_{l-1}^T\\
\Rightarrow \frac{\part L}{\part w_l} &= \delta_l \delta_{l+1}...\delta_L w_{l+1} w_{l+2}...w_L h_{l-1}^T
\end{aligned}
$$
$\bullet$ Vanishing Gradients 

When performing backward propagation, the earlier leayers would have much lower gradients. If using sigmoid function as activation function and initialize weights using Gaussian distribution, then $w_i\delta_i \le \frac{1}{4}$ so that the earlier the layers, the lower the gradient of weight.

$\bullet$ Exploding Gradients

Similarly to vanishing gradients, if initialize the weights with very high magnitude, the earlier the layers the larger the gradients of weight. 

Solutions: 1. Use ReLu activation function; 2. Perform Batch Normalization; 3. Use Residual Blocks in very deep Neural Networks; 4. Xavier initialization (with tanh activation function), initialize weights at layer $l$ multiplying by a factor of $\sqrt{\frac{1}{n^{l-1}}}$ or $\sqrt{\frac{2}{n^{l-1}+ n^l}}$, where the initial weights neither too much bigger than 1 nor too much less than 1;

$\textbf{Batch Normalization}$

In forward propagation 

$\mu = \frac{1}{m} \sum_i s_i, \sigma^2 = \frac{1}{m} \sum_i (s_i - \mu)^2$

$s_{i,norm} = (s_i -\mu)/\sqrt{\sigma^2 + \epsilon}, \tilde s_i = \gamma s_{i,norm} + \beta$

$h_{i+1} = \sigma(\tilde s_i)$

Same backward propagation procedure 

Intuition: solving "covariate shift" problem, where positive and negative samples shifting to different distributions.

### Recommendation System

2020-01-11 Webinar

$\bullet$ Content Based Filter

User-item interactions matrix + user features + item features

Pros: cold start problem solved

Cons: complex structure (information collection, feature engineering)

$\bullet$ Collaborateive Filtering

User-item interactions matrix: row - users; column - product

Values: Boolean - click or not, endorsed or not...; Categorical - review, star...

1. Memory based filtering

   Define a model for user-item interactions where users and items representations have to be learned from interactions matrix

   User-User method (very sensitive to the interaction matrix): extract a row vector from the interaction matrix $\rightarrow$ search the K nearest neighbors of this user in the matrix $\rightarrow$ recommend the most popular items among the neighbors

   Item-Item method (large variance): identify the prefered item of user we want to recommend for and extract the column $\rightarrow$ find the K nearest neighbors of this item in the matrix $\rightarrow$  recomment the item to users who have high review on other items

2. Model based filtering

   Define no mobel for user-item interactions and rely on similarities between users or items in terms of observed iteractions 

   Matrix factorisation: interaction matrix = user matrix * item matrix + reconstruction error matrix 

Pros: don't have to record the user information and product information 

Cons: there is no record for new users/cold start problem (solution: random strategy; maximum expectation strategy; exploitary strategy)

$\bullet$ Hybrid Methods

Bias vs. Variance

High variance (overfitting): recommend well to single user, but not good for multiple users (personalization)

High bias (underfitting): recommend well to multiple users, but not good for single user

### Computer Vision

#### Image Classification 

$\textbf{Convolutional Layer}$ 

$\bullet$ Padding (p)

Maintain the same size for input image and convolutional image since convolutional operator shrinkage the image size. Avoid the loss of information.

$\bullet$ Convolution Operator

$(x*k)_{ij} = \sum_{pq}x_{i+p, j+q}k_{p,q}$

Fully connected layers have too many parameters and have very poor performance on computer vision tasks. Intuition: only connected to a subregion of the input layer to detect objects (edge, corner, color...). 

Parameter sharing: a feature detector that's useful in one part of the image is probably useful in another part of image.

Sparsity of connections: in each layer, each output value depends only on a small number of inputs.

$\bullet$ Kernel/Filter

Kernel is a detector, to detect whether the input follows the features encoded in the kernel (large $\rightarrow$ similar, small $\rightarrow$ dissimilar)

$\bullet$ Chennels 

$\bullet$ Stride (s)

For a $n \times n$ image, after convolutional operator by $f \times f$ filter with $p$ paddings and $s$ stride, the output size is $\lfloor{\frac{n+2p-f}{s}+1}\rfloor \times \lfloor{\frac{n+2p-f}{s}+1}\rfloor$. For a $n \times n \times n_c$ image, after convolutional operator by $f\times f \times n_c$ filter, the output size is $n-f+1 \times n-f+1 \times n_c'$, where $n_c'$ is the number of filters.

$\textbf{Pooling Layer}$

Insert a pooling layer in-between successive convolutional layers to reduce the size of representation and down-sampling. By pooling, we gain robustness to the exact spatial location of features.

Max pooling

Average pooling

$\textbf{Fully Connected Layer}$



$\textbf{Data Augmentation}$

Rotation, shift, rescaling, flipping, noise... 

First sample a batch, then for some of the input, randomly do augmentation.

$\textbf{Residual Networks}$

Gradient vanish problem: input information cannot be passed to very deep layers due to zero weight.
$$
\begin{aligned}
s_{l+1} &= w_{l+1} h_l + b_{l+1}\\
h_{l+1} &= g(s_{l+1})\\
s_{l+2} &= w_{l+2} h_{l+1} + b_{l+2}\\
h_{l+2} &= g(s_{l+2} + h_l) = g( w_{l+2} h_{l+1} + b_{l+2}+h_l)
\end{aligned}
$$
In very deep network or applying weight decay, the weight item $w_{l+2}$ might be zero, in such case $h_{l+2} = g(h_l) = h_l$, then the performance of the network doesn't get hurt and the information of $h_l$ can be passed to deeper layers.

$\textbf{1 $\times$ 1 convolution}$

Network in Network. Convolution by a $1\times 1$ filter, look at element-wise production and sum over channels. Decrease or increase the number of channels; adds non-linearity by activation function to make deeper and more complicated network; aggregate information among chennals

#### Classification with Localization

Bounding box: center coordinate ($b_x, b_y$), height ($b_h$), width ($b_w$) as output vector.

Target label $y$: 

$p_c$ - whether there is object in the picture or probability of whether there is an object

$(b_x,b_y, b_h, b_w)$ - continous, bounding box location and size

(c_1,...,c_i,...,c_n) - one-hot encoding of objective class

Loss function (squared error):

$L(\hat y, y) = \begin{cases} (\hat p_c - p_c)^2 & p_c = 0\\ \sum_{i = 1}^n (\hat c_i - c_i)^2 + (\hat b_x - b_x)^2 + (\hat b_y - b_y)^2 + (\hat b_h - b_h)^2 + (\hat b_w - b_w)^2 & p_c = 1\end{cases}$

#### Detection

Landmark Detection (coordinates): guesture, emotion, pose

$\textbf{Sliding Windows Detection}$

Preparation: a trained CNN to identify whether there is a car or not.

Select a sliding window size and use CNN to identify whether there is a car in the sliding window. Then shift sliding window horizontally and vertically in the whole picture to cover all of the image. After that, try to use larger sizes of sliding windows. Computational cost is high. Use simplier classifiers. 

Turn the last couple of fully connected layers into convolutional layers, then if perform the same convolutional operations as the existing CNN on a larger size of picture, the output would be running a sliding window over the original picture. However, such algorithms cannot have accurate bounding boxes. 

$\textbf{Bounding Box Prediction}$

-Yolo Algorithm

1. Divide the image into $n \times n$ grid cells.
2. For each grid cell, have a label vector $y$ contains $p_c, (b_x,b_y, b_h, b_w), (c_1,...,c_i,...,c_n)$.
3. Each object in training image is assigned to grid cell that contains that object's midpoint.
4. The target output is $n\times n \times size(y)$ dimension, since for each of $n \times n$ grid cells, there is an $y$ vector. 

-Intersection over Union Function

$\text{IoU} = \frac{\text{size of intersection of bounding box}}{\text{size of union of bounding box}}$, the predict bounding box is correct if $\text{IoU}\ge 0.5$ with ground truth bounding box. More generally, IoU is a measure of the overlap between two bounding boxes.

-Non-Max Suppression

Detect each object only once. When having multiple detections, have the probability of detection as $p_c$, select the one box with the highest probability.

1. Discard all boxes with low $p_c$
2. Pick the box with the largest $p_c$ output that as a prediction
3. Discard any remaining box with IoU $\ge$ 0.5 with the box output in the previous step.

-Anchor Boxes

Pre-defined anchor boxes, now the target variable $y$ contains the information of multiple anchor boxes. Each object in training image is assigned to grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU. Solve the problem with two objects in one grid cell; specialize the algorithm to objects with different shapes.

-Yolo Detection Algorithm

1. Divide the image into $n \times n$ grid cells.
2. For each grid cell, have a label vector $y$ contains $[p_c, (b_x,b_y, b_h, b_w), (c_1,...,c_i,...,c_n),p_c, (b_x,b_y, b_h, b_w), (c_1,...,c_i,...,c_n)] $ for 2 bounding boxes (for example), so the output size would be $n\times n \times \text{num}_{\text{anchor_box}}\times (5 + \text{num}_{\text{classes}})$
3.  Each object in training image is assigned to grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU. 
4. Outputting the non-max supressed results 

#### Face Recognition (One-shot Learning)

Learning from one example to recognize the person again. May only have one training set in the database. Rather than learning the image directly, learning a similarity function describe the degree of difference between images. 

Siamese network: use the output of fully connected layer as the encoding of a picture, compare the encodings between two different pictures to judge whether they are the same person or not. Learning the encoding such that the differences for the same person are small and for different person are large.

Triplet Loss: always look at three input images, one anchor, one positive and negative. Choose triplets that're hard to train on. $\alpha$ is margin, to prevent the neural network classifying everthing to be 0. Training set has multiple images for a single person, and choose the 'hard' examples to train the network.
$$
||f(A) - f(P)||^2 \le ||f(A) - f(N) ||^2\\
||f(A) - f(P)||^2 - ||f(A) - f(N) ||^2 + \alpha \le 0\\
L(A,P,N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N) ||^2 + \alpha, 0)\\
J = \sum_{i = 1}^m L(A^{i}, P^{i}, N^{i})
$$

####Face Recognition (Binary Classification)

Intuition: pass two images to a CNN to compute the encoding of the iamge, compare the euclidean distance/ $\chi^2$ similarity between the encodings to see whether they are the same image.

$\hat y = \sigma(\sum_{k =1} ^ n w_k|f(x^{i}_k)-f(x^{j}_k)|+b)$

#### Neural Style Transfer

Objective: generate image $G$ with similar content as content image $C$ and similar style as style image $S$.

Cost Function $J(G) = \alpha J_{content}(C,G) + \beta J_{style} (S,G)$

Content Cost Function $J_{content} (C,G) = \frac{1}{2} ||a^{[l](C)} - a^{[l](G)}||^2$ describes the similarity between the activation of layer $l$ for generated and content image. $l$ should not either too shallow or too deep. Shallow layers capture local features (edges, borders...) and deep layers capture global features.

Style Cost Function $J_{style} = \sum_{l}\frac{1}{(2n_H^l n_w^l n_c^l)^2}\sum_k \sum_{k'}||G_{kk'}^{[l](S)} - G_{kk'}^{[l](G)}||^2, G_{kk'} = \sum_{i}\sum_j a_{ijk} a_{ijk'}$, where $a_{ijk}$ is the activation at $(i,j,k)$. The style cost function describes the differences in the correlation matrix between style and generated image. The correlation matrix describes the correlation between activations at different locations between two layers, which furtherly describe what pixels appear at the same time or don't appear at the same time. The style cost function should take all low and high level features into consideration, so it sums over all layers.

### Sequence Model (Natural Language Processing)

####Recurrent Neural Network (Language Modeling)

$$
a^t = b+ Wh^{t-1} + Ux^t\\
h^t = \tanh(a^t)\\
\hat y ^t = softmax(c+Vh^t)\\
L^t(\hat y^t, y^t) = - y^t \log(\hat y^t)\\
L = \sum_t L^t(\hat y^t, y^t)
$$

When taking derivative with respect to $W$, it's depends on all components before step $t$, so this is back propagation through time.

$\textbf{Sampling Novel Sequence}$

Sample from dictionary for the word at current step based on softmax probability, then use the sampled word at current step as the input of the next step, repeat the process until get End of Sentence or exceed the length of a sentence. 

####Gated Recurrent Unit

Solve the vanishing gradient problem, at the same time, introduce dependency for early and late words.
$$
\tilde c^{t} = \tanh(W_c[\Gamma_r \times c^{t-1}, x^{t}]+ b_c)\\
\Gamma_u = \sigma(W_u [c^{t-1}, x^{t}] + b_u)\\
\Gamma_r = \sigma(W_r[c^{t-1}, x^{t}] + b_r)\\
c^{t} = \Gamma_u \tilde c^{t} + (1-\Gamma_u) c^{t-1}\\
a^{t} = c^{t}
$$

####Long Short Term Memory (LSTM)

$$
\tilde c^{t} = \tanh(W_c[ a^{t-1}, x^{t}]+ b_c)\\
\Gamma_u = \sigma(W_u [a^{t-1}, x^{t}] + b_u) \quad \text{update gate}\\
\Gamma_f = \sigma(W_f [a^{t-1}, x^t] + b_f) \quad \text{forget gate}\\
\Gamma_o = \sigma(W_o[a^{t-1}, x^{t}] + b_o) \quad \text{output gate}\\
c^{t} = \Gamma_u \tilde c^{t} + \Gamma_f c^{t-1}\\
a^{t} = \Gamma_o c^{t}
$$

####Word Embeddings

Word anologies cannot be studied by one-hot encoding, since the inner product of any two different one-hot encoding vectors is zero. Use featurized representation for word embedding. Analogy reasoning. Need to learning embedding matrix $E$. 

$\textbf{Word2Vec}$

Suppose Vocab size = 10,000k 

Objective: supervised learning the mapping from context word $c$ to target word $t$; the target word can be the word in the sentence skip several steps.

Model $o_c \rightarrow E \rightarrow e_c = Eo_c \rightarrow \text{softmax} \rightarrow \hat y$

Softmax layer $p(t|c) = \frac{\exp(\theta_t^T e_c)}{\sum_{j = 1}^{10000000} exp(\theta_j^T e_j)}$, where $\theta_t$ is the parameter associated with output target $t$.

Loss function, cross-entropy loss $L(\hat y, y) = - \sum_{i = 1}^{10000000} y_i \log \hat y_i$

Problems: 1. Computationally expensive when computing the cross entropy loss (solution: hierachical softmax classifier) 2. How to sample context word $c$? Uniform distribution.

$\textbf{Negative Sampling}$

Given context word pair, classify whether a valid word pair.

Model $P(y = 1 | c,t) = \sigma(\theta_t^T e_c)$, train 1 positive example and k randomly sampled examples. 

$\textbf{GloVe (Global Vectors for Word Representation)}$

$x_{ij}$ is the number of times $i$ appears in context of $j$, $x_{ij} = x_{ji}$.

Model $\min \sum_{i=1}^{10000000}\sum_{j=1}^{10000000}f(x_{ij})(\theta_i^T e_j + b_i + b_j' - \log(x_{ij}))^2$

Cannot garuentee the dimension of the embeding matrix is interpretable. 

$\textbf{Sentiment Classification}$

Input: sentences Output: sentiment 

Many-to-One Model: $o_c \rightarrow E \rightarrow e_c = Eo_c \rightarrow \text{RNN} \rightarrow \text{softmax} \rightarrow \hat y$

$\textbf{Attention Model}$

Encoder and decoder structure in machine translate, but it doesn't work well in the long sentences. Attention model focus on part of sentence at one time. Compute attention weights.



