# Lecture 1: A Crash Course on Deep Learning

## Elements of Machine Learning

First, we start from three machine learning keypoints as below:

- Model: a specific format or formula, such as $y=kx+b$, $x$ and $y$ are features and label (class, annotation or others) respectively from one sample. However, both $w$ and $b$ are unknown for us (of course, we can initialize them with some random values). To summarize, input $x_{i}$ in, class probability $\hat{y_{i}}$ out.
- Objective: an object to reflect the gap/difference of a model between perfect status and current status.
- Training: a procedure to minimize the gap. Due to stability of $x$ and $y$, the only changeable are $k$ and $b$.

![](./img/elements-of-ml.png)

On this page, we can see a dog picture as an input $x_{i}$, representing an feature vector. And after model transformation (the formula to caculate $\hat{y_{i}}$), we get the class probability of this picture.

### Model

$x_{i}$ represents a sample or an example (here is an image). $i$ means the current order of this dataset (ith image in this image dataset). $feature_{0}$, $feature_{1}$ till $feature_{m}$ are all the features of one sample/example (For a sample or an example, we assume it has $m$ features). Each feature may have its specific meaning and includes one or multiple number (integer/float/double type) or characters (string/char type).

$\hat{y_{i}}$ represents an inference or predicted result for example $x_{i}$. The formula of $\hat{y_{i}}$ is of logistic regression format (log-like linear model). $w$ acts as $b$ in $\hat{y_{i}} = kx_{i} + b$, calculated step by step in training and initalized with an random or fixed value.

The format of formula calculating $\hat{y_{i}}$ is logistic regression model. Although its name contains 'regression', it's a classification model. Logistic regression model comes from logistic distribution: assuming $X$ is continuous random variable, $y$ obeys logistic distribution. That is, $X$ has distribution function $F(x)$ and density function $f(x)$ as below: 

$$
F(x) = P(X \leq x) = \frac{1}{1 + e^{-(x-\mu)/\gamma}}
$$

$$
f(x) = {F}'(x) = \frac{e^{-(x-\mu)/\gamma}}{\gamma (1 + e^{-(x-\mu)/\gamma})^2}
$$

Binomial logistic regression model is a classification model, represented using conditional probability $P(Y|X)$ and its format is parameterized logistic distribution. Here $X$'s range is real number and $y$'s value is either $0$ or $1$. We use supervised learning method to estimate parameters.

Binomial logistic regression has following conditional probability distribution:

$$
P(Y=1|x）= \frac{e^{w \dot x + b}}{1 + e^{w \dot x + b}}
$$

$$
P(Y=0|x）= \frac{ 1 }{1 + e^{w \dot x + b}}
$$

$x \in {\Re}^{n}$ is input, $Y \in \{0,1\}$ is output, $w \in {\Re}^{n}$ and $b \in \Re$ both are parameters. $w$ is weight vector and $b$ is bias. $w \dot x$ is the inner product of $w$ and $x$.

Given an example $x$, we can get $P(Y=1|x)$ and $P(Y=0|x)$ using formulas above. Logistic regression will compare two conditional probabilities and classify example $x$ as bigger probability class.

For convenient, expand weight vector $w$ and $x$ as $w = (w^{(1)}, w^{(2)}, ..., w^{(n)}, b)^T$, $x = (x^{(1)}, x^{(2)}, ..., x^{(n)}, 1)^T$. Then logistic regression model is as below:

$$
P(Y=1|x) = \frac{e^{ w \dot x}}{ 1 + e^{(w \dot x)}}
$$

$$
P(Y=0|x) = \frac{1}{1+e^{w \dot x}}
$$

### Objective

According to specific model, we can use supervised learning method formulate the concrete problem with its model as an objective.

$$
L(w) = \sum_{i=1}^{n} l (y_{i}, \hat{y_{i}) + \lambda ||w||^2
$$

$l(y_{i}, \hat{y_{i}})$ in formula above is loss gap between predicted value (or label) and ground-truth label. The loss has many a format:

1. 0-1 loss function
$$
L(Y, f(x)) = 
$$
2. quadratic loss function
$$
L(Y, f(X)) = (Y - f(X))^2
$$
3. absolute loss function
$$
L(Y, f(x)) = |Y - f(X)|
$$
4. logarithmic loss function or log-likelihood loss function
$$
L(Y, P(Y|X)) = -\log{P(Y|X)}
$$

Smaller the loss function value is, the better model is. Because both model input and output $(X,Y)$ are random variables, they obey joint distribution $P(X|Y)$. Thus, the expectation of loss function is:

$$
abc
$$

### Training

$$
w \leftarrow w - 
$$

Remember the parameters $w$ and $b$ are randomly initialized or with fixed value (zero or one etc).

The solution procedure of objective is to find better parameters, step by step.
