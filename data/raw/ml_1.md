# CS229 Lecture notes

**Andrew Ng**

## Supervised learning

Let's start by talking about a few examples of supervised learning problems. Suppose we have a dataset giving the living
areas and prices of 47 houses from Portland, Oregon:

The following table:

| Living area ($feet^2$) | Price (1000$s) |
|:-----------------------|:---------------|
| 2104                   | 400            |
| 1600                   | 330            |
| 2400                   | 369            |
| 1416                   | 232            |
| 3000                   | 540            |

We can plot this data:
Given data like this, how can we learn to predict the prices of other houses in Portland, as a function of the size of
their living areas?

To establish notation for future use, we'll use $x^{(i)}$ to denote the "input" variables (living area in this example),
also called input features, and $y^{(i)}$ to denote the "output" or target variable that we are trying to predict (
price). A pair $(x^{(i)},y^{(i)})$ is called a training example, and the dataset that we'll be using to learn a list of
m training examples $\{(x^{(i)},y^{(i)}); i=1,...,m\}$ is called a training set. Note that the superscript "i" in the
notation is simply an index into the training set, and has nothing to do with exponentiation.

We will also use $\mathcal{X}$ denote the space of input values, and $\mathcal{Y}$ the space of output values. In this
example, $\mathcal{X}=\mathcal{Y}=\mathbb{R}$.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a
function $h:\mathcal{X}\mapsto\mathcal{Y}$ so that $h(x)$ is a "good" predictor for the corresponding value of y. For
historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

Training set $\rightarrow$ Learning algorithm $\rightarrow$ h
x (living area of house.) $\rightarrow$ h $\rightarrow$ predicted y (predicted price of house)

When the target variable that we're trying to predict is continuous, such as in our housing example, we call the
learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the
living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

---

### Part I Linear Regression

To make our housing example more interesting, let's consider a slightly richer dataset in which we also know the number
of bedrooms in each house:

| Living area ($feet^2$) | #bedrooms | Price (1000$s) |
|:-----------------------|:----------|:---------------|
| 2104                   | 3         | 400            |
| 1600                   | 3         | 330            |
| 2400                   | 3         | 369            |
| 1416                   | 2         | 232            |
| 3000                   | 4         | 540            |

*(Table data from source)*

Here, the x's are two-dimensional vectors in $\mathbb{R}^2$. For instance, $x_1^{(i)}$ is the living area of the i-th
house in the training set, and $x_2^{(i)}$ is its number of bedrooms. (In general, when designing a learning problem, it
will be up to you to decide what features to choose, so if you are out in Portland gathering housing data, you might
also decide to include other features such as whether each house has a fireplace, the number of bathrooms, and so on.
We'll say more about feature selection later, but for now let's take the features as given.)

To perform supervised learning, we must decide how we're going to represent functions/hypotheses h in a computer. As an
initial choice, let's say we decide to approximate y as a linear function of x:

$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2$$

Here, the $\theta_i$ are the parameters (also called weights) parameterizing the space of linear functions mapping
from $\mathcal{X}$ to $\mathcal{Y}$. When there is no risk of confusion, we will drop the subscript in $h_\theta(x)$ and
write it more simply as $h(x)$. To simplify our notation, we also introduce the convention of letting $x_0=1$ (this is
the intercept term), so that

$$h(x)=\sum_{i=0}^n\theta_ix_i=\theta^Tx,$$

where on the right-hand side above we are viewing $\theta$ and x both as vectors, and here n is the number of input
variables (not counting $x_0$).

Now, given a training set, how do we pick, or learn, the parameters $\theta$? One reasonable method seems to be to
make $h(x)$ close to y, at least for the training examples we have. To formalize this, we will define a function that
measures, for each value of the $\theta$'s, how close the $h(x^{(i)})$'s are to the corresponding $y^{(i)}$'s. We define
the cost function:

$$J(\theta)=\frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2.$$

If you've seen linear regression before, you may recognize this as the familiar least-squares cost function that gives
rise to the ordinary least squares regression model. Whether or not you have seen it previously, let's keep going, and
we'll eventually show this to be a special case of a much broader family of algorithms.

#### 1 LMS algorithm

We want to choose $\theta$ so as to minimize $J(\theta)$. To do so, let's use a search algorithm that starts with some "
initial guess" for $\theta$, and that repeatedly changes $\theta$ to make $J(\theta)$ smaller, until hopefully we
converge to a value of $\theta$ that minimizes $J(\theta)$. Specifically, let's consider the gradient descent algorithm,
which starts with some initial $\theta$, and repeatedly performs the update:

$$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta).$$

(This update is simultaneously performed for all values of $j=0,...,n$.) Here, $\alpha$ is called the learning rate.
This is a very natural algorithm that repeatedly takes a step in the direction of steepest decrease of J.

In order to implement this algorithm, we have to work out what is the partial derivative term on the right hand side.
Let's first work it out for the case of if we have only one training example (x, y), so that we can neglect the sum in
the definition of J. We have:

$$\frac{\partial}{\partial\theta_j}J(\theta)=\frac{\partial}{\partial\theta_j}\frac{1}{2}(h_\theta(x)-y)^2$$
$$=2\cdot\frac{1}{2}(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_j}(h_\theta(x)-y)$$
$$=(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_j}\left(\sum_{i=0}^n\theta_ix_i-y\right)$$
$$=(h_\theta(x)-y)x_j$$

For a single training example, this gives the update rule:

$$\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}.$$

The rule is called the LMS update rule (LMS stands for "least mean squares"), and is also known as the Widrow-Hoff
learning rule. This rule has several properties that seem natural and intuitive. For instance, the magnitude of the
update is proportional to the error term $(y^{(i)}-h_\theta(x^{(i)}))$; thus, for instance, if we are encountering a
training example on which our prediction nearly matches the actual value of $y^{(i)}$, then we find that there is little
need to change the parameters; in contrast, a larger change to the parameters will be made if our
prediction $h_\theta(x^{(i)})$ has a large error (i.e., if it is very far from $y^{(i)}$).

We'd derived the LMS rule for when there was only a single training example. There are two ways to modify this method
for a training set of more than one example. The first is replace it with the following algorithm:

Repeat until convergence {
&nbsp;&nbsp;&nbsp;&nbsp;$\theta_j:=\theta_j+\alpha\sum_{i=1}^m(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)} \quad \text{(for every j).}$
}

The reader can easily verify that the quantity in the summation in the update rule above is
just $\partial J(\theta)/\partial\theta_j$ (for the original definition of J). So, this is simply gradient descent on
the original cost function J. This method looks at every example in the entire training set on every step, and is called
batch gradient descent. Note that, while gradient descent can be susceptible to local minima in general, the
optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus
gradient descent always converges (assuming the learning rate $\alpha$ is not too large) to the global minimum. Indeed,
J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

We use the notation "$a:=b$" to denote an operation (in a computer program) in which we set the value of a variable a to
be equal to the value of b. In other words, this operation overwrites a with the value of b. In contrast, we will
write "$a=b$" when we are asserting a statement of fact, that the value of a is equal to the value of b.

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient
descent, which was initialized at (48,30). The x's in the figure (joined by straight lines) mark the successive values
of $\theta$ that gradient descent went through.

When we run batch gradient descent to fit $\theta$ on our previous dataset, to learn to predict housing price as a
function of living area, we obtain $\theta_0=71.27$, $\theta_1=0.1345$. If we plot $h_\theta(x)$ as a function of x (
area), along with the training data, we obtain the following figure:
If the number of bedrooms were included as one of the input features as well, we
get $\theta_0=89.60$, $\theta_1=0.1392$, $\theta_2=-8.738$.

The above results were obtained with batch gradient descent. There is an alternative to batch gradient descent that also
works very well. Consider the following algorithm:

Loop {
&nbsp;&nbsp;&nbsp;&nbsp;for $i=1$ to m, {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)} \quad \text{(for every j).}$
&nbsp;&nbsp;&nbsp;&nbsp;}
}

In this algorithm, we repeatedly run through the training set, and each time we encounter a training example, we update
the parameters according to the gradient of the error with respect to that single training example only. This algorithm
is called stochastic gradient descent (also incremental gradient descent). Whereas batch gradient descent has to scan
through the entire training set before taking a single step—a costly operation if m is large—stochastic gradient descent
can start making progress right away, and continues to make progress with each example it looks at. Often, stochastic
gradient descent gets "close" to the minimum much faster than batch gradient descent. (Note however that it may never "
converge" to the minimum, and the parameters $\theta$ will keep oscillating around the minimum of $J(\theta)$; but in
practice most of the values near the minimum will be reasonably good approximations to the true minimum.) For these
reasons, particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient
descent.

#### 2 The normal equations

Gradient descent gives one way of minimizing J. Let's discuss a second way of doing so, this time performing the
minimization explicitly and without resorting to an iterative algorithm. In this method, we will minimize J by
explicitly taking its derivatives with respect to the $\theta_j$'s, and setting them to zero. To enable us to do this
without having to write reams of algebra and pages full of matrices of derivatives, let's introduce some notation for
doing calculus with matrices.

##### 2.1 Matrix derivatives

For a function $f:\mathbb{R}^{m\times n}\mapsto\mathbb{R}$ mapping from m-by-n matrices to the real numbers, we define
the derivative of f with respect to A to be:

$$\nabla_A f(A)=\begin{bmatrix}\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1n}} \\ \vdots & \cdots & \vdots \\ \frac{\partial f}{\partial A_{m1}} & \cdots & \frac{\partial f}{\partial A_{mn}}\end{bmatrix}$$

Thus, the gradient $\nabla_A f(A)$ is itself an m-by-n matrix, whose (i, j)-element is $\partial f/\partial A_{ij}$. For
example, suppose $A=\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix}$ is a 2-by-2 matrix, and the
function $f:\mathbb{R}^{2\times2}\mapsto\mathbb{R}$ is given by $f(A)=\frac{3}{2}A_{11}+5A_{12}^2+A_{21}A_{22}$.
Here, $A_{ij}$ denotes the (i, j) entry of the matrix A. We then have

$$\nabla_A f(A)=\begin{bmatrix}\frac{3}{2} & 10A_{12} \\ A_{22} & A_{21}\end{bmatrix}.$$

We also introduce the trace operator, written "tr." For an n-by-n (square) matrix A, the trace of A is defined to be the
sum of its diagonal entries:

$$trA=\sum_{i=1}^n A_{ii}$$

If a is a real number (i.e., a 1-by-1 matrix), then $tr a=a$. (If you haven't seen this "operator notation" before, you
should think of the trace of A as $tr(A)$, or as application of the "trace" function to the matrix A. It's more commonly
written without the parentheses, however.)

The trace operator has the property that for two matrices A and B such that AB is square, we have that $trAB=trBA$. (
Check this yourself!) As corollaries of this, we also have, e.g.,

$$trABC=trCAB=trBCA,$$
$$trABCD=trDABC=trCDAB=trBCDA.$$

The following properties of the trace operator are also easily verified. Here, A and B are square matrices, and a is a
real number:

$$trA=trA^T$$
$$tr(A+B)=trA+trB$$
$$tr(aA)=a trA$$

We now state without proof some facts of matrix derivatives (we won't need some of these until later this quarter).
Equation (4) applies only to non-singular square matrices A, where $|A|$ denotes the determinant of A. We have:

$$\nabla_A trAB=B^T \quad (1)$$
$$\nabla_{A^T} f(A)=(\nabla_A f(A))^T \quad (2)$$
$$\nabla_A trABA^T C=CAB+C^T AB^T \quad (3)$$
$$\nabla_A |A|=|A|(A^{-1})^T \quad (4)$$

To make our matrix notation more concrete, let us now explain in detail the meaning of the first of these equations.
Suppose we have some fixed matrix $B\in\mathbb{R}^{n\times m}$. We can then define a
function $f:\mathbb{R}^{m\times n}\mapsto\mathbb{R}$ according to $f(A)=trAB$. Note that this definition makes sense,
because if $A\in\mathbb{R}^{m\times n}$, then AB is a square matrix, and we can apply the trace operator to it; thus, f
does indeed map from $\mathbb{R}^{m\times n}$ to $\mathbb{R}$. We can then apply our definition of matrix derivatives to
find $\nabla_A f(A)$, which will itself by an m-by-n matrix. Equation (1) above states that the (i, j) entry of this
matrix will be given by the (i, j)-entry of $B^T$ or equivalently, by $B_{ji}$.

The proofs of Equations (1-3) are reasonably simple, and are left as an exercise to the reader. Equations (4) can be
derived using the adjoint representation of the inverse of a matrix.

##### 2.2 Least squares revisited

Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of $\theta$ that
minimizes $J(\theta)$. We begin by re-writing J in matrix-vectorial notation.

Given a training set, define the design matrix X to be the m-by-n matrix (actually $m$-by-$n+1$ if we include the
intercept term) that contains the training examples' input values in its rows:

$$X=\begin{bmatrix}-(x^{(1)})^T- \\ -(x^{(2)})^T- \\ \vdots \\ -(x^{(m)})^T-\end{bmatrix}$$

Also, let $\vec{y}$ be the m-dimensional vector containing all the target values from the training set:

$$\vec{y}=\begin{bmatrix}y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)}\end{bmatrix}.$$

Now, since $h_\theta(x^{(i)})=(x^{(i)})^T\theta$, we can easily verify that

$$X\theta-\vec{y}=\begin{bmatrix}(x^{(1)})^T\theta \\ \vdots \\ (x^{(m)})^T\theta\end{bmatrix}-\begin{bmatrix}y^{(1)} \\ \vdots \\ y^{(m)}\end{bmatrix} = \begin{bmatrix}h_\theta(x^{(1)})-y^{(1)} \\ \vdots \\ h_\theta(x^{(m)})-y^{(m)}\end{bmatrix}$$

Thus, using the fact that for a vector z, we have that $z^T z=\sum_i z_i^2$.

$$\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y})=\frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$$
$$=J(\theta)$$

Finally, to minimize J, let's find its derivatives with respect to $\theta$. Combining Equations (2) and (3), we find
that

$$\nabla_{A^T} trABA^T C=B^T A^T C^T+BA^T C \quad (5)$$

Hence,

$$\nabla_\theta J(\theta)=\nabla_\theta\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y})$$
$$=\frac{1}{2}\nabla_\theta(\theta^T X^T X\theta-\theta^T X^T\vec{y}-\vec{y}^T X\theta+\vec{y}^T\vec{y})$$
$$=\frac{1}{2}\nabla_\theta tr(\theta^T X^T X\theta-\theta^T X^T\vec{y}-\vec{y}^T X\theta+\vec{y}^T\vec{y})$$
$$=\frac{1}{2}\nabla_\theta(tr \theta^T X^T X\theta-2tr\vec{y}^T X\theta)$$
$$=\frac{1}{2}(X^T X\theta+X^T X\theta-2X^T\vec{y})$$
$$=X^T X\theta-X^T\vec{y}$$

In the third step, we used the fact that the trace of a real number is just the real number; the fourth step used the
fact that $trA=trA^T$, and the fifth step used Equation (5) with $A^T=\theta$, $B=B^T=X^T X$, and $C=I$ and Equation (
1). To minimize J, we set its derivatives to zero, and obtain the normal equations:

$$X^T X\theta=X^T\vec{y}$$

Thus, the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation

$$\theta=(X^T X)^{-1}X^T\vec{y}$$

#### 3 Probabilistic interpretation

When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost
function J, be a reasonable choice? In this section, we will give a set of probabilistic assumptions, under which
least-squares regression is derived as a very natural algorithm.

Let us assume that the target variables and the inputs are related via the equation

$$y^{(i)}=\theta^T x^{(i)}+\epsilon^{(i)},$$

where $\epsilon^{(i)}$ is an error term that captures either unmodeled effects (such as if there are some features very
pertinent to predicting housing price, but that we'd left out of the regression), or random noise. Let us further assume
that the $\epsilon^{(i)}$ are distributed IID (independently and identically distributed) according to a Gaussian
distribution (also called a Normal distribution) with mean zero and some variance $\sigma^2$. We can write this
assumption as "$\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2)$." I.e., the density of $\epsilon^{(i)}$ is given by

$$p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$$

This implies that

$$p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}\right)$$

The notation "$p(y^{(i)}|x^{(i)};\theta)$" indicates that this is the distribution of $y^{(i)}$ given $x^{(i)}$ and
parameterized by $\theta$. Note that we should not condition on $\theta$ ("$p(y^{(i)}|x^{(i)},\theta)$"), since $\theta$
is not a random variable. We can also write the distribution of $y^{(i)}$
as $y^{(i)} | x^{(i)}; \theta\sim\mathcal{N}(\theta^T x^{(i)},\sigma^2)$.

Given X (the design matrix, which contains all the $x^{(i)}$'s) and $\theta$, what is the distribution of the $y^{(i)}$'
s? The probability of the data is given by $p(\vec{y}|X;\theta)$. This quantity is typically viewed a function
of $\vec{y}$ (and perhaps X), for a fixed value of $\theta$. When we wish to explicitly view this as a function
of $\theta$, we will instead call it the likelihood function:

$$L(\theta)=L(\theta;X,\vec{y})=p(\vec{y}|X;\theta)$$

Note that by the independence assumption on the $\epsilon^{(i)}$'s (and hence also the $y^{(i)}$'s given the $x^{(i)}$'
s), this can also be written

$$L(\theta)=\prod_{i=1}^m p(y^{(i)}|x^{(i)};\theta)$$
$$=\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}\right)$$

Now, given this probabilistic model relating the $y^{(i)}$'s and the $x^{(i)}$'s, what is a reasonable way of choosing
our best guess of the parameters $\theta$? The principal of maximum likelihood says that we should choose $\theta$ so as
to make the data as high probability as possible. I.e., we should choose $\theta$ to maximize $L(\theta)$.

Instead of maximizing $L(\theta)$, we can also maximize any strictly increasing function of $L(\theta)$. In particular,
the derivations will be a bit simpler if we instead maximize the log likelihood $l(\theta)$:

$$l(\theta)=\log L(\theta)$$
$$=\log\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}\right)$$
$$=\sum_{i=1}^m \log\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}\right)$$
$$=m \log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\cdot\frac{1}{2}\sum_{i=1}^m(y^{(i)}-\theta^T x^{(i)})^2.$$

Hence, maximizing $l(\theta)$ gives the same answer as minimizing

$$\frac{1}{2}\sum_{i=1}^m(y^{(i)}-\theta^T x^{(i)})^2,$$

which we recognize to be $J(\theta)$, our original least-squares cost function.

To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding
the maximum likelihood estimate of $\theta$. This is thus one set of assumptions under which least-squares regression
can be justified as a very natural method that's just doing maximum likelihood estimation. (Note however that the
probabilistic assumptions are by no means necessary for least-squares to be a perfectly good and rational procedure, and
there may and indeed there are other natural assumptions that can also be used to justify it.)

Note also that, in our previous discussion, our final choice of $\theta$ did not depend on what was $\sigma^2$, and
indeed we'd have arrived at the same result even if $\sigma^2$ were unknown. We will use this fact again later, when we
talk about the exponential family and generalized linear models.

#### 4 Locally weighted linear regression

Consider the problem of predicting y from $x\in\mathbb{R}$. The leftmost figure below shows the result of fitting
a $y=\theta_0+\theta_1x$ to a dataset. We see that the data doesn't really lie on straight line, and so the fit is not
very good.

Instead, if we had added an extra feature $x^2$, and fit $y=\theta_0+\theta_1x+\theta_2x^2$, then we obtain a slightly
better fit to the data. (See middle figure) Naively, it might seem that the more features we add, the better. However,
there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5-th order
polynomial $y=\sum_{j=0}^5\theta_jx^j$. We see that even though the fitted curve passes through the data perfectly, we
would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). Without
formally defining what these terms mean, we'll say the figure on the left shows an instance of underfitting—in which the
data clearly shows structure not captured by the model—and the figure on the right is an example of overfitting. (Later
in this class, when we talk about learning theory we'll formalize some of these notions, and also define more carefully
just what it means for a hypothesis to be good or bad.)

As discussed previously, and as shown in the example above, the choice of features is important to ensuring good
performance of a learning algorithm. (When we talk about model selection, we'll also see algorithms for automatically
choosing a good set of features.) In this section, let us talk briefly talk about the locally weighted linear
regression (LWR) algorithm which, assuming there is sufficient training data, makes the choice of features less
critical. This treatment will be brief, since you'll get a chance to explore some of the properties of the LWR algorithm
yourself in the homework.

In the original linear regression algorithm, to make a prediction at a query point x (i.e., to evaluate $h(x)$), we
would:

1. Fit $\theta$ to minimize $\sum_i(y^{(i)}-\theta^Tx^{(i)})^2$
2. Output $\theta^Tx$.

In contrast, the locally weighted linear regression algorithm does the following:

1. Fit $\theta$ to minimize $\sum_iw^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2$.
2. Output $\theta^Tx$

Here, the $w^{(i)}$'s are non-negative valued weights. Intuitively, if $w^{(i)}$ is large for a particular value of i,
then in picking $\theta$, we'll try hard to make $(y^{(i)}-\theta^Tx^{(i)})^2$ small. If $w^{(i)}$ is small, then
the $(y^{(i)}-\theta^Tx^{(i)})^2$ error term will be pretty much ignored in the fit. A fairly standard choice for the
weights is

$$w^{(i)}=\exp\left(-\frac{(x^{(i)}-x)^2}{2\tau^2}\right)$$

Note that the weights depend on the particular point x at which we're trying to evaluate x. Moreover, if $|x^{(i)}-x|$
is small, then $w^{(i)}$ is close to 1; and if $|x^{(i)}-x|$ is large, then $w^{(i)}$ is small. Hence, $\theta$ is
chosen giving a much higher "weight" to the (errors on) training examples close to the query point x. (Note also that
while the formula for the weights takes a form that is cosmetically similar to the density of a Gaussian distribution,
the $w^{(i)}$ do not directly have anything to do with Gaussians, and in particular the $w^{(i)}$ are not random
variables, normally distributed or otherwise.) The parameter $\tau$ controls how quickly the weight of a training
example falls off with distance of its $x^{(i)}$ from the query point x; $\tau$ is called the bandwidth parameter, and
is also something that you'll get to experiment with in your homework.

Locally weighted linear regression is the first example we're seeing of a non-parametric algorithm. The (unweighted)
linear regression algorithm that we saw earlier is known as a parametric learning algorithm, because it has a fixed,
finite number of parameters (the $\theta_i$'s), which are fit to the data. Once we've fit the $\theta_i$ and stored them
away, we no longer need to keep the training data around to make future predictions. In contrast, to make predictions
using locally weighted linear regression, we need to keep the entire training set around. The term "non-parametric" (
roughly) refers to the fact that the amount of stuff we need to keep in order to represent the hypothesis h grows
linearly with the size of the training set. If x is vector-valued, this is generalized to
be $w^{(i)}=\exp(-(x^{(i)}-x)^T(x^{(i)}-x)/(2\tau^2))$, or $w^{(i)}=\exp(-(x^{(i)}-x)^T\Sigma^{-1}(x^{(i)}-x)/2)$, for
an appropriate choice of $\tau$ or $\Sigma$.

### Part II

Classification and logistic regression

Let's now talk about the classification problem. This is just like the regression problem, except that the values $y$ we
now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification
problem in which $y$ can take on only two values, 0 and 1. (Most of what we say here will also generalize to the
multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some
features of a piece of email, and $y$ may be 1 if it is a piece of spam mail, and 0 otherwise. 0 is also called the
negative class, and 1 the positive class, and they are sometimes also denoted by the symbols "-" and "+."
Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the label for the training example.

#### 5 Logistic regression

We could approach the classification problem ignoring the fact that $y$ is discrete-valued, and use our old linear
regression algorithm to try to predict $y$ given $x$. However, it is easy to construct examples where this method
performs very poorly. Intuitively, it also doesn't make sense for $h_\theta(x)$ to take values larger than 1 or smaller
than 0 when we know that $y \in \{0, 1\}$. To fix this, let's change the form for our hypotheses $h_\theta(x)$. We will
choose

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}},$$

where

$$g(z) = \frac{1}{1 + e^{-z}}$$

is called the logistic function or the sigmoid function.

Notice that $g(z)$ tends towards 1 as $z \rightarrow \infty$ and $g(z)$ tends towards 0 as $z \rightarrow -\infty$.
Moreover, $g(z)$, and hence also $h(x)$, is always bounded between 0 and 1. As before, we are keeping the convention of
letting $x_0 = 1$, so that $\theta^T x = \theta_0 + \sum_{j=1}^n \theta_j x_j$.

For now, let's take the choice of $g$ as given. Other functions that smoothly increase from 0 to 1 can also be used, but
for a couple of reasons that we'll see later (when we talk about GLMs, and when we talk about generative learning
algorithms), the choice of the logistic function is a fairly natural one. Before moving on, here's a useful property of
the derivative of the sigmoid function, which we write as $g'$:

$$g'(z) = \frac{d}{dz} \frac{1}{1 + e^{-z}}$$
$$= \frac{1}{(1 + e^{-z})^2} (e^{-z})$$
$$= \frac{1}{(1 + e^{-z})} \cdot \left(1 - \frac{1}{(1 + e^{-z})}\right)$$
$$= g(z)(1 - g(z)).$$

So, given the logistic regression model, how do we fit $\theta$ for it? Following how we saw least squares regression
could be derived as the maximum likelihood estimator under a set of assumptions, let's endow our classification model
with a set of probabilistic assumptions, and then fit the parameters via maximum likelihood.

Let us assume that

$$P(y=1 | x; \theta) = h_\theta(x)$$
$$P(y=0 | x; \theta) = 1 - h_\theta(x)$$

Note that this can be written more compactly as

$$p(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$

Assuming that the $m$ training examples were generated independently, we can then write down the likelihood of the
parameters as

$$L(\theta) = p(\vec{y} | X; \theta)$$
$$= \prod_{i=1}^m p(y^{(i)} | x^{(i)}; \theta)$$
$$= \prod_{i=1}^m (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1 - y^{(i)}}$$

As before, it will be easier to maximize the log likelihood:

$$l(\theta) = \log L(\theta)$$
$$= \sum_{i=1}^m y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log(1 - h(x^{(i)}))$$

How do we maximize the likelihood? Similar to our derivation in the case of linear regression, we can use gradient
ascent. Written in vectorial notation, our updates will therefore be given
by $\theta := \theta + \alpha \nabla_\theta l(\theta)$. (Note the positive rather than negative sign in the update
formula, since we're maximizing, rather than minimizing, a function now.) Let's start by working with just one training
example $(x, y)$, and take derivatives to derive the stochastic gradient ascent rule:

$$\frac{\partial}{\partial \theta_j} l(\theta) = \left(y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)}\right) \frac{\partial}{\partial \theta_j} g(\theta^T x)$$
$$= \left(y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)}\right) g(\theta^T x) (1 - g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x$$
$$= (y(1 - g(\theta^T x)) - (1 - y)g(\theta^T x)) x_j$$
$$= (y - h_\theta(x)) x_j$$

Above, we used the fact that $g'(z) = g(z)(1 - g(z))$. This therefore gives us the stochastic gradient ascent rule

$$\theta_j := \theta_j + \alpha(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}$$

If we compare this to the LMS update rule, we see that it looks identical; but this is not the same algorithm,
because $h_\theta(x^{(i)})$ is now defined as a non-linear function of $\theta^T x^{(i)}$. Nonetheless, it's a little
surprising that we end up with the same update rule for a rather different algorithm and learning problem. Is this
coincidence, or is there a deeper reason behind this? We'll answer this when we get to GLM models. (See also the extra
credit problem on Q3 of problem set 1.)

#### 6 Digression: The perceptron learning algorithm

We now digress to talk briefly about an algorithm that's of some historical interest, and that we will also return to
later when we talk about learning theory. Consider modifying the logistic regression method to "force" it to output
values that are either 0 or 1 exactly. To do so, it seems natural to change the definition of $g$ to be the threshold
function:

$$g(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

If we then let $h_\theta(x) = g(\theta^T x)$ as before but using this modified definition of $g$, and if we use the
update rule

$$\theta_j := \theta_j + \alpha(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}$$

then we have the perceptron learning algorithm.

In the 1960s, this "perceptron" was argued to be a rough model for how individual neurons in the brain work. Given how
simple the algorithm is, it will also provide a starting point for our analysis when we talk about learning theory later
in this class. Note however that even though the perceptron may be cosmetically similar to the other algorithms we
talked about, it is actually a very different type of algorithm than logistic regression and least squares linear
regression; in particular, it is difficult to endow the perceptron's predictions with meaningful probabilistic
interpretations, or derive the perceptron as a maximum likelihood estimation algorithm.

#### 7 Another algorithm for maximizing $l(\theta)$

Returning to logistic regression with $g(z)$ being the sigmoid function, let's now talk about a different algorithm for
maximizing $l(\theta)$. To get us started, let's consider Newton's method for finding a zero of a function.
Specifically, suppose we have some function $f: \mathbb{R} \mapsto \mathbb{R}$, and we wish to find a value of $\theta$
so that $f(\theta) = 0$. Here, $\theta \in \mathbb{R}$ is a real number. Newton's method performs the following update:

$$\theta := \theta - \frac{f(\theta)}{f'(\theta)}.$$

This method has a natural interpretation in which we can think of it as approximating the function $f$ via a linear
function that is tangent to $f$ at the current guess $\theta$, solving for where that linear function equals to zero,
and letting the next guess for $\theta$ be where that linear function is zero.

In the leftmost figure, we see the function $f$ plotted along with the line $y = 0$. We're trying to find $\theta$ so
that $f(\theta) = 0$; the value of $\theta$ that achieves this is about 1.3. Suppose we initialized the algorithm
with $\theta = 4.5$. Newton's method then fits a straight line tangent to $f$ at $\theta = 4.5$ and solves for where
that line evaluates to 0. (Middle figure.) This gives us the next guess for $\theta$, which is about 2.8. The rightmost
figure shows the result of running one more iteration, which updates $\theta$ to about 1.8. After a few more iterations,
we rapidly approach $\theta = 1.3$.

Newton's method gives a way of getting to $f(\theta) = 0$. What if we want to use it to maximize some function $l$? The
maxima of $l$ correspond to points where its first derivative $l'(\theta)$ is zero. So, by
letting $f(\theta) = l'(\theta)$, we can use the same algorithm to maximize $l$, and we obtain the update rule:

$$\theta := \theta - \frac{l'(\theta)}{l''(\theta)}.$$

(Something to think about: How would this change if we wanted to use Newton's method to minimize rather than maximize a
function?)

Lastly, in our logistic regression setting, $\theta$ is vector-valued, so we need to generalize Newton's method to this
setting. The generalization of Newton's method to this multidimensional setting (also called the Newton-Raphson method)
is given by

$$\theta := \theta - H^{-1} \nabla_\theta l(\theta).$$

Here, $\nabla_\theta l(\theta)$ is, as usual, the vector of partial derivatives of $l(\theta)$ with respect to
the $\theta_i$'s; and $H$ is an $n$-by-$n$ matrix (actually, $n+1$-by-$n+1$, assuming that we include the intercept
term) called the Hessian, whose entries are given by

$$H_{ij} = \frac{\partial^2 l(\theta)}{\partial \theta_i \partial \theta_j}.$$

Newton's method typically enjoys faster convergence than (batch) gradient descent, and requires many fewer iterations to
get very close to the minimum. One iteration of Newton's can, however, be more expensive than one iteration of gradient
descent, since it requires finding and inverting an $n$-by-$n$ Hessian; but so long as $n$ is not too large, it is
usually much faster overall. When Newton's method is applied to maximize the logistic regression log likelihood
function $l(\theta)$, the resulting method is also called Fisher scoring.