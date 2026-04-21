## Generative Learning algorithms

So far, we've mainly been talking about learning algorithms that model $p(y|x;\theta)$, the conditional distribution of
y given z. For instance, logistic regression modeled $p(y|x;\theta)$ as $h_{\theta}(x)=g(\theta^{T}x)$ where g is the
sigmoid function. In these notes, we'll talk about a different type of learning algorithm.

Consider a classification problem in which we want to learn to distinguish between elephants $(y=1)$ and dogs $(y=0)$,
based on some features of an animal. Given a training set, an algorithm like logistic regression or the perceptron
algorithm (basically) tries to find a straight line that is, a decision boundary that separates the elephants and dogs.
Then, to classify a new animal as either an elephant or a dog, it checks on which side of the decision boundary it
falls, and makes its prediction accordingly.

Here's a different approach. First, looking at elephants, we can build a model of what elephants look like. Then,
looking at dogs, we can build a separate model of what dogs look like. Finally, to classify a new animal, we can match
the new animal against the elephant model, and match it against the dog model, to see whether the new animal looks more
like the elephants or more like the dogs we had seen in the training set.

Algorithms that try to learn $p(y|x)$ directly (such as logistic regression), or algorithms that try to learn mappings
directly from the space of inputs X to the labels {0,1}, (such as the perceptron algorithm) are called discriminative
learning algorithms. Here, we'll talk about algorithms that instead try to model $p(x|y)$ (and $p(y)$). These algorithms
are called generative learning algorithms. For instance, if y indicates whether an example is a dog (0) or an elephant (
1), then $p(x|y=0)$ models the distribution of dogs' features, and $p(x|y=1)$ models the distribution of elephants
features.

After modeling $p(y)$ (called the class priors) and $p(x|y)$, our algorithm can then use Bayes rule to derive the
posterior distribution on y given r:

$$p(y|x)=\frac{p(x|y)p(y)}{p(x)}$$

Here, the denominator is given by $p(x)=p(x|y=1)p(y=1)+p(x|y=0)p(y=0)$ (you should be able to verify that this is true
from the standard properties of probabilities), and thus can also be expressed in terms of the quantities $p(x|y)$
and $p(y)$ that we've learned. Actually, if were calculating $p(y|x)$ in order to make a prediction, then we don't
actually need to calculate the denominator, since

$$arg~max_{y}p(y|x)=arg~max_{y}\frac{p(x|y)p(y)}{p(x)} = arg~max_{y}p(x|y)p(y)$$

### 1 Gaussian discriminant analysis

The first generative learning algorithm that we'll look at is Gaussian discriminant analysis (GDA). In this model, we'll
assume that $p(x|y)$ is distributed according to a multivariate normal distribution. Let's talk briefly about the
properties of multivariate normal distributions before moving on to the GDA model itself.

#### 1.1 The multivariate normal distribution

The multivariate normal distribution in n-dimensions, also called the multivariate Gaussian distribution, is
parameterized by a mean vector $\mu\in\mathbb{R}^{n}$ and a covariance matrix $\Sigma\in\mathbb{R}^{n\times n}$,
where $\Sigma\ge0$ is symmetric and positive semi-definite. Also written $\mathcal{N}(\mu,\Sigma)$, its density is given
by:

$$p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))$$

In the equation above, $|\Sigma|$ denotes the determinant of the matrix $\Sigma$.

For a random variable X distributed $\mathcal{N}(\mu,\Sigma)$, the mean is (unsurprisingly) given by $\mu$:

$$E[X]=\int_{x}x~p(x;\mu,\Sigma)dx=\mu$$

The covariance of a vector-valued random variable Z is defined as $Cov(Z)=E[(Z-E[Z])(Z-E[Z])^{T}]$. This generalizes the
notion of the variance of a real-valued random variable. The covariance can also be defined
as $Cov(Z)=E[ZZ^{T}]-(E[Z])(E[Z])^{T}$ (You should be able to prove to yourself that these two definitions are
equivalent.) If $X\sim\mathcal{N}(\mu,\Sigma)$, then

$$Cov(X)=\Sigma$$

Here're some examples of what the density of a Gaussian distribution looks like:

*[Description of Plot: Three 3D surface plots of a bell-shaped multivariate Gaussian distribution. The leftmost plot is a standard normal distribution with mean zero and identity covariance. The middle plot is taller and narrower, representing a smaller covariance ($\Sigma=0.6I$). The rightmost plot is flatter and wider, representing a larger covariance ($\Sigma=2I$).]*

The left-most figure shows a Gaussian with mean zero (that is, the $2\times1$ zero-vector) and covariance
matrix $\Sigma=I$ (the $2\times2$ identity matrix). A Gaussian with zero mean and identity covariance is also called the
standard normal distribution. The middle figure shows the density of a Gaussian with zero mean and $\Sigma=0.6I$ and in
the rightmost figure shows one with, $\Sigma=2I$. We see that as $\Sigma$ becomes larger, the Gaussian becomes more "
spread-out," and as it becomes smaller, the distribution becomes more "compressed."

Let's look at some more examples.

*[Description of Plot: Three 3D surface plots showing Gaussians with a mean of 0 but varying off-diagonal covariance entries. The leftmost is a standard Gaussian. The middle and rightmost plots are progressively compressed toward the $45^{\circ}$ line ($x_1=x_2$) as the off-diagonal entries increase to 0.5 and 0.8 respectively.]*

The figures above show Gaussians with mean 0, and with covariance matrices respectively

$$\Sigma=[\begin{matrix}1&0\\ 0&1\end{matrix}] \quad \Sigma=[\begin{matrix}1&0.5\\ 0.5&1\end{matrix}] \quad \Sigma=[\begin{matrix}1&0.8\\ 0.8&1\end{matrix}]$$

The leftmost figure shows the familiar standard normal distribution, and we see that as we increase the off-diagonal
entry in $\Sigma$, the density becomes more "compressed" towards the $45^{\circ}$ line (given by $x_{1}=x_{2}$). We can
see this more clearly when we look at the contours of the same three densities:

*[Description of Plot: Three 3D surface plots of Gaussian distributions. These are compressed in the opposite direction (perpendicular to the $45^{\circ}$ line) due to negative off-diagonal entries (-0.5 and -0.8).]*

Here's one last set of examples generated by varying $\Sigma$:

The plots above used, respectively,
$$\Sigma=[\begin{matrix}1&-0.5\\ -0.5&1\end{matrix}]$$
$$\Sigma=[\begin{matrix}1&-0.8\\ -0.8&1\end{matrix}]$$
$$\Sigma=[\begin{matrix}3&0.8\\ 0.8&1\end{matrix}]$$

From the leftmost and middle figures, we see that by decreasing the off-diagonal elements of the covariance matrix, the
density now becomes "compressed" again, but in the opposite direction. Lastly, as we vary the parameters, more generally
the contours will form ellipses (the rightmost figure showing an example).

As our last set of examples, fixing $\Sigma=I$, by varying $\mu$, we can also move the mean of the density around.

*[Description of Plot: Not explicitly shown in the document, but described as figures generated by shifting the mean $\mu$ around the plane while keeping the identity covariance matrix constant.]*

The figures above were generated using $\Sigma=I$ and respectively
$$\mu=[\begin{matrix}-0.5\\ 0\end{matrix}]$$
$$\mu=[\begin{matrix}1\\ 0\end{matrix}]$$;$$\mu=[\begin{matrix}-1\\ -1.5\end{matrix}]$$

#### 1.2 The Gaussian Discriminant Analysis model

When we have a classification problem in which the input features z are continuous-valued random variables, we can then
use the Gaussian Discriminant Analysis (GDA) model, which models $p(x|y)$ using a multivariate normal distribution.

The model is:
$$y\sim Bernoulli(\phi)$$
$$x|y=0\sim\mathcal{N}(\mu_{0},\Sigma)$$
$$x|y=1\sim\mathcal{N}(\mu_{1},\Sigma)$$

Writing out the distributions, this is:
$$p(y)=\phi^{y}(1-\phi)^{1-y}$$
$$p(x|y=0)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu_{0})^{T}\Sigma^{-1}(x-\mu_{0}))$$
$$p(x|y=1)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu_{1})^{T}\Sigma^{-1}(x-\mu_{1}))$$

Here, the parameters of our model are $\phi$, $\Sigma$, $\mu_{0}$ and $\mu_{1}$. (Note that while there're two different
mean vectors $\mu_{0}$ and $\mu_{1}$, this model is usually applied using only one covariance matrix $\Sigma$.) The
log-likelihood of the data is given by

$$l(\phi,\mu_{0},\mu_{1},\Sigma)=log\prod_{i=1}^{m}p(x^{(i)},y^{(i)};\phi,\mu_{0},\mu_{1},\Sigma)$$
$$=log\prod_{i=1}^{m}p(x^{(i)}|y^{(i)};\mu_{0},\mu_{1},\Sigma)p(y^{(i)};\phi).$$

By maximizing $l$ with respect to the parameters, we find the maximum likelihood estimate of the parameters (see problem
set 1) to be:

$$\phi=\frac{1}{m}\sum_{i=1}^{m}1\{y^{(i)}=1\}$$
$$\mu_{0}=\frac{\sum_{i=1}^{m}1\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^{m}1\{y^{(i)}=0\}}$$
$$\mu_{1}=\frac{\sum_{i=1}^{m}1\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^{m}1\{y^{(i)}=1\}}$$
$$\Sigma=\frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^{T}.$$

Pictorially, what the algorithm is doing can be seen in as follows:

*[Description of Plot: A 2D scatter plot displaying two classes of data points (circles and crosses). Overlaid on each class cluster are the concentric contour lines of a fitted Gaussian distribution. Because both distributions share the same covariance matrix, the contour ellipses are identical in shape and orientation but centered on different means. A straight line runs between the two clusters, representing the linear decision boundary where $p(y=1|x)=0.5$.]*

Shown in the figure are the training set, as well as the contours of the two Gaussian distributions that have been fit
to the data in each of the two classes. Note that the two Gaussians have contours that are the same shape and
orientation, since they share a covariance matrix $\Sigma$, but they have different means $\mu_{0}$ and $\mu_{1}$. Also
shown in the figure is the straight line giving the decision boundary at which $p(y=1|x)=0.5$. On one side of the
boundary, we'll predict $y=1$ to be the most likely outcome, and on the other side, we'll predict $y=0$.

#### 1.3 Discussion: GDA and logistic regression

The GDA model has an interesting relationship to logistic regression. If we view the
quantity $p(y=1|x;\phi,\mu_{0},\mu_{1},\Sigma)$ as a function of $x$, we'll find that it can be expressed in the form

$$p(y=1|x;\phi,\Sigma,\mu_{0},\mu_{1})=\frac{1}{1+exp(-\theta^{T}x)},$$

where $\theta$ is some appropriate function of $\phi, \Sigma, \mu_0, \mu_1$. This is exactly the form that logistic
regression a discriminative algorithm used to model $p(y=1|x)$.

When would we prefer one model over another? GDA and logistic regression will, in general, give different decision
boundaries when trained on the same dataset. Which is better?

We just argued that if $p(x|y)$ is multivariate gaussian (with shared $\Sigma$), then $p(y|x)$ necessarily follows a
logistic function. The converse, however, is not true; i.e., $p(y|x)$ being a logistic function does not imply $p(x|y)$
is multivariate gaussian. This shows that GDA makes stronger modeling assumptions about the data than does logistic
regression. It turns out that when these modeling assumptions are correct, then GDA will find better fits to the data,
and is a better model. Specifically, when $p(x|y)$ is indeed gaussian (with shared $\Sigma$), then GDA is asymptotically
efficient. Informally, this means that in the limit of very large training sets (large m), there is no algorithm that is
strictly better than GDA (in terms of, say, how accurately they estimate $p(y|x)$). In particular, it can be shown that
in this setting, GDA will be a better algorithm than logistic regression; and more generally, even for small training
set sizes, we would generally expect GDA to better.

In contrast, by making significantly weaker assumptions, logistic regression is also more robust and less sensitive to
incorrect modeling assumptions. There are many different sets of assumptions that would lead to $p(y|x)$ taking the form
of a logistic function. For example, if $x|y=0\sim Poisson(\lambda_{0})$, and $x|y=1\sim Poisson(\lambda_{1})$,
then $p(y|x)$ will be logistic. Logistic regression will also work well on Poisson data like this. But if we were to use
GDA on such data and fit Gaussian distributions to such non-Gaussian data then the results will be less predictable, and
GDA may (or may not) do well.

To summarize: GDA makes stronger modeling assumptions, and is more data efficient (i.e., requires less training data to
learn "well") when the modeling assumptions are correct or at least approximately correct. Logistic regression makes
weaker assumptions, and is significantly more robust to deviations from modeling assumptions. Specifically, when the
data is indeed non-Gaussian, then in the limit of large datasets, logistic regression will almost always do better than
GDA. For this reason, in practice logistic regression is used more often than GDA. (Some related considerations about
discriminative vs. generative models also apply for the Naive Bayes algorithm that we discuss next, but the Naive Bayes
algorithm is still considered a very good, and is certainly also a very popular, classification algorithm.)

### 2 Naive Bayes

In GDA, the feature vectors were continuous, real-valued vectors. Let's now talk about a different learning algorithm in
which the $x_{i}$ are discrete-valued.

For our motivating example, consider building an email spam filter using machine learning. Here, we wish to classify
messages according to whether they are unsolicited commercial (spam) email, or non-spam email. After learning to do
this, we can then have our mail reader automatically filter out the spam messages and perhaps place them in a separate
mail folder.

Classifying emails is one example of a broader set of problems called text classification. Let's say we have a training
set (a set of emails labeled as spam or non-spam). We'll begin our construction of our spam filter by specifying the
features $x_{i}$ used to represent an email.

We will represent an email via a feature vector whose length is equal to the number of words in the dictionary.
Specifically, if an email contains the i-th word of the dictionary, then we will set $x_{i}=1$; otherwise, we
let $x_{i}=0$. For instance, the vector

$$x = \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 1 \\ 0 \end{bmatrix} \begin{matrix} \text{a} \\ \text{aardvark} \\ \text{aardwolf} \\ \vdots \\ \text{buy} \\ \text{zygmurgy} \end{matrix}$$

is used to represent an email that contains the words "a" and "buy," but not "aardvark," "aardwolf" or "zygmurgy." The
set of words encoded into the feature vector is called the vocabulary, so the dimension of $x$ is equal to the size of
the vocabulary.

Having chosen our feature vector, we now want to build a generative model. So, we have to model $p(x|y)$. But if we
have, say, a vocabulary of 50000 words, then $x\in\{0,1\}^{50000}$ (x is a 50000-dimensional vector of 0's and 1's), and
if we were to model $p(x|y)$ explicitly with a multinomial distribution over the $2^{50000}$ possible outcomes, then
we'd end up with a $(2^{50000}-1)$-dimensional parameter vector. This is clearly too many parameters.

To model $p(x|y)$, we will therefore make a very strong assumption. We will assume that the $x_{i}$ are conditionally
independent given y. This assumption is called the Naive Bayes (NB) assumption, and the resulting algorithm is called
the Naive Bayes classifier. For instance, if $y=1$ means spam email; "buy" is word 2087 and "price" is word 39831; then
we are assuming that if I tell you $y=1$ (that a particular piece of email is spam), then knowledge of $x_{2087}$ (
knowledge of whether "buy" appears in the message) will have no effect on your beliefs about the value of $x_{39831}$ (
whether "price" appears). More formally, this can be written $p(x_{2087}|y)=p(x_{2087}|y,x_{39831})$. (Note that this is
not the same as saying that $x_{2087}$ and $x_{39831}$ are independent, which would have been
written $p(x_{2087})=p(x_{2087}|x_{39831})$; rather, we are only assuming that $x_{2087}$ and $x_{39831}$ are
conditionally independent given y.)

We now have:
$$p(x_{1},...,x_{50000}|y)$$
$$=p(x_{1}|y)p(x_{2}|y,x_{1})p(x_{3}|y,x_{1},x_{2})\cdot\cdot\cdot p(x_{50000}|y,x_{1},...,x_{4999})$$
$$=p(x_{1}|y)p(x_{2}|y)p(x_{3}|y)\cdot\cdot\cdot p(x_{50000}|y)$$
$$=\prod_{i=1}^{n}p(x_{i}|y)$$

The first equality simply follows from the usual properties of probabilities, and the second equality used the NB
assumption. We note that even though the Naive Bayes assumption is an extremely strong assumptions, the resulting
algorithm works well on many problems.

Our model is parameterized by $\phi_{i|y=1}=p(x_{i}=1|y=1)$, $\phi_{i|y=0}=p(x_{i}=1|y=0)$, and $\phi_{y}=p(y=1)$. As
usual, given a training set $\{(x^{(i)},y^{(i)}); i=1,...,m\}$ we can write down the joint likelihood of the data:

$$\mathcal{L}(\phi_{y},\phi_{j|y=0},\phi_{j|y=1})=\prod_{i=1}^{m}p(x^{(i)},y^{(i)}).$$

Maximizing this with respect to $\phi_{y}$, $\phi_{i|y=0}$ and $\phi_{i|y=1}$ gives the maximum likelihood estimates:

$$\phi_{j|y=1}=\frac{\sum_{i=1}^{m}1\{x_{j}^{(i)}=1\wedge y^{(i)}=1\}}{\sum_{i=1}^{m}1\{y^{(i)}=1\}}$$
$$\phi_{j|y=0}=\frac{\sum_{i=1}^{m}1\{x_{j}^{(i)}=1\wedge y^{(i)}=0\}}{\sum_{i=1}^{m}1\{y^{(i)}=0\}}$$
$$\phi_{y}=\frac{\sum_{i=1}^{m}1\{y^{(i)}=1\}}{m}$$

In the equations above, the "$\wedge$" symbol means "and." The parameters have a very natural interpretation. For
instance, $\phi_{j|y=1}$ is just the fraction of the spam $(y=1)$ emails in which word j does appear.

Having fit all these parameters, to make a prediction on a new example with features $x$, we then simply calculate

$$p(y=1|x)=\frac{p(x|y=1)p(y=1)}{p(x)}$$
$$=\frac{(\prod_{i=1}^{n}p(x_{i}|y=1))p(y=1)}{(\prod_{i=1}^{n}p(x_{i}|y=1))p(y=1)+(\prod_{i=1}^{n}p(x_{i}|y=0))p(y=0)},$$

and pick whichever class has the higher posterior probability.

Lastly, we note that while we have developed the Naive Bayes algorithm mainly for the case of problems where the
features $x_{i}$ are binary-valued, the generalization to where $x_{i}$ can take values in $\{1,2,...,k_{i}\}$ is
straightforward. Here, we would simply model $p(x_{i}|y)$ as multinomial rather than as Bernoulli. Indeed, even if some
original input attribute (say, the living area of a house, as in our earlier example) were continuous valued, it is
quite common to discretize it that is, turn it into a small set of discrete values and apply Naive Bayes. For instance,
if we use some feature $x_{i}$ to represent living area, we might discretize the continuous values as follows:

| Living area (sq. feet) | <400 | 400-800 | 800-1200 | 1200-1600 | >1600 |
|:-----------------------|:-----|:--------|:---------|:----------|:------|
| $T_i$                  | 1    | 2       | 3        | 4         | 5     |

Thus, for a house with living area 890 square feet, we would set the value of the corresponding feature $x_{i}$ to 3. We
can then apply the Naive Bayes algorithm, and model $p(x_{i}|y)$ with a multinomial distribution, as described
previously. When the original, continuous-valued attributes are not well-modeled by a multivariate normal distribution,
discretizing the features and using Naive Bayes (instead of GDA) will often result in a better classifier.

#### 2.1 Laplace smoothing

The Naive Bayes algorithm as we have described it will work fairly well for many problems, but there is a simple change
that makes it work much better, especially for text classification. Let's briefly discuss a problem with the algorithm
in its current form, and then talk about how we can fix it.

Consider spam/email classification, and let's suppose that, after completing CS229 and having done excellent work on the
project, you decide around June 2003 to submit the work you did to the NIPS conference for publication. (NIPS is one of
the top machine learning conferences, and the deadline for submitting a paper is typically in late June or early July.)
Because you end up discussing the conference in your emails, you also start getting messages with the word "nips" in it.
But this is your first NIPS paper, and until this time, you had not previously seen any emails containing the word "
nips"; in particular "nips" did not ever appear in your training set of spam/non-spam emails. Assuming that "nips" was
the 35000th word in the dictionary, your Naive Bayes spam filter therefore had picked its maximum likelihood estimates
of the parameters $\phi_{35000|y}$ to be

$$\phi_{35000|y=1}=\frac{\sum_{i=1}^{m}1\{x_{35000}^{(i)}=1\wedge y^{(i)}=1\}}{\sum_{i=1}^{m}1\{y^{(i)}=1\}}=0$$
$$\phi_{35000|y=0}=\frac{\sum_{i=1}^{m}1\{x_{35000}^{(i)}=1\wedge y^{(i)}=0\}}{\sum_{i=1}^{m}1\{y^{(i)}=0\}}=0$$

I.e., because it has never seen "nips" before in either spam or non-spam training examples, it thinks the probability of
seeing it in either type of email is zero. Hence, when trying to decide if one of these messages containing "nips" is
spam, it calculates the class posterior probabilities, and obtains

$$p(y=1|x)=\frac{\prod_{i=1}^{n}p(x_{i}|y=1)p(y=1)}{\prod_{i=1}^{n}p(x_{i}|y=1)p(y=1)+\prod_{i=1}^{n}p(x_{i}|y=0)p(y=0)}$$
$$=\frac{0}{0}$$

This is because each of the terms $\prod_{i=1}^{n}p(x_{i}|y)$ includes a term $p(x_{35000}|y)=0$ that is multiplied into
it. Hence, our algorithm obtains $0/0$, and doesn't know how to make a prediction.

Stating the problem more broadly, it is statistically a bad idea to estimate the probability of some event to be zero
just because you haven't seen it before in your finite training set. Take the problem of estimating the mean of a
multinomial random variable z taking values in {1,..., k}. We can parameterize our multinomial with $\phi_{i}=p(z=i)$.
Given a set of m independent observations $\{z^{(1)},...,z^{(m)}\}$, the maximum likelihood estimates are given by

$$\phi_{j}=\frac{\sum_{i=1}^{m}1\{z^{(i)}=j\}}{m}.$$

As we saw previously, if we were to use these maximum likelihood estimates, then some of the $\phi_{j}$'s might end up
as zero, which was a problem. To avoid this, we can use Laplace smoothing, which replaces the above estimate with

$$\phi_{j}=\frac{\sum_{i=1}^{m}1\{z^{(i)}=j\}+1}{m+k}.$$

Here, we've added 1 to the numerator, and k to the denominator. Note that $\sum_{j=1}^{k}\phi_{j}=1$ still holds (check
this yourself!), which is a desirable property since the $\phi_{j}$ 's are estimates for probabilities that we know must
sum to 1. Also, $\phi_{j}\ne0$ for all values of j, solving our problem of probabilities being estimated as zero. Under
certain (arguably quite strong) conditions, it can be shown that the Laplace smoothing actually gives the optimal
estimator of the $\phi_{j}$ s.

Returning to our Naive Bayes classifier, with Laplace smoothing, we therefore obtain the following estimates of the
parameters:

$$\phi_{j|y=1}=\frac{\sum_{i=1}^{m}1\{x_{j}^{(i)}=1\wedge y^{(i)}=1\}+1}{\sum_{i=1}^{m}1\{y^{(i)}=1\}+2}$$
$$\phi_{j|y=0}=\frac{\sum_{i=1}^{m}1\{x_{j}^{(i)}=1\wedge y^{(i)}=0\}+1}{\sum_{i=1}^{m}1\{y^{(i)}=0\}+2}$$

(In practice, it usually doesn't matter much whether we apply Laplace smoothing to $\phi_{y}$ or not, since we will
typically have a fair fraction each of spam and non-spam messages, so $\phi_{y}$ will be a reasonable estimate
of $p(y=1)$ and will be quite far from 0 anyway.)

#### 2.2 Event models for text classification

To close off our discussion of generative learning algorithms, let's talk about one more model that is specifically for
text classification. While Naive Bayes as we've presented it will work well for many classification problems, for text
classification, there is a related model that does even better.

In the specific context of text classification, Naive Bayes as presented uses the what's called the multi-variate
Bernoulli event model. In this model, we assumed that the way an email is generated is that first it is randomly
determined (according to the class priors $p(y)$) whether a spammer or non-spammer will send you your next message.
Then, the person sending the email runs through the dictionary, deciding whether to include each word i in that email
independently and according to the probabilities $p(x_{i}=1|y)=\phi_{i|y}$. Thus, the probability of a message was given
by $p(y)\prod_{i=1}^{n}p(x_{i}|y)$.

Here's a different model, called the multinomial event model. To describe this model, we will use a different notation
and set of features for representing emails. We let $x_{i}$ denote the identity of the i-th word in the email.
Thus, $x_{i}$ is now an integer taking values in $\{1,...,|V|\}$ where |V| is the size of our vocabulary (dictionary).
An email of n words is now represented by a vector $(x_{1},x_{2},...,x_{n})$ of length n; note that n can vary for
different documents. For instance, if an email starts with "A NIPS...." then $x_{1}=1$ ("a" is the first word in the
dictionary), and $x_{2}=35000$ (if "nips" is the 35000th word in the dictionary).

In the multinomial event model, we assume that the way an email is generated is via a random process in which
spam/non-spam is first determined (according to $p(y)$) as before. Then, the sender of the email writes the email by
first generating $x_{1}$ from some multinomial distribution over words $(p(x_{1}|y))$. Next, the second word $x_{2}$ is
chosen independently of $x_{1}$ but from the same multinomial distribution, and similarly for $x_{3}$, $x_{4}$, and so
on, until all n words of the email have been generated. Thus, the overall probability of a message is given
by $p(y)\prod_{i=1}^{n}p(x_{i}|y)$.

Note that this formula looks like the one we had earlier for the probability of a message under the multi-variate
Bernoulli event model, but that the terms in the formula now mean very different things. In particular $x_{i}|y$ is now
a multinomial, rather than a Bernoulli distribution.

The parameters for our new model are $\phi_{y}=p(y)$ as before, $\phi_{k|y=1}=p(x_{j}=k|y=1)$ (for any j)
and $\phi_{i|y=0}=p(x_{j}=k|y=0)$. Note that we have assumed that $p(x_{j}|y)$ is the same for all values of j (i.e.,
that the distribution according to which a word is generated does not depend on its position j within the email).

If we are given a training set $\{(x^{(i)},y^{(i)}); i=1,...,m\}$
where $x^{(i)}=(x_{1}^{(i)},x_{2}^{(i)},...,x_{n_{i}}^{(i)})$ (here, $n_{i}$ is the number of words in the i-training
example), the likelihood of the data is given by

$$\mathcal{L}(\phi,\phi_{k|y=0},\phi_{k|y=1})=\prod_{i=1}^{m}p(x^{(i)},y^{(i)})$$
$$=\prod_{i=1}^{m}(\prod_{j=1}^{n_{i}}p(x_{j}^{(i)}|y;\phi_{k|y=0},\phi_{k|y=1}))p(y^{(i)};\phi_{y}).$$

Maximizing this yields the maximum likelihood estimates of the parameters:

$$\phi_{k|y=1}=\frac{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}1\{x_{j}^{(i)}=k\wedge y^{(i)}=1\}}{\sum_{i=1}^{m}1\{y^{(i)}=1\}n_{i}}$$
$$\phi_{k|y=0}=\frac{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}1\{x_{j}^{(i)}=k\wedge y^{(i)}=0\}}{\sum_{i=1}^{m}1\{y^{(i)}=0\}n_{i}}$$
$$\phi_{y}=\frac{\sum_{i=1}^{m}1\{y^{(i)}=1\}}{m}.$$

If we were to apply Laplace smoothing (which needed in practice for good performance) when estimating $\phi_{k|y=0}$
and $\phi_{k|y=1}.$ we add 1 to the numerators and $|V|$ to the denominators, and obtain:

$$\phi_{k|y=1}=\frac{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}1\{x_{j}^{(i)}=k\wedge y^{(i)}=1\}+1}{\sum_{i=1}^{m}1\{y^{(i)}=1\}n_{i}+|V|}$$
$$\phi_{k|y=0}=\frac{\sum_{i=1}^{m}\sum_{j=1}^{n_{i}}1\{x_{j}^{(i)}=k\wedge y^{(i)}=0\}+1}{\sum_{i=1}^{m}1\{y^{(i)}=0\}n_{i}+|V|}.$$

While not necessarily the very best classification algorithm, the Naive Bayes classifier often works surprisingly well.
It is often also a very good "first thing to try," given its simplicity and ease of implementation.