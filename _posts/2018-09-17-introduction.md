---
layout: single
toc: true

--- 
# Motivation

> All models are wrong, but some are useful. — George E. P. Box

During the winter of 2018, I decided to work through the [SKLearn
library](http://scikit-learn.org/) in Python and try to rigorously understand
most, if not all, of the algorithms implemented in it. To my surprise, I found
that it was rather difficult to find comprehensive explorations of the
algorithms which were also accessible to non-experts; most blogs skimmed over
the mathematical underpinnings of machine learning, and most papers presupposed
great familiarity with the field.

This section of my blog is devoted to exploring ML in a way that is
comprehensive and rigorous, but still practical and accessible to a relatively
broad audience. Obviously, not every reader will be interested in every aspect
of each post. You might want to simply gain a practical understanding of when to
use a certain clustering algorithm; or you might want to learn why expectation-
maximization optimization really works. However, I'm hoping that most people
will find *something* interesting in these posts.

I think there are at least two reasons that it's worth deeply understanding ML
algorithms:

1. First, it's fun! The math behind statistical inference and machine learning
is *really* cool.
2. More practically, it will allow you to write more effective code. It's much
easier to figure out why your model isn't working if you actually understand how
your model works.

# Post Structure

 In order to let readers easily jump around, I will loosely divide each post
into the following four sections:

1. Motivation and Intuition
2. Mathematical Derivation
3. Implementation from Scratch (i.e without using machine learning Libraries -
this is to make sure we really understand how each algorithms works)
4. Practical Use (i.e. using libraries like SKLearn, Tensorflow, Pytorch)

It's also worth noting that in both the 1st and 4th sections, I will focus
extensively on use cases of each of the algorithms - i.e. the kinds of
situations in which you'd use a Naives Bayes Classifier over a SVN or vice
versa. Admittedly, not every post will follow this exact structure, but each
post should generally cover these kinds of information.

# Blog Structure



# Background

If you're just starting to learn about Data Science, here are a couple pieces of
background that you might find handy.

## Math

## Modeling

### Generative Models

A **model** is a set of assumptions (and, usually, equations) which frame the
way the world works. For example, you might model a series of dice rolls in a
board game by assuming they are independent from each other - in other words,
the outcome of the first rolls does not affect the outcome of the second roll.

However, we'll usually be interested in **generative models.** A generative
model with respect to some observed data is a model which is capable of *fully
simulating* the observed data. For example, if you observed a bunch of dice
rolls in a row, as follows:

$$ 1, \, 3, \, 4, \, 6, \, 2, \, 5, \, 5, \, 3, \, 1, \, 6$$

just knowing that each roll is independent of the others is *not* enough to
simulate the data. To do that, you'd need to also to specify the *probability
distribution* of the value of each roll - in other words, the probability that
any roll will land as a $1$ or a $2$ or a $3$, etc.

We like generative models for at least two reasons! First, they let us do fun
(and useful) math - for example, we can estimate the probability that the
observed data occurs under a model. As we'll see, calculating the likelihood of
observed data is extremely important in techniques like Maximum Likelihood
Estimation and more. Second, generative models can also *do* cool things: for
example, a generative model for natural language can [write a chapter of Harry
Potter](https://medium.com/deep-writing/harry-potter-written-by-artificial-
intelligence-8a9431803da6) or even create [fake pictures of real
celebrities](https://www.youtube.com/watch?v=VrgYtFhVGmg).

### The Data Generating Process

The **Data Generating Process** (DGP) is the "true" generative model. To be more
specific, in most problems we assume there is some underlying joint probability
distribution which governs the data we observe, and the DGP is that underlying
distribution. The DGP is a bit like the government in this sense - we can never
know exactly what it's doing internally, but we can use external data to get a
rough sense of what's going on.

In the example in the next section, which will tie together all of this
material, we have a "God's eye view" and can see the DGP in all its glory, but
only because I literally made up the data. In reality, you will never know the
DGP - the point of ML is broadly to create models which approximate it. 
 
### Parametric and Nonparametric Models

Like people, models tend to come in families. For example, the normal
distribution is not a single distribution - it's a family of extremely similar
distributions, each of which depends on two values: a *mean* and *variance.* We
generally call values which help index families of models *parameters*.

Unfortunately, we don't usually know the values of parameters we're interested
in. As a result, we have to *estimate* the parameter

<div class = 'notice--warning'> Warning: the term "estimator" is extremely confusing. The key point to remember is that an <strong> estimate </strong> is nonrandom, whereas an <strong> estimator </strong> is a function of
random data which serves as a guess for a parameter of interest. </div>
 


{% highlight python %}
import numpy as np
np.random.seed(210)

# Simulate data w random noise
true_lambda = 457 + np.pi 
data = np.random.poisson(true_lambda, size = 1000) 
data = data + 50 * np.random.randn(size = 1000) 
{% endhighlight %}
 
### Supervised and Unsupervised Learning

### Overfitting

# Last Notes

I'm always learning, and I'm sure I will make mistakes in the blog. If you find
inaccuracies in my posts, please let me know either by [opening an issue on
GitHub](https://github.com/amspector100/amspector100.github.io) (preferred) or
emailing me at amspector100@gmail.com. 
