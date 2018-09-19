---
layout: single
title:  "Machine Learning Blog: An Introduction"
categories:
  - ML
tags:
  - preliminary
date:   2018-09-07
class: wide
published: false
---

## Motivation 

> All models are wrong, but some are useful. — George E. P. Box

During the winter of 2018, I decided to work through the [SKLearn library](http://scikit-learn.org/) in Python and try to rigorously understand most, if not all, of the algorithms implemented in it. To my surprise, I found that it was rather difficult to find comprehensive explorations of the algorithms which were also accessible to non-experts; most blogs skimmed over the mathematical underpinnings of machine learning, and most papers presupposed great familiarity with the field. 

This section of my blog is devoted to exploring ML in a way that is comprehensive and rigorous, but still practical and accessible to a relatively broad audience. Obviously, not every reader will be interested in every aspect of each post. You might want to simply gain a practical understanding of when to use a certain clustering algorithm; or you might want to learn why expectation-maximization optimization really works. However, I'm hoping that most people will find *something* interesting in these posts. 

I think there are at least two reasons that it's worth deeply understanding ML algorithms:

1. First, it's fun! The math behind statistical inference and machine learning is *really* cool. 
2. More practically, it will allow you to write more effective code. It's much easier to figure out why your model isn't working if you actually understand how your model works. 

## Blog Structure

The ML algorithms which currently work generally solve the following sort of problem: given a bunch of data points $ x_1, x_2, \dots, x_n $, assign some features $y$ to each input $x_i$. These inputs $x$ could be anything from numbers and vectors to sound waves and images, and the $y$ could be equally diverse. Moreover, there are quite a large number of models devoted to solving these sorts of problems, from regressions and clustering to deep neural nets. 

There are a lot of other types of problems in machine learning: for example, there might be some latent variables which affect the output that you must learn from the input. For example, if you're trying to train a model to read text in a human-like, emotive voice, your model will probably have to infer the *emotional content* from the text. However, (generally speaking), 


## Post Structure

 In order to let readers easily jump around, I will loosely divide each post into the following four sections:

1. Intuition and Motivation
2. Mathematical Derivation
3. Implementation from Scratch (i.e without using machine learning Libraries - this is to make sure we really understand how each algorithms works)
4. Practical Use (i.e. using libraries like SKLearn, Tensorflow, Pytorch)

It's also worth noting that in both the 1st and 4th sections, I will focus extensively on use cases of each of the algorithms - i.e. the kinds of situations in which you'd use a Naives Bayes Classifier over a SVN or vice versa. Admittedly, not every post will follow this exact structure, but each post should generally cover these kinds of information. 

## Background

We generally divide these algorithms into two classes: supervised and unsupervised learning algorithms.

### Supervised Learning Algorithms

Supervised learning algorithms use *training data* to learn patterns and then make new *predictions* on other data. For example, we might give a supervised learning model a set of 2d points with colors associated to them, and then ask it to predict the colors of another two points. 

### Unsupervised Learning Algorithms

Unsupervised algorithms are designed to automatically detect patterns in data without requiring any training data. For example, the following algorithm (called a Gaussian Mixture Model, or GMM) can take the following data as an input:

{% raw %}![](/assets/images/ML/unsup_init_data.png){% endraw %}

and the GMM will automatically cluster it into something like the following:

{% raw %}![](/assets/images/ML/gmm_output.png){% endraw %}

The GMM did not require any training data; 

There are basically two types of machine learning that work *really well* today in machine learning. First, there are **unsupervised learning algorithms.**


My plan is to cover the following three topics (each of which will require a series of posts). Of course, you don't need to read the posts in order - it just might be helpful! :) 

### Unsupervised Learning Models



### A Quick Review: Statistical Inference

Statistical Inference broadly refers to the process of *generating a model* from *observed data*. 

My initial posts will center on the canonical tools of inferential statistics, including things like general linear models, maximum-likelihood estimators, and Bayesian approaches like MAP estimators (you may not know what these are yet). I'm starting with these kinds of models because they are really the building blocks of machine learning: reviewing statistical inference will help develop both a theoretical framework and practical tools which we can use to create more complicated models.

### Supervised Learning Methods


## Last Notes

I'm always learning, and I'm sure I will make mistakes in the blog. If you find inaccuracies in my posts, please let me know either by [opening an issue on GitHub](https://github.com/amspector100/amspector100.github.io) or emailing me at amspector100@gmail.com. 

<!-- To do: 
(1) Make clear I'll go over use cases and hyper-parameter tuning and such. 
(2) Make an outline of the algorithms I'm going to cover. Explain why I'm covering these topics.  
(3) Subscription to specific categories


 --> 