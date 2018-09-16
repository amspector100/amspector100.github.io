---
layout: single
title:  "What is Maximimum Likelihood Estimation?"
categories:
  - ML
tags:
  - preliminary
date:   2018-09-16
class: wide
---


## Intuition and Introduction

*Maximum Likelihood Estimation*, or MLE, is a technique for guessing *unknown parameters* for *models* of observed data. Specifically, the *Maximum Likelihood Estimator* for an unknown parameter is the value maximizes the probability of the observed data. To understand how this works intuitively, consider the following example. (You could also just skip straight to the [math](#math)). 

Imagine you flip a penny 100 times, and it lands heads every time. You probably have some internal model of how coin flips work - for example, it's reasonable to assume that each toss is independent of the other coin tosses. However, there's a key parameter in this model you're missing: for a weighted coin, you don't know the probability $p$ that any individual toss will come up as heads. However, you've observed that the penny landed heads 100 times in a row, so you infer that $p$ is pretty close to $1$, because that makes the observed data more probable. This is a simple example of MLE.

## <a name="math"></a> Math

### Finding Maximum Likelihood Estimators

>  You take the log so fast that you don’t even see the actual data. — Andrew Gelman

### Connection to Information Theory and 

## Fun Examples