---
layout: post
title: Useful Transformations for the Application of a Lasso Penalty
---

The lasso is a regression method in which we apply an $\ell_1$ penalty to the . It's useful because it performs feature selection - it'll only estimate the regressors it likes, and sets the corresponding parameters for the rest to zero. However, we might not always want to apply a lasso penalty uniformly - or even at all - to some parameters in the problem. In this post, I'll detail how to rewrite those cases into a vanilla lasso problem. 

To be clear, let's suppose we have the $T \times N$ design matrix $\mathbf{X}$ consisting of $T$ observations of $N$ features. We also have the . The lasso objective is 

$$\hat{\boldsymbol{\beta}} = \frac{1}{N}|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \lambda |\boldsymbol{\beta}|_1$$



Typically, we

