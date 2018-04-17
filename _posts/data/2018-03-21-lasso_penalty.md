---
layout: post
title: Useful Variations on the Lasso Penalty
---

<hr>
The lasso is a regression method in which we apply an $\ell_1$ penalty to the regression coefficients. It's useful because it performs feature selection: the lasso will only estimate the parameters for the regressors it likes, while the rest get set to zero. However, we might not always want to apply a lasso penalty uniformly - or even at all - to some coefficients. In this post, I'll detail how to rewrite those cases into a vanilla lasso problem. 

<h2 align="center">Setup</h2>

To be clear, let's suppose we have the $T \times N$ design matrix $\mathbf{X}$ consisting of $T$ observations of $N$ features. Furthermore, our dependent variable is denoted by the $T\times 1$ vector $\mathbf{y}$ while the the $N\times 1$ vector $\boldsymbol{\beta}$ contains the parameters to be estimated. Denoting the $\ell_p$ norm as , the lasso objective can be written as

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \lambda |\boldsymbol{\beta}|_1\right\\}
\end{align}

where $\lambda$ is a hyperparameter that we usually choose during cross-validation. It tells us how strongly we should enforce sparsity when choosing $\boldsymbol{\beta}$. 

Right now, we've written the problem such that the same penalty term is applied across all the coefficients. Sometimes, though, we'll have groups of regressors which we know have different levels of sparsity. To be general, let's suppose we want to apply a different regularization strength $\lambda_i$ to each coefficient $\beta_i$. The optimization procedure in this case is 

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \sum\_{i=1}^N \lambda_i |\beta_i|\right\\} \\\\\\
&= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + |\boldsymbol{\Lambda} \boldsymbol{\beta}|_1\right\\}.
\end{align}

where $\boldsymbol{\Lambda} = \text{diag}\left(\lambda_1, \lambda_2, \ldots, \lambda_N\right)$. 

<hr>
<h2 align="center">Case 1: Non-zero Penalties</h2>
<hr>

For now, let's assume that $\lambda_i>0$ for all $i$. In the above optimization problem, define $\boldsymbol{\beta}' = \boldsymbol{\Lambda} \boldsymbol{\beta}$, so that

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + |\boldsymbol{\beta}'|_1\right\\}.
\end{align}

This is starting to look like a vanilla Lasso: the pesky $\boldsymbol{\Lambda}$ has been absorbed into the parameters. A consequence of this is that the "new" penalty term is simply equal to one. We're not done yet, as we need to rewrite $\boldsymbol{\beta}$ in the reconstruction term. To do so, we need a $\boldsymbol{\Lambda}$: it's not immediately available, but we can concoct one as follows:

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\Lambda}^{-1} \boldsymbol{\Lambda}\boldsymbol{\beta}|^2_2 + |\boldsymbol{\beta}'|_1\right\\} \\\\\
&=\underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\Lambda}^{-1} \boldsymbol{\beta}'|^2_2 + |\boldsymbol{\beta}'|_1\right\\} \\\\\
&= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}' \boldsymbol{\beta}'|^2_2 + |\boldsymbol{\beta}'|_1\right\\},
\end{align}

where $\mathbf{X}' = \mathbf{X}\boldsymbol{\Lambda}^{-1}$. What we're left with is a vanilla Lasso problem! The procedure is simple: for some choice of $\boldsymbol{\Lambda}$, transform your design matrix $\mathbf{X} \rightarrow \mathbf{X}\boldsymbol{\Lambda}^{-1}$ and run an ordinary Lasso with regularization term set to one. 

This approach fails, however, if any of the $\lambda_i$ are equal to zero because $\boldsymbol{\Lambda}$ becomes singular. In this case, we need to take a different approach.

<hr>
<h2 align="center">Case 2: Unpenalized Coefficients</h2> 
<hr>

Let's now consider the case when some of the $\lambda_i$ are equal to zero. This implies that a subset of the $\beta_i$ are unpenalized in the regression. To make this explicit, let's split the $\beta_i$ into two sets of coefficients: those that are penalized $\boldsymbol{\beta}\_{\text{P}}$, and those that aren't $\boldsymbol{\beta}\_{\text{NP}}$. We refer to the corresponding penalties on $\boldsymbol{\beta}\_{\text{P}}$ as $\boldsymbol{\Lambda}\_{\text{P}}$. 

Thus, the optimization problem can be written as 

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{P}}, \hat{\boldsymbol{\beta}}\_{\text{NP}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}, \  \hat{\boldsymbol{\beta}}\_{\text{NP}}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\_{\text{NP}}\boldsymbol{\beta}\_{\text{NP}} + \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\|^2_2 + |\boldsymbol{\beta}\_{\text{P}}|_1\right\\} \\\\\
x^2 &= 5
\end{align}



