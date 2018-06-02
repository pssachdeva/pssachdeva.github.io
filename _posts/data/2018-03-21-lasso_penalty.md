---
layout: post
title: Useful Variations on the Lasso Penalty
---

<hr>
The lasso is a regression method in which we apply an $\ell_1$ penalty to the regression coefficients. It's useful because it performs feature selection: the lasso will only estimate the parameters for the regressors it likes, while the rest get set to zero. However, we might not always want to apply a lasso penalty uniformly - or even at all - to some coefficients. In this post, I'll detail how to rewrite those cases into a vanilla lasso problem. 

<hr class="rule-header-top">
<h2 align="center">Setup</h2>
<hr class="rule-header-bottom">

To be clear, let's suppose we have the $T \times N$ design matrix $\mathbf{X}$ consisting of $T$ observations of $N$ features. Furthermore, our dependent variable is denoted by the $T\times 1$ vector $\mathbf{y}$ while the the $N\times 1$ vector $\boldsymbol{\beta}$ contains the parameters to be estimated. Denoting the $\ell_p$ norm as $\vert\cdot\vert\_p$, the lasso objective can be written as

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

<hr class="rule-header-top">
<h2 align="center">Case 1: Non-zero Penalties</h2>
<hr class="rule-header-bottom">

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

<hr class="rule-header-top">
<h2 align="center">Case 2: Unpenalized Coefficients</h2> 
<hr class="rule-header-bottom">

Let's now consider the case when some of the $\lambda_i$ are equal to zero. This implies that a subset of the $\beta_i$ are unpenalized in the regression. To make this explicit, let's split the $\beta_i$ into two sets of coefficients: those that are penalized $\boldsymbol{\beta}\_{\text{P}}$, and those that aren't $\boldsymbol{\beta}\_{\text{NP}}$; their respective design matrices are $\mathbf{X}\_{\text{P}}$ and $\mathbf{X}\_{\text{NP}}$. Lastly, we refer to the corresponding penalties on $\boldsymbol{\beta}\_{\text{P}}$ as $\boldsymbol{\Lambda}\_{\text{P}}$. 

Thus, the optimization problem can be written as 

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{P}}, \hat{\boldsymbol{\beta}}\_{\text{NP}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}, \  \hat{\boldsymbol{\beta}}\_{\text{NP}}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\_{\text{NP}}\boldsymbol{\beta}\_{\text{NP}} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\|^2_2 + |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\right\\} 
\end{align}

Now, let's think about this like coordinate descent: suppose we already have a putative $\boldsymbol{\beta}\_{\text{P}}$, and we need to compute $\boldsymbol{\beta}\_{\text{NP}}$. Then, the optimization procedure becomes 

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{NP}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{NP}}}{\operatorname{argmin}} \left\\{\left|\mathbf{y}- \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}} - \mathbf{X}\_{\text{NP}}\boldsymbol{\beta}\_{\text{NP}}\right|^2_2 \right\\} 
\end{align}

where we've removed the penalty term since we're no longer optimizing for $\boldsymbol{\beta}\_{\text{P}}$. This optimization procedure is nothing other than a linear regression of the residuals $\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}$ on the non-penalized regressors $\mathbf{X}\_{\text{NP}}$. Thus, the solution is simply that of standard linear regression:

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{NP}} &= \left(\mathbf{X}\_{\text{NP}}^T\mathbf{X}\_{\text{NP}}\right)^{-1} \mathbf{X}\_{\text{NP}}^T \left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right).
\end{align}

Thus, everytime we have a guess at the optimal penalized parameters $\boldsymbol{\beta}\_{\text{P}}$, we already have a closed-form solution for  the non-penalized parameters $\boldsymbol{\beta}\_{\text{NP}}$. We can just go ahead and toss this closed-form expression in the original optimization procedure, and now optimize for $\boldsymbol{\beta}\_{\text{P}}$:

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{P}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}}{\operatorname{argmin}} \left\\{|\mathbf{y} - \mathbf{X}\_{\text{NP}}\left[\left(\mathbf{X}\_{\text{NP}}^T\mathbf{X}\_{\text{NP}}\right)^{-1} \mathbf{X}\_{\text{NP}}^T \left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right)\right] - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\|^2_2 \right. \\\\\
& \qquad \qquad \qquad \left. + |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\right\\} 
\\\\\
&= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}}{\operatorname{argmin}} \left\\{\left|\left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right) - \mathbf{P}\_{\text{NP}}\left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right)\right|^2_2+ |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\right\\} \\\\\
&= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}}{\operatorname{argmin}} \Big\\{\left|\mathbf{M}\_{\text{NP}}\mathbf{y} - \mathbf{M}\_{\text{NP}}\mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right|^2_2+ |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\Big\\},
\end{align}

where $\mathbf{P}\_{\text{NP}}$ is the projection matrix:

\begin{align}
\mathbf{P}\_{\text{NP}} &= \mathbf{X}\_{\text{NP}}\left(\mathbf{X}\_{\text{NP}}^T\mathbf{X}\_{\text{NP}}\right)^{-1} \mathbf{X}\_{\text{NP}}^T
\end{align}

and $\mathbf{M}\_{\text{NP}}$ is its corresponding residual matrix:

\begin{align}
\mathbf{M}\_{\text{NP}} &= \mathbf{I} - \mathbf{P}\_{\text{NP}}
\end{align}

ruff $\Big\\{\Big\\}$
