---
layout: post
title: Useful Variations on the Lasso Penalty
excerpt: Transforming a lasso regression with different penalties into a vanilla lasso regression.
tags: [Statistics]
---
<hr class="rule-header-title-top">
<h1 align="center">{{page.title}}</h1>
<hr class="rule-header-title-bottom">
The lasso is a regression method in which an $\ell_1$ penalty is applied to the regression coefficients. It's useful because it performs feature selection: the lasso will only estimate the parameters for the regressors it likes, while the rest get set to zero. However, we might not always want to apply a lasso penalty uniformly across the coefficients. Instead, we might want to double the lasso penalty for some coefficients, or even turn it off completely for others. In this post, I'll detail how to rewrite those cases into a vanilla lasso problem. 

<hr class="rule-header-top">
<h2 align="center">Setup</h2>
<hr class="rule-header-bottom">

To be clear, let's suppose we have the $T \times N$ design matrix $\mathbf{X}$ consisting of $T$ observations of $N$ features. Furthermore, the response variable is denoted by the $T\times 1$ vector $\mathbf{y}$ while the $N\times 1$ vector $\boldsymbol{\beta}$ contains the parameters to be estimated. Denoting the $\ell_p$ norm as $\vert\cdot\vert\_p$, the lasso objective can be written as

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \lambda |\boldsymbol{\beta}|_1\Big\\},
\end{align}

where $\lambda$ is a hyperparameter that we usually choose during cross-validation. It tells us how strongly we should enforce sparsity when choosing $\boldsymbol{\beta}$. 

Right now, we've written the problem such that the same penalty term is applied across all the coefficients. Sometimes, though, we'll want to enforce different levels of sparsity to different parameters. To be general, let's suppose we want to apply regularization strength $\lambda_i$ to each coefficient $\beta_i$. The optimization procedure becomes

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \sum\_{i=1}^N \lambda_i |\beta_i|\Big\\} \\\\\\
&= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + |\boldsymbol{\Lambda} \boldsymbol{\beta}|_1\Big\\}.
\end{align}

where $\boldsymbol{\Lambda} = \text{diag}\left(\lambda_1, \lambda_2, \ldots, \lambda_N\right)$. Our goal, then, is to rewrite equation (3) in a similar manner as equation (1), with only one penalty.

<hr class="rule-header-top">
<h2 align="center">Case 1: Non-zero Penalties</h2>
<hr class="rule-header-bottom">

For now, let's assume that $\lambda_i>0$ for all $i$. In the above optimization problem, define $\boldsymbol{\beta}' = \boldsymbol{\Lambda} \boldsymbol{\beta}$, so that

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + |\boldsymbol{\beta}'|_1\Big\\}.
\end{align}

This is starting to look like a vanilla Lasso: the pesky $\boldsymbol{\Lambda}$ has been absorbed into the parameters. As a consequence, the "new" penalty term is simply equal to one. We're not done yet, as we need to rewrite $\boldsymbol{\beta}$ in the reconstruction term. To do so, we need a $\boldsymbol{\Lambda}$: it's not immediately available, but we can concoct one as follows:

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\Lambda}^{-1} \boldsymbol{\Lambda}\boldsymbol{\beta}|^2_2 + |\boldsymbol{\beta}'|_1\Big\\} \\\\\
&=\underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\Lambda}^{-1} \boldsymbol{\beta}'|^2_2 + |\boldsymbol{\beta}'|_1\Big\\} \\\\\
&= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}' \boldsymbol{\beta}'|^2_2 + |\boldsymbol{\beta}'|_1\Big\\},
\end{align}

where $\mathbf{X}' = \mathbf{X}\boldsymbol{\Lambda}^{-1}$. What we're left with is a vanilla Lasso problem! The procedure is simple: for some choice of $\boldsymbol{\Lambda}$, transform your design matrix $\mathbf{X} \rightarrow \mathbf{X}\boldsymbol{\Lambda}^{-1}$ and run an ordinary Lasso with a $\lambda=1$.

This approach fails, however, if any of the $\lambda_i$ are equal to zero, since $\boldsymbol{\Lambda}$ becomes singular. In this case, we need to take a different approach.

<hr class="rule-header-top">
<h2 align="center">Case 2: Unpenalized Coefficients</h2> 
<hr class="rule-header-bottom">

Now, we'll assume some of the $\lambda_i$ are equal to zero. This implies that a subset of the $\beta_i$ are unpenalized in the regression. To make this explicit, let's split $\boldsymbol{\beta}$ into two sets of coefficients: those that are penalized $\boldsymbol{\beta}\_{\text{P}}$, and those that aren't $\boldsymbol{\beta}\_{\text{NP}}$; their respective design matrices are $\mathbf{X}\_{\text{P}}$ and $\mathbf{X}\_{\text{NP}}$. Lastly, we refer to the corresponding penalties on $\boldsymbol{\beta}\_{\text{P}}$ as $\boldsymbol{\Lambda}\_{\text{P}}$. 

Thus, the optimization problem can be written as 

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{P}}, \hat{\boldsymbol{\beta}}\_{\text{NP}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}, \  \hat{\boldsymbol{\beta}}\_{\text{NP}}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\_{\text{NP}}\boldsymbol{\beta}\_{\text{NP}} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\|^2_2 + |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\Big\\}.
\end{align}

Let's think about this like coordinate descent: suppose we already have a putative $\boldsymbol{\beta}\_{\text{P}}$ and we just need to compute $\boldsymbol{\beta}\_{\text{NP}}$. We have

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{NP}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{NP}}}{\operatorname{argmin}} \Big\\{\left|\mathbf{y}- \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}} - \mathbf{X}\_{\text{NP}}\boldsymbol{\beta}\_{\text{NP}}\right|^2_2 \Big\\},
\end{align}

where we've removed the penalty term since we're no longer optimizing for $\boldsymbol{\beta}\_{\text{P}}$. Equation (9) is nothing other than an unregularized regression of the residuals $\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}$ on the non-penalized regressors $\mathbf{X}\_{\text{NP}}$. That's just ordinary linear regression! Its solution is simply

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{NP}} &= \left(\mathbf{X}\_{\text{NP}}^T\mathbf{X}\_{\text{NP}}\right)^{-1} \mathbf{X}\_{\text{NP}}^T \left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right).
\end{align}

Thus, everytime we have a guess at the optimal penalized parameters $\boldsymbol{\beta}\_{\text{P}}$, we already have a closed-form solution for  the non-penalized parameters $\boldsymbol{\beta}\_{\text{NP}}$. We can just go ahead and toss this closed-form expression in the original optimization expression, and now optimize for $\boldsymbol{\beta}\_{\text{P}}$:

\begin{align}
\hat{\boldsymbol{\beta}}\_{\text{P}} &= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\_{\text{NP}}\left[\left(\mathbf{X}\_{\text{NP}}^T\mathbf{X}\_{\text{NP}}\right)^{-1} \mathbf{X}\_{\text{NP}}^T \left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right)\right] - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\|^2_2 \notag \\\\\
&\qquad  \qquad \qquad \qquad \qquad + |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\Big\\} 
\\\\\
&= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}}{\operatorname{argmin}} \Big\\{\left|\left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right) - \mathbf{P}\_{\text{NP}}\left(\mathbf{y} - \mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right)\right|^2_2+ |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\Big\\} \\\\\
&= \underset{\hat{\boldsymbol{\beta}}\_{\text{P}}}{\operatorname{argmin}} \Big\\{\left|\mathbf{M}\_{\text{NP}}\mathbf{y} - \mathbf{M}\_{\text{NP}}\mathbf{X}\_{\text{P}} \boldsymbol{\beta}\_{\text{P}}\right|^2_2+ |\boldsymbol{\Lambda}\_{\text{P}}\boldsymbol{\beta}\_{\text{P}}|_1\Big\\},
\end{align}

where $\mathbf{P}\_{\text{NP}}$ is the projection matrix:

\begin{align}
\mathbf{P}\_{\text{NP}} &= \mathbf{X}\_{\text{NP}}\left(\mathbf{X}\_{\text{NP}}^T\mathbf{X}\_{\text{NP}}\right)^{-1} \mathbf{X}\_{\text{NP}}^T
\end{align}

and $\mathbf{M}\_{\text{NP}}$ is its corresponding residual matrix:

\begin{align}
\mathbf{M}\_{\text{NP}} &= \mathbf{I} - \mathbf{P}\_{\text{NP}}.
\end{align}

The projection matrix will project a vector of true dependent variables ($\mathbf{y}$) onto their predicted values ($\hat{\mathbf{y}}$) according to the linear regression; the residual matrix returns the residuals between the true and predicted dependent variables ($\mathbf{y} - \hat{\mathbf{y}}$).

Thus, what we're left with is a simple Lasso problem. The procedure can be summarized as follows: calculate the residual matrix $\mathbf{M}\_{\text{NP}}$ corresponding to the non-penalized design matrix, and apply it to both the dependent variable $\mathbf{y}$ and the penalized regressors $\mathbf{X}\_{\text{P}}$. Then, perform a Lasso on these transformed variables (potentially turning to Case 1 if the penalties vary) to obtain $\hat{\boldsymbol{\beta}}\_{\text{P}}$. Finally, perform ordinary least squares to obtain the estimate for the non-penalized parameters $\hat{\boldsymbol{\beta}}\_{\text{NP}}$.

<hr class="rule-header-top">
<h2 align="center">Summary</h2> 
<hr class="rule-header-bottom">
Consider a lasso optimization procedure with potentially distinct regularization penalties:
\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \Big\\{|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \sum\_{i=1}^N \lambda_i |\beta_i|\Big\\}.
\end{align}
We can solve this with a uniform penalty $\lambda$ as follows:
<ol>
	<li><b>Project out the non-penalized coefficients:</b> Calculate the residual matrix for the non-penalized coefficients ($\lambda_i=0$): $$\mathbf{M}_{\text{NP}} = \mathbf{I} - \mathbf{X}_{\text{NP}}\left(\mathbf{X}_{\text{NP}}^T\mathbf{X}_{\text{NP}}\right)^{-1} \mathbf{X}_{\text{NP}}^T$$
	and apply it to both the response variable $\left(\mathbf{y} \rightarrow \mathbf{M}_{\text{NP}}\mathbf{y}\right)$ and penalized design matrix $\left(\mathbf{X}_{\text{P}} \rightarrow \mathbf{M}_{\text{NP}} \mathbf{X}_{\text{P}}\right)$.
	</li> 
	<li><b>Rescale the projected design matrix:</b> Transform the projected design matrix according to the diagonal matrix of the lasso penalties: $$\mathbf{M}_{\text{NP}} \mathbf{X}_{\text{P}} \rightarrow \mathbf{M}_{\text{NP}} \mathbf{X}_{\text{P}} \boldsymbol{\Lambda}_{\text{P}}^{-1}.$$</li>
	<li><b>Apply a lasso:</b> Lasso regress the projected response variable $\mathbf{M}_{\text{NP}}\mathbf{y}$ on the projected and scaled design matrix $\mathbf{M}_{\text{NP}} \mathbf{X}_{\text{P}} \boldsymbol{\Lambda}_{\text{P}}^{-1}$ with a regularization penalty $\lambda=1$ to obtain estimates for the penalized parameters $\hat{\boldsymbol{\beta}}_{\text{P}}$.</li>
	<li><b>Apply an ordinary least squares:</b> Finally, ordinary least squares regress the residuals of the response variable $\mathbf{y} - \mathbf{X}_{\text{P}} \hat{\boldsymbol{\beta}}_{\text{P}}$ on the non-penalized design matrix $\mathbf{X}_{\text{NP}}$ to obtain the non-penalized parameters $\hat{\boldsymbol{\beta}}_{\text{NP}}$.</li>
</ol>