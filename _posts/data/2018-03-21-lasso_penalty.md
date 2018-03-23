---
layout: post
title: Useful Transformations for the Application of a Lasso Penalty
---

The lasso is a regression method in which we apply an $\ell_1$ penalty to the regression coefficients. It's useful because it performs feature selection: the lasso will only estimate the parameters for the regressors it likes, while the rest get set to zero. However, we might not always want to apply a lasso penalty uniformly - or even at all - to some coefficients. In this post, I'll detail how to rewrite those cases into a vanilla lasso problem. 

<h2 align="center">Setup</h2>
To be clear, let's suppose we have the $T \times N$ design matrix $\mathbf{X}$ consisting of $T$ observations of $N$ features. We also have the . The lasso objective is 
\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\\{\frac{1}{N} \right\\}
\end{align}

\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\{\frac{1}{N}|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \lambda |\boldsymbol{\beta}|_1\right\}
\end{align}

where $\lambda$ is a hyperparameter that we usually choose during cross-validation. It tells us how strongly we should enforce sparsity when choosing $\boldsymbol{\beta}$. 

Right now, we've written the problem such that the same penalty term is applied across all the coefficients. Sometimes, though, we'll have groups of regressors which we know have different levels of sparsity. To be general, let's suppose we want to apply a different regularization strength $\lambda_i$ to each coefficient $\beta_i$. The optimization procedure in this case is 

\begin{align}
\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \left\{\frac{1}{N}|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}|^2_2 + \sum_{i=1}^N \lambda_i |\beta_i|\right\}
\end{align}

<h2 align="center">Lasso</h2>
\begin{align}
		\boldsymbol{\hat{\beta}}_1, \boldsymbol{\hat{\beta}}_2 &= \argmin_{\boldsymbol{\beta}_1, \boldsymbol{\beta}_2} ||\mathbf{y} - \mathbf{X}_2 \boldsymbol{\beta}_1 - \mathbf{X}_1 \boldsymbol{\beta}_2||^2 + \lambda ||\boldsymbol{\beta}_1||_1.
\end{align}
We can reformulate this problem into an ordinary Lasso regression. To see this, let us first expand the reconstruction term in the objective function:
\begin{align}
	||\mathbf{y} - \mathbf{X}_1 \boldsymbol{\beta}_1 - \mathbf{X}_2 \boldsymbol{\beta}_2||^2 &= \left(\mathbf{y} - \mathbf{X}_1 \boldsymbol{\beta}_1 - \mathbf{X}_2\boldsymbol{\beta}_2\right)^T\left(\mathbf{y} - \mathbf{X}_1 \boldsymbol{\beta}_1 - \mathbf{X}_2\boldsymbol{\beta}_2\right) \\
	&= \left(\mathbf{y}_1 - \mathbf{X}_1 \boldsymbol{\beta}_1\right)^T\left(\mathbf{y}_1 - \mathbf{X}_1 \boldsymbol{\beta}_1\right) - (\mathbf{y}_1 - \mathbf{X}_1\boldsymbol{\beta})^T \mathbf{X}_2 \boldsymbol{\beta}_2 \notag \\
	& \qquad  -\boldsymbol{\beta}_2 ^T\mathbf{X}_2^T(\mathbf{y} - \mathbf{X}_1 \boldsymbol{\beta}_1)+ \boldsymbol{\beta}_2^T \mathbf{X}_2^T \mathbf{X}_2 \boldsymbol{\beta}_2.
\end{align}
Now suppose we have a projector into the column space of $\mathbf{X}_2$, namely 
\begin{align}
	\mathbf{P}_2 &= \mathbf{X}_2 (\mathbf{X}_2^T \mathbf{X}_2)^{-1} \mathbf{X}_2^T.
\end{align}
