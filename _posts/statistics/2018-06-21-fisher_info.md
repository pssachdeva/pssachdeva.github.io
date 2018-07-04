---
layout: post
title: A Derivation of the Linear Fisher Information
excerpt: A slog through the derivation of the Fisher information under the assumption of Gaussian noise.
---
<hr class="rule-header-title-top">
<h1 align="center">{{page.title}}</h1>
<hr class="rule-header-title-bottom">

Now, suppose we can write $x$ in some $N$-dimensional representation $\mathbf{r}$ through a function $\mathbf{f}$:

\begin{align}
\mathbf{r} &= \mathbf{f}(x) + \boldsymbol{\epsilon}
\end{align}

where $\boldsymbol{\epsilon}$ is drawn from an $N$-dimensional Gaussian distribution with zero mean and covariance $\boldsymbol{\Sigma}(x)$ (i.e., the covariance is potentially dependent on $x$). Then, the conditional distribution $P[\mathbf{r}|x]$ is a simple Gaussian distribution by virtue of the Gaussian noise $\boldsymbol{\epsilon}$:

\begin{align}
P[\mathbf{r}|x] &= \frac{1}{(2\pi)^N \det \boldsymbol{\Sigma}(x)}\exp\left[-\frac{1}{2}(\mathbf{r}-\mathbf{f}(x))^T \boldsymbol{\Sigma}^{-1}(x) (\mathbf{r} - \mathbf{f}(x))\right]
\end{align}