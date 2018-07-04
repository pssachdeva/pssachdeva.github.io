---
layout: post
title: A Derivation of the Linear Fisher Information
excerpt: A slog through the derivation of the Fisher information under the assumption of Gaussian noise.
---
<hr class="rule-header-title-top">
<h1 align="center">{{page.title}}</h1>
<hr class="rule-header-title-bottom">

Now, suppose we can write $x$ in some $N$-dimensional representation $\mathbf{r}$ through the function $f$:

\begin{align}
\mathbf{r} &= f(x) + \boldsymbol{\epsilon}
\end{align}

where $\boldsymbol{\epsilon}$ is drawn from an $N$-dimensional Gaussian distribution with zero mean and covariance $\boldsymbol{\Sigma}$.