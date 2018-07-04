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

where $\boldsymbol{\epsilon}$ is drawn from an $N$-dimensional Gaussian distribution with zero mean and covariance $\boldsymbol{\Sigma}(x)$ (i.e., the covariance is potentially dependent on $x$). Then, the conditional distribution $P[\mathbf{r}\vert x]$ is a simple Gaussian distribution by virtue of the Gaussian noise $\boldsymbol{\epsilon}$:

\begin{align}
P[\mathbf{r}|x] &= \frac{1}{\sqrt{(2\pi)^N \det \boldsymbol{\Sigma}(x)}}\exp\left[-\frac{1}{2}(\mathbf{r}-\mathbf{f}(x))^T \boldsymbol{\Sigma}^{-1}(x) (\mathbf{r} - \mathbf{f}(x))\right].
\end{align}

Thus, the Fisher information of $x$ given the representation $\mathbf{r}$ can be written as 

\begin{align}
I_F(x) &= \mathbb{E}\left[\left(\frac{d}{dx} \log P[\mathbf{r}\vert x]\right)^2\right] \\\\\\
&=  \int d\mathbf{r} P[\mathbf{r} \vert x] \left(\frac{d}{dx} \log P[\mathbf{r} \vert x]\right)^2
\end{align}

First, let's compute the log-likelihood:

\begin{align}
\log P[\mathbf{r}|x] &= -\frac{N}{2}\log(2\pi) -\frac{1}{2} \log \det \boldsymbol{\Sigma} \notag \\\\\\
&\qquad -\frac{1}{2} \left(\mathbf{r} - \mathbf{f}(x)\right)^T \boldsymbol{\Sigma}^{-1}(x) \left(\mathbf{r} - \mathbf{f}(x)\right).
\end{align}