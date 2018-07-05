---
layout: post
title: A Derivation of the Linear Fisher Information
excerpt: A slog through the derivation of the Fisher information under the assumption of Gaussian noise.
---
<hr class="rule-header-title-top">
<h1 align="center">{{page.title}}</h1>
<hr class="rule-header-title-bottom">

<hr class="rule-header-top">
<h2 align="center">Introduction</h2>
<hr class="rule-header-bottom">
Suppose we have some scalar parameter $x$ and we transform it to an $N$-dimensional representation $\mathbf{r}$ through a function $\mathbf{f}$:

\begin{align}
\mathbf{r} &= \mathbf{f}(x) + \boldsymbol{\epsilon}.
\end{align}

The $\boldsymbol{\epsilon}$ term indicates that the transformation is noisy. Then, how easy is it to decode $x$ given $\mathbf{r}$? One way to assess an estimator $\hat{x}(\mathbf{r})$ is to calculate its variance $\text{Var}(\hat{x})$ over possible data points $\mathbf{r}$. If the variance is large, then in a given instance, we can't necessarily trust our decoder.

The Fisher information places a lower bound on the variance <i>any</i> estimator of $x$ can possess:

\begin{align}
I_F(x) \leq \frac{1}{Var(\hat{x}(\mathbf{r}))}.
\end{align}

Thus, if we have a small Fisher information, then any estimator for $x$ will have a large variance: decoding $x$ from $\mathbf{r}$ will always be hard. 

where $\boldsymbol{\epsilon}$ is drawn from an $N$-dimensional Gaussian distribution with zero mean and covariance $\boldsymbol{\Sigma}(x)$ (i.e., the covariance is potentially dependent on $x$). Then, the conditional distribution $P[\mathbf{r}\vert x]$ is a simple Gaussian distribution by virtue of the Gaussian noise $\boldsymbol{\epsilon}$:

\begin{align}
P[\mathbf{r}|x] &= \frac{1}{\sqrt{(2\pi)^N \det \boldsymbol{\Sigma}(x)}}\exp\left[-\frac{1}{2}(\mathbf{r}-\mathbf{f}(x))(x)^{-1} (\mathbf{r} - \mathbf{f}(x))\right].
\end{align}

Thus, the Fisher information of $x$ given the representation $\mathbf{r}$ can be written as 

\begin{align}
I_F(x) &= \mathbb{E}\left[\left(\frac{d}{dx} \log P[\mathbf{r}\vert x]\right)^2\right] \\\\\\
&=  \int d\mathbf{r} P[\mathbf{r} \vert x] \left(\frac{d}{dx} \log P[\mathbf{r} \vert x]\right)^2
\end{align}

First, let's compute the log-likelihood:

\begin{align}
\log P[\mathbf{r}|x] &= -\frac{N}{2}\log(2\pi) -\frac{1}{2} \log \det \boldsymbol{\Sigma} \notag \\\\\\
&-\frac{1}{2} \left(\mathbf{r} - \mathbf{f}(x)\right)^T \boldsymbol{\Sigma}(x)^{-1} \left(\mathbf{r} - \mathbf{f}(x)\right).
\end{align}

We then need the derivative with respect to $x$:

\begin{align}
\frac{d}{dx} \log P[\mathbf{r}|x] &= 0 - \frac{1}{2}\frac{1}{\det \boldsymbol{\Sigma}(x)} \frac{d}{dx} \det\boldsymbol{\Sigma}(x) \\\\\\
&- \frac{1}{2} \frac{d}{dx} (\mathbf{r} - \mathbf{f}(x))^T\boldsymbol{\Sigma}(x)^{-1}(\mathbf{r}-\mathbf{f}(x)) \\\\\\
&-\frac{1}{2}  (\mathbf{r} - \mathbf{f}(x))^T \frac{d}{dx}\boldsymbol{\Sigma}(x)^{-1} (\mathbf{r}-\mathbf{f}(x)) \\\\\\
&-\frac{1}{2}  (\mathbf{r} - \mathbf{f}(x))^T \boldsymbol{\Sigma}(x)^{-1} \frac{d}{dx}(\mathbf{r}-\mathbf{f}(x))
\end{align}

First, we can evaluate the derivative of a determinant using <a href="https://en.wikipedia.org/wiki/Jacobi%27s_formula">Jacobi's formula</a>, which states that 

\begin{align}
\frac{d}{dx} \det \mathbf{A}(x) &= \det \mathbf{A}(x) \text{ tr}\left[\mathbf{A}(x)^{-1} \mathbf{A}'(x)\right]
\end{align}

where $\mathbf{A}'(x) = \frac{d}{dx}\mathbf{A}(x)$. Thus,

\begin{align}
\frac{d}{dx} \log P[\mathbf{r}|x] &= -\frac{1}{2}\text{tr}\left[\boldsymbol{\Sigma}(x)^{-1} \boldsymbol{\Sigma}'(x)\right] +\frac{1}{2} \mathbf{f}'(x)^T \boldsymbol{\Sigma}(x)^{-1} (\mathbf{r}-\mathbf{f}(x)) \\\\\\
& \qquad \qquad - \frac{1}{2} (\mathbf{r} - \mathbf{f}(x))^T \boldsymbol{\Sigma}'(x)^{-1} (\mathbf{r}-\mathbf{f}(x)) \\\\\\
& \qquad \qquad +\frac{1}{2}(\mathbf{r} - \mathbf{f}(x))^T \boldsymbol{\Sigma}(x)^{-1} \mathbf{f}'(x).
\end{align}

In the above expression, the second and fourth terms are equal since they're both scalars and transposes of each other. As for the derivative of the matrix inverse, we note that

\begin{align}
\frac{d}{dx} \boldsymbol{\Sigma}(x)^{-1}  &= -\boldsymbol{\Sigma}(x)^{-1} \frac{d\boldsymbol{\Sigma}(x)}{dx} \boldsymbol{\Sigma}(x)^{-1}
\end{align}

which can be derived by differentiating the definition of the matrix inverse $\boldsymbol{\Sigma}(x) \boldsymbol{\Sigma}(x)^{-1} = \mathbf{I}$. Thus, we have

\begin{align}
\frac{d}{dx} \log P[\mathbf{r}|x] &= -\frac{1}{2}\text{tr}\left[\boldsymbol{\Sigma}(x)^{-1} \boldsymbol{\Sigma}'(x)\right] + \mathbf{f}'(x)^T \boldsymbol{\Sigma}(x)^{-1} (\mathbf{r} - \mathbf{f}(x)) \\\\\\
& \qquad + \frac{1}{2} (\mathbf{r} - \mathbf{f}(x))^T\boldsymbol{\Sigma}(x)^{-1} \boldsymbol{\Sigma}'(x) \boldsymbol{\Sigma}(x)^{-1}(\mathbf{r} - \mathbf{f}(x))^T
\end{align}
