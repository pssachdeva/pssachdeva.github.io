---
layout: post
title: A Derivation of the Linear Fisher Information
excerpt: >-
  A slog through the derivation of the Fisher information under the assumption
  of Gaussian noise.
published: true
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

The $\boldsymbol{\epsilon}$ term indicates that the transformation is noisy. How easy is it to decode $x$ given $\mathbf{r}$? One way to assess an estimator $\hat{x}(\mathbf{r})$ is to calculate its variance $\text{Var}(\hat{x})$ over possible data points $\mathbf{r}$. If the variance is large, we can't necessarily trust our decoder in a given instance.

The Fisher information $I_F(x)$ places a lower bound on the variance <i>any</i> estimator of $x$ can possess:

\begin{align}
\text{Var}\left[\hat{x}(\mathbf{r})\right] \geq \frac{1}{I_F(x)}.
\end{align}

Thus, if we have a small Fisher information, then even the best estimator for $x$ will have a large variance: decoding $x$ from $\mathbf{r}$ will always be hard. 

Given a log-likelihood $\log P[\mathbf{r} \vert x]$, there is a mathematical expression for the Fisher information:

\begin{align}
I_F(x) &= \mathbb{E}_{\mathbf{r}\vert x}\left[\left(\frac{d}{dx} \log P[\mathbf{r}\vert x]\right)^2\right]. \\\\\\
&=  \int d\mathbf{r} P[\mathbf{r} \vert x] \left(\frac{d}{dx} \log P[\mathbf{r} \vert x]\right)^2
\end{align}

That is, the Fisher information is the average square of the score function (the derivative of the log-likelihood with respect to the parameter).

In computational neuroscience, equation (1) is a simple encoding scheme modeling neural activity: an incoming stimulus $x$ is transformed into a (noisy) neural representation $\mathbf{r}$. Neural systems probably want to decode the stimulus at some point, so the Fisher information is a quantity of interest. In equation (1), we can analytically determine the Fisher information when the noise $\boldsymbol{\epsilon}$ is Gaussian. 

In this post, we'll slog through that derivation and highlight the <b>linear Fisher information</b>, a component of the end expression that has been particularly important in the computational neuroscience literature. 

<hr class="rule-header-top">
<h2 align="center">Deriving the Fisher Information</h2>
<hr class="rule-header-bottom">

Assume $\boldsymbol{\epsilon}$ is drawn from an $N$-dimensional Gaussian distribution with zero mean and covariance $\boldsymbol{\Sigma}(x)$ (i.e., the covariance is potentially dependent on $x$). Then, the conditional distribution $P[\mathbf{r}\vert x]$ is a simple Gaussian distribution:

\begin{align}
P[\mathbf{r}\vert x] &= \frac{1}{\sqrt{(2\pi)^N \det \boldsymbol{\Sigma}(x)}}\exp\left[-\frac{1}{2}(\mathbf{r}-\mathbf{f}(x))(x)^{-1} (\mathbf{r} - \mathbf{f}(x))\right].
\end{align}

First, let's compute the log-likelihood:

\begin{align}
\log P[\mathbf{r}\vert x] &= -\frac{N}{2}\log(2\pi) -\frac{1}{2} \log \det \boldsymbol{\Sigma} \notag \\\\\\
&-\frac{1}{2} \left(\mathbf{r} - \mathbf{f}(x)\right)^T \boldsymbol{\Sigma}(x)^{-1} \left(\mathbf{r} - \mathbf{f}(x)\right).
\end{align}

We then need the derivative with respect to $x$:

\begin{align}
\frac{d}{dx} \log P[\mathbf{r}\vert x] &= 0 - \frac{1}{2}\frac{1}{\det \boldsymbol{\Sigma}(x)} \frac{d}{dx} \det\boldsymbol{\Sigma}(x) \\\\\\
&- \frac{1}{2} \frac{d(\mathbf{r} - \mathbf{f}(x))^T}{dx} \boldsymbol{\Sigma}(x)^{-1}(\mathbf{r}-\mathbf{f}(x)) \\\\\\
&-\frac{1}{2}  (\mathbf{r} - \mathbf{f}(x))^T \frac{d\boldsymbol{\Sigma}(x)^{-1}}{dx} (\mathbf{r}-\mathbf{f}(x)) \\\\\\
&-\frac{1}{2}  (\mathbf{r} - \mathbf{f}(x))^T \boldsymbol{\Sigma}(x)^{-1} \frac{d(\mathbf{r}-\mathbf{f}(x))}{dx}
\end{align}

First, we can evaluate the derivative of a determinant using <a href="https://en.wikipedia.org/wiki/Jacobi%27s_formula">Jacobi's formula</a>, which states that 

\begin{align}
\frac{d}{dx} \det \mathbf{A}(x) &= \det \mathbf{A}(x) \text{ Tr}\left[\mathbf{A}(x)^{-1} \mathbf{A}'(x)\right]
\end{align}

where $\mathbf{A}'(x) = \frac{d}{dx}\mathbf{A}(x)$. Thus,

\begin{align}
\frac{d}{dx} \log P[\mathbf{r} \vert x] &= -\frac{1}{2}\text{Tr}\left[\boldsymbol{\Sigma}(x)^{-1} \boldsymbol{\Sigma}'(x)\right] +\frac{1}{2} \mathbf{f}'(x)^T \boldsymbol{\Sigma}(x)^{-1} (\mathbf{r}-\mathbf{f}(x)) \\\\\\
& \qquad \qquad - \frac{1}{2} (\mathbf{r} - \mathbf{f}(x))^T \boldsymbol{\Sigma}'(x)^{-1} (\mathbf{r}-\mathbf{f}(x)) \\\\\\
& \qquad \qquad +\frac{1}{2}(\mathbf{r} - \mathbf{f}(x))^T \boldsymbol{\Sigma}(x)^{-1} \mathbf{f}'(x).
\end{align}

In the above expression, the second and fourth terms are equal since they're both scalars and transposes of each other. As for the derivative of the matrix inverse, we note that

\begin{align}
\frac{d}{dx} \boldsymbol{\Sigma}(x)^{-1}  &= -\boldsymbol{\Sigma}(x)^{-1} \frac{d\boldsymbol{\Sigma}(x)}{dx} \boldsymbol{\Sigma}(x)^{-1}
\end{align}

which can be derived by differentiating the definition of the matrix inverse $\boldsymbol{\Sigma}(x) \boldsymbol{\Sigma}(x)^{-1} = \mathbf{I}$. Thus, we have

\begin{align}
\frac{d}{dx} \log P[\mathbf{r}\vert x] &= -\frac{1}{2}\text{Tr}\left[\boldsymbol{\Sigma}(x)^{-1} \boldsymbol{\Sigma}'(x)\right] + \mathbf{f}'(x)^T \boldsymbol{\Sigma}(x)^{-1} (\mathbf{r} - \mathbf{f}(x)) \\\\\\
& \qquad + \frac{1}{2} (\mathbf{r} - \mathbf{f}(x))^T\boldsymbol{\Sigma}(x)^{-1} \boldsymbol{\Sigma}'(x) \boldsymbol{\Sigma}(x)^{-1}(\mathbf{r} - \mathbf{f}(x))^T \\\\\\
&= -\frac{1}{2}\text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] + \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r} - \mathbf{f}) \notag \\\\\\
& \qquad + \frac{1}{2} (\mathbf{r} - \mathbf{f})^T\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}(\mathbf{r} - \mathbf{f}).
\end{align}

In the last line, we removed the dependence on $x$ to save space. 

We'll need to square this expression to calculate the Fisher information. This is annoying, but let's be organized:

\begin{align}
\left(\frac{d}{dx} \log P[\mathbf{r}\vert x]\right)^2 &= \frac{1}{4} \text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right]^2 \notag \\\\\\
&+ \left[\mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r} - \mathbf{f})\right]^2 \notag \\\\\\
& + \frac{1}{4} \left[(\mathbf{r} - \mathbf{f})^T\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}(\mathbf{r} - \mathbf{f})\right]^2 \notag \\\\\\
& -\text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \cdot \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r} - \mathbf{f})\notag \\\\\\
& -\frac{1}{2}\text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \cdot (\mathbf{r} - \mathbf{f})^T\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}(\mathbf{r} - \mathbf{f}) \notag \\\\\\
& + \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r} - \mathbf{f}) \cdot (\mathbf{r} - \mathbf{f})^T\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}(\mathbf{r} - \mathbf{f}) \\\\\\
&= E_1 + E_2 + E_3 + E_4 + E_5 + E_6. \label{eqn:addends}
\end{align}

The Fisher information is the expectation of this expression over $P[\mathbf{r}\vert x]$. We've split up our expression into six addends, which we've listed in equation \eqref{eqn:addends}.  Furthermore, terms like $\boldsymbol{\Sigma}(x)$ and $\mathbf{f}(x)$ have no $\mathbf{r}$ dependence and therefore are constants. That makes the expectation of the first term easy:

\begin{align}
E_1 = \mathbb{E}_{\mathbf{r}\vert x}\left[\frac{1}{4} \text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right]^2\right] &= \frac{1}{4} \text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right]^2
\end{align}

Next, note the that addends with an odd number of $(\mathbf{r}-\mathbf{f})$ terms will vanish, since we are taking an expectation over a Gaussian. For example, the expectation of the fourth term is 
\begin{align}
E_4 &= \mathbb{E}_{\mathbf{r}\vert x}\left[-\text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r} - \mathbf{f})\right] \\\\\\
&= -\text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} \mathbb{E}\left[(\mathbf{r} - \mathbf{f})\right] \\\\\\
&= 0
\end{align}

since $\mathbb{E}\left[\mathbf{r}\right] = \mathbf{f}$ by definition. Similarly, the expectation of the sixth term $E_6$ also vanishes, since it contains three instances of $(\mathbf{r}-\mathbf{f})$. 

The expectation of the second term is 

\begin{align}
E_2 &= \mathbb{E}_{\mathbf{r}\vert x}\left[\left[\mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r} - \mathbf{f})\right]^2\right] \\\\\\
&= \mathbb{E}\left[\mathbf{f}'^T \boldsymbol{\Sigma}^{-1} (\mathbf{r}-\mathbf{f})(\mathbf{r}-\mathbf{f})^T \boldsymbol{\Sigma}^{-1} \mathbf{f}'\right] \\\\\\
&= \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} \mathbb{E}\left[(\mathbf{r}-\mathbf{f})(\mathbf{r}-\mathbf{f})^T\right] \boldsymbol{\Sigma}^{-1} \mathbf{f}' \\\\\\
&= \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma} \boldsymbol{\Sigma}^{-1} \mathbf{f}' \\\\\\
&= \mathbf{f}'^T \boldsymbol{\Sigma}^{-1} \mathbf{f}',
\end{align}

which, incidentally, is the linear Fisher information. 

Next, the expectation of the fifth term is 
\begin{align}
E_5 &= -\frac{1}{2} \text{Tr} \left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \mathbb{E}\left[(\mathbf{r}-\mathbf{f})^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1} (\mathbf{r}-\mathbf{f}) \right].
\end{align}

Here, we evaluating over a quadratic form for which we'll need to invoke the identity
\begin{align}
\mathbb{E}\left[\boldsymbol{\epsilon}^T \boldsymbol{\Lambda} \boldsymbol{\epsilon} \right] &= \text{Tr}\left[\boldsymbol{\Lambda} \text{Cov}(\boldsymbol{\epsilon})\right] + \mathbb{E}[\boldsymbol{\epsilon}]^T \boldsymbol{\Lambda} \mathbb{E}[\boldsymbol{\epsilon}]
\end{align}
where $\boldsymbol{\Lambda}$ is a matrix and $\boldsymbol{\epsilon}$ is a random vector. In our case, the expectation of $(\mathbf{r}-\mathbf{f})$ vanishes, so we're only concerned with the first term. The expectation becomes 
\begin{align}
E_5 &= -\frac{1}{2} \text{Tr}\left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \cdot \text{Tr}\left[\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}\right] \\\\\\
&= -\frac{1}{2} \text{Tr} \left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right] \cdot \text{Tr}\left[\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}'\right] \\\\\\
&= -\frac{1}{2} \text{Tr} \left[\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}'\right]^2.
\end{align}

We've left the trickiest for last. The expectation of the third term is effectively the expectation over the product of two quadratic forms: 

\begin{align}
E_3 &= \frac{1}{4} \mathbb{E}\left[(\mathbf{r} - \mathbf{f})^T\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}(\mathbf{r} - \mathbf{f})(\mathbf{r} - \mathbf{f})^T\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}(\mathbf{r} - \mathbf{f})\right]
\end{align}

for which we'll need the identity (see <a href="https://www.jstor.org/stable/25051849">here</a>).

\begin{align}
\mathbb{E}\left[\boldsymbol{\epsilon}^T \mathbf{A} \boldsymbol{\epsilon}\boldsymbol{\epsilon}^T \mathbf{B} \boldsymbol{\epsilon}\right] &= \text{Tr}\left[\mathbf{A}\mathbf{C}\right]\text{Tr}\left[\mathbf{B}\mathbf{C}\right] + 2 \text{Tr}\left[\mathbf{A}\mathbf{C} \mathbf{B}\mathbf{C}\right]
\end{align}
where $\mathbf{C} = \text{Cov}(\boldsymbol{\epsilon})$. Applying the identity gives us our last addend: 

\begin{align}
E_3 &= \frac{1}{4} \text{Tr}\left[\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\right]^2 + \frac{1}{2}\text{Tr}\left[\left(\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}' \boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\right)^2\right] \\\\\\
&= \frac{1}{4}\text{Tr}\left[\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma}'\right]^2
\end{align}