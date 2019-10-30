---
layout: post
title: A Multilevel Analysis of Patrol Stops and Searches in the San Francisco Bay Area
excerpt: Do California Highway Patrol stops in the Bay Area exhibit racial disparities?
tags: [Analysis]
---
<hr class="rule-header-title-top">
<h1 align="center">{{page.title}}</h1>
<hr class="rule-header-title-bottom">
In this post, I'll summarize the analysis and results from my final project
in a multilevel modeling class at Berkeley. I used a logistic multilevel model
to predict whether searches conducted during traffic stops in the Bay Area
exhibited racial disparities. I found - consistent with much of the literature
on this topic - that, given a traffic stop, black and non-white Hispanic drivers
are more likely to be searched than other races.

<hr class="rule-header-top">
<h2 align="center">Introduction</h2>
<hr class="rule-header-bottom">
In recent years, there's been an increased focus in assessing the degree to
which the practices of law enforcement agencies exhibit racial bias. For example,
the prevalence of social media and technology has allowed particular incidents
of police violence against people of color to be broadcast in the public
discourse. Such incidents have sparked an outcry amongst activists who are 
calling for reform in the criminal justice system. Meanwhile, a multitude
of court cases, such as Floyd v. City of New York
<a class="reference">[1],</a>
<span class="citation">
    <a href="https://ccrjustice.org/sites/default/files/assets/files/Floyd-Liability-Opinion-8-12-13.pdf">
        <i>Floyd, et al. vs. City of New York, et al.</i>, 2013
    </a>
</span> have considered how police
practices such as stop-and-frisk may violate either the Fourth Amendment
(unreasonable search and seizures) or the Fourteenth Amendment (equal
protection clause).

There's a rich literature on using probabilistic models to assess the degree to
which police interactions with civilians exhibit racial disparities. Details on
some of those studies will be elaborated on in future posts, but for now, I'll
include references to a subset of them that informed this analysis
<a class="reference">[2]</a>
<span class="citation">
    <a href="https://www.tandfonline.com/doi/abs/10.1198/016214506000001040">
        Gelman, Fagan, & Kiss, <i>Journal of the American Statistical Association</i>, 2007.
    </a>
</span>
<a class="reference">[3]</a>
<span class="citation">
    <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141854">
        Ross, <i>PLoS One</i>, 2015.
    </a>
</span>
<a class="reference">[4]</a>
<span class="citation">
    <a href="https://5harad.com/papers/frisky.pdf">
        Goel, Rao, & Shroff, <i>The Annals of Applied Statistics</i>, 2016.
    </a>
</span>
<a class="reference">[5].</a>
<span class="citation">
    <a href="https://5harad.com/papers/100M-stops.pdf">
        Pierson et al., <i>Stanford Computational Policy Lab</i>, 2017.
    </a>
</span>
The specific model I used was heavily based on <a class="reference">[6].</a>
<span class="citation">
    <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3049597">
        Hannon, <i>Race and Justice</i>, 2017.
    </a>
</span>

In my analysis, I utilized the recently consolidated dataset from the
<a href="https://openpolicing.stanford.edu/">Stanford Open Policing Project</a>.
This dataset acts as a standardized repository of traffic and pedestrian stops
across 31 state police agencies. Each sample in the dataset is a traffic stop
and includes features such as driver demographics, violation information, and
whether searches or arrests were conducted. I complemented this data with
demographic and crime data at the county level, taken from the American
Community Surveys and the Criminal Justice Profiles from the California State
Attorney General’s office.

The main questions I aimed to answer were as follows:
<ol>
    <li>Given a traffic stop, do there exists racial disparities in the rates
        at which searches are conducted throughout the Bay Area?</li>
    <li>If these racial disparities exist, how do they vary across the counties of the Bay Area?</li>
    <li>How are these putative racial disparities moderated by demographic and
        crime rate features for each county?</li>
</ol>
In answering these research questions, I control for the driver’s gender (which
is known to correlate with stop rates) as well as the reason for the stop (e.g.
a DUI may result in a higher probability of a search). In addition, I control
for county-level features that may influence the search rate, such as property
and violent crime rates (searches may be more common in counties with higher
crime), poverty rate, and the percent of residents that are non-white.
Importantly, I control for stop rate and population as these exhibit large
variances amongst the counties, and certain counties with smaller population
exhibit abnormally large numbers of stops.

<hr class="rule-header-top">
<h2 align="center">The Data</h2>
<hr class="rule-header-bottom">

<div style="text-align:center">
<img src="/pics/mlm/mlm_n_stops.png"/>
<b>Figure 1:</b> Stop rate in each Bay Area county
</div>
<hr class="rule-header-top">
<h2 align="center">Multilevel Model</h2>
<hr class="rule-header-bottom">
I considered three separate models, each including more features than the previous
model. For brevity, I'll detail the complete model, and highlight which subset
of variables was used for the prior models. 

The complete model was a random intercepts logistic regression with interaction terms:

\begin{align}
&\text{logit}\\{\text{Pr}(\texttt{search}\_{ij}=1|\mathbf{x}\_{ij}, \mathbf{z}\_j)\\} = (\beta_0 + \zeta_j) + \beta_1 \times \texttt{driver_male}\_{ij} \\\\\
&+\beta_2 \times \texttt{driver_black}\_{ij} + \beta_3 \times \texttt{driver_hispanic}\_{ij} \\\\\
&+\beta_4 \times \texttt{driver_asian}\_{ij} + \beta_5 \times \texttt{driver_other}\_{ij} \\\\\
&+\beta_6 \times \texttt{moving}\_{ij} + \beta_7 \times \texttt{dui}\_{ij} + \beta_8 \times \texttt{equipment}\_{ij} \\\\\
&+\beta\_{9} \times \texttt{poverty_rate}\_j + \beta\_{10} \times \texttt{percent_nonwhite}\_j \\\\\
&+\beta\_{11} \times \texttt{violent_crime}\_j \\\\\
&+\beta\_{12} \times \texttt{driver_black}\_{ij} \times \texttt{violent_crime}\_j \\\\\
&+\beta\_{13}\times \texttt{driver_hispanic}\_{ij} \times \texttt{violent_crime}\_j
\end{align}
There's a lot going on here, so let's break it down. First, note that $i$ indexes the stop, while $j$ indexes the county.

Next, note the breakdown in variable terms:
<ul>
    <li>The first feature ($\beta_1$) controls for gender.</li>
    <li>The next four features ($\beta_2, \beta_3, \beta_4, \beta_5$) encode race. Hispanic refers to non-white hispanic.</li>
    <li>The next three features ($\beta_6, \beta_7, \beta_8$) encode the violation that led to the stop (e.g. moving violation, DUI, or equipment problem).</li>
    <li>The following four features are at the county-level ($\beta_9, \beta_{10}, \beta_{11}, \beta_{12}$) and respectively encode the population (in thousands), poverty rate (percentage of people below the poverty line), violent crime rate (number of violent crimes per 100,000 inhabitants), and property crime rate (number of property crimes per 100,000 inhabitants).</li>
    <li>The last two features encode interaction terms between race and crime rates.</li>
</ul>
Thus, the three models were as follows:

* <b>Model 1:</b> Baseline random intercept model.
* <b>Model 2:</b> Random intercept model with stop-level information (driver
  demographic information and violations).
* <b>Model 3:</b> Random intercept model with stop-level information and
  county-level features, including interaction terms.


<hr class="rule-header-top">
<h2 align="center">Results</h2>
<hr class="rule-header-bottom">
The three models detailed above were fit to the data using Stata. The resulting
parameter estimates are depicted in Table 1. Note that all estimates are assumed
to be significant at the $p=0.05$ level, except when denoted by a dagger.

<div style="text-align:center">
    <b>Table 1:</b> Fitted Parameters for Multilevel Models <br>
    $^{\dagger}$ denotes parameters that are <i>not</i> significant at the $p=0.05$ level.
    <table style="width:100%">
        <tr class="double-border">
            <th></th>
            <th>Model 1</th>
            <th>Model 2</th>
            <th>Model 3</th>
        </tr>
        <tr class="double-border">
            <td><b>Intercept</b></td>
            <td>$-3.42 \ (0.06)$</td>
            <td>$-4.16 \ (0.06)$</td>
            <td>$-4.27 \ (0.20)$</td>
        </tr>
        <tr>
            <td><b>Male</b></td>
            <td></td>
            <td>$+0.39 \ (0.02)$</td>
            <td>$+0.38 \ (0.01)$</td>
        </tr>
        <tr>
            <td><b>Black</b></td>
            <td></td>
            <td>$+0.73 \ (0.02)$</td>
            <td>$+0.35 \ (0.08)$</td>
        </tr>
        <tr>
            <td><b>Hispanic</b></td>
            <td></td>
            <td>$+0.63 \ (0.02)$</td>
            <td>$+0.70 \ (0.05)$</td>
        </tr>
        <tr>
            <td><b>Asian</b></td>
            <td></td>
            <td>$-0.24 \ (0.03)$</td>
            <td>$-0.24 \ (0.02)$</td>
        </tr>
        <tr class="double-border">
            <td><b>Other</b></td>
            <td></td>
            <td>$-0.35 \ (0.03)$</td>
            <td>$-0.34 \ (0.03)$</td>
        </tr>
        <tr>
            <td><b>Moving</b></td>
            <td></td>
            <td>$+0.36 \ (0.02)$</td>
            <td>$+0.36 \ (0.02)$</td>
        </tr>
        <tr>
            <td><b>DUI</b></td>
            <td></td>
            <td>$+1.83 \ (0.02)$</td>
            <td>$+1.83 \ (0.02)$</td>
        </tr>
        <tr class="double-border">
            <td><b>Equipment</b></td>
            <td></td>
            <td>$-0.05 \ (0.02)^{\dagger}$</td>
            <td>$-0.05 \ (0.02)$</td>
        </tr>
        <tr>
            <td><b>Poverty Rate</b></td>
            <td></td>
            <td></td>
            <td>$+0.04 \ (0.02)^{\dagger}$</td>
        </tr>
        <tr>
            <td><b>% Non-White</b></td>
            <td></td>
            <td></td>
            <td>$-0.01\ (0.00)^{\dagger}$</td>
        </tr>
        <tr class="double-border">
            <td><b>Violent Crime</b></td>
            <td></td>
            <td></td>
            <td>$+0.02\ (0.03)^{\dagger}$</td>
        </tr>
        <tr>
            <td><b>Black x Crime</b></td>
            <td></td>
            <td></td>
            <td>$+0.07\ (0.01)$</td>
        </tr>
        <tr class="double-border">
            <td><b>Hispanic x Crime</b></td>
            <td></td>
            <td></td>
            <td>$-0.01\ (0.01)^{\dagger}$</td>
        </tr>
        <tr class="double-border">
            <td><b>Error Variance</b></td>
            <td>$+0.18\ (0.04)$</td>
            <td>$+0.19\ (0.04)$</td>
            <td>$+0.13\ (0.03)$</td>
        </tr>
    </table>
</div>

The grand mean of the intercepts across the models were all large and negative,
implying that searches do not occur often. In addition, the error variance for
model 1 was $0.184$ with a residual intraclass correlation of $0.01$. This small
ICC passes the likelihood ratio test at the $p = 0.001$ level, implying that a
multi-level model is appropriate despite a small effect size.

In Model 2, we added individual level covariates including the driver's
violation and demographic information. The addition of these variables improved
the likelihood while maintaining the small but significant residual ICC
($\hat{\rho}=0.01$). Importantly, the coefficients for black and Hispanic result
in conditional odds ratios are $2.07$ and $1.88$, respectively. This implies
that, given a county, they are more likely to be searched than a white driver
(conversely, Asian and ``Other'' drivers are less likely to be searched than a
white driver) given a stop. Thus, this provides evidence that there are racial
disparities in search rates across the Bay Area but that they do not
strongly vary across the Bay Area.

The complete model achieved the highest log-likelihood of the four models. In
addition, it maintained the small but significant ICC ($\hat{\rho} = 0.005$).
The inclusion of the interaction terms did not impact most of the level-1 covariates
except for whether the driver was black or Hispanic, as might be expected. The
conditional odds ratios for black, hispanic, Asian, and other drivers are
$1.42$, $2.01$, $0.79$, and $0.71$ compared to white drivers, respectively.
Thus, the racial disparities among black and Hispanic drivers persist.
Furthermore, it is worthwhile to note that the conditional odds ratio for male
drivers $(1.47)$ implies that there may exist gender disparities, as the model
implies that men are more likely to be searched than women. Among the violation
variables, a DUI is the most significant with the highest effect size
(conditional odds ratio of $6.25$), implying that drivers pulled over under the
suspicion of DUI are much more likely to be searched.

In Model 4, all level-2 covariates are not significant at the $p=0.05$ level.
The Hispanic interaction term is also not significant, while the black
interaction term is highly significant. Its conditional odds ratio is $1.074$,
implying that among black drivers, each additional violent crime per 1,000
people results in a $7\%$ increase in the chance of a stop. This can result in a
substantial increase in the likelihood that a black driver is pulled over
depending on the county. Thus, violent crime rate among the Bay Area counties
moderates the racial disparities of search rates among black drivers.
<hr class="rule-header-top">
<h2 align="center">Context and Avenues for Future Work</h2>
<hr class="rule-header-bottom">

Ideally, I would have operated at a finer spatial resolution (e.g. census
tracts) as racial disparities are more likely to be moderated at this level
(rather than the county level, which may blur the lines between segregated
neighborhoods). Unfortunately, our dataset only provided stops at the county
level. A study with data at the census tract level likely would have revealed m
correlating with socioeconomic factors, as past studies have shown. In contrast,
our study did not observe strong variations among counties in the Bay Area,
likely due to the small number of clusters and the large population in each
county.

This analysis could have benefited from additional features, such as the
driver’s age, ethnicity, the driver’s criminal history that was available to the
officer, and for what reason the officer initiated a search. For example, past
studies have examined such reasons in stop-and-frisk data and found that reasons
such as “furtive movements” are often used but not predictive of a crime, thus
providing a mechanism for which racial biases tie into frisk rates.