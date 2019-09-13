---
layout: post
title: Preventing Interactions with the Juvenile Justice System
excerpt: Journal club
---
<hr class="rule-header-title-top">
<h2 align="center"><a href="https://onyilam.github.io/juvenile.pdf">{{page.title}}</a></h2>
<h4 align="center"><i>Reza Borhani, Yaeli Cohen, Onyi Lam, Hareem Naveed,
                      Kevin H. Wilson, Chad Kenney, Rayid Ghani</i></h4>
<hr class="rule-header-title-bottom">
tl;dr: The authors created models that predicted whether students in the Milwaukee
Public School systems would interact with the criminal justice system. These models
could help facilitate early-intervention programs to prevent future interactions
with the criminal justice system.
<hr class="rule-header-top">
<h3>The Problem</h3>
<hr class="rule-header-bottom">
<ul>
    <li>Students that have significant interactions with the juvenile justice
        system often have difficulties reintegrating back into society. Reintegration 
        struggles are correlated with a host of other issues, ranging from decreased
        likelihood of graduation to higher mortality rates.</li>
    <li>In 2015, Milwaukee Public Schools had a graduation rate of 58%, compared to
        the state-wide graduation rate of 88%. Meanwhile, juvenile arrest rates
        in Milwaukee have increased 163% between 2011 and 2015, in contrast to
        a steady decrease nationally.</li>
    <li>Milwaukee Public Schools (MPS) uses targeted interventions in an effort to prevent
        future interactions with the criminal justice system. Their current rules-based
        approach flags roughly 22,000 students which far exceeds their capacity
        to intervene with only 5,000 students per year.</li>
    <li>This leads to the the task: <b>can MPS improve on their ability to target
        "at-risk" students by using a machine learning model to predict which
        students are most likely to have future interactions with the criminal
        justice system?</b></li>
</ul>
<hr class="rule-header-top">
<h3>The Approach</h3>
<hr class="rule-header-bottom">
<ul>
    <li>The authors considered two datasets.
        <ul>
            <li>The first was data from the <b>Milwaukee Public School (MPS)</b> system
            on enrolled students from 2004 to 2015. The features in the MPS
            dataset included demographics, attendance records, disciplinary
            events, test assessment data, and school programs that the student
            was enrolled in.</li>
            <li>The second dataset, from the <b>Milwaukee District Attorney's (MDA)</b>
            office, consisted of juvenile interactions with the criminal justice
            system from 2009 to 2015. Importantly, this dataset only includes
            offenses that are referred to the DA's office, which is only a
            subset of <i>all</i> interactions with the criminal justice system.
            The authors note that this implies that only serious crimes are
            included in the dataset. The features include demographic
            information as well as details on the offense.</li>
        </ul>
    </li>
    <li>A great deal of effort went into <b>cleaning the data and matching
    individuals from each dataset</b>. Ultimately, the authors ended up with
    9,451 unique individuals in the MDA dataset, and linked 86% of them with
    MPS records.</li>
    <li>The authors used <b>standard approaches</b>, ranging from logistic regression,
    random forests, AdaBoost, etc. The feature generation was
    straightforward, with a focus on demographic characteristics, history of
    abuse, and incidents of truancy.</li>
    <li>Recall that the goal of the study was to aid MPS in selecting students
    with the <b>highest risk</b>, as their current approach flagged too many students.
    Thus, the authors prioritized <b>precision in the top 1%</b>, i.e., ensuring the
    model is as accurate as possible in the 1% of students most likely to interact
    with the criminal justice system. The authors also prioritized <b>stability</b>
    of the algorithm over time (i.e., if it works in 2010, it should work in 2015).</li>
</ul>
<hr class="rule-header-top">
<h3>Results and Analysis</h3>
<hr class="rule-header-bottom">
<ul>
    <li>The best performing model was a <b>Random Forest</b>. For the top 1% of predicted risk scores, they obtained a precision of 0.3 and recall of 0.1. This implies that, in the top 1% of risk scores predicted by the model, 30% are students who actually ended up interacting with the justice system (precision), and these students constitute 10% of all students who interacted with the justice system (recall).</li>
    <li>A direct comparison of their model to the benchmark of the rules-based approach that MPS uses <b>reveals a large reduction in false positives</b>:
    <table style="width:100%">
    <tr>
        <th></th>
        <th>Flags</th>
        <th>Correctly Identifies</th>
    </tr>
    <tr>
        <td><b>Heuristic Model (MPS)</b></td>
        <td>22,000</td>
        <td>1,310</td>
    </tr>
    <tr>
        <td><b>Their Model</b></td>
        <td>12,000</td>
        <td>1,630</td>
    </tr>
    </table>
    It's unclear at what percentile the authors thresholded their risk scores for this comparison.</li>
    <li>The most important features in their model included (i) the number of "child in need of protective services" records, (ii) age, (iii) number of discipline incidents in the last 2 years, and (iv) the average number of absence days over the years. It is unclear how the authors determined these features were the most important.</li>
</ul>
<hr class="rule-header-top">
<h3>Contextualizing the Study</h3>
<hr class="rule-header-bottom">
The authors use of a random forest greatly exceeds MPS's heuristic model at
predicting future interactions with the criminal
justice system. In particular, the reduction of false positives is
promising, as it would lead to a decrease in needless interventions. Presumably,
interventions were occurring in the time frames considered in the dataset. It would
be useful to extend the model such that it incorporates these interventions.

It is important to note that consideration of interactions with the criminal justice
system cannot occur without contextualizing those interactions with the systemic
and historical racial disparities exhibited by the criminal justice system. These
racial disparities extend beyond criminal justice, such as in
disciplinary actions within K-12 public schools
<a class="reference">[1].</a>
<span class="citation">
    <a href="https://www.pnas.org/content/116/17/8255">
        Riddle & Sinclair, <i>PNAS</i>, 2019.
    </a>
</span>
The dataset
that was used to train this model will be impacted by the very biases that lead to
the racial disparities in the first place.

Therefore, an analysis on how the algorithm treats people of color <i>specifically</i>
is imperative before depoyment. While the authors state that race is not one of
the most important features used, there is an abundance of work demonstrating
that protected variables can still impact machine learning algorithms, even if
they don't do so <i>directly</i>
<a class="reference">[2].</a>
<span class="citation">
    <a href="https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning">
        Hardt, Price, & Srebro, <i>NeurIPS</i>, 2016.
    </a>
</span>

Additionally, an assumption built into targeted interventions is that
<i>the students</i> must be changed rather than the <i>criminal justice system</i>
or the <i>education system</i> itself. This is not to say that targeted interventions
are not capable of preventing future interactions with the criminal justice system.
Rather, the impact of targeted interventions must be
assessed. First, do they in fact decrease future interactions with the criminal
justice system, and second, are they the most cost effective way of reducing 
the interactions?

<hr class="rule-header-title-bottom">
