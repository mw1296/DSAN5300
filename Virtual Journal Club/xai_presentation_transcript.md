# XAI Journal Club — Presenter Transcript
**Paper:** Dwivedi et al. (2023). Explainable AI (XAI): Core Ideas, Techniques, and Solutions. *ACM Computing Surveys*, 55(9). doi:10.1145/3561048

---

## Slide 1 — Title

Hi everyone, thanks for joining. Today Troy and I are presenting a paper on Explainable Artificial Intelligence, or XAI, by Dwivedi and colleagues, published in ACM Computing Surveys. It's a fundational paper for the moden XAI area, a heavily cited paper — nearly a thousand citations since publication in 2023. This XAI methodology bridges the gap between academic theory and the practical needs of a data scientist.

While the name says 'Artificial Intelligence,' XAI is fundamentally about Machine Learning. It provides data scentists a tool to move from simply seeing what a model predicts to understanding the logic and features that drove that output."

---

## Slide 2 — Modern ML Models Are Black Boxes

Before we get into XAI, I want to establish why it exists in the first place.

Think back to the models you learned first in your data science courses — logistic regression, linear regression, decision trees. These are what we call white box models. With logistic regression, every coefficient directly tells you how much a feature shifts the prediction. With a decision tree, you can trace any prediction from the root node all the way to the leaf and read the exact logic step by step. The model explains itself. No extra interpretation needed.

In contrast, neural networks and ensemble methods like XGBoost and Random Forest are black box models. A neural network might have millions of weights distributed across hidden layers. There is no human-readable path from input to output. You feed in the data, you get a prediction, but the internal computation is completely opaque. XGBoost works similarly — it combines hundreds of decision trees, and while each individual tree has a path you could trace, the ensemble as a whole is not visible.

Here is the core tension that XAI is built around: it tells data scientists what statistical/ML methods to use that might shed the lights to the black box model otupt.

---

## Slide 3 — Agenda

Here's what we'll cover in the next twenty minutes.

Part One, focuses on the problem — specifically, why high accuracy alone is not enough, and what real consequences follow from unexplainable AI systems.

Part Two, covers the methods — the taxonomy the paper introduces, and a practical walkthrough of SHAP and LIME, which are the two tools you're most likely to use.

Part Three, focuses on the takeaway — A guide for choosing the right tool for your when explanation is legally required versus optional,

I also want to flag a central question that I'll come back to, because it came up in my own reading: if you already know SHAP and LIME and can run them on any model, what does the label "XAI" actually add? That question will be the thread running through the whole presentation.

---

## Slide 4 — Why Accuracy Alone Isn't Enough

High accuracy is a performance metric, but explainability is a trust metric. A model can be 96% accurate and still be completely wrong about the why.

If you look at this matrix, we categorize risk by the Cost of Opaque Decisions:

The 'Safe' Zone (Bottom Left): For things like Spam Filtering or Search Ranking, the cost of a mistake is low. Here, a 'black box' is acceptable. For example, movie recommendations on Netflix.

The 'Critical' Zone (Top Right): In Medical Diagnosis or Credit Risk, the stakes are life-altering. In these high-stakes domains, an accurate model that cannot explain itself is legally and ethically 'unacceptable'. For exmaple, cancer diagnosis. A doctor uses an AI to spot a tumor. The cost of being wrong is death, so the human expert demands an explanation. They won't sign off on the decision unless they see the "why" (e.g., "The model flagged this cell because of its irregular border").

For those of us who want to work in credit lending, medical image recognition and Insurance Pricing, XAI is our primary tool for compliance.


**side note:** Top Left: Critical Automation
[High Cost of Error | Low Human Participation]

Example: High-Frequency Trading or Self-Driving Cars

Why: These systems move faster than a human can react. If a self-driving car misidentifies a pedestrian as a shadow, the cost is a life. Because a human isn't "in the loop" to catch the mistake in real-time, the model's logic must be perfectly explainable during the development phase to prevent catastrophic failure.

Top Left: Critical Automation
[High Cost of Error | Low Human Participation]

Example: High-Frequency Trading or Self-Driving Cars

Why: These systems move faster than a human can react. If a self-driving car misidentifies a pedestrian as a shadow, the cost is a life. Because a human isn't "in the loop" to catch the mistake in real-time, the model's logic must be perfectly explainable during the development phase to prevent catastrophic failure.
---

## Slide 5 — The role of XAI

This slide shows you where the XAI sits in the ML process. And this is a much simplified ML Ops workflow. What we have been practicing is the blue area - process data --> train model --> tune model --> make predcitions. XAI comes along after we've finalized the model that achieves the best score, then we fit the model with new data and use other ML methods to explain the model. And these methods are those white box methods which is easy to interprete.

---

## Slide 6 — Taxonomy of XAI Techniques

Now let's move into Part Two — the methods themselves. The paper introduces a taxonomy that classifies every XAI technique along three axes, and this taxonomy is genuinely useful as a practical navigation tool.

The first axis is scope: global versus local. A global explanation describes the overall behavior of the model — which features matter most across all predictions. A local explanation zooms in on a single prediction — why did the model make this specific tree split.

The second axis is model dependency: meaning whether the method depends on the specific model. Tree SHAP, for example, exploits the structure of decision trees and can only be applied to tree-based models. Model-agnostic tools work on any model.

The third axis is transparency: white box versus black box. White box models are self-explanatory — logistic regression, decision trees, and linear models can be read directly. Black box models — neural networks, XGBoost — require post-hoc explanations.

The critical point for everyone in this room who works with deep learning: neural networks always live in the black box rows. Once you add hidden layers, there is no white-box option. You will always need a post-hoc tool.


---


## Slide 8 — SHAP: The Workhorse of Black-Box Explanation (9:00 – 11:00)

Let's look at SHAP in more detail, because it's the most important practical tool the paper covers.

SHAP stands for SHapley Additive exPlanations. The intuition is that SHAP analyzes what is each feature's fair contribution to that final prediction? It computes this by calculating the average marginal contribution of each feature across all possible subsets of features. The result for each feature is called its Shapley value. Then sum them up and add to the expected average value.

Looking at the example on the right — the loan denial case. The baseline is 0.50. Debt ratio pushes the prediction up by 0.22 toward denial. Credit score pushes it up by 0.14. Loan amount pushes it up by 0.08. Income and age push it slightly back toward approval, but not enough to overcome the others. Final prediction: 0.87. You can now explain, precisely and defensibly, why this loan was denied.

There are two key variants. Tree SHAP computes exact Shapley values by exploiting the tree structure — it's fast and exact, but only works on tree-based models like XGBoost and Random Forest. Kernel SHAP is model-agnostic — it works on any model including neural networks — but uses a weighted regression approximation, so it's slower and approximate.

**side note:** add notes about Kernal SHPA. understand what that is. Depper understanding about how the statisical fundation of SHAP.
---

## Slide 9 — LIME: Local Approximation of Any Black Box (11:00 – 13:00)

LIME stands for Local Interpretable Model-agnostic Explanations. Like SHAP, it explains individual predictions. But where SHAP uses game theory, LIME uses a much simpler idea: if you want to understand how a black-box model behaves near one specific data point, build a simple interpretable model — a linear regression — that approximates the black box in that local neighborhood.

The process has four steps. First, take the one data point you want to explain — say, one specific loan applicant. Second, perturb that data point hundreds of times: slightly vary the age, the income, the debt ratio, and run each perturbed version through the black-box model to collect predictions. Third, fit a local linear model on those perturbed samples, weighting them by how similar they are to the original point. Fourth, read the coefficients of that linear model — those are your feature contributions for that single prediction.

Now the comparison. LIME is faster than Kernel SHAP, and it works on any model. But it has two significant weaknesses. First, stability: if you run LIME twice on the same instance, you may get different feature weights, because the neighborhood is defined arbitrarily and the sampling introduces randomness. Second, it offers no additivity guarantee — unlike SHAP, the weights don't necessarily sum to anything meaningful.

The practical rule: use LIME for quick, interactive exploration when you're trying to get a rough sense of what's driving a prediction. Use SHAP for anything you're going to report to a stakeholder, include in an audit trail, or use for compliance purposes. SHAP is stable and mathematically guaranteed. LIME is not.

---

## Slide 10 — PDP


--
## Slide 11 — Which Method Do I Run and When?

This is the practical slide — the decision guide. And it directly answers the question that came up in our prep sessions: after training a neural network, which tool do I run to understand the reasoning?

For a neural network: if you want to know which features matter overall, use Kernel SHAP in global mode. If you want to explain one specific prediction, use a SHAP waterfall plot. If you want to do quick exploratory what-if analysis, LIME is fine — just don't use it for final reporting.

For XGBoost or Random Forest: use Tree SHAP for everything. It's faster than Kernel SHAP, gives exact values, and works for both global summaries and local waterfall plots. And again — do not use sklearn's .feature_importances_.

For logistic regression: the coefficients are already the explanation. Globally, you sort them by magnitude. For a single prediction, you compute beta times x for each feature — multiply the coefficient by the actual feature value — and you have the exact contribution of each feature to that prediction. No additional tool is needed.

For any model, if your question is about fairness and bias: run SHAP in global mode and look at the mean absolute SHAP value for protected attributes like age, gender, or race. If any of those features appear near the top of the ranking, that's a red flag that deserves investigation.

The rule of thumb I'd take away from this table: SHAP for anything that goes to a stakeholder or audit. LIME for quick interactive exploration. Coefficients if you're on a white-box model — no add-on needed.

---

## Slide 12 — What This Paper Actually Contributes

Stepping back to the big picture — what does Dwivedi et al. actually contribute that wasn't already in the field?

First, as I said at the start: XAI is not a new algorithm. SHAP, LIME, and PDP all existed before this paper. But the XAI methodoglogy itself is new. The paper's contribution is organizational. It names the field, defines the vocabulary, and places all these scattered tools into a coherent taxonomy. That matters because it gives practitioners and researchers a shared language to talk about what they're doing.

Second, the taxonomy is genuinely actionable. The three axes — global versus local, white box versus black box, model-specific versus model-agnostic — give you a reliable path from "I have this model and this question" to "I should use this specific tool." That decision structure is the paper's most practical gift.

---

## Slide 13 — Open Discussion Questions

I'll wrap up with four questions I'd like to put to the group.

The first one is philosophical: the paper frames XAI as a methodology rather than a new algorithm. Does that framing understate or overstate its contribution? There's a reasonable argument that organizing a field and making existing tools accessible is itself a major contribution. There's also a reasonable argument that a survey paper without novel methods is less impactful than primary research.

The second question sits at the boundary of XAI and ethics. SHAP tells you which features drove a prediction. It does not tell you whether those features are ethically appropriate to use. If age is the top SHAP feature in a loan model, that's an explanation — but is it a justification? Where does XAI's responsibility end and ethical AI's responsibility begin?

The third question is a policy one. There is a real accuracy-interpretability tradeoff in many domains. A logistic regression on a medical dataset will be less accurate than a deep neural net. Should regulators mandate explainability even if it means accepting a less accurate model in healthcare or finance? Where do you draw that line?

And the fourth question looks forward. Everything we've discussed today — SHAP, LIME, the taxonomy — was designed for classical ML models: tabular data, defined features, fixed architectures. LLMs generate free-form text and make decisions in ways that don't map cleanly onto feature importance. Are SHAP and LIME even the right tools for generative AI, or does the field need entirely new explanation methods?

I'll leave it there. Thank you for your time, and I'm happy to open it up for discussion.

---

*Reference: Dwivedi, R., Dave, D., Naik, H., Singhal, S., Omer, R., Patel, P., Qian, B., Wen, Z., Shah, T., Morgan, G., & Ranjan, R. (2023). Explainable AI (XAI): Core Ideas, Techniques, and Solutions. ACM Computing Surveys, 55(9), Article 194. https://doi.org/10.1145/3561048*
