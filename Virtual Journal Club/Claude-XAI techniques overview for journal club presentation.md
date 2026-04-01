# XAI techniques overview for journal club presentation

**Created:** 3/30/2026 15:15:18
**Updated:** 3/31/2026 20:51:05
**Exported:** 3/31/2026 21:08:47
**Link:** [https://claude.ai/chat/0a27ab29-6e03-4c6d-adb0-21205adc4425](https://claude.ai/chat/0a27ab29-6e03-4c6d-adb0-21205adc4425)

## Prompt:
3/30/2026, 3:15:19 PM

> File: Explainable AI (XAI) Core Ideas Techniques and Solutions.pdf

This is the core literature I will mainly use to generate my presentation for a virtual journal club assignment in one of my data science course. Below in quote attached requirement of this journal club presentation:
What i need help from you is:
step1: lead me through this literature review as i do not want to read the whole article. i would like to first understand the xai techniques methodologies statistics from a data scientist perspective, meaning i want to know details of the technical part.  after I have a clear grasp of this techinicque, we can move on to step2.

"Overview The content in the course lectures typically focuses on fundamentals, theory, and code implementation, rather than exploring the research frontier, or what is happening on the "cutting edge". However, it is also important to get a feel for what is currently technologically relevant, and where the field is heading. To this end, we will be doing a virtual journal club this semester. There will be various remote meeting time-slots of the journal club. In each session, there will be a student led presentation, no longer than 20 minutes. This will be followed by a 20 to 30 minute group discussion with the audience. The goal of the presentations is to summarize the "state of the art" for various applications and to explore the future of the field and how it may effect society. This will also allow you to network, make friends, and meet other students in the program.Instructions Article selection Your group must find, read, and summarize at least one peer-reviewed academic article Make sure each article is directly relevant to the class. IMPORTANT: YOUR ARTICLES AND PRESENTATION MUST FOCUS ON APPLICATIONS, NOT THEORY OR METHODS All articles, slides, and presentations shared with the class MUST be academically and professionally suitable. Avoid any violent, graphic, hateful, or politically biased content. Please use common sense by keeping everything appropriate and academically relevant, if in doubt, ask the professor before sharing. This is an open ended project. You are expected to carefully research the topic and then use storytelling skills to present a concise, informative, engaging, and well-organized narrative. Make sure your presentations flow smoothly and are highly visual, rather than primarily text based. Citation Make sure to cite all sources! Provide citations on any slide in which non-original information is being used URLs are fine for images and websites but please provide full citations for journal and news articles. Sharing articles You MUST share these PDF's with the class at least one week before the start of the time when you are presenting. This will help facilitate a robust Q&A session. https://docs.google.com/spreadsheets/d/1gdL5AnR41GC8p67PGpfLk9qVnLsH4wgSXOvvX1_LF28/edit?usp=sharingLinks to an external site. If the journal is not open-source then please host a PDF on Google drive, making it accessible to GU members, and share a link below Important: This is a student-led assignment. Meaning YOU are responsible for organizing groups and meeting times, finding and sharing the articles, reading them, and presenting them, and taking part in a robust discussion. Motivation: The goal of this assignment is to strengthen the following skills: Self-education: Self-education and the ability to digest academic literature are crucial in data science, since the field evolves so rapidly. Staying current with new methodologies, tools, and theories through academic papers and self-learning enables practitioners to innovate, solve complex problems, and maintain a competitive edge in an ever-changing landscape. Presentation: Presenting complex technical topics to a large group is an essential skill for nearly all modern careers, remember, practice makes perfect! Presenter instructions Group selection You can form your own presentation groups To do so, edit the following google-doc file to claim your time-slot, and share your article https://docs.google.com/spreadsheets/d/1gdL5AnR41GC8p67PGpfLk9qVnLsH4wgSXOvvX1_LF28/edit?usp=sharingLinks to an external site. Topics and week selection are first come first serve Maximum presenter group size: Two students (you can also present individually if you want)"



## Response:
3/30/2026, 3:17:27 PM

I'll walk you through the technical content of this XAI survey from a data scientist's perspective. Let me break down the key techniques systematically.

````plaintext
Tool: read_me
````

Great, let me walk you through this paper systematically from a data scientist's perspective. I'll cover the XAI technique landscape with visuals to make it click.

---

## What is XAI and why does it matter?

The core problem: modern ML models (deep nets, gradient boosting ensembles) are highly accurate but essentially *black boxes*. You can't easily see *why* they made a prediction. XAI is the collection of methods that open that box.

The paper organizes XAI across three key axes:

**Axis 1 — Scope:** Global (explains the whole model's behavior) vs. Local (explains one specific prediction)

**Axis 2 — Model dependency:** Model-specific (works only for one algorithm type, e.g., decision trees) vs. Model-agnostic (works on any ML model)

**Axis 3 — Transparency:** White box (inherently interpretable model, like linear regression) vs. Black box (post-hoc explanation applied after training, like SHAP on a neural net)

Here's how all the major techniques map across this taxonomy:

````plaintext
Tool: show_widget
````

Click any technique to dive deeper. Now let me walk through the most technically important methods one by one.

---

## The two major pillars: Feature-based vs. Example-based techniques

### Feature-based techniques

These answer: *"Which input features drove this prediction, and by how much?"*

**PDP (Partial Dependence Plots)** — You freeze all features except one, vary it across its range, and plot the average model output. Useful for global behavior but assumes feature independence, which is often violated.

**ICE (Individual Conditional Expectation)** — Same idea as PDP but plots one line per data instance instead of the average. You can spot heterogeneous effects that PDP averages away.

**ALE (Accumulated Local Effects)** — The paper's recommended upgrade over PDP. Instead of marginalizing over the full feature distribution, it accumulates *local* differences within small feature intervals. This handles correlated features correctly and is computationally cheaper.

**LIME** — Fits a simple interpretable model (typically linear) in the *local neighborhood* of one data point. It perturbs the input, gets predictions from the black-box model, then trains a local surrogate on those perturbed predictions. The surrogate's coefficients become your explanation. Key weakness: the definition of "neighborhood" is arbitrary, and the linear approximation breaks down for complex nonlinear boundaries.

**SHAP (Shapley Additive Explanations)** — The most rigorous of the group. Rooted in cooperative game theory: it asks "what is each feature's *fair* contribution to the prediction?" It computes the average marginal contribution of a feature across all possible feature subsets. Satisfies desirable axioms (efficiency, symmetry, linearity). Provides both local and global explanations. The tradeoff is computational cost.

````plaintext
Tool: show_widget
````

**SHAP variants for efficiency:** Since computing all 2ⁿ feature subsets is intractable at scale, the paper covers two practical shortcuts: *Kernel SHAP* (uses a weighted linear regression approximation — model-agnostic but slower) and *Tree SHAP* (exploits the tree structure for exact, polynomial-time computation — only works on tree-based models like XGBoost/Random Forest).

---

### Example-based techniques

These answer: *"Can you show me a concrete example that explains this prediction?"*

**Counterfactuals** — "What is the *minimum* change to the input that would flip the prediction?" If a loan is denied, a counterfactual says: "if your income were $5K higher and your debt ratio 0.1 lower, you'd be approved." They work on black-box models and are model-agnostic. Best suited to binary classification; performance degrades with more classes.

**Anchors** — The flip side: "What is the *sufficient* condition that *locks in* this prediction?" Instead of what would change it, anchors tell you what rules guarantee it. Uses If-Then rules (e.g., "IF max heart rate ≤ 138 AND blocked vessels > 1, THEN heart disease is predicted"). More generalizable than LIME within the defined region.

**CEM (Contrastive Explanations Method)** — Combines both directions: it finds *pertinent positives* (features that must be present to keep the prediction) and *pertinent negatives* (features whose removal would flip it). Richer than either counterfactuals or anchors alone.

**Integrated Gradients** — A gradient-based local method for neural networks. It attributes importance to each input feature by integrating the gradients of the model output along a straight path from a baseline input (e.g., all zeros) to the actual input. Satisfies axiomatic properties: completeness (attributions sum to the output difference from baseline) and sensitivity (if a feature changes the output, it gets nonzero attribution).

---

Here's how to pick the right technique for your situation:

````plaintext
Tool: show_widget
````

---

## Key tradeoffs a data scientist needs to know

A few things the paper highlights that are especially important in practice:

**Interpretability vs. accuracy tradeoff** — Simple, inherently interpretable models (linear, decision tree) are easy to explain but sacrifice predictive power on complex data. Black-box models like gradient-boosted ensembles or deep nets get better accuracy but require post-hoc XAI. You often can't have both fully.

**LIME vs. SHAP in practice** — LIME is faster and easier to set up. SHAP is slower but theoretically consistent and additive (SHAP values always sum to the prediction difference from the mean baseline). For KNN models, LIME is better on compute cost. For XGBoost/tree models, Tree SHAP dominates: it's exact and fast. LIME can't natively handle XGBoost's `xgb.DMatrix` input format.

**When features are correlated** — PDP and feature importance both assume feature independence. If your features are correlated (e.g., height and weight), these methods can generate explanations based on impossible data point combinations. ALE plots are the correct choice when you suspect correlation.

**Global surrogate models** — An interesting middle path: train your black-box model, get its predictions, then train a simple interpretable model (decision tree or linear) to approximate those predictions. The interpretable model becomes a proxy explanation. The risk: the surrogate may be accurate on average but wrong for specific subsets of data.

---

That covers the full technical landscape. Do you feel comfortable with the mechanics? Once you confirm, we can move on to **Step 2 — building the actual presentation slides**, where I'll help you structure this into a compelling 20-minute narrative focused on applications (as required by your assignment).



## Prompt:
3/30/2026, 3:19:21 PM

Explain SHAP global feature importance



## Response:
3/30/2026, 3:21:11 PM

Great question — let me walk you through SHAP global feature importance with a concrete, visual explanation.

---

## What does "global" mean in SHAP?

Local SHAP gives you a set of φᵢ values for **one prediction** — how much each feature pushed that specific prediction up or down from the baseline. Global SHAP aggregates those local explanations **across all predictions in your dataset** to describe the model's overall behavior.

The key insight: you're not approximating global behavior from scratch. You're *summarizing* many local explanations. This makes it more faithful than methods like PDP that compute global effects directly.

---

## How global SHAP values are computed

**Step 1 — Compute local SHAP values for every instance.** For each row in your dataset, run SHAP to get a φᵢ value for every feature. If you have 1,000 rows and 10 features, you end up with a 1,000 × 10 matrix of SHAP values.

**Step 2 — Aggregate across instances.** The most common aggregation is mean absolute SHAP: for each feature, take |φᵢ| across all instances and average. Features with larger average |φᵢ| matter more to the model overall. Using the absolute value is important — a feature that sometimes pushes predictions up and sometimes down is still influential.

````plaintext
Tool: show_widget
````

---

## The three global SHAP visualizations

There are three distinct plots used in practice, each answering a different question.

**1. Feature importance bar chart** — The simplest. For each feature, plot mean |φᵢ| across all instances. Gives you a ranking but loses directional and distributional information.

**2. Beeswarm / summary plot** — The most informative global view. Each dot is one instance. The x-axis is the SHAP value (negative = pushes prediction down, positive = pushes up). Dots are colored by feature value (red = high, blue = low). Features are sorted by mean |φᵢ|. This lets you see direction *and* distribution simultaneously.

**3. SHAP dependence plot** — Picks one feature, plots its SHAP value (y-axis) against its raw feature value (x-axis). Reveals nonlinear relationships and interaction effects. Similar to a PDP but plots actual SHAP contributions per instance rather than averaged model outputs.

````plaintext
Tool: show_widget
````

Reading this beeswarm plot: the spread of dots along the x-axis tells you how much influence a feature has (wider = more important). The color tells you the direction of the relationship. For debt ratio, blue dots (high debt) cluster on the left (pushing denials) and red dots (low debt) cluster on the right (pushing approvals) — a clear, monotonic negative relationship with loan approval. Income shows the opposite pattern. Age has a much narrower spread, meaning it barely moves predictions either way.

---

## What global SHAP tells you that other methods don't

**vs. permutation importance** — Permutation importance measures how much model accuracy *drops* when you shuffle a feature. It's a blunt measure that doesn't tell you direction, doesn't show per-instance variation, and is sensitive to correlated features (shuffle one correlated feature, the model partially compensates with the other). SHAP directly measures each feature's contribution to each prediction.

**vs. built-in tree feature importance** (e.g., Gini importance in sklearn Random Forest) — Built-in importance is biased toward high-cardinality features and doesn't account for feature interactions properly. SHAP importance is unbiased and theoretically grounded.

**vs. PDP** — PDP shows the average effect of a feature across its range, but can be misleading when features are correlated (it extrapolates to unrealistic feature combinations). SHAP global summaries are computed from actual predictions on real data, so they never extrapolate.

---

## The one critical limitation

SHAP global importance is an *aggregated* description. It tells you what the model does *on average across your dataset*. It doesn't guarantee that the feature importance structure is the same in every subgroup of the data. A feature might be critical for predicting outcomes for one demographic and irrelevant for another — but the global mean |φᵢ| would blend those together. This is why local SHAP (drilling into individual predictions) remains essential alongside the global view.

Ready to move to Step 2 and build the presentation?



## Prompt:
3/30/2026, 3:26:03 PM

i'm reading up the this 4-square white box vs. black box, my feed back is can you replace some of the technologies with something that we are more familier with? for example, PDP/ICE ALE plots neural-symbolic Anchors, Integrated gradients, Counterfactuals are sth I never use before. We students are familiar with classification models, logistic regression, linear regression, deep learning.



## Response:
3/30/2026, 3:26:48 PM

Good feedback. Let me redo that taxonomy map using techniques you already know, and show where the familiar models sit in the XAI landscape.

````plaintext
Tool: show_widget
````

Here's the mental model that ties it together:

**White box models** (top half) — logistic regression, linear regression, and decision trees explain themselves. For logistic regression, the coefficients *are* the explanation: a positive β means that feature pushes the prediction toward the positive class, and the magnitude tells you how strongly. For a decision tree, you can literally trace the path from root to leaf for any single prediction — no extra tool needed.

**Random forest** sits at an interesting edge. It has built-in feature importance (mean decrease in Gini impurity across all splits), so it partially self-explains globally. But because it's an ensemble of hundreds of trees, you can't trace a single prediction path the way you can with one tree — which is why it starts needing SHAP for local explanations.

**Deep learning** is always in the black box half. The moment you add hidden layers, the internal computation becomes too nonlinear and high-dimensional to read directly. You always need an add-on tool — SHAP or LIME — to understand what it's doing.

**The add-on tools** (bottom half) are model-agnostic, meaning you can bolt them onto anything: a logistic regression, a random forest, or a 50-layer neural net. SHAP gives you both a global summary (average |φᵢ| across your whole dataset) and a local waterfall plot (per-prediction breakdown). LIME only does local explanations, but is faster to compute.

Does this framing make more sense as a starting point? We can now build the presentation around this scaffolding.



## Prompt:
3/30/2026, 3:38:31 PM

looks like the texts, labels on x and y axis are overlapping and hard to read



## Response:
3/30/2026, 3:39:05 PM

Let me fix the spacing and layout so everything is clean and readable.

````plaintext
Tool: show_widget
````

The main changes: rotated axis labels moved further left so they don't crowd the quadrant boxes, column headers have more breathing room at the top, all cards are consistently sized with generous internal padding, and the italic context notes sit cleanly at the bottom of each quadrant without overlapping anything. Everything should be readable now. Does this work better?



## Prompt:
3/30/2026, 3:39:53 PM

thanks, the context inside the boxes are clear, but the labels on the row headers are still overalpping - white box and black box labels.



## Response:
3/30/2026, 3:40:24 PM

Let me fix just the row headers — I'll move them above each row instead of rotating them on the left side.

````plaintext
Tool: show_widget
````

The row headers "White box" and "Black box" now sit as plain horizontal text directly above each row — no rotation, no overlap. Is this cleaner?



## Prompt:
3/30/2026, 3:40:54 PM

i like them sitting at the sides of the row headers which makes the readers easier to understand



## Response:
3/30/2026, 3:41:27 PM

Got it — let me put them back on the sides but fix the overlap by spacing the two labels far enough apart vertically.

````plaintext
Tool: show_widget
````

The two row labels are now rotated on the left side as you wanted, but each one is centered within its own row's vertical space — "White box" anchored to the top half, "Black box" to the bottom half — so they sit cleanly apart with no overlap. Does this look right?



## Prompt:
3/30/2026, 3:41:36 PM

still overlapping, can you put self-explanatory and White box in 2 rows?



## Response:
3/30/2026, 3:42:15 PM

Let me stack "White box" and "self-explanatory" as two separate lines within the rotation.

````plaintext
Tool: show_widget
````

Each row label now has "White box" and "self-explanatory" on two separate lines, both rotating together as a unit — and same for "Black box" and "needs add-on tool" on the bottom row. They should sit cleanly within their respective halves without touching each other. Does that look right now?



## Prompt:
3/30/2026, 3:43:13 PM

they are still stacking. put "white box" and "black box" labels to left a little bit.



## Response:
3/30/2026, 3:43:45 PM

Let me shift both row labels further left so they have their own clear space.

````plaintext
Tool: show_widget
````

"White box" and "Black box" are now at x=10, and "self-explanatory" / "needs add-on tool" are shifted further left at x=−8, giving each line its own lane. Does that separate them cleanly now?



## Prompt:
3/31/2026, 8:27:25 PM

so, based on your summary, this literature or this method is basically to instruct how to select the model based on the needs? i want to be more clear how XAI helps to explain the black box part.



## Response:
3/31/2026, 8:28:16 PM

Great question — and actually this reframes the paper quite importantly. XAI is **not** primarily about model selection. Let me clarify what it's really doing.

---

## The core problem XAI solves

When you train a neural network or XGBoost model, you get a function that takes inputs and produces outputs. But the internal computation is opaque — you have no idea *why* it made a specific prediction. XAI is the set of tools that **reverse-engineers that reasoning after the fact** and presents it in human-readable form.

Think of it this way: the model is a locked box. XAI doesn't open the box — it watches the box's behavior very carefully from the outside and builds an explanation of what's probably happening inside.

Here's a concrete walkthrough of what that looks like in practice:

````plaintext
Tool: show_widget
````

---

So to directly answer your question — XAI doesn't change the model or help you pick one. It sits **on top of a trained black box** and produces a human-readable answer to: *"why did it say that?"*

There are three concrete things this unlocks that you can't do without it:

**1. Debugging** — If debt ratio has an unexpectedly huge SHAP value, you might discover the model is picking up on a data leakage issue or a biased training set. You can only find this with an explanation tool.

**2. Legal and regulatory compliance** — In finance, healthcare, and hiring, regulations (like GDPR in Europe) require that automated decisions be explainable to the person affected. A neural net alone cannot satisfy this — SHAP or LIME output can.

**3. Building trust** — A model with 92% accuracy that nobody understands won't get deployed in a hospital or a bank. XAI provides the audit trail that lets stakeholders actually trust and act on the model's predictions.

The taxonomy diagram we built earlier is really just a map of *which XAI tool to reach for* depending on what kind of model you already have and what kind of explanation you need — global vs. local, white box vs. black box. Does this make the purpose clearer?



## Prompt:
3/31/2026, 8:45:25 PM

okay so XAI is telling us a methodology how to combine the ML methods to understand the blackbox output instead of a new ML method? I mean, i can do the neural network and then run SHAP without knowing XAI,. what has XAI told me new?



## Response:
3/31/2026, 8:46:10 PM

That's a really sharp observation — and honestly, you've put your finger on the most important critique of this paper. Let me give you a direct answer.

---

You're right that **XAI is not a new algorithm**. If you already know SHAP and LIME and how to apply them to your neural network, you are already *doing* XAI — you just weren't calling it that. So what does the XAI label actually add?

There are three things:

---

**1. XAI gives a unified vocabulary and framework**

Before XAI became a field, SHAP, LIME, permutation importance, and saliency maps were scattered across different research communities — statistics, deep learning, NLP — with no common language. XAI packages them under one framework with consistent terminology: global vs. local, model-agnostic vs. model-specific, post-hoc vs. intrinsic.

This matters in practice because when a regulator, a hospital administrator, or a judge asks *"how does your model make decisions?"*, you need a structured answer — not just a SHAP plot you happened to run. XAI provides the vocabulary to have that conversation.

---

**2. XAI defines when explanation is legally and ethically required**

This is the genuinely new contribution — not a technical one, but a sociotechnical one. XAI formalizes the idea that some decisions **must** be explainable, regardless of model accuracy.

````plaintext
Tool: show_widget
````

A data scientist who just runs SHAP for curiosity is doing analysis. A data scientist who runs SHAP because the bank is legally required to justify every loan denial under GDPR is doing XAI — because XAI is the *framework that tells you this is necessary in the first place*.

---

**3. XAI catches something accuracy metrics completely miss**

This is the most technically interesting contribution. Consider this scenario:

- You train a neural net to detect pneumonia from chest X-rays
- It achieves 96% accuracy on your test set
- You deploy it

Without XAI, you're done. With XAI, you run SHAP saliency maps and discover the model learned to detect the **hospital tags and patient ID markers** in the corner of the X-ray — not the lung tissue — because sicker patients happened to come from one hospital more often in your training data. The model is 96% accurate *and completely wrong* about what it's learning.

This is called **Clever Hans** behavior — named after a horse that appeared to do math but was actually reading subtle cues from its trainer. XAI is the only tool that catches it.

---

## So what's the honest answer to your question?

| | What you knew before XAI | What XAI adds |
|---|---|---|
| SHAP on a neural net | Yes, SHAP existed | XAI tells you *when* you're obligated to run it |
| Model accuracy | Always measured | XAI shows accuracy can be high while reasoning is wrong |
| Feature importance | Built into random forest | XAI standardizes it across all model types |
| Debugging | Ad hoc | XAI makes it systematic and auditable |

The paper's real contribution is not a new algorithm — it's a **structured argument** that explanation should be a first-class requirement in any ML system deployed on humans, plus a taxonomy of which tools satisfy that requirement in which situations. For your presentation, that framing — *"we already had the tools, XAI tells us we're obligated to use them"* — is actually a very compelling story.



## Prompt:
3/31/2026, 8:49:27 PM

with so many methods such as shap, lime and pdp, does XAI provides guidlines what are the methods to run to answer waht quedstions or to unearth the black box result of what model? for example, after running neural network, in order to answer the question of reasoning, shall i follow up with SHAP or LIME or other models?



## Response:
3/31/2026, 8:51:05 PM

Yes — and this is actually one of the most useful things the paper does. Let me build you a clean decision guide.

````plaintext
Tool: show_widget
````

Pick your model and your question — the guide gives you the recommended tool, why it's the right choice, what the output looks like, any pitfalls to watch for, and what the alternatives are.

---

The short answer to your question is **yes** — this is exactly the practical guidance XAI provides. The key rules the paper establishes are:

**For neural networks** — always SHAP. No built-in interpretation exists. Use global SHAP for overall behavior, local SHAP waterfall for individual predictions, LIME only for quick interactive exploration (it's less stable).

**For XGBoost / Random Forest** — use Tree SHAP specifically, not generic SHAP. It exploits the tree structure for exact results instead of sampling approximations. And critically: *ignore sklearn's built-in `.feature_importances_`* for final reporting — it's biased toward high-cardinality features.

**For logistic / linear regression** — the coefficients already answer most questions directly. Reaching for SHAP or LIME on a logistic regression is usually unnecessary overhead unless you're comparing it side by side with a black-box model.

This is the decision logic the paper formalizes — and what makes XAI more than just "run SHAP on everything."



## Prompt:
undefined



## Response:
undefined





---
Powered by [Claude Exporter](https://www.ai-chat-exporter.net)