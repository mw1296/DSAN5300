## HW: Resampling methods, Model selection, and Non-linearity
`Source`: ISLP Chapter 5,6,7 (https://www.statlearning.com/)

**Note**: Both markdown and PDF files are provided, in case you want the plain-text from the markdown file.

**Instructions**: Please complete the following exercises using `python` in a single Jupyter notebook, export the results to HTML or PDF for submission to Canvas. 

**Note**: You do not need to code algorithms from scratch, you can use StatsModels or any other package. 

**Grading**: Each sub-bullet below will have a rubric category for the TA to score and quantify the quality of your submission for this portion of the deliverable , e.g. on 5.1.a there will be a rubric category with something like '100%, 85%, 70%, 55%, 0%'

### Chapter-5

**5.1** We now review $k$-fold cross-validation.

* (a) Explain how $k$-fold cross-validation is implemented.
* (b) What are the advantages and disadvantages of $k$-fold crossvalidation relative to:
  * i. The validation set approach?
  * ii. LOOCV?

**5.2** In Chapter 4, we used logistic regression to predict the probability of default using income and balance on the Default data set. We will now estimate the test error of this logistic regression model using the validation set approach. Do not forget to set a random seed before beginning your analysis.

* (a) Fit a logistic regression model that uses income and balance to predict default.
* (b) Using the validation set approach, estimate the test error of this model. In order to do this, you must perform the following steps:
  * i. Split the sample set into a training set and a validation set.
  * ii. Fit a multiple logistic regression model using only the training observations.
  * iii. Obtain a prediction of default status for each individual in the validation set by computing the posterior probability of default for that individual, and classifying the individual to the default category if the posterior probability is greater than 0.5 .
  * iv. Compute the validation set error, which is the fraction of the observations in the validation set that are misclassified.
* (c) Repeat the process in (b) three times, using three different splits of the observations into a training set and a validation set. Comment on the results obtained.
* (d) Now consider a logistic regression model that predicts the probability of default using income, balance, and a dummy variable for student. Estimate the test error for this model using the validation set approach. Comment on whether or not including a dummy variable for student leads to a reduction in the test error rate.

**5.3** We will now perform cross-validation on a simulated data set.
* (a) Generate a simulated data set as follows:

```
rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = x - 2 * x**2 + rng.normal(size=100)
```

In this data set, what is $n$ and what is $p$ ? Write out the model used to generate the data in equation form.

* (b) Create a scatterplot of $X$ against $Y$. Comment on what you find.
* (c) Set a random seed, and then compute the LOOCV errors that result from fitting the following four models using least squares:
  * i. $Y=\beta_0+\beta_1 X+\epsilon$
  * ii. $Y=\beta_0+\beta_1 X+\beta_2 X^2+\epsilon$
  * iii. $Y=\beta_0+\beta_1 X+\beta_2 X^2+\beta_3 X^3+\epsilon$
  * iv. $Y=\beta_0+\beta_1 X+\beta_2 X^2+\beta_3 X^3+\beta_4 X^4+\epsilon$.

Note you may find it helpful to use the data.frame() function to create a single data set containing both $X$ and $Y$.

* (d) Repeat (c) using another random seed, and report your results. Are your results the same as what you got in (c)? Why?
* (e) Which of the models in (c) had the smallest LOOCV error? Is this what you expected? Explain your answer.
* (f) Comment on the statistical significance of the coefficient estimates that results from fitting each of the models in (c) using least squares. Do these results agree with the conclusions drawn based on the cross-validation results?

### Chapter-6

**6.1** It is well-known that ridge regression tends to give similar coefficient values to correlated variables, whereas the lasso may give quite different coefficient values to correlated variables. We will now explore this property in a very simple setting.

Suppose that $n=2, p=2, x_{11}=x_{12}, x_{21}=x_{22}$. Furthermore, suppose that $y_1+y_2=0$ and $x_{11}+x_{21}=0$ and $x_{12}+x_{22}=0$, so that the estimate for the intercept in a least squares, ridge regression, or lasso model is zero: $\hat{\beta}_0=0$.

* (a) Write out the ridge regression optimization problem in this setting.
* (b) Argue that in this setting, the ridge coefficient estimates satisfy $\hat{\beta}_1=\hat{\beta}_2$.
* (c) Write out the lasso optimization problem in this setting.
* (d) Argue that in this setting, the lasso coefficients $\hat{\beta}_1$ and $\hat{\beta}_2$ are not unique - in other words, there are many possible solutions to the optimization problem in (c). Describe these solutions.

**6.2** In this exercise, we will generate simulated data, and will then use this data to perform forward and backward stepwise selection.

* (a) Create a random number generator and use its normal() method to generate a predictor $X$ of length $n=100$, as well as a noise vector $\epsilon$ of length $n=100$.
* (b) Generate a response vector $Y$ of length $n=100$ according to the model, where $\beta_0, \beta_1, \beta_2$, and $\beta_3$ are constants of your choice.

$$
Y=\beta_0+\beta_1 X+\beta_2 X^2+\beta_3 X^3+\epsilon
$$

* (c) Use forward stepwise selection in order to select a model containing the predictors $X, X^2, \ldots, X^{10}$. What is the model obtained according to $C_p$ ? Report the coefficients of the model obtained.
* (d) Repeat (c), using backwards stepwise selection. How does your answer compare to the results in (c)?
* (e) Now fit a lasso model to the simulated data, again using $X, X^2$, $\ldots, X^{10}$ as predictors. Use cross-validation to select the optimal value of $\lambda$. Create plots of the cross-validation error as a function of $\lambda$. Report the resulting coefficient estimates, and discuss the results obtained.
* (f) Now generate a response vector $Y$ according to the model and perform forward stepwise selection and the lasso. Discuss the results obtained.

$$
Y=\beta_0+\beta_7 X^7+\epsilon,
$$



**6.3** In this exercise, we will predict the number of applications received using the other variables in the College data set.

* (a) Split the data set into a training set and a test set.
* (b) Fit a linear model using least squares on the training set, and report the test error obtained.
* (c) Fit a ridge regression model on the training set, with $\lambda$ chosen by cross-validation. Report the test error obtained.
* (d) Fit a lasso model on the training set, with $\lambda$ chosen by crossvalidation. Report the test error obtained, along with the number of non-zero coefficient estimates.
* (e) Fit a PCR model on the training set, with $M$ chosen by crossvalidation. Report the test error obtained, along with the value of $M$ selected by cross-validation.
* (f) Fit a PLS model on the training set, with $M$ chosen by crossvalidation. Report the test error obtained, along with the value of $M$ selected by cross-validation.
* (g) Comment on the results obtained. How accurately can we predict the number of college applications received? Is there much difference among the test errors resulting from these five approaches?

### Chapter-7

**7.1** Suppose that a curve $\hat{g}$ is computed to smoothly fit a set of $n$ points using the following formula:
$$
\hat{g}=\arg \min _g\left(\sum_{i=1}^n\left(y_i-g\left(x_i\right)\right)^2+\lambda \int\left[g^{(m)}(x)\right]^2 d x\right),
$$

where $g^{(m)}$ represents the $m$ th derivative of $g$ (and $g^{(0)}=g$ ). Provide example sketches of $\hat{g}$ in each of the following scenarios.

* (a) $\lambda=\infty, m=0$.
* (b) $\lambda=\infty, m=1$.
* (c) $\lambda=\infty, m=2$
* (d) $\lambda=\infty, m=3$.
* (e) $\lambda=0, m=3$.

**7.2** Consider two curves, $\hat{g}_1$ and $\hat{g}_2$, defined by
$$
\begin{aligned}
& \hat{g}_1=\arg \min _g\left(\sum_{i=1}^n\left(y_i-g\left(x_i\right)\right)^2+\lambda \int\left[g^{(3)}(x)\right]^2 d x\right), \\
& \hat{g}_2=\arg \min _g\left(\sum_{i=1}^n\left(y_i-g\left(x_i\right)\right)^2+\lambda \int\left[g^{(4)}(x)\right]^2 d x\right),
\end{aligned}
$$

where $g^{(m)}$ represents the $m$ th derivative of $g$.

* (a) As $\lambda \rightarrow \infty$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller training RSS?
* (b) As $\lambda \rightarrow \infty$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller test RSS?
* (c) For $\lambda=0$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller training and test RSS?

**7.3** This question uses the variables dis (the weighted mean of distances to five Boston employment centers) and nox (nitrogen oxides concentration in parts per 10 million) from the Boston data. We will treat dis as the predictor and nox as the response.

* (a) Use the poly() function from the ISLP.models module to fit a cubic polynomial regression to predict nox using dis. Report the regression output, and plot the resulting data and polynomial fits.
* (b) Plot the polynomial fits for a range of different polynomial degrees (say, from 1 to 10), and report the associated residual sum of squares.
* (c) Perform cross-validation or another approach to select the optimal degree for the polynomial, and explain your results.
* (d) Use the bs () function from the ISLP.models module to fit a regression spline to predict nox using dis. Report the output for the fit using four degrees of freedom. How did you choose the knots? Plot the resulting fit.
* (e) Now fit a regression spline for a range of degrees of freedom, and plot the resulting fits and report the resulting RSS. Describe the results obtained.
* (f) Perform cross-validation or another approach in order to select the best degrees of freedom for a regression spline on this data. Describe your results.

**7.4.** This question relates to the College data set.

* (a) Split the data into a training set and a test set. Using out-of-state tuition as the response and the other variables as the predictors, perform forward stepwise selection on the training set in order to identify a satisfactory model that uses just a subset of the predictors.
* (b) Fit a GAM on the training data, using out-of-state tuition as the response and the features selected in the previous step as the predictors. Plot the results, and explain your findings.
* (c) Evaluate the model obtained on the test set, and explain the results obtained.
* (d) For which variables, if any, is there evidence of a non-linear relationship with the response?