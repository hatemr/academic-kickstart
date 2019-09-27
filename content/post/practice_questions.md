+++
title = "ML Practice Questions"
date = 2019-09-27  #T13:23:10+01:00
draft = false
#tags = ["Getting started"]
categories = []
#markup: mmark
summary = "Practice Questions on Machine Learning"
#disable_comments: true
+++

## Linear Regression

1. What is linear regression?
  * We model a continuous target variable as a linear combination of predictor variables. The "true" relationship need not be linear, we just treat the relationship as linear to be able to predict stuff.

2. How do you train a linear regression model?
  * After assuming the linear relationship, mx+b, you define a loss funciton as the mean squared error. Then you initilize the parameters, compute the loss, and iteratively take steps in the direction which reduces the loss the most. 
  * This direction is defined by the gradient. The gradient points uphill in the steapest direction, so to minimize you go the opposite direction, which is downhill.
  * The gradient is a derivative in multidimensional space. For $mx+b$, there are two parameters so the gradient is a 2d vector.
  * You update the parameters by a step size times the gradient, $w_t = w_{t-1} - \eta \nabla_{w}L$.

3. How do you change linear regression if the target variable is binary?
  * Since the target variable is 0 or 1, you squeeze the prediction to the range 0-1. You squeeze by applying using the logistic function $ \frac{1}{1+d^{-1}} $ to the $w^{T}x + b$ term. Finally, you assume this value represents the probability of the target variable being 1. This is called logistic regression.

4. What else changes in logistic regression? The training? MSE?
  * Since we are now modeling the probability of $y=1$ as a binomial variable where the probability parameter is a linear combination of the predictor variables. When we maximize joint probability, over all samples, we compute the log-loss the function look different from mse: 

  $$\sum_i y_i \log \sigma(w \cdot x+b)+(1-y_i)\log(1-\sigma(w \cdot x+b))$$

5. Why don't we use the MSE for the loss for classification, practically speaking?
  * The "logistic" loss is convex and easier to compute. We prefer to minimize _accuracy loss_ (i.e. "0/1" loss), but this loss is non-convex and thus hard to minimize.
  * Since we can't use the _accuracy loss_, which we really want, we must use a _surrogate_ loss function which approximates the 1/0 loss as closely as possible. The logistic loss is closer to the _accuracy loss_ than the MSE, as shown below [^1]:

![mse v logistic loss](/img/mse_v_logistic_loss.png)

Finally, note that logistic regression uses the logistic loss; training a logistic regression with the MSE loss wouldn't be logistic regression anymore. It would be just a classifier.

## Next up: PCA and Decision Trees
1. What is PCA?

[^1]: [What are the main reasons not to use MSE as a cost functoin for Logistic Regression?](https://www.quora.com/What-are-the-main-reasons-not-to-use-MSE-as-a-cost-function-for-Logistic-Regression)