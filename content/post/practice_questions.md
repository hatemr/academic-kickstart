+++
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
  * After assuming the linear relationship, mx+b, you define a loss funciton as the mean squared error. Then you initilize the parameters, compute the loss, and iteratively take steps in the weight space in the direction which reduces the loss the most. 
  * This direction is defined by the gradient. The gradient points uphill in the steapest direction, so to minimize you go the opposite direction, which is downhill.
  * The gradient is a derivative in multidimensional space. For $mx+b$, there are two parameters so the gradient is a 2d vector.
  * You update the parameters by a step size times the gradient $w^{t} = w^{t-1} - \eta \nabla_{w}L$.

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

## PCA and Decision Trees
1. What is PCA?
  * PCA takes a data matrix and creates a new set of variables that are linear combinations of the original variables. The new variables, or _principal components_ are constructed in such a way that the first principal component has the largest variance possible, followed by the second principal component, and so on. The principal components are all orthogonal to each other. The principal components can be computed by finding the eigenvectors of the data covariance matrix. PCA is mostly useful for dimensionality reduction; reducing the number of variables to a feasible number, while maintaining as much variation in the data as possible.
  * Finding the new axes is the key part of PCA. Once you find the new axes, you can convert new data into the new axes.
  * In T-SNE, we apply the intuition that points close together in 3d should be close in 2d also. We build a neural network to map higher-dimensional points to lower-dimensional space. Minimize the KL divergence of probabilities of points being neighbors.

2. What are decision trees?
  * Decision trees split a feature space into non-overlapping regions, then predicts the average of target values in that region. Decision treess split the feature space by iteratively choosing a variable and value to split the predictions on; the variable and value are chosen according to whichever combination reduces an “impurity measure” the most at each step. Impurity is usually measure as entropy; the extent to which the two groups from the split have mostly one class, instead of a mix of two. The algorithm will continue to split the space until a stopping condition is reached, such as there being a small enough number of observations in the resulting leaves, or the depth of the tree reaches a maximum.

## SVM, Adaboost, and gradient boosting
1. What are SVMs?
  * SVMs are binary classifiers which try to find a hyperplane that splits the two classes in their feature space. Observations lying above the hyperplane are predicted as 1, observations below are predicted as -1. If the data are indeed linearly separable, then the hyperplane is chosen by maximizing the _margin_; the distance of the closest point to the hyperplane. No data points fall within the _margin_.

  * The data are often not linearly separable, so the model is modified to allow observations to violate the margin and the hyperplane. The number and severity of violations allowed is controlled with a tuning parameter, which acts as a "budget" for the amount of violations. When the budget is small, the model has low bias and high variance. High budget means high bias but low variance.

  * For non-linear class boundaries, we could expand the feature space using functions, like polynomials, of the predictors. In the enlarged feature space, the decision boundary that results is linear. However, with many new features, the computations could be unmanageable. The support vector machine allows us to enlarge the feature space in a way that leads to efficient computations.

  * The SVM is an extension of the support vector classifier that results from enlarging the feature space in a specific way, using _kernels_. Kernels are a generalization of the inner product, and they measure the similarity of two observations. Some popular kernels are the polynomial and radial kernels. Kernels are appealing because taking a kernel in the orginal feature space is less expensive than computing an inner product in the expanded feature space.

2. What is Adaboost?
  * Adaboost combine multiple 'base' classifiers to form a model whose performance can be significantly better than that of any of the base classifiers. Overall performance can be good even when performance of each base classifier is only slightly better than random guessing. The base classifiers are trained in sequence, using weighted versions of the data with higher weight going to observations that were classified wrong in the previous classifier in the sequence. Once all the classifiers have been trained, their predictions are combined through a weighted majority voting scheme.


[^1]: [What are the main reasons not to use MSE as a cost functoin for Logistic Regression?](https://www.quora.com/What-are-the-main-reasons-not-to-use-MSE-as-a-cost-function-for-Logistic-Regression)