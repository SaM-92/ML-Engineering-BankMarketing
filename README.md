# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
In this project, our goal is to identify the most effective strategies for improving future marketing campaigns for a financial institution. We approach this as a classification problem, where we aim to predict whether a client will subscribe to a term deposit (represented by the variable ‘y’) based on various factors. By doing so, we hope to increase the effectiveness of the institution’s marketing efforts."


**The top-performing model attained an accuracy of 91.5% and was achieved through hyperparameter tuning of a logistic regression model**

## Scikit-learn Pipeline

In this project, we used logistic regression as our classification algorithm and optimized its hyperparameters using the HyperDriveConfig() class from the Azure ML library. Our data was stored in a blob storage container, from which we retrieved, cleaned, and split it into training and test sets. As an example of hyperparameter tuning, we optimized the Regularization Strength and Maximum Iterations parameters



**What are the benefits of the parameter sampler we chose?**

Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. This penalty term encourages the model to have smaller weights, which can help to reduce the complexity of the model and avoid overfitting1. The Regularization Strength parameter controls the amount of regularization applied to the model. By tuning this parameter, you can find the optimal balance between model complexity and performance.

The Maximum Iterations parameter controls the number of iterations used by the optimisation algorithm to find the optimal values for the model parameters. By tuning this parameter, you can ensure that the optimisation algorithm runs for a sufficient number of iterations to find a good solution, without running for too long and wasting computational resources.

Overall, by tuning these hyperparameters, you can improve the performance of your model by finding the optimal balance between model complexity and performance, and by ensuring that the optimisation algorithm runs for a sufficient number of iterations.

We got the following values for the Regularization and Maximum Iterations:
hyperparameters : {"C": 0.2414056777891848, "max_iter": 150}

**What are the benefits of the early stopping policy we chose?**

The early stopping policy we selected, BanditPolicy, can enhance the efficiency of the hyperparameter tuning process by terminating poorly performing runs prematurely. This can save time and computational resources by reducing the number of runs that need to be completed. Early stopping is a vital technique in machine learning because it helps avoid overfitting and enhances the model's ability to generalise to new data.



## AutoML
The AutoML generates a model along with its corresponding hyperparameters, which are automatically tuned to optimise the performance of the model on the given dataset.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

Both models achieved nearly identical accuracies, with logistic regression (91.5%) using hyperdrive slightly outperforming AutoML (91.4%). In terms of architecture, the hyperdrive experiment was focused solely on logistic regression, whereas AutoML encompassed a broader selection of machine learning models while simultaneously performing hyperparameter tuning. It is worth noting that the AutoML experiment was terminated after 30 minutes, which may explain why a better result than the logistic regression experiment was not achieved.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

In the current project, particularly in the hyperdrive section, our focus was exclusively on logistic regression. However, for future endeavors, it is advisable to incorporate other machine learning models to enable comprehensive comparisons. Moreover, in terms of utilizing AutoML, the time constraint was set at a maximum of 30 minutes for the execution period, but this limitation can be removed to provide more time for AutoML to discover the optimal model. Additionally, a potential area for future enhancement lies in performing data preprocessing prior to feeding it into the AutoML pipeline.  

## Proof of cluster clean up
**Image of cluster marked for deletion**

![Alt text](./image/udacity.JPG)




