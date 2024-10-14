# Email Spam Detection System
**Project Overview**

This project focuses on developing an email spam detection system using various machine learning algorithms. The system classifies emails as either "spam" or "ham" (legitimate) based on textual features extracted from email content. The system leverages Natural Language Processing (NLP) techniques and evaluates multiple machine learning models such as Support Vector Machines (SVM), Random Forest, Decision Tree, Naive Bayes, AdaBoost, and Clustering methods to identify the most effective classifier.

The project aims to tackle the limitations of existing spam detection systems by improving accuracy, reducing false positives, and enhancing scalability. The models are evaluated using precision, recall, F1 score, and accuracy metrics.

**Features**

*Text Preprocessing*: Tokenization, stop-word removal, stemming, and feature extraction using TF-IDF.

*Multiple Classifiers*: Implementation of SVM, Decision Tree, Random Forest, Naive Bayes, AdaBoost, K-Means Clustering, and Agglomerative Clustering for spam detection.

*Model Comparison*: Comprehensive comparison of different algorithms to determine the most effective spam classifier.

*Hyperparameter Tuning*: Optimization of model performance through hyperparameter tuning.

*Advanced Evaluation*: Metrics like precision, recall, F1 score, and confusion matrices for robust model performance evaluation.


**Algorithms Used**

1. *Support Vector Machine (SVM)*: A linear classifier effective for high-dimensional text data.

2. *Random Forest*: An ensemble method that builds multiple decision trees for classification, improving accuracy and reducing overfitting.

3. *Decision Tree*: A simple, interpretable model used as a baseline classifier.

4. *AdaBoost*: An ensemble technique that enhances the performance of weak classifiers by focusing on misclassified samples.

5. *Naive Bayes*: A probabilistic classifier based on Bayes' theorem, particularly effective for text-based data.

6. *K-Means Clustering*: An unsupervised learning algorithm used to group emails into clusters for exploratory analysis.

7. *Agglomerative Clustering*: A hierarchical clustering method useful for identifying nested structures in data.

**Evaluation Metrics**

The performance of each algorithm is evaluated based on the following metrics:

*Accuracy*: The overall correctness of the model.

*Precision*: The proportion of correctly identified spam emails out of all emails predicted as spam.

*Recall*: The proportion of correctly identified spam emails out of all actual spam emails.

*F1 Score*: The harmonic mean of precision and recall, balancing the trade-off between the two.

**Results**

The project evaluates each model using different train-test splits (80-20, 70-30, and 60-40) and provides detailed results for each split. Overall, SVM and Random Forest demonstrate the best performance with high accuracy, precision, and recall, while AdaBoost and Decision Tree offer consistent but slightly lower results.

