# Predictive Analysis of Red Wine Quality

## Abstract
This project employs machine learning techniques to predict and predict wine quality based on physicochemical properties. By analyzing the red wine dataset from the UCI Machine Learning Repository, our goal is to develop a model that defines the criteria for high-quality wines, aiding both enthusiasts and professionals in making informed decisions.

## Documentation

### Introduction
The project focuses on predicting the quality of red wine using their physicochemical properties. This project aims to apply machine learning to establish criteria for selecting high-quality wine. The motivation behind this projects comes from the challenges faced by consumers, particularly amateurs, in discerning high-quality wines among the diverse types available. This project aims for providing a systematic approach to quality assessment.

### Problem Statement
The problem at hand is that accurately predicting wine quality is challenging due to the multitude of influential variables, especially for amateur wine consumers who often lack a clear framework for assessing quality. This project leverages a dataset featuring attributes like `acidity`, `sugar`, and `alcohol` to determine the most predictive factors of wine quality and develop models that can classify wines based on these features.

### Significance
This problem is significant as it aims to solve a real-world issue for wine enthusiasts and industry professionals. Accurate wine quality prediction could benefit consumers in making informed purchasing decisions and assist producers in quality control. The ability to predict wine quality effectively can benefit producers by improving their product quality, aid retailers in stock selection, and assist consumers in making more informed purchasing decisions. We explore various machine learning models, including logistic regression, random forests, and k-nearest neighbors, to identify the most effective approach for predicting wine quality. The project’s findings, particularly the effectiveness of the random forest classifier, provide insights into how machine learning can improve decision-making in the wine industry.

### Potential Use Cases
- **Wine Producers:** Use the model to adjust production processes to enhance wine quality.
- **Retailers:** Better categorize and recommend wines to customers based on quality predictions.
- **Amateur Wine Buyers:** Use the model as a guide to selecting wines based on predicted quality rather than price or brand alone.
- **Quality Control Labs:** Adopt the model to perform routine quality checks and ensure compliance with quality standards.

## Setup
We used a dataset from the UCI Machine Learning Repository, containing 1,599 instances of red wine, each with 12 attributes related to quality. Data was split in an 80/20 ratio for training and testing. Outliers were removed to ensure data integrity.

### Dataset Description
The dataset comprises physicochemical (inputs) and sensory (output) variables from tests. It includes variables like `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, and `quality`. The 'quality' variable is a score between 0 and 10 given by human experts.

- **Number of Instances:** 1599 for red wine.
- **Number of Attributes:** 12 per set, including the output attribute (quality).
- **Basic Statistics:** Detailed statistics including mean, median, mode, and standard deviation for each attribute are provided.

### Experimental Setup
The models evaluated are logistic regression with best subset selection, K-Nearest Neighbors (KNN), and Random Forests, each optimized for accuracy through cross-validation and performance metrics:

- **Logistic Regression with Best Subset Selection:** Exhaustive search over all possible combinations of features based on cross-validation accuracy to determine the best subset of wine features.
- **K-Nearest Neighbors (KNN):** Parameters such as 'n_neighbors' and the distance metric will be tuned to optimize performance.
- **Random Forests:** Parameters like 'number of trees', 'depth of the trees', and 'number of features' to consider for the best split will be tuned.

Each model will undergo validation process, employing K-folds cross-validation and various performance metrics (accuracy, F1 score, confusion matrix, etc.) 

### Computing Environment
The experiments will be conducted using Python, specifically leveraging libraries such as scikit-learn for machine learning algorithms, pandas for data manipulation, and matplotlib for data visualization. Execution will primarily occur on local machines.

### Methodology 
1. **Logistic Regression with Best Subset Selection:**
We utilized logistic regression combined with Best Subset Selection, focusing on a robust set of nine predictors: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `sulphates`, and `alcohol`. We integrate Stratified K-Fold cross-validation (K=4) to ensure that each class is evenly represented during model training, thereby maintaining model generalizability and preventing class imbalance.

Performance highlights:
   - Weighted average F1-score: 0.61
   - Overall accuracy: 63%
   - Challenges: Predicting extreme classes due to fewer training samples.

2. **Random Forest Classifier:**
The Random Forest classifier excels in predicting categorical outcomes such as wine quality because it leverages an ensemble of decision trees, which reduces overfitting and enhances overall accuracy. This method is effective at handling datasets with many variables and complex, non-linear relationships between features, typical of assessments like wine quality. Random Forest also inherently performs feature selection, prioritizing more significant features during training. Moreover, it is robust to noise and does not require input scaling, simplifying the preprocessing steps. Overall, these characteristics make Random Forest a strong choice for modeling and predicting categories based on multiple influencing factors.

Performance highlights:
   - Overall accuracy: 73%
   - Strong performance in predicting middle-class qualities with high precision and F1 scores.
   - Struggles with extreme quality categories due to class imbalance.

3. **K-Nearest Neighbors:**
KNN is highly adaptable, effectively handling both categorical and continuous variables, and is non-parametric, making no assumptions about data distribution. This flexibility is particularly advantageous for analyzing complex, real-world datasets. After conducting a grid search for the KNN model, the optimal parameter for n_neighbors was identified as 1.

Performance highlights:
   - Best cross-validation score: 0.61, indicating moderate predictive performance
   - Overall accuracy: 62%, slightly lower than the Random Forest Classifier model
   - Misclassifications noted between adjacent quality classes, such as 5 and 6, with about 30 instances
   - Effective in capturing true positives, particularly in middle classes (5, 6, and 7)
   - Challenges in predicting minority classes (3 and 8) due to class imbalance

### Results
The evaluation of our models revealed that the Random Forest Classifier significantly outperformed the Logistic Regression and K-Nearest Neighbors models. The key findings include:

- **Main Results**: The Random Forest Classifier achieved the highest overall accuracy of 73%, showcasing its strength in handling complex, non-linear relationships among features and its robustness against overfitting. This model also displayed high precision and recall in the mid-quality categories of wine, with F1-scores notably higher in these groups compared to others.

- **Supplementary Results**: Key parameter choices in the Random Forest model included the number of trees (n_estimators set to 600), and the minimum number of samples required to split a node (min_samples_split set to 5). These parameters were optimized through Randomized SearchCV, enhancing the model's ability to generalize across different subsets of the data effectively.

### Discussion
The superior performance of the Random Forest Classifier can be attributed to its ensemble method, which integrates multiple decision trees to reduce variance and bias, thereby improving prediction accuracy. Feature importance analysis highlighted that alcohol, sulphates, and volatile acidity are the most significant predictors of wine quality. These findings align with known wine science, where these factors are critical in defining the taste, balance, and preservation of wine.
- Comparison: Compared to existing methods, our model's ability to systematically evaluate and rank the importance of different physicochemical properties provides a more empirical approach to wine quality assessment than traditional tasting and scoring, which can be subjective.
- Challenges and Improvements: While our model performs well on average, it struggles with extreme quality categories, likely due to the underrepresentation of these classes in the dataset. Future work could improve upon this by incorporating a more balanced dataset or applying techniques like SMOTE for synthetic data generation to better train the model on minority classes.

### Conclusion
This project successfully leveraged machine learning to predict red wine quality based on physicochemical properties, with the Random Forest Classifier emerging as the most effective model. We demonstrated that machine learning could provide a quantifiable and objective basis for wine quality assessment, which is traditionally subjective. By identifying key physicochemical properties that influence wine quality, our findings offer valuable insights that can assist wine producers, consumers, and retailers in making informed decisions. The project not only addressed the challenges faced by amateur wine consumers but also enhanced the overall wine selection process, contributing to the broader field of food science and quality control.

### References
1. Dataset:
UCI Machine Learning Repository: Wine Quality Data Set. https://archive.ics.uci.edu/ml/datasets/wine+quality
2. Papers:
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016
Forina, M., et al. (1986). Multivariate data analysis as a discriminating method of the origin of wines. Vitis, 25, 189-201.
3. Kaggle Notebooks:
Wine Quality Prediction Models. https://www.kaggle.com/code/rajyellow46/wine-quality
4. GitHub Repositories:
Wine Quality Prediction using Machine Learning. GitHub repository by user1234. https://github.com/user1234/wine-quality-prediction
Scikit-learn Machine Learning Examples. https://github.com/scikit-learn/scikit-learn/tree/main/examples
5. Blog Posts:
"How Machine Learning Can Help in Deciding the Quality of Wines." https://www.datacamp.com/community/tutorials/wine-quality-machine-learning
"Predicting Wine Quality with Several Classification Techniques." https://towardsdatascience.com/predicting-wine-quality-with-several-classification-techniques-179038ea6434
