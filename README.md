# Predictive Analysis of Red Wine Quality

## Abstract
This project utilizes machine learning techniques to classify and predict wine quality based on physicochemical properties. By analyzing a dataset from the UCI Machine Learning Repository, our goal is to develop a model that can provide insights into the criteria that define high-quality wines, assisting both enthusiasts and professionals in making informed choices.

## Documentation

### Introduction
The project focuses on predicting the quality of red wine using their physicochemical properties. This project aims to apply machine learning to establish criteria for selecting high-quality wine. The motivation behind this study stems from the challenges faced by consumers, particularly amateurs, in discerning high-quality wines among the diverse types available. This project aligns with the goal of providing clear criteria for identifying the best wine.

### Problem Statement
The problem at hand is the difficulty in accurately predicting wine quality due to the multitude of variables involved. Amateur wine consumers often lack a clear framework for assessing quality, which this project addresses by using a dataset from the UCI Machine Learning Repository. The dataset includes various continuous and categorical variables, such as acidity, sugar, and alcohol content, which affect wine quality. The project's goal is to determine which features are most predictive of wine quality and to develop models that can accurately classify wines based on these features.

### Significance
This problem is significant as it aims to solve a real-world issue for wine enthusiasts and industry professionals. Accurate wine quality prediction could benefit consumers in making informed purchasing decisions and assist producers in quality control. The ability to predict wine quality effectively can benefit producers by improving their product quality, aid retailers in stock selection, and assist consumers in making more informed purchasing decisions. The project uses machine learning to identify predictive features and evaluate different models, including logistic regression, random forests, and k-nearest neighbors, to establish the best approach for wine quality prediction. The project’s findings, particularly the effectiveness of the random forest classifier, provide insights into how machine learning can improve decision-making in the wine industry.

### Potential Use Cases
- **Wine Producers:** Can use the model to adjust production processes to enhance wine quality.
- **Retailers:** Can better categorize and recommend wines to customers based on quality predictions.
- **Wine Consumers:** Amateur wine enthusiasts can use the model as a guide to selecting wines based on predicted quality rather than price or brand alone.
- **Quality Control Labs:** Labs can adopt the model to perform routine quality checks and ensure compliance with quality standards.

## Setup
In this project, we utilized a red wine quality dataset from the UCI Machine Learning Repository, which contains 1,599 instances, each featuring 12 attributes related to wine quality, such as acidity, sugar, and alcohol content. The dataset includes a quality rating, which serves as the output variable for our predictive models. The data was split into training and test sets in an 80/20 ratio, and outliers were removed using box plots to ensure the dataset's integrity. 

The experimental setup focused on three models: logistic regression with best subset selection, random forest classifier, and k-nearest neighbors. We employed cross-validation for hyperparameter tuning, optimizing parameters for each model to maximize accuracy and other performance metrics. The project was conducted using Python-based libraries on a standard computing setup, allowing us to explore and evaluate different models effectively. The models were selected based on their ability to address the problem at hand, considering key metrics like accuracy, precision, recall, and F1 score.

### Dataset Description
The dataset used in this project is the Wine Quality dataset available from the UCI Machine Learning Repository. It comprises physicochemical (inputs) and sensory (the output) variables based on physicochemical tests. The dataset has two sets, one for red wine and one for white wine. Each set contains variables such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and quality. The 'quality' variable is a score between 0 and 10 given by human experts.

- **Number of Instances:** 1599 for red wine and 4898 for white wine.
- **Number of Attributes:** 12 per set, including the output attribute (quality).
- **Basic Statistics:** Detailed statistics including mean, median, mode, and standard deviation for each attribute are provided in this section.

### Experimental Setup
The models planned for evaluation include logistic regression with best subset selection, K-Nearest Neighbors (KNN), and Random Forests. For each model, the following setups will be considered:

- **Logistic Regression with Best Subset Selection:** Parameters such as 'n_neighbors' and the distance metric will be tuned to optimize performance.
- **K-Nearest Neighbors (KNN):** Parameters such as 'n_neighbors' and the distance metric will be tuned to optimize performance.
- **Random Forests:** Parameters like the number of trees, depth of the trees, and the number of features to consider for the best split will be tuned.

Each model will undergo a rigorous validation process, likely employing techniques such as cross-validation and various performance metrics (accuracy, F1 score, confusion matrix, etc.). 

### Computing Environment
The experiments will be conducted using Python, specifically leveraging libraries such as scikit-learn for machine learning algorithms, pandas for data manipulation, and matplotlib for data visualization. Execution will primarily occur on local machines or cloud-based platforms, depending on the computational resources required.

### Problem Setup
- **Modeling Details for Neural Networks (TBD):** If neural network models are considered later, details about the network structure, number of layers, activation functions, and optimizer choices will be included here.


### Methodology 

Random Forest Classifier:
For the second model, we chose Random Forest Classifier. The Random Forest classifier excels in predicting categorical outcomes such as wine quality because it leverages an ensemble of decision trees, which reduces overfitting and enhances overall accuracy. This method is effective at handling datasets with many variables and complex, non-linear relationships between features, typical of assessments like wine quality. Random Forest also inherently performs feature selection, prioritizing more significant features during training. Moreover, it is robust to noise and does not require input scaling, simplifying the preprocessing steps. Overall, these characteristics make Random Forest a strong choice for modeling and predicting categories based on multiple influencing factors.

The Random Forest Classifier model shown has achieved an overall accuracy of 73%, which is the highest among the three models tested, using the Randomized SearchCV method for hyperparameter tuning across five folds of cross-validation. The best parameters identified were 600 trees ('n_estimators'), a minimum of 5 samples required to split an internal node ('min_samples_split'), and at least 1 sample required at each leaf node ('min_samples_leaf'). From the classification report, it's evident that the model performs well particularly in predicting wine qualities rated 5 and 6, with precision-recall balances resulting in F1-scores of 0.80 and 0.74, respectively. However, the model struggles with the extreme quality categories (3, 4, 7, and 8), likely due to fewer training samples in these categories, as reflected in the support column and very low F1-scores. This discrepancy highlights the model's limitations in handling imbalanced datasets. The confusion matrix further illustrates these strengths and weaknesses, showing concentrated predictions around the middle quality ratings and sparse or inaccurate predictions for the extremes.

k-Nearest Neighbors: 
For the last model, we used k-nearest neighbors (KNN). KNN can classify and predict outcomes by analyzing the labels of the nearest data points in feature space. It is effective for tasks where patterns in data are indicative of similar outcomes. What is more, KNN is very adaptable--it is flexible with both categorical and continuous input variables. It is also inherently non-parametric, whcih means that it makes no assumptions about the underlying data distribution. This can be advantageous in handling real-world, complex datasets. The adaptability and the intuitive nature of KNN made it a reliable choice for our scenario. 

After conducting a grid search for the KNN model, the optimal parameter for n_neighbors was identified as 1. The best cross-validation score achieved was 0.61, indicating a moderate predictive performance. The overall accuracy was 62%, meaning it performed less well than the Random Forest Classifier model. Analysis of the confusion matrix revealed a number of misclassifications, particularly between adjacent quality classes such as 5 and 6, with approximately 30 instances. However, the model was generally effective in capturing true positives, especially in distinguishing among the middle classes (5, 6, and 7). The model's poorer performance in predicting minority classes (3 and 8) is likely attributable to class imbalance, which means that there are certain classes in the dataset that are significantly underrepresented compared to others

### Results and Discussion (TBD)
Since Random Forest Classifier out-performed the other models, we decided to use Random Forest Classifier to see the feature importance. This may generate some insights in wine selection. Based on the feature importance visualization from the Random Forest model, we can draw several conclusions about the relationship between physicochemical properties and red wine quality:

- **Alcohol is Key**: Alcohol content has the highest importance score, suggesting it's the most significant predictor of red wine quality. Higher alcohol levels may be associated with the ripeness of grapes and overall balance, which are indicative of a wine's quality.

- **Impact of Sulphates**: Sulphates, which likely refer to sulfur dioxide used as a preservative, are the second most important feature. Their presence at certain levels could influence the freshness and shelf life of wine, impacting its perceived quality.

- **Role of Acidity**: Volatile acidity and citric acid come next, underscoring the importance of acid content in wine taste and stability. The balance of acidity is crucial for the palate's perception and wine's aging potential.

- **Influence of Sulfur Dioxide**: Total sulfur dioxide is another top feature, highlighting its role in preventing oxidation and maintaining wine's freshness.

In conclusion, the project successfully demonstrated the potential of machine learning for predicting red wine quality based on physicochemical properties. Among the models evaluated, the random forest classifier stood out as the most reliable, achieving the highest accuracy, precision, recall, and F1 score. This model's strength lies in its ability to handle feature interactions, its robustness against overfitting, and its capacity to capture non-linear relationships, all of which are critical for accurate wine quality prediction. The analysis revealed that alcohol, sulphates, and volatile acidity are the top three features that could serve as criteria when choosing high-quality wine.

The findings highlight how machine learning can enhance decision-making in the wine industry, benefiting both producers and consumers. By leveraging predictive modeling, the project provides a robust framework for establishing clear criteria for identifying high-quality wines, thereby addressing the challenges faced by amateur wine consumers and improving the overall wine selection process.

### References (TBD)
