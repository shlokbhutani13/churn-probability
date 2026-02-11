Results and Model Evaluation

We trained and evaluated two supervised machine learning models to predict customer churn using the Telco Customer Churn dataset: Logistic Regression and Random Forest.

The dataset is class-imbalanced, which reflects a real-world churn prediction scenario where fewer customers leave compared to those who stay.

Performance Summary

Logistic Regression
Accuracy: 0.7381
Precision: 0.5043
Recall: 0.7834
F1 Score: 0.6136
ROC-AUC: 0.8413

Random Forest
Accuracy: 0.7821
Precision: 0.6159
Recall: 0.4759
F1 Score: 0.5370
ROC-AUC: 0.8211

Model Selection

Although Random Forest achieved slightly higher accuracy, Logistic Regression was selected as the final model. This is because Logistic Regression achieved significantly higher recall, meaning it correctly identifies most customers who are likely to churn. In churn prediction, missing a churner is more costly than incorrectly flagging a loyal customer.

Logistic Regression also achieved the highest ROC-AUC score, indicating better overall class separation. Additionally, it is more interpretable, making it suitable for real business decision-making.

Outputs

The training pipeline generates and saves the following artifacts:
Confusion matrix visualization
ROC curve visualization
Detailed classification metrics
Trained model saved as a joblib file

All outputs are stored in the reports and models directories.

Key Takeaways

This project demonstrates a full end-to-end machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and selection. It highlights the importance of choosing evaluation metrics based on business goals rather than relying solely on accuracy.

Tech Stack

Python
Pandas and NumPy
Scikit-learn
Matplotlib and Seaborn
Joblib

Future Improvements

Handle class imbalance using class-weighted training
Perform feature importance analysis
Tune decision thresholds based on business cost
Deploy the model as an API or web application