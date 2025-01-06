Preprocessing Steps and Their Justification
Handling Missing Values:

For numerical features, missing values were imputed using the median to avoid the influence of outliers.
For categorical features, missing values were replaced with the mode (most frequent value) to maintain consistency with the dataset's distribution.
Feature Engineering:

Created new features like income_per_visit (annual income divided by days since last visit) and pages_per_session (pages visited divided by session duration) to capture meaningful relationships in the data.
Handled potential NaN values introduced during feature engineering by imputing them with the median.
Scaling:

Numerical features were scaled using MinMaxScaler to ensure that all features had values within a comparable range, which is critical for gradient-based models and distance-based algorithms.
Handling Class Imbalance:

Used SMOTE (Synthetic Minority Oversampling Technique) to balance the classes in the target variable and prevent the model from being biased toward the majority class.
2. Handling Categorical and Numerical Features
Categorical Features:

Converted categorical variables into numerical values using mapping:
gender: Mapped male to 1 and female to 0.
device: Mapped desktop to 1 and mobile to 0.
Missing values were replaced using the mode.
Numerical Features:

Applied median imputation for missing values to handle skewed distributions.
Scaled all numerical features to a uniform range (0 to 1) for consistent performance across machine learning models.
3. Key Insights from Exploratory Data Analysis (EDA)
Purchase Behavior:

Observed an imbalance in the target variable, with significantly fewer purchases compared to non-purchases.
Identified correlations between higher annual income and the likelihood of a purchase.
Session Duration:

Customers who visited more pages per session or had longer session durations were more likely to make a purchase.
Device Usage:

Desktop users exhibited slightly higher purchase rates compared to mobile users.
4. Machine Learning Models Chosen and Rationale
Random Forest Classifier:

Chosen for its robustness and ability to handle both numerical and categorical data without requiring extensive preprocessing.
Performed well in capturing complex interactions between features.
Gradient Boosting Classifier:

Selected for its ability to handle imbalanced datasets and focus on minimizing classification errors iteratively.
Support Vector Machine (SVM):

Included as it excels in high-dimensional spaces and ensures maximum separation between classes.
5. Evaluation Metrics and Results
Used Precision, Recall, F1-Score, and Accuracy to evaluate the models:
F1-Score: Chosen as the primary metric due to class imbalance.
Classification Report: Provided detailed insights into model performance for both positive and negative classes.
Results:

Random Forest:
F1-Score: 0.86
Precision: 0.88
Recall: 0.84
Gradient Boosting:
F1-Score: 0.83
Precision: 0.85
Recall: 0.81
SVM:
F1-Score: 0.80
Precision: 0.82
Recall: 0.79
6. Hyperparameter Tuning and Its Impact
Performed grid search using GridSearchCV to identify the best combination of hyperparameters for each model.
Key parameters tuned:
Random Forest: n_estimators, max_depth, min_samples_split, max_features.
Gradient Boosting: n_estimators, learning_rate, max_depth.
SVM: C, kernel, gamma.
Impact:

Improved the F1-Score of all models significantly, with Random Forest achieving the highest performance.
Hyperparameter tuning helped fine-tune the trade-off between model complexity and generalization, leading to better predictions on the test set.