# SALES-PREDECTION
PREDICT WHETER THE CUSTOMER WILL PURCHSE AN ITE WHEN HE/SHE VISITED TOTHE WEBSIDE BASED ON DIFREN PAREMENTER

this problem belongs to classifications we are going to predict the purchace of customer based on the parameter 0 or1 


# dataset contains the following columns:

customer_id: Unique identifier for each customer. <br>
age: Age of the customer.<br>
gender: Gender of the customer.<br>
annual_income: Annual income of the customer.<br>
last_visited_days_ago: Days since the customer last visited the platform.<br>
session_duration: Duration of the customer's session on the platform (in minutes).<br>
pages_visited: Number of pages visited during the session.<br>
device: Type of device used (e.g., desktop, mobile).<br>
purchase: Target variable indicating whether the customer made a purchase (1) or not (0).<br>



# Missing Values Analysis: Identifies and handles missing values.

Numerical columns are filled with the mean.<br>
Categorical columns are filled with the mode.<br>

# Exploratory Data Analysis (EDA):
Visualizes distributions for numerical variables using histograms.<br>
Analyzes categorical variable distributions with count plots.<br>
Displays a correlation heatmap to explore relationships between numerical variables.<br>
Includes a pairplot to observe patterns between key variables and the target variable.<br>

# Preprocessing: Converts categorical variables into numerical format:
Gender: male to 0, female to 1.<br>
Device: desktop to 0, mobile to 1.<br>
<br>


# install presequrisied packages such as:-
pip install warnings <br>
pip install pandas<br>
pip install numpy <br>
pip install seaborn <br>
pip install matplotlib.pyplot <br>
pip install imbalanced-learn<br>

# import following packages additionally :-

from sklearn.model_selection import train_test_split<br>
from sklearn.preprocessing import StandardScaler<br>
from sklearn.linear_model import LogisticRegression<br>
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier<br>
from sklearn.svm import SVC<br>
from sklearn.neighbors import KNeighborsClassifier<br>
from xgboost import XGBClassifier<br>
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report<br>
from sklearn.impute import SimpleImputer<br>
from imblearn.over_sampling import SMOTE<br>
import pickle<br>




# Created new features: income_per_visit (annual income divided by last visited days) and pages_per_session (pages visited divided by session duration).
Normalized all numerical features using standard scaling.<br>

# Define features and target
X = df.drop(columns=['customer_id', 'purchase'])<br>
y = df['purchase']<br>

# The highest VIF is for pages_per_session (1.72), which is well below the threshold.
No corrective action is needed for multicollinearity in this case.<br>


# Step 2: Handle class imbalance
 by applying SMOTE methods<br>

Split data
X_train, X_test, y_train, y_test
<br>

# models we are trained with :-
models = {<br>
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'k-NN': KNeighborsClassifier()
}



# ensure random state with fine tunning process for respective  model process for good Accuracy 
results for each model  taking parameter with <br>
Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
by train and test  weget the result 

# Best Model: Random Forest with 85% accuracy and f1 score is 84.78668<br>

# then save the model with joblib by using pickle module 


# note :
for testing ,the data should be preprocessas same as training data or else the  result will be differ<br>







# Handle Missing Values: Including missing values in the device column.<br>
# Feature Engineering: Added new features.  <br>
# Train Multiple Models: Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost, k-NN.<br>
# Evaluate Models: Based on Accuracy, Precision, Recall, and F1 Score.<br>
# Save Best Model: Best performing model based on F1 Score.<br>
# Prediction Function: Preprocess and predict on new customer data.<br>

# RESULTS
Model	    F1-Score	Accuracy	ROC-AUC	Training Time<br>
Logistic Regression	0.85	0.88	0.91	Fast<br>
Random Forest	0.90	0.92	0.95	Moderate<br>
XGBoost	   0.89	        0.91	       0.94	        Slow<br>
 (SVM)  	0.88	0.90	0.93	Very Slow<br>
Decision Tree	0.83	0.87	0.88	Fast<br>


BEST MODEL<br>
# RANDOM FOREST <br>

