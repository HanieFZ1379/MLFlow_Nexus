# Step 1| Import Libraries
# 1. to handle the data
import pandas as pd
import numpy as np
from scipy.stats import boxcox

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler
# 4. Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# 5. For Classification task.
from sklearn.neighbors import KNeighborsClassifier
# 6. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 7. Save Model
import pickle
# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Step 2 | Read Dataset
df = pd.read_csv("./assets/dataset/heart.csv")
print(df.head())

# Step 3 | Dataset Overview & Preparation
## Step 3.1 | Rename Variables
df.rename(columns={
    "age":"Age",
    "sex":"Sex",
    "cp":"ChestPain",
    "trestbps":"RestingBloodPressure",
    "chol":"Cholesterol",
    "fbs":"FastingBloodSugar",
    "restecg":"RestingECG",
    "thalach":"MaxHeartRate",
    "exang":"ExcerciseAngina",
    "oldpeak":"OldPeak",
    "slope":"STSlope",
    "ca":"nMajorVessels",
    "thal":"Thalium",
    "target":"Status"
}, inplace=True)

mappings = {
    'Sex': {0: "Female", 1: "Male"},
    'ChestPain': {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic"
    },
    "FastingBloodSugar": {0:False, 1:True},
    "RestingECG": {
        0:"Normal",
        1:"Abnormality",
        2:"Hypertrophy"
    },
    "ExcerciseAngina": {0:"No", 1:"Yes"},
    "STSlope": {
        0:"Upsloping",
        1:"Flat",
        2:"Downsloping"
    },
    "Thalium": {
        0:"Normal",
        1:"Fixed defect",
        2:"Reversible defect",
        3:"Not described"
    },
    "Status": {0:"No Disease", 1:"Heart Disease"}
}

def map_values(x, mapping):
    return mapping.get(x, x)

df_copy = df.copy()
for feature, mapping in mappings.items():
    df_copy[feature] = df_copy[feature].map(lambda x: map_values(x, mapping))
    df_copy[feature] = df_copy[feature].astype(object)

print("_" * 100)
print(df.head())
print("_" * 100)
print(df_copy.head())

## Step 3.2 | Basic Information
print("_" * 100)
print(df_copy.info())
print("_" * 100)
print(df_copy.shape)

## Step 3.3 | Statistical Summary
stats_heart_df = df_copy.copy()
print("_" * 100)
print("*********** Numerical Features ************")
print(stats_heart_df.describe().T)
print("_" * 100)
print("*********** Categorical Features ************")
print(stats_heart_df.describe(include="object").T)
# Step 4 | Exploratary Data Analysis (EDA)
heart_df_eda = df_copy.copy()

## Step 4.1 | Univariate Analysis
# find outliers using IQR method
def find_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_condition = (data < lower_bound) | (data > upper_bound)
    outliers = data[outlier_condition]
    
    return outliers, lower_bound, upper_bound

outliers = {}

def skewness_dist(df, variable):
    skewness = df[variable].skew()

    print(f"Skewness of the {variable} variable: {skewness:.3f}")

    if skewness > 0:
        print("The distribution is right-skewed.")
    elif skewness < 0:
        print("The distribution is left-skewed.")
    else:
        print("The distribution is approximately symmetric.")

# Feature: Age
print("_" * 100)
skewness_dist(heart_df_eda, "Age")
#Outliers of Age variable
age_outliers, age_lower_bound, age_upper_bound = find_outliers(heart_df_eda['Age'])
print("Lower Bound:", age_lower_bound)
print("Upper Bound:", age_upper_bound)
print("Outliers:", len(age_outliers))
outliers.update({"Age":len(age_outliers)})
#There are no outliers in Age variable.

# Feature: RestingBloodPressure
print("_" * 100)
skewness_dist(heart_df_eda, "RestingBloodPressure")
#Outliers of Resting Blood Pressure column:
rbp_outliers, rbp_lower_bound, rbp_upper_bound = find_outliers(heart_df_eda['RestingBloodPressure'])
print("Lower Bound:", rbp_lower_bound)
print("Upper Bound:", rbp_upper_bound)
print("Outliers:", len(rbp_outliers))
outliers.update({"RestingBloodPressure":len(rbp_outliers)})

# Feature: Cholesterol
print("_" * 100)
skewness_dist(heart_df_eda, "Cholesterol")
ch_outliers, ch_lower_bound, ch_upper_bound = find_outliers(heart_df_eda['Cholesterol'])
print("Lower Bound:", ch_lower_bound)
print("Upper Bound:", ch_upper_bound)
print("Outliers:", len(ch_outliers))
outliers.update({"Cholesterol":len(ch_outliers)})

# Feature: MaxHeartRate
print("_" * 100)
skewness_dist(heart_df_eda, "MaxHeartRate")
ecg_outliers, ecg_lower_bound, ecg_upper_bound = find_outliers(heart_df_eda['MaxHeartRate'])
print("Lower Bound:", ecg_lower_bound)
print("Upper Bound:", ecg_upper_bound)
print("Outliers':", len(ecg_outliers))
outliers.update({"MaxHeartRate":len(ecg_outliers)})

# Feature: OldPeak
print("_" * 100)
skewness_dist(heart_df_eda, "OldPeak")
op_outliers, op_lower_bound, op_upper_bound = find_outliers(heart_df_eda['OldPeak'])
print("Lower Bound:", op_lower_bound)
print("Upper Bound:", op_upper_bound)
print("Outliers':", len(op_outliers))
outliers.update({"OldPeak":len(op_outliers)})

print(outliers)

## Step 4.2 | Bivariate Analysis
numerical_features = ['Age', 'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'OldPeak']
categorical_features = ['Sex', 'ChestPain', 'FastingBloodSugar', 'RestingECG', 'ExcerciseAngina', 'STSlope', 'Thalium', 'nMajorVessels']

# Step 5 | Preprocessing
## Step 5.1 | Handling Outliers
outliers_df = pd.DataFrame(list(outliers.items()), columns=['Variable', 'Outliers'])
print("_" * 100)
print(outliers_df)

def box_cox_transform(heart_df):
    transformed_df = heart_df.copy()
    features_to_transform = ["Age", "RestingBloodPressure", "Cholesterol", "MaxHeartRate", "OldPeak"]

    for feature in features_to_transform:
        if np.any(heart_df[feature] <= 0):
            min_value = abs(heart_df[feature].min()) + 1
            heart_df[feature] += min_value
        transformed_feature, lambda_value = boxcox(heart_df[feature])
        transformed_df[feature] = transformed_feature
    return transformed_df

transformed_df = box_cox_transform(df_copy)

# Age variable 
print("_" * 100)
skewness_dist(heart_df_eda, "Age")
skewness_dist(transformed_df, "Age")

#Max Heart Rate variable
print("_" * 100)
skewness_dist(heart_df_eda, "MaxHeartRate")
skewness_dist(transformed_df, "MaxHeartRate")
ecg_outliers_bc, ecg_lower_bound_bc, ecg_upper_bound_bc = find_outliers(transformed_df['MaxHeartRate'])
print("Lower Bound:", ecg_lower_bound_bc)
print("Upper Bound:", ecg_upper_bound_bc)
print("Outliers:", len(ecg_outliers_bc))

#Resting Blood Pressure variable 
print("_" * 100)
skewness_dist(heart_df_eda, "RestingBloodPressure")
skewness_dist(transformed_df, "RestingBloodPressure")
rbp_outliers_bc, rbp_lower_bound_bc, rbp_upper_bound_bc = find_outliers(transformed_df['RestingBloodPressure'])
print("Lower Bound:", rbp_lower_bound_bc)
print("Upper Bound:", rbp_upper_bound_bc)
print("Outliers:", len(rbp_outliers_bc))

#Cholesterol variable 
print("_" * 100)
skewness_dist(heart_df_eda, "Cholesterol")
skewness_dist(transformed_df, "Cholesterol")
ch_outliers_bc, ch_lower_bound_bc, ch_upper_bound_bc = find_outliers(transformed_df['Cholesterol'])
print("Lower Bound:", ch_lower_bound_bc)
print("Upper Bound:", ch_upper_bound_bc)
print("Outliers:", len(ch_outliers_bc))
# Old Peak variable 
print("_" * 100)
skewness_dist(heart_df_eda, "OldPeak")
skewness_dist(transformed_df, "OldPeak")
op_outliers_bc, op_lower_bound_bc, op_upper_bound_bc = find_outliers(transformed_df['OldPeak'])
print("Lower Bound:", op_lower_bound_bc)
print("Upper Bound:", op_upper_bound_bc)
print("Outliers:", len(op_outliers_bc))
transformed = transformed_df.copy()
## Step 5.2 | Missing Values
print("_" * 100)
print("Missing Values")
print(df.isnull().sum())
## Step 5.3 | Duplicated Values
duplicated_rows = df.duplicated()
print("_" * 100)
print(df[duplicated_rows])
print("_" * 100)
print(transformed[duplicated_rows])
df.drop(index=164, axis=0, inplace=True)
transformed.drop(index=164, axis=0, inplace=True)

# Step 6 | Feature Scaling
X = df.drop(["Status"], axis=1)  
y = df["Status"] 
col = list(df.columns.drop("Status"))
sc = StandardScaler()
X[col] = sc.fit_transform(X[col])
print("_" * 100)
print(X.head())
#Splitting the data into the training and testing set  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42, shuffle= True)
print("_" * 100)
print("Shape of Train sets:", X_train.shape)
print("Shape of Test sets:", X_test.shape)
print("Shape of Train set Labels", y_train.shape)
print("Shape of Test set Labels",y_test.shape)
print("_" * 100)
print("Train Label:\n",pd.DataFrame(y_train).value_counts())
print("_" * 100)
print("Test Label:\n",pd.DataFrame(y_test).value_counts())

# Step 7 | Modeling
clf_knn=KNeighborsClassifier()
parametrs_knn={'n_neighbors':[3,5,7, 9, 11], 'metric':['euclidean','manhattan','chebyshev'], 'weights': ['uniform', 'distance']}
grid_clf_knn=GridSearchCV(clf_knn, parametrs_knn, cv=5, n_jobs=-1)
grid_clf_knn.fit(X_train, y_train)
# Conditional check to confirm model training and best estimator selection
if grid_clf_knn.best_estimator_:
    # Save the trained model to a file
    with open('heart_diagnosis_disease_model.pkl', 'wb') as f:
        pickle.dump(grid_clf_knn.best_estimator_, f)
    print("Model saved successfully.")
    best_model_knn=grid_clf_knn.best_estimator_
    y_pred_knn=best_model_knn.predict(X_test)
else:
    print("Model training was not successful; no model to save.")  

ac_knn = accuracy_score(y_test, y_pred_knn)
cr_knn = classification_report(y_test, y_pred_knn)
print("Accuracy score for model " f'{best_model_knn} : ',ac_knn)
print("-" * 100)
print("classification_report for model " f'{best_model_knn} : \n',cr_knn)



# Cross-validation scores for model generalization check

cv_scores = cross_val_score(best_model_knn, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", cv_scores.mean())
cm_rnf = confusion_matrix(y_test, y_pred_knn)








# For model tracking
import mlflow
import mlflow.sklearn
import platform
import psutil
from datetime import datetime
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name, model_name):
    """
    Registers the best model from a given experiment in MLflow's Model Registry.
    
    Parameters:
        experiment_name: Name of the MLflow experiment
        model_name: Name to register the model under
    Returns:
        registered_model_version: The registered model version object
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        
        client = MlflowClient()
        
        # Get experiment ID
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Creating experiment: {experiment_name}")
            mlflow.create_experiment(experiment_name)
            experiment = client.get_experiment_by_name(experiment_name)
        
        # Search runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"]  # Order by accuracy descending
        )
        
        if not runs:
            raise Exception("No runs found in the experiment")
        
        # Get the run with the highest accuracy
        best_run = runs[0]
        
        # Create a registered model if it doesn't exist
        try:
            client.create_registered_model(model_name)
            print(f"Created new registered model '{model_name}'")
        except Exception as e:
            print(f"Model '{model_name}' already exists")
        
        # Register the model version
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{best_run.info.run_id}/heart_disease_model",
            run_id=best_run.info.run_id
        )
        
        # Transition the model version to 'Production' stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        
        print(f"Model registered successfully (Version: {model_version.version})")
        print(f"Model metrics from best run:")
        print(f"Accuracy: {best_run.data.metrics.get('accuracy', 'N/A')}")
        print(f"CV Mean: {best_run.data.metrics.get('cv_mean', 'N/A')}")
        
        return model_version
    
    except Exception as e:
        print(f"Error registering model: {e}")
        return None

def log_metrics_to_mlflow(model, X, y, X_test, y_test, y_pred, cv_scores, best_params):
    """
    Logs detailed metrics and parameters to MLflow.
    """
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('Heart Disease Prediction')
    
    with mlflow.start_run():
        # Log system information
        mlflow.log_param("os_name", platform.system())
        mlflow.log_param("os_version", platform.version())
        mlflow.log_param("processor", platform.processor())
        mlflow.log_param("ram_gb", round(psutil.virtual_memory().total / (1024 ** 3), 2))
        mlflow.log_param("runtime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Log model parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Get detailed classification metrics
        cr_dict = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Log basic metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        
        # Log metrics for each class
        for class_label in ['0', '1']:  # For binary classification
            prefix = f"class_{class_label}"
            mlflow.log_metric(f"{prefix}_precision", cr_dict[class_label]['precision'])
            mlflow.log_metric(f"{prefix}_recall", cr_dict[class_label]['recall'])
            mlflow.log_metric(f"{prefix}_f1", cr_dict[class_label]['f1-score'])
            mlflow.log_metric(f"{prefix}_support", cr_dict[class_label]['support'])
        
        # Log macro and weighted averages
        for avg_type in ['macro avg', 'weighted avg']:
            prefix = avg_type.replace(' ', '_')
            mlflow.log_metric(f"{prefix}_precision", cr_dict[avg_type]['precision'])
            mlflow.log_metric(f"{prefix}_recall", cr_dict[avg_type]['recall'])
            mlflow.log_metric(f"{prefix}_f1", cr_dict[avg_type]['f1-score'])
            mlflow.log_metric(f"{prefix}_support", cr_dict[avg_type]['support'])
        
        # Log confusion matrix values
        mlflow.log_metric("true_negatives", cm[0][0])
        mlflow.log_metric("false_positives", cm[0][1])
        mlflow.log_metric("false_negatives", cm[1][0])
        mlflow.log_metric("true_positives", cm[1][1])
        
        # Log cross-validation metrics
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{i+1}", score)
        
        # Log the model with input example and signature
        input_example = X_test.iloc[:5]  # First 5 rows as input example
        signature = mlflow.models.infer_signature(X_test, y_pred)
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="heart_disease_model", 
            signature=signature, 
            input_example=input_example
        )

# Update the function call in your main script
log_metrics_to_mlflow(
    model=best_model_knn,
    X=X,  # Full dataset
    y=y,  # Full labels
    X_test=X_test,  # Test features
    y_test=y_test,  # Test labels
    y_pred=y_pred_knn,
    cv_scores=cv_scores,
    best_params=grid_clf_knn.best_params_
)

# Then, register the best model
registered_model = register_best_model(
    experiment_name="Heart Disease Prediction",
    model_name="heart_disease_classifier"
)