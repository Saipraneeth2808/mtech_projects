import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
import pickle

# Modeling
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo

def fetch_heart_disease_dataset():
    return fetch_ucirepo(id=45)

heart_disease = fetch_heart_disease_dataset()

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
print(heart_disease.metadata)
print(heart_disease.variables)
print(y)
print(X)


## Step 1: Data Acquisition & Initial Cleaning
## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def prepare_dataframe(X, y):
    # Combine into one DataFrame
    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

    # Inspect the data:
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())  # check missing values

    # Handle missing values:
    # If any column has missing values, decide:
    # Drop rows with missing values (if few).
    # Or impute (mean/median for numeric, mode for categorical).
    df.fillna(df.median(), inplace=True)
    return df

df = prepare_dataframe(X, y)

# Encode categorical features:
# Identify categorical columns (e.g., sex, cp, thal).
# Use OneHotEncoder or pd.get_dummies() for quick EDA
df_encoded = pd.get_dummies(df, drop_first=True)


## Step 2: Basic EDA Visualizations
## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Class balance:
sns.countplot(x='num', data=df)
plt.title('Class Balance')
plt.show()

# Histogram for all the features
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()


# Feature Correlation
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Save dataset
save_dir = os.path.join(os.getcwd(), "mlOps")
os.makedirs(save_dir, exist_ok=True)

# Fetch UCI Heart Disease
ds = heart_disease
X = ds.data.features.copy()
y = ds.data.targets.copy()

# Ensure y is a Series named 'num'
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
y.name = 'num'

# Combine and normalize column names
df = pd.concat([X, y], axis=1)
df.columns = [c.strip().lower() for c in df.columns]

# Optional: make binary target (presence of heart disease)
df['target'] = (df['num'] > 0).astype(int)

# Save files
df.to_csv(os.path.join(save_dir, 'heart.csv'), index=False)               # features + num + target
df.drop(columns=['num']).to_csv(os.path.join(save_dir, 'heart_binary.csv'), index=False)  # binary target only

print('Saved:')
print(' -', os.path.join(save_dir, 'heart.csv'))
print(' -', os.path.join(save_dir, 'heart_binary.csv'))


## Step 3 — Feature Engineering & Model Development
## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

### 3.1 Prepare data & target

# Binary target: presence of heart disease (typical formulation)
# UCI 'num' is 0..4; treat >0 as disease present (1), else absent (0)
df['target'] = (df['num'] > 0).astype(int)

# Feature set (drop original multiclass target)
X = df.drop(columns=['num', 'target'])
y = df['target']


### 3.2 Feature typing
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numeric_cols     = [c for c in X.columns if c not in categorical_cols]  # includes age, trestbps, chol, thalach, oldpeak, ca

print("Categorical:", categorical_cols)
print("Numeric    :", numeric_cols)


### 3.3 Preprocessing pipeline
def build_preprocessor(numeric_cols, categorical_cols):
    # Numeric: median imputation ➜ standard scaling
    # Categorical: most‑frequent imputation ➜ one‑hot (ignore unseen categories in CV)
    numeric_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ('num',  numeric_tf,     numeric_cols),
            ('cat',  categorical_tf, categorical_cols)
        ],
        remainder='drop'  # just to be explicit
    )
    
    return preprocess

preprocess = build_preprocessor(numeric_cols, categorical_cols)


### 3.4 Model candidates & tuning grids
# We’ll optimize primarily for ROC‑AUC (good for imbalanced/binary problems) and report accuracy/precision/recall alongside.

# Logistic Regression: regularization strength C, penalty l2, solver liblinear/saga, with class_weight='balanced'.
# Random Forest: n_estimators, max_depth, min_samples_leaf, max_features, with class_weight='balanced'.

def build_logreg_pipeline(preprocess):
    return Pipeline(steps=[
        ('prep', preprocess),
        ('clf', LogisticRegression(
            max_iter=5000,
            class_weight='balanced'
        ))
    ])


def build_rf_pipeline(preprocess):
    return Pipeline(steps=[
        ('prep', preprocess),
        ('clf', RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

logreg_pipe = build_logreg_pipeline(preprocess)
rf_pipe = build_rf_pipeline(preprocess)

# Hyperparameter grids
logreg_grid = {
    'clf__penalty': ['l2'],
    'clf__solver':  ['liblinear', 'saga'],
    'clf__C':       [0.01, 0.1, 1.0, 3.0, 10.0]
}

rf_grid = {
    'clf__n_estimators':   [200, 400, 800],
    'clf__max_depth':      [None, 5, 10, 20],
    'clf__min_samples_leaf':[1, 2, 4],
    'clf__max_features':   ['sqrt', 'log2']
}


# ## 3.5 Cross‑validation & model selection
# Use StratifiedKFold to preserve class balance across folds. Optimize by ROC‑AUC; keep refit on the best params.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_grid_search(pipe, grid, skf):
    return GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        refit=True
    )

logreg_gs = build_grid_search(logreg_pipe, logreg_grid, skf)
rf_gs = build_grid_search(rf_pipe, rf_grid, skf)


now = datetime.now()
datetime_string = now.isoformat()
scoring = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc_ovr"
}


mlflow.set_tracking_uri("http://host.docker.internal:5000")


def train_and_log_with_mlflow(
    experiment_name,
    run_name,
    grid_search,
    X,
    y,
    scoring,
    artifact_path,
    pkl_name=None
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        grid_search.fit(X, y)

        cv_results = cross_validate(
            grid_search.best_estimator_,
            X, y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1
        )

        metrics = {
            "cv_accuracy": cv_results["test_accuracy"].mean(),
            "cv_precision": cv_results["test_precision"].mean(),
            "cv_recall": cv_results["test_recall"].mean(),
            "cv_f1": cv_results["test_f1"].mean(),
            "cv_roc_auc": cv_results["test_roc_auc"].mean()
        }

        mlflow.log_metrics(metrics)
        mlflow.log_params(grid_search.best_params_)

        mlflow.sklearn.log_model(
            grid_search.best_estimator_,
            artifact_path=artifact_path
        )

        if pkl_name:
            joblib.dump(grid_search.best_estimator_, pkl_name)
            mlflow.log_artifact(pkl_name)

train_and_log_with_mlflow(
    experiment_name="LogisticRegression_GridSearch_" + datetime_string,
    run_name="LogisticRegression_GridSearch_" + datetime_string,
    grid_search=logreg_gs,
    X=X,
    y=y,
    scoring=scoring,
    artifact_path="logRegModel",
    pkl_name="logRegModel.pkl"
)

train_and_log_with_mlflow(
    experiment_name="RandomForest_GridSearch_" + datetime_string,
    run_name="RandomForest_GridSearch_" + datetime_string,
    grid_search=rf_gs,
    X=X,
    y=y,
    scoring=scoring,
    artifact_path="rfModel"
)



# ## 3.6 Final evaluation with stratified cross‑validation
# Report accuracy, precision, recall, ROC‑AUC for each best model using cross_validate. (This evaluates generalization beyond the refit scores.)
scoring = {
    'accuracy':  'accuracy',
    'precision': 'precision',
    'recall':    'recall',
    'roc_auc':   'roc_auc'
}

# Evaluate best Logistic Regression
logreg_cv = cross_validate(
    logreg_gs.best_estimator_,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

# Evaluate best Random Forest
rf_cv = cross_validate(
    rf_gs.best_estimator_,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

def summarize_cv(name, cvres):
    print(f"\n{name} (5-fold CV)")
    for k in ['test_accuracy','test_precision','test_recall','test_roc_auc']:
        print(f"  {k.replace('test_','').upper():10s}: "
              f"mean={cvres[k].mean():.3f}  std={cvres[k].std():.3f}")

summarize_cv("Logistic Regression", logreg_cv)
summarize_cv("Random Forest",      rf_cv)


# ## 3.7 Cross‑validated ROC curve (optional)
# Plot ROC using out‑of‑fold probabilities from cross_val_predict.
# Cross-validated probabilities
y_proba_lr = cross_val_predict(
    logreg_gs.best_estimator_, X, y, cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]

y_proba_rf = cross_val_predict(
    rf_gs.best_estimator_, X, y, cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y, y_proba_rf)
auc_lr = roc_auc_score(y, y_proba_lr)
auc_rf = roc_auc_score(y, y_proba_rf)

plt.figure(figsize=(7,5))
plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={auc_lr:.3f})', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={auc_rf:.3f})', lw=2)
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cross-validated ROC Curves')
plt.legend()
plt.grid(alpha=0.2)
plt.show()


# ## 3.8 Inspect feature contributions
# 
# For Random Forest, you can review global feature importances (after one‑hot).
# For Logistic Regression, inspect coefficients per one‑hot feature.
# RF feature importances
rf_best = rf_gs.best_estimator_
feat_names = rf_best.named_steps['prep'].get_feature_names_out()
importances = rf_best.named_steps['clf'].feature_importances_

fi = pd.DataFrame({'feature': feat_names, 'importance': importances}) \
       .sort_values('importance', ascending=False)
print("\nTop 10 RF features:\n", fi.head(10))

# Logistic coefficients
lr_best = logreg_gs.best_estimator_
coefs = lr_best.named_steps['clf'].coef_.ravel()
lr_df = pd.DataFrame({'feature': feat_names, 'coef': coefs}) \
          .sort_values('coef', ascending=False)
print("\nTop +ve LR features:\n", lr_df.head(10))
print("\nTop -ve LR features:\n", lr_df.tail(10))

def cv_report(name, cvres):
    metrics = ['test_accuracy','test_precision','test_recall','test_roc_auc']
    row = {m.replace('test_',''): (cvres[m].mean(), cvres[m].std()) for m in metrics}
    print(f"\n{name} CV summary")
    for m,(mu,s) in row.items():
        print(f"{m:9s}: {mu:.3f} ± {s:.3f}")

cv_report("Logistic Regression", logreg_cv)
cv_report("Random Forest",      rf_cv)


# # Step 4: Save best model as a PKL file
# 
# As per evaluation metrics (Accuracy, Precision, Recall, ROC/AUC) above, best model is logistic regression.
# Thus saving logistic regression model as a Pickle file.

# Save model as PKL
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(lr_best, f)

print("Best model saved as logreg_model.pkl")
print("Best params:", logreg_gs.best_params_)


# # Sample Prediction of one data sample
sample_patient = {
    "age": 54,
    "sex": 1,
    "cp": 2,
    "trestbps": 180,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 4.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

X_sample = pd.DataFrame([sample_patient])

prediction = lr_best.predict(X_sample)[0]
probability = lr_best.predict_proba(X_sample)[0]
confidence = max(probability)

print(prediction)
print(confidence)

