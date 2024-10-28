# import pandas as pd
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# import time

# # Load and preprocess data
# data = pd.read_csv("modified_employee_data.csv", encoding="ISO-8859-1", engine="python")
# data = data.drop_duplicates()
# data['avg_training_score'] = data['avg_training_score'].fillna(data['avg_training_score'].mode()[0])
# X = data.drop(columns=['is_promoted', 'employee_id'])
# y = data['is_promoted']

# X = pd.get_dummies(X)
# X = X.apply(pd.to_numeric, errors='coerce')

# # Upsampling with SMOTE
# smt = SMOTE()
# X_up, y_up = smt.fit_resample(X, y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_up, y_up, test_size=0.3, random_state=2)

# # Define models
# models = {
#     "Random Forest": RandomForestClassifier(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "XGBoost": XGBClassifier(verbosity=0),
#     "Logistic Regression": LogisticRegression(max_iter=1000)
# }

# # Scaling features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train and evaluate models
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_pred_prob = model.predict_proba(X_test)[:, 1]

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print(f"\nModel: {name}")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print(f"F1 Score: {f1:.2f}")
#     print(classification_report(y_test, y_pred))

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(f'Confusion Matrix for {name}')
#     plt.grid(False)
#     plt.show()

#     # ROC curve
#     fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#     roc_auc = auc(fpr, tpr)
#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.grid()
#     plt.show()

# # Hyperparameter tuning with GridSearchCV for Random Forest
# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [4, 5, 6, 7, 8],
#     'criterion': ['gini', 'entropy']
# }
# start = time.time()
# RN = RandomForestClassifier()
# Grid_RN = GridSearchCV(estimator=RN, param_grid=param_grid, cv=2)
# Grid_RN.fit(X_train, y_train)
# end = time.time()
# RF_time1 = end - start

# # Final prediction and recall score
# y_pred = Grid_RN.predict(X_test)
# rn1 = round(recall_score(y_test, y_pred, average='micro'), 3)
# print('Recall', round(recall_score(y_test, y_pred, average='micro'), 3), '%')


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.sklearn

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-1", engine="python")
    data = data.drop_duplicates()
    data['avg_training_score'] = data['avg_training_score'].fillna(data['avg_training_score'].mode()[0])
    X = data.drop(columns=['is_promoted', 'employee_id'])
    y = data['is_promoted']
    
    X = pd.get_dummies(X)
    X = X.apply(pd.to_numeric, errors='coerce')
    return X, y

def perform_smote_upsampling(X, y):
    smt = SMOTE()
    return smt.fit_resample(X, y)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def log_top_features(model, X_columns, model_name):
    try:
        if model_name == "Logistic Regression":
            coefficients = model.coef_[0]
            coef_df = pd.DataFrame({
                'Feature': X_columns,
                'Coefficient': coefficients
            }).sort_values(by='Coefficient', key=abs, ascending=False)
            top5 = coef_df.head(5)
            print("\nTop 5 Coefficients for Logistic Regression:")
            print(top5)
            mlflow.log_text(str(top5), "top_5_coefficients.txt")
        else:
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': X_columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            top5 = feat_imp_df.head(5)
            print(f"\nTop 5 Feature Importances for {model_name}:")
            print(top5)
            mlflow.log_text(str(top5), f"top_5_importances_{model_name}.txt")
    except AttributeError:
        print(f"Model {model_name} does not support feature importance logging.")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, param_grids):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Promotion Prediction Models")
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            try:
                print(f"\nTuning {name}...")
                param_grid = param_grids[name]
                grid_search = GridSearchCV(model, param_grid, cv=3)
                
                start = time.time()
                grid_search.fit(X_train, y_train)
                end = time.time()
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                mlflow.log_params(best_params)
                mlflow.log_metric("Training Time", end - start)
                
                y_pred = best_model.predict(X_test)
                y_pred_prob = best_model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1 Score", f1)
                
                print(f"\nModel: {name}")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1:.2f}")
                print(classification_report(y_test, y_pred))
            
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix for {name}')
                plt.grid(False)
                plt.savefig(f'cm_{name}.png')
                mlflow.log_artifact(f'cm_{name}.png') 
                plt.show()
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid()
                plt.savefig(f'roc_{name}.png')
                mlflow.log_artifact(f'roc_{name}.png')
                plt.show()
                
                log_top_features(best_model, X.columns, name)
                
                mlflow.sklearn.log_model(best_model, f"best_model_{name}")
            finally:
                mlflow.end_run()

def main():
    X, y = load_and_preprocess_data("modified_employee_data.csv")
    
    X_up, y_up = perform_smote_upsampling(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_up, y_up, test_size=0.3, random_state=2)
    
    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(verbosity=0),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    param_grids = {
        "Random Forest": {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        },
        "Decision Tree": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        },
        "Logistic Regression": {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear']
        }
    }
    
    X_train, X_test = scale_features(X_train, X_test)
    
    train_and_evaluate_models(X_train, X_test, y_train, y_test, models, param_grids)

if __name__ == "__main__":
    main()
