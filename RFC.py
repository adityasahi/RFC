import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                            rand_score, adjusted_rand_score, mutual_info_score, 
                            normalized_mutual_info_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
# Loading data and extracting each type
data = pd.read_csv('dataset.csv')
data = data.set_index('Unnamed: 0')

# Visualization 1: Heatmap of original data 
plt.figure(figsize=(12, 10))
sns.heatmap(data.iloc[:, :100].corr(), cmap='coolwarm') # only did 100 here but will change soon
plt.title('Original Data Correlation Heatmap ')
plt.savefig('original_heatmap.png', bbox_inches='tight')
plt.close()



cancer_types = []
for idx in data.index:
    cancer_type = ''.join([c for c in idx if c.isalpha()]).lower()
    cancer_types.append(cancer_type)


# 0 = BRCA
# 1 = LUAD
# 2 = PRAD
labels = pd.Series(cancer_types, index=data.index)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
brca_indices = labels[labels == 'brca'].index
luad_indices = labels[labels == 'luad'].index
prad_indices = labels[labels == 'prad'].index

#Starting with same amount of samples for each
# 300 for training and 150 for testing
brca_train = brca_indices[:300]
luad_train = luad_indices[:300]
prad_train = prad_indices[:300]

brca_test = brca_indices[300:450]
luad_test = luad_indices[300:450]
prad_test = prad_indices[300:450]

# Combine indices
train_indices = np.concatenate([brca_train, luad_train, prad_train])
test_indices = np.concatenate([brca_test, luad_test, prad_test])


X_train_full = data.loc[train_indices]
X_test_full = data.loc[test_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]
y_test = encoded_labels[np.isin(data.index, test_indices)]


#First using varaince
features = 4800  # Features that varience outputs
variances = data.var() # calculate variance for each feature
top_variance_features = variances.nlargest(features).index #Selects the top features
X_train_var = X_train_full[top_variance_features] # Filter training and testing data
X_test_var = X_test_full[top_variance_features]

#using ANOVA F-test 
# This test focuses on differences between groups
# Shuld work very well for k nearest
#Scored based on ability to seperate classes
features_anova = 1500  # features choses
anova_selector = SelectKBest(f_classif, k=features_anova) #Sort through
X_train_anova = anova_selector.fit_transform(X_train_var, y_train) #Filter for Training and testing
X_test_anova = anova_selector.transform(X_test_var)

#using mutual information to select features
# Focuses on scoring features based on them being similar to another
features2 = 900  # Number of features to select based on mutual information
mi_selector = SelectKBest(mutual_info_classif, k=features2)
X_train_selected = mi_selector.fit_transform(X_train_anova, y_train)
X_test_selected = mi_selector.transform(X_test_anova)

#printing features used at each step
print(f"Number of features after variance selection: {features}")
print(f"Number of features after ANOVA selection: {features_anova}")
print(f"Number of features after mutual information selection: {features2}")


# Visualization 2: Heatmap of selected features
plt.figure(figsize=(12, 10))
selected_df = pd.DataFrame(X_train_selected, 
                         columns=[f"MI_{i}" for i in range(X_train_selected.shape[1])])
sns.heatmap(selected_df.iloc[:, :100].corr(), cmap='coolwarm')
plt.title('Selected Features Correlation Heatmap (First 100 Features)')
plt.savefig('selected_heatmap.png', bbox_inches='tight')
plt.close()


# Perform dimensionality reduction with PCA on the original data and selected data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_full)
X_test_pca = pca.transform(X_test_full)

X_train_scaled_pca = pca.fit_transform(X_train_selected)
X_test_scaled_pca = pca.transform(X_test_selected)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Create a subplot showing results without feature selection
for label in np.unique(y_train):
    axes[0].scatter(X_train_pca[y_train == label, 0], X_train_pca[y_train == label, 1],
        label=f"Class {label}", edgecolor='k', alpha=0.7)
axes[0].set_title('PCA without feature selection')
axes[0].set_xlabel("Component #1")
axes[0].set_ylabel("Component #2")
axes[0].grid(True)
axes[0].legend()

# Create a subplot showing results with feature selection
for label in np.unique(y_train):
    axes[1].scatter(X_train_scaled_pca[y_train == label, 0], X_train_scaled_pca[y_train == label, 1],
        label=f"Class {label}", edgecolor='k', alpha=0.7)
axes[1].set_title('PCA with feature selection')
axes[1].set_xlabel("Component #1")
axes[1].set_ylabel("Component #2")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()


# using pipeline for scaling and knn
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
# parameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__p': [1, 2]  # in case for minkowski metric
}

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with cross-validation
grid_sea = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_sea.fit(X_train_selected, y_train)
print("\nBest parameters found:")
# evaluating the best parameters
best_model = grid_sea.best_estimator_
y_pred = best_model.predict(X_test_selected)

# Evaluation metrics
print("\n=== Best Model Evaluation ===")
print("Best Parameters:", grid_sea.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_sea.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))






# implementing dimensionality reduction here to see what it would do in regards to training the model
print("\n=== Evaluating KNN with PCA-Reduced Features ===")
pca_model = PCA(n_components=2)
X_train_pca = pca_model.fit_transform(X_train_selected)
X_test_pca = pca_model.transform(X_test_selected)

pipeline_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid_pca = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid_pca = GridSearchCV(pipeline_pca, param_grid_pca, cv=cv, scoring='accuracy', n_jobs=-1)
grid_pca.fit(X_train_pca, y_train)
best_pca_model = grid_pca.best_estimator_
y_pred_pca = best_pca_model.predict(X_test_pca)

print("Best PCA Parameters:", grid_pca.best_params_)
print(f"PCA Test Accuracy: {accuracy_score(y_test, y_pred_pca):.4f}")
print("PCA Classification Report:")
print(classification_report(y_test, y_pred_pca, target_names=label_encoder.classes_))




# this is the LDA part
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

print("\n=== Evaluating KNN with LDA-Reduced Features ===")
lda_components = min(len(np.unique(y_train)) - 1, X_train_selected.shape[1])
lda_model = LinearDiscriminantAnalysis(n_components=lda_components)

X_train_lda = lda_model.fit_transform(X_train_selected, y_train)
X_test_lda = lda_model.transform(X_test_selected)

pipeline_lda = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid_lda = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid_lda = GridSearchCV(pipeline_lda, param_grid_lda, cv=cv, scoring='accuracy', n_jobs=-1)
grid_lda.fit(X_train_lda, y_train)
best_lda_model = grid_lda.best_estimator_
y_pred_lda = best_lda_model.predict(X_test_lda)

print("Best LDA Parameters:", grid_lda.best_params_)
print(f"LDA Test Accuracy: {accuracy_score(y_test, y_pred_lda):.4f}")
print("LDA Classification Report:")
print(classification_report(y_test, y_pred_lda, target_names=label_encoder.classes_))






# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Class-wise accuracy
print("\nClass-wise Accuracy:")
cm = confusion_matrix(y_test, y_pred)
for i, cancer in enumerate(label_encoder.classes_):
    class_total = sum(y_test == i)
    class_correct = cm[i, i]
    class_accuracy = (class_correct / class_total) * 100
    print(f"{cancer.upper()}: {class_accuracy:.2f}%")

# Cross-validation scores for best model
print("\nCross-validation scores for best model:")
cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=cv)
print(f"Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# Visualization 3: Accuracy vs k values
results = pd.DataFrame(grid_sea.cv_results_)
k_values = results['param_knn__n_neighbors'].unique()
k_values.sort()
plt.figure(figsize=(10, 6))
for metric in ['euclidean', 'manhattan']:
    subset = results[results['param_knn__metric'] == metric]
    plt.plot(subset['param_knn__n_neighbors'], 
             subset['mean_test_score'],
             marker='o',
             label=f"{metric} distance")

plt.title('KNN Performance by Number of Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean CV Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('knn_performance.png', bbox_inches='tight')
plt.close()

# Clustering evaluation
print("\n=== Clustering Evaluation ===")
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_test_selected)

metrics = {
    'Silhouette': silhouette_score(X_test_selected, cluster_labels),
    'CH': calinski_harabasz_score(X_test_selected, cluster_labels),
    'DBI': davies_bouldin_score(X_test_selected, cluster_labels),
    'RI': rand_score(y_test, cluster_labels),
    'ARI': adjusted_rand_score(y_test, cluster_labels),
    'MI': mutual_info_score(y_test, cluster_labels),
    'NMI': normalized_mutual_info_score(y_test, cluster_labels)
}

print("\nClustering Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# Visualization 4: PCA comparison before/after feature selection
pca = PCA(n_components=2)

# Original data PCA
X_pca_original = pca.fit_transform(StandardScaler().fit_transform(X_train_full))
# Selected data PCA
X_pca_selected = pca.fit_transform(StandardScaler().fit_transform(X_train_selected))

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca_original[:, 0], y=X_pca_original[:, 1], hue=y_train)
plt.title(f'Original Data PCA\n({X_train_full.shape[1]} features)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca_selected[:, 0], y=X_pca_selected[:, 1], hue=y_train)
plt.title(f'Selected Features PCA\n({X_train_selected.shape[1]} features)')

plt.tight_layout()
plt.savefig('pca_comparison.png', bbox_inches='tight')
plt.close()

# Visualization 5: Clustering results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca_selected[:, 0], y=X_pca_selected[:, 1], hue=y_test)
plt.title('True Cancer Type Distribution')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca_selected[:, 0], y=X_pca_selected[:, 1], hue=cluster_labels)
plt.title('KMeans Cluster Assignment')

plt.tight_layout()
plt.savefig('clustering_results.png', bbox_inches='tight')
plt.close()

print("\nAll visualizations saved as PNG files:")
print("- original_heatmap.png")
print("- selected_heatmap.png")
print("- confusion_matrix.png")
print("- knn_performance.png")
print("- pca_comparison.png")
print("- clustering_results.png")
# percent correct
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall KNN Accuracy: {accuracy * 100:.2f}%")

# Create pipeline for Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid for Random Forest
rf_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
}

# Grid Search
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, n_jobs=-1, verbose=1)
rf_grid.fit(X_train_final, y_train)

# Predictions
rf_preds = rf_grid.predict(X_test_final)

# Evaluation
rf_acc = accuracy_score(y_test, rf_preds)
print(f"\n🔍 Best Random Forest Parameters: {rf_grid.best_params_}")
print(f"✅ Random Forest Test Accuracy: {rf_acc:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, rf_preds, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.savefig("rf_confusion_matrix.png")
plt.show()