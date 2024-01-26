import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the data
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Display summary information
print("Summary of the training data:")
print(train_df.info())
print("\nFirst few rows of the training data:")
print(train_df.head())

# Visualize the distribution of classes in training data
#plt.figure(figsize=(8, 6))
#sns.countplot(train_df["label"])
#plt.title('Class Distribution in Fashion-MNIST (Training Data)')
#plt.show()


# Separate features and labels
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

# Apply PCA for dimensionality reduction
n_components = 300  # You can adjust this based on your requirements
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)


# Visualize 2D PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', edgecolor='k', alpha=0.7)
plt.title("2D PCA Visualization of Fashion-MNIST (Training Data)")
plt.show()


# Fit K-Means on the reduced training dataset
kmeans = KMeans(n_clusters=len(np.unique(y_train)))
y_pred_train = kmeans.fit_predict(X_train_pca)

# Evaluate clustering results using classification labels
conf_matrix = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title('Confusion Matrix - Clustering Results on Training Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize 2D PCA with Cluster Assignments
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', edgecolor='k', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Cluster Centers')
plt.title("2D PCA Visualization with KMeans Clusters")
plt.legend()
plt.show()


# Split the training dataset into training and validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=42)


# Train the classifier
classifier = LogisticRegression()
classifier.fit(X_train_split, y_train_split)

# Predict on validation set
y_pred_val = classifier.predict(X_val_split)

# Evaluate the classifier
print("Classification Report on Validation Data:\n", classification_report(y_val_split, y_pred_val))
