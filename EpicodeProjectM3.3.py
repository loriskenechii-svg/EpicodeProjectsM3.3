#1. - DATA EXPLORATION E PREPROCESSING

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Caricamemento e analisi del dataset
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

print(df.head())
print(df.describe())
print(df.dtypes)

X = df.drop(columns=["target"])
y = df["target"]

scaler = StandardScaler()
X_std = scaler.fit_transform(X)


#2. - DATA DIMENSIONALITY REDUCTION

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)


#3. - ADDESTRAMENTO E VALUTAZIONE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#1. K-nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)

#2. Support Vector Machine (SVM)
svm = SVC(kernel="linear")

# Addestramento modelli
knn.fit(X_pca, y)
svm.fit(X_pca, y)

# Calcolo e stampo dell'Accuracy Store
knn_pred = knn.predict(X_pca)
svm_pred = svm.predict(X_pca)

knn_accuracy = accuracy_score(y, knn_pred)
svm_accuracy = accuracy_score(y, svm_pred)

print(f"Accuracy KNN: {knn_accuracy}")
print(f"Accuracy SVM: {svm_accuracy}")


#4. ANALISI E VISUALIZZAZIONE

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

DecisionBoundaryDisplay.from_estimator(
    knn, X_pca, ax=ax1, response_method="predict",
    multiclass_colors="viridis", alpha=0.3
)

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
ax1.set_title("KNN")

DecisionBoundaryDisplay.from_estimator(
    svm, X_pca, ax=ax2, response_method="predict",
    multiclass_colors="viridis", alpha=0.3
)

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
ax2.set_title("SVM")

plt.show()