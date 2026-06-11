import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Select only Income and Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create KMeans Model
kmeans = KMeans(n_clusters=5, random_state=42)

# Predict Clusters
df['Cluster'] = kmeans.fit_predict(X)

# Plot Clusters
plt.figure(figsize=(8,6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

print("\nCustomers in each Cluster:")
print(df['Cluster'].value_counts())

plt.show()
