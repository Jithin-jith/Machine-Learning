# Import necessary libraries
from sklearn.datasets import load_iris            # For loading the iris dataset
from sklearn.model_selection import train_test_split  # For splitting the dataset into train/test sets
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LogisticRegression  # ML algorithm
from sklearn.metrics import accuracy_score, classification_report  # For evaluating the model
import joblib
import numpy as np

# Step 1: Load the iris dataset
iris = load_iris()   # This returns a dictionary-like object with features and labels
X = iris.data        # Features (sepal length, sepal width, petal length, petal width)
y = iris.target      # Target labels (0: Setosa, 1: Versicolor, 2: Virginica)

# Step 2: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale the features (important for algorithms like logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
X_test = scaler.transform(X_test)        # Only transform test data

# Step 4: Create the model and train it
model = LogisticRegression()    # You can also try other models like SVC, DecisionTreeClassifier, etc.
model.fit(X_train, y_train)     # Train the model on the training data

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test)
# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Detailed evaluation

# Optional: Show some predictions
for i in range(5):
    print(f"Predicted: {iris.target_names[y_pred[i]]}, Actual: {iris.target_names[y_test[i]]}")

# save
joblib.dump(model, "logistic_regression.pkl") 
joblib.dump(scaler,"std_scaler.pkl")
