# %%
import time
import pandas as pd

from core import RandomForest, compute_oob_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# %%
df = pd.read_csv("../data/diabetes_binary_health_indicators_BRFSS2015.csv")

df.head()

# %%
X = df.drop("Diabetes_binary", axis=1).values
y = df["Diabetes_binary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "max_features": ["sqrt", "log2"],
}

best_score = -1
best_model = None
best_params = None

for n_estimators in param_grid["n_estimators"]:
    for max_depth in param_grid["max_depth"]:
        for max_features in param_grid["max_features"]:

            model = RandomForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=42,
            )

            model.fit(X_train, y_train)
            score = model.oob_score(X_train, y_train)

            print(n_estimators, max_depth, max_features, "->", score)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = (n_estimators, max_depth, max_features)

print("BEST CONFIGURATION:", best_params, best_score)


# %%
importances = compute_oob_importance(best_model, X_train, y_train)

print("Feature importance:")
print(importances)


# %%
start = time.time()
best_model.fit(X_train, y_train)
custom_time = time.time() - start

pred = best_model.predict(X_test)
custom_acc = accuracy_score(y_test, pred)

print("Custom RF:")
print("Accuracy:", custom_acc)
print("Time:", custom_time)


# %%
rf = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    max_features=best_params[2],
    oob_score=True,
    random_state=42,
)

start = time.time()
rf.fit(X_train, y_train)
sk_time = time.time() - start

pred_sk = rf.predict(X_test)
sk_acc = accuracy_score(y_test, pred_sk)

print("\nSklearn RF:")
print("Accuracy:", sk_acc)
print("Time:", sk_time)


# %%
print("Custom:", custom_acc, "| time:", custom_time)
print("Sklearn:", sk_acc, "| time:", sk_time)



