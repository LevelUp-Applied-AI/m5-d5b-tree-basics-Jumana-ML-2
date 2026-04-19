"""
Module 5 Week B — Core Skills Drill: Tree-Based Model Basics

Complete the three functions below.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def train_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        max_depth: Maximum tree depth.
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    #Create and fit a DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X_train, y_train)
    return tree


def get_feature_importances(model, feature_names):
    """Extract feature importances sorted by importance (descending).

    Args:
        model: Fitted tree-based model with feature_importances_ attribute.
        feature_names: List of feature names.

    Returns:
        Dictionary mapping feature name to importance value, sorted descending.
    """
    #Extract importances and return as a sorted dictionary
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))
    return dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))


def train_balanced_forest(X_train, y_train, X_test, y_test,
                          n_estimators=100, random_state=42):
    """Train a RandomForest with balanced class weights and return metrics.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        n_estimators: Number of trees.
        random_state: Random seed.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1'.
    """
    #Train RandomForestClassifier with class_weight='balanced',
    #       predict on test set, compute and return metrics
    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    return metrics


if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    features = ["tenure", "monthly_charges", "total_charges",
                "num_support_calls", "senior_citizen", "has_partner",
                "has_dependents", "contract_months"]
    X = df[features]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Task 1
    tree = train_decision_tree(X_train, y_train)
    if tree:
        print(f"Decision tree trained, depth={tree.get_depth()}")

    # Task 2
    if tree:
        importances = get_feature_importances(tree, features)
        if importances:
            print(f"Top features: {list(importances.items())[:3]}")

    # Task 3
    metrics = train_balanced_forest(X_train, y_train, X_test, y_test)
    if metrics:
        print(f"Balanced RF: {metrics}")
