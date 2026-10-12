import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


def test_training_and_inference_return_binary_predictions():
    """The classifier trains and predicts one binary label per row."""
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])

    model = train_model(features, labels)
    predictions = inference(model, features)

    assert isinstance(model, RandomForestClassifier)
    assert predictions.shape == labels.shape
    assert set(predictions).issubset({0, 1})


def test_model_can_be_saved_and_loaded(tmp_path):
    """A serialized model reloads with its predictions intact."""
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    model = train_model(features, labels)
    model_path = tmp_path / "models" / "model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    np.testing.assert_array_equal(
        inference(model, features), inference(loaded_model, features)
    )


def test_categorical_slice_returns_precision_recall_and_f1():
    """Slice scoring evaluates only rows with the requested category value."""
    train = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "income": [10, 20, 30, 40],
            "salary": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )
    categorical_features = ["group"]
    X_train, y_train, encoder, label_binarizer = process_data(
        train,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)

    metrics = performance_on_categorical_slice(
        train,
        "group",
        "A",
        categorical_features,
        "salary",
        encoder,
        label_binarizer,
        model,
    )

    assert len(metrics) == 3
    assert all(0.0 <= metric <= 1.0 for metric in metrics)
