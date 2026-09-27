"""Authorship attribution (Doyle / Dostoyevsky / Austen) from single sentences.

Compares softmax regression and the numpy MLP against the same MLP built in Keras,
all trained on the same TF-IDF features and the same split.

    python scripts/authorship.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nnscratch.models import MLP, SoftmaxRegression  # noqa: E402
from nnscratch.text import TfIdf, load_tsv, split, tokenize  # noqa: E402
from nnscratch.train import AdaGrad, classification_report, fit  # noqa: E402

HIDDEN, EPOCHS, BATCH, L2 = 128, 100, 128, 1e-8


def keras_mlp(n_features, n_classes, X, y, X_valid, y_valid):
    import tensorflow as tf
    tf.random.set_seed(0)
    reg = tf.keras.regularizers.l2(L2)
    model = tf.keras.Sequential([
        tf.keras.Input((n_features,), sparse=True),
        tf.keras.layers.Dense(HIDDEN, activation="sigmoid", kernel_regularizer=reg),
        tf.keras.layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X, y, validation_data=(X_valid, y_valid), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    return model, history.history["val_accuracy"]


def main():
    labels, texts = load_tsv(ROOT / "data" / "authors.tsv")
    authors = sorted(set(labels))
    y = np.array([authors.index(a) for a in labels])
    docs = [tokenize(t) for t in texts]
    train, valid, test = split(len(docs), [0.7, 0.1, 0.2], seed=0)
    counts = {a: int((y == i).sum()) for i, a in enumerate(authors)}
    print(f"{len(docs)} sentences {counts}; train/valid/test = {len(train)}/{len(valid)}/{len(test)}")

    tfidf = TfIdf().fit([docs[i] for i in train])
    X = {name: tfidf.transform([docs[i] for i in idx]) for name, idx in
         [("train", train), ("valid", valid), ("test", test)]}
    Y = {"train": y[train], "valid": y[valid], "test": y[test]}
    print(f"vocabulary: {len(tfidf)} stems")

    results, curves = {}, {}
    candidates = {
        "Softmax regression (numpy)": (SoftmaxRegression(len(tfidf), len(authors), l2=L2), AdaGrad(0.01)),
        "MLP (numpy, from scratch)": (MLP(len(tfidf), HIDDEN, len(authors), l2=L2, seed=0), AdaGrad(0.01)),
    }
    for name, (model, optimizer) in candidates.items():
        print(name, flush=True)
        start = time.time()
        history = fit(model, optimizer, X["train"], Y["train"], EPOCHS, BATCH, on_epoch_end=lambda e, m=model: {
            "valid_acc": float((m.predict(X["valid"]) == Y["valid"]).mean())})
        curves[name] = [row["valid_acc"] for row in history]
        results[name] = classification_report(Y["test"], model.predict(X["test"]), len(authors))
        results[name]["train_seconds"] = time.time() - start

    name = "MLP (Keras)"
    print(name, flush=True)
    start = time.time()
    model, curves[name] = keras_mlp(len(tfidf), len(authors), X["train"], Y["train"], X["valid"], Y["valid"])
    pred = model.predict(X["test"], verbose=0).argmax(axis=1)
    results[name] = classification_report(Y["test"], pred, len(authors))
    results[name]["train_seconds"] = time.time() - start

    print(f"\n{'model':30s} {'accuracy':>9s} {'macro F1':>9s}   per-author F1 ({', '.join(authors)})")
    for name, r in results.items():
        print(f"{name:30s} {r['accuracy']:9.4f} {r['macro_f1']:9.4f}   {[round(f, 3) for f in r['f1']]}")

    out = {"authors": authors, "split_sizes": [len(train), len(valid), len(test)], "vocab": len(tfidf),
           "test": results, "valid_accuracy_curves": curves}
    (ROOT / "results" / "authorship.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
