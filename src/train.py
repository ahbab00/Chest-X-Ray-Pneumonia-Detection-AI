import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from data import load_datasets
from model import build_custom_cnn, build_densenet_model, unfreeze_top

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_callbacks(model_name: str):
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_{model_name}.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_history(history, phase=1, model_name="model"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (train_key, val_key), title in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"],
    ):
        ax.plot(history.history[train_key], label=f"Train {title}")
        ax.plot(history.history[val_key], label=f"Val {title}")
        ax.set_title(f"{title} - Phase {phase}")
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"training_curves_phase{phase}_{model_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data" / "chest_xray"))
    parser.add_argument("--model_type", choices=["cnn", "densenet"], default="densenet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.20)
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    train_ds, val_ds, _, class_weights = load_datasets(args.data_dir, args.val_split)
    print(f"Class weights: {class_weights}")

    if args.model_type == "cnn":
        model = build_custom_cnn()
        model.summary()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(args.model_type),
        )
        plot_history(history, phase=1, model_name=args.model_type)

    else:
        model = build_densenet_model()
        model.summary()
        print("Phase 1: Training classifier head...")

        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(args.model_type),
        )
        plot_history(h1, phase=1, model_name=args.model_type)

        print("Phase 2: Fine-tuning backbone...")
        model = unfreeze_top(model, n_layers=100, new_lr=1e-5)

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(f"{args.model_type}_ft"),
        )
        plot_history(h2, phase=2, model_name=args.model_type)

    print("Training complete.")


if __name__ == "__main__":
    main()

