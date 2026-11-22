import matplotlib.pyplot as plt
import sys
import json
from typing import List, Dict, Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from copy import deepcopy


def plot_history(
    training_data_output: List[Tuple[float, int]],
    validation_data_output: List[Tuple[float, int]],
    graph: Optional[Figure] = None,
    *,
    name: Optional[str] = None,
    train_data_size: int = 45_000,
    validation_data_size: int = 5_000,
    color: Optional[str] = None,
    axes: Optional[Tuple[Axes, Axes]] = None,
) -> Figure:
    if axes is not None:
        ax_loss, ax_acc = axes
        graph = ax_loss.figure
    elif graph is None:
        graph, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        plt.figure(graph)
        ax_loss, ax_acc = graph.axes

    if name is None:
        name = f"Run {len(ax_loss.lines) // 2 + 1}"

    for split, history, size, style in (
        ("train", training_data_output, train_data_size, "-"),
        ("val", validation_data_output, validation_data_size, "--"),
    ):
        epochs = range(1, len(history) + 1)
        losses = [loss / size for loss, _ in history]
        accuracies = [hits / size for _, hits in history]
        label = f"{name} {split}"

        loss_line, = ax_loss.plot(
            epochs, losses, color=color, lw=2, ls=style, label=label,
        )
        color = loss_line.get_color()
        ax_acc.plot(
            epochs, accuracies, color=color, lw=2, ls=style, label=label,
        )

    ax_loss.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax_acc.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (fraction)", ylim=(0, 1))
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)

    return graph

def load_training_output(json_path: str) -> Dict:
    progress_json: Dict = {}
    with open(json_path, "r") as progress:
        progress_json = json.load(progress)

    return progress_json

def main() -> None:
    training_runs = load_training_output("progress.json")
    test_runs = training_runs.pop("test")

    graph, rows = plt.subplots(
        len(training_runs), 2,
        figsize=(12, 2.8 * len(training_runs)),
        squeeze=False, layout="constrained",
    )

    for (name, history), (ax_loss, ax_acc) in zip(training_runs.items(), rows):
        plot_history(
            history["train"], history["val"],
            name=name, axes=(ax_loss, ax_acc),
        )

        ax_loss.set_title(f"{name} – Loss")
        ax_acc.set_title(f"{name} – Accuracy")

    graph.savefig("training_history.png", dpi=200)
    plt.show()

    graph, (ax_loss, ax_acc) = plt.subplots(
        2, 1, figsize=(10, 8), layout="constrained"
    )

    # Mean test loss, lowest to highest
    loss_order = sorted(test_runs, key=lambda name: test_runs[name][2])
    losses = [test_runs[name][2] for name in loss_order]

    bars = ax_loss.bar(loss_order, losses)
    ax_loss.bar_label(bars, fmt="%.3f", padding=3)
    ax_loss.set_title("Mean test loss")
    ax_loss.margins(y=0.15)

    # Test accuracy, lowest to highest
    accuracy_order = sorted(test_runs, key=lambda name: test_runs[name][3])
    accuracies = [test_runs[name][3] * 100 for name in accuracy_order]

    labels = [f"{accuracy:.2f}%" for accuracy in accuracies]
    for i in range(1, len(accuracies)):
        gain = accuracies[i] - accuracies[i - 1]
        labels[i] += f"\n(+{gain:.2f} pp)"

    bars = ax_acc.bar(accuracy_order, accuracies)
    ax_acc.bar_label(bars, labels=labels, padding=3)
    ax_acc.set_title("Test accuracy and gain over the previous bar")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 115)
    ax_acc.set_yticks(range(0, 101, 20))

    for ax in (ax_loss, ax_acc):
        ax.tick_params(axis="x", labelrotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    graph.savefig("test_results.png", dpi=200)
    plt.show()
    

if __name__ == "__main__":
    main()