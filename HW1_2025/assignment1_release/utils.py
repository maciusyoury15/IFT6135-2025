import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def generate_plots(model, list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert model in ['mlp', 'resnet18', 'mlpmixer']
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)

    titles = {
        'train_accs': "Training Accuracy Over Epochs",
        'valid_accs': "Validation Accuracy Over Epochs",
        'train_losses': "Training Loss Over Epochs",
        'valid_losses': "Validation Loss Over Epochs"
    }
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        ax.set_title(f'{model}: {titles[yaxis]}')
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))
        

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss 
    """
    exp_logits = torch.exp(logits)  # Exponentiate
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)  # Sum over classes
    probs = exp_logits / sum_exp_logits  # Normalize to get probabilities

    # Compute negative log likelihood for correct class probabilities
    true_class_probs = probs[torch.arange(labels.shape[0]), labels]  # Extract true class probs
    loss = -torch.log(true_class_probs + 1e-12)  # Compute loss

    return loss.mean()  # Mean loss over batch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc


def comparison_efficiency_by_layer_type(comprehensive_results: list, logdir: str):
    # Visualization
    plt.figure(figsize=(15, 10))

    # Accuracy Comparison
    plt.subplot(2, 2, 1)
    names = [result["configuration_name"] for result in comprehensive_results]
    best_accuracies = [result["best_validation_accuracy"] for result in comprehensive_results]
    plt.bar(names, best_accuracies)
    plt.title('Best Validation Accuracy by Layer Width')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    # Training vs Validation Accuracy Plots
    plt.subplot(2, 2, 2)
    for result in comprehensive_results:
        plt.plot(result["validation_accuracies"], 
                 label=f'{result["configuration_name"]} (Val)')
        plt.plot(result["train_accuracies"], 
                 linestyle='--', 
                 label=f'{result["configuration_name"]} (Train)')
    plt.title('Training and Validation Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Loss Comparison
    plt.subplot(2, 2, 3)
    for result in comprehensive_results:
        plt.plot(result["validation_losses"], 
                 label=f'{result["configuration_name"]} (Val Loss)')
        plt.plot(result["train_losses"], 
                 linestyle='--', 
                 label=f'{result["configuration_name"]} (Train Loss)')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Overfitting Metric
    plt.subplot(2, 2, 4)
    overfitting_metrics = []
    for result in comprehensive_results:
        # Calculate the gap between train and validation accuracy
        accuracy_gaps = [
            train - val for train, val in 
            zip(result["train_accuracies"], result["validation_accuracies"])
        ]
        overfitting_metric = np.mean(accuracy_gaps[-5:])  # Average of last 5 epochs
        overfitting_metrics.append(overfitting_metric)
    
    plt.bar(names, overfitting_metrics)
    plt.title('Overfitting Metric\n(Lower is Better)')
    plt.ylabel('Accuracy Gap (Train - Val)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(logdir)
    plt.close()