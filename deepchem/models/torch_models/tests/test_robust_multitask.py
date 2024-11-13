import numpy as np
from deepchem.models.torch_models import RobustMultitask
import pytest
import os
try:
    import torch
    import torch.nn as nn

    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass

# Inputs and parameters used while training the tensorflow model to obtain weights.
n_tasks_tf = 3
n_features_tf = 100
layer_sizes_tf = [512, 1024]

@pytest.mark.torch
def test_robustmultitask_construction():
    """Test that RobustMultiTask Model can be constructed without crash.
    """

    model = RobustMultitask(
        n_tasks=1,
        n_features=100,
        mode="regression",
        layer_sizes=[128, 256],
    )

    assert model is not None


@pytest.mark.torch
def test_robustmultitask_forward():
    """Test that the forward pass of RobustMultiTask Model can be executed without crash
    and that the output has the correct value.
    """

    n_tasks = n_tasks_tf
    n_features = n_features_tf
    layer_sizes = layer_sizes_tf
    torch_model = RobustMultitask(n_tasks=n_tasks,
                                  n_features=n_features,
                                  layer_sizes=layer_sizes,
                                  mode='classification')

    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "tensorflow_robust_multitask_classifier_weights.npz"))

    move_weights(torch_model, weights)

    input_x = weights["input"]
    output = weights["output"]

    torch_out = torch_model(torch.from_numpy(input_x).float())[0]
    torch_out = torch_out.cpu().detach().numpy()
    assert np.allclose(output, torch_out,
                       atol=1e-4), "Predictions are not close"

def move_weights(torch_model, weights):
    """Porting weights from Tensorflow to PyTorch"""

    def to_torch_param(weights):
        """Convert numpy weights to torch parameters to be used as model weights"""
        weights = weights.T
        return nn.Parameter(torch.from_numpy(weights))

    torch_weights = {
        k: to_torch_param(v) for k, v in weights.items() if k != "output"
    }

    # Shared layers (512, 1024)
    torch_model.shared_layers[0].weight = torch_weights["shared-layers-dense-w"]
    torch_model.shared_layers[0].bias = torch_weights["shared-layers-dense-b"]

    torch_model.shared_layers[3].weight = torch_weights["shared-layers-dense_1-w"]
    torch_model.shared_layers[3].bias = torch_weights["shared-layers-dense_1-b"]

    # Bypass layers for tasks
    for i in range(3):  # Three tasks
        torch_model.bypass_layers[i][0].weight = torch_weights[f"bypass-layers-dense_{2 + i * 2}-w"]
        torch_model.bypass_layers[i][0].bias = torch_weights[f"bypass-layers-dense_{2 + i * 2}-b"]

        # Output layers for each task
        torch_model.output_layers[i].weight = torch_weights[f"bypass-layers-dense_{3 + i * 2}-w"]
        torch_model.output_layers[i].bias = torch_weights[f"bypass-layers-dense_{3 + i * 2}-b"]