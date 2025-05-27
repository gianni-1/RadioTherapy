import sys
import os
import os
import pytest
import torch
from system_manager import SystemManager

@pytest.fixture
def temp_root(tmp_path):
    """Create a temporary root directory."""
    return str(tmp_path)

@pytest.fixture
def manager(temp_root):
    """Initialize a SystemManager with minimal settings."""
    return SystemManager(
        root_dir=temp_root,
        transforms=None,
        resolutions=[],
        energies=[],
        quad_energies=[1],
        quad_weights=[1.0],
        batch_size=1,
        device=torch.device('cpu'),
        num_epochs=1,
        learning_rate=0.1,
        patience=1,
        cube_size=(1, 1, 1),
        seed=42
    )

def test_run_inference_no_models_and_no_checkpoint(manager):
    """
    If no models_by_energy are loaded and no checkpoint is provided,
    run_inference should raise RuntimeError.
    """
    manager.models_by_energy = {}
    with pytest.raises(RuntimeError) as exc:
        manager.run_inference("dummy.nii")
    assert "No trained models found" in str(exc.value)

def test_run_inference_invalid_checkpoint_type(manager):
    """
    Providing a checkpoint of unsupported type should raise ValueError.
    """
    manager.models_by_energy = {1: (None, None, None)}
    with pytest.raises(ValueError) as exc:
        manager.run_inference("dummy.nii", model_checkpoint=123)
    assert "Invalid model checkpoint" in str(exc.value)

def test_run_inference_unsupported_ct_extension(manager):
    """
    Providing an unsupported CT file extension should raise ValueError.
    """
    manager.models_by_energy = {1: (None, None, None)}
    # Create a dummy .txt file
    dummy_txt = os.path.join(manager.root_dir, "scan.txt")
    with open(dummy_txt, "w") as f:
        f.write("not a ct")
    with pytest.raises(ValueError) as exc:
        manager.run_inference(dummy_txt)
    assert "Unsupported CT file format" in str(exc.value)