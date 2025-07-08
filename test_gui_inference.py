#!/usr/bin/env python3
"""
Test that the GUI would produce corrected inference output
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sourcecode')))

import torch
import numpy as np
from system_manager import SystemManager

def test_gui_inference():
    """Simulate what the GUI would do for inference"""
    print("Testing GUI inference path...")
    
    # Use the same real training data
    input_path = "traindata/11_5/inputcube/235101017859661465075472232303048949736_0.npy"
    ground_truth_path = "traindata/11_5/outputcube/235101017859661465075472232303048949736_0.npy"
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"Ground truth file not found: {ground_truth_path}")
        return
    
    # Load ground truth for comparison
    ground_truth = np.load(ground_truth_path)
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Ground truth range: {ground_truth.min():.6f} to {ground_truth.max():.6f}")
    
    # Create SystemManager with GUI-like configuration
    system_manager = SystemManager(
        root_dir=".",
        transforms=None,
        resolutions=[(64, 64, 64)],
        energies=[11.5],
        quad_energies=[11.5],
        quad_weights=[1.0],
        batch_size=1,
        device=torch.device('cpu'),
        num_epochs=1,
        learning_rate=0.001,
        patience=10,
        cube_size=64
    )
    
    # Test inference exactly as the GUI would call it
    try:
        print("Running inference (as GUI would)...")
        output_path = system_manager.run_inference(
            input_path,
            model_checkpoint="unified_energy_conditioned_model_res16.0_energies3.ckpt"
        )
        
        print(f"Inference completed successfully!")
        print(f"Output saved to: {output_path}")
        
        # Load and analyze the result
        import nibabel as nib
        img = nib.load(output_path)
        data = img.get_fdata()
        
        print(f"Output shape: {data.shape}")
        print(f"Output range: {data.min():.6f} to {data.max():.6f}")
        
        # Calculate correlation with ground truth
        if data.shape == ground_truth.shape:
            correlation = np.corrcoef(data.flatten(), ground_truth.flatten())[0, 1]
            print(f"Correlation with ground truth: {correlation:.6f}")
            
            # Check if the correlation is good (should be ~0.23)
            if correlation > 0.2:
                print("✅ SUCCESS: GUI inference produces meaningful correlation!")
            else:
                print("❌ FAILURE: GUI inference correlation too low")
        else:
            print(f"❌ FAILURE: Shape mismatch - output {data.shape} vs ground truth {ground_truth.shape}")
            
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui_inference()
