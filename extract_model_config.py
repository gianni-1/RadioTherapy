#!/usr/bin/env python3
"""
Extract model configuration from checkpoint
"""

import torch
from pathlib import Path

def main():
    model_path = "unified_energy_conditioned_model_res16.0_energies3.ckpt"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("=== CHECKPOINT CONTENTS ===")
    print(f"Keys: {list(checkpoint.keys())}")
    
    if 'training_config' in checkpoint:
        print("\n=== TRAINING CONFIG ===")
        config = checkpoint['training_config']
        print(f"Config keys: {list(config.keys())}")
        
        # Look for model-specific configs
        for key in config:
            if 'model' in key.lower() or 'unet' in key.lower() or 'net' in key.lower():
                print(f"{key}: {config[key]}")
    
    # Analyze UNet structure
    if 'unet' in checkpoint:
        unet_state = checkpoint['unet']
        print("\n=== UNET ANALYSIS ===")
        
        # Find key layers to understand architecture
        conv_layers = []
        for key, tensor in unet_state.items():
            if 'conv' in key and 'weight' in key:
                conv_layers.append((key, tensor.shape))
        
        print("Key convolutional layers:")
        for key, shape in conv_layers[:10]:  # First 10
            print(f"  {key}: {shape}")
        
        # Look for specific patterns
        print("\n=== ARCHITECTURE CLUES ===")
        
        # Input layer
        if 'conv_in.conv.weight' in unet_state:
            shape = unet_state['conv_in.conv.weight'].shape
            print(f"Input layer: {shape} -> in_channels={shape[1]}, first_out_channels={shape[0]}")
        
        # Output layer
        if 'out.2.conv.weight' in unet_state:
            shape = unet_state['out.2.conv.weight'].shape
            print(f"Output layer: {shape} -> out_channels={shape[0]}")
        
        # Find channel progression
        down_channels = []
        up_channels = []
        
        for key, tensor in unet_state.items():
            if 'down_blocks' in key and 'conv1.conv.weight' in key:
                down_channels.append((key, tensor.shape))
            elif 'up_blocks' in key and 'conv1.conv.weight' in key:
                up_channels.append((key, tensor.shape))
        
        print("\nDown block channels:")
        for key, shape in down_channels[:5]:
            print(f"  {key}: {shape}")
        
        print("\nUp block channels:")
        for key, shape in up_channels[:5]:
            print(f"  {key}: {shape}")
        
        # Cross-attention dimensions
        attn_keys = []
        for key, tensor in unet_state.items():
            if 'attn2.to_k.weight' in key:
                attn_keys.append((key, tensor.shape))
        
        print("\nCross-attention keys:")
        for key, shape in attn_keys[:3]:
            print(f"  {key}: {shape} -> context_dim={shape[1]}")
    
    print("\n=== SUGGESTED MODEL CONFIG ===")
    if 'unet' in checkpoint:
        unet_state = checkpoint['unet']
        
        # Extract parameters
        in_channels = unet_state['conv_in.conv.weight'].shape[1]
        out_channels = unet_state['out.2.conv.weight'].shape[0]
        
        # Try to infer num_channels from down blocks
        first_down = unet_state['down_blocks.0.resnets.0.conv1.conv.weight'].shape[0]
        second_down = unet_state['down_blocks.1.resnets.0.conv1.conv.weight'].shape[0]
        
        # Check if there's a third down block
        third_down = None
        if 'down_blocks.2.resnets.0.conv1.conv.weight' in unet_state:
            third_down = unet_state['down_blocks.2.resnets.0.conv1.conv.weight'].shape[0]
        
        # Cross-attention dim
        cross_attn_dim = unet_state['down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight'].shape[1]
        
        print(f"in_channels: {in_channels}")
        print(f"out_channels: {out_channels}")
        print(f"num_channels: ({first_down}, {second_down}" + (f", {third_down}" if third_down else "") + ")")
        print(f"cross_attention_dim: {cross_attn_dim}")

if __name__ == "__main__":
    main()
