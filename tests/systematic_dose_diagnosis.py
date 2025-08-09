#!/usr/bin/env python3
"""
Systematic Dose Normalization Diagnosis Script
Phase 1: Root Cause Investigation

This script performs comprehensive analysis of the training data pipeline
to identify the exact source of the 21x dose scaling discrepancy.
"""

import logging
import numpy as np
import torch
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add sourcecode to path
sys.path.append('sourcecode')

try:
    import nibabel as nib
    has_nibabel = True
except ImportError:
    has_nibabel = False


class DoseNormalizationDiagnostic:
    """Comprehensive diagnosis of dose normalization pipeline."""
    
    def __init__(self, traindata_dir="traindata", checkpoint_path=None):
        self.traindata_dir = Path(traindata_dir)
        self.checkpoint_path = checkpoint_path
        self.results = {
            'raw_dose_statistics': {},
            'training_normalization_params': {},
            'checkpoint_normalization_params': {},
            'inference_parameters': {},
            'medical_plausibility': {},
            'data_flow_analysis': {},
            'recommendations': []
        }
    
    def analyze_raw_dose_data(self):
        """Phase 1.1: Analyze raw dose data in traindata directory."""
        logger.info("=" * 60)
        logger.info("PHASE 1.1: RAW DOSE DATA ANALYSIS")
        logger.info("=" * 60)
        
        if not self.traindata_dir.exists():
            logger.error(f"Training data directory not found: {self.traindata_dir}")
            return False
        
        energy_stats = {}
        global_min, global_max = float('inf'), float('-inf')
        total_files = 0
        
        # Search all energy subdirectories
        for energy_dir in self.traindata_dir.iterdir():
            if not energy_dir.is_dir():
                continue
                
            outputcube_dir = energy_dir / "outputcube"
            if not outputcube_dir.exists():
                logger.warning(f"No outputcube directory in {energy_dir}")
                continue
            
            logger.info(f"\nAnalyzing energy directory: {energy_dir.name}")
            
            dose_files = list(outputcube_dir.glob("*.npy")) + list(outputcube_dir.glob("*.nii*"))
            if not dose_files:
                logger.warning(f"No dose files found in {outputcube_dir}")
                continue
            
            energy_doses = []
            for dose_file in dose_files[:10]:  # Sample first 10 files for speed
                try:
                    if dose_file.suffix.lower() in ['.nii', '.gz']:
                        if has_nibabel:
                            dose_data = nib.load(str(dose_file)).get_fdata()
                        else:
                            logger.warning(f"Cannot load {dose_file} - nibabel not available")
                            continue
                    else:
                        dose_data = np.load(str(dose_file))
                    
                    # Handle different array shapes
                    if dose_data.ndim > 3:
                        dose_data = dose_data.squeeze()
                    
                    energy_doses.append(dose_data)
                    global_min = min(global_min, dose_data.min())
                    global_max = max(global_max, dose_data.max())
                    total_files += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load {dose_file}: {e}")
            
            if energy_doses:
                all_values = np.concatenate([d.flatten() for d in energy_doses])
                energy_stats[energy_dir.name] = {
                    'file_count': len(energy_doses),
                    'min': float(all_values.min()),
                    'max': float(all_values.max()),
                    'mean': float(all_values.mean()),
                    'std': float(all_values.std()),
                    'median': float(np.median(all_values)),
                    'percentile_95': float(np.percentile(all_values, 95)),
                    'percentile_99': float(np.percentile(all_values, 99)),
                    'shape': energy_doses[0].shape
                }
                
                logger.info(f"  Files analyzed: {len(energy_doses)}")
                logger.info(f"  Value range: {all_values.min():.6f} to {all_values.max():.6f}")
                logger.info(f"  Mean: {all_values.mean():.6f}, Std: {all_values.std():.6f}")
                logger.info(f"  95th percentile: {np.percentile(all_values, 95):.6f}")
                logger.info(f"  99th percentile: {np.percentile(all_values, 99):.6f}")
        
        self.results['raw_dose_statistics'] = {
            'global_min': float(global_min) if global_min != float('inf') else None,
            'global_max': float(global_max) if global_max != float('-inf') else None,
            'total_files_analyzed': total_files,
            'energy_statistics': energy_stats
        }
        
        # Medical plausibility check
        if global_max != float('-inf'):
            logger.info(f"\n MEDICAL PLAUSIBILITY CHECK:")
            logger.info(f"   Global maximum dose: {global_max:.6f}")
            if global_max < 10:
                logger.warning("  Maximum dose < 10 - suspiciously low for radiotherapy!")
            elif global_max > 100:
                logger.warning("  Maximum dose > 100 - suspiciously high!")
            else:
                logger.info("✓ Dose range appears medically plausible")
        
        return True
    
    def analyze_training_normalization(self):
        """Phase 1.2: Analyze training normalization logic."""
        logger.info("=" * 60)
        logger.info("PHASE 1.2: TRAINING NORMALIZATION ANALYSIS")
        logger.info("=" * 60)
        
        # Try to import training pipeline to understand normalization
        try:
            # Check if training pipeline files exist
            training_files = ['training_pipeline.py', 'sourcecode/training_pipeline.py', 'system_manager.py']
            found_file = None
            
            for tf in training_files:
                if Path(tf).exists():
                    found_file = tf
                    break
            
            if found_file:
                logger.info(f"Found training configuration in: {found_file}")
                
                # Read the file to analyze normalization logic
                with open(found_file, 'r') as f:
                    content = f.read()
                
                # Look for dose normalization patterns
                norm_patterns = ['clip_max', 'clip_min', 'dose_normalization', 'ScaleIntensityRanged', 'NormalizeIntensityd']
                found_patterns = {}
                
                for pattern in norm_patterns:
                    occurrences = content.count(pattern)
                    if occurrences > 0:
                        found_patterns[pattern] = occurrences
                        logger.info(f"  Found '{pattern}': {occurrences} occurrences")
                
                # Look for specific normalization values
                import re
                clip_max_matches = re.findall(r'clip_max["\s]*[=:]["\s]*([0-9.]+)', content)
                clip_min_matches = re.findall(r'clip_min["\s]*[=:]["\s]*([0-9.]+)', content)
                
                if clip_max_matches:
                    logger.info(f"  Found clip_max values: {clip_max_matches}")
                if clip_min_matches:
                    logger.info(f"  Found clip_min values: {clip_min_matches}")
                
                self.results['training_normalization_params'] = {
                    'source_file': found_file,
                    'normalization_patterns': found_patterns,
                    'clip_max_values': clip_max_matches,
                    'clip_min_values': clip_min_matches
                }
            else:
                logger.warning("No training pipeline files found")
                self.results['training_normalization_params'] = None
                
        except Exception as e:
            logger.warning(f"Error analyzing training normalization: {e}")
            self.results['training_normalization_params'] = None
    
    def analyze_checkpoint_parameters(self):
        """Phase 1.3: Analyze checkpoint normalization parameters."""
        logger.info("=" * 60)
        logger.info("PHASE 1.3: CHECKPOINT PARAMETER ANALYSIS")
        logger.info("=" * 60)
        
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            logger.warning("No checkpoint path provided or file not found")
            return False
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract dose normalization parameters
            dose_params = checkpoint.get("dose_normalization_params", {})
            models_by_energy = checkpoint.get("models_by_energy", {})
            
            logger.info(f"Checkpoint contains {len(models_by_energy)} energy-specific models")
            logger.info(f"Available model keys: {list(models_by_energy.keys())}")
            
            if dose_params:
                logger.info(f"\nDose normalization parameters in checkpoint:")
                for key, params in dose_params.items():
                    clip_min = params.get('clip_min', 'N/A')
                    clip_max = params.get('clip_max', 'N/A')
                    logger.info(f"  {key}:")
                    logger.info(f"    clip_min: {clip_min}")
                    logger.info(f"    clip_max: {clip_max}")
                    
                    # Medical plausibility check for clip_max
                    if isinstance(clip_max, (int, float)) and clip_max < 10:
                        logger.warning(f"    ⚠️  clip_max={clip_max} is suspiciously low for radiotherapy!")
            else:
                logger.warning("No dose_normalization_params found in checkpoint!")
            
            # Check individual model parameters
            model_params = {}
            for energy_key, model_dict in models_by_energy.items():
                if isinstance(model_dict, dict):
                    model_params[energy_key] = {
                        'clip_min': model_dict.get('clip_min'),
                        'clip_max': model_dict.get('clip_max'),
                        'scale_factor': model_dict.get('scale_factor')
                    }
            
            logger.info(f"\nIndividual model parameters:")
            for energy_key, params in model_params.items():
                logger.info(f"  {energy_key}:")
                for param, value in params.items():
                    logger.info(f"    {param}: {value}")
            
            self.results['checkpoint_normalization_params'] = {
                'dose_normalization_params': dose_params,
                'model_parameters': model_params,
                'available_energies': list(models_by_energy.keys())
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze checkpoint: {e}")
            return False
    
    def analyze_inference_flow(self):
        """Phase 1.4: Analyze inference denormalization flow."""
        logger.info("=" * 60)
        logger.info("PHASE 1.4: INFERENCE FLOW ANALYSIS")
        logger.info("=" * 60)
        
        try:
            from inference_module import InferenceModule
            
            # Analyze InferenceModule's denormalization logic
            logger.info("Analyzing InferenceModule denormalization logic...")
            
            # Check source code for denormalization methods
            import inspect
            
            # Get source of InferenceModule
            source = inspect.getsource(InferenceModule)
            
            # Look for denormalization-related methods
            if 'denormalize_dose' in source:
                logger.info("✓ Found denormalize_dose method in InferenceModule")
            else:
                logger.warning("  No denormalize_dose method found in InferenceModule")
            
            if 'clip_max' in source:
                logger.info("✓ InferenceModule references clip_max parameters")
            else:
                logger.warning("  InferenceModule doesn't reference clip_max parameters")
            
            # Count occurrences of normalization-related terms
            norm_terms = ['normalize', 'denormalize', 'clip_min', 'clip_max', 'scale_factor']
            for term in norm_terms:
                count = source.count(term)
                logger.info(f"  '{term}' appears {count} times in InferenceModule")
            
            self.results['inference_parameters'] = {
                'has_denormalize_dose': 'denormalize_dose' in source,
                'references_clip_max': 'clip_max' in source,
                'normalization_term_counts': {term: source.count(term) for term in norm_terms}
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze inference module: {e}")
            self.results['inference_parameters'] = None
    
    def perform_data_flow_analysis(self):
        """Phase 1.5: Comprehensive data flow analysis."""
        logger.info("=" * 60)
        logger.info("PHASE 1.5: DATA FLOW ANALYSIS")
        logger.info("=" * 60)
        
        # Compare raw data statistics with checkpoint parameters
        raw_stats = self.results.get('raw_dose_statistics', {})
        checkpoint_params = self.results.get('checkpoint_normalization_params', {})
        
        if raw_stats.get('global_max') and checkpoint_params:
            raw_max = raw_stats['global_max']
            
            # Compare with checkpoint clip_max values
            dose_params = checkpoint_params.get('dose_normalization_params', {})
            if dose_params:
                logger.info("Comparing raw data with checkpoint normalization:")
                for key, params in dose_params.items():
                    clip_max = params.get('clip_max')
                    if isinstance(clip_max, (int, float)):
                        ratio = raw_max / clip_max
                        logger.info(f"  {key}: raw_max={raw_max:.6f}, clip_max={clip_max:.6f}, ratio={ratio:.2f}x")
                        
                        if ratio > 15:
                            logger.warning(f"     Raw data is {ratio:.1f}x larger than clip_max - major discrepancy!")
                            self.results['recommendations'].append(
                                f"CRITICAL: {key} clip_max={clip_max:.6f} is {ratio:.1f}x too small. "
                                f"Should be close to {raw_max:.6f}"
                            )
        
        # Analyze training vs inference energy coverage
        raw_energies = set(self.results['raw_dose_statistics'].get('energy_statistics', {}).keys())
        checkpoint_energies = set()
        
        if checkpoint_params:
            available_energies = checkpoint_params.get('available_energies', [])
            for energy_key in available_energies:
                # Extract energy value from key like "res64x64x64_e11.5"
                if '_e' in energy_key:
                    energy_val = energy_key.split('_e')[1]
                    checkpoint_energies.add(f"energy_{energy_val}")
        
        logger.info(f"Raw data energies: {raw_energies}")
        logger.info(f"Checkpoint energies: {checkpoint_energies}")
        
        missing_in_checkpoint = raw_energies - checkpoint_energies
        if missing_in_checkpoint:
            logger.warning(f"Energies in raw data but missing in checkpoint: {missing_in_checkpoint}")
        
        self.results['data_flow_analysis'] = {
            'raw_energies': list(raw_energies),
            'checkpoint_energies': list(checkpoint_energies),
            'missing_energies': list(missing_in_checkpoint)
        }
    
    def generate_recommendations(self):
        """Phase 1.6: Generate specific recommendations."""
        logger.info("=" * 60)
        logger.info("PHASE 1.6: RECOMMENDATIONS")
        logger.info("=" * 60)
        
        # Analyze all findings and generate actionable recommendations
        recommendations = []
        
        raw_stats = self.results.get('raw_dose_statistics', {})
        checkpoint_params = self.results.get('checkpoint_normalization_params', {})
        
        # Check for major dose scaling issues
        if raw_stats.get('global_max'):
            raw_max = raw_stats['global_max']
            
            if checkpoint_params:
                dose_params = checkpoint_params.get('dose_normalization_params', {})
                if dose_params:
                    max_clip_max = max([p.get('clip_max', 0) for p in dose_params.values() if isinstance(p.get('clip_max'), (int, float))])
                    if max_clip_max > 0:
                        ratio = raw_max / max_clip_max
                        if ratio > 10:
                            recommendations.append({
                                'priority': 'CRITICAL',
                                'issue': 'Dose normalization parameters severely wrong',
                                'description': f'Raw dose maximum ({raw_max:.6f}) is {ratio:.1f}x larger than largest clip_max ({max_clip_max:.6f})',
                                'solution': 'Recalculate dose normalization parameters from raw training data',
                                'implementation': 'Run fix_dose_normalization_parameters() function'
                            })
        
        # Check for missing normalization parameters
        if not checkpoint_params or not checkpoint_params.get('dose_normalization_params'):
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Missing dose normalization parameters in checkpoint',
                'description': 'Checkpoint does not contain dose_normalization_params',
                'solution': 'Regenerate checkpoint with proper dose normalization parameters',
                'implementation': 'Retrain model with corrected dose normalization pipeline'
            })
        
        # Check for inference module issues
        inference_params = self.results.get('inference_parameters')
        if inference_params and not inference_params.get('has_denormalize_dose'):
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Missing denormalize_dose method in InferenceModule',
                'description': 'InferenceModule may not properly denormalize dose outputs',
                'solution': 'Add proper dose denormalization logic to InferenceModule',
                'implementation': 'Implement denormalize_dose() method with energy-specific parameters'
            })
        
        self.results['recommendations'].extend(recommendations)
        
        # Print all recommendations
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\n RECOMMENDATION {i} [{rec['priority']}]:")
            logger.info(f"   Issue: {rec['issue']}")
            logger.info(f"   Description: {rec['description']}")
            logger.info(f"   Solution: {rec['solution']}")
            logger.info(f"   Implementation: {rec['implementation']}")
    
    def save_results(self, output_path="dose_diagnosis_results.json"):
        """Save all analysis results to JSON file."""
        logger.info(f"\nSaving diagnosis results to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info("✓ Results saved")
    
    def create_visualization(self, output_path="dose_diagnosis_visualization.png"):
        """Create visualization of key findings."""
        logger.info(f"\nCreating diagnosis visualization...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Dose Normalization Diagnosis Results', fontsize=16)
            
            # Plot 1: Raw dose statistics by energy
            raw_stats = self.results.get('raw_dose_statistics', {})
            energy_stats = raw_stats.get('energy_statistics', {})
            
            if energy_stats:
                energies = list(energy_stats.keys())
                max_values = [stats['max'] for stats in energy_stats.values()]
                mean_values = [stats['mean'] for stats in energy_stats.values()]
                
                axes[0, 0].bar(range(len(energies)), max_values, alpha=0.7, label='Maximum')
                axes[0, 0].bar(range(len(energies)), mean_values, alpha=0.7, label='Mean')
                axes[0, 0].set_xlabel('Energy')
                axes[0, 0].set_ylabel('Dose Value')
                axes[0, 0].set_title('Raw Dose Statistics by Energy')
                axes[0, 0].set_xticks(range(len(energies)))
                axes[0, 0].set_xticklabels(energies, rotation=45)
                axes[0, 0].legend()
            
            # Plot 2: Checkpoint clip_max vs raw data
            checkpoint_params = self.results.get('checkpoint_normalization_params', {})
            dose_params = checkpoint_params.get('dose_normalization_params', {})
            
            if dose_params and energy_stats:
                clip_maxes = []
                raw_maxes = []
                labels = []
                
                for key, params in dose_params.items():
                    clip_max = params.get('clip_max')
                    if isinstance(clip_max, (int, float)):
                        clip_maxes.append(clip_max)
                        # Try to match with raw data
                        raw_max = raw_stats.get('global_max', 0)
                        raw_maxes.append(raw_max)
                        labels.append(key.split('_')[-1] if '_' in key else key)
                
                if clip_maxes:
                    x = np.arange(len(labels))
                    width = 0.35
                    
                    axes[0, 1].bar(x - width/2, clip_maxes, width, label='Checkpoint clip_max', alpha=0.7)
                    axes[0, 1].bar(x + width/2, raw_maxes, width, label='Raw data max', alpha=0.7)
                    axes[0, 1].set_xlabel('Energy/Resolution')
                    axes[0, 1].set_ylabel('Dose Value')
                    axes[0, 1].set_title('Checkpoint vs Raw Data Comparison')
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels(labels, rotation=45)
                    axes[0, 1].legend()
                    axes[0, 1].set_yscale('log')  # Log scale to show differences
            
            # Plot 3: Recommendations priority
            recommendations = self.results.get('recommendations', [])
            if recommendations:
                priorities = [rec['priority'] for rec in recommendations]
                priority_counts = {p: priorities.count(p) for p in set(priorities)}
                
                axes[1, 0].pie(priority_counts.values(), labels=priority_counts.keys(), autopct='%1.1f%%')
                axes[1, 0].set_title('Recommendations by Priority')
            
            # Plot 4: Summary statistics
            summary_text = f"""Diagnosis Summary:

Raw Data:
• Total files analyzed: {raw_stats.get('total_files_analyzed', 'N/A')}
• Global max dose: {raw_stats.get('global_max', 'N/A'):.6f}
• Energy directories: {len(energy_stats)}

Checkpoint:
• Available models: {len(checkpoint_params.get('available_energies', []))}
• Has dose params: {'Yes' if dose_params else 'No'}

Critical Issues Found: {len([r for r in recommendations if r.get('priority') == 'CRITICAL'])}
Total Recommendations: {len(recommendations)}"""
            
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Diagnosis Summary')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Visualization saved: {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
    
    def run_full_diagnosis(self, checkpoint_path=None):
        """Run complete diagnosis pipeline."""
        logger.info("🔍 STARTING SYSTEMATIC DOSE NORMALIZATION DIAGNOSIS")
        logger.info("=" * 80)
        
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        
        # Execute all analysis phases
        success = True
        success &= self.analyze_raw_dose_data()
        self.analyze_training_normalization()
        if self.checkpoint_path:
            success &= self.analyze_checkpoint_parameters()
        self.analyze_inference_flow()
        self.perform_data_flow_analysis()
        self.generate_recommendations()
        
        # Save results and create visualization
        self.save_results()
        self.create_visualization()
        
        logger.info("=" * 80)
        logger.info(" DIAGNOSIS COMPLETE")
        logger.info("=" * 80)
        
        return success


def main():
    """Main function for running systematic diagnosis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Systematic Dose Normalization Diagnosis")
    parser.add_argument('--traindata-dir', default='traindata', help='Training data directory')
    parser.add_argument('--checkpoint', help='Path to model checkpoint for analysis')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create diagnostic instance
    diagnostic = DoseNormalizationDiagnostic(
        traindata_dir=args.traindata_dir,
        checkpoint_path=args.checkpoint
    )
    
    # Run full diagnosis
    success = diagnostic.run_full_diagnosis()
    
    if success:
        logger.info(" Diagnosis completed successfully")
        logger.info(" Check dose_diagnosis_results.json for detailed findings")
        logger.info(" Check dose_diagnosis_visualization.png for visual summary")
        return 0
    else:
        logger.error(" Diagnosis failed or incomplete")
        return 1


if __name__ == '__main__':
    exit(main())
