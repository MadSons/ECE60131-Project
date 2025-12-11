import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(filepath):
    """Parse individual_results.txt file and return metrics dictionary."""
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('*'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
    
    # Add eucl_mean as ADE_4s if not already present
    if 'eucl_mean' in metrics and 'ADE_4s' not in metrics:
        metrics['ADE_4s'] = metrics['eucl_mean']
    
    return metrics

def load_all_results(base_folder):
    """Load all results from the folder structure."""
    mantra_data = {'digital': None, 'analog': []}
    cvae_data = {'digital': None, 'analog': []}
    
    # Get all subdirectories
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        results_file = os.path.join(folder_path, 'individual_results.txt')
        
        if not os.path.exists(results_file):
            continue
        
        metrics = parse_results_file(results_file)
        
        # Categorize based on folder name
        if 'MANTRA_default_IRM_digital' in folder_name:
            mantra_data['digital'] = metrics
        elif 'MANTRA_default_IRM_analog' in folder_name:
            mantra_data['analog'].append(metrics)
        elif 'CVAE_digital' in folder_name:
            cvae_data['digital'] = metrics
        elif 'CVAE_analog' in folder_name:
            cvae_data['analog'].append(metrics)
    
    return mantra_data, cvae_data

def create_boxplots(mantra_data, cvae_data, output_file='comparison_plot.png'):
    """Create side-by-side box plots with digital results as diamond markers."""
    
    # Define metrics to plot (in order)
    metrics = ['ADE_1s', 'ADE_2s', 'ADE_3s', 'ADE_4s', 
               'horizon10s', 'horizon20s', 'horizon30s', 'horizon40s']
    
    metric_labels = ['ADE 1s', 'ADE 2s', 'ADE 3s', 'ADE 4s',
                     'FDE 1s', 'FDE 2s', 'FDE 3s', 'FDE 4s']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('MANTRA vs CVAE: Analog Runs with Digital Baseline', fontsize=16, fontweight='bold')
    
    for idx, (data, model_name, ax) in enumerate([(mantra_data, 'MANTRA', axes[0]), 
                                                    (cvae_data, 'CVAE', axes[1])]):
        
        # Prepare data for box plots
        analog_values = []
        digital_values = []
        positions = []
        
        for i, metric in enumerate(metrics):
            # Collect analog run values
            values = [run[metric] for run in data['analog'] if metric in run]
            analog_values.append(values)
            
            # Collect digital value
            if data['digital'] and metric in data['digital']:
                digital_values.append(data['digital'][metric])
            else:
                digital_values.append(np.nan)
            
            positions.append(i + 1)
        
        # Create box plots
        bp = ax.boxplot(analog_values, positions=positions, widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='blue'),
                        capprops=dict(color='blue'))
        
        # Overlay digital results as diamond markers
        ax.plot(positions, digital_values, 'D', color='darkgreen', 
                markersize=10, label='Digital', zorder=5, markeredgecolor='black', markeredgewidth=1)
        
        # Formatting
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Displacement [m]', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} Results', fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left')
        
        # Add count of analog runs
        n_runs = len(data['analog'])
        ax.text(0.98, 0.98, f'Analog runs: {n_runs}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set the same y-axis limits for both subplots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.show()

def print_summary_statistics(mantra_data, cvae_data):
    """Print summary statistics for all metrics."""
    metrics = ['ADE_1s', 'ADE_2s', 'ADE_3s', 'ADE_4s', 
               'horizon10s', 'horizon20s', 'horizon30s', 'horizon40s']
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for model_name, data in [('MANTRA', mantra_data), ('CVAE', cvae_data)]:
        print(f"\n{model_name}:")
        print("-" * 80)
        
        for metric in metrics:
            analog_values = [run[metric] for run in data['analog'] if metric in run]
            
            if analog_values:
                digital_val = data['digital'][metric] if data['digital'] and metric in data['digital'] else None
                
                print(f"\n{metric}:")
                print(f"  Analog - Mean: {np.mean(analog_values):.4f}, "
                      f"Std: {np.std(analog_values):.4f}, "
                      f"Min: {np.min(analog_values):.4f}, "
                      f"Max: {np.max(analog_values):.4f}")
                if digital_val is not None:
                    print(f"  Digital: {digital_val:.4f}")

if __name__ == "__main__":
    # Set your base folder path
    base_folder = "test/"
    
    # Load all results
    print("Loading results from folder structure...")
    mantra_data, cvae_data = load_all_results(base_folder)
    
    print(f"Loaded {len(mantra_data['analog'])} MANTRA analog runs")
    print(f"Loaded {len(cvae_data['analog'])} CVAE analog runs")
    
    # Create box plots
    print("\nGenerating comparison plots...")
    create_boxplots(mantra_data, cvae_data)
    
    # Print summary statistics
    print_summary_statistics(mantra_data, cvae_data)