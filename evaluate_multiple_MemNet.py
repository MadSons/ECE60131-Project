import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
import evaluate_MemNet

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluate_MemNet.Validator on multiple MANTRA checkpoints "
                    "and compare metrics (withIRM vs noIRM)."
    )

    #  common parameters 
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="batch size for DataLoader"
    )
    parser.add_argument(
        "--past_len", type=int, default=20,
        help="length of past trajectory"
    )
    parser.add_argument(
        "--future_len", type=int, default=40,
        help="length of future trajectory"
    )
    parser.add_argument(
        "--preds", type=int, default=5,
        help="number of predictions (num_prediction)"
    )
    parser.add_argument(
        "--dim_embedding_key", type=int, default=48,
        help="dimensionality of the key embedding"
    )
    parser.add_argument(
        "--dataset_file", type=str, default="kitti_dataset.json",
        help="path to the dataset JSON file"
    )
    parser.add_argument(
        "--saveImages", type=str, default='None',
        help="whether to save qualitative images (None / 'All' / 'Subset')"
    )
    parser.add_argument(
        "--saved_memory", action="store_true", default=True,
        help="if set, load existing memories from each model's folder instead of building anew"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--analog_runs", type=int, default=10,
        help="number of times to run analog tests for averaging (default: 10)"
    )

    return parser.parse_args()


def run_single_evaluation(config, run_label):
    """Run a single evaluation and return metrics"""
    print(f"Running {run_label}")
    
    # Instantiate Validator (will create its own test/<timestamp>_<info> folder)
    validator = evaluate_MemNet.Validator(config)

    # ─── write (or load) memory ───────────────────────────────────────
    validator._memory_writing(config.saved_memory)

    # ─── run evaluation ───────────────────────────────────────────────
    metrics = validator.evaluate(validator.test_loader)
    
    # Save per‐run results.txt in that run's folder for reference:
    out_folder = validator.folder_test
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "individual_results.txt"), "w") as f:
        f.write(f"**** {run_label} ****\n")
        f.write(f"Model checkpoint: {config.model}\n")
        f.write(f"Memories folder: {config.memories_path}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    return metrics


def main():
    args = parse_args()

    # ─── list MANTRA checkpoints here ────────────────────────────────────
    # Each value is a tuple: (model_checkpoint_path, memories_folder_for_that_model, ae, controller)
    mantra_models = {
        "MANTRA_default": (
            "pretrained_models/MANTRA/model_IRM_statedict.pt",
            "pretrained_models/MANTRA/memories/",
            "pretrained_models/model_AE/model_ae_statedict.pt",
            "pretrained_models/model_controller/model_controller_statedict.pt"
        )
    }

    # ─── container to accumulate all metrics ───────────────────────────────────
    # Will hold: results[(model_label, irm_flag, analog_flag)] = { metric_name: [values], … }
    all_results = {}

    for model_label, (model_path, memories_path, model_ae, model_controller) in mantra_models.items():
        for irm_flag in (True, ):
            # Digital run (single run)
            analog_flag = False
            run_label = f"{model_label}_{'IRM' if irm_flag else 'noIRM'}_{'analog' if analog_flag else 'digital'}"
            print(f"\n==== Testing {run_label} ====\n")

            config = SimpleNamespace(
                cuda=torch.cuda.is_available(),
                batch_size=args.batch_size,
                past_len=args.past_len,
                future_len=args.future_len,
                preds=args.preds,
                dim_embedding_key=args.dim_embedding_key,
                dataset_file=args.dataset_file,
                model_ae=model_ae,
                model_controller=model_controller,
                model=model_path,
                visualize_dataset=False,
                saved_memory=args.saved_memory,
                memories_path=memories_path,
                withIRM=irm_flag,
                saveImages=args.saveImages,
                device=args.device,
                analog=analog_flag,
                info=run_label
            )

            metrics = run_single_evaluation(config, run_label)
            # Convert to lists for consistency with analog results
            all_results[(model_label, irm_flag, analog_flag)] = {k: [v] for k, v in metrics.items()}

            # Analog runs (multiple runs for averaging)
            analog_flag = True
            run_label_base = f"{model_label}_{'IRM' if irm_flag else 'noIRM'}_analog"
            print(f"\n==== Testing {run_label_base} ({args.analog_runs} runs) ====\n")

            # Initialize storage for analog results
            analog_results = {}
            
            for run_idx in range(args.analog_runs):
                run_label = f"{run_label_base}_run{run_idx+1}"
                
                config = SimpleNamespace(
                    cuda=torch.cuda.is_available(),
                    batch_size=args.batch_size,
                    past_len=args.past_len,
                    future_len=args.future_len,
                    preds=args.preds,
                    dim_embedding_key=args.dim_embedding_key,
                    dataset_file=args.dataset_file,
                    model_ae=model_ae,
                    model_controller=model_controller,
                    model=model_path,
                    visualize_dataset=False,
                    saved_memory=args.saved_memory,
                    memories_path=memories_path,
                    withIRM=irm_flag,
                    saveImages=args.saveImages,
                    device=args.device,
                    analog=analog_flag,
                    info=run_label
                )

                metrics = run_single_evaluation(config, run_label)
                
                # Accumulate results
                for k, v in metrics.items():
                    if k not in analog_results:
                        analog_results[k] = []
                    analog_results[k].append(v)

            all_results[(model_label, irm_flag, analog_flag)] = analog_results

    # ─── ensure /results directory exists ──────────────────────────────────────
    results_dir = "results_multiple_mantra"
    os.makedirs(results_dir, exist_ok=True)

    # ─── write a detailed summary text file with statistics ────────────────
    summary_path = os.path.join(results_dir, "results_summary.txt")
    
    with open(summary_path, "w") as out:
        out.write("="*80 + "\n")
        out.write("MANTRA Model Evaluation Results Summary\n")
        out.write("="*80 + "\n\n")
        
        for (model_label, irm_flag, analog_flag), results in all_results.items():
            config_name = f"{model_label} ({'IRM' if irm_flag else 'noIRM'}) ({'Analog' if analog_flag else 'Digital'})"
            out.write(f"{config_name}:\n")
            out.write("-" * len(config_name) + "-\n")
            
            if analog_flag:
                out.write(f"Number of runs: {len(results[list(results.keys())[0]])}\n\n")
                # Calculate statistics
                for metric_name in sorted(results.keys()):
                    values = np.array(results[metric_name])
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    out.write(f"{metric_name:12}: {mean_val:.4f} ± {std_val:.4f}\n")
            else:
                out.write("Single run:\n\n")
                for metric_name in sorted(results.keys()):
                    out.write(f"{metric_name:12}: {results[metric_name][0]:.4f}\n")
            out.write("\n")

    print(f"\nDetailed summary saved to {summary_path}\n")

    # ─── generate improved bar charts with error bars ─────────────────
    metrics_list = [
        "eucl_mean", "ADE_1s", "ADE_2s", "ADE_3s",
        "horizon10s", "horizon20s", "horizon30s", "horizon40s"
    ]

    # Prepare data for plotting
    keys = list(all_results.keys())
    x_labels = []
    for (m, irm_flag, analog_flag) in keys:
        label = f"{m}\n{'IRM' if irm_flag else 'noIRM'}\n{'Analog' if analog_flag else 'Digital'}"
        x_labels.append(label)

    # Set up colors for different models (same model = same color)
    model_colors = {'MANTRA_default': '#2E86AB'}

    # Create subplots for better organization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_list):
        ax = axes[idx]
        
        means = []
        stds = []
        colors_list = []
        alphas_list = []
        
        for k in keys:
            model_label, irm_flag, analog_flag = k
            values = np.array(all_results[k][metric_name])
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            
            # Same color for same model, different alpha for digital vs analog
            colors_list.append(model_colors[model_label])
            alphas_list.append(0.5 if analog_flag else 1.0)  # analog=0.5, digital=1.0
        
        # Create bars with error bars
        bars = []
        for i in range(len(means)):
            bar = ax.bar(i, means[i], yerr=stds[i], 
                        capsize=5, color=colors_list[i], 
                        alpha=alphas_list[i], edgecolor='black', linewidth=0.5)
            bars.extend(bar)
        
        # Customize the subplot
        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            if std > 0:
                ax.text(i, mean + std + mean*0.01,
                       f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(i, mean + mean*0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('MANTRA Model Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(results_dir, "all_metrics_comparison.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {combined_plot_path}")
    
    # Also create individual plots for each metric (improved versions)
    for metric_name in metrics_list:
        plt.figure(figsize=(10, 6))
        
        means = []
        stds = []
        colors_list = []
        alphas_list = []
        
        for k in keys:
            model_label, irm_flag, analog_flag = k
            values = np.array(all_results[k][metric_name])
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            
            # Same color for same model, different alpha for digital vs analog
            colors_list.append(model_colors[model_label])
            alphas_list.append(0.5 if analog_flag else 1.0)  # analog=0.5, digital=1.0
        
        # Create bars with error bars
        bars = []
        for i in range(len(means)):
            bar = plt.bar(i, means[i], yerr=stds[i], 
                         capsize=8, color=colors_list[i], 
                         alpha=alphas_list[i], edgecolor='black', linewidth=1)
            bars.extend(bar)
        
        # Customize the plot
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            if std > 0:
                plt.text(i, mean + std + mean*0.02,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                plt.text(i, mean + mean*0.02,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        png_name = os.path.join(results_dir, f"{metric_name}_comparison.png")
        plt.savefig(png_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved individual plot: {png_name}")

    # ─── create a CSV file for easy data analysis ─────────────────
    csv_path = os.path.join(results_dir, "results_data.csv")
    with open(csv_path, "w") as f:
        # Header
        f.write("Model,IRM,Analog,Metric,Mean,Std,NumRuns\n")
        
        # Data rows
        for (model_label, irm_flag, analog_flag), results in all_results.items():
            for metric_name, values in results.items():
                values_array = np.array(values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array, ddof=1) if len(values_array) > 1 else 0.0
                num_runs = len(values_array)
                
                f.write(f"{model_label},{irm_flag},{analog_flag},{metric_name},{mean_val:.6f},{std_val:.6f},{num_runs}\n")
    
    print(f"CSV data saved to {csv_path}")
    print(f"\nAll results saved to: {results_dir}/")


if __name__ == "__main__":
    main()