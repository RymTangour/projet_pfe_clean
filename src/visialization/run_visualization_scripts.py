import multiprocessing
import subprocess
import argparse
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
def run_script(script_name):
    args = sys.argv[1:]
    # Construct the full path relative to the current script's directory
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = ["python", f"{script_path}.py"] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Visualization for Storage and Memory Data')
    parser.add_argument('--data_dir', type=str,default='/home/rym/projet/projet_pfe/processed_results_final',
                      help='Directory containing the data structure with storage.npy and memory.npy files')
    parser.add_argument('--output_dir', type=str, default='artifacts/plots',
                      help='Directory to save the output plots')
    parser.add_argument('--perplexity', type=float, default=40.0,
                      help='Perplexity parameter for t-SNE (default: 40)')
    parser.add_argument('--learning_rate', type=float, default=200.0,
                      help='Learning rate for t-SNE (default: 200)')
    parser.add_argument('--max_iter', type=int, default=2500,
                      help='Number of iterations for t-SNE (default: 2500)')
    parser.add_argument('--method', type=str, choices=['tsne', 'umap','pca'], default='tsne',
                      help='Visualization method to use (default: tsne)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_samples', type=int, default=100000,
                      help='Maximum number of samples to use for visualization (default: 10000)')
    parser.add_argument('--outlier-method', type=str, default='iqr', 
                      choices=['iqr', 'zscore', 'isolation_forest'],
                      help='Method for outlier detection (default: iqr)')
    parser.add_argument('--iqr-factor', type=float, default=2.0,
                      help='IQR factor for outlier detection (default: 2.0)')
    parser.add_argument('--zscore-threshold', type=float, default=5.0,
                      help='Z-score threshold for outlier detection (default: 5.0)')
    parser.add_argument('--contamination', type=float, default=0.1,
                      help='Contamination parameter for Isolation Forest (default: 0.1)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    scripts = [
    "2D_embedding",
    "3D_embedding",
    "clustering_visualization",
    "data_exploration",
    "feature_analysis",
    "pca_visualization"
                ]

    
    with multiprocessing.Pool(processes=len(scripts)) as pool:
        pool.map(run_script, scripts)