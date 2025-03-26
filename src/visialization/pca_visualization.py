import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
import pandas as pd
from pathlib import Path
import logging
import argparse
import warnings

from src import get_logger 
from visualization_feature_processor import DataLoader , EnhancedVisualizer
from run_visualization_scripts import parse_args

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()
   

class PlotPca(EnhancedVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def plot_pca_explained_variance(self, features, data_type, output_file=None, n_components=None):

        
        self.logger.info(f"Creating PCA explained variance plot for {data_type} data")
        
        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.warning(f"{data_type} data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if data_type == "storage":
            epsilon = 1e-10
            log_features = np.log1p(np.abs(features) + epsilon)
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            processed_features = scaler.fit_transform(log_features)
        elif data_type == "memory":
            from sklearn.preprocessing import PowerTransformer
            scaler = PowerTransformer(method='yeo-johnson')
            processed_features = scaler.fit_transform(features)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            processed_features = scaler.fit_transform(features)
        
        max_components = min(processed_features.shape[0], processed_features.shape[1])
        if n_components is None:
            n_components = min(23, max_components)
        else:
            n_components = min(n_components, max_components)
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(processed_features)
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.5]})
        
        ax[0].bar(
            range(1, n_components + 1),
            pca.explained_variance_ratio_,
            alpha=0.7,
            color='steelblue'
        )
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Variance Ratio')
        ax[0].set_title(f'Explained Variance per Principal Component ({data_type.capitalize()} Data)')
        ax[0].grid(True, linestyle='--', alpha=0.5)
        
        for i, v in enumerate(pca.explained_variance_ratio_):
            ax[0].text(i + 1, v + 0.01, f'{v:.2%}', ha='center', fontsize=8)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        ax[1].plot(
            range(1, n_components + 1),
            cumulative_variance,
            'o-',
            linewidth=2,
            color='darkorange',
            label='Cumulative Explained Variance'
        )
        
        thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        threshold_components = []
        for threshold in thresholds:
            if any(cumulative_variance >= threshold):
                components_needed = np.where(cumulative_variance >= threshold)[0][0] + 1
                threshold_components.append((threshold, components_needed))
                ax[1].axhline(
                    y=threshold,
                    color='gray',
                    linestyle='--',
                    alpha=0.5
                )
                ax[1].text(
                    n_components - 1,
                    threshold + 0.01,
                    f'{threshold:.0%}',
                    ha='right',
                    va='bottom',
                    color='gray'
                )
        
        ax[1].set_xlabel('Number of Principal Components')
        ax[1].set_ylabel('Cumulative Explained Variance')
        ax[1].set_title(f'Cumulative Explained Variance ({data_type.capitalize()} Data)')
        ax[1].set_xticks(range(1, n_components + 1, max(1, n_components // 10)))
        ax[1].set_ylim([0, 1.05])
        ax[1].grid(True, linestyle='--', alpha=0.5)
        
        threshold_text = "\n".join([
            f"Components for {t:.0%} variance: {c}" for t, c in threshold_components
        ])
        plt.figtext(
            0.02,
            0.02,
            threshold_text,
            fontsize=10,
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout()
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved PCA explained variance plot to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        
        return pca
    
    
    def visualize_all_pca(self, data_dict, output_dir=None):
        """Create PCA visualizations for all data types"""
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
    
        self.logger.info(f"Storage shape: {data_dict['storage'].shape}")
        self.logger.info(f"Memory shape: {data_dict['memory'].shape}")
        self.logger.info(f"Combined shape: {data_dict['combined'].shape}")
       
       
        
        try:
            
            self.logger.info("Creating PCA explained variance plots")
            self.plot_pca_explained_variance(
                data_dict["storage"], 
                "storage",
                output_file=output_dir / "pca_storage_variance.png" if output_dir else None
            )
            
            self.plot_pca_explained_variance(
                data_dict["memory"], 
                "memory",
                output_file=output_dir / "pca_memory_variance.png" if output_dir else None
            )
            
            self.plot_pca_explained_variance(
                data_dict["combined"], 
                "combined",
                output_file=output_dir / "pca_combined_variance.png" if output_dir else None
            )
            
           
            
        except Exception as e:
            self.logger.error(f"Error in PCA visualization process: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return None

   
def main():
    args = parse_args()
    try:
   
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = DataLoader(args.data_dir)
        data_dict = data_loader.load_data()
            
        logger.info("Initializing visualizer")
        logger.info("Initializing PCA visualizer")
        plot_pca = PlotPca(
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            random_state=args.seed,
            outlier_method=args.outlier_method,
            iqr_factor=args.iqr_factor,
            zscore_threshold=args.zscore_threshold,
            contamination=args.contamination
        )
        
        for key in ["storage", "memory", "combined"]:
            logger.info(f"{key} data shape: {data_dict[key].shape}")
            logger.info(f"{key} data min: {np.min(data_dict[key])}, max: {np.max(data_dict[key])}")
            logger.info(f"{key} data mean: {np.mean(data_dict[key])}, std: {np.std(data_dict[key])}")
            logger.info(f"NaN values in {key}: {np.isnan(data_dict[key]).sum()}")
            logger.info(f"Inf values in {key}: {np.isinf(data_dict[key]).sum()}")
        
        logger.info("Creating PCA visualizations")
        pca_results = plot_pca.visualize_all_pca(data_dict, args.output_dir)
        
        if pca_results:
            logger.info("PCA visualization complete")
            
            for data_type, result in pca_results.items():
                pca = result["pca"]
                logger.info(f"PCA for {data_type} - Top 5 components explained variance: "
                        f"{pca.explained_variance_ratio_[:5]}")
                logger.info(f"PCA for {data_type} - Cumulative explained variance (2 components): "
                        f"{np.sum(pca.explained_variance_ratio_[:2]):.2%}")
        else:
            logger.warning("PCA visualization failed or returned no results")
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    

if __name__=='__main__':
    main()