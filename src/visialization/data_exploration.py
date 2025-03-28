import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import argparse
import warnings

from src import get_logger 
from visualization_feature_processor import DataLoader , EnhancedVisualizer
from run_visualization_scripts import parse_args

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()
class Explore(EnhancedVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger  
    def plot_missing_data(self, data, output_file=None):
        
        self.logger.info("Analyzing missing data patterns")
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        missing = data.isnull().sum()
        missing_percent = missing / len(data) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percent': missing_percent.values
        })
        
        missing_df = missing_df.sort_values('Missing Values', ascending=False)
        
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        if len(missing_df) == 0:
            self.logger.info("No missing values found in the dataset.")
            
            for ax in axes:
                ax.set_visible(False)
            
            fig.text(
                0.5, 
                0.5, 
                f"No Missing Values Found\nDataset Shape: {data.shape[0]} rows × {data.shape[1]} columns", 
                fontsize=16,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', boxstyle='round,pad=0.5')
            )
        else:
            sns.barplot(x='Missing Values', y='Column', data=missing_df, ax=axes[0])
            axes[0].set_title('Missing Values by Column')
            
            if len(data.columns) <= 50:  
                sns.heatmap(
                    data.isnull(),
                    yticklabels=False,
                    cbar=False,
                    cmap='viridis',
                    ax=axes[1]
                )
                axes[1].set_title('Missing Value Patterns')
            else:
                axes[1].text(
                    0.5, 
                    0.5, 
                    "Too many columns for heatmap visualization",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1].transAxes
                )
        
        plt.tight_layout()
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved missing data plot to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        
        return missing_df

    def plot_class_balance(self, target, output_file=None):
        
        self.logger.info("Analyzing class balance")
        
        if not isinstance(target, pd.Series):
            target = pd.Series(target)
        
        class_counts = target.value_counts()
        class_percents = target.value_counts(normalize=True) * 100
        
        class_df = pd.DataFrame({
            'Class': class_counts.index,
            'Count': class_counts.values,
            'Percentage': class_percents.values
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.barplot(x='Class', y='Count', data=class_df, ax=axes[0])
        axes[0].set_title('Class Distribution (Counts)')
        
        for i, v in enumerate(class_df['Count']):
            axes[0].text(i, v + 0.1, str(v), ha='center')
        
        axes[1].pie(
            class_df['Count'],
            labels=class_df['Class'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        axes[1].axis('equal')
        axes[1].set_title('Class Distribution (Percentage)')
        
        n_classes = len(class_df)
        max_count = class_df['Count'].max()
        min_count = class_df['Count'].min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        summary = (
            f"Number of classes: {n_classes}\n"
            f"Total samples: {class_df['Count'].sum()}\n"
            f"Max/Min ratio: {imbalance_ratio:.2f}\n"
            f"Majority class: {class_df.iloc[0]['Class']} ({class_df.iloc[0]['Percentage']:.1f}%)\n"
            f"Minority class: {class_df.iloc[-1]['Class']} ({class_df.iloc[-1]['Percentage']:.1f}%)"
        )
        
        plt.figtext(
            0.5,
            0.01,
            summary,
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved class balance plot to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        
        return class_df

def main():
    args = parse_args()
    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logger.info(f"Created output directory: {args.output_dir}")
        
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = DataLoader(args.data_dir)
        data_dict = data_loader.load_data()
        
        original_labels = data_dict["original_labels"]
        binary_labels = data_dict["binary_labels"]  
        
        
        logger.info("Initializing Correlation visualizer")
        explore = Explore(
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            random_state=args.seed,
            outlier_method=args.outlier_method,
            iqr_factor=args.iqr_factor,
            zscore_threshold=args.zscore_threshold,
            contamination=args.contamination
        )
        
        # Print data summary stats
        storage_features = data_dict["storage"]
        memory_features = data_dict["memory"]
        combined_features = data_dict["combined"]
        
        logger.info(f"Storage data shape: {storage_features.shape}")
        logger.info(f"Memory data shape: {memory_features.shape}")
        logger.info(f"Combined data shape: {combined_features.shape}")
        
        logger.info("Generating plots")
        for key in ["storage", "memory", "combined"]:
            binary_output_file = os.path.join(args.output_dir, "binary_class_balance.png")
            explore.plot_class_balance(
                binary_labels,
                output_file=binary_output_file
            )
            logger.info(f"Generated binary class balance plot")

            original_output_file = os.path.join(args.output_dir, "original_class_balance.png")
            explore.plot_class_balance(
                original_labels,
                output_file=original_output_file
            )
            logger.info(f"Generated original class balance plot")
            output_file_missing = os.path.join(args.output_dir, f"{key}_missing.png")
            explore.plot_missing_data(
                data_dict[key],  
                output_file=output_file_missing
            )
        
        logger.info("All visualizations completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__=='__main__':
    main()