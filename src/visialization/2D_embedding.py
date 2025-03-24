import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from pathlib import Path
import logging
import argparse
import warnings
from sklearn.metrics.pairwise import pairwise_distances
from src import get_logger 
from visualization_feature_processor import DataLoader , EnhancedVisualizer
from run_visualization_scripts import parse_args


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()


class Cluster(EnhancedVisualizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_specialized_visualization(self, features, labels, plot_title, data_type, 
                               output_file=None, use_original_labels=False, method='tsne'):
        """Create visualization with customized preprocessing per data type"""
        # First check if features and labels have matching lengths
        if len(features) != len(labels):
            self.logger.error(f"Mismatched lengths: features {len(features)}, labels {len(labels)}")
            raise ValueError("Features and labels must have the same length")
            
        # Get initial shape
        initial_shape = features.shape
        self.logger.info(f"Initial features shape: {initial_shape}, Labels length: {len(labels)}")
        
        # Sample data if needed for performance
        features, labels = self.sample_data_if_needed(features, labels)
        self.logger.info(f"After sampling: Features shape: {features.shape}, Labels length: {len(labels)}")
        
        # Apply specialized preprocessing based on data type
        if data_type == "storage":
            preprocessed_features, outlier_mask = self.preprocess_storage_features(features)
        elif data_type == "memory":
            preprocessed_features, outlier_mask = self.preprocess_memory_features(features)
        elif data_type == "combined":
            preprocessed_features, outlier_mask = self.preprocess_combined_features(features[:, :5], features[:, 5:])
            self.logger.info(f"Combined data type: {data_type}. Using generic preprocessing.Rymm{preprocessed_features.shape}")
        else:
            self.logger.warning(f"Unknown data type: {data_type}. Using generic preprocessing.Rymm{preprocessed_features.shape}")
            # Legacy preprocessing as fallback
            preprocessed_features, outlier_mask = self.preprocess_storage_features(features)
        
        # If outliers were removed, also filter labels
        if outlier_mask is not None:
            labels = labels[outlier_mask]
            self.logger.info(f"After outlier removal: Features shape: {preprocessed_features.shape}, "
                        f"Labels length: {len(labels)}")
                        
        # Generate embedding based on chosen method
        if method == 'tsne':
            # Adjust t-SNE parameters based on data type
            if data_type == "memory":
                perplexity = min(25, preprocessed_features.shape[0] - 1)  # Lower perplexity for memory
                learning_rate = 150
                early_exaggeration = 14.0  # Higher exaggeration for memory
                n_iter = 3000  # More iterations for memory
            elif data_type == "combined":
                perplexity = min(30, preprocessed_features.shape[0] - 1)
                learning_rate = 180
                early_exaggeration = 13.0
                n_iter = 2800
            else:  # storage or default
                perplexity = min(self.perplexity, preprocessed_features.shape[0] - 1)
                learning_rate = self.learning_rate
                early_exaggeration = 12.0
                n_iter = self.max_iter
                
            self.logger.info(f"Applying t-SNE for {data_type} data with perplexity={perplexity}, "
                        f"learning_rate={learning_rate}, max_iter={n_iter}, "
                        f"early_exaggeration={early_exaggeration}")
            
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,
                    max_iter=n_iter,
                    n_iter_without_progress=300,
                    min_grad_norm=1e-07,
                    metric='euclidean',
                    init='pca',
                    verbose=1,
                    random_state=self.random_state,
                    method='barnes_hut',
                    angle=0.5,
                )
                embedding = tsne.fit_transform(preprocessed_features)
                
                # Handle NaN values in the embedding
                if np.isnan(embedding).any():
                    self.logger.warning("t-SNE produced NaN values. Replacing with zeros.")
                    embedding = np.nan_to_num(embedding)
                    
            except Exception as e:
                self.logger.error(f"t-SNE failed: {str(e)}")
                self.logger.info("Falling back to PCA")
                pca = PCA(n_components=2, random_state=self.random_state)
                embedding = pca.fit_transform(preprocessed_features)
            
        elif method == 'umap':
            try:
                import umap
                
                # Customize UMAP parameters based on data type
                if data_type == "memory":
                    n_neighbors = 10  # Fewer neighbors for memory data
                    min_dist = 0.05   # Smaller min_dist for memory data
                elif data_type == "combined":
                    n_neighbors = 12
                    min_dist = 0.08
                else:  # storage or default
                    n_neighbors = 15
                    min_dist = 0.1
                    
                self.logger.info(f"Applying UMAP for {data_type} data with n_neighbors={n_neighbors}, min_dist={min_dist}")
                
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=self.random_state
                )
                embedding = reducer.fit_transform(preprocessed_features)
            except ImportError:
                self.logger.error("UMAP not installed. Please install with: pip install umap-learn")
                self.logger.info("Falling back to PCA")
                pca = PCA(n_components=2, random_state=self.random_state)
                embedding = pca.fit_transform(preprocessed_features)
            except Exception as e:
                self.logger.error(f"UMAP failed: {str(e)}")
                self.logger.info("Falling back to PCA")
                pca = PCA(n_components=2, random_state=self.random_state)
                embedding = pca.fit_transform(preprocessed_features)
        
        elif method == 'pca':
            self.logger.info(f"Applying PCA for  data")
            # Added PCA as a primary method option
            if data_type == "storage":
                from sklearn.preprocessing import RobustScaler, StandardScaler
            
                robust_scaler = RobustScaler()
                robust_features = robust_scaler.fit_transform(preprocessed_features)
        
        # Then apply standard scaling to normalize the data
                std_scaler = StandardScaler()
                preprocessed_features = std_scaler.fit_transform(robust_features)
           
            try:
                # Apply PCA dimensionality reduction
                pca = PCA(
                    n_components=2,
                    random_state=self.random_state,
                )
                embedding = pca.fit_transform(preprocessed_features)
                
                # Calculate explained variance ratio for annotation
                explained_variance = pca.explained_variance_ratio_
                total_explained_variance = np.sum(explained_variance) * 100
                
                # Store for later annotation
                self.pca_explained_variance = explained_variance
                self.pca_total_explained_variance = total_explained_variance
                
                self.logger.info(f"PCA explained variance ratio: {explained_variance}")
                self.logger.info(f"Total explained variance: {total_explained_variance:.2f}%")
                
            except Exception as e:
                self.logger.error(f"PCA failed: {str(e)}")
                # If PCA fails (unlikely), use a simpler approach
                self.logger.info("Falling back to standard scaling and PCA")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)  # Use original features as fallback
                pca = PCA(n_components=2, random_state=self.random_state)
                embedding = pca.fit_transform(scaled_features)
        else:
            self.logger.warning(f"Unknown method: {method}. Falling back to PCA.")
            pca = PCA(n_components=2, random_state=self.random_state)
            embedding = pca.fit_transform(preprocessed_features)
        
        # Apply specialized enhancement based on data type
        embedding = self.enhance_separation_specialized(embedding, labels, data_type)
        
        # Create DataFrame for plotting
        df_viz = pd.DataFrame(embedding, columns=['Dim 1', 'Dim 2'])
        df_viz['Label'] = labels
        
        # Create plot
        plt.figure(figsize=(14, 12))
        
        if use_original_labels:
            # Use original program names as labels
            unique_labels = np.unique(labels)
            self.logger.info(f"Plotting with {len(unique_labels)} unique program labels")
            
            # Custom marker size based on data type
            if data_type == "memory":
                marker_size = 100  # Larger markers for memory
            elif data_type == "combined":
                marker_size = 95
            else:
                marker_size = 90
                
            # Use a colormap with enough distinct colors for all programs
            # Get distinct colors for each program
            if len(unique_labels) <= 10:
                cmap = plt.cm.tab10
            elif len(unique_labels) <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.nipy_spectral
                
            colors = [cmap(i) for i in np.linspace(0, 1, len(unique_labels))]
            program_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # Plot each program with its own color
            for i, label in enumerate(unique_labels):
                indices = df_viz['Label'] == label
                
                # Set fixed marker properties for all programs
                alpha = 0.8
                marker = 'o'
                size = marker_size
                color = program_color_map[label]
                
                plt.scatter(
                    df_viz.loc[indices, 'Dim 1'], 
                    df_viz.loc[indices, 'Dim 2'],
                    c=[color],  # Use assigned color from the colormap
                    label=label,
                    alpha=alpha,
                    edgecolors='w',
                    s=size,
                    marker=marker
                )
                
                # Add label at the centroid of each cluster if there are points
                if np.sum(indices) > 0:
                    centroid_x = np.mean(df_viz.loc[indices, 'Dim 1'])
                    centroid_y = np.mean(df_viz.loc[indices, 'Dim 2'])
                    plt.text(centroid_x, centroid_y, label, 
                            fontsize=12, fontweight='bold', 
                            ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
        else:
            colors = {'Benign': 'dodgerblue', 'Malicious': 'orange'}
            if data_type == "memory":
                marker_size = 100  # Larger markers for memory
            elif data_type == "combined":
                marker_size = 95
            else:
                marker_size = 90

            for label, color in colors.items():
                indices = df_viz['Label'] == label
                if np.sum(indices) > 0:  # Only plot if there are points with this label
                    plt.scatter(
                        df_viz.loc[indices, 'Dim 1'], 
                        df_viz.loc[indices, 'Dim 2'],
                        c=color,
                        label=label,
                        alpha=0.8,
                        edgecolors='w',
                        linewidth=0.5,
                        s=marker_size
                    )
        
        # Set the title and labels based on the dimensionality reduction method
        method_name = method.upper()
        plt.title(f"{plot_title} ({data_type.capitalize()} Data)", fontsize=18, pad=20)
        plt.xlabel(f'{method_name} Dimension 1', fontsize=14)
        plt.ylabel(f'{method_name} Dimension 2', fontsize=14)
        
        if not use_original_labels or len(unique_labels) <= 15:
            legend = plt.legend(title="Program Type" if not use_original_labels else "Program Name", 
                    fontsize=12, markerscale=1.2, loc='best')
            if use_original_labels:
                from matplotlib.lines import Line2D
                malicious_programs = [prog for prog in unique_labels if prog in self.ransomware_programs]
                benign_programs = [prog for prog in unique_labels if prog not in self.ransomware_programs]
                
                # Create a legend entry for malicious and benign categories
                custom_lines = []
                custom_labels = []
                
                if malicious_programs:
                    custom_lines.append(Line2D([0], [0], color='red', lw=4))
                    custom_labels.append('Malicious Programs')
                    
                if benign_programs:
                    custom_lines.append(Line2D([0], [0], color='blue', lw=4))
                    custom_labels.append('Benign Programs')
            
                if custom_lines:
                    second_legend = plt.legend(custom_lines, custom_labels, 
                                            loc='lower right', fontsize=12, 
                                            title="Program Categories")
                    
                    # Add the original legend back
                    plt.gca().add_artist(legend)
        
        # Add grid and style improvements
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Add annotation about the specialized preprocessing
        if data_type == "memory":
            preprocessing_text = "Preprocessing: MinMaxScaler + Variance Filtering"
        elif data_type == "combined":
            preprocessing_text = "Preprocessing: StandardScaler + Variance Filtering"
        else:
            preprocessing_text = "Preprocessing: RobustScaler + PCA"
            
        # Add method-specific details to annotation
        if method == 'tsne':
            if data_type == "memory":
                method_text = f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate}, exaggeration={early_exaggeration}"
            elif data_type == "combined":
                method_text = f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate}, max_iter={n_iter}"
            else:
                method_text = f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate}, max_iter={n_iter}"
        elif method == 'umap':
            if data_type == "memory":
                method_text = f"UMAP: n_neighbors=10, min_dist=0.05"
            elif data_type == "combined":
                method_text = f"UMAP: n_neighbors=12, min_dist=0.08"
            else:
                method_text = f"UMAP: n_neighbors=15, min_dist=0.1"
        elif method == 'pca':
            # Add PCA-specific annotation showing variance explained
            if hasattr(self, 'pca_explained_variance'):
                var1 = self.pca_explained_variance[0] * 100
                var2 = self.pca_explained_variance[1] * 100
                total_var = self.pca_total_explained_variance
                method_text = f"PCA: Dim1={var1:.1f}%, Dim2={var2:.1f}%, Total={total_var:.1f}%"
            else:
                method_text = "PCA: 2 components"
        else:
            method_text = f"{method.upper()}: standard parameters"
            
        plt.figtext(0.02, 0.02, preprocessing_text + "\n" + method_text, 
                fontsize=8, ha='left', va='bottom')

        if use_original_labels:
            color_legend_text = "Using distinct colors for each program"
        else:
            color_legend_text = "Color coding: Orange = Malicious, Blue = Benign"
            
        plt.figtext(0.5, 0.01, color_legend_text, fontsize=10, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
        
        plt.close()
    
        return embedding, df_viz
        
    def visualize_all_specialized(self, data_dict, output_dir=None, method='tsne'):
        """Create specialized visualizations for all data types"""
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
     
        # Show data shapes for reference
        self.logger.info(f"Storage shape: {data_dict['storage'].shape}")
        self.logger.info(f"Memory shape: {data_dict['memory'].shape}")
        self.logger.info(f"Combined shape: {data_dict['combined'].shape}")
        self.logger.info(f"Binary labels length: {len(data_dict['binary_labels'])}")
        self.logger.info(f"Original labels length: {len(data_dict['original_labels'])}")

        if (len(data_dict['binary_labels']) != data_dict['storage'].shape[0] or
            len(data_dict['original_labels']) != data_dict['storage'].shape[0]):
            self.logger.error("Label and feature lengths do not match. Cannot proceed with visualization.")
            return None, None
        
        try:

            self.logger.info(f"Creating {method.upper()} visualization for storage features with program labels")
            storage_embedding, storage_df = self.create_specialized_visualization(
                data_dict["storage"],
                data_dict["original_labels"],
                f"{method.upper()} of Storage Features (Program Classification)",
                data_type="storage",
                output_file=output_dir / f"{method}_storage_programs.png" if output_dir else None,
                use_original_labels=True,
                method=method
            )
        
            self.logger.info(f"Creating specialized {method.upper()} visualization for memory features with program labels")
            memory_embedding, memory_df = self.create_specialized_visualization(
                data_dict["memory"],
                data_dict["original_labels"],
                f"{method.upper()} of Memory Features (Program Classification)",
                data_type="memory",
                output_file=output_dir / f"{method}_memory_programs.png" if output_dir else None,
                use_original_labels=True,
                method=method
            )    
            self.logger.info(f"Creating specialized {method.upper()} visualization for combined features with program labels")
            combined_embedding, combined_df = self.create_specialized_visualization(
                data_dict["combined"],
                data_dict["original_labels"],
                f"{method.upper()} of Combined Features (Program Classification)",
                data_type="combined",
                output_file=output_dir / f"{method}_combined_programs.png" if output_dir else None,
                use_original_labels=True,
                method=method
            )
            self.logger.info(f"Creating specialized {method.upper()} visualization for memory features with binary labels")
            self.create_specialized_visualization(
                data_dict["memory"],
                data_dict["binary_labels"],
                f"{method.upper()} of Memory Features (Binary Classification)",
                data_type="memory",
                output_file=output_dir / f"{method}_memory_binary.png" if output_dir else None,
                use_original_labels=False,
                method=method
            )
           
            self.logger.info(f"Creating specialized {method.upper()} visualization for storage features with binary labels")
            self.create_specialized_visualization(
                data_dict["storage"],
                data_dict["binary_labels"],
                f"{method.upper()} of Storage Features (Binary Classification)",
                data_type="storage",
                output_file=output_dir / f"{method}_storage_binary.png" if output_dir else None,
                use_original_labels=False,
                method=method
            )
            
            # Combined with binary labels
            self.logger.info(f"Creating specialized {method.upper()} visualization for combined features with binary labels")
            self.create_specialized_visualization(
                data_dict["combined"],
                data_dict["binary_labels"],
                f"{method.upper()} of Combined Features (Binary Classification)",
                data_type="combined",
                output_file=output_dir / f"{method}_combined_binary.png" if output_dir else None,
                use_original_labels=False,
                method=method
            )
            
            return memory_embedding, memory_df
            
        except Exception as e:
            self.logger.error(f"Error in visualization process: {str(e)}")
            return None, None


def main():
    args = parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = DataLoader(args.data_dir)
        data_dict = data_loader.load_data()
        
        # Create visualizations
        logger.info(f"Creating {args.method.upper()} visualizations")
        visualizer = Cluster(
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            random_state=args.seed,
            outlier_method=args.outlier_method,
            iqr_factor=args.iqr_factor,
            zscore_threshold=args.zscore_threshold,
            contamination=args.contamination
        )
        
        # Run visualizations with the selected method
        if args.method == 'tsne':
            embedding, df = visualizer.visualize_all_specialized(data_dict, args.output_dir, method='tsne')
            logger.info("t-SNE visualization complete")
        elif args.method == 'pca':
            embedding, df = visualizer.visualize_all_specialized(data_dict, args.output_dir, method='pca')
            logger.info("PCA visualization complete")
        elif args.method == 'umap':
            try:
                import umap
                embedding, df = visualizer.visualize_all_specialized(data_dict, args.output_dir, method='umap')
                logger.info("UMAP visualization complete")
            except ImportError:
                logger.error("UMAP not installed. Please install with: pip install umap-learn")
                logger.info("Falling back to t-SNE visualization")
                embedding, df = visualizer.visualize_all_specialized(data_dict, args.output_dir, method='tsne')
        # Add this before visualization in main()
        for key in ["storage", "memory", "combined"]:
            print(f"{key} data shape: {data_dict[key].shape}")
            print(f"{key} data min: {np.min(data_dict[key])}, max: {np.max(data_dict[key])}")
            print(f"{key} data mean: {np.mean(data_dict[key])}, std: {np.std(data_dict[key])}")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()