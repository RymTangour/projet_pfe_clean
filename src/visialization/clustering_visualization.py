# Import numpy and other required libraries here to avoid import errors
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import LabelEncoder
import logging
import argparse
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
from collections import Counter

from src import get_logger 
from visualization_feature_processor import DataLoader , EnhancedVisualizer
from run_visualization_scripts import parse_args

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()

class Cluster(EnhancedVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _gpu_memory_efficient_preprocessing(self, features, labels=None, max_samples=5000, scale_method='robust'):
        if max_samples is not None and max_samples < features.shape[0]:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            self.logger.info(f"Sampled {max_samples} samples for processing")
        
        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.warning("Data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaler = scalers.get(scale_method, StandardScaler())
        preprocessed_features = scaler.fit_transform(features)
        
        if preprocessed_features.shape[1] > 100:
            variances = np.var(preprocessed_features, axis=0)
            var_threshold = np.percentile(variances, 50)
            high_var_indices = np.where(variances > var_threshold)[0]
            
            if labels is not None and len(np.unique(labels)) > 1:
                numeric_labels = (LabelEncoder().fit_transform(labels) 
                                  if not np.issubdtype(labels.dtype, np.number) 
                                  else labels)
                
                mi_scores = mutual_info_classif(preprocessed_features, numeric_labels)
                mi_threshold = np.percentile(mi_scores, 70)
                high_mi_indices = np.where(mi_scores > mi_threshold)[0]
                
                selected_indices = np.union1d(
                    high_var_indices[:50], 
                    high_mi_indices[:50]
                )[:100]
            else:
                selected_indices = high_var_indices[:100]
            
            selected_features = preprocessed_features[:, selected_indices]
        else:
            selected_features = preprocessed_features
            selected_indices = np.arange(preprocessed_features.shape[1])
        
        return selected_features, selected_indices, labels

    def _gpu_optimal_cluster_search(self, features, labels, max_clusters=12):
        numeric_labels = (LabelEncoder().fit_transform(labels) 
                          if not np.issubdtype(labels.dtype, np.number) 
                          else labels)
        
        n_unique_labels = len(np.unique(numeric_labels))
        
        min_k = max(2, n_unique_labels - 2)
        max_k = min(max_clusters, n_unique_labels + 5)
        
        self.logger.info(f"Searching for optimal clusters between {min_k} and {max_k}")
        
        best_score, best_k = -1, None
        
        for k in range(min_k, max_k + 1):
            try:
                cluster_assignments = fcluster(
                    linkage(features, method='ward'), 
                    k, 
                    criterion='maxclust'
                )
                
                if np.min(np.bincount(cluster_assignments)) < 5:
                    continue
                
                sil_score = silhouette_score(features, cluster_assignments)
                ari_score = adjusted_rand_score(numeric_labels, cluster_assignments)
                
                combined_score = (0.3 * sil_score) + (0.7 * ari_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
            except Exception as e:
                self.logger.warning(f"Cluster search error for k={k}: {e}")
        
        return best_k or n_unique_labels

    def plot_enhanced_hierarchical_clustering(
        self, 
        features, 
        labels=None, 
        data_type='combined', 
        n_clusters=None, 
        output_file=None,
        linkage_method='ward', 
        distance_metric='euclidean', 
        try_optimal_clusters=True, 
        max_clusters=12,
        scale_method='robust',
        max_samples=5000
    ):
        torch.cuda.empty_cache()
        
        selected_features, selected_indices, labels = self._gpu_memory_efficient_preprocessing(
            features, labels, max_samples, scale_method
        )
        
        sample_numbers = [f"Sample_{i+1}" for i in range(selected_features.shape[0])]
        feature_numbers = [f"Feature_{i+1}" for i in range(features.shape[1])]
        selected_feature_numbers = [feature_numbers[i] for i in selected_indices]
        
        if try_optimal_clusters and n_clusters is None and labels is not None:
            n_clusters = self._gpu_optimal_cluster_search(selected_features, labels, max_clusters)
            self.logger.info(f"Optimal clusters determined: {n_clusters}")
        
        if distance_metric != 'euclidean':
            dist_matrix = pdist(selected_features, metric=distance_metric)
            Z = linkage(dist_matrix, method=linkage_method)
        else:
            Z = linkage(selected_features, method=linkage_method)
        
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1], height_ratios=[1, 4, 0.5])
        
        ax_dendrogram_top = plt.subplot(gs[0, 0])
        ax_dendrogram_left = plt.subplot(gs[1, 0])
        ax_heatmap = plt.subplot(gs[1, 1])
        ax_labels = plt.subplot(gs[2, 0])
        
        dend_result = dendrogram(Z, no_plot=True)
        reordered_indices = dend_result['leaves']
        
        dendrogram(
            Z,
            ax=ax_dendrogram_top,
            orientation='top',
            color_threshold=None if n_clusters is None else Z[-(n_clusters-1), 2],
            above_threshold_color='gray',
            leaf_rotation=90,
            leaf_font_size=8,
            labels=sample_numbers
        )
        ax_dendrogram_top.set_xticks([])
        ax_dendrogram_top.set_xticklabels([])
        
        dendrogram(
            Z,
            ax=ax_dendrogram_left,
            orientation='left',
            color_threshold=None if n_clusters is None else Z[-(n_clusters-1), 2],
            above_threshold_color='gray',
            leaf_font_size=8,
            labels=None
        )
        ax_dendrogram_left.set_yticks([])
        ax_dendrogram_left.set_yticklabels([])
        
        heatmap_data = selected_features[reordered_indices, :]
        
        im = ax_heatmap.imshow(
            heatmap_data,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )
        
        if len(selected_feature_numbers) <= 20:
            ax_heatmap.set_xticks(np.arange(len(selected_feature_numbers)))
            ax_heatmap.set_xticklabels(selected_feature_numbers, rotation=90, fontsize=8)
        
        plt.colorbar(im, ax=ax_heatmap)
        
        method_str = f" (Method: {linkage_method}, Distance: {distance_metric})"
        fig.suptitle(f'Hierarchical Clustering Dendrogram ({data_type.capitalize()} Data){method_str}', fontsize=16)
        ax_dendrogram_top.set_title('Sample Clustering', fontsize=12)
        ax_heatmap.set_title('Feature Values', fontsize=12)
        
        if n_clusters is not None:
            cluster_assignments = fcluster(Z, n_clusters, criterion='maxclust')
            
            ax_dendrogram_top.axhline(
                y=Z[-(n_clusters-1), 2],
                color='r',
                linestyle='--',
                alpha=0.8
            )
            ax_dendrogram_top.text(
                len(selected_features)/2,
                Z[-(n_clusters-1), 2] + 0.1 * Z[-(n_clusters-1), 2],
                f'{n_clusters} clusters',
                color='r',
                fontweight='bold',
                ha='center'
            )
        
        if labels is not None and len(np.unique(labels)) > 1:
            if not np.issubdtype(labels.dtype, np.number):
                label_encoder = LabelEncoder()
                numeric_labels = label_encoder.fit_transform(labels)
                unique_labels = label_encoder.classes_
                label_names = unique_labels
            else:
                numeric_labels = labels
                unique_labels = np.unique(labels)
                label_names = [f"Class {label}" for label in unique_labels]
            
            reordered_labels = numeric_labels[reordered_indices]
            
            n_classes = len(unique_labels)
            class_cmap = plt.cm.get_cmap('tab20', n_classes) if n_classes <= 20 else plt.cm.get_cmap('nipy_spectral', n_classes)
            
            label_colors = [class_cmap(int(lbl)) for lbl in reordered_labels]
            ax_labels.imshow([reordered_labels], aspect='auto', cmap=class_cmap)
            ax_labels.set_title('Actual Classifications', fontsize=10)
            ax_labels.set_yticks([])
            
            if n_clusters is not None:
                ax_clusters = plt.subplot(gs[2, 1])
                
                reordered_clusters = cluster_assignments[reordered_indices]
                
                cluster_cmap = plt.cm.get_cmap('Paired', n_clusters) if n_clusters <= 12 else plt.cm.get_cmap('tab20', n_clusters)
                ax_clusters.imshow([reordered_clusters - 1], aspect='auto', cmap=cluster_cmap)
                ax_clusters.set_title('Cluster Assignments', fontsize=10)
                ax_clusters.set_yticks([])
                
                ari = adjusted_rand_score(numeric_labels, cluster_assignments)
                nmi = normalized_mutual_info_score(numeric_labels, cluster_assignments)
                homogeneity = homogeneity_score(numeric_labels, cluster_assignments)
                completeness = completeness_score(numeric_labels, cluster_assignments)
                
                plt.figtext(
                    0.5,
                    0.01,
                    f"Cluster-Class Overlap Metrics: ARI = {ari:.3f}, NMI = {nmi:.3f}, "
                    f"Homogeneity = {homogeneity:.3f}, Completeness = {completeness:.3f}",
                    fontsize=11,
                    ha='center',
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
                )
                
                from matplotlib.patches import Patch
                
                if n_classes > 10:
                    n_cols = 3
                    fontsize = 8
                else:
                    n_cols = 2
                    fontsize = 9
                    
                class_legend_elements = [Patch(facecolor=class_cmap(i), label=f"{label_names[i]}") 
                                        for i in range(len(unique_labels))]
                cluster_legend_elements = [Patch(facecolor=cluster_cmap(i), label=f"Cluster {i+1}") 
                                        for i in range(n_clusters)]
                
                ax_labels.legend(handles=class_legend_elements, loc='upper left', 
                                bbox_to_anchor=(0, -0.1), ncol=n_cols, fontsize=fontsize)
                ax_clusters.legend(handles=cluster_legend_elements, loc='upper left', 
                                bbox_to_anchor=(0, -0.1), ncol=min(5, n_clusters), fontsize=fontsize)
                
                class_cluster_dict = {}
                
                for cl in range(1, n_clusters + 1):
                    class_distribution = Counter(numeric_labels[cluster_assignments == cl])
                    class_dist_names = {label_names[cls]: count for cls, count in class_distribution.items()}
                    class_cluster_dict[f"Cluster {cl}"] = class_dist_names
                
                cluster_class_text = "Cluster contents:\n"
                for cluster, class_dist in class_cluster_dict.items():
                    total = sum(class_dist.values())
                    top_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_str = ", ".join([f"{cls} ({count}/{total}, {count/total:.1%})" for cls, count in top_classes])
                    cluster_class_text += f"{cluster}: {top_str}\n"
                
                plt.figtext(
                    0.02,
                    0.5,
                    cluster_class_text,
                    fontsize=9,
                    ha='left',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
                )
                
            class_counts = {label_names[i]: np.sum(numeric_labels == i) for i in range(len(unique_labels))}
            
            count_text = "\n".join([
                f"{label}: {count}" for label, count in class_counts.items()
            ])
            
            plt.figtext(
                0.02,
                0.02,
                f"Sample counts:\n{count_text}",
                fontsize=9,
                ha='left',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
            )
            
        plt.figtext(
            0.98,
            0.02,
            f"Data type: {data_type}\nSamples: {selected_features.shape[0]}\nFeatures: {len(selected_indices)} selected of {features.shape[1]} total",
            fontsize=9,
            ha='right',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved hierarchical clustering plot to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        
        results = {
            'linkage_matrix': Z,
            'method': linkage_method,
            'distance': distance_metric,
            'selected_features': selected_indices,
        }
        
        if n_clusters is not None:
            results['cluster_assignments'] = cluster_assignments
            
        if n_clusters is not None and labels is not None:
            results['metrics'] = {
                'ari': ari,
                'nmi': nmi,
                'homogeneity': homogeneity,
                'completeness': completeness
            }
        
        torch.cuda.empty_cache()
        
        return results

    def plot_minimum_spanning_tree(self, features, labels=None, data_type='combined', output_file=None, use_gpu=True, max_samples=5000):
        """
        Create a minimum spanning tree visualization to show relationships between samples.
        GPU-optimized version with sample limiting to handle large datasets.
        
        Args:
            features (numpy.ndarray): The feature matrix
            labels (numpy.ndarray, optional): Labels for coloring nodes
            data_type (str): Type of data being analyzed
            output_file (str, optional): Path to save the figure
            use_gpu (bool): Whether to use GPU acceleration
            max_samples (int): Maximum number of samples to use for visualization
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Sample data if it's too large
        n_original = features.shape[0]
        if n_original > max_samples:
            self.logger.info(f"Dataset too large ({n_original} samples), sampling {max_samples} points for visualization")
            # Use stratified sampling if labels are provided
            if labels is not None:
                from sklearn.model_selection import train_test_split
                _, sample_features, _, sample_labels = train_test_split(
                    features, labels, 
                    test_size=max_samples/n_original, 
                    stratify=labels if len(np.unique(labels)) < max_samples/10 else None,
                    random_state=self.random_state
                )
                features = sample_features
                labels = sample_labels
            else:
                # Random sampling without labels
                indices = np.random.RandomState(self.random_state).choice(
                    n_original, size=max_samples, replace=False
                )
                features = features[indices]
                if labels is not None:
                    labels = labels[indices]
        
        n_samples = features.shape[0]
        self.logger.info(f"Creating minimum spanning tree visualization for {data_type} data with {n_samples} samples")
        
        # Handle NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.warning(f"{data_type} data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate distances (chunked approach for large datasets)
        def calculate_distances_chunked(feat_matrix, chunk_size=1000, gpu=False):
            """Calculate pairwise distances in memory-efficient chunks"""
            n = feat_matrix.shape[0]
            distances = np.zeros((n, n))
            
            if gpu:
                try:
                    import cupy as cp
                    import cupyx.scipy.spatial.distance as gpu_distance
                    
                    for i in range(0, n, chunk_size):
                        end_i = min(i + chunk_size, n)
                        chunk_i = cp.asarray(feat_matrix[i:end_i])
                        
                        for j in range(0, n, chunk_size):
                            end_j = min(j + chunk_size, n)
                            chunk_j = cp.asarray(feat_matrix[j:end_j])
                            
                            # Calculate distances for this chunk
                            chunk_distances = gpu_distance.cdist(chunk_i, chunk_j, metric='euclidean')
                            distances[i:end_i, j:end_j] = cp.asnumpy(chunk_distances)
                            
                            # Free GPU memory after each chunk
                            del chunk_j, chunk_distances
                            cp.get_default_memory_pool().free_all_blocks()
                        
                        # Free GPU memory after each outer chunk
                        del chunk_i
                        cp.get_default_memory_pool().free_all_blocks()
                    
                    return distances
                
                except (ImportError, ModuleNotFoundError):
                    self.logger.warning("CuPy not available. Falling back to CPU calculation.")
                    return calculate_distances_chunked(feat_matrix, chunk_size, gpu=False)
            else:
                # CPU version with sklearn
                from sklearn.metrics import pairwise_distances
                for i in range(0, n, chunk_size):
                    end_i = min(i + chunk_size, n)
                    chunk_i = feat_matrix[i:end_i]
                    
                    for j in range(i, n, chunk_size):  # Start from i to avoid redundant calculations
                        end_j = min(j + chunk_size, n)
                        chunk_j = feat_matrix[j:end_j]
                        
                        # Calculate distances for this chunk
                        chunk_distances = pairwise_distances(chunk_i, chunk_j, metric='euclidean', n_jobs=-1)
                        
                        # Store distances (symmetric matrix)
                        distances[i:end_i, j:end_j] = chunk_distances
                        if i != j:  # Fill in the symmetric part
                            distances[j:end_j, i:end_i] = chunk_distances.T
                
                return distances
        
        # Use a chunked approach for distance calculation
        chunk_size = min(1000, n_samples // 2)
        distances = calculate_distances_chunked(features, chunk_size=chunk_size, gpu=use_gpu)
        
        # Create MST using memory-efficient sparse approach
        import scipy.sparse as sparse
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        # For each node, keep only connections to k nearest neighbors
        k = min(50, n_samples // 5)  # Adaptive k based on dataset size
        
        # Build sparse graph with k-nearest neighbors approach
        rows, cols, data = [], [], []
        for i in range(n_samples):
            # Get k nearest neighbors for this point
            nearest = np.argsort(distances[i])[:k+1]
            for j in nearest:
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    data.append(distances[i, j])
        
        # Create sparse distance matrix
        sparse_distances = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        
        # Compute the minimum spanning tree
        mst_sparse = minimum_spanning_tree(sparse_distances)
        
        # Convert to NetworkX graph
        G = nx.from_scipy_sparse_array(mst_sparse)
        mst = G  # The result is already an MST
        
        # Set up node colors based on labels
        if labels is not None:
            # Convert labels to numeric if they are strings
            if not np.issubdtype(np.array(labels).dtype, np.number):
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                numeric_labels = label_encoder.fit_transform(labels)
                unique_labels = label_encoder.classes_
            else:
                numeric_labels = np.array(labels)
                unique_labels = np.unique(labels)
            
            # Choose colormap based on number of classes
            if len(unique_labels) <= 10:
                cmap = plt.cm.tab10
            elif len(unique_labels) <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.nipy_spectral
            
            # Create color map
            node_colors = [cmap(numeric_labels[i] / len(unique_labels)) for i in range(len(numeric_labels))]
            
            # Create a mapping for the legend
            legend_elements = [
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=cmap(i / len(unique_labels)),
                    markersize=10,
                    label=unique_labels[i] if not np.issubdtype(np.array(labels).dtype, np.number) else f"Class {i}"
                )
                for i in range(len(unique_labels))
            ]
        else:
            node_colors = ['skyblue'] * n_samples
            legend_elements = None
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Try different layouts and use the best one
        try:
            # For large graphs, try to use Graphviz if available
            if n_samples > 1000:
                try:
                    from networkx.drawing.nx_agraph import graphviz_layout
                    pos = graphviz_layout(mst, prog='sfdp')
                    self.logger.info("Using Graphviz sfdp layout for large graph")
                except:
                    self.logger.warning("Graphviz not available, falling back to spring layout")
                    pos = nx.spring_layout(mst, seed=self.random_state)
            else:
                pos = nx.spring_layout(mst, seed=self.random_state)
        except:
            try:
                pos = nx.kamada_kawai_layout(mst)
            except:
                pos = nx.shell_layout(mst)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            mst,
            pos,
            node_size=100,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='k',
            linewidths=0.5
        )
        
        # Process edges in batches to reduce memory consumption
        def get_edge_widths(graph, batch_size=500):
            """Process edges in batches to reduce memory consumption"""
            all_edges = list(graph.edges(data=True))
            edge_widths = []
            
            for i in range(0, len(all_edges), batch_size):
                batch = all_edges[i:i+batch_size]
                weights = [1 / edge[2]['weight'] for edge in batch]
                edge_widths.extend(weights)
            
            return edge_widths
        
        # Get edge widths
        edge_widths = get_edge_widths(mst)
        
        # Normalize edge widths for better visualization
        if edge_widths:
            edge_widths = np.array(edge_widths)
            min_width = edge_widths.min() if len(edge_widths) > 0 else 0
            max_width = edge_widths.max() if len(edge_widths) > 0 else 0
            
            # Avoid division by zero
            if max_width > min_width:
                edge_widths = 1 + 5 * (edge_widths - min_width) / (max_width - min_width)
            else:
                edge_widths = np.ones_like(edge_widths)
            
            nx.draw_networkx_edges(
                mst,
                pos,
                width=edge_widths,
                alpha=0.6,
                edge_color='gray'
            )
        
        # Add node labels if the network is small
        if n_samples <= 50:
            # Create labels dictionary
            if labels is not None:
                node_labels = {i: str(labels[i]) for i in range(len(labels))}
            else:
                node_labels = {i: str(i) for i in range(n_samples)}
            
            nx.draw_networkx_labels(
                mst,
                pos,
                labels=node_labels,
                font_size=8,
                font_color='black',
                font_weight='bold'
            )
        
        # Add title and legend
        plt.title(f"Minimum Spanning Tree - {data_type.capitalize()} Data", fontsize=16)
        if legend_elements:
            plt.legend(handles=legend_elements, title="Classes", loc="best")
        
        # Remove axis
        plt.axis('off')
        
        # Add information text about preprocessing
        info_text = f"Preprocessing: {data_type.capitalize()} specialized pipeline\n"
        info_text += f"Original samples: {n_original}, Visualized: {n_samples}\n"
        info_text += f"Nodes: {n_samples}, Edges: {len(mst.edges())}\n"
        info_text += f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}"
        plt.figtext(0.02, 0.02, info_text, fontsize=10, ha='left')
        
        # Save or display the figure
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved MST visualization to {output_file}")
        else:
            plt.show()
        
        plt.close()
        
        return mst
def main():
    args = parse_args()
    try:
        # Create output directory if it doesn't exist
        import os
        from pathlib import Path
        
        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
   
        # Load data
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = DataLoader(args.data_dir)
        data_dict = data_loader.load_data()
        
        cluster = Cluster(
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
        for key in ["storage", "memory", "combined"]:
            if key not in data_dict:
                logger.warning(f"{key} data not found in data_dict")
                continue
                
            logger.info(f"{key} data shape: {data_dict[key].shape}")
            logger.info(f"{key} data min: {np.min(data_dict[key])}, max: {np.max(data_dict[key])}")
            logger.info(f"{key} data mean: {np.mean(data_dict[key])}, std: {np.std(data_dict[key])}")
            logger.info(f"NaN values in {key}: {np.isnan(data_dict[key]).sum()}")
            logger.info(f"Inf values in {key}: {np.isinf(data_dict[key]).sum()}")
           
            
            # Create MST visualization
            try:
                # Create output filename for MST
                if output_dir:
                    mst_output_file_original = output_dir / f"minimum_spanning_tree_{key}_original_classification.png"
                    mst_output_file_binary = output_dir / f"minimum_spanning_tree_{key}_binary_classification.png"

                    
                    logger.info(f"Creating minimum spanning tree visualization for {key} data")
                    _ = cluster.plot_minimum_spanning_tree(
                        features=data_dict[key],
                        labels=data_dict['original_labels'],
                        data_type=key,
                        output_file=mst_output_file_original
                    )
                    
                    _ = cluster.plot_minimum_spanning_tree(
                        features=data_dict[key],
                        labels=data_dict['binary_labels'],
                        data_type=key,
                        output_file=mst_output_file_binary
                    )
                logger.info(f"Completed MST visualization for {key} data")
            except Exception as e:
                logger.error(f"Error in MST visualization: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Try different clustering approaches and parameters
            methods_to_try = [
                {'linkage': 'ward', 'distance': 'euclidean'},
                {'linkage': 'complete', 'distance': 'correlation'},
                {'linkage': 'average', 'distance': 'cosine'}
            ]
            
            scaling_methods = ['robust', 'standard']
            
            for labels_to_use in [data_dict['binary_labels'], data_dict['original_labels']]:
                # Determine which label type we're using for file naming
                label_type = "binary" if np.array_equal(labels_to_use, data_dict['binary_labels']) else "original"
                
                best_ari = -1
                best_config = None
                best_results = None
                
                # Try each combination
                for method in methods_to_try:
                    for scaling in scaling_methods:
                        try:
                            # Create descriptive output filename
                            method_name = f"{method['linkage']}_{method['distance']}"
                            output_file = None
                            if output_dir:
                                output_file = output_dir / f"hierarchical_clustering_{key}_{label_type}_{method_name}_{scaling}.png"
                            
                            # Run enhanced clustering with automatic cluster number detection
                            logger.info(f"Trying {method_name} with {scaling} scaling for {key} data ({label_type} labels)")
                            
                            results = cluster.plot_enhanced_hierarchical_clustering(
                                features=data_dict[key],
                                labels=labels_to_use,
                                data_type=f"{key} ({label_type} labels)",
                                try_optimal_clusters=True,  # Automatically detect clusters
                                linkage_method=method['linkage'],
                                distance_metric=method['distance'],
                                scale_method=scaling,
                                output_file=output_file,
                                max_samples=5000  # Pass max_samples with default value 5000
                            )
                            
                            # Track best performing configuration
                            if 'metrics' in results and results['metrics']['ari'] > best_ari:
                                best_ari = results['metrics']['ari']
                                best_config = {
                                    'method': method_name,
                                    'scaling': scaling,
                                    'metrics': results['metrics']
                                }
                                best_results = results
                                
                            logger.info(f"Completed {method_name} clustering for {key} data ({label_type} labels)")
                            
                        except Exception as e:
                            logger.error(f"Error in {method_name} clustering: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                
                # Report best configuration
                if best_config:
                    logger.info(f"Best clustering configuration for {key} data ({label_type} labels):")
                    logger.info(f"Method: {best_config['method']}, Scaling: {best_config['scaling']}")
                    logger.info(f"Metrics: ARI={best_config['metrics']['ari']:.3f}, NMI={best_config['metrics']['nmi']:.3f}")
                    
                    # Create a detailed final plot with the best configuration
                    if output_dir:
                        final_output = output_dir / f"hierarchical_clustering_{key}_{label_type}_BEST.png"
                        
                        # Extract best parameters
                        best_method = best_config['method'].split('_')[0]
                        best_distance = best_config['method'].split('_')[1]
                        
                        # Get optimal number of clusters from best results
                        n_clusters = None
                        if 'cluster_assignments' in best_results:
                            n_clusters = len(np.unique(best_results['cluster_assignments']))
                        
                        # Run one more time with the best parameters
                        _ = cluster.plot_enhanced_hierarchical_clustering(
                            features=data_dict[key],
                            labels=labels_to_use,
                            data_type=f"{key} ({label_type} labels - Best Configuration)",
                            n_clusters=n_clusters,
                            linkage_method=best_method,
                            distance_metric=best_distance,
                            scale_method=best_config['scaling'],
                            output_file=final_output,
                            max_samples=5000  # Pass max_samples with default value 5000
                        )
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__=='__main__':
    main()
