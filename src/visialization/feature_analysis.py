import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from src import get_logger 
from visualization_feature_processor import DataLoader , EnhancedVisualizer
from run_visualization_scripts import parse_args

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()

   
class Correlation(EnhancedVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define list of known malicious programs
        self.malicious_programs = [ 'AESCrypt', 'Conti', 'Darkside', 'LockBit', 'REvil', 'Ryuk', 'WannaCry' ]


    def plot_feature_correlations(self, features, labels=None, data_type='combined', 
                    feature_names=None, output_file=None, n_top_features=30,
                    show_values=True, custom_cmap=None, mask_upper_triangle=False,
                    method='pearson', threshold=0.7, cluster=False,
                    highlight_correlations=True, class_names=None, title_suffix=None,
                    storage_memory_split=None, is_binary=False):
        
        # Import hierarchical clustering modules if clustering is enabled
        if cluster:
            from scipy.cluster.hierarchy import linkage, dendrogram
        
        self.logger.info(f"Creating correlation heatmap for {data_type} data using {method} method")
        
        # Handle NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.warning(f"{data_type} data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # No preprocessing - use raw features directly
        preprocessed_features = features
        outlier_mask = np.ones(features.shape[0], dtype=bool)
        self.logger.info(f"Using raw {data_type} data without preprocessing")
        
        # Apply outlier mask to labels if provided
        if labels is not None and outlier_mask is not None:
            labels = labels[outlier_mask]
        
        # Process labels if provided
        if labels is not None:
            if not np.issubdtype(np.array(labels).dtype, np.number):
                # For string labels, use the actual program names directly
                unique_labels = np.unique(labels)
                
                # Add "Malicious" to label names that are in the malicious_programs list
                label_names = []
                for label in unique_labels:
                    if label in self.malicious_programs:
                        label_names.append(f"{label} (Malicious)")
                    else:
                        label_names.append(label)
                
                # Create a mapping dictionary for string labels
                label_dict = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_dict[label] for label in labels])
            else:
                # For numeric labels, use provided class_names if available
                numeric_labels = np.array(labels)
                
                # Use actual program names if they're provided in class_names
                if class_names is not None:
                    label_names = []
                    for i in np.unique(numeric_labels):
                        prog_name = class_names.get(i, f"Program {i}")
                        if prog_name in self.malicious_programs:
                            label_names.append(f"{prog_name} (Malicious)")
                        else:
                            label_names.append(prog_name)
                else:
                    # Default to "Program X" instead of "Class X"
                    label_names = [f"Program {i}" for i in np.unique(numeric_labels)]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(preprocessed_features.shape[1])]
        
        # Ensure we have the correct number of feature names
        if len(feature_names) != preprocessed_features.shape[1]:
            self.logger.warning(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({preprocessed_features.shape[1]}). Adjusting.")
            if len(feature_names) > preprocessed_features.shape[1]:
                feature_names = feature_names[:preprocessed_features.shape[1]]
            else:
                feature_names.extend([f"Feature {i+1}" for i in range(len(feature_names), preprocessed_features.shape[1])])
        
        # Create DataFrame for correlation calculation - only numerical data
        features_df = pd.DataFrame(preprocessed_features, columns=feature_names)
        
        # Add labels for correlation calculation using actual values
        if labels is not None:
            # For all unique labels, create separate column for each program/label
            for i, label_idx in enumerate(sorted(np.unique(numeric_labels))):
                # Get the original label value
                if not np.issubdtype(np.array(labels).dtype, np.number):
                    # For string labels, get the original label from label_names
                    original_label = label_names[i] if i < len(label_names) else f"Label {label_idx}"
                else:
                    # For numeric labels, use the actual numeric value
                    original_label = f"Label {label_idx}"
                
                # Use original label as column name
                features_df[original_label] = (numeric_labels == label_idx).astype(int)
        
        # Calculate correlation matrix using specified method
        correlation_matrix = features_df.corr(method=method)
        
        # For feature-to-class correlation only, filter the matrix
        if labels is not None:
            # Identify program/label columns (all non-feature columns)
            feature_cols = feature_names.copy()
            program_cols = [col for col in correlation_matrix.columns if col not in feature_cols]
            
            # Extract just the feature-to-program correlations
            feature_program_corr = correlation_matrix.loc[feature_cols, program_cols]
            
            # Replace the full correlation matrix with just feature-to-program correlations
            correlation_matrix = feature_program_corr
        
        # Select top features based on correlation with programs
        if correlation_matrix.shape[0] > n_top_features:
            # Calculate maximum absolute correlation with any program
            max_corr = correlation_matrix.abs().max(axis=1)
            # Sort features by their maximum correlation
            sorted_features = max_corr.sort_values(ascending=False).index.tolist()
            # Keep only top n features
            selected_features = sorted_features[:n_top_features]
            # Filter correlation matrix to keep only selected features
            correlation_matrix = correlation_matrix.loc[selected_features, :]
        
        # Calculate optimal figure size based on content
        num_features = len(correlation_matrix)
        num_programs = len(correlation_matrix.columns)
        
        # Calculate width and height based on content size
        # Width depends on number of programs (columns)
        # Height depends on number of features (rows)
        width = max(10, min(20, 8 + num_programs * 0.8))  # Base width + adjustment per program
        height = max(8, min(24, 6 + num_features * 0.3))  # Base height + adjustment per feature
        
        # If we have very few features and programs, don't make the figure too small
        if num_features < 5 and num_programs < 5:
            width, height = max(width, 10), max(height, 8)
        
        # If we have many features but few programs, make it taller and narrower
        if num_features > 20 and num_programs < 5:
            width = max(10, min(15, width))
            height = max(height, min(30, num_features * 0.4))
        
        # If we have many programs but few features, make it wider and shorter
        if num_programs > 10 and num_features < 10:
            width = max(width, min(24, num_programs * 1.0))
            height = max(8, min(15, height))
        
        self.logger.info(f"Setting figure size to {width}x{height} for {num_features} features and {num_programs} programs")
        
        # Create figure with calculated dimensions
        plt.figure(figsize=(width, height))
        
        # Select colormap based on data type or use custom
        if custom_cmap:
            cmap = custom_cmap
        else:
            if data_type == "memory":
                cmap = "coolwarm"
            elif data_type == "storage":
                cmap = "RdBu_r"
            else:  # combined or default
                cmap = plt.cm.coolwarm.copy()
                cmap.set_bad('black')
        
        # Adjust font size based on grid size
        annot_fontsize = max(6, min(10, 12 - (num_features + num_programs) * 0.05))
        
        # Plot the heatmap
        sns.heatmap(
            correlation_matrix,
            annot=show_values,  # Show correlation values based on parameter
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            square=False,  # Set to False to allow better fitting to the figure size
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} correlation"},
            fmt=".2f" if show_values else "",  # Format for the annotations
            annot_kws={"size": annot_fontsize} if show_values else {}
        )
        
        # Add title
        classification_type = "Binary Labels" if is_binary else "All Classes"
        plot_title = f"Feature-{classification_type} Correlation - {data_type.capitalize()} Data ({method.capitalize()})"
        if title_suffix:
            plot_title += f" - {title_suffix}"
        plt.title(plot_title, fontsize=14)
        
        # Rotate x-axis labels for better readability
        locs, labels = plt.xticks()

        # Create new labels with line breaks
        new_labels = []
        for label in labels:
            # Get the original text
            orig_text = label.get_text()
            
            # If label contains parentheses, split at the opening parenthesis
            if '(' in orig_text:
                main_text = orig_text.split('(')[0].strip()
                paren_text = '(' + orig_text.split('(')[1]
                new_labels.append(f"{main_text}\n{paren_text}")
            else:
                # If no parentheses, just use the original text
                new_labels.append(orig_text)

        # Set the new tick labels with horizontal orientation
        plt.xticks(
            locs, 
            new_labels, 
            rotation=0,  # Horizontal labels
            ha='center',  # Center-align horizontally
            fontsize=max(7, min(10, 10 - num_programs * 0.05))
        )

        # Add more space at the bottom of the plot for the labels
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])         
        plt.yticks(rotation=0, fontsize=max(7, min(10, 10 - num_features * 0.02)))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Add information text about features
        feature_count = correlation_matrix.shape[0]
        program_count = correlation_matrix.shape[1]
        
        info_text = f"Preprocessing: Raw data (no preprocessing)\n"
        info_text += f"Features: {feature_count}/{features.shape[1]} (showing top features by correlation)\n"
        info_text += f"Classes: {program_count}\n"
        

        plt.figtext(0.02, -0.02, info_text, fontsize=9, ha='left')
        
        # Add information about high correlations if requested
        if highlight_correlations and threshold > 0:
            # Find features with high correlation to any program
            high_corr_features = []
            
            for feature in correlation_matrix.index:
                for program_col in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[feature, program_col]
                    if abs(corr_val) > threshold:
                        high_corr_features.append((feature, program_col, corr_val))
            
            if high_corr_features:
                # Sort by absolute correlation
                high_corr_features.sort(key=lambda x: abs(x[2]), reverse=True)
                
                high_corr_text = f"Features with |correlation| > {threshold}:\n\n"
                for feature, program_col, corr_val in high_corr_features[:15]:  # Show top 15
                    high_corr_text += f"{feature} & {program_col}: {corr_val:.3f}\n"
                
                if len(high_corr_features) > 15:
                    high_corr_text += f"... and {len(high_corr_features) - 15} more"
                
                # Position the text box dynamically based on figure size
                # For wider figures, place it further to the right
                x_pos = min(0.75, max(0.5, 1.0 - (10.0/width)))
                
                plt.figtext(
                    x_pos,
                    0.01,
                    high_corr_text,
                    fontsize=9,
                    ha='left',
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
                )
        
        # Save or display the figure
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved correlation heatmap to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
        
        plt.close()
        
        return correlation_matrix
    def plot_feature_importance(self, features, labels, data_type, feature_names=None, n_top_features=20, output_file=None):
        """
        Visualize feature importance using Random Forest classifier.
        
        Args:
            features (numpy.ndarray): The feature matrix
            labels (numpy.ndarray): Target labels (binary or multiclass)
            data_type (str): 'storage', 'memory', or 'combined'
            feature_names (list, optional): Names of features (default: indices)
            n_top_features (int, optional): Number of top features to display
            output_file (str, optional): Path to save the figure
        """
        self.logger.info(f"Creating feature importance plot for {data_type} data")
        
        # Handle NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.warning(f"{data_type} data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
        
        # Ensure we don't have too many feature names
        if len(feature_names) > features.shape[1]:
            feature_names = feature_names[:features.shape[1]]
        elif len(feature_names) < features.shape[1]:
            feature_names.extend([f'Feature {i+1}' for i in range(len(feature_names), features.shape[1])])
        
        # Encode labels if they are not numeric
        if not np.issubdtype(labels.dtype, np.number):
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
        else:
            encoded_labels = labels
        
        # Train a random forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(features, encoded_labels)
        
        # Get feature importances
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        
        # Sort feature indices by importance
        indices = np.argsort(importances)[::-1]
        
        # Limit to top N features
        n_top_features = min(n_top_features, len(indices))
        top_indices = indices[:n_top_features]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(
            range(n_top_features),
            importances[top_indices],
            xerr=std[top_indices],
            align='center',
            alpha=0.8,
            color='steelblue',
            ecolor='black',
            capsize=5
        )
        
        # Add labels and title
        plt.yticks(range(n_top_features), [feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {n_top_features} Feature Importance ({data_type.capitalize()} Data)')
        
        # Add importance values as text
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + std[top_indices][i] + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{importances[top_indices][i]:.3f}',
                va='center'
            )
        
        plt.tight_layout()
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved feature importance plot to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        
        return importances, indices
    
    def plot_feature_distributions(self, features, labels, feature_names=None, data_type='combined', n_features=10, output_file=None):
        """
        Plot distribution of top features across different classes.
        
        Args:
            features (numpy.ndarray): The feature matrix
            labels (numpy.ndarray): Labels for samples
            feature_names (list, optional): Names of features
            data_type (str): 'storage', 'memory', or 'combined'
            n_features (int): Number of top features to show
            output_file (str, optional): Path to save the figure
        """
        self.logger.info(f"Creating feature distribution plots for {data_type} data")
        
        # Handle NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.warning(f"{data_type} data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert labels to numeric if they are strings
        if not np.issubdtype(labels.dtype, np.number):
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
            unique_labels = label_encoder.classes_
        else:
            numeric_labels = labels
            unique_labels = np.unique(labels)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
        
        # Ensure we don't have too many feature names
        if len(feature_names) > features.shape[1]:
            feature_names = feature_names[:features.shape[1]]
        elif len(feature_names) < features.shape[1]:
            feature_names.extend([f'Feature {i+1}' for i in range(len(feature_names), features.shape[1])])
        
        # Calculate the feature variances for selection
        variances = np.var(features, axis=0)
        top_feature_idx = np.argsort(variances)[::-1][:n_features]
        
        # Choose colormap based on number of classes
        if len(unique_labels) <= 10:
            cmap = plt.cm.tab10
        elif len(unique_labels) <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral
        
        # Determine layout based on number of features
        n_cols = min(3, n_features)
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Create figure with increased size for better visibility
        # Increase overall figure size for better visualization
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        
        # Create a GridSpec to accommodate the legend on the left
        gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[0.2] + [1] * n_cols)
        
        # Create axes for the plots
        axes = []
        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j < n_features:
                    axes.append(fig.add_subplot(gs[i, j + 1]))  # +1 to skip legend column
        
        # Create a separate axis for the legend
        legend_ax = fig.add_subplot(gs[:, 0])
        legend_ax.axis('off')  # Hide the axis
        
        # Generate common handles and labels for the legend
        handles = []
        labels_list = []
        
        # Plot distributions for each selected feature
        for i, idx in enumerate(top_feature_idx):
            if i >= len(axes):
                break
                
            ax = axes[i]
            feature_vals = features[:, idx]
            
            # Plot KDE for each class
            for j, label in enumerate(np.unique(numeric_labels)):
                class_vals = feature_vals[numeric_labels == label]
                class_label = unique_labels[j] if not np.issubdtype(labels.dtype, np.number) else f"Class {label}"
                
                if len(class_vals) > 1:  # Need at least 2 points for KDE
                    sns.kdeplot(
                        class_vals,
                        ax=ax,
                        color=cmap(j / len(unique_labels)),
                        label=class_label,
                        fill=True,
                        alpha=0.3
                    )
                else:
                    # Fall back to histogram for small classes
                    ax.hist(
                        class_vals,
                        color=cmap(j / len(unique_labels)),
                        label=class_label,
                        alpha=0.5,
                        bins=1
                    )
                
                # Collect legend handles and labels only once
                if i == 0:
                    handles.append(plt.Line2D([0], [0], color=cmap(j / len(unique_labels)), lw=4))
                    labels_list.append(class_label)
            
            # Add feature statistics
            ax.axvline(np.mean(feature_vals), color='red', linestyle='--')
            ax.axvline(np.median(feature_vals), color='green', linestyle=':')
            
            if i == 0:
                handles.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=2))
                labels_list.append('Mean')
                handles.append(plt.Line2D([0], [0], color='green', linestyle=':', lw=2))
                labels_list.append('Median')
            
            # Calculate statistics
            feature_mean = np.mean(feature_vals)
            feature_std = np.std(feature_vals)
            feature_min = np.min(feature_vals)
            feature_max = np.max(feature_vals)
            
            stats_text = f"μ={feature_mean:.2f}\nσ={feature_std:.2f}\nmin={feature_min:.2f}\nmax={feature_max:.2f}"
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,  # Increased font size
                va='top',
                ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )
            
            # Set title and labels
            ax.set_title(feature_names[idx], fontsize=12)
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            
            # Remove legend from individual plots
            if hasattr(ax, 'legend_') and ax.legend_ is not None:
                ax.legend_.remove()
        
        # Create a single legend for all plots in the dedicated legend axis
        legend_ax.legend(handles, labels_list, loc='center', fontsize=12, frameon=True, 
                        title="Classes and Statistics", title_fontsize=14)
        
        # Add title for the entire figure
        fig.suptitle(f'Feature Distributions by Class ({data_type.capitalize()} Data)', fontsize=16, y=0.98)
        
        # Adjust the layout with more space between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8, w_pad=0.8)
        
        if output_file:
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved feature distribution plot to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        
        return top_feature_idx
def main():
    args = parse_args()
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logger.info(f"Created output directory: {args.output_dir}")
        
        # Load data
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = DataLoader(args.data_dir)
        data_dict = data_loader.load_data()
        
        # Extract original labels and binary labels
        original_labels = data_dict["original_labels"]
        binary_labels = data_dict["binary_labels"]  # This should be available from your DataLoader
        
        # Create visualizer instance
        logger.info("Initializing Correlation visualizer")
        correlation = Correlation(
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
        
        # Determine the storage-memory split point for combined data
        # If combined features count equals storage + memory, use storage count as split
        # Otherwise, guess the midpoint
        if combined_features.shape[1] == storage_features.shape[1] + memory_features.shape[1]:
            split_point = storage_features.shape[1]
            logger.info(f"Combined features detected as concatenation of storage ({split_point}) and memory ({memory_features.shape[1]}) features")
        else:
            split_point = combined_features.shape[1] // 2
            logger.info(f"Unable to determine exact split point, using midpoint: {split_point}")
        
        # Generate correlation plots for each data type with original multi-class labels
        # 1. Storage features correlation
        output_file = os.path.join(args.output_dir, f"correlation_storage_all_classes.png")
        logger.info(f"Generating correlation plot for storage data with all classes, saving to {output_file}")
        correlation.plot_feature_correlations(
            features=storage_features, 
            labels=original_labels,
            data_type="storage",
            output_file=output_file,
            title_suffix="All Classes",
            mask_upper_triangle=False,
            n_top_features=storage_features.shape[1],  # Use all features
            is_binary=False
        )
        
        # 2. Memory features correlation
        output_file = os.path.join(args.output_dir, f"correlation_memory_all_classes.png")
        logger.info(f"Generating correlation plot for memory data with all classes, saving to {output_file}")
        correlation.plot_feature_correlations(
            features=memory_features, 
            labels=original_labels,
            data_type="memory",
            output_file=output_file,
            title_suffix="All Classes",
            mask_upper_triangle=False,
            n_top_features=memory_features.shape[1],  # Use all features
            is_binary=False
        )
        
        # 3. Combined features correlation
        output_file = os.path.join(args.output_dir, f"correlation_combined_all_classes.png")
        logger.info(f"Generating correlation plot for combined data with all classes, saving to {output_file}")
        correlation.plot_feature_correlations(
            features=combined_features, 
            labels=original_labels,
            data_type="combined",
            output_file=output_file,
            title_suffix="All Classes",
            mask_upper_triangle=False,
            n_top_features=combined_features.shape[1],  # Use all features
            storage_memory_split=split_point,  # Pass the split point
            is_binary=False
        )
        
        # Generate correlation plots for each data type with binary class labels
        # 1. Storage features correlation with binary classes
        output_file = os.path.join(args.output_dir, f"correlation_storage_binary_classes.png")
        logger.info(f"Generating correlation plot for storage data with binary classes, saving to {output_file}")
        correlation.plot_feature_correlations(
            features=storage_features, 
            labels=binary_labels,
            data_type="storage",
            output_file=output_file,
            title_suffix="Binary Classification",
            mask_upper_triangle=False,
            n_top_features=storage_features.shape[1],  # Use all features
            is_binary=True
        )
        
        # 2. Memory features correlation with binary classes
        output_file = os.path.join(args.output_dir, f"correlation_memory_binary_classes.png")
        logger.info(f"Generating correlation plot for memory data with binary classes, saving to {output_file}")
        correlation.plot_feature_correlations(
            features=memory_features, 
            labels=binary_labels,
            data_type="memory",
            output_file=output_file,
            title_suffix="Binary Classification",
            mask_upper_triangle=False,
            n_top_features=memory_features.shape[1],  # Use all features
            is_binary=True
        )
        
        # 3. Combined features correlation with binary classes
        output_file = os.path.join(args.output_dir, f"correlation_combined_binary_classes.png")
        logger.info(f"Generating correlation plot for combined data with binary classes, saving to {output_file}")
        correlation.plot_feature_correlations(
            features=combined_features, 
            labels=binary_labels,
            data_type="combined",
            output_file=output_file,
            title_suffix="Binary Classification",
            mask_upper_triangle=False,
            n_top_features=combined_features.shape[1],  # Use all features
            storage_memory_split=split_point,  # Pass the split point
            is_binary=True
        )
        
        # Generate feature importance plots for multiclass classification
        logger.info("Generating feature importance plots for multiclass classification")
        
        # 1. Storage features importance
        output_file = os.path.join(args.output_dir, f"importance_storage_all_classes.png")
        correlation.plot_feature_importance(
            features=storage_features,
            labels=original_labels,
            data_type="storage",
            output_file=output_file,
            n_top_features=min(20, storage_features.shape[1])
        )
        
        # 2. Memory features importance
        output_file = os.path.join(args.output_dir, f"importance_memory_all_classes.png")
        correlation.plot_feature_importance(
            features=memory_features,
            labels=original_labels,
            data_type="memory",
            output_file=output_file,
            n_top_features=min(20, memory_features.shape[1])
        )
        
        # 3. Combined features importance
        output_file = os.path.join(args.output_dir, f"importance_combined_all_classes.png")
        correlation.plot_feature_importance(
            features=combined_features,
            labels=original_labels,
            data_type="combined",
            output_file=output_file,
            n_top_features=min(20, combined_features.shape[1])
        )
        
        # Generate feature importance plots for binary classification
        logger.info("Generating feature importance plots for binary classification")
        
        # 1. Storage features importance
        output_file = os.path.join(args.output_dir, f"importance_storage_binary_classes.png")
        correlation.plot_feature_importance(
            features=storage_features,
            labels=binary_labels,
            data_type="storage",
            output_file=output_file,
            n_top_features=min(20, storage_features.shape[1])
        )
        
        # 2. Memory features importance
        output_file = os.path.join(args.output_dir, f"importance_memory_binary_classes.png")
        correlation.plot_feature_importance(
            features=memory_features,
            labels=binary_labels,
            data_type="memory",
            output_file=output_file,
            n_top_features=min(20, memory_features.shape[1])
        )
        
        # 3. Combined features importance
        output_file = os.path.join(args.output_dir, f"importance_combined_binary_classes.png")
        correlation.plot_feature_importance(
            features=combined_features,
            labels=binary_labels,
            data_type="combined",
            output_file=output_file,
            n_top_features=min(20, combined_features.shape[1])
        )
        
        # Generate feature distribution plots for multiclass classification
        logger.info("Generating feature distribution plots for multiclass classification")
        
        # 1. Storage features distribution
        output_file = os.path.join(args.output_dir, f"distribution_storage_all_classes.png")
        correlation.plot_feature_distributions(
            features=storage_features,
            labels=original_labels,
            data_type="storage",
            output_file=output_file,
            n_features=min(10, storage_features.shape[1])
        )
        
        # 2. Memory features distribution
        output_file = os.path.join(args.output_dir, f"distribution_memory_all_classes.png")
        correlation.plot_feature_distributions(
            features=memory_features,
            labels=original_labels,
            data_type="memory",
            output_file=output_file,
            n_features=min(10, memory_features.shape[1])
        )
        
        # 3. Combined features distribution
        output_file = os.path.join(args.output_dir, f"distribution_combined_all_classes.png")
        correlation.plot_feature_distributions(
            features=combined_features,
            labels=original_labels,
            data_type="combined",
            output_file=output_file,
            n_features=min(10, combined_features.shape[1])
        )
        
        # Generate feature distribution plots for binary classification
        logger.info("Generating feature distribution plots for binary classification")
        
        # 1. Storage features distribution with binary classes
        output_file = os.path.join(args.output_dir, f"distribution_storage_binary_classes.png")
        correlation.plot_feature_distributions(
            features=storage_features,
            labels=binary_labels,
            data_type="storage",
            output_file=output_file,
            n_features=min(10, storage_features.shape[1])
        )
        
        # 2. Memory features distribution with binary classes
        output_file = os.path.join(args.output_dir, f"distribution_memory_binary_classes.png")
        correlation.plot_feature_distributions(
            features=memory_features,
            labels=binary_labels,
            data_type="memory",
            output_file=output_file,
            n_features=min(10, memory_features.shape[1])
        )
        
        # 3. Combined features distribution with binary classes
        output_file = os.path.join(args.output_dir, f"distribution_combined_binary_classes.png")
        correlation.plot_feature_distributions(
            features=combined_features,
            labels=binary_labels,
            data_type="combined",
            output_file=output_file,
            n_features=min(10, combined_features.shape[1])
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