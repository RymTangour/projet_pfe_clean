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


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger()

class FeatureProcessor:
    @staticmethod
    def extract_features(data, data_type):
        if data_type == "storage":
            return data
        elif data_type == "memory":
            return data
    
    @staticmethod
    def extract_combined_features(storage_data, memory_data):
        """Combine features from storage and memory data"""
        return np.hstack((
            FeatureProcessor.extract_features(storage_data, "storage"),
            FeatureProcessor.extract_features(memory_data, "memory")
        ))

class DataLoader:    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.malicious_programs = [
            'AESCrypt', 'Conti', 'Darkside', 'LockBit', 
            'REvil', 'Ryuk', 'WannaCry'
        ]
    
    def load_data(self):
        storage_data = []
        memory_data = []
        combined_data = []
        all_binary_labels = []
        all_original_labels = []
        batches = [d for d in self.base_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(batches)} batches")
        
        for batch_dir in batches:
            self.logger.info(f"Processing batch: {batch_dir.name}")
            
            # Find all RAM type directories
            ram_type_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
            
            for ram_type_dir in ram_type_dirs:
                ram_type = ram_type_dir.name
                self.logger.info(f"Processing RAM type: {ram_type}")
                
                # Find all program directories (these will be our class labels)
                program_dirs = [d for d in ram_type_dir.iterdir() if d.is_dir()]
                
                for program_dir in program_dirs:
                    program_name = program_dir.name
                    binary_label = "Malicious" if program_name in self.malicious_programs else "Benign"
                    
                    # Find all timestamp directories
                    timestamp_dirs = [d for d in program_dir.iterdir() if d.is_dir()]
                    self.logger.info(f"Found {len(timestamp_dirs)} timestamp directories for {program_name} ({binary_label})")
                    
                    processed_count = 0
                    for timestamp_dir in timestamp_dirs:
                        storage_file = timestamp_dir / "storage.npy"
                        memory_file = timestamp_dir / "memory.npy"
                        
                        if not storage_file.exists() or not memory_file.exists():
                            continue
                        
                        try:
                            # Load raw data
                            storage_features = np.load(storage_file)
                            memory_features = np.load(memory_file)
                            
                            # Check if we have data and shapes match
                            if storage_features.size == 0 or memory_features.size == 0:
                                self.logger.warning(f"Empty data for {timestamp_dir}")
                                continue
                                
                            if storage_features.shape[0] != memory_features.shape[0]:
                                self.logger.warning(f"Shape mismatch at {timestamp_dir}: storage {storage_features.shape} vs memory {memory_features.shape}")
                                continue
                            num_samples = storage_features.shape[0]
                            storage_only_features = FeatureProcessor.extract_features(storage_features, "storage")
                            memory_only_features = FeatureProcessor.extract_features(memory_features, "memory")
                            combined_features = FeatureProcessor.extract_combined_features(storage_features, memory_features)
                            storage_data.append(storage_only_features)
                            memory_data.append(memory_only_features)
                            combined_data.append(combined_features)
                            all_binary_labels.extend([binary_label] * num_samples)
                            all_original_labels.extend([program_name] * num_samples)
                            
                            processed_count += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error processing {timestamp_dir}: {str(e)}")
                    
                    self.logger.info(f"Processed {processed_count} samples for program: {program_name} ({binary_label})")
        
        if not storage_data or not memory_data or not combined_data:
            raise ValueError("No data was loaded or extracted")
        
        # Stack the data
        try:
            storage_array = np.vstack(storage_data)
            memory_array = np.vstack(memory_data)
            combined_array = np.vstack(combined_data)
        except ValueError as e:
            # If stacking fails, try to diagnose the issue
            shapes = [arr.shape for arr in storage_data]
            self.logger.error(f"Failed to stack arrays. Shapes: {shapes}")
            raise e
        
        # Convert labels to numpy arrays
        binary_labels_array = np.array(all_binary_labels)
        original_labels_array = np.array(all_original_labels)
        
        # Verify lengths match
        if (len(binary_labels_array) != storage_array.shape[0] or 
            len(original_labels_array) != storage_array.shape[0]):
            self.logger.error(f"Label length mismatch: binary labels {len(binary_labels_array)}, "
                             f"original labels {len(original_labels_array)}, data {storage_array.shape[0]}")
            raise ValueError("Label and data lengths do not match")
        
        # Log summary of binary classification    
        unique_binary_labels, binary_counts = np.unique(binary_labels_array, return_counts=True)
        self.logger.info(f"Binary classification summary:")
        for label, count in zip(unique_binary_labels, binary_counts):
            self.logger.info(f"  {label}: {count} samples")
        
        # Log summary of original labels
        unique_original_labels, original_counts = np.unique(original_labels_array, return_counts=True)
        self.logger.info(f"Original classification summary:")
        for label, count in zip(unique_original_labels, original_counts):
            self.logger.info(f"  {label}: {count} samples")
        
        self.logger.info(f"Final data shapes: Storage {storage_array.shape}, Memory {memory_array.shape}, "
                        f"Combined {combined_array.shape}, Labels {len(binary_labels_array)}")
        
        return {
            "storage": storage_array,
            "memory": memory_array,
            "combined": combined_array,
            "binary_labels": binary_labels_array,
            "original_labels": original_labels_array
        }

class EnhancedVisualizer:
    
    def __init__(self, perplexity=40, learning_rate=200, max_iter=2500, random_state=42,outlier_method='iqr', iqr_factor=2.0, zscore_threshold=5.0, contamination=0.1):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.outlier_method = outlier_method
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
            # Define a colormap for program-specific plots
        self.program_cmap = plt.colormaps['tab20']
        
        # Define ransomware programs for special coloring
        self.ransomware_programs = ['Conti', 'Darkside', 'LockBit', 'REvil', 'Ryuk', 'WannaCry']
    def remove_outliers(self,features, method='iqr', **kwargs):
    
        logger = logging.getLogger(__name__)
        logger.info(f"Checking for outliers using {method} method")
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            logger.warning("Data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if method == 'zscore':
            zscore_threshold = kwargs.get('zscore_threshold', 5.0)  # Changed from 3.0 to 5.0
            logger.info(f"Using Z-score method with threshold: {zscore_threshold}")
           
            medians = np.median(features, axis=0)
            mad = np.median(np.abs(features - medians), axis=0)
            mad = np.where(mad == 0, 1e-8, mad)  # Avoid division by zero
            
            # Calculate modified z-scores (more robust than standard z-scores)
            modified_z_scores = 0.6745 * np.abs(features - medians) / mad
            
            # Find samples where all features have reasonable z-scores
            non_outlier_mask = np.all(modified_z_scores < zscore_threshold, axis=1)
            
        elif method == 'iqr':
            # Get factor with more lenient default for extreme data patterns
            iqr_factor = kwargs.get('iqr_factor', 2.0)  # Changed from default 1.5 to 2.0
            logger.info(f"Using IQR method with factor: {iqr_factor}")
            
            # Apply feature-wise outlier detection
            non_outlier_mask = np.ones(features.shape[0], dtype=bool)
            
            # Apply IQR to each feature separately - this works better with mixed scale data
            for i in range(features.shape[1]):
                feature = features[:, i]
                q1 = np.percentile(feature, 25)
                q3 = np.percentile(feature, 75)
                iqr = q3 - q1
                
                # Skip features with zero IQR (constant features)
                if iqr == 0:
                    continue
                    
                lower_bound = q1 - (iqr_factor * iqr)
                upper_bound = q3 + (iqr_factor * iqr)
                
                # Update mask - only mark as outlier if multiple features indicate it
                feature_mask = (feature >= lower_bound) & (feature <= upper_bound)
                non_outlier_mask = non_outlier_mask & feature_mask
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            # Get contamination param with default appropriate for this data
            contamination = kwargs.get('contamination', 0.1)  # Expect 10% outliers
            logger.info(f"Using Isolation Forest with contamination: {contamination}")
            
            # For high-dimensional data, reduce dimensions first
            if features.shape[1] > 20:
                from sklearn.decomposition import PCA
                logger.info("Applying PCA before Isolation Forest due to high dimensionality")
                pca = PCA(n_components=min(20, features.shape[0], features.shape[1]))
                features_reduced = pca.fit_transform(features)
            else:
                features_reduced = features
            
            # Train isolation forest
            isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200  # More estimators for better accuracy
            )
            
            # Predict outliers (-1 for outliers, 1 for inliers)
            predictions = isolation_forest.fit_predict(features_reduced)
            non_outlier_mask = predictions == 1
            
        else:
            logger.warning(f"Unknown outlier detection method: {method}. Using IQR.")
            # Default to IQR for unknown methods
            iqr_factor = kwargs.get('iqr_factor', 2.0)
            
            non_outlier_mask = np.ones(features.shape[0], dtype=bool)
            for i in range(features.shape[1]):
                feature = features[:, i]
                q1 = np.percentile(feature, 25)
                q3 = np.percentile(feature, 75)
                iqr = q3 - q1
                
                if iqr == 0:
                    continue
                    
                lower_bound = q1 - (iqr_factor * iqr)
                upper_bound = q3 + (iqr_factor * iqr)
                
                feature_mask = (feature >= lower_bound) & (feature <= upper_bound)
                non_outlier_mask = non_outlier_mask & feature_mask
        
        logger.info(f"Identified {features.shape[0] - np.sum(non_outlier_mask)} outliers out of {features.shape[0]} samples")
        
        return features[non_outlier_mask], non_outlier_mask
   

    def preprocess_storage_features(self,features):
   
        logger = logging.getLogger(__name__)
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            logger.warning("Data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape if needed
        if len(features.shape) > 2:
            logger.info(f"Reshaping features from {features.shape} to 2D.")
            features = features.reshape(features.shape[0], -1)
        features, outlier_mask = self.remove_outliers(features, method=self.outlier_method, 
                                                 iqr_factor=self.iqr_factor, 
                                                 zscore_threshold=self.zscore_threshold,
                                                 contamination=self.contamination)
        logger.info("Applying log transformation to handle extreme value ranges")
        epsilon = 1e-10  # Small constant to avoid log(0)
        log_features = np.log1p(np.abs(features) + epsilon)
        

        logger.info("Applying RobustScaler for storage features")
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(log_features)
        
        pca_components = min(23, features.shape[1])  # Reduced from 50 based on data patterns
        logger.info(f"Applying PCA to reduce storage dimensions to {pca_components}")
        pca = PCA(n_components=pca_components, random_state=self.random_state)
        reduced_features = pca.fit_transform(scaled_features)
        explained_var = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_var:.2f}")
        
        return reduced_features, outlier_mask

    def preprocess_memory_features(self,features):
       
        logger = logging.getLogger(__name__)
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            logger.warning("Data contains NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape if needed
        if len(features.shape) > 2:
            logger.info(f"Reshaping features from {features.shape} to 2D.")
            features = features.reshape(features.shape[0], -1)
        feature_means = np.mean(np.abs(features), axis=0)
        large_value_cols = np.where(feature_means > 1e10)[0]
        small_value_cols = np.where(feature_means <= 1e10)[0]
        
        logger.info(f"Identified {len(large_value_cols)} columns with extremely large values")
        features, outlier_mask = self.remove_outliers(features, method=self.outlier_method, 
                                                 iqr_factor=self.iqr_factor, 
                                                 zscore_threshold=self.zscore_threshold,
                                                 contamination=self.contamination)
        # Process large value columns with log transformation and standardization
        if len(large_value_cols) > 0:
            large_features = features[:, large_value_cols]
            epsilon = 1e-10
            large_features_log = np.log1p(np.abs(large_features) + epsilon)
            scaler_large = StandardScaler()
            large_features_scaled = scaler_large.fit_transform(large_features_log)
        else:
            large_features_scaled = np.array([]).reshape(features.shape[0], 0)
        
        # Process small value columns with MinMaxScaler as in original code
        if len(small_value_cols) > 0:
            small_features = features[:, small_value_cols]
            scaler_small = PowerTransformer(method='yeo-johnson')
            small_features_scaled = scaler_small.fit_transform(small_features)
        else:
            small_features_scaled = np.array([]).reshape(features.shape[0], 0)
        
        # Recombine the differently scaled features
        scaled_features = np.hstack((small_features_scaled, large_features_scaled))
        variances = np.var(scaled_features, axis=0)
        # Keep features with variance above 75th percentile (more aggressive than original)
        high_var_indices = np.where(variances > np.percentile(variances, 75))[0]
        logger.info(f"Keeping {len(high_var_indices)} high-variance features out of {scaled_features.shape[1]}")
        
        if len(high_var_indices) > 2:
            filtered_features = scaled_features[:, high_var_indices]
        else:
            filtered_features = scaled_features
        
        # For memory data, use fewer PCA components based on the observed patterns
        if filtered_features.shape[1] > 10:  # Reduced from 20 based on data patterns
            pca_components = min(23, filtered_features.shape[0], filtered_features.shape[1])
            logger.info(f"Applying PCA with {pca_components} components for memory features")
            pca = PCA(n_components=pca_components, random_state=self.random_state)
            reduced_features = pca.fit_transform(filtered_features)
            explained_var = np.sum(pca.explained_variance_ratio_)
            logger.info(f"Memory PCA explained variance: {explained_var:.2f}")
        else:
            reduced_features = filtered_features
        
        return reduced_features , outlier_mask

    def preprocess_combined_features(self, storage_features, memory_features):
        """
        Preprocess and combine storage and memory features with consistent outlier detection.
        
        Args:
            storage_features (numpy.ndarray): Raw storage features
            memory_features (numpy.ndarray): Raw memory features
            
        Returns:
            tuple: (combined_reduced_features, combined_mask) - PCA-reduced combined features and outlier mask
        """
        logger = logging.getLogger(__name__)
        
        # Check for NaN or Inf values in both feature sets
        if np.isnan(storage_features).any() or np.isinf(storage_features).any():
            logger.warning("Storage data contains NaN or Inf values. Replacing with zeros.")
            storage_features = np.nan_to_num(storage_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(memory_features).any() or np.isinf(memory_features).any():
            logger.warning("Memory data contains NaN or Inf values. Replacing with zeros.")
            memory_features = np.nan_to_num(memory_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape if needed
        if len(storage_features.shape) > 2:
            logger.info(f"Reshaping storage features from {storage_features.shape} to 2D.")
            storage_features = storage_features.reshape(storage_features.shape[0], -1)
        
        if len(memory_features.shape) > 2:
            logger.info(f"Reshaping memory features from {memory_features.shape} to 2D.")
            memory_features = memory_features.reshape(memory_features.shape[0], -1)
        
        # First identify outliers in both datasets separately
        _, outlier_mask_storage = self.remove_outliers(storage_features, method=self.outlier_method, 
                                                iqr_factor=self.iqr_factor, 
                                                zscore_threshold=self.zscore_threshold,
                                                contamination=self.contamination)
        
        _, outlier_mask_memory = self.remove_outliers(memory_features, method=self.outlier_method, 
                                                iqr_factor=self.iqr_factor, 
                                                zscore_threshold=self.zscore_threshold,
                                                contamination=self.contamination)
        
        # Combine the masks - keep only samples that are not outliers in BOTH datasets
        combined_mask = outlier_mask_storage & outlier_mask_memory
        logger.info(f"Combined outlier detection: keeping {np.sum(combined_mask)} out of {len(combined_mask)} samples")
        
        # Apply the combined mask to both feature sets
        storage_filtered = storage_features[combined_mask]
        memory_filtered = memory_features[combined_mask]
        
        # Now preprocess each filtered dataset
        # Process storage features
        logger.info("Preprocessing filtered storage features")
        epsilon = 1e-10  # Small constant to avoid log(0)
        log_storage = np.log1p(np.abs(storage_filtered) + epsilon)
        
        scaler_storage = RobustScaler()
        scaled_storage = scaler_storage.fit_transform(log_storage)
        
        storage_pca_components = min(23, storage_filtered.shape[1])
        logger.info(f"Applying PCA to reduce storage dimensions to {storage_pca_components}")
        pca_storage = PCA(n_components=storage_pca_components, random_state=self.random_state)
        reduced_storage = pca_storage.fit_transform(scaled_storage)
        storage_explained_var = np.sum(pca_storage.explained_variance_ratio_)
        logger.info(f"Storage PCA explained variance: {storage_explained_var:.2f}")
        
        # Process memory features
        logger.info("Preprocessing filtered memory features")
        feature_means = np.mean(np.abs(memory_filtered), axis=0)
        large_value_cols = np.where(feature_means > 1e10)[0]
        small_value_cols = np.where(feature_means <= 1e10)[0]
        
        logger.info(f"Identified {len(large_value_cols)} memory columns with extremely large values")
        
        # Process large value columns with log transformation and standardization
        if len(large_value_cols) > 0:
            large_features = memory_filtered[:, large_value_cols]
            large_features_log = np.log1p(np.abs(large_features) + epsilon)
            scaler_large = StandardScaler()
            large_features_scaled = scaler_large.fit_transform(large_features_log)
        else:
            large_features_scaled = np.array([]).reshape(memory_filtered.shape[0], 0)
        
        # Process small value columns with PowerTransformer
        if len(small_value_cols) > 0:
            small_features = memory_filtered[:, small_value_cols]
            scaler_small = RobustScaler()
            small_features_scaled = scaler_small.fit_transform(small_features)
        else:
            small_features_scaled = np.array([]).reshape(memory_filtered.shape[0], 0)
        
        # Recombine the differently scaled memory features
        scaled_memory = np.hstack((small_features_scaled, large_features_scaled))
        
        # Filter by variance
        variances = np.var(scaled_memory, axis=0)
        high_var_indices = np.where(variances > np.percentile(variances, 75))[0]
        logger.info(f"Keeping {len(high_var_indices)} high-variance memory features out of {scaled_memory.shape[1]}")
        
        if len(high_var_indices) > 2:
            filtered_memory = scaled_memory[:, high_var_indices]
        else:
            filtered_memory = scaled_memory
        
        # Apply PCA to memory features if needed
        if filtered_memory.shape[1] > 10:
            memory_pca_components = min(23, filtered_memory.shape[0], filtered_memory.shape[1])
            logger.info(f"Applying PCA with {memory_pca_components} components for memory features")
            pca_memory = PCA(n_components=memory_pca_components, random_state=self.random_state)
            reduced_memory = pca_memory.fit_transform(filtered_memory)
            memory_explained_var = np.sum(pca_memory.explained_variance_ratio_)
            logger.info(f"Memory PCA explained variance: {memory_explained_var:.2f}")
        else:
            reduced_memory = filtered_memory
        
        # Log the shapes of processed features
        logger.info(f"Processed storage features shape: {reduced_storage.shape}")
        logger.info(f"Processed memory features shape: {reduced_memory.shape}")
        
        # Standardize both feature sets to ensure equal contribution before combining
        scaler_final_storage = StandardScaler()
        scaler_final_memory = StandardScaler()
        
        storage_std = scaler_final_storage.fit_transform(reduced_storage)
        memory_std = scaler_final_memory.fit_transform(reduced_memory)
        
        # Combine the standardized features
        combined_features = np.hstack((storage_std, memory_std))
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        # Apply final PCA to the combined features
        final_pca_components = min(23, combined_features.shape[0], combined_features.shape[1])
        logger.info(f"Applying PCA with {final_pca_components} components for combined features")
        final_pca = PCA(n_components=final_pca_components, random_state=self.random_state)
        final_reduced_features = final_pca.fit_transform(combined_features)
        combined_explained_var = np.sum(final_pca.explained_variance_ratio_)
        logger.info(f"Combined PCA explained variance: {combined_explained_var:.2f}")
        
        return final_reduced_features, combined_mask
        
    def enhance_separation_specialized(self, embedding, labels, data_type):
        """Enhanced cluster separation customized by data type"""
        # Critical validation: ensure embedding and labels have matching lengths
        if len(embedding) != len(labels):
            self.logger.error(f"Length mismatch in enhance_separation: embedding {len(embedding)} vs labels {len(labels)}")
            return embedding
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if data_type == "storage":
            # For storage: slight adjustment toward cluster centers (original method)
            adjustment_strength = 0.2
        elif data_type == "memory":
            # For memory: stronger adjustment toward cluster centers
            adjustment_strength = 0.2
        elif data_type == "combined":
            # For combined: moderate adjustment
            adjustment_strength = 0.2
        else:
            adjustment_strength = 0.2
            
        self.logger.info(f"Enhancing separation for {data_type} data with strength {adjustment_strength}")
        
        # Map labels to numeric values
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_idx[label] for label in labels])
        
        # Initialize and fit kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans.fit(embedding)
        centers = kmeans.cluster_centers_
        
        # Adjust points to move toward their true class center
        enhanced_embedding = embedding.copy()
        
        # Group points by label for per-cluster adjustments
        for label_idx, label in enumerate(unique_labels):
            label_mask = numeric_labels == label_idx
            if np.sum(label_mask) > 0:
                # Calculate the centroid of points with this label
                true_centroid = np.mean(enhanced_embedding[label_mask], axis=0)
                
                # For combined and memory data, apply a more aggressive adjustment
                if data_type in ["memory", "combined"]:
                    # Calculate distance to centroid for each point in this cluster
                    for i in np.where(label_mask)[0]:
                        vector_to_center = true_centroid - enhanced_embedding[i]
                        dist_from_center = np.linalg.norm(vector_to_center)
                        
                        # Adjust strength based on distance - points far from center get larger adjustments
                        if data_type == "memory":
                            # Stronger correction for outliers
                            point_strength = adjustment_strength * (1 + dist_from_center / 5)
                            enhanced_embedding[i] = enhanced_embedding[i] + point_strength * vector_to_center
                        else:  # combined
                            enhanced_embedding[i] = enhanced_embedding[i] + adjustment_strength * vector_to_center
                else:  # storage - use original approach
                    for i in np.where(label_mask)[0]:
                        vector_to_center = centers[label_idx] - enhanced_embedding[i]
                        enhanced_embedding[i] = enhanced_embedding[i] + adjustment_strength * vector_to_center
        
        # For memory data, add a final step: increase spread between clusters
        if data_type == "memory":
            # Calculate centroids for each cluster after adjustments
            cluster_centroids = []
            for label_idx in range(n_clusters):
                label_mask = numeric_labels == label_idx
                if np.sum(label_mask) > 0:
                    centroid = np.mean(enhanced_embedding[label_mask], axis=0)
                    cluster_centroids.append(centroid)
            
            cluster_centroids = np.array(cluster_centroids)
            
            # Calculate center of all centroids
            if len(cluster_centroids) > 0:
                center_of_mass = np.mean(cluster_centroids, axis=0)
                
                # Move each cluster away from center
                spread_factor = 1.3  # How much to spread clusters
                for label_idx in range(n_clusters):
                    label_mask = numeric_labels == label_idx
                    if np.sum(label_mask) > 0:
                        # Vector from center of mass to this cluster's centroid
                        cluster_centroid = np.mean(enhanced_embedding[label_mask], axis=0)
                        direction = cluster_centroid - center_of_mass
                        
                        # Normalize the direction vector if it's not zero
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 1e-10:  # Avoid division by very small numbers
                            direction = direction / direction_norm
                            
                            # Move all points in this cluster away from center
                            enhanced_embedding[label_mask] += spread_factor * direction
        
        return enhanced_embedding
    
    def sample_data_if_needed(self, features, labels, max_samples=100000):
        
        """Sample data if it exceeds max_samples to improve visualization performance"""
        if features.shape[0] > max_samples:
            self.logger.info(f"Sampling {max_samples} points from {features.shape[0]} for visualization")
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            return features[indices], labels[indices]
        return features, labels
    

