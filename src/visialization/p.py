from src import get_logger
from visualization_feature_processor import DataLoader , EnhancedVisualizer
class Cluster(EnhancedVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def create_specialized_visualization(self, features, labels, plot_title, data_type,output_file=None, use_original_labels=False, method='tsne'):
        if len(features)!=len(labels):
            self.logger.error(f"Mismatch lengths: features {len(features)}, labels {len((labels))}")
            raise ValueError("Features and labels must have the same length")
        
        initial_shape=features.shape
        self.logger.info(f"Initial features shape: {initial_shape}, labels length: {len(labels)}")

        features, labels= self.sample_data_if_needed(features, labels)
        self.logger.info(f"After sampling: Features shape: {features.shape}, labels length: {len(labels)}")

        if data_type=="storage":
            preprocessed_features, outlier_mask=self.preprocess_storage_features(features)
        elif data_type=="memory":
            preprocessed_features,outlier_mask=self.preprocess_memory_features(features)
        elif data_type=="combined":
            preprocessed_features, outlier_mask= self.preprocess_combined_features(features)
        else:
            self.logger.warning(f"Unknowen data type: {data_type},Using storage preprocessing algorithm")
            preprocessed_features, outlier_mask=self.preprocess_storage_features(features)

        if outlier_mask is not None:
            labels=labels[outlier_mask]
            self.logger.info(f"After outlier removal: Features shape: {preprocessed_features.shape},"
                             f"Labels  length: {len(labels)}")
        
        if method=='tsne':
            if data_type=='memory':
                perplexity=perplexity