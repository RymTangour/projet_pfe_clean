from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import PowerTransformer, RobustScaler
import umap
from src import get_logger
from src.visialization.run_visualization_scripts import parse_args
from visualization_feature_processor import DataLoader , EnhancedVisualizer
logger=get_logger()
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
            preprocessed_features, outlier_mask = self.preprocess_combined_features(features[:, :5], features[:, 5:])
        else:
            self.logger.warning(f"Unknowen data type: {data_type},Using storage preprocessing algorithm")
            preprocessed_features, outlier_mask=self.preprocess_storage_features(features)

        if outlier_mask is not None:
            labels=labels[outlier_mask]
            self.logger.info(f"After outlier removal: Features shape: {preprocessed_features.shape},"
                             f"Labels  length: {len(labels)}")
        
        if method=='tsne':
            if data_type=='memory':
                perplexity=1500
                learning_rate=200
                early_exaggeration=14
                n_iter=3000
            elif data_type=='combined':
                perplexity=30
                learning_rate=200
                early_exaggeration=14
                n_iter=3000
            else:
                perplexity=30
                learning_rate=200
                early_exaggeration=14
                n_iter=3000
            self.logger.info(f"Applying t-SNE for {data_type} data with perpexity={perplexity}, learning_rate={learning_rate},"
                            f" max_iter={n_iter}, early_exaggeration={early_exaggeration}")

            try:
                tsne=TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    learning_rate=learning_rate,
                    max_iter=n_iter,
                    n_iter_without_progress=300,
                    min_grad_norm=1e-07,
                    init='pca',
                    verbose=1,
                    random_state=self.random_state,
                    method='barnes_hut',
                    angle=0.5,
                    n_jobs=-1
                    
                )
                embedding = tsne.fit_transform(preprocessed_features)
                if np.isnan(embedding).any():
                    self.logger.error(f"NaN values in t-SNE embedding for {data_type} data, Replacing with zeros")
                    raise ValueError("NaN values in t-SNE embedding")
                    embedding=np.nan_to_num(embedding)

            except Exception as e:
                self.logger.error(f'Error in t-SNE for {data_type} data: {str(e)}')
                self.logger.info(f'Using PCA instead  of TSNE for {data_type} data')
                pca=PCA(n_components=2, random_state=self.random_state)
                embedding=pca.fit_transform(preprocessed_features)

        elif method=='umap':
            try:

                if data_type=='memory':
                    n_neighbors=150
                    min_dist=0.01
                    spread=1

                elif data_type=='combined':
                    n_neighbors=150
                    min_dist=0.01
                    spread=1
                else:
                    n_neighbors=150
                    min_dist=0.01
                    spread=1

                reducer=umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=self.random_state
                )
                embedding=reducer.fit_transform(preprocessed_features)

                if np.isnan(embedding).any():
                    self.logger.warning(f'NaN values in UMAP embedding for {data_type} data, Replacing with zeros.')
                    embedding=np.nan_to_num(embedding)

            except (ImportError, Exception) as e:
                error_type="ImportError" if isinstance(e, ImportError) else "Runtime error"
                self.logger.error(f"UMAP failed {error_type}  :{str(e)}" )
                self.logger.info(f'Falling back to PCA for {data_type} data')
                pca=PCA(n_components=2, random_state=self.random_state)
                embedding=pca.fit_transform(preprocessed_features)

        elif method=='pca':
            self.logger.info(f'Applying PCA for {data_type} data')
          
            try:
                pca=PCA(n_components=2, random_state=self.random_state)
                embedding=pca.fit_transform(preprocessed_features)
                if np.isnan(embedding).any():
                    self.logger.warning(f'NaN values in PCA embedding for {data_type} data, Replacing with zeros.')
                    embedding=np.nan_to_num(embedding)


            except Exception as e:
                self.logger.error(f'PCA failed for data type {data_type}: {str(e)}')
                
        else:
            self.logger.error(f'Unknowen method: {method}, Using PCA as default.')
            pca=PCA(n_components=2, random_state=self.random_state)
            embedding=pca.fit_transform(preprocessed_features)
            if np.isnan(embedding).any():
                self.logger.warning(f'NaN values in PCA embedding for {data_type} data, Replacing with zeros.')
                embedding=np.nan_to_num(embedding)


        
        embedding= self.enhance_separation_specialized(embedding, labels, data_type)

        df_viz=pd.DataFrame(embedding, columns=['Dim 1','Dim 2'])
        df_viz['Label']=labels

        plt.figure(figsize=(14, 12))
        if use_original_labels:
            unique_labels=np.unique(labels)
            self.logger.info(f"Program labels: {unique_labels}")
            self.logger.info(f"Plotting with {len(unique_labels)} unique program labels")
            cmap=plt.cm.tab20
            colors=[cmap(i) for i in np.linspace(0,1, len(unique_labels))]
            program_color_map={label: colors[i] for i, label in enumerate(unique_labels)}

            for i, label in enumerate(unique_labels):
                indices=df_viz['Label']==label
                alpha=0.8
                marker_size=100
                marker='o'
                color=program_color_map[label]
                plt.scatter(
                    df_viz.loc[indices, 'Dim 1'],
                    df_viz.loc[indices, 'Dim 2'],
                    color=color,
                    alpha=alpha,
                    edgecolors='w',
                    s=marker_size,
                    marker=marker,
                )
                if np.sum(indices)>0:
                    centroid_x=np.mean(df_viz.loc[indices, 'Dim 1'])
                    centroid_y=np.mean(df_viz.loc[indices, 'Dim 2'])
                    plt.text(
                        centroid_x, centroid_y,label,
                        frontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3')
                    )

        else:
            colors={'Benign': 'blue', 'Malicious': 'orange'}
            for label, color in colors.items():
                indices=df_viz['Label']==label
                if np.sum(indices)>0:
                    plt.scatter(
                        df_viz.loc[indices, 'Dim 1'],
                        df_viz.loc[indices, 'Dim 2'],
                        color=color,
                        alpha=0.8,
                        s=100,
                        marker='o',
                        edgecolors='w',
                        linewidth=0.5,
                        label=label
                    )

        method_name=method.upper()
        plt.title(f'{plot_title}({data_type.capitalize()} Data) ', fontsize=18, pad=20)
        plt.xlabel(f'{method_name}  Dimension 1', fontseize=14)
        plt.ylabel(f'{method_name}  Dimension 2', fontsize=14)
        if not use_original_labels or len(unique_labels)<=12:
            legend= plt.legend(title="Program Type" if not use_original_labels else "Progeam Name",
                               fontseize=12,markerscale=1.2, loc='best')
        
        if method=='tsne':
            if data_type=="memory":
                method_text=f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate},exaggeration={early_exaggeration}, max_iter={n_iter}"
            if data_type=="combined":
                method_text=f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate},exaggeration={early_exaggeration}, max_iter={n_iter}"
            else :
                method_text=f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate},exaggeration={early_exaggeration}, max_iter={n_iter}"
        if method=='umap':
            if data_type=='memory':
                method_text=f"UMAP: n_neighbors={n_neighbors}, spread={spread}, min_dist={min_dist}"
            if data_type=='combined':
                method_text=f"UMAP: n_neighbors={n_neighbors}, spread={spread}, min_dist={min_dist}"
            else:
                method_text=f"UMAP: n_neighbors={n_neighbors}, spread={spread}, min_dist={min_dist}"

        if method=='pca':
            method_text="PCA: 2 components"
        else:
            method_text=f"{method.upper()}: standard_parameters"
        plt.figtext(0.02,0.02, method_text, frontsize=10, ha='left', va='bottom')

        if output_file:
            try:
                plt.savefig(output_file, dpi=600, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save figuren:{str(e)}")
        else:
            plt.show()

        plt.close()
        return embedding, df_viz
                
    def plot_pca_explained_variance(self,features,data_type,output_file=None,n_components=None) :
        try: 
            features, labels= self.sample_data_if_needed(features, labels)
            self.logger.info(f"After sampling: Features shape: {features.shape}, labels length: {len(labels)}")

            if data_type=="storage":
                preprocessed_features, outlier_mask=self.preprocess_storage_features(features)
            elif data_type=="memory":
                preprocessed_features,outlier_mask=self.preprocess_memory_features(features)
            elif data_type=="combined":
                preprocessed_features, outlier_mask = self.preprocess_combined_features(features[:, :5], features[:, 5:])
            else:
                self.logger.warning(f"Unknowen data type: {data_type},Using storage preprocessing algorithm")
                preprocessed_features, outlier_mask=self.preprocess_storage_features(features)

            if outlier_mask is not None:
                labels=labels[outlier_mask]
                self.logger.info(f"After outlier removal: Features shape: {preprocessed_features.shape},"
                                f"Labels  length: {len(labels)}")
        
            if data_type=='memory':
                scaler=PowerTransformer(method='yeo-johnson')
                preprocessed_features=scaler.fit_transform(preprocessed_features)
            if data_type=='combined':
                scaler=StandardScaler()
                preprocessed_features=scaler.fit_transform(preprocessed_features)
            else:
                scaler=RobustScaler()
                preprocessed_features=scaler.fit_transform(preprocessed_features)  

            pca=PCA(n_components=None, random_state=self.random_state)
            pca_varaince=pca.fit_transform(preprocessed_features)
            explained_variance=pca.explained_variance_ratio_
            total_explained_variance=np.sum(explained_variance)
            n_components=preprocessed_features.shape[0]
            self.logger.info(f'Explained variance ratio for PCA: {explained_variance}') 
            self.logger.info(f'Total explained variance for PCA: {total_explained_variance}')
        except Exception as e:
            self.logger.error(f'PCA explained variance calculation failed for {data_type} data: {str(e)}')
            self.explained_variance_ratio=None
            self.total_explained_variance=None
        fig,ax=plt.subplot(2,1,  figsize=(12,10), gridspec_k={'height_ratios': [1,1.5]})
        ax[0].bar(
            range(1, n_components+1),
            explained_variance,
            alpha=0.7,
            color='steelblue'
        )
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Varaince Ratio')
        ax[0].set_title(f'Explained Variance per Principal Component ({data_type.capitalize()} Data)')
        ax[0].grid(True, linestyle='--', alpha=0.5)

        for i, v in enumerate(explained_variance):
            ax[0].text(i+1, v+0.01, f'{v:.2%}', ha='center', fontseize=8)
        cumulative_variance=np.cumsum(explained_variance)
        ax[1].plot(
            range(1,n_components+1),
            cumulative_variance,
            'o- ',
            linewidth=2,
            color='darkorange',
            label='Cumulative Explained Variance'
        )
        thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        threshold_components=[]
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
        ax[1].set_xticks(range(1, n_components + 1))
        ax[1].set_ylim([0, 1.05])
        ax[1].grid(True, linestyle='--', alpha=0.5)
        threshold_text = "\n".join([
            f"Components for {t:.0%} variance: {c}" for t, c in threshold_components
        ])

        plt.figtext(
            0.02,
            0.02,
            threshold_text,
            frontsize=10,
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white',  alpha=0.8,edgecolor='gray', boxstyle='round,pad=0.5')
        )
        plt.tight_layout()

        if output_file:
            try:
                plt.savefig(output_file)

            except Exception as e:
                self.logger.error(f"Failed to save figure: {str(e)}")
        else:
            plt.show()
            
        plt.close()
        return pca
    
    def visualize_all_specialized(self,data_dict,output_dir=None, method='tsne'):
        if output_dir is not None:
            output_dir=Path(output_dir)
            output_dir.mkdir(exist_ok=True,parents=True)

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
            for key1 in ["storage","memory","combined"]:
                for key2 in ["original_labels","binary_labels"]:
                    self.logger.info(f"Creating{method.upper()} visualization for {key1} features with {key2} labels")
                    self.create_specialized_visualization(
                        data_dict[key1],
                        data_dict[key2],
                        f"{method.upper()} of {key1} features, ({key2} Classification.)",
                        output_file=output_dir / f"{method}_{key1}_{key2}.png" if output_dir else None,
                        use_original_labels=True if key2=='original_labels' else False,
                        method=method
                    )
                self.logger.info("Creating PCA visualizations for {key1} features")
                self.plot_pca_explained_variance(
                    data_dict[key1],
                    key1,
                    output_file=output_dir / "pca_{key1}_variance.png" if output_dir else None

                    )


        
        except Exception as e:
            self.logger.error(f"Error in visualization process: {str(e)}")
        return None

def main():
    args = parse_args()
    
    try:
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = DataLoader(args.data_dir)
        data_dict = data_loader.load_data()
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
      
        visualizer.visualize_all_specialized(data_dict, args.output_dir, args.method) 
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
       
        



        

















