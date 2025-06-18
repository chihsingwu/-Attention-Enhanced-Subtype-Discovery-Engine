# Attention-Enhanced Subtype Discovery Engine
# A novel pipeline combining PCA, cosine similarity attention, and survival analysis
# for cancer subtype identification and prognostic biomarker discovery

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, adjusted_rand_score
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

class AttentionEnhancedEngine:
    """
    Attention-Enhanced Subtype Discovery Engine for cancer genomics analysis.
    
    This engine implements a novel pipeline that combines:
    - PCA dimensionality reduction
    - Cosine similarity attention mechanism
    - K-means clustering for subtype discovery
    - Feature ranking and noise reduction
    - Comprehensive survival analysis
    
    The attention mechanism creates patient similarity networks to identify
    cohesive subgroups and outlier patterns in high-dimensional genomic data.
    """
    
    def __init__(self, n_components=10, n_clusters=3, attention_temperature=1.0):
        """
        Initialize the Attention-Enhanced Subtype Discovery Engine.
        
        Parameters:
        -----------
        n_components : int, default=10
            Number of principal components for PCA reduction
        n_clusters : int, default=3
            Number of clusters for subtype discovery
        attention_temperature : float, default=1.0
            Temperature parameter for attention weight calculation
        """
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.attention_temperature = attention_temperature
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.results_ = {}

    def preprocess_data(self, data_df):
        """
        Standardize gene expression data for downstream analysis.
        
        Parameters:
        -----------
        data_df : DataFrame
            Gene expression matrix (samples × genes)
            
        Returns:
        --------
        X_scaled : array-like
            Standardized expression data
        """
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data_df)
        return X_scaled

    def compute_pca_attention(self, X_scaled):
        """
        Perform PCA reduction and compute cosine similarity attention matrix.
        
        The attention mechanism creates a patient similarity network where
        high attention weights indicate similar expression patterns.
        
        Parameters:
        -----------
        X_scaled : array-like
            Standardized expression data
            
        Returns:
        --------
        X_reduced : array-like
            PCA-transformed data
        attention_matrix : array-like
            Cosine similarity attention weights between samples
        """
        # PCA dimensionality reduction
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Compute cosine similarity attention matrix
        cosine_sim = cosine_similarity(X_reduced)
        
        # Apply temperature scaling for attention focus control
        attention_matrix = np.exp(cosine_sim / self.attention_temperature)
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        return X_reduced, attention_matrix

    def discover_subtypes(self, X_reduced):
        """
        Perform K-means clustering to identify cancer subtypes.
        
        Parameters:
        -----------
        X_reduced : array-like
            PCA-transformed data
            
        Returns:
        --------
        cluster_labels : array-like
            Subtype assignments for each sample
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_reduced)
        return cluster_labels

    def rank_features(self, data_df, cluster_labels):
        """
        Rank genes by inter-cluster variance for biomarker identification.
        
        Parameters:
        -----------
        data_df : DataFrame
            Original gene expression data
        cluster_labels : array-like
            Cluster assignments
            
        Returns:
        --------
        top_genes : list
            Top-ranked genes based on inter-cluster variance
        ranked_scores : Series
            Complete ranking scores for all genes
        """
        # Calculate inter-cluster variance for each gene
        cluster_df = data_df.copy()
        cluster_df['cluster'] = cluster_labels
        
        # Compute variance between cluster means
        cluster_means = cluster_df.groupby('cluster').mean()
        inter_cluster_var = cluster_means.var(axis=0)
        
        # Rank genes by variance
        ranked_scores = inter_cluster_var.sort_values(ascending=False)
        top_genes = ranked_scores.head(20).index.tolist()
        
        return top_genes, ranked_scores

    def survival_analysis(self, survival_df, cluster_labels):
        """
        Perform comprehensive survival analysis comparing subtypes.
        
        Parameters:
        -----------
        survival_df : DataFrame
            Survival data with 'time' and 'event' columns
        cluster_labels : array-like
            Subtype assignments
            
        Returns:
        --------
        dict : Survival analysis results including Cox model and statistics
        """
        df = survival_df.copy()
        df['cluster'] = cluster_labels
        
        # Cox proportional hazards model
        cox_model = CoxPHFitter()
        cox_model.fit(df, duration_col='time', event_col='event', formula='cluster')
        
        # Log-rank test for survival differences
        groups = []
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df[df['cluster'] == cluster_id]
            groups.append((cluster_data['time'], cluster_data['event']))
        
        if len(groups) >= 2:
            logrank_result = logrank_test(*groups[0], *groups[1])
            survival_stats = {
                'p_value': logrank_result.p_value,
                'test_statistic': logrank_result.test_statistic,
                'significant': logrank_result.p_value < 0.05
            }
        else:
            survival_stats = {'error': 'Need at least 2 clusters for comparison'}
        
        return {
            'cox_model': cox_model,
            'logrank_stats': survival_stats
        }

    def validate_clustering(self, X_reduced, cluster_labels):
        """
        Validate clustering quality using multiple metrics.
        
        Parameters:
        -----------
        X_reduced : array-like
            PCA-transformed data
        cluster_labels : array-like
            Cluster assignments
            
        Returns:
        --------
        dict : Validation metrics
        """
        # Silhouette score for cluster quality
        silhouette_avg = silhouette_score(X_reduced, cluster_labels)
        
        # Stability assessment through multiple runs
        stability_scores = []
        for _ in range(10):
            kmeans_temp = KMeans(n_clusters=self.n_clusters, random_state=None, n_init=10)
            temp_labels = kmeans_temp.fit_predict(X_reduced)
            stability_scores.append(adjusted_rand_score(cluster_labels, temp_labels))
        
        mean_stability = np.mean(stability_scores)
        
        return {
            'silhouette_score': round(silhouette_avg, 3),
            'mean_stability': round(mean_stability, 3),
            'cluster_quality': 'Excellent' if silhouette_avg > 0.7 else
                              'Good' if silhouette_avg > 0.5 else
                              'Fair' if silhouette_avg > 0.3 else 'Poor',
            'stability_assessment': 'Stable' if mean_stability > 0.7 else 'Moderate'
        }

    def plot_abalone_yangzhou_visualization(self, X_reduced, attention_matrix, cluster_labels, 
                                          survival_df=None, top_genes=None):
        """
        Create comprehensive visualization suite (Abalone Yangzhou Style).
        
        The 'Abalone Yangzhou' visualization includes:
        - PCA scatter plot with subtype coloring
        - Attention heatmap showing patient similarity network
        - Survival curves by subtype
        - Top biomarker expression patterns
        
        Parameters:
        -----------
        X_reduced : array-like
            PCA-transformed data
        attention_matrix : array-like
            Attention weights matrix
        cluster_labels : array-like
            Subtype assignments
        survival_df : DataFrame, optional
            Survival data for Kaplan-Meier plots
        top_genes : list, optional
            Top biomarker genes for expression visualization
        """
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Color palette for subtypes
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_clusters))
        
        # 1. PCA Scatter Plot with Subtype Coloring
        ax1 = fig.add_subplot(gs[0, :2])
        for i, color in zip(np.unique(cluster_labels), colors):
            mask = cluster_labels == i
            ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[color], label=f'Subtype {i}', alpha=0.7, s=60)
        
        explained_var = self.pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax1.set_title('Cancer Subtype Discovery in PCA Space', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Attention Heatmap (Patient Similarity Network)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Sort samples by cluster for better visualization
        sort_idx = np.argsort(cluster_labels)
        sorted_attention = attention_matrix[sort_idx][:, sort_idx]
        
        im = ax2.imshow(sorted_attention, cmap='YlOrRd', aspect='auto')
        ax2.set_title('Cosine Similarity Attention Matrix\n(Patient Similarity Network)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Patient Index (sorted by subtype)')
        ax2.set_ylabel('Patient Index (sorted by subtype)')
        
        # Add cluster boundaries
        cluster_counts = np.bincount(cluster_labels)
        boundaries = np.cumsum(cluster_counts)[:-1] - 0.5
        for boundary in boundaries:
            ax2.axhline(boundary, color='white', linewidth=2)
            ax2.axvline(boundary, color='white', linewidth=2)
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Survival Curves by Subtype
        if survival_df is not None:
            ax3 = fig.add_subplot(gs[1, :2])
            df_with_clusters = survival_df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            for cluster_id, color in zip(np.unique(cluster_labels), colors):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                
                kmf = KaplanMeierFitter()
                kmf.fit(cluster_data['time'], 
                       event_observed=cluster_data['event'],
                       label=f'Subtype {cluster_id} (n={len(cluster_data)})')
                
                kmf.plot_survival_function(ax=ax3, color=color, linewidth=3)
            
            ax3.set_title('Kaplan-Meier Survival Curves by Cancer Subtype', 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Survival Probability')
            ax3.grid(True, alpha=0.3)
        
        # 4. Top Biomarker Expression Heatmap
        if top_genes is not None and hasattr(self, '_original_data'):
            ax4 = fig.add_subplot(gs[1, 2:])
            
            # Get expression data for top genes
            top_gene_data = self._original_data[top_genes[:10]].T  # Top 10 genes
            
            # Sort by cluster
            sorted_data = top_gene_data.iloc[:, sort_idx]
            
            sns.heatmap(sorted_data, ax=ax4, cmap='RdBu_r', center=0, 
                       cbar_kws={'shrink': 0.8}, yticklabels=True, xticklabels=False)
            ax4.set_title('Top 10 Biomarker Expression Patterns', 
                         fontsize=14, fontweight='bold')
            ax4.set_xlabel('Patients (sorted by subtype)')
            ax4.set_ylabel('Biomarker Genes')
        
        # 5. Feature Importance Bar Plot
        if hasattr(self, '_ranked_scores'):
            ax5 = fig.add_subplot(gs[2, :2])
            top_scores = self._ranked_scores.head(15)
            
            bars = ax5.bar(range(len(top_scores)), top_scores.values, 
                          color='steelblue', alpha=0.7)
            ax5.set_title('Top 15 Biomarker Genes by Inter-Cluster Variance', 
                         fontsize=14, fontweight='bold')
            ax5.set_xlabel('Gene Rank')
            ax5.set_ylabel('Inter-Cluster Variance Score')
            ax5.set_xticks(range(len(top_scores)))
            ax5.set_xticklabels(top_scores.index, rotation=45, ha='right')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Cluster Statistics Summary
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # Create summary statistics table
        cluster_sizes = np.bincount(cluster_labels)
        summary_text = "📊 ABALONE YANGZHOU ANALYSIS SUMMARY\n\n"
        summary_text += f"🔬 Total Samples: {len(cluster_labels)}\n"
        summary_text += f"🧬 Principal Components: {self.n_components}\n"
        summary_text += f"🎯 Discovered Subtypes: {self.n_clusters}\n\n"
        
        for i, size in enumerate(cluster_sizes):
            percentage = (size / len(cluster_labels)) * 100
            summary_text += f"Subtype {i}: {size} patients ({percentage:.1f}%)\n"
        
        if hasattr(self, '_validation_results'):
            val_results = self._validation_results
            summary_text += f"\n📈 Clustering Quality: {val_results['cluster_quality']}\n"
            summary_text += f"🎲 Silhouette Score: {val_results['silhouette_score']}\n"
            summary_text += f"🔄 Stability: {val_results['stability_assessment']}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('🦪 ABALONE YANGZHOU FRIED RICE 🍛\nAttention-Enhanced Cancer Subtype Discovery', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

    def fit_predict(self, data_df, survival_df=None):
        """
        Execute the complete Attention-Enhanced Subtype Discovery pipeline.
        
        Parameters:
        -----------
        data_df : DataFrame
            Gene expression matrix (samples × genes)
        survival_df : DataFrame, optional
            Survival data with 'time' and 'event' columns
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print("🔬 Starting Attention-Enhanced Subtype Discovery...")
        
        # Store original data for visualization
        self._original_data = data_df.copy()
        
        # 1. Data preprocessing
        print("📊 Preprocessing gene expression data...")
        X_scaled = self.preprocess_data(data_df)
        
        # 2. PCA + Attention mechanism
        print("🧠 Computing PCA and attention matrix...")
        X_reduced, attention_matrix = self.compute_pca_attention(X_scaled)
        
        # 3. Subtype discovery
        print("🎯 Discovering cancer subtypes...")
        cluster_labels = self.discover_subtypes(X_reduced)
        
        # 4. Feature ranking
        print("🔍 Ranking biomarker genes...")
        top_genes, ranked_scores = self.rank_features(data_df, cluster_labels)
        self._ranked_scores = ranked_scores
        
        # 5. Clustering validation
        print("✅ Validating clustering quality...")
        validation_results = self.validate_clustering(X_reduced, cluster_labels)
        self._validation_results = validation_results
        
        # 6. Survival analysis (if provided)
        survival_results = None
        if survival_df is not None:
            print("📈 Performing survival analysis...")
            survival_results = self.survival_analysis(survival_df, cluster_labels)
        
        # Store results
        self.results_ = {
            'embedding': X_reduced,
            'attention_matrix': attention_matrix,
            'cluster_labels': cluster_labels,
            'top_genes': top_genes,
            'gene_rankings': ranked_scores,
            'validation': validation_results,
            'survival_analysis': survival_results,
            'explained_variance': self.pca.explained_variance_ratio_
        }
        
        print("🎉 Analysis completed successfully!")
        
        # Generate Abalone Yangzhou visualization
        print("🦪 Generating Abalone Yangzhou visualization...")
        self.plot_abalone_yangzhou_visualization(
            X_reduced, attention_matrix, cluster_labels, 
            survival_df, top_genes
        )
        
        return self.results_

    def get_summary_report(self):
        """
        Generate a comprehensive summary report of the analysis.
        
        Returns:
        --------
        str : Formatted summary report
        """
        if not self.results_:
            return "No analysis results available. Please run fit_predict() first."
        
        results = self.results_
        
        report = """
🦪 ABALONE YANGZHOU FRIED RICE - ANALYSIS REPORT 🍛
====================================================

🔬 METHODOLOGY OVERVIEW:
- PCA dimensionality reduction with attention mechanism
- Cosine similarity patient network construction
- K-means clustering for subtype discovery
- Inter-cluster variance biomarker ranking
- Comprehensive survival and validation analysis

📊 ANALYSIS RESULTS:
"""
        
        # Basic statistics
        n_samples = len(results['cluster_labels'])
        n_subtypes = len(np.unique(results['cluster_labels']))
        
        report += f"• Total samples analyzed: {n_samples}\n"
        report += f"• Cancer subtypes identified: {n_subtypes}\n"
        report += f"• Principal components used: {self.n_components}\n"
        report += f"• Explained variance (PC1+PC2): {sum(results['explained_variance'][:2]):.1%}\n\n"
        
        # Subtype distribution
        report += "🎯 SUBTYPE DISTRIBUTION:\n"
        cluster_counts = np.bincount(results['cluster_labels'])
        for i, count in enumerate(cluster_counts):
            percentage = (count / n_samples) * 100
            report += f"• Subtype {i}: {count} patients ({percentage:.1f}%)\n"
        report += "\n"
        
        # Top biomarkers
        report += "🧬 TOP BIOMARKER GENES:\n"
        for i, gene in enumerate(results['top_genes'][:10], 1):
            score = results['gene_rankings'][gene]
            report += f"{i:2d}. {gene:12s} (score: {score:.3f})\n"
        report += "\n"
        
        # Validation results
        val_results = results['validation']
        report += "✅ CLUSTERING VALIDATION:\n"
        report += f"• Silhouette score: {val_results['silhouette_score']}\n"
        report += f"• Cluster quality: {val_results['cluster_quality']}\n"
        report += f"• Stability assessment: {val_results['stability_assessment']}\n"
        report += f"• Mean stability score: {val_results['mean_stability']}\n\n"
        
        # Survival analysis
        if results['survival_analysis']:
            surv_results = results['survival_analysis']
            report += "📈 SURVIVAL ANALYSIS:\n"
            if 'logrank_stats' in surv_results:
                logrank = surv_results['logrank_stats']
                if 'p_value' in logrank:
                    report += f"• Log-rank test p-value: {logrank['p_value']:.4f}\n"
                    report += f"• Significant survival difference: {'Yes' if logrank['significant'] else 'No'}\n"
            report += "• Cox proportional hazards model fitted successfully\n\n"
        
        # Recommendations
        report += "💡 CLINICAL IMPLICATIONS:\n"
        if val_results['silhouette_score'] > 0.5 and val_results['mean_stability'] > 0.7:
            report += "• High-confidence subtype discovery - suitable for clinical translation\n"
            report += "• Strong biomarker candidates identified for validation\n"
        elif val_results['silhouette_score'] > 0.3:
            report += "• Moderate-confidence results - recommend validation with larger cohort\n"
        else:
            report += "• Low-confidence clustering - consider alternative parameters or methods\n"
        
        if results['survival_analysis'] and results['survival_analysis']['logrank_stats'].get('significant', False):
            report += "• Significant prognostic value detected - potential for risk stratification\n"
        
        report += "\n🎯 NEXT STEPS:\n"
        report += "• Validate top biomarkers in independent cohorts\n"
        report += "• Perform functional enrichment analysis on identified genes\n"
        report += "• Consider drug target analysis for subtype-specific therapies\n"
        
        return report


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    """
    Example usage of the Attention-Enhanced Subtype Discovery Engine
    
    Required data format:
    - data_df: Gene expression DataFrame (samples × genes)
    - survival_df: DataFrame with columns ['time', 'event', ...] 
    """
    
    print("🦪 Attention-Enhanced Subtype Discovery Engine 🍛")
    print("Ready for cancer genomics analysis with attention mechanism!")
    
    # Initialize engine
    engine = AttentionEnhancedEngine(
        n_components=10, 
        n_clusters=3, 
        attention_temperature=1.0
    )
    
    # Example workflow (uncomment when you have real data)
    """
    # Load your data
    data_df = pd.read_csv('gene_expression.csv', index_col=0)
    survival_df = pd.read_csv('survival_data.csv')
    
    # Run complete analysis
    results = engine.fit_predict(data_df, survival_df)
    
    # Generate summary report
    report = engine.get_summary_report()
    print(report)
    
    # Access specific results
    print("Top 5 biomarker genes:", results['top_genes'][:5])
    print("Clustering validation:", results['validation'])
    """