# Abalone Yangzhou Fried Rice Visualization Demo
# Generate sample data and showcase the complete visualization suite

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_demo_data():
    """
    Generate realistic demo data simulating cancer genomics study.
    """
    # Generate 200 patients with 50 genes (simulating S100 family + related genes)
    n_patients = 200
    n_genes = 50
    
    # Create three distinct cancer subtypes with different expression patterns
    subtype1_patients = 70  # Aggressive subtype
    subtype2_patients = 80  # Intermediate subtype  
    subtype3_patients = 50  # Indolent subtype
    
    # Gene expression patterns for each subtype
    # Subtype 1: High S100A4, S100A8 (aggressive)
    expr1 = np.random.normal(8, 1.5, (subtype1_patients, n_genes))
    expr1[:, :5] += np.random.normal(3, 0.5, (subtype1_patients, 5))  # Boost first 5 genes
    
    # Subtype 2: Moderate expression (intermediate)
    expr2 = np.random.normal(6, 1.2, (subtype2_patients, n_genes))
    expr2[:, 5:10] += np.random.normal(2, 0.5, (subtype2_patients, 5))  # Boost genes 6-10
    
    # Subtype 3: Low overall expression (indolent)
    expr3 = np.random.normal(4, 1.0, (subtype3_patients, n_genes))
    expr3[:, 10:15] += np.random.normal(1.5, 0.3, (subtype3_patients, 5))  # Boost genes 11-15
    
    # Combine expression data
    expression_data = np.vstack([expr1, expr2, expr3])
    
    # Create gene names (S100 family + related cancer genes)
    gene_names = ['S100A4', 'S100A8', 'S100A9', 'S100A11', 'S100A12'] + \
                ['EGFR', 'TP53', 'MYC', 'BRCA1', 'KRAS'] + \
                [f'GENE_{i:02d}' for i in range(11, n_genes+1)]
    
    # Create patient IDs
    patient_ids = [f'Patient_{i:03d}' for i in range(1, n_patients+1)]
    
    # Expression DataFrame
    expr_df = pd.DataFrame(expression_data, 
                          index=patient_ids, 
                          columns=gene_names)
    
    # Generate survival data with subtype-dependent outcomes
    true_labels = np.array([0]*subtype1_patients + [1]*subtype2_patients + [2]*subtype3_patients)
    
    # Survival times (aggressive subtype has shorter survival)
    survival_times = []
    events = []
    
    for label in true_labels:
        if label == 0:  # Aggressive subtype
            time = np.random.exponential(15) + 5  # Shorter survival
            event = np.random.choice([0, 1], p=[0.3, 0.7])  # Higher event rate
        elif label == 1:  # Intermediate subtype
            time = np.random.exponential(25) + 10
            event = np.random.choice([0, 1], p=[0.5, 0.5])
        else:  # Indolent subtype
            time = np.random.exponential(40) + 15  # Longer survival
            event = np.random.choice([0, 1], p=[0.7, 0.3])  # Lower event rate
        
        survival_times.append(time)
        events.append(event)
    
    survival_df = pd.DataFrame({
        'patient_id': patient_ids,
        'time': survival_times,
        'event': events,
        'age': np.random.normal(65, 12, n_patients),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_patients)
    })
    
    return expr_df, survival_df, true_labels

def create_abalone_yangzhou_demo():
    """
    Create the complete Abalone Yangzhou Fried Rice visualization demo.
    """
    # Generate demo data
    print("🍚 Preparing ingredients for Abalone Yangzhou Fried Rice...")
    expr_df, survival_df, true_labels = generate_demo_data()
    
    # Data preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expr_df)
    
    # PCA + Attention mechanism
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Cosine similarity attention matrix
    cosine_sim = cosine_similarity(X_reduced)
    attention_matrix = np.exp(cosine_sim) / np.exp(cosine_sim).sum(axis=1, keepdims=True)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_reduced)
    
    # Feature ranking (inter-cluster variance)
    cluster_df = expr_df.copy()
    cluster_df['cluster'] = cluster_labels
    cluster_means = cluster_df.groupby('cluster').mean()
    inter_cluster_var = cluster_means.var(axis=0)
    top_genes = inter_cluster_var.sort_values(ascending=False).head(10).index.tolist()
    
    print("🔥 Stir-frying the data in high heat...")
    
    # Create the magnificent visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Color palette - rich like abalone sauce
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][:3]
    
    # 1. PCA Scatter Plot - The main dish base 🍚
    ax1 = fig.add_subplot(gs[0, :2])
    for i, color in enumerate(colors):
        mask = cluster_labels == i
        ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   c=color, label=f'Subtype {i}', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.set_title('🍚 Cancer Subtype Discovery in PCA Space\n(The Rice Base)', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#FAFAFA')
    
    # 2. Attention Heatmap - The premium abalone sauce 🦪
    ax2 = fig.add_subplot(gs[0, 2:])
    sort_idx = np.argsort(cluster_labels)
    sorted_attention = attention_matrix[sort_idx][:, sort_idx]
    
    im = ax2.imshow(sorted_attention, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
    ax2.set_title('🦪 Cosine Similarity Attention Matrix\n(Premium Abalone Sauce)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Patient Index (sorted by subtype)')
    ax2.set_ylabel('Patient Index (sorted by subtype)')
    
    # Add cluster boundaries with style
    cluster_counts = np.bincount(cluster_labels)
    boundaries = np.cumsum(cluster_counts)[:-1] - 0.5
    for boundary in boundaries:
        ax2.axhline(boundary, color='white', linewidth=3, alpha=0.8)
        ax2.axvline(boundary, color='white', linewidth=3, alpha=0.8)
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # 3. Survival Curves - The timing perfection ⏰
    ax3 = fig.add_subplot(gs[1, :2])
    df_with_clusters = survival_df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    survival_stats = []
    for i, color in enumerate(colors):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
        
        kmf = KaplanMeierFitter()
        kmf.fit(cluster_data['time'], 
               event_observed=cluster_data['event'],
               label=f'Subtype {i} (n={len(cluster_data)})')
        
        kmf.plot_survival_function(ax=ax3, color=color, linewidth=3, alpha=0.8)
        
        # Calculate median survival
        median_survival = kmf.median_survival_time_
        survival_stats.append(f'Subtype {i}: {median_survival:.1f} months')
    
    ax3.set_title('⏰ Kaplan-Meier Survival Curves\n(Perfect Timing Like Yangzhou Chef)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (months)', fontsize=12)
    ax3.set_ylabel('Survival Probability', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#FAFAFA')
    
    # 4. Top Biomarker Heatmap - The premium ingredients 🧬
    ax4 = fig.add_subplot(gs[1, 2:])
    top_expr_data = expr_df[top_genes].T
    sorted_expr = top_expr_data.iloc[:, sort_idx]
    
    sns.heatmap(sorted_expr, ax=ax4, cmap='RdBu_r', center=sorted_expr.mean().mean(), 
               cbar_kws={'shrink': 0.8, 'label': 'Expression Level'}, 
               yticklabels=True, xticklabels=False, 
               linewidths=0.1, linecolor='white')
    ax4.set_title('🧬 Top 10 Biomarker Expression\n(Premium Ingredients)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Patients (sorted by subtype)')
    ax4.set_ylabel('Biomarker Genes')
    
    # 5. Feature Importance - The seasoning balance 🧂
    ax5 = fig.add_subplot(gs[2, :2])
    top_scores = inter_cluster_var.sort_values(ascending=False).head(10)
    
    bars = ax5.bar(range(len(top_scores)), top_scores.values, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']*2, 
                  alpha=0.8, edgecolor='white', linewidth=1)
    ax5.set_title('🧂 Top 10 Biomarker Importance\n(Seasoning Balance)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Gene Rank')
    ax5.set_ylabel('Inter-Cluster Variance Score')
    ax5.set_xticks(range(len(top_scores)))
    ax5.set_xticklabels(top_scores.index, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_facecolor('#FAFAFA')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_scores.values)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Chef's Summary - The master's touch 👨‍🍳
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    # Create an elegant summary
    cluster_sizes = np.bincount(cluster_labels)
    summary_text = "👨‍🍳 CHEF'S ANALYSIS SUMMARY 👨‍🍳\n\n"
    summary_text += "═" * 40 + "\n"
    summary_text += f"🍽️  Total Diners: {len(cluster_labels)} patients\n"
    summary_text += f"🍚  Rice Varieties: {len(colors)} cancer subtypes\n"
    summary_text += f"🦪  Abalone Quality: Premium attention mechanism\n"
    summary_text += f"🧬  Premium Ingredients: {len(top_genes)} top biomarkers\n\n"
    
    summary_text += "📊 PORTION DISTRIBUTION:\n"
    for i, size in enumerate(cluster_sizes):
        percentage = (size / len(cluster_labels)) * 100
        summary_text += f"   Subtype {i}: {size} patients ({percentage:.1f}%)\n"
    
    summary_text += f"\n🍳 COOKING TECHNIQUE:\n"
    summary_text += f"   • PCA Components: {pca.n_components_}\n"
    summary_text += f"   • Attention Temperature: 1.0°C\n"
    summary_text += f"   • Explained Variance: {sum(pca.explained_variance_ratio_[:2]):.1%}\n"
    
    summary_text += f"\n⭐ CHEF'S RECOMMENDATION:\n"
    if len(set(cluster_labels)) == 3:
        summary_text += "   Perfect balance of flavors achieved!\n"
        summary_text += "   Ready for clinical tasting.\n"
    
    summary_text += f"\n🥢 SURVIVAL TASTE TEST:\n"
    for stat in survival_stats[:3]:
        summary_text += f"   • {stat}\n"
    
    # Create a fancy text box
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF8DC", 
                     edgecolor="#DAA520", linewidth=2, alpha=0.9))
    
    # Add decorative elements
    ax6.text(0.95, 0.05, "🦪🍚🥢", transform=ax6.transAxes, 
            fontsize=24, ha='right', va='bottom', alpha=0.7)
    
    # Super title with style
    plt.suptitle('🦪 ABALONE YANGZHOU FRIED RICE 🍛\nAttention-Enhanced Cancer Subtype Discovery', 
                 fontsize=20, fontweight='bold', y=0.98,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFE4B5", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    print("🎉 Abalone Yangzhou Fried Rice is ready to serve!")
    print("👨‍🍳 Bon appétit! The visualization feast awaits...")
    
    plt.show()
    
    return {
        'expression_data': expr_df,
        'survival_data': survival_df,
        'cluster_labels': cluster_labels,
        'attention_matrix': attention_matrix,
        'top_genes': top_genes,
        'pca_components': X_reduced
    }

# Run the demo
if __name__ == "__main__":
    print("🔥 Welcome to the Abalone Yangzhou Fried Rice Kitchen! 🔥")
    print("👨‍🍳 Preparing the most exquisite cancer genomics visualization...")
    
    results = create_abalone_yangzhou_demo()
    
    print("\n🍽️ Your Abalone Yangzhou Fried Rice is served!")
    print("🦪 Notice the rich attention matrix - like premium abalone sauce")
    print("🍚 Each rice grain (data point) perfectly positioned")
    print("🧬 Premium biomarker ingredients highlighted")
    print("⏰ Survival curves cooked to perfection")
    
    print(f"\n📊 Dish Analysis:")
    print(f"   • {len(results['expression_data'])} patients (rice grains)")
    print(f"   • {len(set(results['cluster_labels']))} subtypes (flavor profiles)")
    print(f"   • {len(results['top_genes'])} premium biomarkers")
    print(f"   • Attention matrix: {results['attention_matrix'].shape}")
