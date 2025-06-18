# -Attention-Enhanced-Subtype-Discovery-Engine
The Attention-Enhanced Subtype Discovery Engine introduces an innovative approach to cancer genomics analysis by integrating dimensionality reduction with attention mechanisms. This pipeline creates patient similarity networks through cosine attention matrices, enabling robust subtype identification and biomarker discovery.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> A novel cancer subtype discovery pipeline combining PCA, cosine similarity attention mechanism, and survival analysis for precision oncology.

##  Overview

The **Attention-Enhanced Subtype Discovery Engine** introduces an innovative approach to cancer genomics analysis by integrating dimensionality reduction with attention mechanisms. This pipeline creates patient similarity networks through cosine attention matrices, enabling robust subtype identification and biomarker discovery.

**Key Innovation:** Unlike traditional clustering methods, our attention mechanism constructs patient-patient similarity networks in PCA space, revealing cohesive subgroups and outlier patterns that may represent novel therapeutic targets.

| Input | Output | Clinical Value |
|-------|--------|----------------|
| 🧬 Gene expression (samples × genes) | 🎯 Cancer subtypes | Precision therapy selection |
| 🏥 Survival data (time, event) | 📊 Prognostic biomarkers | Risk stratification |
| | 🔗 Patient similarity networks | Drug response prediction |

## Quick Start

```bash
pip install numpy pandas scikit-learn lifelines matplotlib seaborn
```

```python
from attention_enhanced_engine import AttentionEnhancedEngine

# Initialize with attention mechanism
engine = AttentionEnhancedEngine(
    n_components=10,           # PCA dimensions
    n_clusters=3,              # Expected subtypes
    attention_temperature=1.0   # Attention focus control
)

# Complete analysis pipeline
results = engine.fit_predict(expression_data, survival_data)

# Generate comprehensive report
report = engine.get_summary_report()
print(report)

# Access key results
top_biomarkers = results['top_genes'][:10]
attention_matrix = results['attention_matrix']
subtype_labels = results['cluster_labels']
```

## Core Architecture

### **Attention Mechanism**
```
Gene Expression → PCA Reduction → Cosine Similarity → Attention Weights
     ↓                ↓               ↓                    ↓
Standardization  Low-dimension   Patient Network    Subtype Discovery
```

**Attention Formula:** `attention_weights = softmax(cosine_similarity(PCA_features) / temperature)`

### **Pipeline Components**
1. **Data Preprocessing**: StandardScaler normalization
2. **PCA Dimensionality Reduction**: Capture major expression patterns  
3. **Attention Matrix**: Cosine similarity patient networks
4. **K-means Clustering**: Attention-guided subtype discovery
5. **Feature Ranking**: Inter-cluster variance biomarker identification
6. **Survival Analysis**: Cox regression and Kaplan-Meier curves

## Abalone Yangzhou Visualization

The engine generates a comprehensive **6-panel visualization suite**:

- **PCA Subtype Plot**: Cancer subtypes in reduced space
- **Attention Heatmap**: Patient similarity network with cluster boundaries
- **Survival Curves**: Kaplan-Meier analysis by subtype
- **Biomarker Heatmap**: Top gene expression patterns
- **Feature Importance**: Inter-cluster variance ranking
- **Summary Statistics**: Analysis quality metrics

*The multi-style combines rich visual information like the premium ingredients in this classic dish.*

## Validation Framework

### **Statistical Validation**
- **Silhouette Score**: Cluster cohesion assessment
- **Stability Testing**: 10-fold clustering consistency (Adjusted Rand Index)
- **Log-rank Test**: Survival difference significance

### **Quality Assessment**
```python
validation_results = {
    'silhouette_score': 0.65,           # Cluster quality
    'mean_stability': 0.78,             # Consistency across runs  
    'cluster_quality': 'Good',          # Automated assessment
    'stability_assessment': 'Stable'    # Reliability rating
}
```

##  Applications

- **Cancer Subtype Discovery**: Identify molecularly distinct patient groups
- **Biomarker Identification**: Rank genes by discriminative power
- **Prognosis Prediction**: Survival-based risk stratification  
- **Drug Target Analysis**: Subtype-specific therapeutic opportunities
- **Clinical Trial Design**: Patient stratification for precision medicine

## Advanced Features

### **Temperature-Controlled Attention**
Adjust attention focus with temperature parameter:
- `temperature < 1.0`: Sharp attention (focus on similar patients)
- `temperature > 1.0`: Soft attention (broader similarity consideration)

### **Biomarker Ranking**
Inter-cluster variance scoring identifies genes with:
- High differential expression between subtypes
- Potential therapeutic targeting value
- Prognostic significance

### **Comprehensive Reporting**
Automated generation of clinical-grade analysis reports including:
- Methodology overview and parameters
- Subtype characteristics and biomarkers
- Validation metrics and confidence assessment
- Clinical implications and next steps

## Example Results

```
 ANALYSIS SUMMARY
• Total Samples: 250 patients
• Cancer Subtypes: 3 distinct groups  
• Top Biomarker: EGFR (score: 2.847)
• Cluster Quality: Good (silhouette: 0.65)
• Survival Difference: Significant (p=0.003)
```

##  Contributing

We welcome contributions to enhance the attention mechanism, add new validation metrics, or extend to other cancer types. Please submit issues and pull requests.

##  License

MIT License - see [LICENSE](LICENSE) file for details.

## 📎 Citation

```bibtex
@software{attention_enhanced_engine,
  title={Attention-Enhanced Subtype Discovery Engine for Cancer Genomics},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[username]/attention-enhanced-subtype-discovery}
}
```

---
![image](https://github.com/user-attachments/assets/e5e905b3-0d11-4031-8fab-355a4a0abd06)


** Note**: This tool is designed for research purposes. Clinical ap
