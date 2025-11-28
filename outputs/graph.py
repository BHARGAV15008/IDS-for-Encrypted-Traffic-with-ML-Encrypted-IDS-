import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the three JSON files
files = ['Results_01.json', 'Results_02.json', 'Results_03.json']
labels = ['Results_01', 'Results_02', 'Results_03']

datas = []
for file in files:
    with open(file, 'r') as f:
        datas.append(json.load(f))

data1, data2, data3 = datas

# Extract main metrics
metrics_list = []

# Results_01 and Results_02 (have phase4)
for i, data in enumerate(datas[:2]):
    m = data['phase4']['metrics']
    metrics_list.append({
        'Model': labels[i],
        'Accuracy': m['accuracy'],
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1-Score': m['f1'],
        'AUC-ROC': m.get('auc_roc', None),
        'FPR': m['fpr'],
        'Specificity': m['specificity']
    })

# Results_03 (metrics at top level)
m3 = data3
tn, fp = m3['confusion_matrix'][0]
fn, tp = m3['confusion_matrix'][1]
total_neg = tn + fp
specificity = tn / total_neg if total_neg > 0 else 0
fpr = fp / total_neg if total_neg > 0 else 0

metrics_list.append({
    'Model': labels[2],
    'Accuracy': m3['accuracy'],
    'Precision': m3['precision'],
    'Recall': m3['recall'],
    'F1-Score': m3['f1'],
    'AUC-ROC': None,
    'FPR': round(fpr, 10),
    'Specificity': round(specificity, 10)
})

df_metrics = pd.DataFrame(metrics_list)

# Plot 1: Key Metrics Comparison (zoomed to see tiny differences)
plt.figure(figsize=(12, 7))
df_plot = df_metrics.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                          var_name='Metric', value_name='Score')

sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette='tab10')
plt.title('Key Metrics Comparison (zoomed view > 0.999)', fontsize=16)
plt.ylim(0.9990, 1.0001)
plt.ylabel('Score')
plt.grid(axis='y', alpha=0.3)
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.7f', fontsize=9)
plt.legend(title='Model')
plt.tight_layout()
plt.show()

# Confusion Matrices (absolute values)
cms = [
    [[data1['phase4']['threshold_metrics']['tn'], data1['phase4']['threshold_metrics']['fp']],
     [data1['phase4']['threshold_metrics']['fn'], data1['phase4']['threshold_metrics']['tp']]],
    [[data2['phase4']['threshold_metrics']['tn'], data2['phase4']['threshold_metrics']['fp']],
     [data2['phase4']['threshold_metrics']['fn'], data2['phase4']['threshold_metrics']['tp']]],
    data3['confusion_matrix']
]

totals = [np.sum(cm) for cm in cms]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (cm, label, total) in enumerate(zip(cms, labels, totals)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black',
                cbar=False, ax=axes[i],
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'])
    axes[i].set_title(f'{label}\nConfusion Matrix\nTotal samples: {total:,}', fontsize=14)
plt.tight_layout()
plt.show()

# Normalized Confusion Matrices (%)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (cm, label) in enumerate(zip(cms, labels)):
    cm_norm = cm / np.sum(cm) * 100
    sns.heatmap(cm_norm, annot=True, fmt='.4f', cmap='Blues', linewidths=1, linecolor='black',
                cbar=False, ax=axes[i], annot_kws={"size": 11},
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'])
    axes[i].set_title(f'{label}\nNormalized Confusion Matrix (%)', fontsize=14)
plt.tight_layout()
plt.show()

# FPR vs Specificity (extreme zoom)
plt.figure(figsize=(10, 6))
df_fpr_spec = df_metrics.melt(id_vars='Model', value_vars=['FPR', 'Specificity'],
                              var_name='Metric', value_name='Score')

sns.barplot(data=df_fpr_spec, x='Metric', y='Score', hue='Model', palette='Set2')
plt.title('False Positive Rate vs Specificity (extreme zoom)', fontsize=16)
plt.ylim(-0.0000001, 0.00012)
plt.ylabel('Score')
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.10f', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Dataset sizes
sample_data = [
    {'Model': 'Results_01', 'Cleaned': data1['phase1']['cleaned_samples'], 'Balanced': data1['phase1']['balanced_samples'], 'Test': totals[0]},
    {'Model': 'Results_02', 'Cleaned': data2['phase1']['cleaned_samples'], 'Balanced': data2['phase1']['balanced_samples'], 'Test': totals[1]},
    {'Model': 'Results_03', 'Cleaned': None, 'Balanced': None, 'Test': totals[2]}
]

df_samples = pd.DataFrame(sample_data)
print("\nDataset Sizes:")
print(df_samples)

plt.figure(figsize=(10, 6))
df_samples_melt = df_samples.melt(id_vars='Model', value_vars=['Cleaned', 'Balanced', 'Test'],
                                  var_name='Dataset', value_name='Samples')
sns.barplot(data=df_samples_melt.dropna(), x='Model', y='Samples', hue='Dataset', palette='viridis')
plt.title('Dataset Sizes Comparison')
plt.yscale('log')
plt.ylabel('Number of Samples (log scale)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
