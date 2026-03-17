import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid', palette='Set2')
print('Libraries loaded successfully.')

# ── 1.1  Load the raw data ────────────────────────────────────────────────────
raw_df = pd.read_csv('Lab_Exam_binary_classification_dataset.csv')
print('Raw shape:', raw_df.shape)
raw_df.head(10)

# ── 1.2  Basic info & data types ─────────────────────────────────────────────
print('=== Data Types ===')
print(raw_df.dtypes)
print()
print('=== Summary Statistics ===')
raw_df.describe(include='all')

# ── 1.3  Missing values ───────────────────────────────────────────────────────
missing = raw_df.isnull().sum()
print('Missing values per column:')
print(missing[missing > 0])
print(f'\nTotal rows with any missing value: {raw_df.isnull().any(axis=1).sum()}')

# Drop rows where Target is missing (cannot train/evaluate without a label)
df = raw_df.dropna(subset=['Target']).copy()
print(f'\nRows retained after dropping missing Target: {len(df)}')

# ── 1.4  Outlier Detection ────────────────────────────────────────────────────
# Feature1 IQR-based check
Q1 = df['Feature1'].quantile(0.25)
Q3 = df['Feature1'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
outliers_f1 = df[df['Feature1'] > upper_fence]
print(f'Feature1 IQR upper fence: {upper_fence:.4f}')
print(f'Outlier rows in Feature1:')
print(outliers_f1[['Feature1', 'Feature2', 'Target']])

# Remove the extreme outlier before modelling
df = df[df['Feature1'] <= upper_fence].copy()
print(f'\nRows after outlier removal: {len(df)}')

# ── 1.5  Class distribution ───────────────────────────────────────────────────
counts = df['Target'].value_counts()
print('Class counts:')
print(counts)
print(f'\nClass balance (Yes %): {counts["Yes"]/len(df)*100:.1f}%')

fig, ax = plt.subplots(figsize=(4, 3))
counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='black', ax=ax)
ax.set_title('Target Class Distribution')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_xticklabels(['No', 'Yes'], rotation=0)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2, p.get_height() + 5),
                ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# ── 1.6  Feature distributions (histograms + KDE) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, feat, color in zip(axes, ['Feature1', 'Feature2'], ['#3498db', '#9b59b6']):
    df[feat].plot(kind='hist', bins=30, color=color, edgecolor='white',
                  alpha=0.8, ax=ax, density=True)
    df[feat].plot(kind='kde', ax=ax, color='black', linewidth=1.5)
    ax.set_title(f'{feat} Distribution')
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')
plt.suptitle('Feature Distributions', y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150)
plt.show()

# ── 1.7  Feature distributions by class (box plots) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
palette = {'No': '#e74c3c', 'Yes': '#2ecc71'}
for ax, feat in zip(axes, ['Feature1', 'Feature2']):
    sns.boxplot(data=df, x='Target', y=feat, palette=palette, ax=ax, width=0.5)
    ax.set_title(f'{feat} by Class')
    ax.set_xlabel('Target')
plt.suptitle('Feature Distribution by Class', y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots_by_class.png', dpi=150)
plt.show()


# ── 1.8  Scatter plot of both features, coloured by class ─────────────────────
palette = {'No': '#e74c3c', 'Yes': '#2ecc71'}
fig, ax = plt.subplots(figsize=(7, 5))
for label, grp in df.groupby('Target'):
    ax.scatter(grp['Feature1'], grp['Feature2'],
               label=label, color=palette[label], alpha=0.55, edgecolors='none', s=25)
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_title('Feature1 vs Feature2 (coloured by class)')
ax.legend(title='Target')
plt.tight_layout()
plt.savefig('scatter_by_class.png', dpi=150)
plt.show()


# ── 1.9  Correlation heatmap ──────────────────────────────────────────────────
df_num = df.copy()
df_num['Target_bin'] = (df_num['Target'] == 'Yes').astype(int)
corr = df_num[['Feature1', 'Feature2', 'Target_bin']].corr()

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

print('\nCorrelation of features with Target:')
print(corr['Target_bin'].drop('Target_bin'))

# ── 2.1  Prepare feature matrix and label vector ──────────────────────────────
X = df[['Feature1', 'Feature2']].values
y = (df['Target'] == 'Yes').astype(int).values    # Yes → 1,  No → 0
feature_names = ['Feature1', 'Feature2']

print(f'X shape: {X.shape}')
print(f'y distribution: 0 (No) = {(y==0).sum()}, 1 (Yes) = {(y==1).sum()}')


# ── 2.2  Train / Test split (80/20, stratified) ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f'Training samples : {len(X_train)}')
print(f'Test samples     : {len(X_test)}')
print(f'Train class dist.: No={( y_train==0).sum()}, Yes={(y_train==1).sum()}')
print(f'Test  class dist.: No={( y_test==0).sum()},  Yes={(y_test==1).sum()}')

# ── 2.3  Feature scaling ──────────────────────────────────────────────────────
# Logistic Regression is sensitive to feature scale (Feature2 is ~100× larger)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print('Training-set statistics after scaling:')
print(f'  Feature1 mean={X_train_sc[:,0].mean():.4f}, std={X_train_sc[:,0].std():.4f}')
print(f'  Feature2 mean={X_train_sc[:,1].mean():.4f}, std={X_train_sc[:,1].std():.4f}')


# ── 2.4  Train Logistic Regression ───────────────────────────────────────────
log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                             class_weight='balanced')   # handle mild imbalance
log_reg.fit(X_train_sc, y_train)

print('=== Learned Model Parameters ===')
print(f'Intercept (bias)        : {log_reg.intercept_[0]:.4f}')
for name, coef in zip(feature_names, log_reg.coef_[0]):
    print(f'Coefficient ({name:8s}) : {coef:.4f}')

# ── 2.5  Cross-validation (5-fold stratified) ─────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale within each fold using a pipeline to avoid leakage
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(C=1.0, max_iter=1000,
                                  random_state=42, class_weight='balanced'))
])

cv_acc  = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
cv_roc  = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

print(f'5-Fold CV Accuracy  : {cv_acc.mean():.4f}  ± {cv_acc.std():.4f}')
print(f'5-Fold CV ROC-AUC   : {cv_roc.mean():.4f}  ± {cv_roc.std():.4f}')


# ── 3.1  Decision Boundary (in original feature space) ───────────────────────
# Build a dense mesh in original space, scale it, then predict class probability
h = 0.01          # mesh step
x1_min, x1_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
x2_min, x2_max = X[:, 1].min() - 5,    X[:, 1].max() + 5

xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max, h),
    np.arange(x2_min, x2_max, (x2_max - x2_min) / ((x1_max - x1_min) / h))
)
mesh_input = np.c_[xx1.ravel(), xx2.ravel()]
mesh_scaled = scaler.transform(mesh_input)
Z = log_reg.predict(mesh_scaled).reshape(xx1.shape)
Z_prob = log_reg.predict_proba(mesh_scaled)[:, 1].reshape(xx1.shape)

# Colours
palette = {0: '#f1948a', 1: '#82e0aa'}   # red=No, green=Yes

fig, ax = plt.subplots(figsize=(9, 6))

# Probability contour fill
contourf = ax.contourf(xx1, xx2, Z_prob, levels=50,
                        cmap='RdYlGn', alpha=0.55, vmin=0, vmax=1)
fig.colorbar(contourf, ax=ax, label='P(Yes)')

# Hard decision boundary (P = 0.5)
ax.contour(xx1, xx2, Z_prob, levels=[0.5],
           colors='black', linewidths=2.0, linestyles='--')

# Original data points
for label, marker in [(0, 'o'), (1, '^')]:
    mask = (y == label)
    ax.scatter(X[mask, 0], X[mask, 1],
               c=palette[label], edgecolors='black', linewidths=0.4,
               s=28, alpha=0.8, marker=marker,
               label='No' if label == 0 else 'Yes', zorder=3)

# Legend items
boundary_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                            label='Decision Boundary (P=0.5)')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [boundary_line], labels + ['Decision Boundary (P=0.5)'],
          loc='upper right', framealpha=0.9)

ax.set_xlabel('Feature1', fontsize=12)
ax.set_ylabel('Feature2', fontsize=12)
ax.set_title('Logistic Regression – Decision Boundary\n(background = predicted class probability)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150)
plt.show()

print('Decision boundary plot saved as decision_boundary.png')



# ── 4.1  Predictions on the held-out test set ─────────────────────────────────
y_pred      = log_reg.predict(X_test_sc)
y_pred_prob = log_reg.predict_proba(X_test_sc)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f'Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)')
print(f'Test ROC-AUC  : {auc:.4f}')
print()
print('=== Classification Report ===')
print(classification_report(y_test, y_pred, target_names=['No (0)', 'Yes (1)']))


# ── 4.2  Confusion Matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'True Negatives  (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'True Positives  (TP): {tp}')



# ── 4.2  Confusion Matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'True Negatives  (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'True Positives  (TP): {tp}')

# ── 4.4  Performance Summary Table ───────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score

prec   = precision_score(y_test, y_pred)
rec    = recall_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred)

summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Yes)', 'Recall (Yes)', 'F1-Score (Yes)',
               'ROC-AUC',
               '5-Fold CV Accuracy', '5-Fold CV ROC-AUC'],
    'Value' : [f'{acc:.4f}', f'{prec:.4f}', f'{rec:.4f}', f'{f1:.4f}',
               f'{auc:.4f}',
               f'{cv_acc.mean():.4f} ± {cv_acc.std():.4f}',
               f'{cv_roc.mean():.4f} ± {cv_roc.std():.4f}']
})
print('=== Model Performance Summary ===')
print(summary.to_string(index=False))
