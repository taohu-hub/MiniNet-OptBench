import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Load the CSV file
df = pd.read_csv('cifar_training.csv')

# Filter to only include final rows where tta_val_acc is not NaN (these are the summary rows)
df_final = df[df['tta_val_acc'].notna()].copy()

# Parse the param_groups string into a list of dicts
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df_final['params'] = df_final['param_groups'].apply(safe_literal_eval)

# Extract key parameters for labeling
df_final['lr'] = df_final['params'].apply(lambda x: x[0].get('lr', None) if x else None)
df_final['momentum'] = df_final['params'].apply(lambda x: x[0].get('momentum', None) if x else None)
df_final['eps'] = df_final['params'].apply(lambda x: x[0].get('eps', None) if x else None)
df_final['wd'] = df_final['params'].apply(lambda x: x[0].get('weight_decay', None) if x else None)

# Create a config label
def create_label(row):
    parts = []
    if row['lr'] is not None:
        parts.append(f"lr={row['lr']}")
    if row['momentum'] is not None:
        parts.append(f"mom={row['momentum']}")
    if row['eps'] is not None:
        parts.append(f"eps={row['eps']}")
    if row['wd'] is not None:
        parts.append(f"wd={row['wd']}")
    return ', '.join(parts)

df_final['config_label'] = df_final.apply(create_label, axis=1)
df_final['full_label'] = df_final['optimizer'] + ' (' + df_final['config_label'] + ')'

# Ensure types are float
df_final['final_train_acc'] = df_final['final_train_acc'].astype(float)
df_final['final_val_acc'] = df_final['final_val_acc'].astype(float)
df_final['tta_val_acc'] = df_final['tta_val_acc'].astype(float)

# Group by optimizer and param_groups for calculations
grouped = df_final.groupby(['optimizer', 'param_groups'])

# 1. Training set comparison between different optimizer's best parameters (judged by training set performance)
# For each optimizer, find the param_groups with the highest mean final_train_acc
best_train_params = grouped['final_train_acc'].mean().reset_index().groupby('optimizer').apply(lambda x: x.loc[x['final_train_acc'].idxmax()])
best_train_df = pd.merge(df_final, best_train_params[['optimizer', 'param_groups']], on=['optimizer', 'param_groups'])

plt.figure(figsize=(10, 6))
sns.boxplot(data=best_train_df, x='optimizer', y='final_train_acc', hue='config_label')
plt.title('1. Training Set Performance: Best Parameters per Optimizer (Judged by Train Acc)')
plt.ylabel('Final Train Accuracy')
plt.xticks(rotation=45)
plt.legend(title='Config')
plt.tight_layout()
plt.show()

# 2. Test set comparison between different optimizer's best parameters (judged by test set performance)
# Using tta_val_acc as test performance
best_test_params = grouped['tta_val_acc'].mean().reset_index().groupby('optimizer').apply(lambda x: x.loc[x['tta_val_acc'].idxmax()])
best_test_df = pd.merge(df_final, best_test_params[['optimizer', 'param_groups']], on=['optimizer', 'param_groups'])

plt.figure(figsize=(10, 6))
sns.boxplot(data=best_test_df, x='optimizer', y='tta_val_acc', hue='config_label')
plt.title('2. Test Set Performance: Best Parameters per Optimizer (Judged by TTA Val Acc)')
plt.ylabel('TTA Val Accuracy')
plt.xticks(rotation=45)
plt.legend(title='Config')
plt.tight_layout()
plt.show()

# 3. Training set comparison between different parameter groups for each fixed optimizer
optimizers = df_final['optimizer'].unique()
for opt in optimizers:
    opt_df = df_final[df_final['optimizer'] == opt]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=opt_df, x='config_label', y='final_train_acc')
    plt.title(f'3. Training Set Performance: Parameter Groups for {opt}')
    plt.ylabel('Final Train Accuracy')
    plt.xlabel('Parameter Groups')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Testing set comparison between different parameter groups for each fixed optimizer
for opt in optimizers:
    opt_df = df_final[df_final['optimizer'] == opt]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=opt_df, x='config_label', y='tta_val_acc')
    plt.title(f'4. Test Set Performance: Parameter Groups for {opt}')
    plt.ylabel('TTA Val Accuracy')
    plt.xlabel('Parameter Groups')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5. Training set performance of all parameters and all optimizers
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_final, x='full_label', y='final_train_acc')
plt.title('5. Training Set Performance: All Parameters and Optimizers')
plt.ylabel('Final Train Accuracy')
plt.xlabel('Optimizer and Config')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 6. Test set performance of all parameters and all optimizers
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_final, x='full_label', y='tta_val_acc')
plt.title('6. Test Set Performance: All Parameters and Optimizers')
plt.ylabel('TTA Val Accuracy')
plt.xlabel('Optimizer and Config')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
