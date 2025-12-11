import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '20251125/Final_Performance_Summary_All_CTRLs_V2.csv'
df = pd.read_csv(file_path)

# 1. Filter out 'PCA_GPR'
df_filtered = df[df['Model'] != 'PCA_GPR'].copy()

# 2. Drop 'Area_Error' and 'EMD' (and empty columns if any)
columns_to_keep = [col for col in df.columns if col not in ['Area_Error', 'EMD'] and 'Unnamed' not in col]
df_filtered = df_filtered[columns_to_keep]

# Display info to understand the remaining structure and models
print("Remaining Models:", df_filtered['Model'].unique())
print("\nColumns:", df_filtered.columns.tolist())

# 3. Visualization
# Define metrics to plot
metrics = ['RMSE', 'MAE', 'Hausdorff_max', 'Chamfer_mean', 'Length_Error', 'Width_Error','Size_Error', 'IoU', 'Dice']

# Set up the plotting style
sns.set(style="whitegrid")
plt.figure(figsize=(20, 15))

# Create a subplot for each metric
for i, metric in enumerate(metrics):
    plt.subplot(3, 3, i + 1)
    # Plot Control Count vs Metric, grouped by Model
    sns.lineplot(data=df_filtered, x='CTRL_Count', y=metric, hue='Model', marker='o')
    plt.title(f'{metric} vs. Control Point Count')
    plt.xlabel('Number of Control Points')
    plt.ylabel(metric)
    plt.legend(title='Model')

plt.tight_layout()
plt.savefig('performance_metrics_plot_combine__2.png')

# 4. Generate a summary table for the text response
summary_table = df_filtered.groupby(['Model', 'CTRL_Count'])[metrics].mean()
print("\nSummary Table Head:")
print(summary_table.head())