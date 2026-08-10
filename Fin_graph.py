import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# [설정] 파일 경로
file_path = '20260106/Final_Performance_Summary_CTRL20_50.csv'

try:
    df = pd.read_csv(file_path)

    # 1. PCA_GPR 제외 및 불필요한 컬럼 제거
    df_filtered = df[df['Model'] != 'PCA_GPR'].copy()
    columns_to_keep = [col for col in df.columns if col not in ['Area_Error', 'EMD'] and 'Unnamed' not in col]
    df_filtered = df_filtered[columns_to_keep]

    # 2. 시각화할 지표 설정
    metrics = ['Hausdorff_max', 'Chamfer_mean', 'IoU'] 

    # 3. 그래프 그리기
    sns.set(style="whitegrid")
    
    # 그래프 크기 및 서브플롯 배치 조정 (1행 3열)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(data=df_filtered, x='CTRL_Count', y=metric, hue='Model', marker='o', ax=ax)
        
        ax.set_title(f'{metric} vs. Control Point Count')
        ax.set_xlabel('Number of Control Points')
        ax.set_ylabel(metric)
        

        # IoU일 때만 Y축 범위 고정 (0.8 ~ 1.0)
        if metric == 'IoU':
            ax.set_ylim(0.8, 1.0)
            ax.legend(title='Model', loc='lower right')
        if metric == 'Hausdorff_max':
            ax.legend(title='Model', loc='upper right')
        if metric == 'Chamfer_mean':
            ax.legend(title='Model', loc='center right')        

    plt.tight_layout()
    plt.savefig('test.png')
    plt.show()
    print("그래프 저장 완료: performance_metrics_plot_combine__2___dfasdfasd.png")

except FileNotFoundError:
    print(f"[Error] 파일을 찾을 수 없습니다: {file_path}")
    print("경로를 확인하거나 이전 단계에서 CSV 생성이 완료되었는지 확인해주세요.")