import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ast_new_folder(code_dir):
    input_excel = os.path.join(code_dir, "Result_5_Signal_Mixing", "Mixing_Robustness_Results.xlsx")

    # ایجاد پوشه مستقل هفتم
    output_dir = os.path.join(code_dir, "Result_7_AST_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_excel)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    baseline_noise_hfd = df[df['Target_Ratio'] == 0.0]['Mean_HFD'].values[0]
    plt.axhline(y=baseline_noise_hfd, color='green', linestyle='--', linewidth=2,
                label=f'Ambient Ocean Noise Baseline ({baseline_noise_hfd:.2f})')

    sns.lineplot(data=df, x='Target_Ratio', y='Mean_HFD', marker='o', color='darkred', linewidth=3,
                 label='Mixed Signal HFD')
    plt.gca().invert_xaxis()

    plt.title('Acoustic Stealth Threshold (AST) Analysis')
    plt.xlabel('Target Signal Ratio (1.0 = Pure Target, 0.0 = Pure Noise)')
    plt.ylabel('Mean Higuchi Fractal Dimension (HFD)')
    plt.legend(loc='lower left')

    ast_point = df[abs(df['Mean_HFD'] - baseline_noise_hfd) < 0.05]['Target_Ratio'].max()
    if not pd.isna(ast_point):
        plt.axvline(x=ast_point, color='blue', linestyle=':', linewidth=2)
        plt.text(ast_point + 0.02, baseline_noise_hfd - 0.1, f'Stealth Threshold:\nTarget Ratio < {ast_point}',
                 color='blue', fontweight='bold')

    plt.tight_layout()
    # ذخیره در پوشه جدید
    output_path = os.path.join(output_dir, "Acoustic_Stealth_Threshold_Plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Success! AST plot saved to NEW folder: {output_dir}")


code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
plot_ast_new_folder(code_directory)


#%% Block 2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dual_ast_analysis(code_dir):
    input_excel = os.path.join(code_dir, "Result_5_Signal_Mixing", "Mixing_Robustness_Results.xlsx")
    output_dir = os.path.join(code_dir, "Result_7_AST_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_excel)

    # استخراج مقادیر پایه نویز اقیانوس (Target Ratio = 0.0)
    noise_hfd = df[df['Target_Ratio'] == 0.0]['Mean_HFD'].values[0]
    noise_zcr = df[df['Target_Ratio'] == 0.0]['Mean_ZCR'].values[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # --- نمودار اول: HFD ---
    ax1.axhline(y=noise_hfd, color='green', linestyle='--', linewidth=2, label=f'Noise Baseline ({noise_hfd:.2f})')
    sns.lineplot(data=df, x='Target_Ratio', y='Mean_HFD', marker='o', color='darkred', linewidth=3, ax=ax1,
                 label='Mixed Signal')
    ax1.invert_xaxis()
    ax1.set_title('HFD Stealth Threshold')
    ax1.set_xlabel('Target Signal Ratio')
    ax1.set_ylabel('Mean HFD')

    # محاسبه آستانه استتار HFD (اختلاف کمتر از 0.05 با نویز)
    ast_hfd = df[abs(df['Mean_HFD'] - noise_hfd) < 0.05]['Target_Ratio'].max()
    if not pd.isna(ast_hfd):
        ax1.axvline(x=ast_hfd, color='blue', linestyle=':', linewidth=2)
        ax1.text(ast_hfd + 0.02, noise_hfd - 0.1, f'AST: {ast_hfd}', color='blue', fontweight='bold')

    # --- نمودار دوم: ZCR ---
    ax2.axhline(y=noise_zcr, color='green', linestyle='--', linewidth=2, label=f'Noise Baseline ({noise_zcr:.4f})')
    sns.lineplot(data=df, x='Target_Ratio', y='Mean_ZCR', marker='s', color='navy', linewidth=3, ax=ax2,
                 label='Mixed Signal')
    ax2.invert_xaxis()
    ax2.set_title('ZCR Stealth Threshold')
    ax2.set_xlabel('Target Signal Ratio')
    ax2.set_ylabel('Mean ZCR')

    # محاسبه آستانه استتار ZCR (تلرانس 5 درصدی نسبت به مقدار پایه نویز)
    threshold_zcr = noise_zcr * 0.05
    ast_zcr = df[abs(df['Mean_ZCR'] - noise_zcr) < threshold_zcr]['Target_Ratio'].max()
    if not pd.isna(ast_zcr):
        ax2.axvline(x=ast_zcr, color='blue', linestyle=':', linewidth=2)
        # تنظیم موقعیت متن بر اساس مقیاس ZCR
        y_pos = noise_zcr - (noise_zcr * 0.2) if noise_zcr > 0 else 0.05
        ax2.text(ast_zcr + 0.02, y_pos, f'AST: {ast_zcr}', color='blue', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "Dual_AST_HFD_vs_ZCR.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Success! Dual AST plot saved to: {output_path}")


code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
plot_dual_ast_analysis(code_directory)