import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns


def higuchi_fd_sweep(x, kmax):
    N = len(x)
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = np.zeros(k)
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) <= 1: continue
            Lmk = np.sum(np.abs(np.diff(x[idxs])))
            norm = (N - 1) / (idxs[-1] - m) / k
            Lk[m] = Lmk * norm
        L[k - 1] = np.mean(Lk)
    ln_L = np.log(L)
    ln_k = np.log(1 / np.arange(1, kmax + 1))
    slope, _ = np.polyfit(ln_k, ln_L, 1)
    return slope


def run_kmax_sweep(base_dir, code_dir, window_size_sec=0.05):
    # ایجاد پوشه اختصاصی نتایج
    output_dir = os.path.join(code_dir, "Result_3_Kmax_Sweep")
    os.makedirs(output_dir, exist_ok=True)

    original_excel = os.path.join(base_dir, "shipsEar.xlsx")
    output_excel = os.path.join(output_dir, "Kmax_Sweep_Results.xlsx")

    df = pd.read_excel(original_excel)
    kmax_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    target_classes = ['Dredger', 'Passengers', 'RORO', 'Natural ambient noise']
    selected_files = {}

    # انتخاب یک فایل نمونه از هر کلاس
    for index, row in df.iterrows():
        vessel_type = row.get('Type')
        filename = row.get('Filename')
        if vessel_type in target_classes and vessel_type not in selected_files:
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath): selected_files[vessel_type] = filepath

    results = []
    for v_type, filepath in selected_files.items():
        print(f"Processing kmax sweep for: {v_type}")
        sample_rate, data = wavfile.read(filepath)
        if data.ndim > 1: data = data[:, 0]
        data = data[:min(len(data), 10 * sample_rate)]  # محدود به 10 ثانیه اول برای افزایش سرعت

        window_samples = int(window_size_sec * sample_rate)
        num_windows = len(data) // window_samples

        for k in kmax_values:
            hfd_list = []
            for i in range(num_windows):
                window = data[i * window_samples: (i + 1) * window_samples]
                if np.max(np.abs(window)) > 0:
                    hfd_list.append(higuchi_fd_sweep(window, k))
            results.append({'Type': v_type, 'kmax': k, 'Mean_HFD': np.mean(hfd_list) if hfd_list else np.nan})

    df_results = pd.DataFrame(results)
    df_results.to_excel(output_excel, index=False)

    # رسم نمودار خطی پارامتریک
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df_results, x='kmax', y='Mean_HFD', hue='Type', marker='o', linewidth=2)
    plt.title('Parametric Sweep: Effect of kmax on Higuchi Fractal Dimension')
    plt.xlabel('kmax (Maximum Time Interval Scale)')
    plt.ylabel('Mean HFD')
    plt.xticks(kmax_values)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmax_Parametric_Sweep.png"))
    plt.close()
    print(f"\nSuccess! Results and plot saved to: {output_dir}")


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
run_kmax_sweep(base_directory, code_directory)