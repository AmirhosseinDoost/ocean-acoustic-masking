import os
import time
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns


def higuchi_fd(x, kmax=10):
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


def katz_fd(x):
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    if d == 0 or L == 0: return 0.0
    n = len(x) - 1
    return np.log10(n) / (np.log10(d / L) + np.log10(n))


def zero_crossing_rate(x):
    return np.sum(np.abs(np.diff(np.sign(x)))) / (2 * len(x))


def process_static_features_full_length(base_dir, code_dir, window_size_sec=0.05):
    output_dir = os.path.join(code_dir, "Result_1_Static_Features")
    os.makedirs(output_dir, exist_ok=True)

    original_excel = os.path.join(base_dir, "shipsEar.xlsx")
    output_excel = os.path.join(output_dir, "ShipsEar_Acoustic_Features_Master.xlsx")

    print("Starting full-length static feature extraction...")
    df = pd.read_excel(original_excel)
    for col in ['Mean_HFD', 'Katz_FD', 'Zero_Crossing_Rate']:
        if col not in df.columns: df[col] = None

    total_files = len(df.dropna(subset=['Filename']))
    processed_count = 0

    for index, row in df.iterrows():
        filename = row.get('Filename')
        if pd.isna(filename): continue
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath): continue

        try:
            start_time = time.time()
            sample_rate, data = wavfile.read(filepath)
            if data.ndim > 1: data = data[:, 0]

            # بدون محدودیت: استفاده از کل طول فایل صوتی
            window_samples = int(window_size_sec * sample_rate)
            num_windows = len(data) // window_samples

            hfd_list, kfd_list, zcr_list = [], [], []
            for i in range(num_windows):
                window = data[i * window_samples: (i + 1) * window_samples]
                if np.max(np.abs(window)) > 0:
                    hfd_list.append(higuchi_fd(window, kmax=10))
                    kfd_list.append(katz_fd(window))
                    zcr_list.append(zero_crossing_rate(window))

            if hfd_list:
                df.at[index, 'Mean_HFD'] = np.mean(hfd_list)
                df.at[index, 'Katz_FD'] = np.mean(kfd_list)
                df.at[index, 'Zero_Crossing_Rate'] = np.mean(zcr_list)

            elapsed = time.time() - start_time
            processed_count += 1
            print(
                f"[{processed_count}/{total_files}] Processed {filename} -> {num_windows} windows in {elapsed:.1f} seconds")

        except Exception as e:
            print(f"Error on {filename}: {e}")

    df.to_excel(output_excel, index=False)
    print(f"\nExcel saved to: {output_excel}")

    # رسم نمودار
    df_plot = df.dropna(subset=['Type', 'Mean_HFD'])
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Mean_HFD', y='Zero_Crossing_Rate', hue='Type', data=df_plot, s=100, alpha=0.8)
    plt.title('Non-linear Dynamics: HFD vs Zero Crossing Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_hfd_zcr.png'))
    plt.close()
    print("Plot saved successfully in Result_1 folder.")


# مسیرهای اجرای کد
base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

process_static_features_full_length(base_directory, code_directory)
#%% block 2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = r"D:\Academic Projects\Passive\Dataset\Code\Result_1_Static_Features"
excel_path = os.path.join(output_dir, "ShipsEar_Acoustic_Features_Master.xlsx")

# خواندن فایل اکسلی که قبلا تولید شده است
df = pd.read_excel(excel_path)
df_plot = df.dropna(subset=['Type', 'Mean_HFD']).copy()

# یکپارچه‌سازی املای نام کلاس‌ها برای جلوگیری از ایجاد ستون‌های تکراری
df_plot['Type'] = df_plot['Type'].replace('fishboat', 'Fishboat')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# رسم نمودار جعبه‌ای (کندل)
sns.boxplot(x='Type', y='Mean_HFD', data=df_plot, hue='Type', palette='viridis', legend=False)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Higuchi Fractal Dimension (HFD) by Vessel Type')
plt.ylabel('Mean HFD (Complexity/Chaos)')
plt.xlabel('Acoustic Source Type')
plt.tight_layout()

# ذخیره تصویر
plot_path = os.path.join(output_dir, 'hfd_boxplot.png')
plt.savefig(plot_path)
plt.close()

print(f"Boxplot saved successfully to: {plot_path}")