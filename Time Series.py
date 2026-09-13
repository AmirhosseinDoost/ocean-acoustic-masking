import os
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


def process_timeseries_categorized(base_dir, code_dir, window_size_sec=0.05):
    # مسیر اختصاصی برای نتایج این بخش
    output_dir = os.path.join(code_dir, "Result_2_Time_Series")
    os.makedirs(output_dir, exist_ok=True)

    original_excel = os.path.join(base_dir, "shipsEar.xlsx")
    timeseries_excel_path = os.path.join(output_dir, "ShipsEar_TimeSeries_HFD.xlsx")
    plots_base_dir = os.path.join(output_dir, "Individual_Audio_Plots")

    os.makedirs(plots_base_dir, exist_ok=True)
    df_meta = pd.read_excel(original_excel)
    all_data = []
    sns.set_theme(style="whitegrid")

    for index, row in df_meta.iterrows():
        filename = row.get('Filename')
        vessel_type = row.get('Type')
        if pd.isna(filename) or pd.isna(vessel_type): continue

        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath): continue

        safe_class_name = str(vessel_type).replace('/', '_').replace('\\', '_').strip()
        class_folder = os.path.join(plots_base_dir, safe_class_name)
        os.makedirs(class_folder, exist_ok=True)

        try:
            sample_rate, data = wavfile.read(filepath)
            if data.ndim > 1: data = data[:, 0]
            window_samples = int(window_size_sec * sample_rate)
            num_windows = len(data) // window_samples

            file_data = {'Filename': filename, 'Type': vessel_type}
            hfd_values = []

            for i in range(num_windows):
                window = data[i * window_samples: (i + 1) * window_samples]
                hfd_val = higuchi_fd(window) if np.max(np.abs(window)) > 0 else np.nan
                file_data[f'Win_{i + 1}'] = hfd_val
                hfd_values.append(hfd_val)

            all_data.append(file_data)
            hfd_plot_vals = np.array([x for x in hfd_values if not np.isnan(x)])

            if len(hfd_plot_vals) > 0:
                time_axis = np.arange(len(hfd_plot_vals)) * window_size_sec
                plt.figure(figsize=(12, 4))
                plt.plot(time_axis, hfd_plot_vals, linewidth=1.2, color='teal')
                plt.title(f'Temporal Dynamics of HFD | {filename}')
                plt.xlabel('Time (Seconds)')
                plt.ylabel('HFD')
                plt.ylim(1.0, 2.0)
                plt.tight_layout()
                plt.savefig(os.path.join(class_folder, f"{str(filename).replace('.wav', '')}_HFD.png"))
                plt.close()

        except Exception as e:
            pass

    pd.DataFrame(all_data).to_excel(timeseries_excel_path, index=False)


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
process_timeseries_categorized(base_directory, code_directory)