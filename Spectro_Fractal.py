import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


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


def run_spectro_fractal_analysis(base_dir, code_dir, window_size_sec=0.05, duration_sec=30):
    output_dir = os.path.join(code_dir, "Result_6_Spectro_Fractal")
    os.makedirs(output_dir, exist_ok=True)

    original_excel = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(original_excel)

    # انتخاب دو کلاس برای مقایسه
    target_classes = ['Passengers', 'Natural ambient noise']

    for v_type in target_classes:
        filename = df[df['Type'] == v_type]['Filename'].dropna().iloc[0]
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath): continue

        print(f"Generating Spectro-Fractal plot for: {v_type}")
        sample_rate, data = wavfile.read(filepath)
        if data.ndim > 1: data = data[:, 0]

        # برش سیگنال به 30 ثانیه اول
        max_samples = min(int(duration_sec * sample_rate), len(data))
        data = data[:max_samples].astype(float)

        # محاسبه HFD در طول زمان
        window_samples = int(window_size_sec * sample_rate)
        num_windows = len(data) // window_samples

        time_axes_hfd = np.linspace(0, duration_sec, num_windows)
        hfd_list = []

        for i in range(num_windows):
            window = data[i * window_samples: (i + 1) * window_samples]
            hfd_list.append(higuchi_fd(window, kmax=10) if np.max(np.abs(window)) > 0 else np.nan)

        # رسم ترکیب طیف‌نگار و HFD
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # تولید اسپکتروگرام (Spectrogram)
        f, t, Sxx = signal.spectrogram(data, sample_rate, nperseg=1024, noverlap=512)
        # تبدیل به مقیاس دسی‌بل و محدود کردن فرکانس برای وضوح بهتر
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        cax = ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax1.set_ylim(0, min(sample_rate / 2, 10000))  # تمرکز روی فرکانس‌های زیر 10 کیلوهرتز
        ax1.set_ylabel('Frequency [Hz]', color='black')
        ax1.set_xlabel('Time [sec]')
        ax1.set_title(f'Spectro-Fractal Analysis: {v_type}')
        fig.colorbar(cax, ax=ax1, label='Intensity [dB]')

        # اضافه کردن محور دوم برای HFD
        ax2 = ax1.twinx()
        ax2.plot(time_axes_hfd, hfd_list, color='red', linewidth=2.5, alpha=0.85, label='Higuchi FD')
        ax2.set_ylabel('Higuchi Fractal Dimension (HFD)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(1.0, 2.1)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"Spectro_Fractal_{v_type.replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print(f"\nSuccess! Spectro-Fractal overlays saved to: {output_dir}")


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

run_spectro_fractal_analysis(base_directory, code_directory)