import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_classical_passive_sonar(code_dir):
    audio_dir = os.path.join(code_dir, "Result_5_Signal_Mixing", "Result_5_Mixed_Audio_Files")
    output_dir = os.path.join(code_dir, "Result_8_Classical_Sonar_FFT")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found at {audio_dir}")
        return

    ratios = np.round(np.arange(0.0, 1.1, 0.1), 1)
    detection_threshold_db = 3.0  # حداقل اختلاف 3 دسی‌بل برای کشف در سونارهای کلاسیک

    # 1. خواندن نویز خالص (0%) برای تعیین کف نویز (Noise Floor)
    noise_file = os.path.join(audio_dir, "Mixed_0pct_Target_100pct_Noise.wav")
    sr, sig_noise = wavfile.read(noise_file)
    f_noise, psd_noise = welch(sig_noise, sr, nperseg=4096)
    psd_noise_db = 10 * np.log10(psd_noise + 1e-10)

    # 2. خواندن هدف خالص (100%) برای پیدا کردن فرکانس شاخص (Tonal Peak)
    target_file = os.path.join(audio_dir, "Mixed_100pct_Target_0pct_Noise.wav")
    _, sig_target = wavfile.read(target_file)
    f_target, psd_target = welch(sig_target, sr, nperseg=4096)
    psd_target_db = 10 * np.log10(psd_target + 1e-10)

    # پیدا کردن قله اصلی فرکانس موتور در محدوده زیر 3000 هرتز
    valid_idx = np.where(f_target < 3000)[0]
    peak_idx = valid_idx[np.argmax(psd_target_db[valid_idx])]
    target_peak_freq = f_target[peak_idx]
    noise_floor_at_peak = psd_noise_db[peak_idx]

    print(f"Target characteristic peak identified at: {target_peak_freq:.1f} Hz")
    print(f"Ambient noise floor at this frequency: {noise_floor_at_peak:.1f} dB")

    results = []

    # 3. تحلیل طیفی تمام فایل‌های ترکیبی
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    for alpha in ratios:
        t_pct = int(alpha * 100)
        n_pct = int((1.0 - alpha) * 100)
        filename = f"Mixed_{t_pct}pct_Target_{n_pct}pct_Noise.wav"
        filepath = os.path.join(audio_dir, filename)

        _, sig_mix = wavfile.read(filepath)
        f_mix, psd_mix = welch(sig_mix, sr, nperseg=4096)
        psd_mix_db = 10 * np.log10(psd_mix + 1e-10)

        # استخراج قدرت سیگنال ترکیبی در فرکانس شاخص
        mix_power_at_peak = psd_mix_db[peak_idx]
        snr_db = mix_power_at_peak - noise_floor_at_peak

        detected = "Yes" if snr_db >= detection_threshold_db else "No"
        results.append({
            'Target_Ratio': alpha,
            'Peak_Power_dB': mix_power_at_peak,
            'SNR_dB': snr_db,
            'Detected_by_FFT': detected
        })

        # رسم طیف فرکانسی فقط برای چند نسبت مهم برای شلوغ نشدن نمودار
        if alpha in [0.0, 0.2, 0.4, 0.7, 1.0]:
            plt.plot(f_mix[valid_idx], psd_mix_db[valid_idx], label=f'Target: {t_pct}%')

    # ذخیره نمودار طیف فرکانسی (FFT Spectrum)
    plt.axvline(x=target_peak_freq, color='red', linestyle='--', linewidth=1.5, label='Target Peak Freq')
    plt.title('Classical Sonar FFT Analysis: Peak Masking by Ocean Noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FFT_Spectrum_Masking.png"), dpi=300)
    plt.close()

    # 4. ذخیره و رسم نمودار عملکرد سونار کلاسیک
    df_results = pd.DataFrame(results)
    df_results.to_excel(os.path.join(output_dir, "Classical_Sonar_Detection_Results.xlsx"), index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x='Target_Ratio', y='SNR_dB', marker='o', color='purple', linewidth=3)
    plt.axhline(y=detection_threshold_db, color='red', linestyle='--', linewidth=2, label='Detection Threshold (3 dB)')
    plt.gca().invert_xaxis()

    # پیدا کردن نقطه کوری سونار کلاسیک
    blind_spot = df_results[df_results['SNR_dB'] < detection_threshold_db]['Target_Ratio'].max()
    if not pd.isna(blind_spot):
        plt.axvline(x=blind_spot, color='black', linestyle=':', linewidth=2)
        plt.text(blind_spot + 0.05, detection_threshold_db + 2, f'Classical Sonar\nBlind Spot: {blind_spot}',
                 fontweight='bold')

    plt.title('Classical Passive Sonar Detection Performance')
    plt.xlabel('Target Signal Ratio (1.0 = Pure Target, 0.0 = Pure Noise)')
    plt.ylabel('Signal-to-Noise Ratio at Peak Frequency (dB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Classical_Sonar_Performance.png"), dpi=300)
    plt.close()

    print(f"\nSuccess! Classical sonar simulation saved to: {output_dir}")
    print("Check 'Classical_Sonar_Performance.png' to see exactly where the FFT fails.")


code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
simulate_classical_passive_sonar(code_directory)