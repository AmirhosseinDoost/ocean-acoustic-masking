import os
import numpy as np
import pandas as pd
from scipy.io import wavfile


def export_mixed_audio_files(base_dir, code_dir, duration_sec=30):
    # ایجاد پوشه اختصاصی برای فایل‌های صوتی خروجی
    output_audio_dir = os.path.join(code_dir, "Result_5_Signal_Mixing", "Result_5_Mixed_Audio_Files")
    os.makedirs(output_audio_dir, exist_ok=True)

    original_excel = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(original_excel)

    # انتخاب کلاس هدف و نویز محیطی
    target_class = 'Passengers'
    noise_class = 'Natural ambient noise'

    target_rows = df[df['Type'] == target_class]
    noise_rows = df[df['Type'] == noise_class]

    if target_rows.empty or noise_rows.empty:
        print("Error: Target or Noise class not found.")
        return

    target_file = target_rows['Filename'].dropna().iloc[0]
    noise_file = noise_rows['Filename'].dropna().iloc[0]

    target_path = os.path.join(base_dir, target_file)
    noise_path = os.path.join(base_dir, noise_file)

    # خواندن فایل‌های صوتی
    sr_target, sig_target = wavfile.read(target_path)
    sr_noise, sig_noise = wavfile.read(noise_path)

    if sig_target.ndim > 1: sig_target = sig_target[:, 0]
    if sig_noise.ndim > 1: sig_noise = sig_noise[:, 0]

    min_sr = min(sr_target, sr_noise)

    # محدود کردن به طول مشخص (مثلا ۳۰ ثانیه) برای هماهنگی حجم فایل‌ها
    max_samples = min(int(duration_sec * min_sr), len(sig_target), len(sig_noise))
    sig_target = sig_target[:max_samples].astype(float)
    sig_noise = sig_noise[:max_samples].astype(float)

    # نرمال‌سازی RMS برای ترکیب عادلانه توان صوتی
    def rms_normalize(signal):
        rms = np.sqrt(np.mean(signal ** 2))
        return signal / rms if rms > 0 else signal

    sig_target_norm = rms_normalize(sig_target)
    sig_noise_norm = rms_normalize(sig_noise)

    # نسبت‌های ترکیب (از 0.0 تا 1.0 با گام‌های 0.1)
    mix_ratios = np.round(np.arange(0.0, 1.1, 0.1), 1)

    print("Generating and exporting mixed audio files...")

    for alpha in mix_ratios:
        # ترکیب خطی سیگنال‌ها
        mixed_signal = (alpha * sig_target_norm) + ((1.0 - alpha) * sig_noise_norm)

        # تبدیل به فرمت استاندارد 16-bit PCM برای ذخیره صحیح به صورت wav
        mixed_signal_normalized = mixed_signal / np.max(np.abs(mixed_signal))  # جلوگیری از کپی‌شدن یا دیستورت صدا
        audio_int16 = (mixed_signal_normalized * 32767).astype(np.int16)

        # نام‌گذاری دقیق فایل بر اساس درصد ترکیب
        target_pct = int(alpha * 100)
        noise_pct = int((1.0 - alpha) * 100)
        filename = f"Mixed_{target_pct}pct_Target_{noise_pct}pct_Noise.wav"
        filepath = os.path.join(output_audio_dir, filename)

        wavfile.write(filepath, min_sr, audio_int16)
        print(f"Exported: {filename}")

    print(f"\nSuccess! All 11 mixed audio files are saved in: {output_audio_dir}")


# مسیرهای اجرا
base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

export_mixed_audio_files(base_directory, code_directory)