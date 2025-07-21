import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.integrate import trapezoid
from moviepy.editor import VideoFileClip
import os
import tempfile

def remove_audio(video_path):
    """Remove audio from video and return path to temp video file without audio"""
    try:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "no_audio_video.mp4")
        
        # Load video and remove audio
        video = VideoFileClip(video_path)
        video = video.without_audio()
        video.write_videofile(output_path, codec="libx264", audio_codec=None)
        
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to remove audio: {str(e)}")
        return video_path  # Return original if fails

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_heart_rate_from_video(video_path):
    # Remove audio first
    clean_video_path = remove_audio(video_path)
    
    cap = cv2.VideoCapture(clean_video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {clean_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        print("[ERROR] Video FPS is zero or invalid.")
        return None

    print(f"[INFO] Video FPS: {fps:.2f}")
    
    red_avg_values = []
    frame_times = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame is None or frame.shape[0] == 0:
            continue

        h, w = frame.shape[:2]
        roi = frame[h//3:h*2//3, w//3:w*2//3]

        red_channel = roi[:, :, 2]
        red_mean = np.mean(red_channel)
        red_avg_values.append(red_mean)
        frame_times.append(frame_count / fps)

    cap.release()

    # Clean up temporary file
    if clean_video_path != video_path:
        try:
            os.remove(clean_video_path)
            os.rmdir(os.path.dirname(clean_video_path))
        except:
            pass

    if len(red_avg_values) < 10:
        print("[ERROR] Not enough data for heart rate analysis.")
        return None

    red_signal = np.array(red_avg_values)
    times = np.array(frame_times)

    red_signal = red_signal - np.mean(red_signal)
    red_signal = red_signal / np.max(np.abs(red_signal))

    filtered_signal = butter_bandpass_filter(red_signal, 0.8, 3.0, fs=fps, order=4)
    peaks, _ = find_peaks(filtered_signal, distance=fps / 2.5)

    duration = times[-1] - times[0]
    num_beats = len(peaks)
    if duration == 0:
        print("[ERROR] Invalid time duration.")
        return None

    bpm = (num_beats / duration) * 60.0

    # ✅ HRV + LF/HF Analysis
    rmssd_ms = None
    sdnn = None
    lf_hf_ratio = None
    lf_power = hf_power = None
    anxiety_risk = "Not enough peaks"

    if len(peaks) >= 5:
        rr_intervals = np.diff(times[peaks])
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        sdnn = np.std(rr_intervals)
        rmssd_ms = rmssd * 1000

        # Interpolation for evenly spaced RR intervals
        fs_interp = 4  # Hz
        t_interp = np.linspace(times[peaks][1], times[peaks][-1], int((times[peaks][-1] - times[peaks][1]) * fs_interp))
        rr_interp = np.interp(t_interp, times[peaks][1:], rr_intervals)

        # Welch's PSD
        nperseg = min(256, len(rr_interp))
        freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg)

        # LF and HF bands
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

        lf_power = trapezoid(psd[lf_mask], freqs[lf_mask])
        hf_power = trapezoid(psd[hf_mask], freqs[hf_mask])

        if hf_power > 0:
            lf_hf_ratio = lf_power / hf_power

            if lf_hf_ratio < 1.0:
                anxiety_risk = "✅ Relaxed"
            elif lf_hf_ratio < 2.5:
                anxiety_risk = "⚠️  Mild Stress"
            else:
                anxiety_risk = "🚨 High Anxiety Risk"
        else:
            anxiety_risk = "⚠️ HF power too low to compute LF/HF ratio"

        print("🧠 HRV Metrics:")
        print(f" - RMSSD       : {rmssd_ms:.2f} ms")
        print(f" - SDNN        : {sdnn:.2f} s")
        print(f" - LF Power    : {lf_power:.4f}")
        print(f" - HF Power    : {hf_power:.4f}")
        print(f" - LF/HF Ratio : {lf_hf_ratio:.2f}")
        print(f" - Anxiety Risk: {anxiety_risk}")
    else:
        print("⚠️ Not enough peaks to calculate HRV & LF/HF")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, filtered_signal, label='Filtered PPG Signal')
    plt.plot(times[peaks], filtered_signal[peaks], 'ro', label='Peaks')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    title = f"HR: {bpm:.1f} BPM"
    if rmssd_ms:
        title += f" | RMSSD: {rmssd_ms:.1f} ms"
    if lf_hf_ratio:
        title += f" | LF/HF: {lf_hf_ratio:.2f}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bpm

def ask_user_to_upload():
    Tk().withdraw()
    print("📁 Please select your finger-on-camera video file...")
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    return file_path

# 🔄 Run
video_path = ask_user_to_upload()
if video_path:
    bpm = get_heart_rate_from_video(video_path)
    if bpm is not None:
        print(f"✅ Estimated Heart Rate: {bpm:.2f} BPM")
    else:
        print("❌ Heart rate estimation failed.")
else:
    print("⚠️ No file selected. Exiting.")