from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.integrate import trapezoid
from moviepy.editor import VideoFileClip
import os
import tempfile
import uuid
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

# Log directory information
logging.debug(f"Current directory: {current_dir}")
logging.debug(f"Template directory: {template_dir}")

# Verify template directory exists
if not os.path.exists(template_dir):
    logging.info(f"Creating template directory: {template_dir}")
    os.makedirs(template_dir)

app = Flask(__name__, template_folder=template_dir)

# Create default index.html if it doesn't exist
index_path = os.path.join(template_dir, 'index.html')
if not os.path.exists(index_path):
    logging.info(f"Creating default index.html at {index_path}")
    with open(index_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>PulseTrack HRV Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #4a6fa5; }
        .container { max-width: 600px; margin: 0 auto; }
        .btn {
            background: #4a6fa5;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PulseTrack HRV Analyzer</h1>
        <p>Heart Rate Variability Analysis Application</p>
        <p><strong>Important:</strong> Please replace this file with the actual frontend HTML file</p>
        <button class="btn" onclick="location.href='/start'">Start Application</button>
    </div>
</body>
</html>""")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_app():
    """Main application route that serves the actual frontend"""
    # For now, we'll serve a basic page - you should replace this with your actual frontend
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PulseTrack HRV Analyzer</title>
        <script>
            alert("Application started! Please replace this with your actual frontend code.");
        </script>
    </head>
    <body>
        <h1>PulseTrack is Running</h1>
        <p>This is a placeholder. You should see your actual application interface here.</p>
        <p>To fix this:</p>
        <ol>
            <li>Create a file called 'index.html' in the 'templates' folder</li>
            <li>Copy your frontend HTML code into that file</li>
            <li>Restart the application</li>
        </ol>
    </body>
    </html>
    """

def remove_audio(video_path):
    """Remove audio from video and return path to temp video file without audio"""
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "no_audio_video.mp4")
        
        video = VideoFileClip(video_path)
        video = video.without_audio()
        video.write_videofile(output_path, codec="libx264", audio_codec=None)
        
        return output_path
    except Exception as e:
        logging.error(f"Failed to remove audio: {str(e)}")
        return video_path

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_heart_rate_from_video(video_path):
    clean_video_path = remove_audio(video_path)
    
    cap = cv2.VideoCapture(clean_video_path)

    if not cap.isOpened():
        logging.error(f"Cannot open video file: {clean_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        logging.error("Video FPS is zero or invalid.")
        return None

    logging.info(f"Video FPS: {fps:.2f}")
    
    red_avg_values = []
    frame_times = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame is None or frame.size == 0:
            continue

        h, w = frame.shape[:2]
        roi = frame[h//3:h*2//3, w//3:w*2//3]

        red_channel = roi[:, :, 2]
        red_mean = np.mean(red_channel)
        red_avg_values.append(red_mean)
        frame_times.append(frame_count / fps)

    cap.release()

    if clean_video_path != video_path:
        try:
            os.remove(clean_video_path)
            os.rmdir(os.path.dirname(clean_video_path))
        except Exception as e:
            logging.error(f"Error cleaning temp files: {str(e)}")

    if len(red_avg_values) < 10:
        logging.error("Not enough data for heart rate analysis.")
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
        logging.error("Invalid time duration.")
        return None

    bpm = (num_beats / duration) * 60.0

    # HRV Analysis
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

        fs_interp = 4
        t_interp = np.linspace(times[peaks][1], times[peaks][-1], int((times[peaks][-1] - times[peaks][1]) * fs_interp))
        rr_interp = np.interp(t_interp, times[peaks][1:], rr_intervals)

        nperseg = min(256, len(rr_interp))
        freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg)

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
                anxiety_risk = "⚠️ Mild Stress"
            else:
                anxiety_risk = "🚨 High Anxiety Risk"
        else:
            anxiety_risk = "⚠️ HF power too low"
    else:
        logging.warning("Not enough peaks to calculate HRV & LF/HF")

    # Generate plot
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
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "bpm": bpm,
        "rmssd_ms": rmssd_ms,
        "sdnn": sdnn,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "anxiety_risk": anxiety_risk,
        "fps": fps,
        "plot": plot_data,
        "pulse_data": filtered_signal.tolist(),
        "times": times.tolist()
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
    video_file.save(temp_video_path)
    
    try:
        results = get_heart_rate_from_video(temp_video_path)
        if results is None:
            return jsonify({"error": "Analysis failed"}), 500
            
        return jsonify(results)
    except Exception as e:
        logging.exception("Error during analysis")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(temp_video_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logging.error(f"Error cleaning temp files: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)