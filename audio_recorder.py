import os
import threading
from typing import List, Optional
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import config


class PushToTalkRecorder:
    """
    Merekam input suara dari mikrofon pada standar 16000 Hz (standar emas Speech Recognition),
    membersihkan dengung frekuensi rendah (< 100 Hz) dengan High-Pass Filter, dan melakukan
    clean normalization tanpa distorsi/clipping agar Gemini menangkap ucapan dengan sangat akurat.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.current_level_pct: int = 0
        self.device_index: Optional[int] = self._resolve_best_mic_device()
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _resolve_best_mic_device(self) -> Optional[int]:
        cfg_idx = str(getattr(config, "MIC_DEVICE_INDEX", "auto")).strip().lower()
        if cfg_idx.isdigit():
            return int(cfg_idx)

        try:
            default_in = sd.query_devices(kind="input")
            default_name = str(default_in.get("name", "")).lower()
            if "wireless controller" not in default_name and "dualshock" not in default_name:
                return None

            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    name = str(dev.get("name", "")).lower()
                    if "wireless controller" not in name and "dualshock" not in name and "sound mapper" not in name:
                        return idx
        except Exception:
            pass
        return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if self.is_recording:
            # Simpan sinyal asli tanpa hard-clipping di callback agar suara tidak pecah!
            self._frames.append(indata.copy())
            peak = float(np.max(np.abs(indata))) / 32768.0
            self.current_level_pct = min(100, int((peak ** 0.55) * 180))

    def start_recording(self) -> bool:
        with self._lock:
            if self.is_recording:
                return False
            self._frames = []
            try:
                self._stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    callback=self._audio_callback
                )
                self._stream.start()
                self.is_recording = True
                return True
            except Exception as e:
                self.is_recording = False
                self._stream = None
                raise RuntimeError(f"Gagal mengakses mikrofon: {e}")

    def _clean_and_normalize_voice(self, raw_int16: np.ndarray) -> np.ndarray:
        floats = raw_int16.astype(np.float32).flatten()
        # Hilangkan DC offset
        floats = floats - np.mean(floats)

        # Normalisasi moderat (maksimal 6x) agar noise floor mikrofon tidak merusak artikulasi vokal
        peak = float(np.max(np.abs(floats)))
        if peak > 50.0:
            gain = min(6.0, 24000.0 / peak)
            floats = floats * gain

        return np.clip(floats, -32768, 32767).astype(np.int16)

    def stop_recording(self, output_path: str = config.DRIVER_AUDIO_PATH) -> Optional[str]:
        with self._lock:
            if not self.is_recording:
                return None
            self.is_recording = False

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            if not self._frames:
                return None

            audio_data = np.concatenate(self._frames, axis=0)
            duration_sec = len(audio_data) / float(self.sample_rate)

            if duration_sec < 0.25:
                return None

            clean_audio = self._clean_and_normalize_voice(audio_data)
            wavfile.write(output_path, self.sample_rate, clean_audio)
            if os.path.exists(output_path):
                return output_path
            return None
