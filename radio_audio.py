import asyncio
import json
import os
import threading
import urllib.request
import urllib.error
from typing import Optional
import numpy as np
import pygame
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample
import config


class F1RadioAudioEngine:
    """
    Menangani sintesis suara GP Lambiase (ElevenLabs REST API + auto-fallback Edge-TTS),
    memproses efek DSP Radio Intercom F1 (Bandpass 300-3400Hz + Soft Clipping + Static Noise + Beep In/Out),
    dan memutarnya secara non-blocking melalui pygame.mixer.
    Kompatibel penuh dengan Python 3.14+.
    """

    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.playback_lock = threading.Lock()
        self._init_mixer()
        self._ensure_beep_files()

    def _init_mixer(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=1024)
        except Exception as e:
            print(f"[Audio Warn] Gagal menginisialisasi pygame.mixer: {e}")

    def _generate_f1_beep(self, freqs: list, duration_sec: float = 0.14) -> np.ndarray:
        """Membuat gelombang nada F1 FIA Team Radio beep (dual-tone + envelope)."""
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        wave = np.zeros_like(t, dtype=np.float32)
        for f in freqs:
            wave += 0.45 * np.sin(2 * np.pi * f * t)
        fade_len = int(0.015 * self.sample_rate)
        if fade_len > 0 and len(wave) > 2 * fade_len:
            wave[:fade_len] *= np.linspace(0, 1, fade_len)
            wave[-fade_len:] *= np.linspace(1, 0, fade_len)
        return (wave * 32767).astype(np.int16)

    def _ensure_beep_files(self):
        """Membuat file beep_in.wav dan beep_out.wav jika belum ada."""
        if not os.path.exists("beep_in.wav"):
            beep_in = self._generate_f1_beep([1850.0, 2400.0], duration_sec=0.14)
            wavfile.write("beep_in.wav", self.sample_rate, beep_in)

        if not os.path.exists("beep_out.wav"):
            beep_out = self._generate_f1_beep([1550.0, 2050.0], duration_sec=0.16)
            wavfile.write("beep_out.wav", self.sample_rate, beep_out)

    def play_ptt_beep_in(self):
        """Memutar bunyi klik radio saat driver menekan tombol PTT."""
        try:
            if pygame.mixer.get_init() and os.path.exists("beep_in.wav"):
                sound = pygame.mixer.Sound("beep_in.wav")
                sound.set_volume(0.45)
                sound.play()
        except Exception:
            pass

    def _bandpass_filter(self, data: np.ndarray, lowcut: float = 300.0, highcut: float = 3400.0, order: int = 4) -> np.ndarray:
        """Scipy Butterworth Bandpass Filter (300 Hz - 3400 Hz) khas walkie-talkie helm F1."""
        nyq = 0.5 * self.sample_rate
        low = max(0.01, lowcut / nyq)
        high = min(0.99, highcut / nyq)
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data)

    def _is_excited_text(self, text: str) -> bool:
        # // Deteksi apakah kalimat merupakan selebrasi kemenangan/podium/pole/simply lovely
        u = text.upper()
        return ("!" in text) and any(
            w in u for w in ("YES!", "GET IN THERE", "POLE POSITION", "CHAMPION", "MEGA", "PODIUM", "SIMPLY LOVELY", "HAHA", "HA-HA", "WOO")
        )

    def _apply_f1_radio_dsp(self, pcm_float: np.ndarray, is_excited: bool = False) -> np.ndarray:
        """
        Menerapkan rantai DSP Radio F1:
        1. Bandpass filter (300 Hz - 3400 Hz)
        2. Pre-emphasis & Soft saturation (dikurangi saat is_excited=True agar dinamika teriakan tetap lepas)
        3. Derau statis (background carrier hiss) tipis
        4. Penggabungan Beep In + Suara + Beep Out
        """
        filtered = self._bandpass_filter(pcm_float, 280.0 if is_excited else 300.0, 3600.0 if is_excited else 3400.0, order=4)

        peak = np.max(np.abs(filtered))
        if peak > 1e-5:
            filtered = filtered / peak

        # // Jika selebrasi, gunakan drive lebih ringan (1.35) agar intonasi tinggi tidak gepeng terkompresi
        drive_amount = 1.35 if is_excited else 2.0
        driven = np.tanh(filtered * drive_amount)

        noise = np.random.normal(0, 0.015, size=driven.shape)
        noise_filtered = self._bandpass_filter(noise, 400.0, 3200.0, order=2)
        radio_voice = (driven * 0.92) + noise_filtered

        max_val = np.max(np.abs(radio_voice))
        if max_val > 1e-5:
            radio_voice = (radio_voice / max_val) * (0.96 if is_excited else 0.90)
        voice_int16 = (radio_voice * 32767).astype(np.int16)

        _, beep_in_data = wavfile.read("beep_in.wav")
        _, beep_out_data = wavfile.read("beep_out.wav")
        pause = np.zeros(int(self.sample_rate * 0.05), dtype=np.int16)

        combined = np.concatenate([beep_in_data, pause, voice_int16, pause, beep_out_data])
        return combined

    def _decode_mp3_file_to_pcm(self, mp3_path: str) -> Optional[np.ndarray]:
        # // Dekode file MP3 menjadi array float32 mono pada self.sample_rate menggunakan pygame.mixer
        if not os.path.exists(mp3_path):
            return None
        self._init_mixer()
        sound = pygame.mixer.Sound(mp3_path)
        raw_bytes = sound.get_raw()
        mixer_freq, _, mixer_channels = pygame.mixer.get_init()

        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if mixer_channels > 1:
            samples = samples.reshape(-1, mixer_channels).mean(axis=1)

        if mixer_freq != self.sample_rate:
            target_len = int(len(samples) * float(self.sample_rate) / float(mixer_freq))
            samples = resample(samples, target_len)
        return samples

    def _synthesize_elevenlabs_pcm(self, text: str) -> Optional[np.ndarray]:
        """
        Memanggil ElevenLabs REST API dengan format mp3_44100_128 (didukung penuh oleh akun Free Tier,
        tidak terkena HTTP 403 seperti pcm_24000).
        Saat is_excited=True (Bravo / Simply Lovely / WDC), menggunakan Voice ID ekspresif & style=1.0.
        """
        api_key = config.ELEVENLABS_API_KEY.strip()
        voice_id = config.GP_VOICE_ID.strip()

        if not api_key or api_key == "your_elevenlabs_api_key_here" or not voice_id or voice_id == "your_gp_lambiase_voice_id_here":
            return None

        is_excited = self._is_excited_text(text)
        # // Gunakan suara ekspresif (Charlie) khusus saat selebrasi jika user memakai Voice ID Daniel yang datar
        active_voice_id = "IKne3meq5aSn9XLyUdCD" if (is_excited and voice_id == "onwK4e9ZLuTAKqWW03F9") else voice_id

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{active_voice_id}?output_format=mp3_44100_128"
        payload = json.dumps({
            "text": text,
            "model_id": "eleven_multilingual_v2" if is_excited else "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.12 if is_excited else 0.50,
                "similarity_boost": 0.80,
                "style": 1.0 if is_excited else 0.25,
                "use_speaker_boost": True
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            },
            method="POST"
        )

        mp3_temp_path = "gp_elevenlabs_raw.mp3"
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                mp3_bytes = resp.read()
                if not mp3_bytes:
                    return None
                with open(mp3_temp_path, "wb") as f:
                    f.write(mp3_bytes)
                return self._decode_mp3_file_to_pcm(mp3_temp_path)
        except Exception:
            return None

    async def _edge_tts_save_mp3(self, text: str, mp3_path: str):
        import edge_tts
        is_excited = self._is_excited_text(text)
        # // Naikkan pitch (+15Hz), rate (+16%), dan volume (+25%) pada Edge-TTS saat selebrasi agar tidak datar
        rate_val = "+16%" if is_excited else "+6%"
        pitch_val = "+15Hz" if is_excited else "+0Hz"
        vol_val = "+25%" if is_excited else "+0%"
        communicate = edge_tts.Communicate(text, voice=config.FALLBACK_VOICE, rate=rate_val, pitch=pitch_val, volume=vol_val)
        await communicate.save(mp3_path)

    def _synthesize_edge_tts_pcm(self, text: str) -> Optional[np.ndarray]:
        """
        Auto-fallback menggunakan Edge-TTS ("en-GB-RyanNeural") dan mendekode MP3 ke PCM array
        menggunakan pygame.mixer.Sound.get_raw().
        """
        mp3_path = config.ENGINEER_RAW_AUDIO_PATH
        try:
            asyncio.run(self._edge_tts_save_mp3(text, mp3_path))
            if not os.path.exists(mp3_path):
                return None

            self._init_mixer()
            sound = pygame.mixer.Sound(mp3_path)
            raw_bytes = sound.get_raw()
            mixer_freq, mixer_format, mixer_channels = pygame.mixer.get_init()

            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if mixer_channels > 1:
                samples = samples.reshape(-1, mixer_channels).mean(axis=1)

            if mixer_freq != self.sample_rate:
                target_len = int(len(samples) * float(self.sample_rate) / float(mixer_freq))
                samples = resample(samples, target_len)

            return samples
        except Exception as e:
            print(f"[Edge-TTS Error] Gagal menghasilkan suara fallback: {e}")
            return None

    def speak_radio(self, text: str, on_source_resolved=None):
        """
        Menjalankan sintesis suara + filter radio F1 + pemutaran audio secara non-blocking di thread terpisah.
        """
        if not text or not text.strip():
            return

        def _worker():
            with self.playback_lock:
                engine_used = "ElevenLabs (GP Voice)"
                pcm_float = self._synthesize_elevenlabs_pcm(text)

                if pcm_float is None:
                    engine_used = f"Edge-TTS ({config.FALLBACK_VOICE})"
                    pcm_float = self._synthesize_edge_tts_pcm(text)

                if on_source_resolved:
                    on_source_resolved(engine_used)

                if pcm_float is None or len(pcm_float) == 0:
                    return

                is_exc = self._is_excited_text(text)
                final_wav_data = self._apply_f1_radio_dsp(pcm_float, is_excited=is_exc)
                wavfile.write(config.ENGINEER_RADIO_WAV_PATH, self.sample_rate, final_wav_data)

                try:
                    self._init_mixer()
                    radio_sound = pygame.mixer.Sound(config.ENGINEER_RADIO_WAV_PATH)
                    radio_sound.set_volume(0.95)
                    channel = radio_sound.play()
                    while channel is not None and channel.get_busy():
                        pygame.time.wait(50)
                except Exception as e:
                    print(f"[Playback Error] Gagal memutar audio radio: {e}")

        threading.Thread(target=_worker, daemon=True).start()
