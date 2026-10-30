import os
from dotenv import load_dotenv

# Muat variabel dari file .env
load_dotenv()

# Konfigurasi Jaringan UDP F1 24
UDP_IP: str = os.getenv("UDP_IP", "127.0.0.1")
UDP_PORT: int = int(os.getenv("UDP_PORT", "20777"))

# Konfigurasi API & Suara
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
GP_VOICE_ID: str = os.getenv("GP_VOICE_ID", "")
FALLBACK_VOICE: str = os.getenv("FALLBACK_VOICE", "en-GB-RyanNeural")

# Konfigurasi Kontrol & File Audio
PTT_KEY: str = os.getenv("PTT_KEY", "v")
# Tombol di stik PS4 / Xbox: bisa diisi "share" (otomatis deteksi Xbox View/Back [6] & PS4 Share [4/8]), atau angka spesifik misal "6"
PTT_CONTROLLER_BUTTON: str = os.getenv("PTT_CONTROLLER_BUTTON", "share")
# Index mikrofon (kosongkan / "auto" agar otomatis menghindari mic kosong stik PS4 Wireless Controller)
MIC_DEVICE_INDEX: str = os.getenv("MIC_DEVICE_INDEX", "auto")
SAMPLE_RATE: int = 24000
DRIVER_AUDIO_PATH: str = "driver_radio.wav"
ENGINEER_RAW_AUDIO_PATH: str = "gp_raw_response.mp3"
ENGINEER_RADIO_WAV_PATH: str = "gp_radio_final.wav"

