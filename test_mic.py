import os
import sys
import time
import threading
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import pygame

SAMPLE_RATE = 16000
TEST_FILE = "test_mic_playback.wav"
MIC_BOOST_GAIN = 6.0  # Digital Pre-Amp Booster 6x lipat untuk mic yang kecil


def list_input_devices():
    """Menampilkan daftar semua perangkat input (mikrofon) yang tersedia."""
    devices = sd.query_devices()
    input_devs = []
    default_in_idx = sd.default.device[0]

    print("\n📋 DAFTAR MIKROFON (INPUT DEVICES) YANG TERDETEKSI DI WINDOWS:")
    print("---------------------------------------------------------------------------")
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", "Unknown")
            is_def = " (DEFAULT WINDOWS)" if idx == default_in_idx else ""
            warn = " ⚠️ [MIC STIK PS4 - BIASANYA 0%!]" if "wireless controller" in name.lower() else ""
            print(f"  [m{idx}] {name}{is_def}{warn}")
            input_devs.append((idx, name))
    print("---------------------------------------------------------------------------")
    return input_devs


def pick_initial_mic():
    """Otomatis hindari mic kosong 'Wireless Controller' milik stik PS4."""
    try:
        default_in = sd.query_devices(kind="input")
        if "wireless controller" not in default_in["name"].lower():
            return None, default_in["name"]
    except Exception:
        pass

    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", "")
            if "wireless controller" not in name.lower() and "mapper" not in name.lower():
                return idx, name
    return None, "Default Input"


def save_mic_to_env(device_idx: int):
    """Menyimpan MIC_DEVICE_INDEX pilihan user ke dalam file .env secara otomatis."""
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("MIC_DEVICE_INDEX="):
            new_lines.append(f"MIC_DEVICE_INDEX={device_idx}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"\nMIC_DEVICE_INDEX={device_idx}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    if os.name == "nt":
        os.system("")

    print("===========================================================================")
    print("      F1 24 INTERACTIVE MICROPHONE (AUTO-BOOST 6X) & PS4 PAD TESTER        ")
    print("===========================================================================")

    list_input_devices()
    current_dev_idx, current_dev_name = pick_initial_mic()

    # Inisialisasi Controller PS4 / Xbox
    pygame.init()
    pygame.joystick.init()
    joysticks = []
    for i in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(i)
        js.init()
        joysticks.append(js)

    if joysticks:
        print(f"🎮 Stik Terdeteksi : {joysticks[0].get_name()} ({joysticks[0].get_numbuttons()} tombol)")
    else:
        print("🎮 Stik Terdeteksi : Belum ada stik terdeteksi")

    print("\n💡 KONTROL CEPAT:")
    print("  👉 Ketik '1' lalu [ENTER]    = KELUAR (EXIT)")
    print("  👉 Ketik 'p' lalu [ENTER]    = Putar ulang suaramu (sudah di-boost kencang!)")
    print("  👉 Ketik 'm0', 'm1', dst     = Ganti mikrofon (misal ketik m2 lalu Enter)")
    print("  👉 Ketik 'list' lalu [ENTER] = Lihat ulang daftar mikrofon\n")

    running = True
    restart_stream = False
    current_pct = 0
    max_pct = 0
    ring_buffer = deque(maxlen=int(SAMPLE_RATE * 4 / 1024))  # Simpan 4 detik terakhir

    def audio_callback(indata, frames, time_info, status):
        nonlocal current_pct, max_pct
        # Terapkan Digital Pre-Amp Boost 6x agar mic kecil langsung naik kencang
        boosted = np.clip(indata.astype(np.float32) * MIC_BOOST_GAIN, -32768, 32767).astype(np.int16)
        peak = float(np.max(np.abs(boosted))) / 32768.0
        pct = min(100, int((peak ** 0.6) * 100))
        current_pct = pct
        if pct > max_pct:
            max_pct = pct
        ring_buffer.append(boosted)

    def input_listener_thread():
        nonlocal running, current_dev_idx, current_dev_name, restart_stream, max_pct
        while running:
            try:
                cmd = sys.stdin.readline()
                if not cmd:
                    break
                cmd = cmd.strip().lower()
                # Ketik '1' (atau 'exit') langsung keluar!
                if cmd in ("1", "exit", "quit", "q"):
                    running = False
                    break
                elif cmd == "list":
                    print("\n")
                    list_input_devices()
                elif cmd in ("p", "play"):
                    if len(ring_buffer) == 0:
                        print("\n⚠️ Belum ada suara yang terekam.")
                    else:
                        print("\n🔊 Memutar rekaman 4 detik terakhir (dengan Auto-Boost + Normalizer)...")
                        audio_arr = np.concatenate(list(ring_buffer), axis=0)
                        peak_val = float(np.max(np.abs(audio_arr)))
                        if peak_val > 20:
                            norm_arr = np.clip((audio_arr.astype(np.float32) / peak_val) * 30000.0, -32768, 32767).astype(np.int16)
                        else:
                            norm_arr = audio_arr
                        wavfile.write(TEST_FILE, SAMPLE_RATE, norm_arr)
                        try:
                            if not pygame.mixer.get_init():
                                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
                            snd = pygame.mixer.Sound(TEST_FILE)
                            ch = snd.play()
                            while ch.get_busy():
                                pygame.time.wait(50)
                            print("✅ Selesai memutar rekaman!\n")
                        except Exception as e:
                            print(f"❌ Gagal memutar audio: {e}\n")
                elif cmd.startswith("m") and cmd[1:].isdigit():
                    new_idx = int(cmd[1:])
                    devs = sd.query_devices()
                    if 0 <= new_idx < len(devs) and devs[new_idx].get("max_input_channels", 0) > 0:
                        current_dev_idx = new_idx
                        current_dev_name = devs[new_idx]["name"]
                        max_pct = 0
                        save_mic_to_env(new_idx)
                        restart_stream = True
                        print(f"\n✅ Berhasil ganti ke Mic [m{new_idx}]: {current_dev_name} (Tersimpan ke .env!)\n")
                    else:
                        print(f"\n❌ ID [m{new_idx}] bukan perangkat mikrofon yang valid.\n")
            except Exception:
                break

    t_in = threading.Thread(target=input_listener_thread, daemon=True)
    t_in.start()

    while running:
        restart_stream = False
        try:
            stream = sd.InputStream(
                device=current_dev_idx,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=1024,
                dtype="int16",
                callback=audio_callback
            )
            stream.start()
        except Exception as e:
            print(f"\n❌ Gagal membuka Mic ({current_dev_name}): {e}")
            print("Ketik 'm<nomor>' untuk pilih mic lain, atau ketik '1' untuk keluar.")
            while running and not restart_stream:
                time.sleep(0.2)
            continue

        while running and not restart_stream:
            pressed_btns = []
            try:
                pygame.event.pump()
                for js in joysticks:
                    for b in range(js.get_numbuttons()):
                        if js.get_button(b):
                            pressed_btns.append(f"Btn {b}")
            except Exception:
                pass

            btn_str = ",".join(pressed_btns) if pressed_btns else "None"
            bars = int(current_pct / 4)  # 25 balok
            bar_visual = "█" * bars + "░" * (25 - bars)

            short_mic_name = (current_dev_name[:20] + "..") if len(current_dev_name) > 22 else current_dev_name
            status_line = (
                f"\r🎙️ Mic[{short_mic_name}] "
                f"[{bar_visual}] {current_pct:>3}% (Peak:{max_pct:>3}%) | 🎮 Pad:[{btn_str:<7}] | Ketik '1'+Enter=Exit, 'p'+Enter=Play "
            )
            sys.stdout.write(status_line)
            sys.stdout.flush()
            time.sleep(0.05)

        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    print("\n\n👋 Keluar dari Mic Tester!")


if __name__ == "__main__":
    main()
