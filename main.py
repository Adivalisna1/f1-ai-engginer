1
import os
import sys
import time
import threading
from collections import deque
import keyboard
import pygame
import config
from f1_telemetry import F1TelemetryListener
from audio_recorder import PushToTalkRecorder
from radio_audio import F1RadioAudioEngine
from ai_brain import RaceEngineerBrain


# ANSI Escape Codes untuk Dashboard CLI Berwarna
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class GPLambiaseRaceEngineerApp:
    def __init__(self):
        self.telemetry = F1TelemetryListener(config.UDP_IP, config.UDP_PORT)
        self.recorder = PushToTalkRecorder()
        self.radio_audio = F1RadioAudioEngine()
        self.brain = RaceEngineerBrain()

        self.running = True
        self.ptt_active = False
        self.processing_ai = False
        self.last_proactive_radio_time = 0.0
        self.last_tts_engine = "Standby"
        self.controller_name = "No Controller"
        self.joysticks = []
        self.radio_logs = deque(maxlen=6)
        self._log_lock = threading.Lock()

        # Aktifkan dukungan ANSI color di terminal Windows
        if os.name == "nt":
            os.system("")

        self._init_controllers()
        self._start_exit_key_listener()
        self.add_log("SYSTEM", f"Pit-Wall Telemetry Listener bound to UDP {config.UDP_IP}:{config.UDP_PORT}")
        self.add_log("GP", f"Radio check, loud and clear. Hold [PS4 SHARE] or ['{config.PTT_KEY.upper()}'] to talk.")

    def _start_exit_key_listener(self):
        """Memungkinkan user mengetik '1' atau 'exit' + Enter di terminal untuk keluar dengan mudah."""
        def _stdin_waiter():
            while self.running:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    cmd = line.strip().lower()
                    if cmd in ("1", "exit", "quit", "q"):
                        self.running = False
                        break
                except Exception:
                    break
        threading.Thread(target=_stdin_waiter, daemon=True).start()

    def _init_controllers(self):
        """Inisialisasi stik PS4 / Xbox (XInput via Steam maupun DirectInput)."""
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            self._refresh_joysticks()
        except Exception as e:
            self.controller_name = f"Gamepad Init Err ({e})"

    def _refresh_joysticks(self):
        try:
            count = pygame.joystick.get_count()
            if count != len(self.joysticks):
                self.joysticks = []
                for i in range(count):
                    js = pygame.joystick.Joystick(i)
                    js.init()
                    self.joysticks.append(js)
                if self.joysticks:
                    self.controller_name = self.joysticks[0].get_name()
                else:
                    self.controller_name = "Disconnected"
        except Exception:
            pass

    def _is_controller_ptt_pressed(self) -> bool:
        """
        Mengecek apakah tombol SHARE di stik PS4 (yang dibaca Xbox View/Back oleh Steam/Windows) ditekan.
        - Xbox / XInput (Steam Input): Tombol View/Back (Share) = Button 6
        - PS4 DualShock 4 SDL2 / DirectInput: Tombol Share = Button 4 (SDL2) atau Button 8 (Legacy DInput)
        """
        try:
            pygame.event.pump()
            self._refresh_joysticks()
            if not self.joysticks:
                return False

            btn_cfg = config.PTT_CONTROLLER_BUTTON.strip().lower()
            for js in self.joysticks:
                name_lower = js.get_name().lower()
                num_buttons = js.get_numbuttons()

                # Jika user mengisi nomor tombol spesifik di .env (misal PTT_CONTROLLER_BUTTON=6)
                if btn_cfg.isdigit():
                    idx = int(btn_cfg)
                    if idx < num_buttons and js.get_button(idx):
                        return True
                    continue

                # Mode otomatis "share": deteksi berdasarkan profil Xbox (XInput) vs PS4 (DualShock)
                if "xbox" in name_lower or "xinput" in name_lower:
                    # Di Xbox Controller (Steam), tombol Share PS4 adalah tombol View/Back (Button 6)
                    target_buttons = [6]
                elif "ps4" in name_lower or "dualshock" in name_lower or "sony" in name_lower or "wireless controller" in name_lower:
                    # Di SDL2 modern, Share PS4 = Button 4 (sedangkan Button 6 adalah L2 digital).
                    # Jika tombol > 14 dan bukan SDL2 standar, fallback cek 4 dan 8.
                    target_buttons = [4, 8]
                else:
                    # Stik generik: cek Button 6 (Xbox Back/Share) dan Button 8
                    target_buttons = [6, 8]

                for b_idx in target_buttons:
                    if b_idx < num_buttons and js.get_button(b_idx):
                        return True
        except Exception:
            pass
        return False

    def add_log(self, speaker: str, message: str):
        timestamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            self.radio_logs.append((timestamp, speaker, message))

    def _set_tts_engine_name(self, name: str):
        self.last_tts_engine = name

    def _handle_ptt_worker(self):
        """Loop pemantau tombol Push-to-Talk (Keyboard 'V' ATAU Tombol SHARE Stik PS4/Xbox) secara non-blocking."""
        was_pressed = False
        while self.running:
            kb_pressed = False
            try:
                kb_pressed = keyboard.is_pressed(config.PTT_KEY)
            except Exception:
                pass

            ctrl_pressed = self._is_controller_ptt_pressed()
            is_pressed = kb_pressed or ctrl_pressed

            if is_pressed and not was_pressed and not self.processing_ai:
                was_pressed = True
                self.ptt_active = True
                try:
                    self.recorder.start_recording()
                except Exception as e:
                    self.add_log("ERROR", str(e))
                    self.ptt_active = False

            elif not is_pressed and was_pressed:
                was_pressed = False
                self.ptt_active = False
                wav_path = self.recorder.stop_recording(config.DRIVER_AUDIO_PATH)
                if wav_path:
                    threading.Thread(
                        target=self._process_driver_transmission,
                        args=(wav_path,),
                        daemon=True
                    ).start()
                else:
                    self.add_log("SYSTEM", "PTT released too quickly (<0.25s). Ignored.")

            time.sleep(0.03)

    def _process_driver_transmission(self, wav_path: str):
        self.processing_ai = True
        try:
            snap = self.telemetry.get_snapshot()
            driver_text, gp_reply = self.brain.process_driver_audio(wav_path, snap)
            self.add_log("DRIVER", driver_text)
            self.add_log("GP", gp_reply)
            self.radio_audio.speak_radio(gp_reply, on_source_resolved=self._set_tts_engine_name)
        except Exception as e:
            self.add_log("ERROR", f"Radio processing failed: {e}")
        finally:
            self.processing_ai = False

    def _check_proactive_race_events(self):
        """
        Memeriksa apakah ada event krusial (Tabrakan/Damage, Safety Car, Perubahan Cuaca, Ban Kritis,
        Overtake, atau Finish) dan memicu GP Lambiase bicara otomatis dengan cooldown hemat token (20 detik).
        """
        events = self.telemetry.pop_events()
        for ev in events:
            self.add_log("EVENT", ev)
            is_urgent = any(
                k in ev
                for k in (
                    "CRITICAL CAR DAMAGE",
                    "SAFETY CAR",
                    "RED FLAG",
                    "CRITICAL WEATHER",
                    "CRITICAL TYRE WEAR",
                    "CHEQUERED FLAG",
                    "FINAL CLASSIFICATION",
                    "OVERTAKE!"
                )
            )
            now = time.time()
            # Cooldown 20 detik untuk panggilan otomatis agar tetap responsif tapi hemat kuota API
            if is_urgent and not self.processing_ai and not self.ptt_active and (now - self.last_proactive_radio_time > 20.0):
                self.last_proactive_radio_time = now
                snap = self.telemetry.get_snapshot()
                threading.Thread(
                    target=self._trigger_proactive_callout,
                    args=(ev, snap),
                    daemon=True
                ).start()

    def _trigger_proactive_callout(self, ev: str, snap: dict):
        self.processing_ai = True
        try:
            gp_line = self.brain.generate_proactive_callout(ev, snap)
            self.add_log("GP", gp_line)
            self.radio_audio.speak_radio(gp_line, on_source_resolved=self._set_tts_engine_name)
        except Exception:
            pass
        finally:
            self.processing_ai = False

    def _wear_color(self, wear: float) -> str:
        if wear < 35.0:
            return Colors.GREEN
        elif wear < 65.0:
            return Colors.YELLOW
        return Colors.RED

    def _render_dashboard(self):
        snap = self.telemetry.get_snapshot()

        conn_badge = (
            f"{Colors.GREEN}{Colors.BOLD}● LIVE TELEMETRY (20777){Colors.RESET}"
            if snap["connected"]
            else f"{Colors.YELLOW}{Colors.BOLD}◌ WAITING FOR F1 24 UDP (20777)...{Colors.RESET}"
        )

        if self.ptt_active:
            lvl = self.recorder.current_level_pct
            bars = int(lvl / 5)  # 20 balok
            meter = "█" * bars + "░" * (20 - bars)
            mic_badge = f"{Colors.RED}{Colors.BOLD}[● REC - TRANSMITTING]{Colors.RESET} {Colors.GREEN}[{meter}] {lvl:>3}%{Colors.RESET}"
        elif self.processing_ai:
            mic_badge = f"{Colors.CYAN}{Colors.BOLD}[⚙ GP ANALYZING TELEMETRY...]{Colors.RESET}"
        else:
            mic_badge = f"{Colors.GREEN}[READY - Hold PS4 'SHARE' or Key '{config.PTT_KEY.upper()}' to Talk]{Colors.RESET} ({Colors.YELLOW}Pad: {self.controller_name}{Colors.RESET})"

        tw = snap["tyre_wear_pct"]
        ts = snap["tyre_surface_temps"]
        ti = snap["tyre_inner_temps"]
        tp = snap["tyre_pressures_psi"]

        lines = [
            "\033[H\033[J",  # Clear screen & move cursor home
            f"{Colors.BLUE}{Colors.BOLD}========================================================================================{Colors.RESET}",
            f" {Colors.BOLD}{Colors.WHITE}ORACLE RED BULL RACING | AI RACE ENGINEER - GIANPIERO LAMBIASE EDITION (F1 24){Colors.RESET}",
            f"{Colors.BLUE}{Colors.BOLD}========================================================================================{Colors.RESET}",
            f" Status : {conn_badge}    | Voice Engine: {Colors.CYAN}{self.last_tts_engine}{Colors.RESET}",
            f" Radio  : {mic_badge}",
            f"{Colors.DIM}----------------------------------------------------------------------------------------{Colors.RESET}",
            f" {Colors.BOLD}{Colors.YELLOW}[SESSION & STANDINGS]{Colors.RESET}",
            f"  Session  : {Colors.WHITE}{snap['session_type']:<18}{Colors.RESET} | Position : {Colors.BOLD}{Colors.CYAN}P{snap['car_position']}{Colors.RESET} (Lap {snap['current_lap']}/{snap['total_laps']})",
            f"  Gaps     : Ahead: {Colors.GREEN}+{snap['delta_front_s']:.3f}s{Colors.RESET}       | Behind   : {Colors.YELLOW}+{snap['delta_behind_s']:.3f}s{Colors.RESET}",
            f"  Weather  : {snap['weather']} (Track {snap['track_temp']}°C / Air {snap['air_temp']}°C) | SC: {Colors.MAGENTA}{snap['safety_car_status']}{Colors.RESET}",
            f"  Forecast : {Colors.CYAN}{snap['rain_forecast_summary']}{Colors.RESET}",
            f"{Colors.DIM}----------------------------------------------------------------------------------------{Colors.RESET}",
            f" {Colors.BOLD}{Colors.YELLOW}[CAR TELEMETRY, POWERTRAIN & SETUP]{Colors.RESET}",
            f"  Speed/RPM: {Colors.WHITE}{snap['speed_kph']:>3} km/h{Colors.RESET} ({snap['rpm']:>5} RPM, Gear {snap['gear']}) | Throttle: {snap['throttle']:>5.1f}% | Brake: {snap['brake']:>5.1f}%",
            f"  Fuel/ERS : Fuel: {Colors.GREEN}{snap['fuel_remaining_laps']:>4.2f} laps{Colors.RESET}   | ERS Battery: {Colors.CYAN}{snap['ers_battery_pct']:>5.1f}%{Colors.RESET} | Brake Bias: {snap['brake_bias']}%",
            f"  Setup    : Wings F/R: {snap['front_wing']}/{snap['rear_wing']} | Diff On/Off: {snap['on_throttle_diff']}%/{snap['off_throttle_diff']}% | ARB F/R: {snap['front_arb']}/{snap['rear_arb']}",
            f"  Damage   : Wing FL/FR/R: {snap['front_left_wing_damage']}%/{snap['front_right_wing_damage']}%/{snap['rear_wing_damage']}% | PU Wear ICE/MGUK/TC: {snap['ice_wear_pct']}%/{snap['mguk_wear_pct']}%/{snap['turbo_wear_pct']}%",
            f"{Colors.DIM}----------------------------------------------------------------------------------------{Colors.RESET}",
            f" {Colors.BOLD}{Colors.YELLOW}[TYRES: {snap['tyre_compound'].upper()} - AGE: {snap['tyre_age_laps']} LAPS]{Colors.RESET}",
            f"  FL Wear: {self._wear_color(tw[0])}{tw[0]:>5.1f}%{Colors.RESET} ({ts[2]}°C Surf / {ti[2]}°C Core / {tp[2]} PSI) | FR Wear: {self._wear_color(tw[1])}{tw[1]:>5.1f}%{Colors.RESET} ({ts[3]}°C Surf / {ti[3]}°C Core / {tp[3]} PSI)",
            f"  RL Wear: {self._wear_color(tw[2])}{tw[2]:>5.1f}%{Colors.RESET} ({ts[0]}°C Surf / {ti[0]}°C Core / {tp[0]} PSI) | RR Wear: {self._wear_color(tw[3])}{tw[3]:>5.1f}%{Colors.RESET} ({ts[1]}°C Surf / {ti[1]}°C Core / {tp[1]} PSI)",
            f"{Colors.DIM}----------------------------------------------------------------------------------------{Colors.RESET}",
            f" {Colors.BOLD}{Colors.YELLOW}[PIT-WALL RADIO COMMUNICATIONS LOG]{Colors.RESET}"
        ]

        with self._log_lock:
            if not self.radio_logs:
                lines.append(f"  {Colors.DIM}No radio transmissions yet.{Colors.RESET}")
            for ts_str, speaker, msg in self.radio_logs:
                if speaker == "GP":
                    tag = f"{Colors.CYAN}{Colors.BOLD}[GP LAMBIASE]{Colors.RESET}"
                elif speaker == "DRIVER":
                    tag = f"{Colors.GREEN}{Colors.BOLD}[DRIVER]     {Colors.RESET}"
                elif speaker == "EVENT":
                    tag = f"{Colors.MAGENTA}{Colors.BOLD}[RACE CTRL]  {Colors.RESET}"
                elif speaker == "ERROR":
                    tag = f"{Colors.RED}{Colors.BOLD}[ALERT]      {Colors.RESET}"
                else:
                    tag = f"{Colors.WHITE}{Colors.BOLD}[PIT WALL]   {Colors.RESET}"
                lines.append(f"  {Colors.DIM}{ts_str}{Colors.RESET} {tag} {msg}")

        lines.append(f"{Colors.BLUE}{Colors.BOLD}========================================================================================{Colors.RESET}")
        lines.append(f" {Colors.DIM}Controls: Hold [PS4 SHARE] or [{config.PTT_KEY.upper()}] to Talk | Ketik '1' + [ENTER] (atau CTRL+C) untuk EXIT{Colors.RESET}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def run(self):
        self.telemetry.start()
        ptt_thread = threading.Thread(target=self._handle_ptt_worker, daemon=True)
        ptt_thread.start()

        try:
            while self.running:
                self._check_proactive_race_events()
                self._render_dashboard()
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.telemetry.stop()
            print(f"\n{Colors.YELLOW}[Pit Wall] Shutting down F1 24 AI Race Engineer. Box, box.{Colors.RESET}")


if __name__ == "__main__":
    app = GPLambiaseRaceEngineerApp()
    app.run()

