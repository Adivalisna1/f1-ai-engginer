import base64
import json
import urllib.request
import urllib.error
from typing import Dict, Tuple, Optional
import config


# Prompt murni transkripsi TANPA daftar contoh kata (agar AI tidak menjiplak/halusinasi kata "box box"!)
STT_PURE_PROMPT = (
    "Transcribe the exact speech in this audio recording. "
    "The speaker may speak in Indonesian (Bahasa Indonesia) or English. "
    "Do NOT guess or invent words. Write ONLY the exact words spoken in lowercase."
)


GP_LAMBIASE_TEXT_SYSTEM_PROMPT = """
You are Gianpiero "GP" Lambiase, Senior Formula 1 Race Engineer speaking over the F1 pit-wall radio.
Rules:
- NEVER call the driver "Max". Call them "mate".
- Even if the driver speaks in Indonesian (Bahasa Indonesia), you understand them 100% and reply in authentic British English as GP Lambiase.
- Reply in 1 to 2 short sentences (max 30 words). No markdown, no asterisks.
- If the driver rants or wants to retire: check WingDmg & PU wear. If <75%, refuse retirement with dry British wit and tell them to keep pushing or box for a wing.
- If the driver asks for car setup tweaks (understeer/oversteer/braking): give specific numbers based on their current Wings, Diff, ARB, or Brake Bias.
""".strip()


class RaceEngineerBrain:
    """
    Opsi 3 Hybrid Architecture (Diperbaiki dari halusinasi prompt):
    1. Pure Speech-to-Text via gemini-flash-latest (model yang jauh lebih pintar mendengar Bahasa Indonesia & Inggris
       dibanding versi lite, dengan thinkingBudget=0 agar tetap cepat ~0.8 detik, dan TANPA daftar kata jebakan!).
    2. Smart Pit-Wall Fast Lane (0.001s): Langsung menjawab 10 kategori pertanyaan balapan (Inggris & Indonesia)
       secara INSTAN dari data telemetri lokal!
    3. Text-to-Text Gemini AI (~0.6s): Untuk pertanyaan bebas/kompleks.
    """

    def __init__(self):
        self.groq_api_key = getattr(config, "GROQ_API_KEY", "").strip()
        self.has_groq_key = bool(self.groq_api_key and self.groq_api_key != "paste_your_gsk_api_key_here")
        self.api_key = config.GEMINI_API_KEY.strip()
        self.has_valid_key = bool(self.api_key and self.api_key != "your_gemini_api_key_here")
        self.stt_model = "gemini-flash-latest"
        self.text_model = "gemini-flash-lite-latest"

    def _call_groq_whisper_stt(self, wav_path: str) -> str:
        # // Kirim file WAV ke endpoint Groq Whisper Large v3 Turbo menggunakan multipart/form-data
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        boundary = "----F1PitWallBoundary7MA4YWxkTrZu0gW"
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        body = bytearray()
        # // Field 1: file audio WAV
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="file"; filename="driver_radio.wav"\r\n')
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(wav_bytes)
        body.extend(b"\r\n")

        # // Field 2: model whisper-large-v3-turbo
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="model"\r\n\r\n')
        body.extend(b"whisper-large-v3-turbo\r\n")

        # // Field 3: temperature 0 untuk mencegah halusinasi
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="temperature"\r\n\r\n')
        body.extend(b"0\r\n")

        # // Field 4: prompt konteks kosakata F1, Kode Sandi (cow, banana, pizza, apple, watermelon, roti, alpha, bravo, simply lovely)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="prompt"\r\n\r\n')
        body.extend(b"F1 team radio: simply lovely, cow, banana, pizza, apple, watermelon, roti, alpha, bravo, wdc, champion, pole, podium, radio check, driver ahead, driver behind, box box, retire.\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "F124-RaceEngineer/1.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("text", "").strip()

    def _resolve_groq_chat_model(self) -> str:
        # // Ambil ID model Text-to-Text yang benar-benar aktif di akun Groq user via /openai/v1/models
        if getattr(self, "_groq_chat_model", None):
            return self._groq_chat_model
        url = "https://api.groq.com/openai/v1/models"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "User-Agent": "F124-RaceEngineer/1.0"
            },
            method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                model_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                # // Prioritaskan GPT OSS 20B / 120B (sesuai dashboard Groq user) atau model chat non-whisper/non-tts
                for pref in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "gpt-oss-20b", "gpt-oss-120b"):
                    if pref in model_ids:
                        self._groq_chat_model = pref
                        return pref
                for mid in model_ids:
                    if not any(x in mid.lower() for x in ("whisper", "orpheus", "guard", "safety", "tts", "vision")):
                        self._groq_chat_model = mid
                        return mid
        except Exception:
            pass
        return "openai/gpt-oss-20b"

    def _call_groq_llm(self, user_prompt: str, system_text: str) -> str:
        # // Panggil Groq Chat Completions dengan max_tokens cukup untuk model Reasoning (gpt-oss-20b)
        model_id = self._resolve_groq_chat_model()
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 400
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "F124-RaceEngineer/1.0"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                # // Bersihkan tag <think>...</think> jika model mengeluarkan blok reasoning
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                if content:
                    return content
                return "Copy that mate, keep your head down and focus on your lap time."
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM {he.code} ({model_id}): {err_body[:80]}")

    def _call_gemini_api(self, model_name: str, parts: list, system_text: Optional[str] = None, max_tokens: int = 80) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_tokens
            }
        }
        if system_text:
            payload["system_instruction"] = {"parts": [{"text": system_text}]}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                candidates = resp_data.get("candidates", [])
                if candidates:
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    text_out = "".join(p.get("text", "") for p in content_parts if "text" in p)
                    if text_out.strip():
                        return text_out.strip()
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            try:
                err_msg = json.loads(err_body).get("error", {}).get("message", err_body)
            except Exception:
                err_msg = err_body
            raise RuntimeError(f"HTTP {he.code} ({model_name}): {err_msg[:90]}")
        raise RuntimeError("Empty transcription")

    def _fast_lane_intent_router(self, transcript: str, snap: Dict) -> Optional[str]:
        """
        Smart Fast Lane (0.001 detik, 0 latensi AI!):
        Mendukung penuh Bahasa Indonesia & Bahasa Inggris!
        """
        t = transcript.lower().strip()
        pos = snap.get("car_position", 1)
        lap = snap.get("current_lap", 1)
        total = snap.get("total_laps", 0)
        rem_laps = max(0, total - lap) if total > 0 else 20
        ahead_name = snap.get("driver_ahead_name", "None")
        behind_name = snap.get("driver_behind_name", "None")
        gap_ahead = snap.get("delta_front_s", 0.0)
        gap_behind = snap.get("delta_behind_s", 0.0)
        compound = snap.get("tyre_compound", "Soft")
        tyre_age = snap.get("tyre_age_laps", 0)
        tw = snap.get("tyre_wear_pct", [0.0, 0.0, 0.0, 0.0])
        max_wear = round(max(tw), 1)
        ts = snap.get("tyre_surface_temps", [90, 90, 90, 90])
        max_temp = max(ts)
        weather = snap.get("weather", "Clear")
        track_temp = snap.get("track_temp", 30)
        forecast = snap.get("rain_forecast_summary", "No rain expected")
        fuel_laps = snap.get("fuel_remaining_laps", 0.0)
        ers_pct = snap.get("ers_battery_pct", 100.0)
        fl_dmg = snap.get("front_left_wing_damage", 0)
        fr_dmg = snap.get("front_right_wing_damage", 0)
        r_dmg = snap.get("rear_wing_damage", 0)
        ice_wear = snap.get("ice_wear_pct", 0)

        # 0A. CODE WORD: "COW" -> RETIRE THE CAR
        if any(w in t.split() for w in ("cow", "cows", "kau", "sapi")) or "code cow" in t:
            max_dmg = max(fl_dmg, fr_dmg, r_dmg, ice_wear)
            if max_dmg >= 75:
                return "Code Cow received mate. Damage is terminal on the telemetry, bring the car back slowly and we will retire it."
            elif max(fl_dmg, fr_dmg) >= 20:
                return f"Code Cow negative mate, we are not retiring. You have {max(fl_dmg, fr_dmg)} percent wing damage, box this lap for a new front wing."
            return "Negative on Code Cow mate, telemetry shows the car is completely fine. Stop complaining, get your head down, and keep pushing."

        # 0B. CODE WORD: "BANANA" -> CEK KONDISI BAN SAAT INI (WEAR, AGE, TEMPS)
        if "banana" in t:
            status_word = (
                "looking healthy, good to keep pushing"
                if max_wear < 40
                else ("getting worn, manage your slip on exit" if max_wear < 65 else "critical now, we need to box soon")
            )
            return (
                f"Copy Code Banana. You are on {compound}s, {tyre_age} laps old. "
                f"Highest wear is {max_wear} percent and peak surface temp is {max_temp} degrees. Tyres are {status_word}."
            )

        # 0C. CODE WORD: "PIZZA" -> ADA YANG NABRAK / CRASH / CONTACT
        if "pizza" in t:
            max_wing = max(fl_dmg, fr_dmg, r_dmg)
            if max_wing >= 20:
                return (
                    f"Copy Code Pizza, we see heavy contact on telemetry: front left wing {fl_dmg} percent, front right {fr_dmg} percent, rear {r_dmg} percent. "
                    f"Box this lap for repairs, watch for debris."
                )
            elif max_wing > 0:
                return f"Copy Code Pizza, minor contact detected: wing damage is {max_wing} percent. Car balance looks okay, stay out and watch for Yellow Flags."
            return f"Copy Code Pizza, incident noted. Your aero is clean with zero damage, watch for debris and possible Safety Car."

        # 0D. CODE WORD: "APPLE" -> CEK CUACA & FORECAST HUJAN
        if any(k in t for k in ("apple", "apel")):
            return f"Copy Code Apple. Current weather is {weather}, track temperature is {track_temp} degrees. {forecast}."

        # 0E. CODE WORD: "WATERMELON" -> SARAN STRATEGI BAN KE DEPAN (STAY OUT VS GANTI BAN BERDASARKAN TELEMETRI)
        if "watermelon" in t:
            # // Tentukan kompon target berdasarkan cuaca & sisa lap
            if "rain" in weather.lower() or "rain" in forecast.lower():
                target_comp = "Full Wets" if ("heavy" in weather.lower() or "storm" in weather.lower()) else "Intermediates"
                if "rain" in weather.lower() and compound.lower() not in ("intermediate", "wet (full wet)", "wet"):
                    return f"Copy Code Watermelon. Track is wet and you are on slicks, box this lap to switch to {target_comp}."
                return f"Copy Code Watermelon. With moisture in the forecast, stay out on your {compound}s for now and we will switch to {target_comp} when the rain hits."

            target_comp = "Softs" if (rem_laps <= 14 and total > 0) else ("Mediums" if rem_laps <= 28 else "Hards")
            if max_wear < 45.0:
                return (
                    f"Copy Code Watermelon. Your {compound}s are only at {max_wear} percent wear, so stay out and extend this stint. "
                    f"When the window opens, our best compound for the final {rem_laps} laps is the {target_comp}."
                )
            else:
                return (
                    f"Copy Code Watermelon. Tyre wear has reached {max_wear} percent on the {compound}s. "
                    f"We recommend boxing soon to switch to fresh {target_comp} for the remaining {rem_laps} laps."
                )

        # 0F. CODE WORD: "ROTI" -> TAKTIK NAIK POSISI (ATTACK/UNDERCUT) VS TAHAN LAWAN DEMI TEAMMATE CLEAN AIR
        if any(k in t for k in ("roti", "bread")):
            if pos == 1:
                return (
                    f"Copy Code Roti. You are P1 in clean air mate. Control the pace, protect your {compound}s, "
                    f"and keep {behind_name} at {gap_behind} seconds behind."
                )
            # // Jika mobil depan dekat (<= 1.8s), instruksikan serangan naik posisi atau undercut
            if 0.0 < gap_ahead <= 1.8:
                if max_wear >= 38.0:
                    return (
                        f"Copy Code Roti. {ahead_name} is {gap_ahead} seconds ahead in dirty air. "
                        f"Let us go for the undercut: push hard this lap and box to jump them in the pit phase."
                    )
                return (
                    f"Copy Code Roti. You are in attack range on {ahead_name}, gap {gap_ahead} seconds. "
                    f"You have {ers_pct} percent ERS battery, deploy overtake on the main straight to take P{pos - 1}."
                )
            # // Jika mobil depan jauh (> 3.0s) dan mobil belakang menempel (<= 1.5s), mainkan team game tahan lawan
            if gap_ahead > 3.0 and 0.0 < gap_behind <= 1.5:
                return (
                    f"Copy Code Roti. Gap ahead to {ahead_name} is {gap_ahead} seconds, so play the team game: "
                    f"back {behind_name} up in the middle sector to give your teammate clean air, and save ERS for the straights."
                )
            # // Kondisi umum: kejar ritme secara bertahap / overcut
            return (
                f"Copy Code Roti. You are P{pos}, {ahead_name} is {gap_ahead} seconds ahead and {behind_name} is {gap_behind} seconds behind. "
                f"Focus on consistent lap times in clean air to set up an overcut."
            )

        # 0G. CODE WORD: "ALPHA" -> ANALISIS TELEMETRI & REKOMENDASI MODIFIKASI SETUP MOBIL
        if any(k in t for k in ("alpha", "alfa")):
            f_wing = snap.get("front_wing", 25)
            r_wing = snap.get("rear_wing", 25)
            on_diff = snap.get("on_throttle_diff", 55)
            off_diff = snap.get("off_throttle_diff", 50)
            f_arb = snap.get("front_arb", 10)
            r_arb = snap.get("rear_arb", 8)
            bb = snap.get("brake_bias", 56)
            tp = snap.get("tyre_pressures_psi", [22.5, 22.5, 23.5, 23.5])
            rear_temp_avg = (ts[0] + ts[1]) / 2.0
            front_temp_avg = (ts[2] + ts[3]) / 2.0

            # // Evaluasi ketidakseimbangan termal & handling dari telemetri
            if snap.get("lockup_detected") or front_temp_avg > 105:
                setup_advice = (
                    f"Front tyres are running hot at {int(front_temp_avg)} degrees with braking lock-up risk. "
                    f"Shift Brake Bias rearward from {bb} to {max(50, bb - 2)} percent, lower front tyre pressure by 0.4 PSI, "
                    f"and add 2 clicks of Front Wing from {f_wing} to {f_wing + 2}."
                )
            elif snap.get("oversteer_slip_detected") or rear_temp_avg > 105:
                setup_advice = (
                    f"Rear tyres are overheating at {int(rear_temp_avg)} degrees with traction slip. "
                    f"Lower On-Throttle Differential from {on_diff} to {max(50, on_diff - 5)} percent, soften Rear ARB from {r_arb} to {max(1, r_arb - 2)}, "
                    f"and add 2 clicks of Rear Wing from {r_wing} to {r_wing + 2}."
                )
            elif front_temp_avg - rear_temp_avg >= 5.0:
                setup_advice = (
                    f"Telemetry shows mild understeer balance. Increase Front Wing from {f_wing} to {f_wing + 2} "
                    f"and lower Off-Throttle Differential from {off_diff} to {max(20, off_diff - 5)} percent for better rotation."
                )
            else:
                setup_advice = (
                    f"Current setup wings {f_wing}/{r_wing} and diff {on_diff}/{off_diff} percent are well balanced. "
                    f"To extract more straight-line speed, trim Rear Wing to {max(1, r_wing - 2)} and keep Brake Bias at {bb} percent."
                )

            pu_note = (
                f" Also note: ICE wear is high at {ice_wear} percent, consider fitting a fresh engine component after this session."
                if ice_wear >= 55
                else ""
            )
            return f"Copy Code Alpha. {setup_advice}{pu_note}"

        # 0H. CODE WORD: "BRAVO" / "SIMPLY LOVELY" -> SELEBRASI P1 / POLE POSITION / PODIUM / JUARA DUNIA (WDC)
        if any(k in t for k in ("bravo", "simply lovely", "lovely", "champion", "wdc", "world champion")):
            sess = snap.get("session_type", "Race")
            is_quali = any(q in sess for q in ("Q1", "Q2", "Q3", "Short Q", "One-Shot Q", "Shootout"))
            if "simply lovely" in t or "lovely" in t:
                return (
                    "Yes!! Ha-ha! Simply lovely indeed, mate!! What a performance!! "
                    "Absolutely clinical out there today... every single lap was on the limit! Get in there!!"
                )
            if "wdc" in t or "champion" in t:
                return (
                    "Yes!! Yes, mate!! You are the World Drivers' Champion!! Ha-ha, unbelievable season!! "
                    "Absolutely clinical from day one... bring that championship trophy home!!"
                )
            if pos == 1 and is_quali:
                return (
                    "Yes, mate!! Pole position!! P-1!! What a mega lap!! "
                    "Purple through the sectors... ha-ha, simply lovely job!!"
                )
            if pos == 1:
                return (
                    "Get in there, mate!! Yes!! P-1, what a win!! Absolutely mega drive today!! "
                    "You controlled that race from start to finish... simply lovely!!"
                )
            if pos <= 3:
                return (
                    f"Yes, mate!! P-{pos}, fantastic podium finish!! "
                    "Great recovery and mega pace out there... enjoy that trophy!!"
                )
            return (
                "Yes, mate!! Mega effort out there!! Fantastic fight all the way to the line... "
                "we will take this momentum into the next Grand Prix!!"
            )

        # 0E. RETIRE / RANTING / NGOMEL MINTA RETIRE / SWEARING
        if any(k in t for k in (
            "retire", "retired", "retiring", "nyerah", "mundur", "quit", "undrivable",
            "garbage", "shitbox", "gak enak", "parah", "capek", "fuck", "fucking", "shit", "hell", "damn", "anjing", "ajg"
        )):
            max_dmg = max(fl_dmg, fr_dmg, r_dmg, ice_wear)
            if max_dmg >= 75:
                return "Understood mate, damage is terminal on the telemetry. Bring the car back slowly and we will retire it in the garage."
            elif max(fl_dmg, fr_dmg) >= 20:
                return f"Calm down mate, we are not retiring the car. You have {max(fl_dmg, fr_dmg)} percent wing damage, box this lap for a new nose and get back out there."
            return "Negative mate, we are not retiring the car. Telemetry shows everything is fine, stop complaining and get your head down."

        # 1. WEATHER / CUACA / RAIN FORECAST
        if any(k in t for k in ("cuaca", "hujan", "mendung", "panas", "gerimis", "weather", "rain", "forecast", "wet", "dry", "track temp")):
            return f"Current weather is {weather}, track temperature is {track_temp} degrees. {forecast}."

        # 2. RADIO CHECK / TES RADIO
        if any(k in t for k in ("radio check", "loud and clear", "cek radio", "can you hear", "tes radio", "halo gp", "hello gp")):
            if not snap.get("connected"):
                return "Loud and clear mate. Telemetry is on standby in the garage, ready when you hit the track."
            return f"Loud and clear mate. You are P{pos} on lap {lap}, systems and telemetry are all green."

        # 3. DRIVER AHEAD / SIAPA DI DEPAN / GAP DEPAN
        if any(k in t for k in ("ahead", "in front", "depan", "front gap", "catching")):
            if pos == 1 or "leader" in ahead_name.lower() or ahead_name == "None":
                return f"You are P1 leading the session mate, clean air ahead. Car behind is {behind_name} at {gap_behind} seconds."
            return f"Car ahead is {ahead_name} in P{max(1, pos - 1)}, gap is {gap_ahead} seconds. Keep your head down."

        # 4. DRIVER BEHIND / SIAPA DI BELAKANG / GAP BELAKANG
        if any(k in t for k in ("behind", "belakang", "chasing", "following")):
            if behind_name == "None":
                return "Nobody directly behind you mate, gap is completely clear."
            return f"Car behind is {behind_name} in P{pos + 1}, gap is {gap_behind} seconds."

        # 5. NEXT TYRE / BAN SELANJUTNYA / REKOMENDASI BAN
        if any(k in t for k in (
            "next tyre", "next tire", "next compound", "what tyre", "what tire", "which tyre",
            "ban selanjutnya", "ban berikutnya", "ganti ban", "ban apa", "saran ban", "rekomendasi ban"
        )):
            if "rain" in weather.lower() or "rain" in forecast.lower():
                if "heavy" in weather.lower() or "storm" in weather.lower():
                    return f"With heavy rain conditions, we recommend boxing for Full Wets mate. {forecast}."
                return f"We see moisture in the forecast, {forecast}. Intermediates are the best call for the next stint."
            if rem_laps <= 14 and total > 0:
                rec = "Softs"
                reason = f"only {rem_laps} laps remaining so you can push flat out"
            elif rem_laps <= 28:
                rec = "Mediums"
                reason = f"{rem_laps} laps to go, giving us the best balance of pace and degradation"
            else:
                rec = "Hards"
                reason = f"a long {rem_laps} lap run to the flag"
            return f"Copy mate, for the next stint we recommend the {rec}, {reason}."

        # 6. TYRE CONDITION / WEAR / KONDISI BAN SEKARANG
        if any(k in t for k in ("tyre", "tire", "kondisi ban", "ban gimana", "keadaan ban", "aus ban", "wear", "degradation", "grip", "ban gw", "ban aku")):
            status_word = "looking healthy" if max_wear < 40 else ("getting worn, manage your slip" if max_wear < 65 else "critical, we need to box soon")
            return (
                f"You are on {compound}s, {tyre_age} laps old. Highest wear is {max_wear} percent and surface peak is {max_temp} degrees. "
                f"Tyres are {status_word}."
            )

        # 7. CAR DAMAGE / KERUSAKAN / SAYAP
        if any(k in t for k in ("damage", "wing", "kerusakan", "rusak", "tabrak", "sayap", "front wing", "broken", "kondisi mobil", "mobil aman")):
            max_wing = max(fl_dmg, fr_dmg, r_dmg)
            if max_wing == 0:
                return f"Aero is completely clean mate, zero wing damage. Engine ICE wear is at {ice_wear} percent."
            elif max_wing < 25:
                return f"Minor aero damage mate: front left {fl_dmg} percent, front right {fr_dmg} percent, rear {r_dmg} percent. Stay out for now."
            else:
                return f"Heavy wing damage detected: front left {fl_dmg} percent, front right {fr_dmg} percent. Box this lap for a front wing replacement."

        # 8. FUEL & BATTERY / BENSIN & ERS
        if any(k in t for k in ("fuel", "bensin", "bahan bakar", "battery", "baterai", "ers", "energy")):
            fuel_note = "Fuel is comfortable" if fuel_laps >= 1.0 else "Fuel is tight, lift and coast into heavy braking zones"
            return f"{fuel_note} at {fuel_laps} laps remaining, and ERS battery is at {ers_pct} percent."

        # 10. BOX BOX / PIT STOP
        if any(k in t for k in ("box box", "box this lap", "pit this lap", "masuk pit", "pit stop", "coming in")):
            next_c = "Intermediates" if "rain" in forecast.lower() else ("Softs" if rem_laps < 15 else "Mediums")
            return f"Copy that, box this lap, box box. Pit crew is ready with a fresh set of {next_c}, watch your pit lane speed limit."

        return None

    def build_telemetry_context(self, snap: Dict) -> str:
        tw = snap.get("tyre_wear_pct", [0, 0, 0, 0])
        ts = snap.get("tyre_surface_temps", [90, 90, 90, 90])
        conn_status = "LIVE ON TRACK" if snap.get("connected") else "IN GARAGE"
        return (
            f"TELEMETRY [{conn_status}]: Session={snap.get('session_type')}, Lap={snap.get('current_lap')}/{snap.get('total_laps')}, "
            f"Pos=P{snap.get('car_position')}, Ahead={snap.get('driver_ahead_name')}(+{snap.get('delta_front_s')}s), "
            f"Behind={snap.get('driver_behind_name')}(+{snap.get('delta_behind_s')}s), Weather={snap.get('weather')}, "
            f"Forecast={snap.get('rain_forecast_summary')}, Tyres={snap.get('tyre_compound')}(Wear={tw}%, Temps={ts}C), "
            f"Fuel={snap.get('fuel_remaining_laps')}L, ERS={snap.get('ers_battery_pct')}%, "
            f"Setup(Wings={snap.get('front_wing')}/{snap.get('rear_wing')}, Diff={snap.get('on_throttle_diff')}/{snap.get('off_throttle_diff')}, "
            f"ARB={snap.get('front_arb')}/{snap.get('rear_arb')}, BB={snap.get('brake_bias')}%), "
            f"WingDmg={snap.get('front_left_wing_damage')}/{snap.get('front_right_wing_damage')}/{snap.get('rear_wing_damage')}%"
        )

    def process_driver_audio(self, wav_path: str, telemetry_snapshot: Dict) -> Tuple[str, str]:
        # // reload key dari config/env jika baru diisi
        if not self.has_groq_key:
            self.groq_api_key = getattr(config, "GROQ_API_KEY", "").strip()
            self.has_groq_key = bool(self.groq_api_key and self.groq_api_key != "paste_your_gsk_api_key_here")

        # // JALUR UTAMA: Groq Cloud (Whisper Large v3 Turbo STT ~0.3s + Fast Lane / Llama 3.3 70B ~0.2s)
        if self.has_groq_key:
            try:
                transcript = self._call_groq_whisper_stt(wav_path)
                if not transcript:
                    return ("[Unintelligible]", "Say again mate, radio was empty.")

                # // Cek Smart Fast Lane (0.001 detik) untuk 10 kategori utama F1
                fast_reply = self._fast_lane_intent_router(transcript, telemetry_snapshot)
                if fast_reply is not None:
                    return (transcript, fast_reply)

                # // Jika pertanyaan bebas/kompleks, lempar teks ke Groq Llama 3.3 (~0.2 detik)
                telemetry_context = self.build_telemetry_context(telemetry_snapshot)
                user_msg = f"{telemetry_context}\nDriver said over radio: \"{transcript}\"\nReply as GP Lambiase in 1-2 short sentences:"
                gp_reply = self._call_groq_llm(user_msg, GP_LAMBIASE_TEXT_SYSTEM_PROMPT)
                gp_reply = gp_reply.replace("GP_RESPONSE:", "").replace("*", "").strip()
                return (transcript, gp_reply)
            except Exception as e:
                return (f"[Groq Error: {str(e)[:75]}]", "Radio static on the pit wall mate, say again?")

        if not self.has_valid_key:
            return (
                "[API Key belum diisi di .env]",
                "Radio check mate, please set your Groq API key in the env file."
            )

        try:
            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")

            # Gunakan self.stt_model (gemini-flash-latest) untuk akurasi penuh transkripsi audio Bahasa Indonesia/Inggris
            telemetry_context = self.build_telemetry_context(telemetry_snapshot)
            single_prompt = (
                "Listen to the attached audio (spoken in Indonesian or English). "
                "First, transcribe the exact spoken words into DRIVER_TRANSCRIPT. "
                "Second, answer as F1 Race Engineer GP Lambiase in GP_RESPONSE using this telemetry:\n"
                f"{telemetry_context}\n\n"
                "Format:\n"
                "DRIVER_TRANSCRIPT: <exact spoken words>\n"
                "GP_RESPONSE: <short 1-sentence reply>"
            )
            stt_parts = [
                {"text": single_prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}
            ]
            try:
                raw_out = self._call_gemini_api(self.stt_model, stt_parts, system_text=GP_LAMBIASE_TEXT_SYSTEM_PROMPT, max_tokens=90)
            except Exception:
                raw_out = self._call_gemini_api(self.text_model, stt_parts, system_text=GP_LAMBIASE_TEXT_SYSTEM_PROMPT, max_tokens=90)

            transcript = "Driver Radio"
            ai_reply = raw_out.strip()
            for line in raw_out.splitlines():
                line_s = line.strip()
                if line_s.upper().startswith("DRIVER_TRANSCRIPT:"):
                    transcript = line_s.split(":", 1)[1].strip().strip('"').strip("*")
                elif line_s.upper().startswith("GP_RESPONSE:"):
                    ai_reply = line_s.split(":", 1)[1].strip().strip('"').strip("*")

            # Jika transkrip cocok dengan 10 kategori utama di Fast Lane, gunakan jawaban presisi Fast Lane!
            fast_reply = self._fast_lane_intent_router(transcript, telemetry_snapshot)
            if fast_reply is not None:
                return (transcript, fast_reply)

            return (transcript, ai_reply)

        except Exception as e:
            return (
                f"[Error: {str(e)[:75]}]",
                "Radio static on the pit wall mate, say again?"
            )

    def generate_proactive_callout(self, event_description: str, telemetry_snapshot: Dict) -> str:
        ev_upper = event_description.upper()
        if "DAMAGE" in ev_upper:
            fl = telemetry_snapshot.get("front_left_wing_damage", 0)
            fr = telemetry_snapshot.get("front_right_wing_damage", 0)
            return f"We see front wing damage on the telemetry mate, left {fl} percent, right {fr} percent. Box this lap if you cannot hold the apex."
        if "SAFETY CAR" in ev_upper:
            return "Safety Car deployed mate, delta positive. Check if you want to box for fresh tyres."
        if "WEATHER" in ev_upper:
            return f"Track conditions changing mate, {telemetry_snapshot.get('weather')}. Keep an eye on grip levels."
        if "TYRE WEAR" in ev_upper:
            return "Tyre wear is past 65 percent mate, watch your rear traction on exit or box this lap."
        if "OVERTAKE" in ev_upper:
            return f"Nice move mate, P{telemetry_snapshot.get('car_position')}. Now get your head down and close the gap ahead."
        if "CHEQUERED FLAG" in ev_upper or "FINAL" in ev_upper:
            if "WDC" in ev_upper or "CHAMPION" in ev_upper:
                return (
                    "Yes!! Yes, mate!! You are the World Drivers' Champion!! Ha-ha, unbelievable season!! "
                    "You deserve every bit of this... simply lovely, bring that championship home!!"
                )
            if "POLE POSITION" in ev_upper:
                return "Yes, mate!! P-1, Pole Position!! What a mega flying lap!!"
            if "VICTORY" in ev_upper:
                return "Get in there, mate!! Yes!! P-1 Victory!! Absolutely clinical drive today, simply lovely!!"
            if "PODIUM" in ev_upper:
                return f"Yes, mate!! Podium finish, P-{telemetry_snapshot.get('car_position')}!! Great job out there, bring it home!!"
            return f"Chequered flag mate, P{telemetry_snapshot.get('car_position')}. Good job, bring the car home."
        return f"Copy that, {event_description}. Stay focused."
