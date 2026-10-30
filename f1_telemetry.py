import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import config


# Mapping Konstanta F1 24
SESSION_TYPES = {
    0: "Unknown",
    1: "FP1", 2: "FP2", 3: "FP3", 4: "Short FP",
    5: "Q1", 6: "Q2", 7: "Q3", 8: "Short Q", 9: "One-Shot Q",
    10: "Sprint Shootout 1", 11: "Sprint Shootout 2", 12: "Sprint Shootout 3",
    13: "Short Sprint Shootout", 14: "One-Shot Sprint Shootout",
    15: "Race", 16: "Sprint Race", 17: "Race 3",
    18: "Time Trial"
}

WEATHER_TYPES = {
    0: "Clear (Kering)",
    1: "Light Cloud (Berawan Tipis)",
    2: "Overcast (Mendung)",
    3: "Light Rain (Hujan Ringan)",
    4: "Heavy Rain (Hujan Deras)",
    5: "Storm (Badai)"
}

SAFETY_CAR_TYPES = {
    0: "None (Track Clear)",
    1: "Full Safety Car",
    2: "Virtual Safety Car (VSC)",
    3: "Formation Lap"
}

TYRE_COMPOUNDS = {
    16: "Soft (C5)", 17: "Medium (C4)", 18: "Hard (C3)",
    7: "Intermediate", 8: "Wet (Full Wet)",
    9: "Dry", 10: "Wet"
}


@dataclass
class RaceState:
    # Status Koneksi
    connected: bool = False
    last_packet_time: float = 0.0
    packet_format: int = 2024

    # Packet 1: Session
    session_type: str = "Waiting for Session..."
    track_id: int = -1
    weather: str = "Clear (Kering)"
    track_temp: int = 30
    air_temp: int = 24
    total_laps: int = 0
    safety_car_status: str = "None (Track Clear)"
    rain_forecast_summary: str = "No rain expected in the next 30m"

    # Packet 2: Lap Data
    current_lap: int = 1
    car_position: int = 1
    prev_car_position: int = 1
    delta_front_ms: int = 0
    delta_behind_ms: int = 0
    driver_ahead_name: str = "None (Clear Track)"
    driver_behind_name: str = "None"
    player_driver_name: str = "Mate"
    driver_names_by_idx: Dict[int, str] = field(default_factory=dict)
    last_lap_time_ms: int = 0
    current_lap_time_ms: int = 0
    pit_status: int = 0

    # Packet 5: Car Setups
    front_wing: int = 25
    rear_wing: int = 25
    on_throttle_diff: int = 55
    off_throttle_diff: int = 50
    front_camber: float = -3.00
    rear_camber: float = -1.50
    front_toe: float = 0.05
    rear_toe: float = 0.20
    front_arb: int = 10
    rear_arb: int = 8
    brake_bias: int = 56
    tyre_pressures_psi: List[float] = field(default_factory=lambda: [22.5, 22.5, 23.5, 23.5])  # RL, RR, FL, FR

    # Packet 6: Car Telemetry
    speed_kph: int = 0
    rpm: int = 0
    gear: int = 0
    throttle: float = 0.0
    brake: float = 0.0
    drs_active: bool = False
    tyre_surface_temps: List[int] = field(default_factory=lambda: [90, 90, 90, 90])  # RL, RR, FL, FR
    tyre_inner_temps: List[int] = field(default_factory=lambda: [95, 95, 95, 95])    # RL, RR, FL, FR
    lockup_detected: bool = False
    oversteer_slip_detected: bool = False

    # Packet 7: Car Status
    fuel_remaining_laps: float = 0.0
    ers_store_energy_pct: float = 100.0
    ers_deploy_mode: int = 1
    tyre_compound: str = "Soft"
    tyre_age_laps: int = 0

    # Packet 10: Car Damage
    tyre_wear_pct: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # RL, RR, FL, FR
    front_left_wing_damage: int = 0
    front_right_wing_damage: int = 0
    rear_wing_damage: int = 0
    floor_damage: int = 0
    ice_wear_pct: int = 0
    mguk_wear_pct: int = 0
    turbo_wear_pct: int = 0

    # Packet 3 & 8: Events & Final Classification
    chequered_flag: bool = False
    final_position: Optional[int] = None
    recent_events: List[str] = field(default_factory=list)


class F1TelemetryListener:
    """
    Non-blocking UDP Listener untuk membaca paket telemetri biner F1 24 (Port 20777).
    Format Header F1 24: 29 bytes (<HBBBBBQfIIBB)
    """
    HEADER_FORMAT = "<HBBBBBQfIIBB"
    HEADER_SIZE = 29

    def __init__(self, ip: str = config.UDP_IP, port: int = config.UDP_PORT):
        self.ip = ip
        self.port = port
        self.state = RaceState()
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.sock: Optional[socket.socket] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    def get_snapshot(self) -> Dict:
        """Mengambil salinan state balapan secara thread-safe."""
        with self.lock:
            # Cek apakah koneksi aktif (paket diterima dalam 5 detik terakhir)
            is_active = (time.time() - self.state.last_packet_time) < 5.0 if self.state.last_packet_time > 0 else False
            self.state.connected = is_active
            return {
                "connected": self.state.connected,
                "session_type": self.state.session_type,
                "weather": self.state.weather,
                "track_temp": self.state.track_temp,
                "air_temp": self.state.air_temp,
                "total_laps": self.state.total_laps,
                "safety_car_status": self.state.safety_car_status,
                "rain_forecast_summary": self.state.rain_forecast_summary,
                "current_lap": self.state.current_lap,
                "car_position": self.state.car_position,
                "delta_front_s": round(self.state.delta_front_ms / 1000.0, 3),
                "delta_behind_s": round(self.state.delta_behind_ms / 1000.0, 3),
                "driver_ahead_name": self.state.driver_ahead_name,
                "driver_behind_name": self.state.driver_behind_name,
                "player_driver_name": self.state.player_driver_name,
                "front_wing": self.state.front_wing,
                "rear_wing": self.state.rear_wing,
                "on_throttle_diff": self.state.on_throttle_diff,
                "off_throttle_diff": self.state.off_throttle_diff,
                "front_camber": round(self.state.front_camber, 2),
                "rear_camber": round(self.state.rear_camber, 2),
                "front_toe": round(self.state.front_toe, 3),
                "rear_toe": round(self.state.rear_toe, 3),
                "front_arb": self.state.front_arb,
                "rear_arb": self.state.rear_arb,
                "brake_bias": self.state.brake_bias,
                "tyre_pressures_psi": list(self.state.tyre_pressures_psi),
                "speed_kph": self.state.speed_kph,
                "rpm": self.state.rpm,
                "gear": self.state.gear,
                "throttle": round(self.state.throttle * 100, 1),
                "brake": round(self.state.brake * 100, 1),
                "tyre_surface_temps": list(self.state.tyre_surface_temps),
                "tyre_inner_temps": list(self.state.tyre_inner_temps),
                "lockup_detected": self.state.lockup_detected,
                "oversteer_slip_detected": self.state.oversteer_slip_detected,
                "fuel_remaining_laps": round(self.state.fuel_remaining_laps, 2),
                "ers_battery_pct": round(self.state.ers_store_energy_pct, 1),
                "tyre_compound": self.state.tyre_compound,
                "tyre_age_laps": self.state.tyre_age_laps,
                "tyre_wear_pct": [round(w, 1) for w in self.state.tyre_wear_pct],
                "front_left_wing_damage": self.state.front_left_wing_damage,
                "front_right_wing_damage": self.state.front_right_wing_damage,
                "rear_wing_damage": self.state.rear_wing_damage,
                "ice_wear_pct": self.state.ice_wear_pct,
                "mguk_wear_pct": self.state.mguk_wear_pct,
                "turbo_wear_pct": self.state.turbo_wear_pct,
                "chequered_flag": self.state.chequered_flag,
                "final_position": self.state.final_position,
                "recent_events": list(self.state.recent_events[-5:])
            }

    def pop_events(self) -> List[str]:
        """Mengambil dan mengosongkan antrean event terbaru secara thread-safe."""
        with self.lock:
            events = list(self.state.recent_events)
            self.state.recent_events.clear()
            return events

    def _add_event(self, event_msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {event_msg}"
        if not self.state.recent_events or self.state.recent_events[-1] != entry:
            self.state.recent_events.append(entry)
            if len(self.state.recent_events) > 15:
                self.state.recent_events.pop(0)

    def _listen_loop(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)

        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
                if len(data) < self.HEADER_SIZE:
                    continue

                # Unpack 29-byte PacketHeader F1 24
                header = struct.unpack(self.HEADER_FORMAT, data[:self.HEADER_SIZE])
                packet_format = header[0]
                packet_id = header[5]
                player_car_idx = header[10]

                if player_car_idx >= 22:
                    continue

                payload = data[self.HEADER_SIZE:]

                with self.lock:
                    self.state.last_packet_time = time.time()
                    self.state.connected = True
                    self.state.packet_format = packet_format

                    if packet_id == 1:
                        self._parse_session_packet(payload)
                    elif packet_id == 2:
                        self._parse_lap_data_packet(payload, player_car_idx)
                    elif packet_id == 3:
                        self._parse_event_packet(payload, player_car_idx)
                    elif packet_id == 4:
                        self._parse_participants_packet(payload, player_car_idx)
                    elif packet_id == 5:
                        self._parse_car_setup_packet(payload, player_car_idx)
                    elif packet_id == 6:
                        self._parse_car_telemetry_packet(payload, player_car_idx)
                    elif packet_id == 7:
                        self._parse_car_status_packet(payload, player_car_idx)
                    elif packet_id == 8:
                        self._parse_final_classification_packet(payload, player_car_idx)
                    elif packet_id == 10:
                        self._parse_car_damage_packet(payload, player_car_idx)

            except socket.timeout:
                continue
            except Exception:
                continue

    def _parse_session_packet(self, payload: bytes):
        """
        Packet 1 (Session Data):
        Offset 0: m_weather (uint8)
        Offset 1: m_trackTemperature (int8)
        Offset 2: m_airTemperature (int8)
        Offset 3: m_totalLaps (uint8)
        Offset 4-5: m_trackLength (uint16)
        Offset 6: m_sessionType (uint8)
        """
        if len(payload) < 20:
            return
        weather_id, track_temp, air_temp, total_laps, _, session_type_id, track_id = struct.unpack("<BbbBHBb", payload[:8])
        new_weather = WEATHER_TYPES.get(weather_id, "Clear (Kering)")
        if new_weather != self.state.weather and self.state.last_packet_time > 0:
            self._add_event(f"CRITICAL WEATHER CHANGE: Track weather is now {new_weather}!")
        self.state.weather = new_weather
        self.state.track_temp = track_temp
        self.state.air_temp = air_temp
        self.state.total_laps = total_laps
        self.state.session_type = SESSION_TYPES.get(session_type_id, f"Session({session_type_id})")
        self.state.track_id = track_id

        # Offset untuk safetyCarStatus dan WeatherForecastSamples pada F1 24
        # m_safetyCarStatus berada setelah marshal zones (5 + 21*5 = 110 bytes dari awal payload -> offset 124)
        if len(payload) >= 126:
            sc_status = payload[124]
            new_sc = SAFETY_CAR_TYPES.get(sc_status, "None (Track Clear)")
            if new_sc != self.state.safety_car_status and sc_status in (1, 2):
                self._add_event(f"SAFETY CAR DEPLOYED: {new_sc}")
            self.state.safety_car_status = new_sc

        # Ekstraksi Weather Forecast Samples (mulai sekitar offset 127)
        if len(payload) >= 136:
            num_samples = payload[126]
            rain_detected = False
            for i in range(min(num_samples, 10)):
                sample_offset = 127 + (i * 8)
                if sample_offset + 8 <= len(payload):
                    _, time_offset, weather_fc, _, _, _, _, rain_pct = struct.unpack(
                        "<BBBbbbbB", payload[sample_offset:sample_offset + 8]
                    )
                    if weather_fc >= 3 or rain_pct >= 40:
                        w_name = WEATHER_TYPES.get(weather_fc, "Rain")
                        self.state.rain_forecast_summary = f"Rain ({w_name}, {rain_pct}% chance) in ~{time_offset} mins"
                        rain_detected = True
                        break
            if not rain_detected and num_samples > 0:
                self.state.rain_forecast_summary = "Dry conditions expected for the next 30m"

    def _parse_lap_data_packet(self, payload: bytes, car_idx: int):
        """
        Packet 2 (Lap Data):
        Pada F1 24, ukuran per mobil adalah 57 bytes (22 mobil = 1254 bytes).
        """
        stride = 57 if len(payload) >= 22 * 57 else (len(payload) // 22 if len(payload) >= 22 * 50 else 53)
        offset = car_idx * stride
        if offset + 38 > len(payload):
            return

        # Fields awal LapData F1 24:
        # 0..3: m_lastLapTimeInMS (I)
        # 4..7: m_currentLapTimeInMS (I)
        # 8..13: sector1/2 times (H B H B)
        # 14..15: m_deltaToCarInFrontMSPart (H)
        # 16: m_deltaToCarInFrontMinutesPart (B)
        # 17..18: m_deltaToRaceLeaderMSPart (H)
        # 19: m_deltaToRaceLeaderMinutesPart (B)
        # 20..31: lapDistance, totalDistance, safetyCarDelta (f, f, f)
        # 32: m_carPosition (B)
        # 33: m_currentLapNum (B)
        # 34: m_pitStatus (B)
        last_lap_ms, curr_lap_ms = struct.unpack("<II", payload[offset:offset + 8])
        delta_front_ms, delta_front_min = struct.unpack("<HB", payload[offset + 14:offset + 17])
        car_pos, curr_lap, pit_status = struct.unpack("<BBB", payload[offset + 32:offset + 35])

        self.state.last_lap_time_ms = last_lap_ms
        self.state.current_lap_time_ms = curr_lap_ms
        self.state.delta_front_ms = (delta_front_min * 60000) + delta_front_ms

        if car_pos > 0:
            # Deteksi otomatis event OVERTAKE jika posisi membaik saat sesi berjalan
            if 0 < car_pos < self.state.car_position and self.state.current_lap >= 1:
                self._add_event(f"OVERTAKE! Moved up from P{self.state.car_position} to P{car_pos}")
            self.state.prev_car_position = self.state.car_position
            self.state.car_position = car_pos

        if curr_lap > 0:
            self.state.current_lap = curr_lap
        self.state.pit_status = pit_status

        # Hitung delta & identifikasi nama pembalap tepat di depan (P-1) dan di belakang (P+1)
        ahead_pos = self.state.car_position - 1
        behind_pos = self.state.car_position + 1
        found_ahead = False
        found_behind = False
        for idx in range(22):
            c_off = idx * stride
            if c_off + 35 <= len(payload):
                other_pos = payload[c_off + 32]
                if other_pos == ahead_pos and ahead_pos >= 1:
                    self.state.driver_ahead_name = self.state.driver_names_by_idx.get(idx, f"Car #{idx}")
                    found_ahead = True
                elif other_pos == behind_pos:
                    d_ms, d_min = struct.unpack("<HB", payload[c_off + 14:c_off + 17])
                    self.state.delta_behind_ms = (d_min * 60000) + d_ms
                    self.state.driver_behind_name = self.state.driver_names_by_idx.get(idx, f"Car #{idx}")
                    found_behind = True
        if not found_ahead:
            self.state.driver_ahead_name = "None (Leader P1)"
        if not found_behind:
            self.state.delta_behind_ms = 0
            self.state.driver_behind_name = "None"

    def _parse_event_packet(self, payload: bytes, player_car_idx: int):
        """
        Packet 3 (Event Packet):
        4 karakter ASCII event code diikuti EventDataDetails.
        """
        if len(payload) < 4:
            return
        event_code = payload[:4].decode("ascii", errors="ignore")

        if event_code == "CHQF":
            self.state.chequered_flag = True
            self._add_event("CHEQUERED FLAG! Session finished.")
        elif event_code == "OVTK" and len(payload) >= 6:
            overtaking_idx, overtaken_idx = struct.unpack("<BB", payload[4:6])
            if overtaking_idx == player_car_idx:
                self._add_event(f"OVERTAKE CONFIRMED on Car #{overtaken_idx}!")
            elif overtaken_idx == player_car_idx:
                self._add_event(f"Position lost to Car #{overtaking_idx}.")
        elif event_code == "RDFL":
            self._add_event("RED FLAG DEPLOYED!")
        elif event_code == "DRSE":
            self._add_event("DRS Enabled by Race Control.")

    def _parse_participants_packet(self, payload: bytes, player_car_idx: int):
        """
        Packet 4 (Participants Data):
        Byte 0: m_numActiveCars
        Diikuti 22 * ParticipantData (pada F1 24 berukuran 58 atau 60 bytes per mobil).
        Field m_name berada pada offset +7 sepanjang 48 bytes (UTF-8 null-terminated).
        """
        if len(payload) < 60:
            return
        stride = (len(payload) - 1) // 22
        if stride < 53:
            return
        for idx in range(22):
            off = 1 + (idx * stride)
            if off + 55 <= len(payload):
                raw_name = payload[off + 7:off + 55].split(b"\x00", 1)[0]
                name_str = raw_name.decode("utf-8", errors="ignore").strip()
                if name_str:
                    self.state.driver_names_by_idx[idx] = name_str
                    if idx == player_car_idx and name_str.lower() not in ("player", "driver"):
                        self.state.player_driver_name = name_str

    def _parse_car_setup_packet(self, payload: bytes, car_idx: int):
        """
        Packet 5 (Car Setups):
        Ukuran CarSetupData F1 24 adalah 49 atau 50 bytes per mobil.
        Format: <BBBBffffBBBBBBBBffffBBf (49 bytes)
        """
        stride = 50 if len(payload) >= 22 * 50 else 49
        offset = car_idx * stride
        if offset + 49 > len(payload):
            return

        unpacked = struct.unpack("<BBBBffffBBBBBBBBffffBBf", payload[offset:offset + 49])
        self.state.front_wing = unpacked[0]
        self.state.rear_wing = unpacked[1]
        self.state.on_throttle_diff = unpacked[2]
        self.state.off_throttle_diff = unpacked[3]
        self.state.front_camber = unpacked[4]
        self.state.rear_camber = unpacked[5]
        self.state.front_toe = unpacked[6]
        self.state.rear_toe = unpacked[7]
        self.state.front_arb = unpacked[10]
        self.state.rear_arb = unpacked[11]
        self.state.brake_bias = unpacked[14]
        # RL, RR, FL, FR tyre pressures
        self.state.tyre_pressures_psi = [
            round(unpacked[16], 1),
            round(unpacked[17], 1),
            round(unpacked[18], 1),
            round(unpacked[19], 1)
        ]

    def _parse_car_telemetry_packet(self, payload: bytes, car_idx: int):
        """
        Packet 6 (Car Telemetry):
        Ukuran CarTelemetryData F1 24 adalah 60 bytes per mobil.
        Format: <HfffBbHBBH4H4B4BHB4f4B
        """
        stride = 60
        offset = car_idx * stride
        if offset + 60 > len(payload):
            return

        unpacked = struct.unpack("<HfffBbHBBH4H4B4BHB4f4B", payload[offset:offset + 60])
        self.state.speed_kph = unpacked[0]
        self.state.throttle = unpacked[1]
        steer = unpacked[2]
        self.state.brake = unpacked[3]
        self.state.gear = unpacked[5]
        self.state.rpm = unpacked[6]
        self.state.drs_active = bool(unpacked[7])

        # Tyre Surface Temps (RL, RR, FL, FR) & Inner Temps
        self.state.tyre_surface_temps = [unpacked[14], unpacked[15], unpacked[16], unpacked[17]]
        self.state.tyre_inner_temps = [unpacked[18], unpacked[19], unpacked[20], unpacked[21]]

        # Deteksi Lock-up Rem & Oversteer/Traction Slip secara heuristik real-time
        # Lock-up: Tekanan rem tinggi (>80%) pada kecepatan menengah/tinggi + lonjakan suhu permukaan ban depan
        front_surface_max = max(self.state.tyre_surface_temps[2], self.state.tyre_surface_temps[3])
        if self.state.brake > 0.85 and self.state.speed_kph > 90 and front_surface_max > 112:
            if not self.state.lockup_detected:
                self.state.lockup_detected = True
                self._add_event("WARNING: Front tyre lock-up detected under heavy braking!")
        else:
            self.state.lockup_detected = False

        # Oversteer / Traction Slip: Throttle tinggi + sudut kemudi korektif besar + panas permukaan ban belakang
        rear_surface_max = max(self.state.tyre_surface_temps[0], self.state.tyre_surface_temps[1])
        if self.state.throttle > 0.75 and abs(steer) > 0.35 and rear_surface_max > 108:
            if not self.state.oversteer_slip_detected:
                self.state.oversteer_slip_detected = True
                self._add_event("WARNING: Rear traction slip / oversteer detected on corner exit!")
        else:
            self.state.oversteer_slip_detected = False

    def _parse_car_status_packet(self, payload: bytes, car_idx: int):
        """
        Packet 7 (Car Status):
        Ukuran CarStatusData F1 24 adalah 55 bytes per mobil.
        """
        stride = 55 if len(payload) >= 22 * 55 else (len(payload) // 22 if len(payload) >= 22 * 47 else 55)
        offset = car_idx * stride
        if offset + 45 > len(payload):
            return

        # Offset dalam CarStatusData:
        # 0..4: tractionControl, antiLockBrakes, fuelMix, frontBrakeBias, pitLimiterStatus (5B)
        # 5..16: fuelInTank (f), fuelCapacity (f), fuelRemainingLaps (f)
        # 25: actualTyreCompound (B)
        # 26: visualTyreCompound (B)
        # 27: tyresAgeLaps (B)
        # 37..40: ersStoreEnergy (f) (Max = 4,000,000 Joules)
        # 41: ersDeployMode (B)
        _, _, _, brake_bias, _, _, _, fuel_rem_laps = struct.unpack("<5Bfff", payload[offset:offset + 17])
        actual_compound, visual_compound, tyre_age = struct.unpack("<BBB", payload[offset + 25:offset + 28])
        ers_joules, ers_mode = struct.unpack("<fB", payload[offset + 37:offset + 42])

        if brake_bias > 0:
            self.state.brake_bias = brake_bias
        self.state.fuel_remaining_laps = fuel_rem_laps
        self.state.tyre_compound = TYRE_COMPOUNDS.get(
            visual_compound, TYRE_COMPOUNDS.get(actual_compound, f"Compound({visual_compound})")
        )
        self.state.tyre_age_laps = tyre_age
        self.state.ers_store_energy_pct = min(100.0, max(0.0, (ers_joules / 4000000.0) * 100.0))
        self.state.ers_deploy_mode = ers_mode

    def _parse_final_classification_packet(self, payload: bytes, car_idx: int):
        """
        Packet 8 (Final Classification):
        Byte 0: m_numCars
        Diikuti 22 * FinalClassificationData (45 bytes per mobil).
        Byte pertama dari FinalClassificationData per mobil adalah m_position.
        """
        if len(payload) < 2:
            return
        stride = 45
        offset = 1 + (car_idx * stride)
        if offset + 1 <= len(payload):
            final_pos = payload[offset]
            if final_pos > 0:
                self.state.final_position = final_pos
                self.state.chequered_flag = True
                is_quali = any(q in self.state.session_type for q in ("Q1", "Q2", "Q3", "Short Q", "One-Shot Q", "Shootout"))
                is_race = "Race" in self.state.session_type
                # // Track ID 21 = Yas Marina Abu Dhabi (GP Penutup Musim F1 24)
                is_season_finale = (self.state.track_id == 21)

                if is_race and is_season_finale and final_pos <= 3:
                    self._add_event("FINAL CLASSIFICATION: WORLD DRIVERS' CHAMPION WDC!")
                elif final_pos == 1 and is_quali:
                    self._add_event("FINAL CLASSIFICATION: P1 POLE POSITION!")
                elif final_pos == 1:
                    self._add_event("FINAL CLASSIFICATION: P1 VICTORY!")
                elif final_pos <= 3:
                    self._add_event(f"FINAL CLASSIFICATION: PODIUM FINISH P{final_pos}!")
                else:
                    self._add_event(f"FINAL CLASSIFICATION: Finished P{final_pos}.")

    def _parse_car_damage_packet(self, payload: bytes, car_idx: int):
        """
        Packet 10 (Car Damage):
        Ukuran CarDamageData F1 24 adalah 42 bytes per mobil.
        Format:
        0..15: m_tyresWear[4] (4f) -> RL, RR, FL, FR
        16..19: m_tyresDamage[4] (4B)
        20..23: m_brakesDamage[4] (4B)
        24: m_frontLeftWingDamage (B)
        25: m_frontRightWingDamage (B)
        26: m_rearWingDamage (B)
        27: m_floorDamage (B)
        28: m_diffuserDamage (B)
        29: m_sidepodDamage (B)
        30..32: drsFault, ersFault, gearBoxDamage (3B)
        33: m_engineDamage (B)
        34: m_engineMGUHWear (B)
        35: m_engineESWear (B)
        36: m_engineCEWear (B)
        37: m_engineICEWear (B)
        38: m_engineMGUKWear (B)
        39: m_engineTCWear (B)
        """
        stride = 42
        offset = car_idx * stride
        if offset + 40 > len(payload):
            return

        tyre_wear = struct.unpack("<4f", payload[offset:offset + 16])
        fl_wing, fr_wing, r_wing, floor_dmg = struct.unpack("<4B", payload[offset + 24:offset + 28])
        ice_wear, mguk_wear, tc_wear = struct.unpack("<3B", payload[offset + 37:offset + 40])

        # Deteksi otomatis tabrakan / kerusakan baru pada sayap atau floor (naik >= 8%)
        if (
            (fl_wing - self.state.front_left_wing_damage >= 8)
            or (fr_wing - self.state.front_right_wing_damage >= 8)
            or (r_wing - self.state.rear_wing_damage >= 8)
            or (floor_dmg - self.state.floor_damage >= 8)
        ):
            self._add_event(
                f"CRITICAL CAR DAMAGE DETECTED! Wing FL:{fl_wing}%, FR:{fr_wing}%, Rear:{r_wing}%, Floor:{floor_dmg}%"
            )

        # Deteksi otomatis ban mulai kritis (melewati ambang 65% untuk pertama kali di stint ini)
        old_max_wear = max(self.state.tyre_wear_pct) if self.state.tyre_wear_pct else 0.0
        new_wear_list = [tyre_wear[2], tyre_wear[3], tyre_wear[0], tyre_wear[1]]
        new_max_wear = max(new_wear_list)
        if old_max_wear < 65.0 <= new_max_wear:
            self._add_event(f"CRITICAL TYRE WEAR ALERT! Max tyre wear has reached {round(new_max_wear, 1)}%.")

        # Simpan sebagai urutan [FL, FR, RL, RR] agar intuitif saat dibaca engineer
        self.state.tyre_wear_pct = new_wear_list
        self.state.front_left_wing_damage = fl_wing
        self.state.front_right_wing_damage = fr_wing
        self.state.rear_wing_damage = r_wing
        self.state.floor_damage = floor_dmg
        self.state.ice_wear_pct = ice_wear
        self.state.mguk_wear_pct = mguk_wear
        self.state.turbo_wear_pct = tc_wear

