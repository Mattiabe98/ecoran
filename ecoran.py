import yaml
import subprocess
import time
import os
import sys
import struct
from typing import List, Dict, Any, Set, Tuple, Optional
import threading
import argparse
import signal
import logging 
from logging.handlers import RotatingFileHandler

try:
    from lib.xAppBase import xAppBase 
except ImportError:
    print("E: Failed to import xAppBase from lib.xAppBase. Ensure the library is correctly installed and accessible.")
    sys.exit(1)

# MSR Addresses
MSR_IA32_TSC = 0x10
MSR_IA32_MPERF = 0xE7

# Verbosity levels
SILENT = 0
ERROR = 1
WARN = 2
INFO = 3
DEBUG_KPM = 4
DEBUG_ALL = 5

LOGGING_LEVEL_MAP = {
    SILENT: logging.CRITICAL + 10, ERROR: logging.ERROR, WARN: logging.WARNING,
    INFO: logging.INFO, DEBUG_KPM: logging.DEBUG, DEBUG_ALL: logging.DEBUG
}

def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            val_bytes = f.read(8) 
            return struct.unpack('<Q', val_bytes)[0] if len(val_bytes) == 8 else None
    except Exception: return None

class CoreMSRData:
    def __init__(self, core_id: int):
        self.core_id, self.mperf, self.tsc, self.busy_percent = core_id, None, None, 0.0

class PowerManager(xAppBase):
    MAX_VOLUME_COUNTER_KBITS = (2**32) - 1 

    def __init__(self, config_path: str, http_server_port: int, rmr_port: int, kpm_ran_func_id: int = 2):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.verbosity = int(self.config.get('console_verbosity_level', INFO))
        self.file_verbosity_cfg = int(self.config.get('file_verbosity_level', DEBUG_KPM))
        self.log_file_path_base = self.config.get('log_file_path', "/mnt/data/ecoran")
        self._setup_logging()

        xapp_base_config_file = self.config.get('xapp_base_config_file', '')
        super().__init__(xapp_base_config_file, http_server_port, rmr_port)
        
        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        self.max_energy_val_rapl = self.config.get('rapl_max_energy_uj_override', 2**60 - 1)
        self.print_interval_s = int(self.config.get('print_interval', 5))
        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 90)) 
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 170)) 
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 99.5))
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_update_interval_s = int(self.config.get('tdp_update_interval_s', 1))
        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.0005))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 3))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 1.5))
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        self.dry_run = bool(self.config.get('dry_run', False))
        self.current_tdp_w = float(self.tdp_min_w) 
        self.last_pkg_energy_uj: Optional[int] = None 
        self.last_energy_read_time: Optional[float] = None 
        self.energy_at_last_log_uj: Optional[int] = None 
        self.max_ru_timing_usage_history: List[float] = []
        self.last_tdp_adjustment_time: float = 0.0
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}
        self.kpm_ran_func_id = kpm_ran_func_id
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None: self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else: self._log(WARN, "xAppBase.e2sm_kpm module unavailable."); self.e2sm_kpm = None
        self.gnb_ids_map = self.config.get('gnb_ids', {}) 
        self.gnb_id_to_du_name_map = {v: k for k, v in self.gnb_ids_map.items()}
        clos_association_config = self.config.get('clos_association', {})
        self.clos_to_du_names_map: Dict[int, List[str]] = {}
        ran_components_in_config = self.config.get('ran_cores', {}).keys()
        for cid, comps in clos_association_config.items():
            if isinstance(comps, list): self.clos_to_du_names_map[int(cid)] = [c for c in comps if c in ran_components_in_config and c.startswith('du')]
            else: self._log(WARN, f"Components for CLOS {cid} not a list. Skipping.")
        
        self.kpm_gnb_tracking: Dict[str, Dict[str, Any]] = {} 
        self.kpm_data_lock = threading.Lock() 
        
        # For per-KPM-interval approximate efficiency logging
        self.last_energy_reading_for_kpm_eff_uj: Optional[int] = None
        self.last_kpm_report_time_for_eff_calc: Optional[float] = None
        self.kpm_eff_lock = threading.Lock() 

        self._validate_config()
        if self.dry_run: self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _setup_logging(self):
        self.logger = logging.getLogger("EcoRANPowerManager")
        self.logger.handlers = []
        self.logger.propagate = False
        console_level = LOGGING_LEVEL_MAP.get(self.verbosity, logging.INFO)
        file_level = LOGGING_LEVEL_MAP.get(self.file_verbosity_cfg, logging.DEBUG)
        overall_logger_level = min(console_level, file_level) if not (self.verbosity == SILENT and self.file_verbosity_cfg == SILENT) else logging.CRITICAL + 10
        self.logger.setLevel(overall_logger_level)

        if self.verbosity > SILENT:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(console_level)
            ch_formatter = logging.Formatter('%(asctime)s %(levelname).1s: %(message)s', datefmt='%H:%M:%S')
            ch.setFormatter(ch_formatter)
            self.logger.addHandler(ch)

        if self.file_verbosity_cfg > SILENT and self.log_file_path_base:
            try:
                os.makedirs(self.log_file_path_base, exist_ok=True)
                log_fn = f"ecoran_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
                log_fp = os.path.join(self.log_file_path_base, log_fn)
                fh = logging.FileHandler(log_fp)
                fh.setLevel(file_level)
                fh_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(module)s:%(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                fh.setFormatter(fh_formatter)
                self.logger.addHandler(fh)
                self._log(INFO, f"File logging started: {log_fp} at level {logging.getLevelName(file_level)}")
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} E: Failed to set up file logging to {self.log_file_path_base}: {e}")

    def _log(self, level_num: int, message: str):
        if hasattr(self, 'logger'):
            if level_num == ERROR: self.logger.error(message)
            elif level_num == WARN: self.logger.warning(message)
            elif level_num == INFO: self.logger.info(message)
            elif level_num >= DEBUG_KPM: self.logger.debug(message)
        elif level_num > SILENT : 
            level_map_fallback = {ERROR: "E:", WARN: "W:", INFO: "INFO:", DEBUG_KPM: "DBG_KPM:", DEBUG_ALL: "DEBUG:"}
            print(f"{time.strftime('%H:%M:%S')} {level_map_fallback.get(level_num, f'LVL{level_num}:')} {message}")
    
    def _validate_config(self):
        if not os.path.exists(self.rapl_base_path) or not os.path.exists(self.power_limit_uw_file):
            print(f"E: RAPL path {self.rapl_base_path} or power limit file missing. Exiting."); sys.exit(1) 
        if not os.path.exists(self.energy_uj_file): self._log(WARN, f"Energy file {self.energy_uj_file} not found.")
        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'): self._log(WARN, "'ru_timing_cores' is defined but empty.")
        elif not self.ru_timing_core_indices: self._log(INFO, "No 'ru_timing_cores' defined.")
        elif self.ru_timing_core_indices:
            tc = self.ru_timing_core_indices[0]; mp = f'/dev/cpu/{tc}/msr'
            if not os.path.exists(mp): print(f"E: MSR file {mp} not found. Exiting."); sys.exit(1)
            if read_msr_direct(tc,MSR_IA32_TSC) is None: print(f"E: Failed MSR read on core {tc}. Exiting."); sys.exit(1)
            self._log(INFO, "MSR access test passed.")
        try: subprocess.run([self.intel_sst_path,"--version"],capture_output=True,check=True,text=True)
        except Exception as e: print(f"E: '{self.intel_sst_path}' failed: {e}. Exiting."); sys.exit(1)
        # ... (Other critical parameter checks with print + sys.exit if they should halt execution) ...
        self._log(INFO, "Configuration and system checks passed.")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f: return yaml.safe_load(f)
        except FileNotFoundError: print(f"E: Config file '{self.config_path}' not found. Exiting."); sys.exit(1)
        except yaml.YAMLError as e: print(f"E: Could not parse config file '{self.config_path}': {e}. Exiting."); sys.exit(1)

    def _parse_core_list_string(self, core_str: str) -> List[int]:
        cores: Set[int] = set();
        if not core_str: return []
        for part_raw in core_str.split(','):
            part = part_raw.strip();
            if not part: continue
            if '-' in part:
                try: s, e = map(int, part.split('-',1)); cores.update(range(s,e+1)) if s<=e else self._log(WARN, f"Invalid core range {s}-{e}.")
                except ValueError: self._log(WARN, f"Invalid core range format '{part}'.")
            else:
                try: cores.add(int(part))
                except ValueError: self._log(WARN, f"Invalid core number '{part}'.")
        return sorted(list(cores))

    def _update_ru_core_msr_data(self):
        if not self.ru_timing_core_indices: return
        for cid in self.ru_timing_core_indices:
            mperf, tsc = read_msr_direct(cid, MSR_IA32_MPERF), read_msr_direct(cid, MSR_IA32_TSC)
            busy = 0.0
            if cid not in self.ru_core_msr_prev_data: self.ru_core_msr_prev_data[cid] = CoreMSRData(cid)
            prev = self.ru_core_msr_prev_data[cid]
            if all(x is not None for x in [prev.mperf, prev.tsc, mperf, tsc]):
                dm, dt = mperf - prev.mperf, tsc - prev.tsc # type: ignore
                if dm < 0: dm += (2**64)
                if dt < 0: dt += (2**64)
                busy = min(100.0, 100.0 * dm / dt) if dt > 0 else prev.busy_percent
            else: busy = prev.busy_percent
            prev.mperf, prev.tsc, prev.busy_percent = mperf, tsc, busy

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        max_b = 0.0; valid = False
        for cid in self.ru_timing_core_indices:
            d = self.ru_core_msr_prev_data.get(cid)
            if d and d.busy_percent is not None: max_b = max(max_b, d.busy_percent); valid=True
        if not valid and not self.max_ru_timing_usage_history: return 0.0
        self.max_ru_timing_usage_history.append(max_b)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg: self.max_ru_timing_usage_history.pop(0)
        return sum(self.max_ru_timing_usage_history)/len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str]) -> None:
        cmd = [self.intel_sst_path] + cmd_list[1:] if cmd_list[0]=="intel-speed-select" else cmd_list
        pcmd = ' '.join(cmd)
        if self.dry_run: self._log(INFO, f"[DRY RUN] Would exec: {pcmd}"); return
        self._log(DEBUG_ALL, f"Executing: {pcmd}")
        try: subprocess.run(cmd, shell=False, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            msg = f"Cmd '{e.cmd}' fail ({e.returncode}).SOUT:{e.stdout.strip()} SERR:{e.stderr.strip()}"
            self._log(ERROR, msg);
            if not self.dry_run: raise RuntimeError(msg)
        except FileNotFoundError:
            self._log(ERROR, f"Cmd '{cmd[0]}' not found.")
            if not self.dry_run:
                raise RuntimeError(f"Cmd not found: {cmd[0]}")

    def _setup_intel_sst(self):
        self._log(INFO, "--- Configuring Intel SST-CP ---")
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])
            for cid_key, freq in self.config.get('clos_min_frequency', {}).items(): 
                self._run_command(["intel-speed-select", "core-power", "config", "-c", str(cid_key), "--min", str(freq)])
            ran_cores = {n:self._parse_core_list_string(str(cs)) for n,cs in self.config.get('ran_cores',{}).items()}
            for cid_key, comps in self.config.get('clos_association',{}).items():
                cid = int(cid_key)
                assoc_cores = set(c for comp in comps if isinstance(comps,list) for c in ran_cores.get(comp,[]))
                if cid==0 and self.ru_timing_core_indices: self._log(INFO,f"Ensuring RU_Timing cores in CLOS 0."); assoc_cores.update(self.ru_timing_core_indices)
                if assoc_cores: self._run_command(["intel-speed-select", "-c", ",".join(map(str,sorted(list(assoc_cores)))), "core-power", "assoc", "-c", str(cid)])
                elif cid==0 and self.ru_timing_core_indices and not comps: self._run_command(["intel-speed-select", "-c", ",".join(map(str,sorted(self.ru_timing_core_indices))), "core-power", "assoc", "-c", str(cid)])
                else: self._log(WARN, f"No cores for CLOS {cid}.")
            self._log(INFO, "--- Intel SST-CP Configuration Complete ---")
        except Exception as e: self._log(ERROR, f"SST-CP setup error: {e}");
            if not self.dry_run: raise RuntimeError(str(e))

    def _read_current_tdp_limit_w(self) -> float:
        if self.dry_run and self.last_tdp_adjustment_time > 0: return self.current_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f: return int(f.read().strip()) / 1e6
        except Exception: self._log(WARN, f"Could not read {self.power_limit_uw_file}. Assuming min_tdp."); return float(self.tdp_min_w)

    def _set_tdp_limit_w(self, tdp_watts: float, context: str = ""):
        clamped_uw = max(int(self.tdp_min_w*1e6), min(int(tdp_watts*1e6), int(self.tdp_max_w*1e6)))
        new_tdp_w = clamped_uw / 1e6
        if self.dry_run:
            if abs(self.current_tdp_w - new_tdp_w)>0.01: self._log(INFO, f"[DRY RUN] {context}. New Target TDP: {new_tdp_w:.1f}W")
            self.current_tdp_w = new_tdp_w; return
        try: 
            with open(self.power_limit_uw_file,'r') as f:
                if int(f.read().strip())==clamped_uw:
                    if abs(self.current_tdp_w-new_tdp_w)>0.01: self.current_tdp_w=new_tdp_w
                    if context: self._log(INFO, f"{context}. New Target TDP: {new_tdp_w:.1f}W (already set)")
                    return
        except Exception as e: self._log(WARN, f"Could not read {self.power_limit_uw_file} before write: {e}.")
        try: 
            self._log(INFO, f"{context}. New Target TDP: {new_tdp_w:.1f}W")
            with open(self.power_limit_uw_file,'w') as f: f.write(str(clamped_uw))
            self.current_tdp_w = new_tdp_w
        except OSError as e: self._log(ERROR, f"OSError writing TDP to {self.power_limit_uw_file}: {e}");
            if not self.dry_run: raise RuntimeError(str(e))
        except Exception as e: self._log(ERROR, f"Exception writing TDP: {e}");
            if not self.dry_run: raise RuntimeError(str(e))

    def _adjust_tdp(self, control_ru_cpu_usage: float):
        error = self.target_ru_cpu_usage - control_ru_cpu_usage; abs_error = abs(error) 
        sens_thresh = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_thresh = sens_thresh * self.adaptive_step_far_thresh_factor
        if abs_error > sens_thresh:
            step = self.tdp_adj_step_w_large if abs_error > far_thresh else self.tdp_adj_step_w_small
            change = -step if error > 0 else step
            if change != 0:
                ctx = f"TDP Adjust: RU CPU {control_ru_cpu_usage:.2f}%, Target {self.target_ru_cpu_usage:.2f}%. Error {error:.2f}%. Action: TDP by {change:.1f}W"
                self._set_tdp_limit_w(self.current_tdp_w + change, context=ctx)

    def _get_pkg_power_w(self) -> Tuple[float, bool]:
        if not os.path.exists(self.energy_uj_file): return 0.0, False
        try:
            with open(self.energy_uj_file, 'r') as f: current_e_uj = int(f.read().strip())
            now = time.monotonic(); pwr_w, ok = 0.0, False
            if self.last_pkg_energy_uj is not None and self.last_energy_read_time is not None:
                dt = now - self.last_energy_read_time
                if dt > 0.001:
                    de = current_e_uj - self.last_pkg_energy_uj
                    if de < 0: 
                        max_r = self.max_energy_val_rapl
                        try: 
                            with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"),'r') as f_max_r: 
                                max_r_val = int(f_max_r.read().strip());
                                if max_r_val > 0: max_r = max_r_val
                        except Exception: pass
                        de += max_r
                    pwr_w = (de / 1e6) / dt
                    if 0 <= pwr_w < 5000: ok = True
                    else: self._log(DEBUG_ALL, f"Unrealistic PkgPwr: {pwr_w:.1f}W"); ok=False; pwr_w=0.0
            if ok or self.last_pkg_energy_uj is None: self.last_pkg_energy_uj, self.last_energy_read_time = current_e_uj, now
            return pwr_w, ok
        except Exception as e: self._log(WARN, f"Exception in _get_pkg_power_w: {e}"); return 0.0, False
        
    def _read_current_energy_uj(self) -> Optional[int]:
        if not os.path.exists(self.energy_uj_file): return None
        try:
            with open(self.energy_uj_file, 'r') as f: return int(f.read().strip())
        except Exception as e: self._log(WARN, f"Could not read {self.energy_uj_file}: {e}"); return None

    def _get_interval_energy_uj(self) -> Optional[float]:
        curr_e_uj = self._read_current_energy_uj()
        if curr_e_uj is None: return None
        if self.energy_at_last_log_uj is None: self.energy_at_last_log_uj = curr_e_uj; return None 
        delta_e = float(curr_e_uj - self.energy_at_last_log_uj)
        if delta_e < 0: 
            max_r = self.max_energy_val_rapl
            try:
                with open(os.path.join(self.rapl_base_path,"max_energy_range_uj"),'r') as f_max_r: 
                    max_r_val = int(f_max_r.read().strip());
                    if max_r_val > 0: max_r = max_r_val
            except Exception as e: self._log(WARN, f"Could not read max_energy_range_uj: {e}"); pass 
            delta_e += max_r
        self.energy_at_last_log_uj = curr_e_uj
        return delta_e

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes,
                                 kpm_report_style: Optional[int] = None, ue_id: Optional[Any] = None):
        current_report_time = time.monotonic()
        current_pkg_energy_uj = self._read_current_energy_uj()

        self._log(DEBUG_KPM, f"KPM CB: Start for {e2_agent_id}, SubID {subscription_id}, Time: {current_report_time:.3f}")
        if not self.e2sm_kpm: self._log(WARN, f"KPM indication from {e2_agent_id}, but e2sm_kpm module unavailable."); return
        
        try:
            meas_report = self.e2sm_kpm.extract_meas_data(indication_msg_bytes)
            # self._log(DEBUG_KPM, f"KPM CB: Extracted meas_report from {e2_agent_id}: {meas_report}") 
            if not meas_report: self._log(WARN, f"KPM CB: Failed to extract KPM measurement data from {e2_agent_id}."); return
            
            dl_volume_kbits_in_period, ul_volume_kbits_in_period = 0.0, 0.0
            measurements = meas_report.get("measData", {})
            # self._log(DEBUG_KPM, f"KPM CB: measData from {e2_agent_id}: {measurements}") 
            if not isinstance(measurements, dict): self._log(WARN, f"KPM CB: Invalid 'measData' from {e2_agent_id}."); return

            for metric_name, value_list in measurements.items():
                if isinstance(value_list, list) and value_list:
                    value_to_convert = value_list[0] 
                    try:
                        if metric_name == 'DRB.RlcSduTransmittedVolumeDL': 
                            dl_volume_kbits_in_period = float(value_to_convert)
                        elif metric_name == 'DRB.RlcSduTransmittedVolumeUL': 
                            ul_volume_kbits_in_period = float(value_to_convert)
                    except (ValueError, TypeError): self._log(WARN, f"KPM CB: Metric '{metric_name}' value '{value_to_convert}' invalid for float().")
                else: self._log(WARN, f"KPM CB: Metric '{metric_name}' from {e2_agent_id} unexpected value format: {value_list}.")

            bits_this_kpm_interval = (dl_volume_kbits_in_period + ul_volume_kbits_in_period) * 1000.0

            # --- Per-DU Approximate Efficiency Calculation & Logging ---
            if current_pkg_energy_uj is not None:
                with self.kpm_eff_lock: 
                    if self.last_energy_reading_for_kpm_eff_uj is not None and \
                       self.last_kpm_report_time_for_eff_calc is not None:
                        
                        delta_time_s = current_report_time - self.last_kpm_report_time_for_eff_calc
                        delta_energy_uj_pkg = float(current_pkg_energy_uj - self.last_energy_reading_for_kpm_eff_uj)

                        if delta_energy_uj_pkg < 0: 
                            max_r = self.max_energy_val_rapl 
                            try: 
                                with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"),'r') as f_max_r: 
                                    max_r_val = int(f_max_r.read().strip());
                                    if max_r_val > 0: max_r = max_r_val
                            except Exception: pass
                            delta_energy_uj_pkg += max_r
                        
                        if delta_energy_uj_pkg > 1e-3 and delta_time_s > 0.001: # Use a small positive threshold for energy
                            du_efficiency_approx = bits_this_kpm_interval / delta_energy_uj_pkg
                            self._log(INFO, f"KPM InstEff ({e2_agent_id}): {bits_this_kpm_interval/1e6:.2f}Mb in ~{delta_time_s:.2f}s, PkgEnergyDelta: {delta_energy_uj_pkg/1e6:.2f}J, ApproxEff: {du_efficiency_approx:.2f} b/uJ")
                    
                    self.last_energy_reading_for_kpm_eff_uj = current_pkg_energy_uj
                    self.last_kpm_report_time_for_eff_calc = current_report_time
            # --- End Per-DU Approximate Efficiency ---

            with self.kpm_data_lock:
                if e2_agent_id not in self.kpm_gnb_tracking:
                    self.kpm_gnb_tracking[e2_agent_id] = {
                        'bits_sum_for_main_interval_dl':0.0, 'bits_sum_for_main_interval_ul':0.0,
                        'num_reports_for_main_interval':0 
                    }
                
                tracking_data = self.kpm_gnb_tracking[e2_agent_id]
                tracking_data['bits_sum_for_main_interval_dl'] += bits_this_kpm_interval 
                tracking_data['bits_sum_for_main_interval_ul'] += (ul_volume_kbits_in_period * 1000.0) 
                tracking_data['num_reports_for_main_interval'] += 1
                
                self._log(DEBUG_KPM, f"KPM CB (Delta Volume): {e2_agent_id}: Reported DL_kbit={dl_volume_kbits_in_period:.0f}, UL_kbit={ul_volume_kbits_in_period:.0f}")
                self._log(DEBUG_KPM, f"KPM CB (Delta Volume): Accumulated for main interval for {e2_agent_id}: {tracking_data}")
        
        except Exception as e: self._log(ERROR, f"Error processing KPM indication from {e2_agent_id}: {e}"); import traceback; traceback.print_exc()

    def _get_and_reset_accumulated_kpm_metrics(self) -> Dict[str, Dict[str, Any]]:
        with self.kpm_data_lock:
            snap = {}
            for gnb_id, data in self.kpm_gnb_tracking.items():
                snap[gnb_id] = {
                    'dl_bits': data.get('bits_sum_for_main_interval_dl', 0.0), 
                    'ul_bits': data.get('bits_sum_for_main_interval_ul', 0.0),
                    'reports_in_interval': data.get('num_reports_for_main_interval', 0)
                }
                data['bits_sum_for_main_interval_dl'] = 0.0
                data['bits_sum_for_main_interval_ul'] = 0.0
                data['num_reports_for_main_interval'] = 0 
        return snap

    def _setup_kpm_subscriptions(self):
        self._log(INFO, "--- Setting up KPM Subscriptions ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable."); return
        nodes = list(self.gnb_ids_map.values())
        if not nodes: self._log(WARN, "No gNB IDs for KPM."); return
        
        # These periods are for the KPM subscription itself
        kpm_report_p = int(self.config.get('kpm_report_period_ms', 1000)) 
        kpm_gran_p = int(self.config.get('kpm_granularity_period_ms', 1000))
        
        metrics = ['DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL']
        self._log(INFO, f"KPM: Subscribing to metrics: {metrics} with ReportPeriod={kpm_report_p}ms, Granularity={kpm_gran_p}ms")
        
        style = 1; successes = 0
        for node_id in nodes:
            cb_adapter = lambda ag, sb, hd, ms, st=style: self._kpm_indication_callback(ag,sb,hd,ms,kpm_report_style=st,ue_id=None)
            if self.dry_run:
                self._log(INFO, f"[DRY RUN] KPM Sub: Node {node_id}, Metrics {metrics}, Style {style}")
                with self.kpm_data_lock: 
                    if node_id not in self.kpm_gnb_tracking: 
                        self.kpm_gnb_tracking[node_id] = {
                            'bits_sum_for_main_interval_dl':0.0, 
                            'bits_sum_for_main_interval_ul':0.0, 
                            'num_reports_for_main_interval':0
                        }
                successes+=1; continue
            try:
                self._log(INFO, f"Subscribing KPM: Node {node_id}, Metrics {metrics}, Report {kpm_report_p}ms, Granularity {kpm_gran_p}ms, Style {style}")
                self.e2sm_kpm.subscribe_report_service_style_1(node_id, kpm_report_p, metrics, kpm_gran_p, cb_adapter)
                with self.kpm_data_lock: # Initialize tracking for this gNB
                    if node_id not in self.kpm_gnb_tracking: 
                        self.kpm_gnb_tracking[node_id] = {
                            'bits_sum_for_main_interval_dl':0.0, 
                            'bits_sum_for_main_interval_ul':0.0, 
                            'num_reports_for_main_interval':0
                        }
                successes+=1
            except Exception as e: self._log(ERROR, f"KPM subscription failed for {node_id}: {e}"); import traceback; traceback.print_exc()
        if successes > 0: self._log(INFO, f"--- KPM Subscriptions: {successes} nodes attempted. ---")
        elif nodes: self._log(WARN, "No KPM subscriptions initiated.")
    
    @xAppBase.start_function
    def run_power_management_xapp(self):
        if os.geteuid() != 0 and not self.dry_run: print("E: Must be root for live run."); sys.exit(1)
        try:
            # Initialize for app's main print interval energy calculation
            self.energy_at_last_log_uj = self._read_current_energy_uj()
            if self.energy_at_last_log_uj is None and not self.dry_run: self._log(WARN, "Could not get initial package energy for main interval.")
            
            # Initialize for per-KPM-interval approximate efficiency
            with self.kpm_eff_lock:
                self.last_energy_reading_for_kpm_eff_uj = self._read_current_energy_uj()
                self.last_kpm_report_time_for_eff_calc = time.monotonic()
            if self.last_energy_reading_for_kpm_eff_uj is None and not self.dry_run:
                self._log(WARN, "Could not get initial package energy for KPM-interval efficiency.")


            if self.ru_timing_core_indices: self._log(INFO, "Priming MSR..."); self._update_ru_core_msr_data(); time.sleep(0.2); self._update_ru_core_msr_data(); self._log(INFO, "MSR primed.")
            else: self._log(INFO, "No RU timing cores defined, MSR priming skipped.")
            self._log(INFO, "Attempting to set initial TDP..."); pwr, ok = self._get_pkg_power_w(); time.sleep(0.2); pwr, ok = self._get_pkg_power_w()
            if ok and pwr > 1.0: self._set_tdp_limit_w(max(self.tdp_min_w, min(pwr, self.tdp_max_w)), context=f"Initial PkgWatt {pwr:.1f}W. Initial TDP Set") 
            else: self._log(WARN, "Invalid initial PkgWatt."); self._set_tdp_limit_w(self._read_current_tdp_limit_w() if not self.dry_run else self.current_tdp_w, context="Initial TDP Set (fallback)") 
            self._log(INFO, f"Effective Initial TDP: {self.current_tdp_w:.1f}W.")
            self._setup_intel_sst(); self._setup_kpm_subscriptions() 
            self.last_tdp_adjustment_time = time.monotonic(); last_print_time = time.monotonic()
            self._log(INFO, f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            self._log(INFO, f"Target RU CPU: {self.target_ru_cpu_usage:.2f}% | RU Cores: {self.ru_timing_core_indices or 'NONE'}")
            self._log(INFO, f"TDP Update: {self.tdp_update_interval_s}s | Print: {self.print_interval_s}s | TDP Range: {self.tdp_min_w}W-{self.tdp_max_w}W")
            self._log(INFO, f"KPM Metrics: DRB.RlcSduTransmittedVolumeDL/UL from gNBs: {', '.join(self.gnb_ids_map.values()) or 'NONE'}")

            while True:
                start_time = time.monotonic()
                if self.ru_timing_core_indices: self._update_ru_core_msr_data()
                control_val = self._get_control_ru_timing_cpu_usage()
                now = time.monotonic()
                if now - self.last_tdp_adjustment_time >= self.tdp_update_interval_s and self.ru_timing_core_indices:
                    self._adjust_tdp(control_val); self.last_tdp_adjustment_time = now
                
                if now - last_print_time >= self.print_interval_s:
                    pwr_w, pwr_ok = self._get_pkg_power_w(); int_e_uj = self._get_interval_energy_uj()
                    kpm_snapshot = self._get_and_reset_accumulated_kpm_metrics() 
                    
                    dl_b, ul_b = sum(d.get('dl_bits',0.0) for d in kpm_snapshot.values()), sum(d.get('ul_bits',0.0) for d in kpm_snapshot.values())
                    tot_b = dl_b + ul_b
                    num_kpm_reports_total = sum(d.get('reports_in_interval', 0) for d in kpm_snapshot.values())

                    srv_eff = "N/A"
                    if int_e_uj is not None and int_e_uj > 1e-9 : 
                        srv_eff = f"{tot_b/int_e_uj:.2f} b/uJ"
                    elif tot_b > 1e-9 : srv_eff = "inf b/uJ"
                    else: srv_eff = "0.00 b/uJ"
                        
                    clos_eff_strs = []
                    if self.clos_to_du_names_map:
                        for cid, dus in self.clos_to_du_names_map.items():
                            clos_b = sum(kpm_snapshot.get(self.gnb_ids_map.get(du_n),{}).get('dl_bits',0.0) + 
                                         kpm_snapshot.get(self.gnb_ids_map.get(du_n),{}).get('ul_bits',0.0) 
                                         for du_n in dus if self.gnb_ids_map.get(du_n))
                            eff_str = "N/A";
                            if int_e_uj is not None and int_e_uj > 1e-9 : 
                                eff_str = f"{clos_b/int_e_uj:.2f} b/uJ"
                            elif clos_b > 1e-9 : eff_str = "inf b/uJ"
                            else: eff_str = "0.00 b/uJ"
                            clos_eff_strs.append(f"CLOS{cid}:{eff_str} ({clos_b/1e6:.2f}Mb)")
                    ru_usage = ", ".join([f"C{i}:{self.ru_core_msr_prev_data[i].busy_percent:>6.2f}%" if i in self.ru_core_msr_prev_data else f"C{i}:N/A" for i in self.ru_timing_core_indices]) or "N/A"
                    
                    pkg_pwr_log_str = f"{pwr_w:.1f}" if pwr_ok else "N/A"
                    energy_interval_j_str = f"{int_e_uj/1e6:.2f}" if int_e_uj is not None else "N/A"

                    log_parts = [f"RU:[{ru_usage}] (AvgMax:{control_val:>6.2f}%)", f"TDP:{self.current_tdp_w:>5.1f}W", 
                                 f"PkgPwr:{pkg_pwr_log_str}W", f"IntEgy:{energy_interval_j_str}J",
                                 f"KPMreps:{num_kpm_reports_total}", 
                                 f"TotBits:{tot_b/1e6:.2f}Mb", f"SrvEff:{srv_eff}", f"CLoSEff:[{' | '.join(clos_eff_strs) or 'No CLoS DUs'}]"]
                    self._log(INFO, " | ".join(log_parts)); last_print_time = now
                time.sleep(max(0, 1.0 - (time.monotonic() - start_time)))
        except KeyboardInterrupt: self._log(INFO, f"\nMonitoring loop interrupted by user.")
        except SystemExit as e: self._log(INFO, f"Application exiting: {e}"); raise 
        except RuntimeError as e: self._log(ERROR, f"Critical runtime error in loop: {e}")
        except Exception as e: self._log(ERROR, f"\nUnexpected error in loop: {e}"); import traceback; traceback.print_exc()
        finally: self._log(INFO, "--- Power Manager xApp run_power_management_xapp finished. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoRAN Power Manager with KPM xApp")
    parser.add_argument("config_path", type=str, help="Path to YAML config.")
    parser.add_argument("--http_server_port",type=int,default=8090,help="HTTP server port.")
    parser.add_argument("--rmr_port",type=int,default=4560,help="RMR port.")
    args = parser.parse_args()
    manager = None
    try:
        manager = PowerManager(args.config_path, args.http_server_port, args.rmr_port)
        if hasattr(manager, 'signal_handler') and callable(manager.signal_handler):
            signal.signal(signal.SIGINT, manager.signal_handler)
            signal.signal(signal.SIGTERM, manager.signal_handler)
            manager._log(INFO, "Registered signal handlers from xAppBase.")
        else: manager._log(WARN, "No 'signal_handler' method in PowerManager/xAppBase.")
        manager.run_power_management_xapp() 
    except RuntimeError as e: print(f"E: Failed to initialize or run PowerManager: {e}"); sys.exit(1)
    except SystemExit as e: print(f"Application terminated: {e}"); sys.exit(0 if str(e) == "0" else 1)
    except Exception as e: print(f"E: An unexpected error at top level: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    finally:
        if manager and hasattr(manager, '_log'): manager._log(INFO,"Application finished.")
        else: print("INFO: Application finished.")
