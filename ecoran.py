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
import logging # Added for file logging
from logging.handlers import RotatingFileHandler # Optional: for log rotation

try:
    from lib.xAppBase import xAppBase 
except ImportError:
    print("E: Failed to import xAppBase from lib.xAppBase. Ensure the library is correctly installed and accessible.")
    sys.exit(1)

# MSR Addresses
MSR_IA32_TSC = 0x10
MSR_IA32_MPERF = 0xE7

# Verbosity levels (used for config values)
SILENT = 0
ERROR = 1
WARN = 2
INFO = 3
DEBUG_KPM = 4
DEBUG_ALL = 5

# Map verbosity config values to logging levels
LOGGING_LEVEL_MAP = {
    SILENT: logging.CRITICAL + 10, # Higher than critical to silence it
    ERROR: logging.ERROR,
    WARN: logging.WARNING,
    INFO: logging.INFO,
    DEBUG_KPM: logging.DEBUG, # Map KPM debug to general DEBUG
    DEBUG_ALL: logging.DEBUG  # Map all debug to general DEBUG
}


def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
    # ... (no changes) ...
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            msr_val_bytes = f.read(8) 
            if len(msr_val_bytes) == 8:
                return struct.unpack('<Q', msr_val_bytes)[0]
            else: return None
    except FileNotFoundError: return None
    except PermissionError: return None
    except OSError as e:
        if e.errno != 2 and e.errno != 13:
             print(f"W: OSError reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"E: Unexpected error reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None

class CoreMSRData:
    # ... (no changes) ...
    def __init__(self, core_id: int):
        self.core_id = core_id
        self.mperf: Optional[int] = None
        self.tsc: Optional[int] = None
        self.busy_percent: float = 0.0

class PowerManager(xAppBase):
    MAX_VOLUME_COUNTER_KBITS = (2**32) - 1 

    def __init__(self, config_path: str, http_server_port: int, rmr_port: int, kpm_ran_func_id: int = 2):
        self.config_path = config_path
        self.config = self._load_config() # Load config first to get logging params
        
        # Initialize logging
        self.logger = logging.getLogger("EcoRANPowerManager")
        self.console_verbosity = int(self.config.get('console_verbosity_level', INFO))
        self.file_verbosity = int(self.config.get('file_verbosity_level', DEBUG_KPM)) # Log more to file by default
        self.log_file_path_base = self.config.get('log_file_path', "/mnt/data/ecoran")
        self._setup_logging()

        # Now call super and initialize other attributes
        xapp_base_config_file = self.config.get('xapp_base_config_file', '')
        super().__init__(xapp_base_config_file, http_server_port, rmr_port)

        # ... (rest of __init__ as before) ...
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
        self.max_ru_timing_usage_history: List[float] = []
        self.last_tdp_adjustment_time: float = 0.0
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}
        self.kpm_ran_func_id = kpm_ran_func_id
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None:
            self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else:
            self._log(WARN, "xAppBase.e2sm_kpm module not found or not initialized. KPM functionality will be disabled.")
            self.e2sm_kpm = None
        self.gnb_ids_map = self.config.get('gnb_ids', {}) 
        self.gnb_id_to_du_name_map = {v: k for k, v in self.gnb_ids_map.items()}
        clos_association_config = self.config.get('clos_association', {})
        self.clos_to_du_names_map: Dict[int, List[str]] = {}
        ran_components_in_config = self.config.get('ran_cores', {}).keys()
        for clos_id, components in clos_association_config.items():
            if not isinstance(components, list):
                self._log(WARN, f"Components for CLOS {clos_id} is not a list in config. Skipping.")
                continue
            self.clos_to_du_names_map[int(clos_id)] = [
                comp for comp in components if comp in ran_components_in_config and comp.startswith('du')
            ]
        self.accumulated_kpm_metrics: Dict[str, Dict[str, Any]] = {} 
        self.kpm_data_lock = threading.Lock()
        self.energy_at_last_log_uj: Optional[int] = None 
        self._validate_config()
        if self.dry_run:
            self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _setup_logging(self):
        self.logger.handlers = [] # Clear any existing handlers
        self.logger.propagate = False # Prevent double logging if root logger is configured
        
        # Determine overall minimum level for the logger itself
        # The logger's level must be <= the lowest level of its handlers
        effective_console_level = LOGGING_LEVEL_MAP.get(self.console_verbosity, logging.INFO)
        effective_file_level = LOGGING_LEVEL_MAP.get(self.file_verbosity, logging.DEBUG)
        
        # Set logger level to the more verbose of the two handlers, or INFO if both are silent
        if self.console_verbosity == SILENT and self.file_verbosity == SILENT:
            self.logger.setLevel(logging.CRITICAL + 10) # Effectively silent
        else:
            self.logger.setLevel(min(effective_console_level, effective_file_level))


        # Console Handler
        if self.console_verbosity > SILENT:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(effective_console_level)
            formatter = logging.Formatter('%(asctime)s %(levelname).1s: %(message)s', datefmt='%H:%M:%S')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # File Handler
        if self.file_verbosity > SILENT and self.log_file_path_base:
            try:
                os.makedirs(self.log_file_path_base, exist_ok=True)
                log_filename = f"ecoran_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
                log_filepath = os.path.join(self.log_file_path_base, log_filename)
                
                # fh = RotatingFileHandler(log_filepath, maxBytes=5*1024*1024, backupCount=3) # Optional: 5MB, 3 backups
                fh = logging.FileHandler(log_filepath)
                fh.setLevel(effective_file_level)
                file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                fh.setFormatter(file_formatter)
                self.logger.addHandler(fh)
                self_initial_log_msg = f"File logging started: {log_filepath} at level {logging.getLevelName(effective_file_level)}"
                if self.console_verbosity > SILENT : print(f"{time.strftime('%H:%M:%S')} INFO: {self_initial_log_msg}") # Also print to console
                self.logger.info(self_initial_log_msg) # Log to file

            except Exception as e:
                print(f"E: Failed to set up file logging to {self.log_file_path_base}: {e}")


    def _log(self, level: int, message: str):
        # Map custom verbosity to logging levels
        if level == ERROR: self.logger.error(message)
        elif level == WARN: self.logger.warning(message)
        elif level == INFO: self.logger.info(message)
        elif level >= DEBUG_KPM: # Both DEBUG_KPM and DEBUG_ALL map to logger.debug
            self.logger.debug(message)
        # SILENT level effectively does nothing if logger/handler levels are set higher

    # ... (rest of the methods from previous version, e.g., _validate_config, _load_config, etc.) ...
    # Make sure they use self._log(LEVEL, "message") instead of print()
    # Example for _validate_config first lines:
    def _validate_config(self):
        if not os.path.exists(self.rapl_base_path) or not os.path.exists(self.power_limit_uw_file):
            # Critical errors that prevent startup should still print to stderr and exit
            print(f"E: RAPL path {self.rapl_base_path} or power limit file missing. Exiting."); sys.exit(1) 
        if not os.path.exists(self.energy_uj_file):
             self._log(WARN, f"Energy file {self.energy_uj_file} not found. PkgPower/Efficiency readings will be N/A.")
        # ... rest of _validate_config using self._log ...
        # ... (Ensure other methods like _parse_core_list_string, _run_command also use self._log) ...
        # ... (the full class from previous message, just ensure prints are replaced by self._log) ...

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
        for core_id in self.ru_timing_core_indices:
            mperf, tsc = read_msr_direct(core_id, MSR_IA32_MPERF), read_msr_direct(core_id, MSR_IA32_TSC)
            busy = 0.0
            if core_id not in self.ru_core_msr_prev_data: self.ru_core_msr_prev_data[core_id] = CoreMSRData(core_id)
            prev = self.ru_core_msr_prev_data[core_id]
            if all(x is not None for x in [prev.mperf, prev.tsc, mperf, tsc]):
                dmperf, dtsc = mperf - prev.mperf, tsc - prev.tsc # type: ignore
                if dmperf < 0: dmperf += (2**64)
                if dtsc < 0: dtsc += (2**64)
                if dtsc > 0: busy = min(100.0, 100.0 * dmperf / dtsc)
                else: busy = prev.busy_percent
            else: busy = prev.busy_percent
            prev.mperf, prev.tsc, prev.busy_percent = mperf, tsc, busy

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        max_busy = 0.0; valid_data = False
        for core_id in self.ru_timing_core_indices:
            data = self.ru_core_msr_prev_data.get(core_id)
            if data and data.busy_percent is not None: max_busy = max(max_busy, data.busy_percent); valid_data = True
        if not valid_data and not self.max_ru_timing_usage_history: return 0.0
        self.max_ru_timing_usage_history.append(max_busy)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg: self.max_ru_timing_usage_history.pop(0)
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str]) -> None:
        cmd_str_list = [self.intel_sst_path] + cmd_list[1:] if cmd_list[0] == "intel-speed-select" else cmd_list
        printable_cmd = ' '.join(cmd_str_list)
        if self.dry_run: self._log(INFO, f"[DRY RUN] Would execute: {printable_cmd}"); return
        self._log(DEBUG_ALL, f"Executing: {printable_cmd}")
        try: subprocess.run(cmd_str_list, shell=False, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            msg = f"Cmd '{e.cmd}' failed ({e.returncode}). STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}"
            self._log(ERROR, msg)
            if not self.dry_run: raise RuntimeError(msg)
        except FileNotFoundError:
            msg = f"Cmd '{cmd_str_list[0]}' not found."
            self._log(ERROR, msg)
            if not self.dry_run: raise RuntimeError(msg)

    def _setup_intel_sst(self):
        self._log(INFO, "--- Configuring Intel SST-CP ---")
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])
            for cid_key, freq in self.config.get('clos_min_frequency', {}).items(): 
                self._run_command(["intel-speed-select", "core-power", "config", "-c", str(cid_key), "--min", str(freq)])
            ran_cores = {n: self._parse_core_list_string(str(cs)) for n, cs in self.config.get('ran_cores', {}).items()}
            for cid_key, comps in self.config.get('clos_association', {}).items():
                cid = int(cid_key)
                assoc_cores = set(c for comp in comps if isinstance(comps,list) for c in ran_cores.get(comp,[]))
                if cid == 0 and self.ru_timing_core_indices: 
                    self._log(INFO, f"Ensuring RU_Timing cores {self.ru_timing_core_indices} in CLOS 0.")
                    assoc_cores.update(self.ru_timing_core_indices)
                if assoc_cores: 
                    self._run_command(["intel-speed-select", "-c", ",".join(map(str, sorted(list(assoc_cores)))), "core-power", "assoc", "-c", str(cid)])
                elif cid == 0 and self.ru_timing_core_indices and not comps: 
                    self._log(INFO, f"Assigning only RU_Timing cores {self.ru_timing_core_indices} to CLOS 0.")
                    self._run_command(["intel-speed-select", "-c", ",".join(map(str, sorted(self.ru_timing_core_indices))), "core-power", "assoc", "-c", str(cid)])
                else: 
                    self._log(WARN, f"No cores for CLOS {cid}.")
            self._log(INFO, "--- Intel SST-CP Configuration Complete ---")
        except Exception as e: 
            self._log(ERROR, f"Intel SST-CP setup error: {e}")
            if not self.dry_run: 
                raise RuntimeError(str(e))

    def _read_current_tdp_limit_w(self) -> float:
        if self.dry_run and self.last_tdp_adjustment_time > 0: return self.current_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f: return int(f.read().strip()) / 1e6
        except Exception: self._log(WARN, f"Could not read {self.power_limit_uw_file}. Assuming min_tdp."); return float(self.tdp_min_w)

    def _set_tdp_limit_w(self, tdp_watts: float, context: str = ""):
        clamped_uw = max(int(self.tdp_min_w*1e6), min(int(tdp_watts*1e6), int(self.tdp_max_w*1e6)))
        new_tdp_w = clamped_uw / 1e6
        if self.dry_run:
            if abs(self.current_tdp_w - new_tdp_w) > 0.01: self._log(INFO, f"[DRY RUN] {context}. New Target TDP: {new_tdp_w:.1f}W")
            self.current_tdp_w = new_tdp_w; return
        try: 
            with open(self.power_limit_uw_file, 'r') as f:
                if int(f.read().strip()) == clamped_uw:
                    if abs(self.current_tdp_w - new_tdp_w)>0.01: self.current_tdp_w = new_tdp_w
                    if context: self._log(INFO, f"{context}. New Target TDP: {new_tdp_w:.1f}W (already set)")
                    return
        except Exception as e: self._log(WARN, f"Could not read {self.power_limit_uw_file} before write: {e}.")
        try: 
            self._log(INFO, f"{context}. New Target TDP: {new_tdp_w:.1f}W")
            with open(self.power_limit_uw_file, 'w') as f: f.write(str(clamped_uw))
            self.current_tdp_w = new_tdp_w
        except OSError as e: 
            self._log(ERROR, f"OSError writing TDP to {self.power_limit_uw_file}: {e}")
            if not self.dry_run: 
                raise RuntimeError(str(e))
        except Exception as e: 
            self._log(ERROR, f"Exception writing TDP: {e}")
            if not self.dry_run: 
                raise RuntimeError(str(e))

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
                                max_r_val = int(f_max_r.read().strip())
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
                    max_r_val = int(f_max_r.read().strip())
                    if max_r_val > 0: max_r = max_r_val
            except Exception as e: self._log(WARN, f"Could not read max_energy_range_uj: {e}"); pass 
            delta_e += max_r
        self.energy_at_last_log_uj = curr_e_uj
        return delta_e

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes,
                                 kpm_report_style: Optional[int] = None, ue_id: Optional[Any] = None):
        self._log(DEBUG_KPM, f"KPM CB: Received from {e2_agent_id}, SubID {subscription_id}, Style {kpm_report_style}, UE {ue_id}")
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

            delta_dl_bits = dl_volume_kbits_in_period * 1000.0 
            delta_ul_bits = ul_volume_kbits_in_period * 1000.0

            with self.kpm_data_lock:
                if e2_agent_id not in self.accumulated_kpm_metrics:
                    self.accumulated_kpm_metrics[e2_agent_id] = {
                        'dl_bits_interval_sum':0.0, 'ul_bits_interval_sum':0.0,
                        'num_reports_processed':0 
                    }
                
                acc_data = self.accumulated_kpm_metrics[e2_agent_id]
                
                acc_data['dl_bits_interval_sum'] += delta_dl_bits
                acc_data['ul_bits_interval_sum'] += delta_ul_bits
                acc_data['num_reports_processed'] += 1 # Increment for every KPM report
                
                self._log(DEBUG_KPM, f"KPM CB (Delta Volume): {e2_agent_id}: Reported DL_kbit={dl_volume_kbits_in_period:.0f}, UL_kbit={ul_volume_kbits_in_period:.0f} => Added Ddl_b={delta_dl_bits:.0f}, Dul_b={delta_ul_bits:.0f}")
                self._log(DEBUG_KPM, f"KPM CB (Delta Volume): Accumulated for {e2_agent_id}: {acc_data}")
        
        except Exception as e: self._log(ERROR, f"Error processing KPM indication from {e2_agent_id}: {e}"); import traceback; traceback.print_exc()

    def _get_and_reset_accumulated_kpm_metrics(self) -> Dict[str, Dict[str, Any]]:
        with self.kpm_data_lock:
            snap = {}
            for gnb_id, data in self.accumulated_kpm_metrics.items():
                snap[gnb_id] = {
                    'dl_bits': data.get('dl_bits_interval_sum', 0.0), 
                    'ul_bits': data.get('ul_bits_interval_sum', 0.0),
                    'reports_in_interval': data.get('num_reports_processed', 0) # Include reports count
                }
                data['dl_bits_interval_sum'] = 0.0
                data['ul_bits_interval_sum'] = 0.0
                data['num_reports_processed'] = 0 # Reset for the next print interval
        return snap

    def _setup_kpm_subscriptions(self):
        self._log(INFO, "--- Setting up KPM Subscriptions ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable."); return
        nodes = list(self.gnb_ids_map.values())
        if not nodes: self._log(WARN, "No gNB IDs for KPM."); return
        metrics = ['DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL']
        self._log(INFO, f"KPM: Subscribing to metrics: {metrics}")
        rep_ms, gran_ms = int(self.config.get('kpm_report_period_ms',1000)), int(self.config.get('kpm_granularity_period_ms',1000))
        style = 1; successes = 0
        for node_id in nodes:
            cb_adapter = lambda ag, sb, hd, ms, st=style: self._kpm_indication_callback(ag,sb,hd,ms,kpm_report_style=st,ue_id=None)
            if self.dry_run:
                self._log(INFO, f"[DRY RUN] KPM Sub: Node {node_id}, Metrics {metrics}, Style {style}")
                with self.kpm_data_lock: 
                    if node_id not in self.accumulated_kpm_metrics: 
                        self.accumulated_kpm_metrics[node_id] = {'dl_bits_interval_sum':0.0, 'ul_bits_interval_sum':0.0, 'num_reports_processed':0}
                successes+=1; continue
            try:
                self._log(INFO, f"Subscribing KPM: Node {node_id}, Metrics {metrics}, Report {rep_ms}ms, Granularity {gran_ms}ms, Style {style}")
                self.e2sm_kpm.subscribe_report_service_style_1(node_id, rep_ms, metrics, gran_ms, cb_adapter)
                with self.kpm_data_lock:
                    if node_id not in self.accumulated_kpm_metrics: 
                        self.accumulated_kpm_metrics[node_id] = {'dl_bits_interval_sum':0.0, 'ul_bits_interval_sum':0.0, 'num_reports_processed':0}
                successes+=1
            except Exception as e: self._log(ERROR, f"KPM subscription failed for {node_id}: {e}"); import traceback; traceback.print_exc()
        if successes > 0: self._log(INFO, f"--- KPM Subscriptions: {successes} nodes attempted. ---")
        elif nodes: self._log(WARN, "No KPM subscriptions initiated.")
    
    @xAppBase.start_function
    def run_power_management_xapp(self):
        if os.geteuid() != 0 and not self.dry_run: print("E: Must be root for live run."); sys.exit(1)
        try:
            self.energy_at_last_log_uj = self._read_current_energy_uj()
            if self.energy_at_last_log_uj is None and not self.dry_run: self._log(WARN, "Could not get initial package energy.")
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
                    if int_e_uj is not None: srv_eff = f"{tot_b/int_e_uj:.2f} b/uJ" if int_e_uj>1e-9 else ("inf b/uJ" if tot_b>1e-9 else "0.00 b/uJ")
                    clos_eff_strs = []
                    if self.clos_to_du_names_map:
                        for cid, dus in self.clos_to_du_names_map.items():
                            clos_b = sum(kpm_snapshot.get(self.gnb_ids_map.get(du_n),{}).get('dl_bits',0.0) + 
                                         kpm_snapshot.get(self.gnb_ids_map.get(du_n),{}).get('ul_bits',0.0) 
                                         for du_n in dus if self.gnb_ids_map.get(du_n))
                            eff_str = "N/A";
                            if int_e_uj is not None: eff_str = f"{clos_b/int_e_uj:.2f} b/uJ" if int_e_uj>1e-9 else ("inf b/uJ" if clos_b>1e-9 else "0.00 b/uJ")
                            clos_eff_strs.append(f"CLOS{cid}:{eff_str} ({clos_b/1e6:.2f}Mb)")
                    ru_usage = ", ".join([f"C{i}:{self.ru_core_msr_prev_data[i].busy_percent:>6.2f}%" if i in self.ru_core_msr_prev_data else f"C{i}:N/A" for i in self.ru_timing_core_indices]) or "N/A"
                    
                    pkg_pwr_log_str = f"{pwr_w:.1f}" if pwr_ok else "N/A"
                    energy_interval_j_str = f"{int_e_uj/1e6:.2f}" if int_e_uj is not None else "N/A"

                    log_parts = [f"RU:[{ru_usage}] (AvgMax:{control_val:>6.2f}%)", f"TDP:{self.current_tdp_w:>5.1f}W", 
                                 f"PkgPwr:{pkg_pwr_log_str}W", f"IntEgy:{energy_interval_j_str}J",
                                 f"KPMreps:{num_kpm_reports_total}", # Added KPM report count
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
    parser.add_argument("--http_server_port",type=int,default=8090,help="HTTP server port.") # Corrected default
    parser.add_argument("--rmr_port",type=int,default=4560,help="RMR port.") # Corrected default
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
