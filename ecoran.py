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
import random
import math
import numpy as np
import json

# Attempt to import contextualbandits library
try:
    from contextualbandits.online import LinUCB # Correct import path
except ImportError:
    print("E: Failed to import LinUCB from contextualbandits.online. Please install the library: pip install contextualbandits")
    sys.exit(1)

try:
    from lib.xAppBase import xAppBase
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from lib.xAppBase import xAppBase
    except ImportError:
        print("E: Failed to import xAppBase from lib.xAppBase. Ensure library is accessible.")
        sys.exit(1)

# MSR Addresses
MSR_IA32_TSC = 0x10
MSR_IA32_MPERF = 0xE7

# Verbosity levels
SILENT = 0; ERROR = 1; WARN = 2; INFO = 3; DEBUG_KPM = 4; DEBUG_ALL = 5
LOGGING_LEVEL_MAP = {
    SILENT: logging.CRITICAL + 10, ERROR: logging.ERROR, WARN: logging.WARNING,
    INFO: logging.INFO, DEBUG_KPM: logging.DEBUG, DEBUG_ALL: logging.DEBUG
}

# --- Context Feature Indices (Ensure this matches _get_current_context_vector order) ---
# This definition helps in understanding the context vector but is not strictly enforced by index in code below.
# The order in _get_current_context_vector and the configured context_dimension_features_only are key.
# CTX_IDX_BIAS = 0
# CTX_IDX_TOTAL_BITS_DL_NORM = 1
# CTX_IDX_TOTAL_BITS_UL_NORM = 2
# CTX_IDX_PRB_TOT_DL_NORM = 3
# CTX_IDX_PRB_TOT_UL_NORM = 4
# CTX_IDX_NUM_ACTIVE_UES_NORM = 5
# CTX_IDX_NUM_ACTIVE_DUS_NORM = 6
# CTX_IDX_RU_CPU_NORM = 7
# CTX_IDX_CURRENT_TDP_NORM = 8
# context_dimension_features_only would be 9 with these features

def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]: # Same as before
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            val_bytes = f.read(8)
            return struct.unpack('<Q', val_bytes)[0] if len(val_bytes) == 8 else None
    except FileNotFoundError: return None
    except PermissionError: return None
    except OSError as e:
        if e.errno != 2 and e.errno != 13 and e.errno != 19:
             print(f"W: OSError reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"E: Unexpected error reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None

class CoreMSRData: # Same as before
    def __init__(self, core_id: int):
        self.core_id, self.mperf, self.tsc, self.busy_percent = core_id, None, None, 0.0

# Our custom LinUCB class is now replaced by the library version.
# We'll instantiate the library's LinUCB in PowerManager.

class PowerManager(xAppBase):
    MAX_VOLUME_COUNTER_KBITS = (2**32) - 1 # ~4 Tb, RLC SDU Volume is often 32-bit kbits

    def __init__(self, config_path: str, http_server_port: int, rmr_port: int, kpm_ran_func_id: int = 2):
        self.config_path = config_path
        self.config = self._load_config()

        self.verbosity = int(self.config.get('console_verbosity_level', INFO))
        self.file_verbosity_cfg = int(self.config.get('file_verbosity_level', DEBUG_KPM))
        self.log_file_path_base = self.config.get('log_file_path', "/mnt/data/ecoran")
        self._setup_logging()

        xapp_base_config_file = self.config.get('xapp_base_config_file', '')
        super().__init__(xapp_base_config_file, http_server_port, rmr_port)

        # --- System Paths and Parameters ---
        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        self.max_energy_val_rapl = self.config.get('rapl_max_energy_uj_override', 2**60 - 1)

        # --- Timing & Control Loop Parameters ---
        self.main_loop_sleep_s = float(self.config.get('main_loop_sleep_s', 0.1))
        self.ru_timing_pid_interval_s = float(self.config.get('ru_timing_pid_interval_s', 1.0))
        self.optimizer_decision_interval_s = float(self.config.get('optimizer_decision_interval_s', 10.0))
        self.stats_print_interval_s = float(self.config.get('stats_print_interval_s', self.optimizer_decision_interval_s))

        # --- TDP Management Parameters ---
        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 90))
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 170))
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 99.5))
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.0005))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1.0))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 3.0))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 1.5))

        # --- System State Variables ---
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        self.dry_run = bool(self.config.get('dry_run', False))
        self.current_tdp_w = float(self.tdp_min_w) # Will be updated from HW
        self.last_pkg_energy_uj: Optional[int] = None
        self.last_energy_read_time: Optional[float] = None
        self.energy_at_last_optimizer_interval_uj: Optional[int] = None
        self.max_ru_timing_usage_history: List[float] = []
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}

        # --- KPM Setup ---
        self.kpm_ran_func_id = kpm_ran_func_id
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None: self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else: self._log(WARN, "xAppBase.e2sm_kpm module unavailable."); self.e2sm_kpm = None
        self.gnb_ids_map = self.config.get('gnb_ids', {}) # Maps DU name to gNB E2NodeID
        self.kpm_data_lock = threading.Lock()
        # Stores summed per-gNB data derived from Style 4 per-UE reports
        self.accumulated_kpm_metrics: Dict[str, Dict[str, Any]] = {}
        # Stores unique UE IDs seen in current optimizer interval (globally unique: gnbID_localUEID)
        self.current_interval_ue_ids: Set[str] = set()
        # Stores per-UE data for calculating active UEs etc. within an interval
        self.current_interval_per_ue_data: Dict[str, Dict[str, float]] = {} # Key: global_ue_id, Value: {metric: value}

        # --- Contextual Bandit (using `contextualbandits` library) ---
        cb_config = self.config.get('contextual_bandit', {})
        bandit_actions_w_str = cb_config.get('actions_tdp_delta_w', {"dec_10": -10.0, "dec_5": -5.0, "hold": 0.0, "inc_5": 5.0, "inc_10": 10.0})
        self.bandit_actions: Dict[str, float] = {k: float(v) for k, v in bandit_actions_w_str.items()}
        self.arm_keys_ordered = list(self.bandit_actions.keys()) 
        if "hold" not in self.bandit_actions:
            self.bandit_actions["hold"] = 0.0
            if "hold" not in self.arm_keys_ordered: self.arm_keys_ordered.append("hold")
        
        # THIS IS THE KEY ATTRIBUTE for number of features we prepare (excluding bias if lib handles it)
        self.context_dimension_features_only = int(cb_config.get('context_dimension_features_only', 8)) 
        self.linucb_alpha = float(cb_config.get('alpha', 1.0)) 
        self.linucb_lambda_ = float(cb_config.get('lambda_', 0.1)) 
        self.linucb_fit_intercept = bool(cb_config.get('fit_intercept', True)) # Default to True

        self._log(INFO, f"Initializing LinUCB with nchoices={len(self.arm_keys_ordered)}, alpha={self.linucb_alpha}, lambda_={self.linucb_lambda_}, fit_intercept={self.linucb_fit_intercept}")
        
        # The library's LinUCB infers ndim from data.
        # If fit_intercept is True, library handles bias, pass ndim as actual feature count.
        # If fit_intercept is False, we add bias, so ndim passed to library is feature_count + 1.
        ndim_for_lib_init = self.context_dimension_features_only
        if not self.linucb_fit_intercept:
            ndim_for_lib_init +=1 # We will be adding a bias term manually

        self.contextual_bandit_model = LinUCB(
            nchoices=len(self.arm_keys_ordered),
            alpha=self.linucb_alpha,
            lambda_=self.linucb_lambda_,
            fit_intercept=self.linucb_fit_intercept 
            # Note: The library's LinUCB does NOT take 'ndim' as a constructor argument.
            # It infers it. My previous manual LinUCB did.
            # The `ndim` parameter in some contextualbandits policies is for the *internal representation*
            # and might be used differently. For LinUCB from this library, we don't pass it at init.
        )
        self.optimizer_target_tdp_w = self.current_tdp_w
        self.last_selected_arm_index: Optional[int] = None
        self.last_context_vector: Optional[np.array] = None
        self.total_bits_from_previous_optimizer_interval: Optional[float] = None
        self.throughput_change_threshold_for_discard = float(cb_config.get('throughput_change_threshold_for_discard', 1.0))
        self.active_ue_throughput_threshold_mbps = float(cb_config.get('active_ue_throughput_threshold_mbps', 1.0))

        self.norm_params = cb_config.get('normalization_parameters', {})
        self._ensure_default_norm_params()


        # --- Timestamps and Logging Variables ---
        self.last_ru_pid_run_time: float = 0.0
        self.last_optimizer_run_time: float = 0.0
        self.last_stats_print_time: float = 0.0
        self.most_recent_calculated_efficiency_for_log: Optional[float] = None
        self.current_num_active_ues_for_log: int = 0

        self._validate_config()
        if self.dry_run: self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _ensure_default_norm_params(self):
        """Ensure essential normalization parameters have defaults if not in config."""
        defaults = {
            'bias': {'min': 1.0, 'max': 1.0},
            'total_bits_dl_per_second': {'min': 0.0, 'max': 1e9}, # 1 Gbps
            'total_bits_ul_per_second': {'min': 0.0, 'max': 1e9}, # 1 Gbps
            'prb_total_dl_percentage': {'min': 0.0, 'max': float(len(self.gnb_ids_map) or 1.0) * 100.0}, # Max 100% per DU
            'prb_total_ul_percentage': {'min': 0.0, 'max': float(len(self.gnb_ids_map) or 1.0) * 100.0},
            'num_active_ues': {'min': 0, 'max': 100.0},
            'num_active_dus': {'min': 0, 'max': float(len(self.gnb_ids_map) or 1.0)},
            'ru_cpu_usage': {'min': 80.0, 'max': 100.0},
            'current_tdp': {'min': float(self.tdp_min_w), 'max': float(self.tdp_max_w)}
        }
        for key, default_val in defaults.items():
            if key not in self.norm_params:
                self.norm_params[key] = default_val
                self._log(INFO, f"Normalization param for '{key}' not in config, using default: {default_val}")


    def _normalize_feature(self, value: float, feature_key: str) -> float: # Same as before
        params = self.norm_params.get(feature_key)
        if not params:
            self._log(WARN, f"Normalization params not found for {feature_key}. Returning raw value: {value}")
            return value 
        val_min = float(params.get('min', 0.0))
        val_max = float(params.get('max', 1.0))
        if val_max == val_min: 
            return 0.5 if value == val_min else (0.0 if value < val_min else 1.0)
        normalized = (value - val_min) / (val_max - val_min)
        return max(0.0, min(1.0, normalized)) 
    
    
    def _get_current_context_vector(self,
                                   current_total_bits_dl_interval: float,
                                   current_total_bits_ul_interval: float,
                                   current_prb_dl_total_percentage: float,
                                   current_prb_ul_total_percentage: float,
                                   current_num_active_ues: int,
                                   current_num_active_dus: int,
                                   current_ru_cpu_avg: float,
                                   current_actual_tdp: float) -> np.array:
        
        interval_s = self.optimizer_decision_interval_s
        if interval_s <=0: interval_s = 1.0 
    
        bits_dl_ps = current_total_bits_dl_interval / interval_s if interval_s > 0 else 0.0
        bits_ul_ps = current_total_bits_ul_interval / interval_s if interval_s > 0 else 0.0
    
        # Order of features must be consistent
        # NO explicit bias term if LinUCB's fit_intercept=True
        feature_values_ordered = [
            bits_dl_ps,
            bits_ul_ps,
            current_prb_dl_total_percentage,
            current_prb_ul_total_percentage,
            float(current_num_active_ues),
            float(current_num_active_dus),
            current_ru_cpu_avg,
            current_actual_tdp
        ]
        feature_keys_ordered = [
            'total_bits_dl_per_second',
            'total_bits_ul_per_second',
            'prb_total_dl_percentage',
            'prb_total_ul_percentage',
            'num_active_ues',
            'num_active_dus',
            'ru_cpu_usage',
            'current_tdp'
        ]
    
        # The configured self.context_dimension_features_only should match len(feature_values_ordered)
        if len(feature_values_ordered) != self.context_dimension_features_only:
            self._log(ERROR, f"Number of actual features ({len(feature_values_ordered)}) "
                             f"does not match configured context_dimension_features_only ({self.context_dimension_features_only}). "
                             "Check feature list and config.")
            # Fallback to avoid crash, but this indicates a config/code mismatch
            return np.ones(self.context_dimension_features_only) * 0.5 
    
        normalized_features = np.array([
            self._normalize_feature(val, key) for val, key in zip(feature_values_ordered, feature_keys_ordered)
        ])
        
        return normalized_features
    
    # ... (Logging, _load_config, _parse_core_list_string, MSR methods, _run_command, _setup_intel_sst, RAPL methods, _run_ru_timing_pid_step, _get_pkg_power_w, _read_current_energy_uj, _get_interval_energy_uj_for_optimizer are UNCHANGED from the final KPM Style 4 version)
    # For brevity, I'll copy a few and indicate others are the same.
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
                self.logger.info(f"File logging started: {log_fp} at level {logging.getLevelName(file_level)}")
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} E: Failed to set up file logging to {self.log_file_path_base}: {e}")
                if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers) and self.verbosity > SILENT:
                    ch_fallback = logging.StreamHandler(sys.stdout)
                    ch_fallback.setLevel(console_level)
                    ch_formatter_fallback = logging.Formatter('%(asctime)s %(levelname).1s: %(message)s', datefmt='%H:%M:%S')
                    ch_fallback.setFormatter(ch_formatter_fallback)
                    self.logger.addHandler(ch_fallback)
                    self.logger.warning("File logging failed, using console fallback for logger.")


    def _log(self, level_num: int, message: str):
        if hasattr(self, 'logger') and self.logger.handlers: 
            if level_num == ERROR: self.logger.error(message)
            elif level_num == WARN: self.logger.warning(message)
            elif level_num == INFO: self.logger.info(message)
            elif level_num >= DEBUG_KPM: self.logger.debug(message)
        elif level_num > SILENT : 
            level_map_fallback = {ERROR: "E:", WARN: "W:", INFO: "INFO:", DEBUG_KPM: "DBG_KPM:", DEBUG_ALL: "DEBUG:"}
            print(f"{time.strftime('%H:%M:%S')} {level_map_fallback.get(level_num, f'LVL{level_num}:')} {message}")

    def _validate_config(self):
        if not os.path.exists(self.rapl_base_path) or not os.path.exists(self.power_limit_uw_file):
            self._log(ERROR, f"RAPL path {self.rapl_base_path} or power limit file missing. Exiting."); sys.exit(1)
        if not os.path.exists(self.energy_uj_file): self._log(WARN, f"Energy file {self.energy_uj_file} not found.")
        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'): self._log(WARN, "'ru_timing_cores' is defined but empty.")
        elif not self.ru_timing_core_indices: self._log(INFO, "No 'ru_timing_cores' defined, RU Timing PID will be disabled.")
        elif self.ru_timing_core_indices:
            tc = self.ru_timing_core_indices[0]; mp = f'/dev/cpu/{tc}/msr'
            if not os.path.exists(mp): self._log(ERROR, f"MSR file {mp} not found. Exiting."); sys.exit(1)
            if read_msr_direct(tc,MSR_IA32_TSC) is None: self._log(ERROR, f"Failed MSR read on core {tc}. Exiting."); sys.exit(1)
            self._log(INFO, "MSR access test passed.")
        try: subprocess.run([self.intel_sst_path,"--version"],capture_output=True,check=True,text=True)
        except Exception as e: self._log(ERROR, f"'{self.intel_sst_path}' failed: {e}. Exiting."); sys.exit(1)
        if 'bias' not in self.norm_params: # Ensure bias normalization is present for context vector
            self.norm_params['bias'] = {'min': 1.0, 'max': 1.0} # Bias is always 1, so normalized is 1 (or 0.5 if min=max=0)
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
                dm, dt = mperf - prev.mperf, tsc - prev.tsc 
                if dm < 0: dm += (2**64) 
                if dt < 0: dt += (2**64)
                busy = min(100.0, 100.0 * dm / dt) if dt > 0 else prev.busy_percent
            else: busy = prev.busy_percent 
            prev.mperf, prev.tsc, prev.busy_percent = mperf, tsc, busy

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        max_b = 0.0; valid_sample_this_round = False
        for cid in self.ru_timing_core_indices:
            d = self.ru_core_msr_prev_data.get(cid)
            if d and d.busy_percent is not None:
                max_b = max(max_b, d.busy_percent)
                valid_sample_this_round = True
        
        if not valid_sample_this_round and not self.max_ru_timing_usage_history: return 0.0 
        elif not valid_sample_this_round and self.max_ru_timing_usage_history:
            return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history)

        self.max_ru_timing_usage_history.append(max_b)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg:
            self.max_ru_timing_usage_history.pop(0)
        
        return sum(self.max_ru_timing_usage_history)/len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str]) -> None:
        cmd_str_list = [self.intel_sst_path] + cmd_list[1:] if cmd_list[0] == "intel-speed-select" else cmd_list
        printable_cmd = ' '.join(cmd_str_list)
        if self.dry_run:
            self._log(INFO, f"[DRY RUN] Would execute: {printable_cmd}")
            return
        
        self._log(DEBUG_ALL, f"Executing: {printable_cmd}")
        try:
            process = subprocess.run(cmd_str_list, shell=False, check=True, capture_output=True, text=True, timeout=10)
            if process.stdout: self._log(DEBUG_ALL, f"Cmd STDOUT: {process.stdout.strip()}")
            if process.stderr: self._log(DEBUG_ALL, f"Cmd STDERR: {process.stderr.strip()}") 
        except subprocess.CalledProcessError as e:
            msg = f"Cmd '{e.cmd}' failed ({e.returncode}). STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}"
            self._log(ERROR, msg)
            raise RuntimeError(msg) 
        except FileNotFoundError:
            msg = f"Cmd '{cmd_str_list[0]}' not found. Ensure it's in PATH or config provides full path."
            self._log(ERROR, msg)
            raise RuntimeError(msg)
        except subprocess.TimeoutExpired:
            msg = f"Cmd '{printable_cmd}' timed out after 10s."
            self._log(ERROR, msg)
            raise RuntimeError(msg)

    def _setup_intel_sst(self):
        self._log(INFO, "--- Configuring Intel SST-CP ---")
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"]) 
            for cid_key, freq_mhz_str in self.config.get('clos_min_frequency', {}).items():
                try:
                    freq_val = str(freq_mhz_str) 
                    self._run_command(["intel-speed-select", "core-power", "config", "-c", str(cid_key), "--min", freq_val])
                    self._log(INFO, f"SST-CP: Set CLOS {cid_key} min frequency to {freq_val} MHz.")
                except Exception as e_clos_freq:
                    self._log(ERROR, f"SST-CP: Failed to set min freq for CLOS {cid_key} to {freq_mhz_str}: {e_clos_freq}")
            
            ran_cores = {name: self._parse_core_list_string(str(core_list_str))
                         for name, core_list_str in self.config.get('ran_cores', {}).items()}

            all_configured_cores = set() 
            for cid_key, component_names_list in self.config.get('clos_association', {}).items():
                clos_id = int(cid_key)
                cores_for_this_clos = set()
                if isinstance(component_names_list, list):
                    for comp_name in component_names_list:
                        cores_for_this_comp = ran_cores.get(comp_name, [])
                        if not cores_for_this_comp:
                            self._log(WARN, f"SST-CP: Component '{comp_name}' for CLOS {clos_id} has no cores defined in 'ran_cores'.")
                        cores_for_this_clos.update(cores_for_this_comp)
                else:
                    self._log(WARN, f"SST-CP: Components for CLOS {clos_id} ('{component_names_list}') is not a list. Skipping.")
                    continue

                if clos_id == 0 and self.ru_timing_core_indices: 
                    self._log(INFO, f"SST-CP: Ensuring RU_Timing cores {self.ru_timing_core_indices} are in CLOS 0.")
                    cores_for_this_clos.update(self.ru_timing_core_indices)
                
                if cores_for_this_clos:
                    core_list_str = ",".join(map(str, sorted(list(cores_for_this_clos))))
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id)])
                    self._log(INFO, f"SST-CP: Associated cores [{core_list_str}] to CLOS {clos_id}.")
                    all_configured_cores.update(cores_for_this_clos)
                elif clos_id == 0 and self.ru_timing_core_indices and not component_names_list: 
                     core_list_str = ",".join(map(str, sorted(list(self.ru_timing_core_indices))))
                     self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id)])
                     self._log(INFO, f"SST-CP: Associated RU_Timing cores [{core_list_str}] to CLOS {clos_id}.")
                     all_configured_cores.update(self.ru_timing_core_indices)
                else:
                    self._log(WARN, f"SST-CP: No cores to associate with CLOS {clos_id} based on components '{component_names_list}'.")
            self._log(INFO, "--- Intel SST-CP Configuration Complete ---")
        except Exception as e:
            self._log(ERROR, f"Intel SST-CP setup error: {e}")
            if not self.dry_run:
                raise RuntimeError(f"SST-CP Setup Failed: {e}")

    def _read_current_tdp_limit_w(self) -> float:
        if self.dry_run and hasattr(self, 'optimizer_target_tdp_w'): 
            return self.optimizer_target_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f:
                return int(f.read().strip()) / 1e6
        except Exception as e:
            self._log(WARN, f"Could not read {self.power_limit_uw_file}, returning configured min_tdp ({self.tdp_min_w}W). Error: {e}")
            return float(self.tdp_min_w)

    def _set_tdp_limit_w(self, tdp_watts: float, context: str = ""):
        clamped_tdp_uw = int(max(self.tdp_min_w * 1e6, min(tdp_watts * 1e6, self.tdp_max_w * 1e6)))
        new_tdp_w = clamped_tdp_uw / 1e6
        significant_change = abs(self.current_tdp_w - new_tdp_w) > 0.01

        if self.dry_run:
            if significant_change:
                self._log(INFO, f"[DRY RUN] {context}. New Target TDP: {new_tdp_w:.1f}W (Previous: {self.current_tdp_w:.1f}W).")
            self.current_tdp_w = new_tdp_w
            return

        try:
            with open(self.power_limit_uw_file, 'r') as f_read:
                current_hw_limit_uw = int(f_read.read().strip())
            if current_hw_limit_uw == clamped_tdp_uw:
                if significant_change: 
                    self._log(INFO, f"{context}. Target TDP: {new_tdp_w:.1f}W (already set in HW, updating internal state).")
                    self.current_tdp_w = new_tdp_w
                return 
        except Exception as e:
            self._log(WARN, f"Could not read {self.power_limit_uw_file} before write: {e}. Proceeding with write.")

        try:
            self._log(INFO, f"{context}. Setting TDP to: {new_tdp_w:.1f}W (from {self.current_tdp_w:.1f}W).")
            with open(self.power_limit_uw_file, 'w') as f_write:
                f_write.write(str(clamped_tdp_uw))
            self.current_tdp_w = new_tdp_w 
        except OSError as e:
            self._log(ERROR, f"OSError writing TDP to {self.power_limit_uw_file}: {e}")
            raise RuntimeError(f"OSError setting TDP: {e}") 
        except Exception as e:
            self._log(ERROR, f"Exception writing TDP: {e}")
            raise RuntimeError(f"Exception setting TDP: {e}")

    def _run_ru_timing_pid_step(self, current_ru_cpu_usage: float):
        if not self.ru_timing_core_indices: return 

        error = self.target_ru_cpu_usage - current_ru_cpu_usage  
        abs_error = abs(error)
        sensitivity_threshold = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        
        if abs_error > sensitivity_threshold:
            step_size = self.tdp_adj_step_w_large if abs_error > (sensitivity_threshold * self.adaptive_step_far_thresh_factor) else self.tdp_adj_step_w_small
            tdp_change_w = -step_size if error > 0 else step_size
            new_target_tdp = self.current_tdp_w + tdp_change_w
            ctx = (f"RU_PID: RU CPU {current_ru_cpu_usage:.2f}% (Target {self.target_ru_cpu_usage:.2f}%), "
                   f"Error {error:.2f}%, TDP Change {tdp_change_w:.1f}W")
            self._set_tdp_limit_w(new_target_tdp, context=ctx)

    def _run_contextual_bandit_optimizer_step(self, current_efficiency_bits_per_uj: Optional[float],
                                             current_context_vector: Optional[np.array],
                                             significant_throughput_change: bool):
        if self.last_selected_bandit_arm is not None and self.last_context_vector is not None:
            if significant_throughput_change:
                self._log(WARN, f"ContextualBandit: Skipping update for arm '{self.last_selected_bandit_arm}' due to significant throughput change.")
            elif current_efficiency_bits_per_uj is not None and math.isfinite(current_efficiency_bits_per_uj):
                self.contextual_bandit.update_arm(self.last_selected_bandit_arm, self.last_context_vector, current_efficiency_bits_per_uj)
            else:
                self._log(WARN, f"ContextualBandit: Invalid efficiency ({current_efficiency_bits_per_uj}) for arm '{self.last_selected_bandit_arm}'. Skipping update.")
        selected_arm_index = 0 # Default to first arm
        selected_arm_key_log = self.arm_keys_ordered[0] # Default
        scores_for_log = [] # Default empty scores

        if current_context_vector is None:
            self._log(WARN, "CB Lib: Current context vector is None. Defaulting to 'hold' arm or first arm.")
            selected_arm_key_default = "hold"
            if "hold" not in self.arm_keys_ordered: selected_arm_key_default = self.arm_keys_ordered[0]
            selected_arm_index = self.arm_keys_ordered.index(selected_arm_key_default)
            selected_arm_key_log = selected_arm_key_default
            scores_for_log = [0.0] * len(self.arm_keys_ordered) # Placeholder scores
        else:
            try:
                scores = self.contextual_bandit_model.decision_function(current_context_vector.reshape(1, -1))
                
                # --- DEBUGGING PRINTS ---
                self._log(DEBUG_ALL, f"CB Lib: Raw scores object: {scores}")
                self._log(DEBUG_ALL, f"CB Lib: Type of scores: {type(scores)}")
                if isinstance(scores, np.ndarray):
                    self._log(DEBUG_ALL, f"CB Lib: Shape of scores: {scores.shape}")
                # --- END DEBUGGING PRINTS ---

                if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[0] == 1:
                    scores_for_current_context = scores[0] # This should be a 1D array of scores
                    selected_arm_index = np.argmax(scores_for_current_context)
                    selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                    scores_for_log = scores_for_current_context # Assign for logging
                else:
                    self._log(ERROR, f"CB Lib: Unexpected scores format. Type: {type(scores)}, Value: {scores}. Defaulting arm choice.")
                    selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                    selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                    scores_for_log = [0.0] * len(self.arm_keys_ordered) # Placeholder

            except Exception as e:
                self._log.error(f"CB Lib: Error during arm selection using decision_function: {e}. Defaulting to random arm.")
                selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                scores_for_log = [0.0] * len(self.arm_keys_ordered) # Placeholder

        # Prepare scores string for logging (robustly)
        if isinstance(scores_for_log, (np.ndarray, list)) and all(isinstance(s, (int, float)) for s in scores_for_log):
            scores_str = ", ".join([f"{self.arm_keys_ordered[i]}:{s:.3f}" for i,s in enumerate(scores_for_log) if i < len(self.arm_keys_ordered)])
        else:
            scores_str = "N/A or invalid format"
            self._log(WARN, f"CB Lib: Could not format scores for logging. scores_for_log: {scores_for_log}")
        
        self._log(INFO, f"CB Lib: Selected ArmIdx '{selected_arm_index}' (Key: {selected_arm_key_log}). Scores: [{scores_str}]")
        
        self.last_selected_arm_index = selected_arm_index
        self.last_context_vector = current_context_vector

        actual_selected_arm_key = self.arm_keys_ordered[selected_arm_index]
        tdp_delta_w = self.bandit_actions.get(actual_selected_arm_key, 0.0)

        base_tdp_for_bandit_decision = self.optimizer_target_tdp_w
        proposed_next_tdp_by_bandit = base_tdp_for_bandit_decision + tdp_delta_w
        
        self._log(INFO, f"CB Lib Action: ArmKey='{actual_selected_arm_key}', Delta={tdp_delta_w:.1f}W. "
                        f"Base TDP: {base_tdp_for_bandit_decision:.1f}W. "
                        f"Proposed TDP: {proposed_next_tdp_by_bandit:.1f}W.")
        if current_context_vector is not None:
             self._log(DEBUG_ALL, f"CB Lib Context: {['{:.2f}'.format(x) for x in current_context_vector]}")

        self._set_tdp_limit_w(proposed_next_tdp_by_bandit, context=f"Optimizer CB Lib (Arm: {actual_selected_arm_key})")
        self.optimizer_target_tdp_w = self._read_current_tdp_limit_w()


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
                    if 0 <= pwr_w < 5000: 
                        ok = True
                    else:
                        self._log(DEBUG_ALL, f"Unrealistic PkgPwr calculated: {pwr_w:.1f}W (dE={de}, dt={dt:.3f}s). Resetting baseline.")
                        ok=False; pwr_w=0.0
                        self.last_pkg_energy_uj, self.last_energy_read_time = current_e_uj, now
                        return pwr_w, ok 
            
            if ok or self.last_pkg_energy_uj is None:
                 self.last_pkg_energy_uj, self.last_energy_read_time = current_e_uj, now
            return pwr_w, ok
        except Exception as e:
            self._log(WARN, f"Exception in _get_pkg_power_w: {e}");
            return 0.0, False

    def _read_current_energy_uj(self) -> Optional[int]:
        if not os.path.exists(self.energy_uj_file): return None
        try:
            with open(self.energy_uj_file, 'r') as f: return int(f.read().strip())
        except Exception as e: self._log(WARN, f"Could not read {self.energy_uj_file}: {e}"); return None

    def _get_interval_energy_uj_for_optimizer(self) -> Optional[float]:
        current_e_uj = self._read_current_energy_uj()
        if current_e_uj is None: return None

        if self.energy_at_last_optimizer_interval_uj is None: 
            self.energy_at_last_optimizer_interval_uj = current_e_uj
            return None 

        delta_e = float(current_e_uj - self.energy_at_last_optimizer_interval_uj)

        if delta_e < 0: 
            max_r = self.max_energy_val_rapl
            try:
                with open(os.path.join(self.rapl_base_path,"max_energy_range_uj"),'r') as f_max_r:
                    max_r_val = int(f_max_r.read().strip());
                    if max_r_val > 0: max_r = max_r_val
            except Exception as e: self._log(WARN, f"Could not read max_energy_range_uj: {e}"); pass
            delta_e += max_r
        
        self.energy_at_last_optimizer_interval_uj = current_e_uj 
        return delta_e


    def _run_contextual_bandit_optimizer_step(self, current_efficiency_bits_per_uj: Optional[float],
                                             current_context_vector: Optional[np.array],
                                             significant_throughput_change: bool):
        # 1. Update bandit with the reward from the PREVIOUS action (if any valid data)
        if self.last_selected_arm_index is not None and self.last_context_vector is not None:
            if significant_throughput_change:
                self._log(WARN, f"CB Lib: Skipping update for arm_idx '{self.last_selected_arm_index}' due to significant throughput change.")
            elif current_efficiency_bits_per_uj is not None and math.isfinite(current_efficiency_bits_per_uj):
                # contextualbandits library `fit` method usually takes: X, actions_taken, rewards
                # For online, it's often: model.fit(X_i, a_i, r_i) where X_i is context, a_i is action index, r_i is reward
                # Check documentation for the specific update method of contextualbandits.LinUCB
                # Assuming it has an `update(action_index, reward, context_vector)` or similar
                # Or a `partial_fit(X, chosen_arm_indicators, rewards)`
                # For now, let's assume a simplified `update` method if the library doesn't have a direct online one,
                # or we adapt to its `partial_fit` or `fit` with single samples.
                # The library's LinUCB `fit(self, X, a, r)` expects:
                # X: Feature matrix (n_samples, n_features)
                # a: Action chosen for each sample (n_samples,) -> array of integers (arm indices)
                # r: Reward received for each sample (n_samples,)
                try:
                    self.contextual_bandit_model.fit(X=self.last_context_vector.reshape(1, -1), # Reshape to (1, n_features)
                                                     a=np.array([self.last_selected_arm_index]), # Action index as array
                                                     r=np.array([current_efficiency_bits_per_uj])) # Reward as array
                    last_arm_key = self.arm_keys_ordered[self.last_selected_arm_index]
                    self._log(INFO, f"CB Lib: Updated ArmIdx '{self.last_selected_arm_index}' (Key: {last_arm_key}) with reward {current_efficiency_bits_per_uj:.3f}.")
                except Exception as e:
                    self._log.error(f"CB Lib: Error during model fit/update: {e}")

            else:
                self._log(WARN, f"CB Lib: Invalid efficiency ({current_efficiency_bits_per_uj}) for arm_idx '{self.last_selected_arm_index}'. Skipping update.")
        
        # 2. Select a new arm using the CURRENT context
        if current_context_vector is None:
            self._log(WARN, "CB Lib: Current context vector is None. Cannot select arm. Defaulting to 'hold'.")
            selected_arm_key = "hold"
            if "hold" not in self.arm_keys_ordered: selected_arm_key = self.arm_keys_ordered[0] # Fallback if hold not present
            selected_arm_index = self.arm_keys_ordered.index(selected_arm_key)
        else:
            # contextualbandits library `predict(X)` returns scores or chosen actions for each row in X
            # chosen_action_indices = self.contextual_bandit_model.predict(current_context_vector.reshape(1, -1)) # Returns array of chosen actions
            # selected_arm_index = chosen_action_indices[0]
            
            # Or, if predict returns scores:
            scores = self.contextual_bandit_model.predict(current_context_vector.reshape(1, -1)) # Should return scores per arm for the context
            selected_arm_index = np.argmax(scores) # Choose arm with highest score/probability
            selected_arm_key = self.arm_keys_ordered[selected_arm_index]
            self._log(INFO, f"CB Lib: Selected ArmIdx '{selected_arm_index}' (Key: {selected_arm_key}). Scores: {['{:.3f}'.format(s) for s in scores[0]]}")


        self.last_selected_arm_index = selected_arm_index
        self.last_context_vector = current_context_vector

        actual_selected_arm_key = self.arm_keys_ordered[selected_arm_index]
        tdp_delta_w = self.bandit_actions.get(actual_selected_arm_key, 0.0)

        base_tdp_for_bandit_decision = self.optimizer_target_tdp_w
        proposed_next_tdp_by_bandit = base_tdp_for_bandit_decision + tdp_delta_w
        
        self._log(INFO, f"CB Lib Action: ArmKey='{actual_selected_arm_key}', Delta={tdp_delta_w:.1f}W. "
                        f"Base TDP: {base_tdp_for_bandit_decision:.1f}W. "
                        f"Proposed TDP: {proposed_next_tdp_by_bandit:.1f}W.")
        if current_context_vector is not None:
             self._log(DEBUG_ALL, f"CB Lib Context: {['{:.2f}'.format(x) for x in current_context_vector]}")

        self._set_tdp_limit_w(proposed_next_tdp_by_bandit, context=f"Optimizer CB Lib (Arm: {actual_selected_arm_key})")
        self.optimizer_target_tdp_w = self._read_current_tdp_limit_w()


    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes):
        self._log(DEBUG_KPM, f"KPM CB: Agent:{e2_agent_id}, Sub(E2EventInstID):{subscription_id}, Time:{time.monotonic():.3f}")
        if not self.e2sm_kpm: self._log(WARN, f"KPM from {e2_agent_id}, but e2sm_kpm unavailable."); return
    
        try:
            kpm_hdr_info = self.e2sm_kpm.extract_hdr_info(indication_hdr_bytes)
            kpm_meas_data = self.e2sm_kpm.extract_meas_data(indication_msg_bytes)
            # self._log(DEBUG_ALL, f"KPM CB RAW HDR: {kpm_hdr_info}")
            # self._log(DEBUG_ALL, f"KPM CB RAW MSG DATA: {kpm_meas_data}") # Can be very verbose
    
            if not kpm_meas_data: 
                self._log(WARN, f"KPM CB Style 4: Failed/empty KPM data from {e2_agent_id}. HDR: {kpm_hdr_info}"); return
            
            ue_meas_data_map = kpm_meas_data.get("ueMeasData", {})
            if not isinstance(ue_meas_data_map, dict):
                self._log(WARN, f"KPM CB Style 4: Invalid 'ueMeasData' from {e2_agent_id}. Data: {kpm_meas_data}"); return
            
            if not ue_meas_data_map: # No UEs in this specific report
                with self.kpm_data_lock: # Still count the report for the gNB
                    if e2_agent_id not in self.accumulated_kpm_metrics:
                        self.accumulated_kpm_metrics[e2_agent_id] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'prb_sum_dl':0.0, 'prb_sum_ul':0.0, 'num_reports':0}
                    self.accumulated_kpm_metrics[e2_agent_id]['num_reports'] += 1
                self._log(DEBUG_KPM, f"KPM CB Style 4: {e2_agent_id}: Report with 0 UEs.")
                return 
            
            gNB_data_this_report = {'dl_bits': 0.0, 'ul_bits': 0.0, 'dl_prb': 0.0, 'ul_prb': 0.0}
            ues_in_this_report_for_this_gnb = set()

            for ue_id_str, per_ue_measurements in ue_meas_data_map.items():
                global_ue_id = f"{e2_agent_id}_{ue_id_str}" 
                ues_in_this_report_for_this_gnb.add(global_ue_id)

                ue_dl_bits_this_ue = 0
                ue_ul_bits_this_ue = 0

                ue_metrics = per_ue_measurements.get("measData", {})
                if not isinstance(ue_metrics, dict): continue # Skip malformed UE data
                
                for metric_name, value_list in ue_metrics.items():
                    if isinstance(value_list, list) and value_list:
                        # Summing values if KPM could report multiple values for a metric in a single UE report (e.g. per QCI)
                        # For RLC SDU volume, it's usually one value. For PRB, usually one.
                        value = sum(val for val in value_list if isinstance(val, (int, float)))

                        try:
                            if metric_name == 'DRB.RlcSduTransmittedVolumeDL':
                                ue_dl_bits_this_ue = float(value) * 1000.0 # kbits to bits
                                gNB_data_this_report['dl_bits'] += ue_dl_bits_this_ue
                            elif metric_name == 'DRB.RlcSduTransmittedVolumeUL':
                                ue_ul_bits_this_ue = float(value) * 1000.0 # kbits to bits
                                gNB_data_this_report['ul_bits'] += ue_ul_bits_this_ue
                            elif metric_name == 'RRU.PrbTotDl': # Assuming this is % for the UE
                                gNB_data_this_report['dl_prb'] += float(value)
                            elif metric_name == 'RRU.PrbTotUl': # Assuming this is % for the UE
                                gNB_data_this_report['ul_prb'] += float(value)
                        except (ValueError, TypeError) as e:
                             self._log(WARN, f"KPM CB Style 4: Metric '{metric_name}' for UE {global_ue_id} value '{value_list}' processing error: {e}.")
                
                # Store per-UE total throughput for active UE calculation
                with self.kpm_data_lock: # Protect shared self.current_interval_per_ue_data
                    if global_ue_id not in self.current_interval_per_ue_data:
                        self.current_interval_per_ue_data[global_ue_id] = {'total_bits': 0.0}
                    self.current_interval_per_ue_data[global_ue_id]['total_bits'] += (ue_dl_bits_this_ue + ue_ul_bits_this_ue)

            with self.kpm_data_lock:
                if e2_agent_id not in self.accumulated_kpm_metrics:
                    self.accumulated_kpm_metrics[e2_agent_id] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'prb_sum_dl':0.0, 'prb_sum_ul':0.0, 'num_reports':0}
                acc = self.accumulated_kpm_metrics[e2_agent_id]
                acc['bits_sum_dl'] += gNB_data_this_report['dl_bits']
                acc['bits_sum_ul'] += gNB_data_this_report['ul_bits']
                acc['prb_sum_dl'] += gNB_data_this_report['dl_prb']
                acc['prb_sum_ul'] += gNB_data_this_report['ul_prb']
                acc['num_reports'] += 1
                
                self.current_interval_ue_ids.update(ues_in_this_report_for_this_gnb)
                
                self._log(DEBUG_KPM, f"KPM CB Style 4: {e2_agent_id}: "
                                     f"Bits(DL={gNB_data_this_report['dl_bits']:.0f}, UL={gNB_data_this_report['ul_bits']:.0f}), "
                                     f"PRB%(DL={gNB_data_this_report['dl_prb']:.1f}, UL={gNB_data_this_report['ul_prb']:.1f}). "
                                     f"Reported {len(ues_in_this_report_for_this_gnb)} UEs. Total unique this interval: {len(self.current_interval_ue_ids)}")

        except Exception as e: self._log(ERROR, f"Error processing KPM Style 4 from {e2_agent_id}: {e}"); import traceback; traceback.print_exc()


    def _get_and_reset_accumulated_kpm_metrics(self) -> Dict[str, Dict[str, Any]]: # Now includes PRB
        with self.kpm_data_lock:
            snap = {}
            for gnb_id, data in self.accumulated_kpm_metrics.items():
                snap[gnb_id] = {
                    'dl_bits': data.get('bits_sum_dl', 0.0),
                    'ul_bits': data.get('bits_sum_ul', 0.0),
                    'dl_prb_sum_percentage': data.get('prb_sum_dl', 0.0), # Sum of PRB percentages reported
                    'ul_prb_sum_percentage': data.get('prb_sum_ul', 0.0),
                    'reports_in_interval': data.get('num_reports', 0)
                }
                data['bits_sum_dl'] = 0.0; data['bits_sum_ul'] = 0.0
                data['prb_sum_dl'] = 0.0; data['prb_sum_ul'] = 0.0
                data['num_reports'] = 0
        return snap

    def _get_and_reset_active_ue_count_and_data(self) -> Tuple[int, Dict[str, Dict[str, float]]]:
        # Calculates active UEs based on throughput threshold from per-UE data collected in callbacks.
        # Also returns the per-UE data for potential further use and clears it.
        active_ue_count = 0
        report_interval_s = float(self.config.get('kpm_subscriptions', {}).get('style4_report_period_ms', 1000)) / 1000.0
        if report_interval_s <= 0: report_interval_s = 1.0 # Avoid div by zero

        threshold_bits = self.active_ue_throughput_threshold_mbps * 1024 * 1024 # Convert Mbps to bits for the interval
        
        # The current_interval_per_ue_data stores SUM of bits over MULTIPLE KPM reports if optimizer_interval > KPM_report_interval
        # So threshold should be Mbps * optimizer_decision_interval_s
        threshold_bits_over_optimizer_interval = self.active_ue_throughput_threshold_mbps * 1e6 * self.optimizer_decision_interval_s


        with self.kpm_data_lock:
            for global_ue_id, ue_data in self.current_interval_per_ue_data.items():
                if ue_data.get('total_bits', 0.0) >= threshold_bits_over_optimizer_interval :
                    active_ue_count += 1
            
            # For logging/debugging, what are the UEs that were active
            # self._log(DEBUG_KPM, f"Active UE count: {active_ue_count} based on threshold {self.active_ue_throughput_threshold_mbps} Mbps over {self.optimizer_decision_interval_s}s.")

            # Return a copy and clear
            per_ue_data_snap = self.current_interval_per_ue_data.copy()
            self.current_interval_per_ue_data.clear()
            # Also clear the set of all seen UEs for the next interval
            all_seen_ues_count = len(self.current_interval_ue_ids) # For logging if needed
            self.current_interval_ue_ids.clear()

        return active_ue_count, per_ue_data_snap # also return all_seen_ues_count if needed

    def _setup_kpm_subscriptions(self): # Same as previous Style 4 only version
        self._log(INFO, "--- Setting up KPM Style 4 Subscriptions (Per-UE Metrics) ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable. Cannot subscribe."); return
        
        nodes = list(self.gnb_ids_map.values()) 
        if not nodes: self._log(WARN, "No gNB IDs configured for KPM subscriptions."); return

        kpm_config = self.config.get('kpm_subscriptions', {})
        
        style4_metrics = kpm_config.get('style4_metrics_per_ue', [
            'DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL',
            'RRU.PrbTotDl', 'RRU.PrbTotUl' # Added PRB metrics
        ])
        style4_report_p_ms = int(kpm_config.get('style4_report_period_ms', 1000))
        style4_gran_p_ms = int(kpm_config.get('style4_granularity_period_ms', style4_report_p_ms))
        
        # Use the example's dummy condition for now, make sure it's robust or configurable to `[]`
        matching_ue_conds_config = kpm_config.get('style4_matching_ue_conditions', 
                                                  [{'testCondInfo': {'testType': ('ul-rSRP', 'true'), 'testExpr': 'lessthan', 'testValue': ('valueInt', 10000)}}])
            
        self._log(INFO, f"KPM Style 4: MetricsPerUE: {style4_metrics}, ReportPeriod={style4_report_p_ms}ms, Granularity={style4_gran_p_ms}ms, Conditions: {matching_ue_conds_config}")
        
        successes = 0
        for node_id_str in nodes:
            if self.dry_run:
                self._log(INFO, f"[DRY RUN] KPM Style 4 Sub: Node {node_id_str}")
                successes+=1; continue
            try:
                self._log(INFO, f"Subscribing KPM Style 4: Node {node_id_str}")
                self.e2sm_kpm.subscribe_report_service_style_4(
                    node_id_str, style4_report_p_ms, matching_ue_conds_config, 
                    style4_metrics, style4_gran_p_ms, self._kpm_indication_callback
                )
                with self.kpm_data_lock:
                    if node_id_str not in self.accumulated_kpm_metrics:
                        self.accumulated_kpm_metrics[node_id_str] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'prb_sum_dl':0.0, 'prb_sum_ul':0.0, 'num_reports':0}
                successes += 1
            except Exception as e: self._log(ERROR, f"KPM Style 4 subscription failed for {node_id_str}: {e}"); import traceback; traceback.print_exc()
        
        if successes > 0: self._log(INFO, f"--- KPM Style 4 Subscriptions: {successes} successful attempts for {len(nodes)} nodes. ---")
        elif nodes: self._log(WARN, "No KPM Style 4 subscriptions successfully initiated.")
        else: self._log(INFO, "--- No KPM nodes to subscribe to. ---")

    @xAppBase.start_function
    def run_power_management_xapp(self):
        if os.geteuid() != 0 and not self.dry_run:
            self._log(ERROR, "Must be root for live run. Exiting."); sys.exit(1)
        
        try:
            # ... (Initializations for TDP, MSR, Energy, SST are THE SAME as before) ...
            self.current_tdp_w = self._read_current_tdp_limit_w()
            self.optimizer_target_tdp_w = self.current_tdp_w
            self._log(INFO, f"Initial current TDP: {self.current_tdp_w:.1f}W. Optimizer target set to this.")

            if self.ru_timing_core_indices:
                self._log(INFO, "Priming MSR data..."); self._update_ru_core_msr_data(); time.sleep(0.1); self._update_ru_core_msr_data(); self._log(INFO, "MSR primed.")
            
            self.energy_at_last_optimizer_interval_uj = self._read_current_energy_uj()
            self.last_pkg_energy_uj = self._read_current_energy_uj() 
            self.last_energy_read_time = time.monotonic()
            if self.energy_at_last_optimizer_interval_uj is None and not self.dry_run: self._log(WARN, "Could not get initial package energy for optimizer.")
            
            self._setup_intel_sst()
            self._setup_kpm_subscriptions() 

            now = time.monotonic()
            self.last_ru_pid_run_time = now
            self.last_optimizer_run_time = now 
            self.last_stats_print_time = now
            self.total_bits_from_previous_optimizer_interval = 0.0 # Initialize

            self._log(INFO, f"\n--- Starting Monitoring & Control Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            self._log(INFO, f"RU PID Interval: {self.ru_timing_pid_interval_s}s | Target RU CPU: {self.target_ru_cpu_usage if self.ru_timing_core_indices else 'N/A'}%")
            self._log(INFO, f"CB Optimizer Interval: {self.optimizer_decision_interval_s}s | TDP Range: {self.tdp_min_w}W-{self.tdp_max_w}W")
            self._log(INFO, f"CB Actions: {self.bandit_actions}, Alpha: {self.linucb_alpha}")
            self._log(INFO, f"CB Context Dim (features only): {self.context_dimension_features_only}, FitIntercept: {self.linucb_fit_intercept}. ActiveUE Thresh: {self.active_ue_throughput_threshold_mbps} Mbps.")
            self._log(INFO, f"Stats Print Interval: {self.stats_print_interval_s}s")

            while self.running: 
                loop_start_time = time.monotonic()

                if self.ru_timing_core_indices: self._update_ru_core_msr_data()
                current_ru_cpu_usage_control_val = self._get_control_ru_timing_cpu_usage()

                if loop_start_time - self.last_ru_pid_run_time >= self.ru_timing_pid_interval_s:
                    if self.ru_timing_core_indices: self._run_ru_timing_pid_step(current_ru_cpu_usage_control_val)
                    self.last_ru_pid_run_time = loop_start_time
                
                if loop_start_time - self.last_optimizer_run_time >= self.optimizer_decision_interval_s:
                    interval_energy_uj = self._get_interval_energy_uj_for_optimizer()
                    kpm_summed_data = self._get_and_reset_accumulated_kpm_metrics() # Summed DL/UL bits & PRB% per gNB
                    current_num_active_ues, _ = self._get_and_reset_active_ue_count_and_data() # Resets per-UE data too
                    self.current_num_active_ues_for_log = current_num_active_ues

                    # Calculate system-wide totals for context
                    total_dl_bits_interval = sum(d.get('dl_bits', 0.0) for d in kpm_summed_data.values())
                    total_ul_bits_interval = sum(d.get('ul_bits', 0.0) for d in kpm_summed_data.values())
                    total_bits_optimizer_interval = total_dl_bits_interval + total_ul_bits_interval
                    
                    total_prb_dl_percentage = sum(d.get('dl_prb_sum_percentage', 0.0) for d in kpm_summed_data.values())
                    total_prb_ul_percentage = sum(d.get('ul_prb_sum_percentage', 0.0) for d in kpm_summed_data.values())

                    num_kpm_reports_processed = sum(d.get('reports_in_interval',0) for d in kpm_summed_data.values())
                    num_active_dus = sum(1 for d in kpm_summed_data.values() if d.get('dl_bits',0) + d.get('ul_bits',0) > 1e-6)

                    significant_throughput_change = False # Logic for this remains same
                    if self.total_bits_from_previous_optimizer_interval is not None: 
                        denominator = self.total_bits_from_previous_optimizer_interval
                        if denominator < 1e-6: 
                            if total_bits_optimizer_interval > 1e6 : significant_throughput_change = True 
                        elif abs(total_bits_optimizer_interval - denominator) / denominator > self.throughput_change_threshold_for_discard:
                            relative_change = abs(total_bits_optimizer_interval - denominator) / denominator
                            self._log(WARN, f"Optimizer: Sig. throughput change ({relative_change*100:.1f}%). Update might be skipped.")
                            significant_throughput_change = True
                    self.total_bits_from_previous_optimizer_interval = total_bits_optimizer_interval

                    current_efficiency_for_bandit: Optional[float] = None
                    if num_kpm_reports_processed > 0 : 
                        if interval_energy_uj is not None and interval_energy_uj > 1e-3: 
                            current_efficiency_for_bandit = total_bits_optimizer_interval / interval_energy_uj
                        elif total_bits_optimizer_interval > 1e-9: current_efficiency_for_bandit = float('inf') 
                        else: current_efficiency_for_bandit = 0.0
                    self.most_recent_calculated_efficiency_for_log = current_efficiency_for_bandit
                    
                    current_actual_tdp_for_context = self.current_tdp_w 
                    current_context_vec = self._get_current_context_vector(
                        total_dl_bits_interval, total_ul_bits_interval,
                        total_prb_dl_percentage, total_prb_ul_percentage,
                        current_num_active_ues, num_active_dus,
                        current_ru_cpu_usage_control_val, current_actual_tdp_for_context
                    )
                    
                    self._run_contextual_bandit_optimizer_step(current_efficiency_for_bandit, current_context_vec, significant_throughput_change)
                    self.last_optimizer_run_time = loop_start_time
                
                if loop_start_time - self.last_stats_print_time >= self.stats_print_interval_s:
                    pkg_pwr_w, pkg_pwr_ok = self._get_pkg_power_w()
                    ru_usage_str = "N/A"
                    if self.ru_timing_core_indices:
                        ru_usage_parts = [f"C{cid}:{self.ru_core_msr_prev_data.get(cid).busy_percent:>6.2f}%" if self.ru_core_msr_prev_data.get(cid) else f"C{cid}:N/A" for cid in self.ru_timing_core_indices]
                        ru_usage_str = f"[{', '.join(ru_usage_parts)}] (AvgMax:{current_ru_cpu_usage_control_val:>6.2f}%)"
                    pkg_pwr_log_str = f"{pkg_pwr_w:.1f}" if pkg_pwr_ok else "N/A"
                    
                    # Using arm_keys_ordered for consistent indexing if bandit library uses integer indices
                    last_arm_key_str = self.arm_keys_ordered[self.last_selected_arm_index] if self.last_selected_arm_index is not None else 'None'
                    
                    # Getting best arm from library might require different method if it doesn't track empirical means directly.
                    # For now, this is a placeholder if library doesn't expose it easily.
                    # best_emp_arm_idx, best_emp_eff = self.contextual_bandit_model.get_best_empirical_arm_stats() # Fictitious
                    # best_emp_arm_key = self.arm_keys_ordered[best_emp_arm_idx] if best_emp_arm_idx is not None else "None"
                    # For now, we don't have a direct equivalent from the library for "best empirical arm" easily.
                    
                    bandit_log = (f"CB_Lib(LastArmIdx:{self.last_selected_arm_index if self.last_selected_arm_index is not None else 'N/A'}, "
                                  f"Key:{last_arm_key_str})")
                    
                    log_parts = [f"RU:{ru_usage_str}", f"TDP_Act:{self.current_tdp_w:>5.1f}W", 
                                 f"TDP_OptTrg:{self.optimizer_target_tdp_w:>5.1f}W", f"PkgPwr:{pkg_pwr_log_str}W", bandit_log]
                    if self.most_recent_calculated_efficiency_for_log is not None:
                         log_parts.append(f"IntEff:{self.most_recent_calculated_efficiency_for_log:.3f}b/uJ")
                    log_parts.append(f"ActiveUEs:{self.current_num_active_ues_for_log}")

                    self._log(INFO, " | ".join(log_parts)); 
                    self.last_stats_print_time = loop_start_time
                
                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, self.main_loop_sleep_s - loop_duration)
                if sleep_time > 0 : time.sleep(sleep_time)

        # ... (except KeyboardInterrupt, SystemExit, etc. are THE SAME as before) ...
        except KeyboardInterrupt: self._log(INFO, "\nLoop interrupted (KeyboardInterrupt).")
        except SystemExit as e: self._log(INFO, f"Application exiting (SystemExit: {e})."); raise 
        except RuntimeError as e: self._log(ERROR, f"Critical runtime error in loop: {e}."); raise 
        except Exception as e: self._log(ERROR, f"\nUnexpected error in loop: {e}"); import traceback; self._log(ERROR, traceback.format_exc()); raise 
        finally: self._log(INFO, "--- Power Manager xApp run_power_management_xapp finished. ---")

if __name__ == "__main__": # Same as before
    parser = argparse.ArgumentParser(description="EcoRAN Power Manager xApp with Contextual Bandit Optimizer")
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--http_server_port", type=int, default=8090, help="HTTP server port.")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port.")
    args = parser.parse_args()
    manager = None
    try:
        manager = PowerManager(args.config_path, args.http_server_port, args.rmr_port)
        signal.signal(signal.SIGINT, manager.signal_handler)
        signal.signal(signal.SIGTERM, manager.signal_handler)
        if hasattr(signal, 'SIGQUIT'): signal.signal(signal.SIGQUIT, manager.signal_handler)
        manager._log(INFO, "Registered signal handlers from xAppBase.")
        manager.run_power_management_xapp()
    except RuntimeError as e: 
        print(f"E: Critical error: {e}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"Critical error: {e}")
        sys.exit(1)
    except SystemExit as e: 
        code = e.code if e.code is not None else 0
        print(f"Application terminated with exit code: {code}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, f"Application terminated (SystemExit: {code}).")
        sys.exit(code) 
    except Exception as e:
        print(f"E: An unexpected error at top level: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"TOP LEVEL UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        msg = "Application finished."
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, msg)
        else: print(f"INFO: {msg} (logger may not be available).")
