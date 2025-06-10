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
    from contextualbandits.online import BootstrappedTS
    from contextualbandits.linreg import LinearRegression
except ImportError:
    print("E: Failed to import LinTS from contextualbandits.online. Please install the library: pip install contextualbandits")
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


def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
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

        # --- System Paths and Parameters ---
        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        self.max_energy_val_rapl = self.config.get('rapl_max_energy_uj_override', 2**60 - 1)

        # --- Timing & Control Loop Parameters ---
        self.main_loop_sleep_s = float(self.config.get('main_loop_sleep_s', 0.1))
        self.ru_timing_pid_interval_s = float(self.config.get('ru_timing_pid_interval_s', 1.0))
        self.optimizer_decision_interval_s = float(self.config.get('optimizer_decision_interval_s', 5.0)) # Default 5s
        self.stats_print_interval_s = float(self.config.get('stats_print_interval_s', self.optimizer_decision_interval_s))

        # --- TDP Management Parameters ---
        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 90))
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 170))
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 99.8)) # Was 99.5
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.0005))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1.0))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 3.0))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 1.5))

        # --- System State Variables ---
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        self.dry_run = bool(self.config.get('dry_run', False))
        self.current_tdp_w = float(self.tdp_min_w) 
        self.last_pkg_energy_uj: Optional[int] = None
        self.last_energy_read_time: Optional[float] = None
        self.energy_at_last_optimizer_interval_uj: Optional[int] = None
        self.max_ru_timing_usage_history: List[float] = []
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}

        # --- KPM Setup ---
        self.kpm_ran_func_id = kpm_ran_func_id
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None: self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else: self._log(WARN, "xAppBase.e2sm_kpm module unavailable."); self.e2sm_kpm = None
        self.gnb_ids_map = self.config.get('gnb_ids', {}) 
        self.kpm_data_lock = threading.Lock()
        self.accumulated_kpm_metrics: Dict[str, Dict[str, Any]] = {}
        self.current_interval_ue_ids: Set[str] = set()
        self.current_interval_per_ue_data: Dict[str, Dict[str, float]] = {}

        # --- Contextual Bandit ---
        cb_config = self.config.get('contextual_bandit', {})
        bandit_actions_w_str = cb_config.get('actions_tdp_delta_w', {"dec_10": -10.0, "dec_5": -5.0, "hold": 0.0, "inc_5": 5.0, "inc_10": 10.0})
        self.workload_drop_threshold = float(cb_config.get('workload_drop_threshold_for_reset', 0.5))
        self.bandit_actions: Dict[str, float] = {k: float(v) for k, v in bandit_actions_w_str.items()}
        self.arm_keys_ordered = list(self.bandit_actions.keys()) 
        if "hold" not in self.bandit_actions:
            self.bandit_actions["hold"] = 0.0
            if "hold" not in self.arm_keys_ordered: self.arm_keys_ordered.append("hold")
        
        self.context_dimension_features_only = int(cb_config.get('context_dimension_features_only', 8)) 

        self.workload_avg_window_size = int(cb_config.get('workload_avg_window_size', 3))
        self.recent_workload_bits: List[float] = []
        # # LinTS specific parameters from config (with defaults)
        # self.lints_lambda_ = float(cb_config.get('lambda_', 1.0))  # Default from LinTS doc
        # self.lints_fit_intercept = bool(cb_config.get('fit_intercept', True))
        # self.lints_v_sq = float(cb_config.get('v_sq', 0.1)) # Recommended to decrease from 1.0, let's try 0.1
        # self.lints_sample_from = str(cb_config.get('sample_from', "coef"))
        # self.lints_method = str(cb_config.get('method', "chol"))
        # self.lints_beta_prior = cb_config.get('beta_prior', "auto") # Can be None, "auto", or specific tuple
        # self.lints_smoothing = cb_config.get('smoothing', None) # Can be None or tuple
        # lints_random_state_cfg = cb_config.get('random_state', None)
        # self.lints_random_state = int(lints_random_state_cfg) if lints_random_state_cfg is not None else None

        # Create LinTS Beta Prior
        
        beta_prior_config = cb_config.get('beta_prior', None)
        self.lints_beta_prior = None # Default
        if isinstance(beta_prior_config, str):
            if beta_prior_config == "auto":
                self.lints_beta_prior = "auto"
            elif beta_prior_config == "uniform_optimistic":
                # This creates the ((a,b), n) tuple correctly in Python
                # It means: for any arm with < 3 observations, sample its
                # reward from a Beta(1,1) distribution (a uniform random number).
                self.lints_beta_prior = ((1, 1), 3) 
                self._log(INFO, "Using 'uniform_optimistic' Beta(1,1) prior.")
        elif beta_prior_config is not None:
             # If you ever want to pass the complex structure directly from YAML
             # it should be loaded here, but this is less safe.
             self.lints_beta_prior = beta_prior_config

        self.lints_smoothing = cb_config.get('smoothing', None)

        # --- CORRECTED BootstrappedTS parameters ---
        bts_config = self.config.get('bootstrapped_ts', {})
        self.bts_nsamples = int(bts_config.get('nsamples', 20))
        # These are for the *base* algorithm, not the bandit itself
        self.bts_lambda_ = float(bts_config.get('lambda_', 1.0))
        self.bts_fit_intercept = bool(bts_config.get('fit_intercept', True))
        
        # Get the beta prior config
        beta_prior_config = bts_config.get('beta_prior', "uniform_optimistic")
        self.bts_beta_prior = None
        if isinstance(beta_prior_config, str):
            if beta_prior_config == "auto":
                self.bts_beta_prior = "auto"
            elif beta_prior_config == "uniform_optimistic":
                self.bts_beta_prior = ((1, 1), 3)
        
        # --- CORRECTED CONSTRUCTOR CALL ---
        self._log(INFO, f"Initializing BootstrappedTS with nsamples={self.bts_nsamples}")
        
        # 1. Create an instance of the base algorithm with its parameters
        base_algo = LinearRegression(
            lambda_=self.bts_lambda_,
            fit_intercept=self.bts_fit_intercept
        )
        
        # 2. Pass the INSTANCE of the base algorithm to the bandit
        self.contextual_bandit_model = BootstrappedTS(
            base_algorithm=base_algo,
            nchoices=len(self.arm_keys_ordered),
            nsamples=self.bts_nsamples,
            batch_train=True, 
            beta_prior=self.bts_beta_prior,
            random_state=42
        )


        self.optimizer_target_tdp_w = self.current_tdp_w
        self.last_selected_arm_index: Optional[int] = None
        self.last_context_vector: Optional[np.array] = None # Type hint for clarity
        self.total_bits_from_previous_optimizer_interval: Optional[float] = None
        self.throughput_change_threshold_for_discard = float(cb_config.get('throughput_change_threshold_for_discard', 1.0))
        self.active_ue_throughput_threshold_mbps = float(cb_config.get('active_ue_throughput_threshold_mbps', 1.0))
        self.was_idle_in_previous_step = True 
        self.norm_params = cb_config.get('normalization_parameters', {})
        self._ensure_default_norm_params()

        # --- Timestamps and Logging Variables ---
        self.last_ru_pid_run_time: float = 0.0
        self.last_optimizer_run_time: float = 0.0
        self.last_stats_print_time: float = 0.0
        self.most_recent_calculated_reward_for_log: Optional[float] = None # For logging
        self.current_num_active_ues_for_log: int = 0
        self.pid_triggered_since_last_decision = False
        self.max_efficiency_seen = 1e-9
        self.max_eff_decay_factor = 0.995 # Add this new parameter

        self.last_action_actual_tdp: Optional[float] = None
        self.last_action_requested_tdp: Optional[float] = None
        
        self.COLORS = {
            'RED': '\033[91m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'WHITE': '\033[97m',
            'BOLD': '\033[1m',
            'RESET': '\033[0m'
        }
        
        self._validate_config()
        if self.dry_run: self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _colorize(self, text: str, color: str) -> str:
        """Wraps text in ANSI color codes for terminal output."""
        color_code = self.COLORS.get(color.upper())
        if color_code:
            return f"{color_code}{text}{self.COLORS['RESET']}"
        return text

    def _ensure_default_norm_params(self):
        """Ensure essential normalization parameters have defaults if not in config."""
        defaults = {
            'num_active_ues': {'min': 0, 'max': 100.0},
            'cpu_headroom': {'min': -5.0, 'max': 20.0}
        }
        for key, default_val in defaults.items():
            if key not in self.norm_params:
                self.norm_params[key] = default_val
                self._log(INFO, f"Normalization param for '{key}' not in config, using default: {default_val}")
  

    def _get_current_context_vector(self,
                                       current_num_active_ues: int,
                                       current_ru_cpu_avg: float,
                                       current_actual_tdp: float,
                                       # The normalized efficiency is now passed in
                                       normalized_efficiency: float 
                                       ) -> np.array:
        
        def _normalize(val, min_val, max_val):
            if (max_val - min_val) < 1e-6: return 0.5
            return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
    
        # --- MINIMALIST FEATURE ENGINEERING ---
    
        # Feature 1: Normalized Efficiency (already calculated and passed in).
        feature_1_norm_eff = normalized_efficiency
    
        # Feature 2: CPU Headroom, normalized.
        cpu_headroom = self.target_ru_cpu_usage - current_ru_cpu_avg
        norm_params_cpu = self.norm_params.get('cpu_headroom', {'min': -5.0, 'max': 20.0})
        feature_2_cpu_headroom = _normalize(cpu_headroom, norm_params_cpu.get('min'), norm_params_cpu.get('max'))
    
        # Feature 3: TDP Position, normalized.
        feature_3_tdp_pos = _normalize(current_actual_tdp, self.tdp_min_w, self.tdp_max_w)
    
        # Feature 4: Number of Active UEs, normalized.
        norm_params_ues = self.norm_params.get('num_active_ues', {'min': 0.0, 'max': 10.0})
        feature_4_num_ues = _normalize(float(current_num_active_ues), norm_params_ues.get('min'), norm_params_ues.get('max'))
        
        final_features = np.array([
            feature_1_norm_eff,
            feature_2_cpu_headroom,
            feature_3_tdp_pos,
            feature_4_num_ues
        ])
        
        self._log(DEBUG_ALL, f"Context Vector (Normalized): [Eff:{final_features[0]:.2f}, "
                             f"CPU_Headroom:{final_features[1]:.2f}, TDP_Pos:{final_features[2]:.2f}, "
                             f"Num_UEs:{final_features[3]:.2f}]")
    
        return final_features
    
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
        # No explicit 'bias' normalization parameter needed if fit_intercept=True for LinTS
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
        requested_tdp_w = tdp_watts
        clamped_tdp_uw = int(max(self.tdp_min_w * 1e6, min(tdp_watts * 1e6, self.tdp_max_w * 1e6)))
        new_tdp_w = clamped_tdp_uw / 1e6
        significant_change = abs(self.current_tdp_w - new_tdp_w) > 0.01

        if self.dry_run:
            if significant_change:
                self._log(INFO, f"[DRY RUN] {context}. New Target TDP: {new_tdp_w:.1f}W (Previous: {self.current_tdp_w:.1f}W).")
            self.current_tdp_w = new_tdp_w # Update internal state even in dry run
            return new_tdp_w, requested_tdp_w

        try:
            with open(self.power_limit_uw_file, 'r') as f_read:
                current_hw_limit_uw = int(f_read.read().strip())
            if current_hw_limit_uw == clamped_tdp_uw:
                if significant_change: # Only log if it's a logical change even if HW is same
                    self._log(INFO, f"{context}. Target TDP: {new_tdp_w:.1f}W (already set in HW, updating internal state).")
                self.current_tdp_w = new_tdp_w # Update internal state
                return new_tdp_w, requested_tdp_w
        except Exception as e:
            self._log(WARN, f"Could not read {self.power_limit_uw_file} before write: {e}. Proceeding with write.")

        try:
            # Colorize the new TDP value in magenta
            colored_tdp = self._colorize(f'{new_tdp_w:.1f}W', 'MAGENTA')
            self._log(INFO, f"{context}. Setting TDP to: {colored_tdp} (from {self.current_tdp_w:.1f}W).")
            with open(self.power_limit_uw_file, 'w') as f_write:
                f_write.write(str(clamped_tdp_uw))
            self.current_tdp_w = new_tdp_w
            return new_tdp_w, requested_tdp_w
        except OSError as e:
            self._log(ERROR, f"OSError writing TDP to {self.power_limit_uw_file}: {e}")
            raise RuntimeError(f"OSError setting TDP: {e}") 
        except Exception as e:
            self._log(ERROR, f"Exception writing TDP: {e}")
            raise RuntimeError(f"Exception setting TDP: {e}")

    def _run_ru_timing_pid_step(self, current_ru_cpu_usage: float):
        if not self.ru_timing_core_indices: return 
        
        pid_critical_ru_cpu_trigger = self.target_ru_cpu_usage # Using the direct target now
    
        if current_ru_cpu_usage > pid_critical_ru_cpu_trigger:
            self.pid_triggered_since_last_decision = True
            tdp_change_w = self.tdp_adj_step_w_large 
            new_target_tdp = self.current_tdp_w + tdp_change_w
            
            ctx = (f"RU_PID_SAFETY_NET: RU CPU {current_ru_cpu_usage:.2f}% > Trigger {pid_critical_ru_cpu_trigger:.2f}%. "
                   f"Forcing TDP Increase by {tdp_change_w:.1f}W")
            self._set_tdp_limit_w(new_target_tdp, context=ctx)
            # CRITICAL: When PID acts, it should influence the bandit's perceived target TDP
            self.optimizer_target_tdp_w = self._read_current_tdp_limit_w() 
            self._log(DEBUG_ALL, f"PID updated optimizer_target_tdp_w to {self.optimizer_target_tdp_w:.1f}W")
    
    def _run_contextual_bandit_optimizer_step(self, current_reward_for_bandit: Optional[float],
                                             current_context_vector: Optional[np.array],
                                             significant_throughput_change: bool):
        # 1. Update bandit with the reward from the PREVIOUS action
        if self.last_selected_arm_index is not None and self.last_context_vector is not None:
            if significant_throughput_change:
                self._log(WARN, f"CB Lib: Skipping update for arm_idx '{self.last_selected_arm_index}' due to sig. throughput change.")
            elif current_reward_for_bandit is not None and math.isfinite(current_reward_for_bandit):
                try:
                    X_update = self.last_context_vector.reshape(1, -1)
                    action_update = np.array([self.last_selected_arm_index], dtype=int)
                    reward_update = np.array([current_reward_for_bandit])

                    self.contextual_bandit_model.partial_fit(X_update, action_update, reward_update) # Use partial_fit for online updates
                    
                    last_arm_key = self.arm_keys_ordered[self.last_selected_arm_index]
                    reward_color = 'GREEN' if current_reward_for_bandit >= 0 else 'RED'
                    
                    colored_key = self._colorize(f'Key: {last_arm_key}', 'CYAN')
                    colored_reward = self._colorize(f'{current_reward_for_bandit:.3f}', reward_color)
                    
                    self._log(INFO, f"CB Lib: Updated ArmIdx '{self.last_selected_arm_index}' ({colored_key}) with reward {colored_reward}.")
                except Exception as e:
                    self._log(ERROR, f"CB Lib: Error during model partial_fit/update: {e}") 
            else:
                self._log(WARN, f"CB Lib: Invalid reward ({current_reward_for_bandit}) for arm_idx '{self.last_selected_arm_index}'. Skipping update.")
        
        # 2. Select new arm
        selected_arm_index = 0 
        selected_arm_key_log = self.arm_keys_ordered[0] if self.arm_keys_ordered else "N/A_NO_ARMS"
        scores_for_logging_str = "N/A_SCORES"

        if current_context_vector is None:
            self._log(WARN, "CB Lib: Current context vector is None. Defaulting to 'hold' or first arm.")
            # Logic to select default arm (e.g., 'hold' or random)
            # For LinTS, decision_function might still work or predict might be used.
            # If truly no context, we might need to rely on beta_prior or smoothing if active.
            # For simplicity, let's assume predict might give a random choice or based on priors.
            try:
                if self.arm_keys_ordered: # Ensure there are arms
                    # Create a dummy context or handle how LinTS predicts with None
                    # This part might need adjustment based on how LinTS handles None context
                    # For now, let's try to get a prediction even with a dummy context if necessary
                    # Or, more simply, pick a random arm or a 'hold' arm.
                    selected_arm_key_default = "hold"
                    if "hold" in self.arm_keys_ordered:
                        selected_arm_index = self.arm_keys_ordered.index("hold")
                    elif self.arm_keys_ordered:
                        selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                    selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                    scores_for_logging_str = "[Defaulted due to None context - random/hold]"
                else: # Should not happen
                    selected_arm_key_log = "N/A_NO_ARMS"
                    scores_for_logging_str = "[Error: No arms and None context]"

            except Exception as e:
                 self._log(ERROR, f"CB Lib: Error during default arm selection for None context: {e}")
                 if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                 if self.arm_keys_ordered: selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]

        else: # Context is available
            try:
                # LinTS typically uses .predict() which directly gives the chosen arm index
                # based on sampling. decision_function might give scores before sampling.
                # Let's assume we want the sampled choice directly.
                # The `predict` method of LinTS in this library returns the chosen arm index.
                reshaped_context = current_context_vector.reshape(1, -1)
                predicted_arm_indices = self.contextual_bandit_model.predict(reshaped_context)
                
                if isinstance(predicted_arm_indices, np.ndarray) and predicted_arm_indices.size == 1:
                    selected_arm_index = int(predicted_arm_indices[0])
                    if 0 <= selected_arm_index < len(self.arm_keys_ordered):
                        selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                        # LinTS doesn't naturally provide "scores" for all arms in the same way LinUCB does,
                        # as it samples coefficients and then picks the max.
                        # We could call decision_function if we want to see the raw E[reward] before sampling,
                        # but the choice is based on the sampled coefficients.
                        try:
                            # For logging scores (expected values before sampling for this step)
                            raw_scores_output = self.contextual_bandit_model.decision_function(reshaped_context)
                            if isinstance(raw_scores_output, np.ndarray) and raw_scores_output.ndim == 2 and raw_scores_output.shape[0] == 1:
                                actual_scores_array = raw_scores_output[0]
                                scores_for_logging_str = ", ".join([
                                    f"{self.arm_keys_ordered[i]}:{s:.3f}" 
                                    for i, s in enumerate(actual_scores_array) 
                                    if i < len(self.arm_keys_ordered)
                                ])
                            else:
                                scores_for_logging_str = "[Scores not in expected array format]"
                        except Exception as e_score:
                            self._log(DEBUG_ALL, f"CB Lib: Could not get decision_function scores for logging: {e_score}")
                            scores_for_logging_str = "[Scores N/A for LinTS predict]"

                    else:
                        self._log(ERROR, f"CB Lib: Predicted arm index {selected_arm_index} out of bounds. Defaulting.")
                        if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                        if self.arm_keys_ordered: selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                        scores_for_logging_str = "[Error: Predicted index out of bounds]"
                else:
                    self._log(ERROR, f"CB Lib: Predict method returned unexpected format: {predicted_arm_indices}. Defaulting.")
                    if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                    if self.arm_keys_ordered: selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                    scores_for_logging_str = "[Error: Unexpected predict format]"

            except Exception as e:
                self._log(ERROR, f"CB Lib: Error during arm selection (predict or processing): {e}. Defaulting arm.")
                if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
                if self.arm_keys_ordered: selected_arm_key_log = self.arm_keys_ordered[selected_arm_index]
                scores_for_logging_str = "[Error: Exception in arm selection]"
        
        colored_selection = self._colorize(selected_arm_key_log, 'BLUE')
        self._log(INFO, f"CB Lib: Selected ArmIdx '{selected_arm_index}' (Key: {colored_selection}). Scores (expected): [{scores_for_logging_str}]")
        
        self.last_selected_arm_index = selected_arm_index
        self.last_context_vector = current_context_vector # Store the non-reshaped one

        actual_selected_arm_key_from_idx = self.arm_keys_ordered[selected_arm_index] if self.arm_keys_ordered and selected_arm_index < len(self.arm_keys_ordered) else "hold" 
        if actual_selected_arm_key_from_idx not in self.bandit_actions: 
            self._log(WARN, f"Selected arm key '{actual_selected_arm_key_from_idx}' not in bandit_actions. Defaulting to 'hold'.")
            actual_selected_arm_key_from_idx = "hold" if "hold" in self.bandit_actions else (self.arm_keys_ordered[0] if self.arm_keys_ordered else "error_no_arm")

        tdp_delta_w = self.bandit_actions.get(actual_selected_arm_key_from_idx, 0.0)
        
        # Base TDP for decision should be the one the bandit intended to operate from if it was set last.
        # If PID interfered, self.current_tdp_w would be different.
        # Using optimizer_target_tdp_w as the base for *this* decision.
        base_tdp_for_bandit_decision = self.optimizer_target_tdp_w 
        proposed_next_tdp_by_bandit = base_tdp_for_bandit_decision + tdp_delta_w
        
        self._log(INFO, f"CB Lib Action: ArmKey='{actual_selected_arm_key_from_idx}', Delta={tdp_delta_w:.1f}W. "
                        f"Base TDP (optimizer target): {base_tdp_for_bandit_decision:.1f}W. "
                        f"Proposed TDP: {proposed_next_tdp_by_bandit:.1f}W.")
        if current_context_vector is not None:
             self._log(DEBUG_ALL, f"CB Lib Context (used for decision): {['{:.2f}'.format(x) for x in current_context_vector]}")

        actual_set_tdp, requested_tdp = self._set_tdp_limit_w(proposed_next_tdp_by_bandit, context=f"Optimizer CB Lib (Arm: {actual_selected_arm_key_from_idx})")
        self.optimizer_target_tdp_w = actual_set_tdp
        self.optimizer_target_tdp_w = self._read_current_tdp_limit_w() # Update target to what was actually set

        self.last_action_actual_tdp = actual_set_tdp
        self.last_action_requested_tdp = requested_tdp

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
            
            if ok or self.last_pkg_energy_uj is None: # Update baseline if first read or successful calculation
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

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes):
        self._log(DEBUG_KPM, f"KPM CB: Agent:{e2_agent_id}, Sub(E2EventInstID):{subscription_id}, Time:{time.monotonic():.3f}")
        if not self.e2sm_kpm: self._log(WARN, f"KPM from {e2_agent_id}, but e2sm_kpm unavailable."); return
    
        try:
            kpm_hdr_info = self.e2sm_kpm.extract_hdr_info(indication_hdr_bytes)
            kpm_meas_data = self.e2sm_kpm.extract_meas_data(indication_msg_bytes)
    
            if not kpm_meas_data: 
                self._log(WARN, f"KPM CB Style 4: Failed/empty KPM data from {e2_agent_id}. HDR: {kpm_hdr_info}"); return
            
            ue_meas_data_map = kpm_meas_data.get("ueMeasData", {})
            if not isinstance(ue_meas_data_map, dict):
                self._log(WARN, f"KPM CB Style 4: Invalid 'ueMeasData' from {e2_agent_id}. Data: {kpm_meas_data}"); return
            
            if not ue_meas_data_map: 
                with self.kpm_data_lock: 
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
                if not isinstance(ue_metrics, dict): continue 
                
                for metric_name, value_list in ue_metrics.items():
                    if isinstance(value_list, list) and value_list:
                        value = sum(val for val in value_list if isinstance(val, (int, float)))
                        try:
                            if metric_name == 'DRB.RlcSduTransmittedVolumeDL':
                                ue_dl_bits_this_ue = float(value) * 1000.0 
                                gNB_data_this_report['dl_bits'] += ue_dl_bits_this_ue
                            elif metric_name == 'DRB.RlcSduTransmittedVolumeUL':
                                ue_ul_bits_this_ue = float(value) * 1000.0 
                                gNB_data_this_report['ul_bits'] += ue_ul_bits_this_ue
                            elif metric_name == 'RRU.PrbTotDl': 
                                gNB_data_this_report['dl_prb'] += float(value)
                            elif metric_name == 'RRU.PrbTotUl': 
                                gNB_data_this_report['ul_prb'] += float(value)
                        except (ValueError, TypeError) as e:
                             self._log(WARN, f"KPM CB Style 4: Metric '{metric_name}' for UE {global_ue_id} value '{value_list}' processing error: {e}.")
                
                with self.kpm_data_lock: 
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


    def _get_and_reset_accumulated_kpm_metrics(self) -> Dict[str, Dict[str, Any]]:
        with self.kpm_data_lock:
            snap = {}
            for gnb_id, data in self.accumulated_kpm_metrics.items():
                snap[gnb_id] = {
                    'dl_bits': data.get('bits_sum_dl', 0.0),
                    'ul_bits': data.get('bits_sum_ul', 0.0),
                    'dl_prb_sum_percentage': data.get('prb_sum_dl', 0.0),
                    'ul_prb_sum_percentage': data.get('prb_sum_ul', 0.0),
                    'reports_in_interval': data.get('num_reports', 0)
                }
                data['bits_sum_dl'] = 0.0; data['bits_sum_ul'] = 0.0
                data['prb_sum_dl'] = 0.0; data['prb_sum_ul'] = 0.0
                data['num_reports'] = 0
        return snap

    def _get_and_reset_active_ue_count_and_data(self) -> Tuple[int, Dict[str, Dict[str, float]]]:
        active_ue_count = 0
        threshold_bits_over_optimizer_interval = self.active_ue_throughput_threshold_mbps * 1e6 * self.optimizer_decision_interval_s

        with self.kpm_data_lock:
            for global_ue_id, ue_data in self.current_interval_per_ue_data.items():
                if ue_data.get('total_bits', 0.0) >= threshold_bits_over_optimizer_interval :
                    active_ue_count += 1
            per_ue_data_snap = self.current_interval_per_ue_data.copy()
            self.current_interval_per_ue_data.clear()
            self.current_interval_ue_ids.clear()
        return active_ue_count, per_ue_data_snap

    def _setup_kpm_subscriptions(self):
        self._log(INFO, "--- Setting up KPM Style 4 Subscriptions (Per-UE Metrics) ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable. Cannot subscribe."); return
        
        nodes = list(self.gnb_ids_map.values()) 
        if not nodes: self._log(WARN, "No gNB IDs configured for KPM subscriptions."); return

        kpm_config = self.config.get('kpm_subscriptions', {})
        style4_metrics = kpm_config.get('style4_metrics_per_ue', ['DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL', 'RRU.PrbTotDl', 'RRU.PrbTotUl'])
        style4_report_p_ms = int(kpm_config.get('style4_report_period_ms', 1000))
        style4_gran_p_ms = int(kpm_config.get('style4_granularity_period_ms', style4_report_p_ms))
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
            self.current_tdp_w = self._read_current_tdp_limit_w()
            # Initialize optimizer_target_tdp_w with the actual current TDP.
            # This is the TDP the bandit should consider as its starting point if PID hasn't intervened.
            self.optimizer_target_tdp_w = self.current_tdp_w
            self._log(INFO, f"Initial current TDP: {self.current_tdp_w:.1f}W. Optimizer base target set to this.")


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
            self.total_bits_from_previous_optimizer_interval = 0.0

            self._log(INFO, f"\n--- Starting Monitoring & Control Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            self._log(INFO, f"RU PID Interval: {self.ru_timing_pid_interval_s}s | Target RU CPU: {self.target_ru_cpu_usage if self.ru_timing_core_indices else 'N/A'}%")
            self._log(INFO, f"CB Optimizer Interval: {self.optimizer_decision_interval_s}s | TDP Range: {self.tdp_min_w}W-{self.tdp_max_w}W")
            # Log LinTS specific parameters instead of LinUCB alpha
            self._log(INFO, f"CB (BootstrapTS) Actions: {self.bandit_actions}")
            self._log(INFO, f"CB Context Dim (features only): {self.context_dimension_features_only}, FitIntercept: {self.bts_fit_intercept}. ActiveUE Thresh: {self.active_ue_throughput_threshold_mbps} Mbps.")
            self._log(INFO, f"Stats Print Interval: {self.stats_print_interval_s}s")

            # --- Rule-based descent parameters (optional) ---
            enable_rule_based_idle_descent = bool(self.config.get('contextual_bandit',{}).get('enable_rule_based_idle_descent', False))
            rule_idle_descent_tdp_thresh = float(self.config.get('contextual_bandit',{}).get('rule_idle_descent_tdp_thresh_w', self.tdp_min_w + 20)) # e.g. 110W
            rule_idle_descent_step_w = float(self.config.get('contextual_bandit',{}).get('rule_idle_descent_step_w', 20.0))
            rule_idle_descent_max_steps = int(self.config.get('contextual_bandit',{}).get('rule_idle_descent_max_steps', 3))
            rule_idle_descent_steps_taken = 0


            while self.running: 
                loop_start_time = time.monotonic()

                if self.ru_timing_core_indices: self._update_ru_core_msr_data()
                current_ru_cpu_usage_control_val = self._get_control_ru_timing_cpu_usage()

                # Update current_tdp_w to reflect actual hardware state before any decisions
                self.current_tdp_w = self._read_current_tdp_limit_w()

                if loop_start_time - self.last_ru_pid_run_time >= self.ru_timing_pid_interval_s:
                    if self.ru_timing_core_indices: self._run_ru_timing_pid_step(current_ru_cpu_usage_control_val)
                    self.last_ru_pid_run_time = loop_start_time
                
                if loop_start_time - self.last_optimizer_run_time >= self.optimizer_decision_interval_s:
                    # Refresh current_tdp_w again as PID might have acted
                    was_pid_triggered_in_interval = False
                    self.current_tdp_w = self._read_current_tdp_limit_w()

                    interval_energy_uj = self._get_interval_energy_uj_for_optimizer()
                    kpm_summed_data = self._get_and_reset_accumulated_kpm_metrics()
                    current_num_active_ues, _ = self._get_and_reset_active_ue_count_and_data()
                    self.current_num_active_ues_for_log = current_num_active_ues

                    total_dl_bits_interval = sum(d.get('dl_bits', 0.0) for d in kpm_summed_data.values())
                    total_ul_bits_interval = sum(d.get('ul_bits', 0.0) for d in kpm_summed_data.values())
                    total_bits_optimizer_interval = total_dl_bits_interval + total_ul_bits_interval
                    
                    total_prb_dl_percentage = sum(d.get('dl_prb_sum_percentage', 0.0) for d in kpm_summed_data.values())
                    total_prb_ul_percentage = sum(d.get('ul_prb_sum_percentage', 0.0) for d in kpm_summed_data.values())
                    num_kpm_reports_processed = sum(d.get('reports_in_interval',0) for d in kpm_summed_data.values())

                    significant_throughput_change = False
                    if self.total_bits_from_previous_optimizer_interval is not None: 
                        denominator = self.total_bits_from_previous_optimizer_interval
                        if denominator < 1e-6: 
                            if total_bits_optimizer_interval > 1e6 : significant_throughput_change = True 
                        elif abs(total_bits_optimizer_interval - denominator) / denominator > self.throughput_change_threshold_for_discard:
                            relative_change = abs(total_bits_optimizer_interval - denominator) / denominator
                            self._log(WARN, f"Optimizer: Sig. throughput change ({relative_change*100:.1f}%). Update might be skipped.")
                            significant_throughput_change = True
                    
                    reset_efficiency_baseline = False
                    
                    # Only perform the check if we have a full window of history to compare against
                    if len(self.recent_workload_bits) >= self.workload_avg_window_size:
                        # Calculate the moving average of the recent past
                        historical_avg_bits = sum(self.recent_workload_bits) / len(self.recent_workload_bits)
                        
                        # Only consider it a potential drop if the historical average was significant
                        if historical_avg_bits > 1e7: # e.g., > 10 Megabits on average per interval
                            
                            # Use a safe divisor
                            safe_historical_avg = max(historical_avg_bits, 1.0)
                            workload_ratio = total_bits_optimizer_interval / safe_historical_avg
                            
                            if workload_ratio < self.workload_drop_threshold:
                                self._log(INFO, f"Workload drop detected (Current Bits: {total_bits_optimizer_interval/1e6:.1f}Mb "
                                              f"< {self.workload_drop_threshold*100:.0f}% of historical avg: {historical_avg_bits/1e6:.1f}Mb). "
                                              f"Resetting max_efficiency_seen.")
                                reset_efficiency_baseline = True
                    
                    # --- Update the memory window ---
                    self.recent_workload_bits.append(total_bits_optimizer_interval)
                    if len(self.recent_workload_bits) > self.workload_avg_window_size:
                        self.recent_workload_bits.pop(0)
                    
                    # --- Act on the flag ---
                    if reset_efficiency_baseline:
                        self.max_efficiency_seen = 0.0
                        # Also clear the history so the baseline can be rebuilt from scratch after the drop
                        self.recent_workload_bits.clear()

                    # Determine effective TDP for reward calculation (the one bandit was aiming for unless PID overrode)
                    # self.optimizer_target_tdp_w was set by the *previous* bandit action OR by PID.
                    # self.current_tdp_w is the *actual* current HW TDP.
                    # For reward, we should evaluate the consequence of the TDP that was *active* during the interval.
                    # This is tricky if PID acts mid-interval or just before reward.
                    # Let's use self.optimizer_target_tdp_w (what the system was set to at start of interval by CB/PID)
                    # This `tdp_for_reward_eval` is the one set at the end of the *previous* optimizer step.
                    tdp_for_reward_eval = self.optimizer_target_tdp_w

                    reward_for_bandit = 0.0
                    # --- HIERARCHICAL REWARD CALCULATION ---

                    # 1. Determine system state and last action
                    current_raw_efficiency = 0.0
                    if num_kpm_reports_processed > 0 and interval_energy_uj is not None and interval_energy_uj > 1e-3:
                        current_raw_efficiency = total_bits_optimizer_interval / interval_energy_uj

                    # --- ADAPTIVE NORMALIZATION & STATE CHANGE LOGIC ---
                    is_active_ue_present = (current_num_active_ues > 0)
                    if (self.total_bits_from_previous_optimizer_interval is not None and
                            self.total_bits_from_previous_optimizer_interval > 1e6): # Avoid triggering on idle fluctuations
                    
                        # Calculate the ratio of current throughput to previous throughput.
                        workload_ratio = total_bits_optimizer_interval / self.total_bits_from_previous_optimizer_interval
                        
                        if workload_ratio < self.workload_drop_threshold:
                            self._log(INFO, f"Workload drop detected (ratio: {workload_ratio:.2f} < {self.workload_drop_threshold}). "
                                          f"Resetting max_efficiency_seen from {self.max_efficiency_seen:.3f} to 0.")
                            self.max_efficiency_seen = 0.0
                    # Decay pushes the agent to hold, disable it.
                    # self.max_efficiency_seen *= self.max_eff_decay_factor 
                    # Ensure the new efficiency measurement can set the max if it's currently zero.
                    self.max_efficiency_seen = max(self.max_efficiency_seen, current_raw_efficiency)
                    
                    # Calculate normalized efficiency SAFELY.
                    safe_max_seen = max(self.max_efficiency_seen, 1e-6)
                    normalized_efficiency = current_raw_efficiency / safe_max_seen
                    normalized_efficiency = min(normalized_efficiency, 1.0) # Clamp
                    
                    # Now update the throughput for the *next* interval's comparison
                    self.total_bits_from_previous_optimizer_interval = total_bits_optimizer_interval
                    
                    # --- CONTEXT VECTOR CREATION ---
                    current_actual_tdp_for_context = self.current_tdp_w
                    current_context_vec = self._get_current_context_vector(
                        current_num_active_ues,
                        current_ru_cpu_usage_control_val,
                        current_actual_tdp_for_context,
                        # Pass the pre-calculated normalized efficiency
                        normalized_efficiency 
                    )


                    # --- HIERARCHICAL REWARD CALCULATION ---
                    
                    # Get last action info
                    action_delta_w = 0.0
                    chosen_arm_key = "N/A"
                    if self.last_selected_arm_index is not None and self.arm_keys_ordered:
                        chosen_arm_key = self.arm_keys_ordered[self.last_selected_arm_index]
                        action_delta_w = self.bandit_actions.get(chosen_arm_key, 0.0)

                    # HIERARCHY 1: PID Trigger
                    if self.pid_triggered_since_last_decision:
                        self.pid_triggered_since_last_decision = False
                        if action_delta_w <= 0:
                            reward_for_bandit = -0.4
                            self._log(WARN, f"CB REWARD: PID TRIGGER OVERRIDE. Action '{chosen_arm_key}' was wrong. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', 'RED')}")
                        else:
                            reward_for_bandit = 0.4 # Strong incentive
                            self._log(INFO, f"CB REWARD: PID TRIGGER OVERRIDE. Action '{chosen_arm_key}' was correct. Applying strong incentive bonus. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', 'GREEN')}")

                    
                    # # HIERARCHY 2: Stressed State
                    # elif current_ru_cpu_usage_control_val > (self.target_ru_cpu_usage * 0.99):
                    #     if action_delta_w > 0:
                    #         reward_for_bandit = 0.05 # Gentle nudge
                    #     else:
                    #         danger_zone_start = self.target_ru_cpu_usage * 0.99
                    #         penalty_progress = (current_ru_cpu_usage_control_val - danger_zone_start) / (self.target_ru_cpu_usage - danger_zone_start)
                    #         reward_for_bandit = -0.2 * np.clip(penalty_progress, 0, 1) # Milder penalty
                    #     reward_color = 'GREEN' if reward_for_bandit >=0 else 'RED'
                    #     self._log(INFO, f"CB Reward (Stressed): CPU at {current_ru_cpu_usage_control_val:.2f}%. Action '{chosen_arm_key}'. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', reward_color)}")

                    # HIERARCHY 3: Normal Operation (Idle or Active)
                    # elif is_active_ue_present:
                    #     # ACTIVE & HEALTHY: Reward is based on shaped efficiency
                    #     reward_for_bandit = normalized_efficiency ** 2 # Shape the reward to be "greedy"
                    #     colored_max_seen = self._colorize(f'{self.max_efficiency_seen:.3f}', 'WHITE')
                    #     self._log(INFO, f"CB Reward (Active/Healthy): RawEff={current_raw_efficiency:.3f} b/uJ, NormEff={normalized_efficiency:.3f} (MaxSeen={colored_max_seen}). Final Shaped Reward={self._colorize(f'{reward_for_bandit:.3f}', 'GREEN')}")
                
                    elif is_active_ue_present:
                        # --- Start with the efficiency-based reward as the default ---
                        reward_for_bandit = normalized_efficiency ** 2
                        
                        # --- Now, apply a penalty if stressed AND action was wrong ---
                        if current_ru_cpu_usage_control_val > (self.target_ru_cpu_usage * 0.99):
                            if action_delta_w <= 0: # If we held or decreased TDP when stressed...
                                self._log(WARN, f"CB REWARD MOD: Stressed CPU at {current_ru_cpu_usage_control_val:.2f}%. Applying penalty of -0.3 to efficiency reward.")
                                reward_for_bandit -= 0.3
                        if (self.last_action_requested_tdp is not None and self.last_action_actual_tdp is not None and action_delta_w > 0 and abs(self.last_action_requested_tdp - self.last_action_actual_tdp) > 1e-3): # Check if the action was clipped
                            # The requested action was outside the operational range. This is an ineffective choice.
                            # We assign a moderate penalty to discourage this.
                            reward_for_bandit =- 0.25 
                            self._log(WARN, f"CB REWARD: Ineffective Action, -0.25 penalty added. Requested {self.last_action_requested_tdp:.1f}W but clipped to {self.last_action_actual_tdp:.1f}W. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', 'RED')}")
                                            # Final clipping and setting the log variable
                        # Log the result
                        colored_max_seen = self._colorize(f'{self.max_efficiency_seen:.3f}', 'WHITE')
                        reward_color = 'GREEN' if reward_for_bandit >= 0 else 'RED'
                        self._log(INFO, f"CB Reward (Active): RawEff={current_raw_efficiency:.3f} b/uJ, NormEff={normalized_efficiency:.3f}, MaxSeen={colored_max_seen}. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', reward_color)}")
                    else: # TRUE IDLE: Not stressed and no UEs
                        holding_zone_width_w = 5.0
                        if tdp_for_reward_eval <= (self.tdp_min_w + holding_zone_width_w):
                            if action_delta_w == 0: reward_for_bandit = 0.2
                            elif action_delta_w < 0: reward_for_bandit = 0.05
                            else: reward_for_bandit = -0.4
                        else:
                            if action_delta_w < 0:
                                normalized_tdp_excursion = (tdp_for_reward_eval - self.tdp_min_w) / (self.tdp_max_w - self.tdp_min_w) if (self.tdp_max_w - self.tdp_min_w) > 0 else 0
                                reward_for_bandit = 0.2 * (1.0 - normalized_tdp_excursion) + 0.3
                            elif action_delta_w == 0: reward_for_bandit = 0.0
                            else:
                                normalized_tdp_excursion = (tdp_for_reward_eval - self.tdp_min_w) / (self.tdp_max_w - self.tdp_min_w) if (self.tdp_max_w - self.tdp_min_w) > 0 else 0
                                reward_for_bandit = -0.3 - 0.2 * normalized_tdp_excursion
                        
                        reward_color = 'GREEN' if reward_for_bandit >= 0 else 'RED'
                        self._log(INFO, f"CB Reward (True Idle): Action '{chosen_arm_key}'. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', reward_color)}")

                    reward_for_bandit = np.clip(reward_for_bandit, -1.0, 1.0)
                    self.most_recent_calculated_reward_for_log = reward_for_bandit

                    # Now, run the optimizer step
                    self._run_contextual_bandit_optimizer_step(reward_for_bandit, current_context_vec, significant_throughput_change)
                    
                    self.last_optimizer_run_time = loop_start_time
                
                if loop_start_time - self.last_stats_print_time >= self.stats_print_interval_s:
                    pkg_pwr_w, pkg_pwr_ok = self._get_pkg_power_w()
                    ru_usage_str = "N/A"
                    if self.ru_timing_core_indices:
                        ru_usage_parts = [f"C{cid}:{self.ru_core_msr_prev_data.get(cid).busy_percent:>6.2f}%" if self.ru_core_msr_prev_data.get(cid) else f"C{cid}:N/A" for cid in self.ru_timing_core_indices]
                        ru_usage_str = f"[{', '.join(ru_usage_parts)}] (AvgMax:{current_ru_cpu_usage_control_val:>6.2f}%)"
                    pkg_pwr_log_str = f"{pkg_pwr_w:.1f}" if pkg_pwr_ok else "N/A"
                    
                    last_arm_key_str = "N/A"
                    if self.last_selected_arm_index is not None and self.last_selected_arm_index < len(self.arm_keys_ordered):
                         last_arm_key_str = self.arm_keys_ordered[self.last_selected_arm_index]
                    
                    bandit_log = (f"CB_Lib(LastArmIdx:{self.last_selected_arm_index if self.last_selected_arm_index is not None else 'N/A'}, "
                                  f"Key:{last_arm_key_str})")
                    
                    log_parts = [f"RU:{ru_usage_str}", f"TDP_Act:{self.current_tdp_w:>5.1f}W", 
                                 f"TDP_OptTrg:{self.optimizer_target_tdp_w:>5.1f}W", f"PkgPwr:{pkg_pwr_log_str}W", bandit_log]
                    if self.most_recent_calculated_reward_for_log is not None: # Use the unified variable
                         log_parts.append(f"LastReward:{self.most_recent_calculated_reward_for_log:.3f}") # Changed log key
                    log_parts.append(f"ActiveUEs:{self.current_num_active_ues_for_log}")

                    self._log(INFO, " | ".join(log_parts)); 
                    self.last_stats_print_time = loop_start_time
                
                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, self.main_loop_sleep_s - loop_duration)
                if sleep_time > 0 : time.sleep(sleep_time)

        except KeyboardInterrupt: self._log(INFO, "\nLoop interrupted (KeyboardInterrupt).")
        except SystemExit as e: self._log(INFO, f"Application exiting (SystemExit: {e})."); raise 
        except RuntimeError as e: self._log(ERROR, f"Critical runtime error in loop: {e}."); raise 
        except Exception as e: self._log(ERROR, f"\nUnexpected error in loop: {e}"); import traceback; self._log(ERROR, traceback.format_exc()); raise 
        finally: self._log(INFO, "--- Power Manager xApp run_power_management_xapp finished. ---")

if __name__ == "__main__":
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
        # manager._log(INFO, "Registered signal handlers from xAppBase.") # Already logged in superclass
        manager.run_power_management_xapp()
    except RuntimeError as e: 
        print(f"E: Critical error: {e}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and hasattr(manager.logger, 'hasHandlers') and manager.logger.hasHandlers(): manager._log(ERROR, f"Critical error: {e}")
        sys.exit(1)
    except SystemExit as e: 
        code = e.code if e.code is not None else 0
        print(f"Application terminated with exit code: {code}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and hasattr(manager.logger, 'hasHandlers') and manager.logger.hasHandlers(): manager._log(INFO, f"Application terminated (SystemExit: {code}).")
        sys.exit(code) 
    except Exception as e:
        print(f"E: An unexpected error at top level: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        if manager and hasattr(manager, '_log') and hasattr(manager.logger, 'hasHandlers') and manager.logger.hasHandlers(): manager._log(ERROR, f"TOP LEVEL UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        msg = "Application finished."
        if manager and hasattr(manager, '_log') and hasattr(manager.logger, 'hasHandlers') and manager.logger.hasHandlers(): manager._log(INFO, msg)
        else: print(f"INFO: {msg} (logger may not be available).")
