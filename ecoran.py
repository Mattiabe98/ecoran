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
import collections

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

        self._setup_logging()
        
        xapp_base_config_file = self.config.get('xapp_base_config_file', '')
        super().__init__(xapp_base_config_file, http_server_port, rmr_port)

        self._initialize_parameters()
        self._initialize_contextual_bandit()
        self._initialize_state_variables()
        
        self.COLORS = {
            'RED': '\033[91m', 'GREEN': '\033[92m', 'YELLOW': '\033[93m',
            'BLUE': '\033[94m', 'MAGENTA': '\033[95m', 'CYAN': '\033[96m',
            'WHITE': '\033[97m', 'BOLD': '\033[1m', 'RESET': '\033[0m'
        }
        
        self._validate_system_paths()
        if self.dry_run: self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _initialize_parameters(self):
        """Load operational parameters from the config file."""
        self.verbosity = int(self.config.get('console_verbosity_level', INFO))
        
        # --- System Paths and Parameters ---
        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        self.max_energy_val_rapl = self.config.get('rapl_max_energy_uj_override', 2**60 - 1)

        # --- Timing & Control Loop Parameters ---
        self.main_loop_sleep_s = float(self.config.get('main_loop_sleep_s', 0.1))
        self.ru_timing_pid_interval_s = float(self.config.get('ru_timing_pid_interval_s', 1.0))
        # DEPRECATED: The optimizer interval is now event-driven (report count) or by timeout.
        # This value is now only used as a fallback for the timeout.
        self.optimizer_decision_interval_s = float(self.config.get('optimizer_decision_interval_s', 5.0))
        self.stats_print_interval_s = float(self.config.get('stats_print_interval_s', self.optimizer_decision_interval_s))

        # --- TDP Management Parameters ---
        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 90))
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 170))
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 99.8))
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.0005))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1.0))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 3.0))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 1.5))
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        self.dry_run = bool(self.config.get('dry_run', False))

        # --- KPM Setup ---
        self.kpm_ran_func_id = int(self.config.get('kpm_ran_func_id', 2))
        self.gnb_ids_map = self.config.get('gnb_ids', {}) 

    def _initialize_contextual_bandit(self):
        """Set up the contextual bandit model and its parameters."""
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

        beta_prior_config = cb_config.get('beta_prior', "uniform_optimistic")
        self.lints_beta_prior = None
        if isinstance(beta_prior_config, str):
            if beta_prior_config == "auto": self.lints_beta_prior = "auto"
            elif beta_prior_config == "uniform_optimistic":
                self.lints_beta_prior = ((1, 1), 3) 
                self._log(INFO, "Using 'uniform_optimistic' Beta(1,1) prior.")
        elif beta_prior_config is not None:
             self.lints_beta_prior = beta_prior_config
        self.lints_smoothing = cb_config.get('smoothing', None)

        bts_config = self.config.get('bootstrapped_ts', {})
        self.bts_nsamples = int(bts_config.get('nsamples', 20))
        self.bts_lambda_ = float(bts_config.get('lambda_', 1.0))
        self.bts_fit_intercept = bool(bts_config.get('fit_intercept', True))
        
        beta_prior_config = bts_config.get('beta_prior', "uniform_optimistic")
        self.bts_beta_prior = None
        if isinstance(beta_prior_config, str):
            if beta_prior_config == "auto": self.bts_beta_prior = "auto"
            elif beta_prior_config == "uniform_optimistic": self.bts_beta_prior = ((1, 1), 3)
        
        self._log(INFO, f"Initializing BootstrappedTS with nsamples={self.bts_nsamples}")
        base_algo = LinearRegression(lambda_=self.bts_lambda_, fit_intercept=self.bts_fit_intercept)
        self.contextual_bandit_model = BootstrappedTS(
            base_algorithm=base_algo, nchoices=len(self.arm_keys_ordered),
            nsamples=self.bts_nsamples, batch_train=True, beta_prior=self.bts_beta_prior,
            njobs_arms=1, njobs_samples=1, random_state=42)

        self.throughput_change_threshold_for_discard = float(cb_config.get('throughput_change_threshold_for_discard', 1.0))
        self.active_ue_throughput_threshold_mbps = float(cb_config.get('active_ue_throughput_threshold_mbps', 1.0))
        self.norm_params = cb_config.get('normalization_parameters', {})
        self._ensure_default_norm_params()

    def _initialize_state_variables(self):
        """Initialize dynamic state variables for the control loop."""
        self.current_tdp_w = float(self.tdp_min_w) 
        self.last_pkg_energy_uj: Optional[int] = None
        self.last_energy_read_time: Optional[float] = None
        self.energy_at_last_optimizer_interval_uj: Optional[int] = None
        self.max_ru_timing_usage_history: List[float] = []
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}
        
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None: self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else: self._log(WARN, "xAppBase.e2sm_kpm module unavailable."); self.e2sm_kpm = None
        
        self.kpm_data_lock = threading.Lock()
        self.accumulated_kpm_metrics: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'num_reports':0}
        )
        self.current_interval_per_ue_data: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: {'total_bits': 0.0})

        self.optimizer_target_tdp_w = self.current_tdp_w
        self.last_selected_arm_index: Optional[int] = None
        self.last_context_vector: Optional[np.array] = None
        self.total_bits_from_previous_optimizer_interval: Optional[float] = None
        self.was_idle_in_previous_step = True 

        self.last_ru_pid_run_time: float = 0.0
        self.last_optimizer_run_time: float = 0.0
        self.last_stats_print_time: float = 0.0
        self.most_recent_calculated_reward_for_log: Optional[float] = None
        self.current_num_active_ues_for_log: int = 0
        self.pid_triggered_since_last_decision = False
        self.last_action_actual_tdp: Optional[float] = None
        self.last_action_requested_tdp: Optional[float] = None
        self.last_raw_efficiency: float = 0.0
        self.last_normalized_efficiency: float = 0.0

        self.last_interval_avg_power_w: float = 0.0
        kpm_config = self.config.get('kpm_subscriptions', {})
        self.optimizer_reports_per_du = int(kpm_config.get('optimizer_reports_per_du', 10))
        self.optimizer_max_interval_s = float(kpm_config.get('optimizer_max_interval_s', self.optimizer_decision_interval_s * 1.5))
        self.expected_du_ids = set(self.gnb_ids_map.values())
        self.num_expected_dus = len(self.expected_du_ids)
        self.reports_received_this_interval: Dict[str, int] = collections.defaultdict(int)
        self._log(INFO, f"Expecting KPM reports from {self.num_expected_dus} DUs: {self.expected_du_ids}")
        self._log(INFO, f"Optimizer will trigger after {self.optimizer_reports_per_du} reports from each DU or after {self.optimizer_max_interval_s}s timeout.")
        self.du_activity_timeout_s = float(kpm_config.get('du_activity_timeout_s', 5.0))
        self._log(INFO, f"A DU will be considered idle if no report is received for {self.du_activity_timeout_s}s.")
        
        self.baseline_window_size = int(self.config.get('contextual_bandit', {}).get('baseline_window_size', 50))
        self.baseline_percentile = float(self.config.get('contextual_bandit', {}).get('baseline_percentile', 90.0))
        self.min_baseline_samples = int(self.config.get('contextual_bandit', {}).get('min_baseline_samples', 10))
        self.stable_efficiency_history = collections.deque(maxlen=self.baseline_window_size)
        self.last_report_time_per_du: Dict[str, float] = {}
        
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
                                       current_pkg_power_w: float, # Changed from current_actual_tdp
                                       normalized_efficiency: float 
                                       ) -> np.array:
        
        def _normalize(val, min_val, max_val):
            if (max_val - min_val) < 1e-6: return 0.5
            return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
    
        feature_1_norm_eff = normalized_efficiency
    
        cpu_headroom = self.target_ru_cpu_usage - current_ru_cpu_avg
        norm_params_cpu = self.norm_params.get('cpu_headroom', {'min': -5.0, 'max': 20.0})
        feature_2_cpu_headroom = _normalize(cpu_headroom, norm_params_cpu.get('min'), norm_params_cpu.get('max'))
    
        # The actual fix: Normalize based on ACTUAL power consumed, not the LIMIT.
        # The range is still based on the TDP limits, as that defines the operational boundaries.
        feature_3_power_pos = _normalize(current_pkg_power_w, self.tdp_min_w, self.tdp_max_w)
    
        norm_params_ues = self.norm_params.get('num_active_ues', {'min': 0.0, 'max': 10.0})
        feature_4_num_ues = _normalize(float(current_num_active_ues), norm_params_ues.get('min'), norm_params_ues.get('max'))
        
        final_features = np.array([
            feature_1_norm_eff,
            feature_2_cpu_headroom,
            feature_3_power_pos, # Use the new feature
            feature_4_num_ues
        ])
        
        self._log(DEBUG_ALL, f"Context Vector (Normalized): [Eff:{final_features[0]:.2f}, "
                             f"CPU_Headroom:{final_features[1]:.2f}, Pwr_Pos:{final_features[2]:.2f}, " # Updated log
                             f"Num_UEs:{final_features[3]:.2f}]")
    
        return final_features
        
    def _setup_logging(self):
        self.file_verbosity_cfg = int(self.config.get('file_verbosity_level', DEBUG_KPM))
        self.log_file_path_base = self.config.get('log_file_path', "/mnt/data/ecoran")
        self.logger = logging.getLogger("EcoRANPowerManager")
        self.logger.handlers = [] 
        self.logger.propagate = False
        console_level = LOGGING_LEVEL_MAP.get(int(self.config.get('console_verbosity_level', INFO)), logging.INFO)
        file_level = LOGGING_LEVEL_MAP.get(self.file_verbosity_cfg, logging.DEBUG)
        overall_logger_level = min(console_level, file_level) if not (console_level > logging.CRITICAL and self.file_verbosity_cfg == SILENT) else logging.CRITICAL + 10
        self.logger.setLevel(overall_logger_level)

        if console_level <= logging.CRITICAL:
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
                if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers) and console_level <= logging.CRITICAL:
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

    def _validate_system_paths(self):
        """Validates that necessary system files and commands are accessible."""
        if not os.path.exists(self.rapl_base_path) or not os.path.exists(self.power_limit_uw_file):
            self._log(ERROR, f"RAPL path {self.rapl_base_path} or power limit file missing. Exiting."); sys.exit(1)
        if not os.path.exists(self.energy_uj_file): self._log(WARN, f"Energy file {self.energy_uj_file} not found.")
        
        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'): self._log(WARN, "'ru_timing_cores' is defined but empty.")
        elif not self.ru_timing_core_indices: self._log(INFO, "No 'ru_timing_cores' defined, RU Timing PID will be disabled.")
        elif self.ru_timing_core_indices:
            tc = self.ru_timing_core_indices[0]; mp = f'/dev/cpu/{tc}/msr'
            if not os.path.exists(mp): self._log(ERROR, f"MSR file {mp} not found. Exiting."); sys.exit(1)
            if read_msr_direct(tc, MSR_IA32_TSC) is None: self._log(ERROR, f"Failed MSR read on core {tc}. Exiting."); sys.exit(1)
            self._log(INFO, "MSR access test passed.")
        
        try:
            subprocess.run([self.intel_sst_path, "--version"], capture_output=True, check=True, text=True)
        except Exception as e:
            self._log(ERROR, f"'{self.intel_sst_path}' failed: {e}. Exiting."); sys.exit(1)
        
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
            
            clos_min_freqs = self.config.get('clos_min_frequency', {}) or {}
            clos_max_freqs = self.config.get('clos_max_frequency', {}) or {}
            clos_weights = self.config.get('clos_weights', {}) or {}
            priority_scheme = self.config.get('clos_priority_scheme', 'ordered').lower()

            if priority_scheme not in ['ordered', 'proportional']:
                self._log(WARN, f"Invalid 'clos_priority_scheme' ('{priority_scheme}'). Defaulting to 'ordered'.")
                priority_scheme = 'ordered'
            self._log(INFO, f"SST-CP: Using '{priority_scheme}' priority scheme.")

            all_clos_ids = sorted(list(set(clos_min_freqs.keys()) | set(clos_max_freqs.keys()) | set(clos_weights.keys())))

            for cid_key in all_clos_ids:
                try:
                    cmd_parts = ["intel-speed-select", "core-power", "config", "-c", str(cid_key)]
                    log_msg_parts = []

                    # Add priority scheme
                    if priority_scheme == 'proportional':
                        cmd_parts.extend(["-p", "0"])
                        weight = clos_weights.get(cid_key)
                        if weight is not None:
                            cmd_parts.extend(["--weight", str(weight)])
                            log_msg_parts.append(f"weight {weight}")
                        else:
                            self._log(WARN, f"SST-CP: Proportional mode enabled, but no weight found for CLOS {cid_key}.")
                    else: # ordered
                        cmd_parts.extend(["-p", "1"])
                    
                    # Add min/max frequencies
                    min_freq = clos_min_freqs.get(cid_key)
                    if min_freq is not None:
                        cmd_parts.extend(["--min", str(min_freq)])
                        log_msg_parts.append(f"min {min_freq}MHz")
                    
                    max_freq = clos_max_freqs.get(cid_key)
                    if max_freq is not None:
                        cmd_parts.extend(["--max", str(max_freq)])
                        log_msg_parts.append(f"max {max_freq}MHz")
                    
                    if len(log_msg_parts) > 0:
                        self._run_command(cmd_parts)
                        log_msg = f"SST-CP: Configured CLOS {cid_key} with " + ", ".join(log_msg_parts) + "."
                        self._log(INFO, log_msg)
                    else:
                        self._log(INFO, f"SST-CP: No min/max/weight configuration for CLOS {cid_key}, skipping config command.")

                except Exception as e_clos_freq:
                    self._log(ERROR, f"SST-CP: Failed to configure CLOS {cid_key}: {e_clos_freq}")

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
        
        pid_critical_ru_cpu_trigger = self.target_ru_cpu_usage
    
        if current_ru_cpu_usage > pid_critical_ru_cpu_trigger:
            self.pid_triggered_since_last_decision = True
            tdp_change_w = self.tdp_adj_step_w_large 
            new_target_tdp = self.current_tdp_w + tdp_change_w
            
            ctx = (f"RU_PID_SAFETY_NET: RU CPU {current_ru_cpu_usage:.2f}% > Trigger {pid_critical_ru_cpu_trigger:.2f}%. "
                   f"Forcing TDP Increase by {tdp_change_w:.1f}W")
            self._set_tdp_limit_w(new_target_tdp, context=ctx)
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

                    self.contextual_bandit_model.partial_fit(X_update, action_update, reward_update)
                    
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
            if "hold" in self.arm_keys_ordered:
                selected_arm_index = self.arm_keys_ordered.index("hold")
            elif self.arm_keys_ordered:
                selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
        else:
            try:
                reshaped_context = current_context_vector.reshape(1, -1)
                predicted_arm_indices = self.contextual_bandit_model.predict(reshaped_context)
                
                if isinstance(predicted_arm_indices, np.ndarray) and predicted_arm_indices.size == 1:
                    selected_arm_index = int(predicted_arm_indices[0])
                else:
                    self._log(ERROR, f"CB Lib: Predict method returned unexpected format: {predicted_arm_indices}. Defaulting.")
                    if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))

            except Exception as e:
                self._log(ERROR, f"CB Lib: Error during arm selection (predict or processing): {e}. Defaulting arm.")
                if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))

        if not (0 <= selected_arm_index < len(self.arm_keys_ordered)):
            self._log(ERROR, f"CB Lib: Predicted arm index {selected_arm_index} out of bounds. Defaulting.")
            if self.arm_keys_ordered: selected_arm_index = random.choice(range(len(self.arm_keys_ordered)))
            else: selected_arm_index = 0
        
        selected_arm_key_log = self.arm_keys_ordered[selected_arm_index] if self.arm_keys_ordered else "N/A"
        
        # Log scores for debugging if possible
        if current_context_vector is not None:
            try:
                raw_scores = self.contextual_bandit_model.decision_function(current_context_vector.reshape(1, -1))
                scores_for_logging_str = ", ".join([f"{self.arm_keys_ordered[i]}:{s:.3f}" for i, s in enumerate(raw_scores[0])])
            except Exception:
                scores_for_logging_str = "[Scores N/A for this model]"

        colored_selection = self._colorize(selected_arm_key_log, 'BLUE')
        self._log(INFO, f"CB Lib: Selected ArmIdx '{selected_arm_index}' (Key: {colored_selection}). Scores (expected): [{scores_for_logging_str}]")
        
        self.last_selected_arm_index = selected_arm_index
        self.last_context_vector = current_context_vector

        tdp_delta_w = self.bandit_actions.get(selected_arm_key_log, 0.0)
        base_tdp_for_bandit_decision = self.optimizer_target_tdp_w 
        proposed_next_tdp_by_bandit = base_tdp_for_bandit_decision + tdp_delta_w
        
        self._log(INFO, f"CB Lib Action: ArmKey='{selected_arm_key_log}', Delta={tdp_delta_w:.1f}W. "
                        f"Base TDP (optimizer target): {base_tdp_for_bandit_decision:.1f}W. "
                        f"Proposed TDP: {proposed_next_tdp_by_bandit:.1f}W.")

        actual_set_tdp, requested_tdp = self._set_tdp_limit_w(proposed_next_tdp_by_bandit, context=f"Optimizer CB Lib (Arm: {selected_arm_key_log})")
        self.optimizer_target_tdp_w = actual_set_tdp
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
                                max_r_val = int(f_max_r.read().strip())
                                if max_r_val > 0: max_r = max_r_val
                        except Exception: pass 
                        de += max_r
                    pwr_w = (de / 1e6) / dt 
                    if 0 <= pwr_w < 5000: ok = True
                    else:
                        self._log(DEBUG_ALL, f"Unrealistic PkgPwr calculated: {pwr_w:.1f}W. Resetting baseline.")
                        self.last_pkg_energy_uj, self.last_energy_read_time = current_e_uj, now
                        return 0.0, False
            
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
                    max_r_val = int(f_max_r.read().strip())
                    if max_r_val > 0: max_r = max_r_val
            except Exception as e: self._log(WARN, f"Could not read max_energy_range_uj: {e}"); pass
            delta_e += max_r
        
        self.energy_at_last_optimizer_interval_uj = current_e_uj 
        return delta_e

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                     indication_hdr_bytes: bytes, indication_msg_bytes: bytes):
            # --- CRITICAL: Add logging at the very top to confirm this function is being called ---
            self._log(DEBUG_KPM, f"KPM CB: Received indication from AgentID: {e2_agent_id}, SubID: {subscription_id}")
    
            if not self.e2sm_kpm:
                self._log(WARN, f"KPM from {e2_agent_id}, but e2sm_kpm module is unavailable. Cannot process.")
                return
        
            try:
                # --- START: NEW DEBUGGING BLOCK ---
                # Let's see the raw data before we try to parse it.
                # This will show us exactly what the gNB is sending.
                # try:
                #     # Assuming the library can return a raw dictionary representation
                #     raw_meas_data_for_log = self.e2sm_kpm.get_meas_data_dict(indication_msg_bytes)
                #     self._log(DEBUG_ALL, f"KPM PAYLOAD from {e2_agent_id}: {json.dumps(raw_meas_data_for_log)}")
                # except Exception as e_log:
                #     self._log(WARN, f"Could not log raw KPM payload: {e_log}")
                kpm_meas_data = self.e2sm_kpm.extract_meas_data(indication_msg_bytes)
        
                if not kpm_meas_data: 
                    kpm_hdr_info = self.e2sm_kpm.extract_hdr_info(indication_hdr_bytes)
                    self._log(WARN, f"KPM CB: KPM data from {e2_agent_id} is empty or failed to parse. HDR: {kpm_hdr_info}")
                    # Still treat this as a "heartbeat" report from the DU
                    with self.kpm_data_lock:
                        self.reports_received_this_interval[e2_agent_id] += 1
                        self.last_report_time_per_du[e2_agent_id] = time.monotonic()
                    return
                
                ue_meas_data_map = kpm_meas_data.get("ueMeasData", {})
                if not isinstance(ue_meas_data_map, dict):
                    self._log(WARN, f"KPM CB: Invalid 'ueMeasData' format from {e2_agent_id}. Data: {kpm_meas_data}")
                    return
                
                gNB_data_this_report = {'dl_bits': 0.0, 'ul_bits': 0.0}
    
                for ue_id_str, per_ue_measurements in ue_meas_data_map.items():
                    global_ue_id = f"{e2_agent_id}_{ue_id_str}" 
                    ue_dl_bits_this_ue, ue_ul_bits_this_ue = 0.0, 0.0
                    ue_metrics = per_ue_measurements.get("measData", {})
                    if not isinstance(ue_metrics, dict): continue 
                    
                    for metric_name, value_list in ue_metrics.items():
                        if isinstance(value_list, list) and value_list:
                            value = sum(val for val in value_list if isinstance(val, (int, float)))
                            # Ensure metric names match your config.yaml
                            if metric_name == 'DRB.RlcSduTransmittedVolumeDL':
                                ue_dl_bits_this_ue = float(value) * 1000.0 
                                gNB_data_this_report['dl_bits'] += ue_dl_bits_this_ue
                            elif metric_name == 'DRB.RlcSduTransmittedVolumeUL':
                                ue_ul_bits_this_ue = float(value) * 1000.0 
                                gNB_data_this_report['ul_bits'] += ue_ul_bits_this_ue
    
                    with self.kpm_data_lock:
                        # Using defaultdict here is fine.
                        self.current_interval_per_ue_data[global_ue_id]['total_bits'] += (ue_dl_bits_this_ue + ue_ul_bits_this_ue)
    
                # --- This block now handles all data accumulation and counter increments ---
                with self.kpm_data_lock:
                    # Use defaultdict for clean accumulation
                    acc = self.accumulated_kpm_metrics[e2_agent_id]
                    acc['bits_sum_dl'] += gNB_data_this_report['dl_bits']
                    acc['bits_sum_ul'] += gNB_data_this_report['ul_bits']
                    acc['num_reports'] += 1
    
                    # This is the logic that was likely failing or not being reached.
                    # Now it's guaranteed to run if the function is called.
                    if e2_agent_id in self.expected_du_ids:
                        self.reports_received_this_interval[e2_agent_id] += 1
                        self.last_report_time_per_du[e2_agent_id] = time.monotonic()
                        self._log(DEBUG_KPM, f"KPM CB: Incremented report count for DU '{e2_agent_id}' to {self.reports_received_this_interval[e2_agent_id]}. Total reports this interval: {len(self.reports_received_this_interval)}")
                    else:
                        self._log(WARN, f"KPM CB: Received report from unexpected DU '{e2_agent_id}'. Ignoring for optimizer count.")
    
            except Exception as e:
                self._log(ERROR, f"KPM CB: CRITICAL ERROR processing KPM from {e2_agent_id}: {e}")
                import traceback
                self._log(ERROR, traceback.format_exc())

    def _get_and_reset_kpm_data_for_cycle(self, interval_duration_s: float) -> Tuple[Dict[str, Any], int]:
        """
        Atomically gets a snapshot of all accumulated KPM data for the optimizer cycle,
        calculates the active UE count, and then resets all KPM accumulators.
        This centralized function prevents race conditions between different data structures.
        
        Returns:
            Tuple[Dict[str, Any], int]: A tuple containing:
                - The snapshot of summed KPM metrics.
                - The number of active UEs in the interval.
        """
        with self.kpm_data_lock:
            # 1. Take a snapshot of the primary accumulator
            kpm_metrics_snapshot = dict(self.accumulated_kpm_metrics)
            
            # 2. Calculate active UEs from the per-UE data before clearing it
            active_ue_count = 0
            threshold_bits = self.active_ue_throughput_threshold_mbps * 1e6 * interval_duration_s
            for ue_data in self.current_interval_per_ue_data.values():
                if ue_data.get('total_bits', 0.0) >= threshold_bits:
                    active_ue_count += 1
            
            # 3. Atomically reset all data structures for the next interval
            self.accumulated_kpm_metrics.clear()
            self.current_interval_per_ue_data.clear()
            # Note: We do NOT reset self.reports_received_this_interval here.
            # That is handled by the main loop's trigger logic.

            self._log(DEBUG_ALL, f"Retrieved KPM data for cycle. Metrics Snapshot: {kpm_metrics_snapshot}, Active UEs: {active_ue_count}")
            
        return kpm_metrics_snapshot, active_ue_count

    def _setup_kpm_subscriptions(self):
        self._log(INFO, "--- Setting up KPM Style 4 Subscriptions (Per-UE Metrics) ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable. Cannot subscribe."); return
        
        nodes = list(self.gnb_ids_map.values()) 
        if not nodes: self._log(WARN, "No gNB IDs configured for KPM subscriptions."); return

        kpm_config = self.config.get('kpm_subscriptions', {})
        style4_metrics = kpm_config.get('style4_metrics_per_ue', ['DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL'])
        style4_report_p_ms = int(kpm_config.get('style4_report_period_ms', 1000))
        style4_gran_p_ms = int(kpm_config.get('style4_granularity_period_ms', style4_report_p_ms))
        matching_ue_conds_config = kpm_config.get('style4_matching_ue_conditions', 
                                                  [{'testCondInfo': {'testType': ('ul-rSRP', 'true'), 'testExpr': 'lessthan', 'testValue': ('valueInt', 10000)}}])
            
        self._log(INFO, f"KPM Style 4: MetricsPerUE: {style4_metrics}, ReportPeriod={style4_report_p_ms}ms, Granularity={style4_gran_p_ms}ms")
        
        successes = 0
        for node_id_str in nodes:
            if self.dry_run:
                self._log(INFO, f"[DRY RUN] Would subscribe KPM Style 4 to Node {node_id_str}"); successes+=1; continue
            try:
                self._log(INFO, f"Subscribing KPM Style 4: Node {node_id_str}")
                self.e2sm_kpm.subscribe_report_service_style_4(
                    node_id_str, style4_report_p_ms, matching_ue_conds_config, 
                    style4_metrics, style4_gran_p_ms, self._kpm_indication_callback)
                with self.kpm_data_lock: self.accumulated_kpm_metrics[node_id_str] # Ensure key exists
                successes += 1
            except Exception as e: self._log(ERROR, f"KPM Style 4 subscription failed for {node_id_str}: {e}"); import traceback; traceback.print_exc()
        
        if successes > 0: self._log(INFO, f"--- KPM Style 4 Subscriptions: {successes} successful attempts for {len(nodes)} nodes. ---")
        elif nodes: self._log(WARN, "No KPM Style 4 subscriptions successfully initiated.")
        else: self._log(INFO, "--- No KPM nodes to subscribe to. ---")


    def _perform_optimizer_cycle(self, interval_duration_s: float):
        """Contains all logic for a single optimizer decision cycle."""
        # 1. --- Gather Current State ---
        if self.ru_timing_core_indices:
            self._update_ru_core_msr_data()
            current_ru_cpu_usage_control_val = self._get_control_ru_timing_cpu_usage()
        else:
            current_ru_cpu_usage_control_val = 0.0

        pid_fired_this_interval = self.pid_triggered_since_last_decision
        self.current_tdp_w = self._read_current_tdp_limit_w()

        # --- Centralized KPM Data Retrieval ---
        kpm_summed_data, current_num_active_ues = self._get_and_reset_kpm_data_for_cycle(interval_duration_s)
        self.current_num_active_ues_for_log = current_num_active_ues
        
        interval_energy_uj = self._get_interval_energy_uj_for_optimizer()

        # --- Data Aggregation & Core Metric Calculation ---
        total_bits_optimizer_interval = sum(d.get('bits_sum_dl', 0.0) + d.get('bits_sum_ul', 0.0) for d in kpm_summed_data.values())
        self._log(DEBUG_ALL, f"Optimizer Cycle: Total bits for interval = {total_bits_optimizer_interval}")
        avg_power_w_interval = (interval_energy_uj / 1e6) / interval_duration_s if interval_energy_uj and interval_duration_s > 0 else 0.0
        current_raw_efficiency = (total_bits_optimizer_interval / interval_energy_uj) if interval_energy_uj and interval_energy_uj > 1e-3 else 0.0
        
        dynamic_baseline = 1e-9
        if len(self.stable_efficiency_history) >= self.min_baseline_samples:
            dynamic_baseline = np.percentile(self.stable_efficiency_history, self.baseline_percentile)
        elif self.stable_efficiency_history:
            dynamic_baseline = np.mean(self.stable_efficiency_history)
        
        current_normalized_efficiency = np.clip(current_raw_efficiency / max(dynamic_baseline, 1e-9), 0.0, 1.0)
        
        # --- Stability Checks ---
        is_workload_stable = True
        significant_throughput_change = False
        if self.total_bits_from_previous_optimizer_interval is not None and self.total_bits_from_previous_optimizer_interval > 1e6:
            workload_ratio = total_bits_optimizer_interval / self.total_bits_from_previous_optimizer_interval
            if workload_ratio < self.workload_drop_threshold:
                self._log(INFO, f"Workload drop detected. Resetting efficiency baseline.")
                is_workload_stable = False
                self.stable_efficiency_history.clear()
            
            if abs(total_bits_optimizer_interval - self.total_bits_from_previous_optimizer_interval) / self.total_bits_from_previous_optimizer_interval > self.throughput_change_threshold_for_discard:
                significant_throughput_change = True
                self._log(WARN, f"Optimizer: Sig. performance score change detected. Update might be skipped.")

        is_cpu_stressed = current_ru_cpu_usage_control_val > (self.target_ru_cpu_usage * 0.99)
        if pid_fired_this_interval or is_cpu_stressed:
            is_workload_stable = False
            if pid_fired_this_interval: self._log(WARN, "Instability: PID triggered. Suppressing efficiency history update.")
            if is_cpu_stressed: self._log(WARN, f"Instability: CPU stressed ({current_ru_cpu_usage_control_val:.2f}%). Suppressing efficiency history update.")

        # --- Reward Calculation ---
        action_delta_w = 0.0
        chosen_arm_key = "N/A"
        if self.last_selected_arm_index is not None:
            chosen_arm_key = self.arm_keys_ordered[self.last_selected_arm_index]
            action_delta_w = self.bandit_actions.get(chosen_arm_key, 0.0)

        reward_for_bandit = 0.0
        is_active_ue_present = current_num_active_ues > 0

        # +++ START: NEW HEADROOM PENALTY/REWARD LOGIC +++
        # This logic runs before the main reward calculation to add a headroom-based component.
        headroom_reward_component = 0.0
        # We can only calculate this if it's not the very first run.
        if self.last_action_actual_tdp is not None and self.last_interval_avg_power_w > 0:
            previous_tdp_limit = self.last_action_actual_tdp
            # Gap from the PREVIOUS state
            previous_headroom_w = max(0, previous_tdp_limit - self.last_interval_avg_power_w)
            # Gap from the CURRENT state
            current_headroom_w = max(0, self.optimizer_target_tdp_w - avg_power_w_interval)
            
            # The change in wasted power. A negative value is good (headroom was reduced).
            headroom_delta = current_headroom_w - previous_headroom_w
    
            # If we increased the limit for no reason, headroom_delta will be positive. Punish it.
            # If we decreased the limit successfully, headroom_delta will be negative. Reward it.
            # This provides a direct incentive to close the gap.
            # The scaling factor (0.05) makes it a gentle but firm nudge.
            headroom_reward_component = -headroom_delta * 0.2
            
            # We only apply this logic if the PID didn't fire, as the PID is a more critical event.
            if not pid_fired_this_interval:
                self._log(INFO, f"CB REWARD MOD: Headroom Δ={headroom_delta:+.1f}W. Applying reward component: {headroom_reward_component:+.3f}")
                reward_for_bandit += headroom_reward_component
        # +++ END: NEW HEADROOM PENALTY/REWARD LOGIC +++
        
        # HIERARCHY 1: PID Trigger
        if pid_fired_this_interval:
            self.pid_triggered_since_last_decision = False
            if action_delta_w <= 0:
                reward_for_bandit = -0.4
                self._log(WARN, f"CB REWARD: PID TRIGGER OVERRIDE. Action '{chosen_arm_key}' was wrong. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', 'RED')}")
            else:
                reward_for_bandit = 0.4
                if is_active_ue_present:
                    reward_for_bandit = max(current_normalized_efficiency - self.last_normalized_efficiency, 0.4)
                self._log(INFO, f"CB REWARD: PID TRIGGER OVERRIDE. Action '{chosen_arm_key}' was correct. Applying strong incentive. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', 'GREEN')}")
        
        # HIERARCHY 2: Normal Operation (Active or Idle)
        elif is_active_ue_present:
            # --- HYBRID REWARD V5: CONDITIONAL COMPOSITION ---
            relative_reward_component = 0.0
            absolute_reward_component = 0.0

            if self.last_raw_efficiency > 1e-9:
                raw_efficiency_change_pct = (current_raw_efficiency - self.last_raw_efficiency) / self.last_raw_efficiency
                reward_shaping_factor = 5.0
                relative_reward_component = np.tanh(raw_efficiency_change_pct * reward_shaping_factor)
            elif current_raw_efficiency > 1e-9:
                relative_reward_component = 0.5

            # The absolute component is only a bonus for GOOD behavior
            if relative_reward_component >= 0:
                absolute_reward_component = (current_normalized_efficiency ** 2)
                reward_for_bandit = (0.8 * relative_reward_component) + (0.2 * absolute_reward_component)
            else:
                # If efficiency got worse, the reward is ONLY the negative relative component. No bonus.
                reward_for_bandit = relative_reward_component

            # --- Detailed Logging ---
            reward_color = 'GREEN' if reward_for_bandit >= 0 else 'RED'
            colored_reward = self._colorize(f'{reward_for_bandit:+.3f}', reward_color)
            self._log(INFO, f"CB Reward (Active): RawEff: {self.last_raw_efficiency:.3f} -> {current_raw_efficiency:.3f} | NormEff: {current_normalized_efficiency:.2f}")
            self._log(INFO, f"                 -> Components: Relative={relative_reward_component:+.3f}, Absolute={absolute_reward_component:+.3f} => Final Reward: {colored_reward}")
            
            # Additive penalties still apply
            if is_cpu_stressed and action_delta_w <= 0:
                self._log(WARN, f"CB REWARD MOD: Stressed CPU at {current_ru_cpu_usage_control_val:.2f}%. Applying penalty of -0.3 to reward.")
                reward_for_bandit -= 0.3
        else: # Idle state logic
            holding_zone_width_w = 5.0
            if self.optimizer_target_tdp_w <= (self.tdp_min_w + holding_zone_width_w):
                if action_delta_w == 0: reward_for_bandit = 0.2
                elif action_delta_w < 0: reward_for_bandit = 0.05
                else: reward_for_bandit = -0.4
            else:
                norm_tdp_excursion = (self.optimizer_target_tdp_w - self.tdp_min_w) / (self.tdp_max_w - self.tdp_min_w) if (self.tdp_max_w > self.tdp_min_w) else 0
                if action_delta_w < 0: reward_for_bandit = 0.2 + 0.3 * norm_tdp_excursion
                elif action_delta_w == 0: reward_for_bandit = 0.0
                else: reward_for_bandit = -0.3 - 0.2 * norm_tdp_excursion
            reward_color = 'GREEN' if reward_for_bandit >= 0 else 'RED'
            self._log(INFO, f"CB Reward (True Idle): Action '{chosen_arm_key}'. Final Reward={self._colorize(f'{reward_for_bandit:.3f}', reward_color)}")

        # Penalty for Ineffective (Clipped) Actions
        if (self.last_action_requested_tdp is not None and 
            self.last_action_actual_tdp is not None and 
            action_delta_w > 0 and 
            abs(self.last_action_requested_tdp - self.last_action_actual_tdp) > 1e-3):
            self._log(WARN, f"CB REWARD: Ineffective Action, -0.25 penalty added. Requested {self.last_action_requested_tdp:.1f}W but clipped to {self.last_action_actual_tdp:.1f}W.")
            reward_for_bandit -= 0.25
        
        # --- Finalize and Run Bandit ---
        reward_for_bandit = np.clip(reward_for_bandit, -1.0, 1.0)
        self.most_recent_calculated_reward_for_log = reward_for_bandit
        
        current_context_vec = self._get_current_context_vector(
            current_num_active_ues, current_ru_cpu_usage_control_val, self.current_tdp_w, current_normalized_efficiency
        )

        self._run_contextual_bandit_optimizer_step(reward_for_bandit, current_context_vec, significant_throughput_change)
        
        # --- Update State for Next Cycle ---
        if is_workload_stable: self.stable_efficiency_history.append(current_raw_efficiency)
        self.last_raw_efficiency = current_raw_efficiency
        self.last_normalized_efficiency = current_normalized_efficiency
        self.total_bits_from_previous_optimizer_interval = total_bits_optimizer_interval
        self.last_interval_avg_power_w = avg_power_w_interval
    @xAppBase.start_function
    def run_power_management_xapp(self):
        if os.geteuid() != 0 and not self.dry_run:
            self._log(ERROR, "Must be root for live run. Exiting."); sys.exit(1)
        
        try:
            self.current_tdp_w = self._read_current_tdp_limit_w()
            self.optimizer_target_tdp_w = self.current_tdp_w
            self._log(INFO, f"Initial current TDP: {self.current_tdp_w:.1f}W. Optimizer base target set to this.")

            if self.ru_timing_core_indices:
                self._log(INFO, "Priming MSR data..."); self._update_ru_core_msr_data(); time.sleep(0.1); self._update_ru_core_msr_data(); self._log(INFO, "MSR primed.")
            
            self.energy_at_last_optimizer_interval_uj = self._read_current_energy_uj()
            self.last_pkg_energy_uj = self._read_current_energy_uj() 
            self.last_energy_read_time = time.monotonic()
            
            self._setup_intel_sst()
            self._setup_kpm_subscriptions() 

            now = time.monotonic()
            self.last_ru_pid_run_time = now
            self.last_optimizer_run_time = now 
            self.last_stats_print_time = now
            self.total_bits_from_previous_optimizer_interval = 0.0

            self._log(INFO, f"\n--- Starting Monitoring & Control Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            self._log(INFO, f"RU PID Interval: {self.ru_timing_pid_interval_s}s | Target RU CPU: {self.target_ru_cpu_usage if self.ru_timing_core_indices else 'N/A'}%")
            self._log(INFO, f"CB (BootstrapTS) Actions: {self.bandit_actions}")

            while self.running: 
                loop_start_time = time.monotonic()
                if self.ru_timing_core_indices: self._update_ru_core_msr_data()
                self.current_tdp_w = self._read_current_tdp_limit_w()

                if loop_start_time - self.last_ru_pid_run_time >= self.ru_timing_pid_interval_s:
                    if self.ru_timing_core_indices:
                        current_ru_cpu_usage = self._get_control_ru_timing_cpu_usage()
                        self._run_ru_timing_pid_step(current_ru_cpu_usage)
                    self.last_ru_pid_run_time = loop_start_time
                
                # --- Adaptive Optimizer Trigger Logic ---
                now = time.monotonic()
                should_run_optimizer = False
                reason = ""

                # Condition 1: Global safety timeout. This always comes first.
                if now - self.last_optimizer_run_time > self.optimizer_max_interval_s:
                    should_run_optimizer = True
                    reason = f"Timeout of {self.optimizer_max_interval_s:.1f}s reached."
                else:
                    # Condition 2: Report count met for all *active* DUs.
                    with self.kpm_data_lock:
                        # Define the set of DUs we currently consider active and should wait for.
                        active_du_set = {
                            du_id for du_id in self.expected_du_ids
                            if (now - self.last_report_time_per_du.get(du_id, 0.0)) < self.du_activity_timeout_s
                        }
                        
                        # --- BUG FIX: Reset report count for DUs that have become inactive ---
                        inactive_dus = self.expected_du_ids - active_du_set
                        for du_id in inactive_dus:
                            if self.reports_received_this_interval.get(du_id, 0) > 0:
                                self._log(DEBUG_KPM, f"DU '{du_id}' became inactive. Resetting its report count from {self.reports_received_this_interval[du_id]} to 0.")
                                self.reports_received_this_interval[du_id] = 0
                        # --- END BUG FIX ---

                        # Only proceed if there is at least one active DU.
                        if active_du_set:
                            # Check if all DUs in the active set have met their report quota.
                            all_active_dus_ready = all(
                                self.reports_received_this_interval.get(du_id, 0) >= self.optimizer_reports_per_du
                                for du_id in active_du_set
                            )

                            if all_active_dus_ready:
                                should_run_optimizer = True
                                counts_str = ", ".join([f"{du.split('_')[-1]}:{self.reports_received_this_interval.get(du, 0)}" for du in active_du_set])
                                reason = f"Active DUs met report count ({counts_str})"
                
                if should_run_optimizer:
                    # This part remains the same as before.
                    self._log(INFO, f"Triggering optimizer: {reason}")
                    actual_interval_s = now - self.last_optimizer_run_time
                    if actual_interval_s < 0.1:
                        self._log(WARN, f"Optimizer triggered with very short duration ({actual_interval_s:.3f}s). Skipping cycle.")
                    else:
                        self._perform_optimizer_cycle(actual_interval_s)
                    
                    self.last_optimizer_run_time = now
                    with self.kpm_data_lock:
                        self.reports_received_this_interval.clear()
                # --- End Trigger Logic ---

                if loop_start_time - self.last_stats_print_time >= self.stats_print_interval_s:
                    pkg_pwr_w, pkg_pwr_ok = self._get_pkg_power_w()
                    ru_usage_str = "N/A"
                    if self.ru_timing_core_indices:
                        ru_usage_avg = self._get_control_ru_timing_cpu_usage()
                        ru_usage_str = f"(AvgMax:{ru_usage_avg:>6.2f}%)"
                    
                    last_arm_key_str = self.arm_keys_ordered[self.last_selected_arm_index] if self.last_selected_arm_index is not None else "N/A"
                    bandit_log = f"CB(LastArm:{last_arm_key_str})"
                    
                    log_parts = [f"RU:{ru_usage_str}", f"TDP_Act:{self.current_tdp_w:>5.1f}W", f"PkgPwr:{pkg_pwr_w if pkg_pwr_ok else 'N/A':>5.1f}W", bandit_log]
                    if self.most_recent_calculated_reward_for_log is not None:
                         log_parts.append(f"LastReward:{self.most_recent_calculated_reward_for_log:.3f}")
                    log_parts.append(f"ActiveUEs:{self.current_num_active_ues_for_log}")

                    self._log(INFO, " | ".join(log_parts)); 
                    self.last_stats_print_time = loop_start_time
                
                loop_duration = time.monotonic() - loop_start_time
                time.sleep(max(0, self.main_loop_sleep_s - loop_duration))

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
        manager.run_power_management_xapp()
    except (RuntimeError, SystemExit) as e: 
        code = e.code if isinstance(e, SystemExit) and e.code is not None else 1
        msg = f"Application terminated with exit code {code}: {e}"
        print(msg, file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR if isinstance(e, RuntimeError) else INFO, msg)
        sys.exit(code) 
    except Exception as e:
        msg = f"An unexpected error at top level: {e}"
        print(msg, file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"TOP LEVEL UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        msg = "Application finished."
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, msg)
        else: print(f"INFO: {msg} (logger may not be available).")
