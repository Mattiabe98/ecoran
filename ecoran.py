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

try:
    from lib.xAppBase import xAppBase # Assuming this is in PYTHONPATH or ./lib/
except ImportError:
    # Attempt relative import if direct fails (e.g. when run as a script)
    try:
        # This is a common way to handle lib imports when the script is in the same dir or one level up
        sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Adjust if lib is elsewhere
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

# --- Context Feature Indices (Example) ---
CTX_IDX_BIAS = 0
CTX_IDX_TOTAL_BITS_NORM = 1 # Bits per second normalized
CTX_IDX_NUM_UES_NORM = 2
CTX_IDX_NUM_ACTIVE_DUS_NORM = 3
CTX_IDX_RU_CPU_NORM = 4
CTX_IDX_CURRENT_TDP_NORM = 5
# CONTEXT_DIMENSION will be set from config, ensure it matches the number of features used + bias

def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            val_bytes = f.read(8)
            return struct.unpack('<Q', val_bytes)[0] if len(val_bytes) == 8 else None
    except FileNotFoundError: return None
    except PermissionError: return None
    except OSError as e:
        # Suppress "No such device" or "Permission denied" during MSR read if core is offline/hotplugged
        # or if permissions are insufficient for a specific core momentarily.
        if e.errno != 2 and e.errno != 13 and e.errno != 19: # errno 19 is ENODEV (No such device)
             print(f"W: OSError reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"E: Unexpected error reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None

class CoreMSRData:
    def __init__(self, core_id: int):
        self.core_id, self.mperf, self.tsc, self.busy_percent = core_id, None, None, 0.0

class LinUCBContextualBandit:
    def __init__(self, arm_keys: List[str], context_dim: int, alpha: float, lambda_reg: float = 0.1):
        self.arm_keys = arm_keys
        self.n_arms = len(arm_keys)
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        self.A = { arm: self.lambda_reg * np.identity(self.context_dim) for arm in self.arm_keys }
        self.b = { arm: np.zeros(self.context_dim) for arm in self.arm_keys }
        self.arm_counts = {arm: 0 for arm in self.arm_keys}
        self.arm_sum_rewards = {arm: 0.0 for arm in self.arm_keys}
        self.logger = logging.getLogger("EcoRANPowerManager.LinUCB")

    def select_arm(self, context_vector: np.array) -> str:
        if not self.arm_keys: # Should not happen if initialized correctly
            self.logger.error("LinUCB: No arms defined!")
            return "hold" # A safe default if possible

        if context_vector.shape[0] != self.context_dim:
            self.logger.error(f"LinUCB: Context vector dim {context_vector.shape[0]} != expected {self.context_dim}. Using random arm.")
            return random.choice(self.arm_keys)

        max_ucb = -float('inf')
        selected_arm_key = self.arm_keys[0] # Default initialization
        ucb_scores_debug = {}

        for arm_key in self.arm_keys:
            try:
                A_inv = np.linalg.inv(self.A[arm_key])
            except np.linalg.LinAlgError:
                self.logger.warning(f"LinUCB: Matrix A for arm {arm_key} is singular. Using identity fallback.")
                A_inv = np.identity(self.context_dim) / self.lambda_reg # Avoid division by zero if lambda_reg is 0
                if self.lambda_reg == 0: A_inv = np.identity(self.context_dim) * 1e6 # Large variance
            
            theta_hat = A_inv @ self.b[arm_key]
            exploitation_term = context_vector @ theta_hat
            
            # Ensure exploration term calculation is robust
            try:
                exploration_radicand = context_vector.T @ A_inv @ context_vector
                if exploration_radicand < 0: # Should not happen with PSD matrices A_inv if A is PSD
                    self.logger.warning(f"LinUCB: Negative radicand {exploration_radicand:.3e} for exploration term in arm {arm_key}. Using 0.")
                    exploration_radicand = 0.0
                exploration_term = self.alpha * np.sqrt(exploration_radicand)
            except Exception as e:
                self.logger.error(f"LinUCB: Error calculating exploration term for arm {arm_key}: {e}. Setting to 0.")
                exploration_term = 0.0

            current_ucb = exploitation_term + exploration_term
            ucb_scores_debug[arm_key] = current_ucb

            if current_ucb > max_ucb:
                max_ucb = current_ucb
                selected_arm_key = arm_key
        
        # Sort scores for logging to easily see the top ones
        sorted_scores_str = ", ".join([f"{k}:{v:.3f}" for k,v in sorted(ucb_scores_debug.items(), key=lambda item: item[1], reverse=True)[:5]]) # Log top 5
        self.logger.info(f"LinUCB: Selected Arm '{selected_arm_key}'. Top Scores: {sorted_scores_str}")
        return selected_arm_key

    def update_arm(self, arm_key: str, context_vector: np.array, reward: float):
        if arm_key not in self.A:
            self.logger.error(f"LinUCB: Attempted to update non-existent arm: {arm_key}")
            return
        if context_vector.shape[0] != self.context_dim:
            self.logger.error(f"LinUCB: Context vector dim mismatch during update for arm {arm_key}.")
            return

        self.A[arm_key] += np.outer(context_vector, context_vector)
        self.b[arm_key] += reward * context_vector
        self.arm_counts[arm_key] += 1
        self.arm_sum_rewards[arm_key] += reward
        avg_reward = self.arm_sum_rewards[arm_key] / self.arm_counts[arm_key]
        self.logger.info(f"LinUCB: Updated Arm '{arm_key}' with reward {reward:.3f}. "
                         f"Count: {self.arm_counts[arm_key]}, AvgReward: {avg_reward:.3f}")

    def get_best_arm_empirically(self) -> Tuple[Optional[str], float]:
        best_arm = None; max_empirical_mean = -float('inf')
        if not self.arm_keys: return None, -float('inf')

        for arm_key in self.arm_keys:
            if self.arm_counts[arm_key] > 0:
                empirical_mean = self.arm_sum_rewards[arm_key] / self.arm_counts[arm_key]
                if empirical_mean > max_empirical_mean:
                    max_empirical_mean = empirical_mean
                    best_arm = arm_key
        return best_arm, max_empirical_mean

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

        self.main_loop_sleep_s = float(self.config.get('main_loop_sleep_s', 0.1))
        self.ru_timing_pid_interval_s = float(self.config.get('ru_timing_pid_interval_s', 1.0))
        self.optimizer_decision_interval_s = float(self.config.get('optimizer_decision_interval_s', 10.0))
        self.stats_print_interval_s = float(self.config.get('stats_print_interval_s', self.optimizer_decision_interval_s))

        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 90))
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 170))
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 99.5))
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))

        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.0005))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1.0))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 3.0))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 1.5))

        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        self.dry_run = bool(self.config.get('dry_run', False))
        self.current_tdp_w = float(self.tdp_min_w)
        self.last_pkg_energy_uj: Optional[int] = None
        self.last_energy_read_time: Optional[float] = None
        self.energy_at_last_optimizer_interval_uj: Optional[int] = None

        self.max_ru_timing_usage_history: List[float] = []
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}

        self.kpm_ran_func_id = kpm_ran_func_id
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None: self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else: self._log(WARN, "xAppBase.e2sm_kpm module unavailable."); self.e2sm_kpm = None

        self.gnb_ids_map = self.config.get('gnb_ids', {})
        self.gnb_id_to_du_name_map = {v: k for k, v in self.gnb_ids_map.items()}
        
        clos_association_config = self.config.get('clos_association', {})
        self.clos_to_du_names_map: Dict[int, List[str]] = {}
        ran_components_in_config = self.config.get('ran_cores', {}).keys()
        for cid_key, comps_list in clos_association_config.items():
            cid = int(cid_key)
            if isinstance(comps_list, list): 
                self.clos_to_du_names_map[cid] = [c for c in comps_list if c in ran_components_in_config and c.startswith('du')]
            else: self._log(WARN, f"Components for CLOS {cid} ('{comps_list}') not a list. Skipping.")
        

        self.kpm_data_lock = threading.Lock()
        self.accumulated_kpm_metrics: Dict[str, Dict[str, Any]] = {}
        self.current_interval_ue_ids: Set[str] = set()

        cb_config = self.config.get('contextual_bandit', {})
        bandit_actions_w_str = cb_config.get('actions_tdp_delta_w', {"dec_10": -10.0, "dec_5": -5.0, "hold": 0.0, "inc_5": 5.0, "inc_10": 10.0})
        self.bandit_actions: Dict[str, float] = {k: float(v) for k, v in bandit_actions_w_str.items()}
        if "hold" not in self.bandit_actions: self.bandit_actions["hold"] = 0.0

        self.context_dimension = int(cb_config.get('context_dimension', 6))
        self.linucb_alpha = float(cb_config.get('alpha', 1.0))
        self.linucb_lambda_reg = float(cb_config.get('lambda_reg', 0.1))

        self.contextual_bandit = LinUCBContextualBandit(
            arm_keys=list(self.bandit_actions.keys()),
            context_dim=self.context_dimension,
            alpha=self.linucb_alpha,
            lambda_reg=self.linucb_lambda_reg
        )
        self.optimizer_target_tdp_w = self.current_tdp_w
        self.last_selected_bandit_arm: Optional[str] = None
        self.last_context_vector: Optional[np.array] = None
        self.total_bits_from_previous_optimizer_interval: Optional[float] = None
        self.throughput_change_threshold_for_discard = float(cb_config.get('throughput_change_threshold_for_discard', 1.0))

        self.norm_params = cb_config.get('normalization_parameters', {}) # Loaded from config
        # Ensure default TDP normalization params match actual TDP range if not specified
        if 'current_tdp' not in self.norm_params:
            self.norm_params['current_tdp'] = {'min': float(self.tdp_min_w), 'max': float(self.tdp_max_w)}
        if 'num_active_dus' not in self.norm_params:
             self.norm_params['num_active_dus'] =  {'min': 0, 'max': float(len(self.gnb_ids_map) or 1.0)}


        self.last_ru_pid_run_time: float = 0.0
        self.last_optimizer_run_time: float = 0.0
        self.last_stats_print_time: float = 0.0
        self.most_recent_calculated_efficiency_for_log: Optional[float] = None
        self.current_num_ues_for_log: int = 0

        self._validate_config()
        if self.dry_run: self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _normalize_feature(self, value: float, feature_key: str) -> float:
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

    def _get_current_context_vector(self, current_total_bits_interval: float, current_num_ues: int,
                                   current_num_active_dus: int, current_ru_cpu_avg: float,
                                   current_actual_tdp: float) -> np.array:
        
        interval_s = self.optimizer_decision_interval_s
        if interval_s <=0: interval_s = 1.0 

        bits_per_second = current_total_bits_interval / interval_s if interval_s > 0 else 0.0

        features = np.zeros(self.context_dimension)
        current_feature_index = 0

        def add_feature(value, key):
            nonlocal current_feature_index
            if current_feature_index < self.context_dimension:
                features[current_feature_index] = self._normalize_feature(value, key)
                current_feature_index += 1
            else:
                self._log(WARN, f"Context vector full, cannot add feature {key}. Dim: {self.context_dimension}")

        # Order must be consistent and match definition (e.g., CTX_IDX constants if used)
        add_feature(1.0, 'bias') # Bias term - normalization params for bias could be min:1, max:1
        add_feature(bits_per_second, 'total_bits_per_second')
        add_feature(float(current_num_ues), 'num_ues')
        add_feature(float(current_num_active_dus), 'num_active_dus')
        add_feature(current_ru_cpu_avg, 'ru_cpu_usage')
        
        # Only add TDP if context_dimension allows
        if 'current_tdp' in self.norm_params and current_feature_index < self.context_dimension:
             add_feature(current_actual_tdp, 'current_tdp')
        
        if current_feature_index != self.context_dimension:
            self._log(ERROR, f"Context vector final length {current_feature_index} != configured dimension {self.context_dimension}!")
            # Pad with 0.5 or handle error - this indicates a mismatch in feature definition vs. dimension
            while current_feature_index < self.context_dimension:
                features[current_feature_index] = 0.5 
                current_feature_index +=1
        return features

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
        
        if current_context_vector is None:
            self._log(WARN, "ContextualBandit: Current context vector is None. Cannot select arm. Holding TDP by choosing 'hold' arm.")
            selected_arm_key = "hold" 
            if "hold" not in self.bandit_actions:
                 selected_arm_key = random.choice(list(self.bandit_actions.keys()))
        else:
            selected_arm_key = self.contextual_bandit.select_arm(current_context_vector)
        
        self.last_selected_bandit_arm = selected_arm_key
        self.last_context_vector = current_context_vector 

        tdp_delta_w = self.bandit_actions.get(selected_arm_key, 0.0) 

        base_tdp_for_bandit_decision = self.optimizer_target_tdp_w
        proposed_next_tdp_by_bandit = base_tdp_for_bandit_decision + tdp_delta_w
        
        self._log(INFO, f"ContextualBandit: Arm='{selected_arm_key}', Delta={tdp_delta_w:.1f}W. "
                        f"Base TDP: {base_tdp_for_bandit_decision:.1f}W. "
                        f"Proposed TDP: {proposed_next_tdp_by_bandit:.1f}W.")
        if current_context_vector is not None:
             self._log(DEBUG_ALL, f"ContextualBandit: Context for decision: {['{:.2f}'.format(x) for x in current_context_vector]}")

        self._set_tdp_limit_w(proposed_next_tdp_by_bandit, context=f"Optimizer CB (Arm: {selected_arm_key})")
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

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes):
        self._log(DEBUG_KPM, f"KPM CB: Agent:{e2_agent_id}, Sub(E2EventInstanceID):{subscription_id}, Time:{time.monotonic():.3f}")
        if not self.e2sm_kpm: self._log(WARN, f"KPM from {e2_agent_id}, but e2sm_kpm unavailable."); return

        try:
            kpm_hdr_info = self.e2sm_kpm.extract_hdr_info(indication_hdr_bytes)
            kpm_meas_data = self.e2sm_kpm.extract_meas_data(indication_msg_bytes)

            if not kpm_meas_data: 
                self._log(WARN, f"KPM CB Style 4: Failed to extract KPM data from {e2_agent_id}. HDR: {kpm_hdr_info}"); return
            
            ue_meas_data_map = kpm_meas_data.get("ueMeasData", {})
            if not isinstance(ue_meas_data_map, dict):
                self._log(WARN, f"KPM CB Style 4: Invalid 'ueMeasData' format from {e2_agent_id}. Data: {kpm_meas_data}")
                return
            
            gNB_total_dl_bits_this_report = 0.0
            gNB_total_ul_bits_this_report = 0.0
            ues_in_this_report_for_this_gnb = set()

            for ue_id_str, per_ue_measurements in ue_meas_data_map.items():
                global_ue_id = f"{e2_agent_id}_{ue_id_str}" 
                ues_in_this_report_for_this_gnb.add(global_ue_id)

                ue_metrics = per_ue_measurements.get("measData", {})
                if not isinstance(ue_metrics, dict):
                    self._log(WARN, f"KPM CB Style 4: Invalid 'measData' for UE {global_ue_id} from {e2_agent_id}.")
                    continue
                
                for metric_name, value_list in ue_metrics.items():
                    if isinstance(value_list, list) and value_list:
                        value = sum(value_list) 
                        try:
                            if metric_name == 'DRB.RlcSduTransmittedVolumeDL':
                                gNB_total_dl_bits_this_report += float(value) * 1000.0
                            elif metric_name == 'DRB.RlcSduTransmittedVolumeUL':
                                gNB_total_ul_bits_this_report += float(value) * 1000.0
                        except (ValueError, TypeError):
                             self._log(WARN, f"KPM CB Style 4: Metric '{metric_name}' for UE {global_ue_id} value '{value_list}' invalid.")
            
            with self.kpm_data_lock:
                if e2_agent_id not in self.accumulated_kpm_metrics:
                    self.accumulated_kpm_metrics[e2_agent_id] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'num_reports':0}
                acc_data = self.accumulated_kpm_metrics[e2_agent_id]
                acc_data['bits_sum_dl'] += gNB_total_dl_bits_this_report
                acc_data['bits_sum_ul'] += gNB_total_ul_bits_this_report
                acc_data['num_reports'] += 1
                
                self.current_interval_ue_ids.update(ues_in_this_report_for_this_gnb)
                
                self._log(DEBUG_KPM, f"KPM CB Style 4: {e2_agent_id}: Added DL_b={gNB_total_dl_bits_this_report:.0f}, UL_b={gNB_total_ul_bits_this_report:.0f}. "
                                     f"Reported {len(ues_in_this_report_for_this_gnb)} UEs. Total unique UEs for interval so far: {len(self.current_interval_ue_ids)}")

        except Exception as e: self._log(ERROR, f"Error processing KPM Style 4 indication from {e2_agent_id}: {e}"); import traceback; traceback.print_exc()


    def _get_and_reset_accumulated_kpm_metrics(self) -> Dict[str, Dict[str, Any]]:
        with self.kpm_data_lock:
            snap = {}
            for gnb_id, data in self.accumulated_kpm_metrics.items():
                snap[gnb_id] = {
                    'dl_bits': data.get('bits_sum_dl', 0.0),
                    'ul_bits': data.get('bits_sum_ul', 0.0),
                    'reports_in_interval': data.get('num_reports', 0)
                }
                data['bits_sum_dl'] = 0.0
                data['bits_sum_ul'] = 0.0
                data['num_reports'] = 0
        return snap

    def _get_and_reset_current_ue_count(self) -> int:
        with self.kpm_data_lock:
            count = len(self.current_interval_ue_ids)
            self.current_interval_ue_ids.clear() 
        return count

    def _setup_kpm_subscriptions(self): # Heavily modified for Style 4 only
        self._log(INFO, "--- Setting up KPM Style 4 Subscriptions (Per-UE Metrics) ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable. Cannot subscribe."); return
        
        nodes = list(self.gnb_ids_map.values()) 
        if not nodes: self._log(WARN, "No gNB IDs configured for KPM subscriptions."); return

        kpm_config = self.config.get('kpm_subscriptions', {})
        
        style4_metrics = kpm_config.get('style4_metrics_per_ue', [
            'DRB.RlcSduTransmittedVolumeDL', 
            'DRB.RlcSduTransmittedVolumeUL',
        ])
        style4_report_p_ms = int(kpm_config.get('style4_report_period_ms', 1000))
        style4_gran_p_ms = int(kpm_config.get('style4_granularity_period_ms', style4_report_p_ms))
        
        default_match_all_cond = [{'testCondInfo': {'testType': ('ul-rSRP', 'true'), 'testExpr': 'lessthan', 'testValue': ('valueInt', 1000)}}]
            
        self._log(INFO, f"KPM Style 4: MetricsPerUE: {style4_metrics}, ReportPeriod={style4_report_p_ms}ms, Granularity={style4_gran_p_ms}ms, Conditions: {default_match_all_cond}")
        
        successes = 0
        for node_id_str in nodes:
            # self._last_ric_sub_id_attempted = None # No longer strictly needed for style mapping if only one style
            # self._last_style_attempted = 4 # No longer strictly needed

            if self.dry_run:
                self._log(INFO, f"[DRY RUN] KPM Style 4 Sub: Node {node_id_str}")
                successes+=1; continue
            try:
                self._log(INFO, f"Subscribing KPM Style 4: Node {node_id_str}")
                
                # --- CORRECTED CALL using positional arguments ---
                self.e2sm_kpm.subscribe_report_service_style_4(
                    node_id_str,                # e2_node_id
                    style4_report_p_ms,         # report_period
                    default_match_all_cond,   # matchingUeConds
                    style4_metrics,             # metric_names (meas_names_per_ue)
                    style4_gran_p_ms,           # granul_period
                    self._kpm_indication_callback # subscription_callback
                )
                # --- END CORRECTED CALL ---
                
                with self.kpm_data_lock:
                    if node_id_str not in self.accumulated_kpm_metrics:
                        self.accumulated_kpm_metrics[node_id_str] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'num_reports':0}
                successes += 1
            except Exception as e: self._log(ERROR, f"KPM Style 4 subscription failed for {node_id_str}: {e}"); import traceback; traceback.print_exc()
        
        if successes > 0: self._log(INFO, f"--- KPM Style 4 Subscriptions: {successes} successful attempts for {len(nodes)} nodes. ---")
        elif nodes: self._log(WARN, "No KPM Style 4 subscriptions successfully initiated.")
        else: self._log(INFO, "--- No KPM nodes to subscribe to. ---")

    def _subscription_response_callback(self, name, path, data, ctype):
        # This method is inherited from xAppBase and called by its HTTP server
        # when the Subscription Manager sends a response to our subscription request.
        # The xAppBase.py provided earlier handles storing the mapping from RIC Subscription ID
        # to the E2EventInstanceID (which is used in RMR headers).
        # We just log it here for visibility.
        try:
            parsed_data = json.loads(data)
            subscription_id_ric = parsed_data.get('SubscriptionId') 
            subscription_instances = parsed_data.get('SubscriptionInstances', [])
            if subscription_instances:
                e2_event_instance_id = subscription_instances[0].get("E2EventInstanceId")
                self._log(INFO,f"Subscription Response CB: RIC_SubID={subscription_id_ric} maps to E2EventInstanceID={e2_event_instance_id}")
            else:
                self._log(WARN, f"Subscription Response CB: No SubscriptionInstances found for RIC_SubID={subscription_id_ric}. Data: {data}")

        except Exception as e:
            self._log(ERROR, f"Error in _subscription_response_callback processing data '{data}': {e}")
        
        # Create and return the HTTP response object expected by xAppBase
        response_payload_str = "{}" # Empty JSON object as payload
        return {'response': 'OK', 'status': 200, 'payload': response_payload_str, 'ctype': 'application/json', 'attachment': None, 'mode': 'plain'}


    @xAppBase.start_function
    def run_power_management_xapp(self):
        if os.geteuid() != 0 and not self.dry_run:
            self._log(ERROR, "Must be root for live run. Exiting."); sys.exit(1)
        
        try:
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
            self.total_bits_from_previous_optimizer_interval = 0.0

            self._log(INFO, f"\n--- Starting Monitoring & Control Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            self._log(INFO, f"RU PID Interval: {self.ru_timing_pid_interval_s}s | Target RU CPU: {self.target_ru_cpu_usage if self.ru_timing_core_indices else 'N/A'}%")
            self._log(INFO, f"Contextual Bandit Optimizer Interval: {self.optimizer_decision_interval_s}s | TDP Range: {self.tdp_min_w}W-{self.tdp_max_w}W")
            self._log(INFO, f"Contextual Bandit Actions (TDP Delta W): {self.bandit_actions}, Alpha: {self.linucb_alpha}, Lambda: {self.linucb_lambda_reg}")
            norm_param_sample_key = 'total_bits_per_second' # Key to show a sample normalization param
            norm_param_sample_val = self.norm_params.get(norm_param_sample_key, 'N/A')
            if isinstance(norm_param_sample_val, dict): # Make it more readable
                norm_param_sample_val = f"min:{norm_param_sample_val.get('min')},max:{norm_param_sample_val.get('max')}"

            self._log(INFO, f"Context Dimension: {self.context_dimension}. Norm Param Sample ({norm_param_sample_key}): {norm_param_sample_val}")
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
                    kpm_data_for_totals = self._get_and_reset_accumulated_kpm_metrics()
                    current_num_ues = self._get_and_reset_current_ue_count() 
                    self.current_num_ues_for_log = current_num_ues

                    total_bits_optimizer_interval = sum(
                        d.get('dl_bits', 0.0) + d.get('ul_bits', 0.0)
                        for d in kpm_data_for_totals.values()
                    )
                    num_kpm_reports_processed = sum(d.get('reports_in_interval',0) for d in kpm_data_for_totals.values())
                    num_active_dus = sum(1 for d in kpm_data_for_totals.values() if d.get('dl_bits',0) + d.get('ul_bits',0) > 1e-6) # Active if some bits transferred

                    significant_throughput_change = False
                    if self.total_bits_from_previous_optimizer_interval is not None: 
                        denominator = self.total_bits_from_previous_optimizer_interval
                        if denominator < 1e-6: 
                            if total_bits_optimizer_interval > 1e6 : 
                                significant_throughput_change = True 
                                self._log(INFO, "Optimizer: Throughput ramped up significantly from near zero.")
                        elif abs(total_bits_optimizer_interval - denominator) / denominator > self.throughput_change_threshold_for_discard:
                            relative_change = abs(total_bits_optimizer_interval - denominator) / denominator
                            self._log(WARN, f"Optimizer: Significant throughput change detected ({relative_change*100:.1f}%). Update for arm {self.last_selected_bandit_arm} might be skipped.")
                            significant_throughput_change = True
                    self.total_bits_from_previous_optimizer_interval = total_bits_optimizer_interval

                    current_efficiency_for_bandit: Optional[float] = None
                    if num_kpm_reports_processed > 0 : 
                        if interval_energy_uj is not None and interval_energy_uj > 1e-3: 
                            current_efficiency_for_bandit = total_bits_optimizer_interval / interval_energy_uj
                        elif total_bits_optimizer_interval > 1e-9 and (interval_energy_uj is None or interval_energy_uj <= 1e-3):
                            current_efficiency_for_bandit = float('inf') 
                        else: 
                            current_efficiency_for_bandit = 0.0
                    
                    self.most_recent_calculated_efficiency_for_log = current_efficiency_for_bandit
                    current_actual_tdp_for_context = self.current_tdp_w 
                    current_context_vec = self._get_current_context_vector(
                        total_bits_optimizer_interval, current_num_ues, num_active_dus,
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
                    
                    best_emp_arm, best_emp_eff = self.contextual_bandit.get_best_arm_empirically()
                    bandit_log = (f"CB(LastSelArm:{self.last_selected_bandit_arm or 'None'}, "
                                  f"BestEmpArm:{best_emp_arm or 'None'}@AvgEff:{best_emp_eff:.3f})")
                    
                    log_parts = [f"RU:{ru_usage_str}", f"TDP_Act:{self.current_tdp_w:>5.1f}W", 
                                 f"TDP_OptTrg:{self.optimizer_target_tdp_w:>5.1f}W", f"PkgPwr:{pkg_pwr_log_str}W", bandit_log]
                    if self.most_recent_calculated_efficiency_for_log is not None:
                         log_parts.append(f"IntEff:{self.most_recent_calculated_efficiency_for_log:.3f}b/uJ")
                    log_parts.append(f"UEsNow:{self.current_num_ues_for_log}")

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
    parser.add_argument("--http_server_port", type=int, default=8090, help="HTTP server port for xAppBase (default: 8090).")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port for xAppBase (default: 4560).")
    args = parser.parse_args()

    manager = None
    try:
        manager = PowerManager(args.config_path, args.http_server_port, args.rmr_port)
        
        # xAppBase.signal_handler calls self.stop(), which sets self.running = False
        signal.signal(signal.SIGINT, manager.signal_handler)
        signal.signal(signal.SIGTERM, manager.signal_handler)
        if hasattr(signal, 'SIGQUIT'): # SIGQUIT might not be available on Windows
             signal.signal(signal.SIGQUIT, manager.signal_handler)
        manager._log(INFO, "Registered signal handlers from xAppBase to control 'self.running'.")

        # The main execution method name in PowerManager is run_power_management_xapp
        manager.run_power_management_xapp()

    except RuntimeError as e: 
        print(f"E: Critical error during PowerManager execution: {e}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"Critical error: {e}")
        sys.exit(1)
    except SystemExit as e: 
        code = e.code if e.code is not None else 0
        print(f"Application terminated with exit code: {code}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, f"Application terminated (SystemExit: {code}).")
        sys.exit(code) 
    except Exception as e:
        print(f"E: An unexpected error occurred at the top level: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"TOP LEVEL UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        msg = "Application finished."
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, msg)
        else: print(f"INFO: {msg} (logger may not be available).")
