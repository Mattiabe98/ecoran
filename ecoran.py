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
import random # For Thompson Sampling
import math   # For Thompson Sampling

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
    def __init__(self, core_id: int):
        self.core_id, self.mperf, self.tsc, self.busy_percent = core_id, None, None, 0.0

# --- NEW: Thompson Sampling for Gaussian Rewards ---
class ThompsonSamplerGaussian:
    def __init__(self, arm_keys: List[str], prior_mean: float = 0.0, prior_precision: float = 0.01, reward_precision: float = 1.0):
        self.arm_keys = arm_keys
        self.n_arms = len(arm_keys)
        
        # Priors for Gaussian distribution (mean mu, precision tau)
        # Precision tau = 1 / variance
        self.prior_mean = prior_mean  # mu_0
        self.prior_precision = prior_precision # tau_0 (small precision = large variance = more exploration initially)
        
        # Precision of the rewards themselves (assumed known for simplicity, or estimated)
        self.reward_precision = reward_precision # tau_r

        # Posterior parameters for each arm: (posterior_mean, posterior_precision)
        self.posterior_params = {
            arm: {'mean': self.prior_mean, 'precision': self.prior_precision, 'sum_rewards': 0.0, 'count': 0}
            for arm in self.arm_keys
        }
        self.logger = logging.getLogger("EcoRANPowerManager.ThompsonSampler") # Use main logger

    def select_arm(self) -> str:
        sampled_means = []
        for arm_key in self.arm_keys:
            params = self.posterior_params[arm_key]
            # Sample from posterior N(mu | mu_k, 1/tau_k)
            # where mu_k = posterior_mean, tau_k = posterior_precision
            try:
                sampled_value = random.gauss(params['mean'], 1.0 / math.sqrt(params['precision'] + 1e-9)) # add epsilon to avoid div by zero
            except OverflowError: # Should not happen if precision is managed
                self.logger.error(f"OverflowError for arm {arm_key} with precision {params['precision']}. Using prior_mean.")
                sampled_value = self.prior_mean

            sampled_means.append(sampled_value)
            self.logger.debug(f"Arm {arm_key}: posterior(mean={params['mean']:.3f}, prec={params['precision']:.3f}), sampled_value={sampled_value:.3f}")

        best_arm_idx = sampled_means.index(max(sampled_means))
        selected_arm_key = self.arm_keys[best_arm_idx]
        self.logger.info(f"Thompson Sampling: Selected Arm '{selected_arm_key}' (Sampled means: {['{:.2f}'.format(x) for x in sampled_means]})")
        return selected_arm_key

    def update_arm(self, arm_key: str, reward: float):
        if arm_key not in self.posterior_params:
            self.logger.error(f"Attempted to update non-existent arm: {arm_key}")
            return

        params = self.posterior_params[arm_key]
        
        # Update based on Bayesian formula for Gaussian mean with known precision
        # See: https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions 
        # (Gaussian likelihood with known variance, Gaussian prior on mean)

        # Old posterior params: mu_k, tau_k
        mu_k = params['mean']
        tau_k = params['precision']
        
        # For a single new reward 'r' with precision 'tau_r':
        # New posterior precision tau_{k+1} = tau_k + tau_r
        # New posterior mean mu_{k+1} = (tau_k * mu_k + tau_r * r) / (tau_k + tau_r)

        # If we consider 'n' rewards for the update step and sum_rewards:
        # This can be simplified if we update one reward at a time, or if we scale reward_precision by n
        # For simplicity, let's assume one reward at a time for now.
        # If 'reward' is an average over an interval, treat it as one sample for this update.

        new_posterior_precision = tau_k + self.reward_precision
        new_posterior_mean = (tau_k * mu_k + self.reward_precision * reward) / new_posterior_precision
        
        params['mean'] = new_posterior_mean
        params['precision'] = new_posterior_precision
        params['sum_rewards'] += reward
        params['count'] += 1
        
        self.logger.info(f"Thompson Sampling: Updated Arm '{arm_key}' with reward {reward:.3f}. New posterior(mean={params['mean']:.3f}, prec={params['precision']:.3f}, count={params['count']})")

    def get_best_arm_empirically(self) -> Tuple[Optional[str], float]:
        best_arm = None
        max_empirical_mean = -float('inf')
        for arm_key, params in self.posterior_params.items():
            if params['count'] > 0:
                empirical_mean = params['sum_rewards'] / params['count']
                if empirical_mean > max_empirical_mean:
                    max_empirical_mean = empirical_mean
                    best_arm = arm_key
        return best_arm, max_empirical_mean

# --- END NEW: Thompson Sampling ---


class PowerManager(xAppBase):
    MAX_VOLUME_COUNTER_KBITS = (2**32) - 1

    def __init__(self, config_path: str, http_server_port: int, rmr_port: int, kpm_ran_func_id: int = 2):
        self.config_path = config_path
        self.config = self._load_config()

        self.verbosity = int(self.config.get('console_verbosity_level', INFO))
        self.file_verbosity_cfg = int(self.config.get('file_verbosity_level', DEBUG_KPM))
        self.log_file_path_base = self.config.get('log_file_path', "/mnt/data/ecoran")
        self._setup_logging() # Logger is now available

        xapp_base_config_file = self.config.get('xapp_base_config_file', '')
        super().__init__(xapp_base_config_file, http_server_port, rmr_port)

        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        self.max_energy_val_rapl = self.config.get('rapl_max_energy_uj_override', 2**60 - 1)
        
        # --- Timing and Control Parameters ---
        self.main_loop_sleep_s = float(self.config.get('main_loop_sleep_s', 0.1)) # Granularity of main loop
        self.ru_timing_pid_interval_s = float(self.config.get('ru_timing_pid_interval_s', 1.0)) # How often RU PID runs
        self.optimizer_decision_interval_s = float(self.config.get('optimizer_decision_interval_s', 10.0)) # How often Bandit runs
        self.stats_print_interval_s = float(self.config.get('stats_print_interval_s', 5.0)) # How often full stats are printed (can be same as optimizer)

        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 90))
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 170))
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 99.5))
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        
        # RU Timing PID parameters
        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.0005)) # % of target usage
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1.0))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 3.0))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 1.5)) # Error ratio for large step
        
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        self.dry_run = bool(self.config.get('dry_run', False))
        self.current_tdp_w = float(self.tdp_min_w) # Actual current TDP, will be read or set
        self.last_pkg_energy_uj: Optional[int] = None
        self.last_energy_read_time: Optional[float] = None
        self.energy_at_last_optimizer_interval_uj: Optional[int] = None # For optimizer efficiency calculation
        
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
        for cid, comps in clos_association_config.items():
            if isinstance(comps, list): self.clos_to_du_names_map[int(cid)] = [c for c in comps if c in ran_components_in_config and c.startswith('du')]
            else: self._log(WARN, f"Components for CLOS {cid} not a list. Skipping.")

        self.kpm_gnb_last_state: Dict[str, Dict[str, Any]] = {}
        self.accumulated_kpm_metrics_for_optimizer: Dict[str, Dict[str, Any]] = {} # For optimizer efficiency calculation
        self.kpm_data_lock = threading.Lock()

        # --- Bandit Optimizer Parameters & State ---
        bandit_config = self.config.get('bandit_optimizer', {})
        bandit_actions_w_str = bandit_config.get('actions_tdp_delta_w', {"dec_10": -10.0, "dec_5": -5.0, "hold": 0.0, "inc_5": 5.0, "inc_10": 10.0})
        self.bandit_actions: Dict[str, float] = {k: float(v) for k, v in bandit_actions_w_str.items()}
        
        # Ensure 'hold' action exists for initialization if needed
        if "hold" not in self.bandit_actions:
            self.bandit_actions["hold"] = 0.0
            self._log(WARN, "Bandit actions did not contain 'hold', adding it with delta 0.0W.")

        bandit_prior_mean = float(bandit_config.get('prior_mean_efficiency', 0.0)) # Initial guess for b/uJ
        bandit_prior_precision = float(bandit_config.get('prior_precision', 0.01)) # Low precision = high variance = more exploration
        bandit_reward_precision = float(bandit_config.get('reward_precision', 1.0)) # Precision of efficiency measurement

        self.thompson_sampler = ThompsonSamplerGaussian(
            arm_keys=list(self.bandit_actions.keys()),
            prior_mean=bandit_prior_mean,
            prior_precision=bandit_prior_precision,
            reward_precision=bandit_reward_precision
        )
        self.optimizer_target_tdp_w = self.current_tdp_w # Bandit's intended TDP target
        self.last_selected_bandit_arm: Optional[str] = None

        # Timestamps for different control loops
        self.last_ru_pid_run_time: float = 0.0
        self.last_optimizer_run_time: float = 0.0
        self.last_stats_print_time: float = 0.0
        # --- END Bandit ---

        self._validate_config()
        if self.dry_run: self._log(INFO, "!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _setup_logging(self):
        self.logger = logging.getLogger("EcoRANPowerManager")
        self.logger.handlers = [] # Clear existing handlers if any (e.g. during re-init)
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
                # Use RotatingFileHandler if logs can get very large
                # fh = RotatingFileHandler(log_fp, maxBytes=20*1024*1024, backupCount=5)
                fh = logging.FileHandler(log_fp)
                fh.setLevel(file_level)
                fh_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(module)s:%(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                fh.setFormatter(fh_formatter)
                self.logger.addHandler(fh)
                # self._log(INFO, f"File logging started: {log_fp} at level {logging.getLevelName(file_level)}") # Causes recursion if logger not fully set
                self.logger.info(f"File logging started: {log_fp} at level {logging.getLevelName(file_level)}")

            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} E: Failed to set up file logging to {self.log_file_path_base}: {e}")
                # Fallback to console if file logging fails but console is enabled
                if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers) and self.verbosity > SILENT:
                    ch_fallback = logging.StreamHandler(sys.stdout)
                    ch_fallback.setLevel(console_level)
                    ch_formatter_fallback = logging.Formatter('%(asctime)s %(levelname).1s: %(message)s', datefmt='%H:%M:%S')
                    ch_fallback.setFormatter(ch_formatter_fallback)
                    self.logger.addHandler(ch_fallback)
                    self.logger.warning("File logging failed, using console fallback for logger.")


    def _log(self, level_num: int, message: str):
        if hasattr(self, 'logger') and self.logger.handlers: # Check if handlers are present
            if level_num == ERROR: self.logger.error(message)
            elif level_num == WARN: self.logger.warning(message)
            elif level_num == INFO: self.logger.info(message)
            elif level_num >= DEBUG_KPM: self.logger.debug(message)
        elif level_num > SILENT : # Fallback if logger not fully initialized or no handlers
            level_map_fallback = {ERROR: "E:", WARN: "W:", INFO: "INFO:", DEBUG_KPM: "DBG_KPM:", DEBUG_ALL: "DEBUG:"}
            print(f"{time.strftime('%H:%M:%S')} {level_map_fallback.get(level_num, f'LVL{level_num}:')} {message}")

    def _validate_config(self):
        if not os.path.exists(self.rapl_base_path) or not os.path.exists(self.power_limit_uw_file):
            print(f"E: RAPL path {self.rapl_base_path} or power limit file missing. Exiting."); sys.exit(1)
        if not os.path.exists(self.energy_uj_file): self._log(WARN, f"Energy file {self.energy_uj_file} not found.")
        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'): self._log(WARN, "'ru_timing_cores' is defined but empty.")
        elif not self.ru_timing_core_indices: self._log(INFO, "No 'ru_timing_cores' defined, RU Timing PID will be disabled.")
        elif self.ru_timing_core_indices:
            tc = self.ru_timing_core_indices[0]; mp = f'/dev/cpu/{tc}/msr'
            if not os.path.exists(mp): print(f"E: MSR file {mp} not found. Exiting."); sys.exit(1)
            if read_msr_direct(tc,MSR_IA32_TSC) is None: print(f"E: Failed MSR read on core {tc}. Exiting."); sys.exit(1)
            self._log(INFO, "MSR access test passed.")
        try: subprocess.run([self.intel_sst_path,"--version"],capture_output=True,check=True,text=True)
        except Exception as e: print(f"E: '{self.intel_sst_path}' failed: {e}. Exiting."); sys.exit(1)
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
                if dm < 0: dm += (2**64) # MPERF and TSC are 64-bit counters
                if dt < 0: dt += (2**64)
                busy = min(100.0, 100.0 * dm / dt) if dt > 0 else prev.busy_percent
            else: busy = prev.busy_percent # Use previous if current read fails
            prev.mperf, prev.tsc, prev.busy_percent = mperf, tsc, busy

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        max_b = 0.0; valid_sample_this_round = False
        for cid in self.ru_timing_core_indices:
            d = self.ru_core_msr_prev_data.get(cid)
            if d and d.busy_percent is not None: # d.busy_percent could be 0.0
                max_b = max(max_b, d.busy_percent)
                valid_sample_this_round = True
        
        if not valid_sample_this_round and not self.max_ru_timing_usage_history:
            return 0.0 # No valid data at all
        elif not valid_sample_this_round and self.max_ru_timing_usage_history:
            # If current sample is invalid, use the last valid average to avoid erratic PID behavior
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
            # Using shell=False is safer, ensure intel-speed-select is in PATH or provide full path
            process = subprocess.run(cmd_str_list, shell=False, check=True, capture_output=True, text=True, timeout=10) # Added timeout
            if process.stdout: self._log(DEBUG_ALL, f"Cmd STDOUT: {process.stdout.strip()}")
            if process.stderr: self._log(DEBUG_ALL, f"Cmd STDERR: {process.stderr.strip()}") # SST often prints info to stderr
        except subprocess.CalledProcessError as e:
            msg = f"Cmd '{e.cmd}' failed ({e.returncode}). STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}"
            self._log(ERROR, msg)
            raise RuntimeError(msg) # Re-raise for critical setup
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
            self._run_command(["intel-speed-select", "core-power", "enable"]) # Global enable
            # Configure CLOS min frequencies
            for cid_key, freq_mhz_str in self.config.get('clos_min_frequency', {}).items():
                try:
                    # intel-speed-select expects frequency in MHz or a factor (e.g., 0.8 for 800MHz if base is 1GHz)
                    # Check documentation for your specific intel-speed-select version
                    # Assuming freq_mhz_str is like "2200" for 2.2GHz
                    freq_val = str(freq_mhz_str) # Keep as string, SST tool parses it
                    self._run_command(["intel-speed-select", "core-power", "config", "-c", str(cid_key), "--min", freq_val])
                    self._log(INFO, f"SST-CP: Set CLOS {cid_key} min frequency to {freq_val} MHz.")
                except Exception as e_clos_freq:
                    self._log(ERROR, f"SST-CP: Failed to set min freq for CLOS {cid_key} to {freq_mhz_str}: {e_clos_freq}")
            
            # Associate cores to CLOS
            ran_cores = {name: self._parse_core_list_string(str(core_list_str))
                         for name, core_list_str in self.config.get('ran_cores', {}).items()}

            all_configured_cores = set() # To check for overlaps or unassigned cores if needed later
            for cid_key, component_names in self.config.get('clos_association', {}).items():
                clos_id = int(cid_key)
                cores_for_this_clos = set()
                if isinstance(component_names, list):
                    for comp_name in component_names:
                        cores_for_this_comp = ran_cores.get(comp_name, [])
                        if not cores_for_this_comp:
                            self._log(WARN, f"SST-CP: Component '{comp_name}' for CLOS {clos_id} has no cores defined in 'ran_cores'.")
                        cores_for_this_clos.update(cores_for_this_comp)
                else:
                    self._log(WARN, f"SST-CP: Components for CLOS {clos_id} ('{component_names}') is not a list. Skipping.")
                    continue

                # Ensure RU Timing cores are handled, typically in a high-priority CLOS (e.g., CLOS 0)
                if clos_id == 0 and self.ru_timing_core_indices: # Assuming CLOS 0 is high priority
                    self._log(INFO, f"SST-CP: Ensuring RU_Timing cores {self.ru_timing_core_indices} are in CLOS 0.")
                    cores_for_this_clos.update(self.ru_timing_core_indices)
                
                if cores_for_this_clos:
                    core_list_str = ",".join(map(str, sorted(list(cores_for_this_clos))))
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id)])
                    self._log(INFO, f"SST-CP: Associated cores [{core_list_str}] to CLOS {clos_id}.")
                    all_configured_cores.update(cores_for_this_clos)
                elif clos_id == 0 and self.ru_timing_core_indices and not component_names: # RU cores only in CLOS0
                     core_list_str = ",".join(map(str, sorted(list(self.ru_timing_core_indices))))
                     self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id)])
                     self._log(INFO, f"SST-CP: Associated RU_Timing cores [{core_list_str}] to CLOS {clos_id}.")
                     all_configured_cores.update(self.ru_timing_core_indices)
                else:
                    self._log(WARN, f"SST-CP: No cores to associate with CLOS {clos_id} based on components '{component_names}'.")
            self._log(INFO, "--- Intel SST-CP Configuration Complete ---")
        except Exception as e:
            self._log(ERROR, f"Intel SST-CP setup error: {e}")
            if not self.dry_run:
                raise RuntimeError(f"SST-CP Setup Failed: {e}")


    def _read_current_tdp_limit_w(self) -> float:
        if self.dry_run and hasattr(self, 'optimizer_target_tdp_w'): # Use the target if in dry run and optimizer has set one
            return self.optimizer_target_tdp_w # Or self.current_tdp_w if that's more accurate for "actual"
        try:
            with open(self.power_limit_uw_file, 'r') as f:
                return int(f.read().strip()) / 1e6
        except Exception as e:
            self._log(WARN, f"Could not read {self.power_limit_uw_file}, returning configured min_tdp ({self.tdp_min_w}W). Error: {e}")
            return float(self.tdp_min_w)

    def _set_tdp_limit_w(self, tdp_watts: float, context: str = ""):
        clamped_tdp_uw = int(max(self.tdp_min_w * 1e6, min(tdp_watts * 1e6, self.tdp_max_w * 1e6)))
        new_tdp_w = clamped_tdp_uw / 1e6

        # Check if the change is significant enough to log/act
        significant_change = abs(self.current_tdp_w - new_tdp_w) > 0.01 # More than 0.01W difference

        if self.dry_run:
            if significant_change:
                self._log(INFO, f"[DRY RUN] {context}. New Target TDP: {new_tdp_w:.1f}W (Previous: {self.current_tdp_w:.1f}W).")
            self.current_tdp_w = new_tdp_w
            return

        try:
            # Read current hardware limit to avoid unnecessary writes
            with open(self.power_limit_uw_file, 'r') as f_read:
                current_hw_limit_uw = int(f_read.read().strip())
            if current_hw_limit_uw == clamped_tdp_uw:
                if significant_change: # Update internal state even if HW is same
                    self._log(INFO, f"{context}. Target TDP: {new_tdp_w:.1f}W (already set in HW, updating internal state).")
                    self.current_tdp_w = new_tdp_w
                return # No need to write
        except Exception as e:
            self._log(WARN, f"Could not read {self.power_limit_uw_file} before write: {e}. Proceeding with write.")

        try:
            self._log(INFO, f"{context}. Setting TDP to: {new_tdp_w:.1f}W (from {self.current_tdp_w:.1f}W).")
            with open(self.power_limit_uw_file, 'w') as f_write:
                f_write.write(str(clamped_tdp_uw))
            self.current_tdp_w = new_tdp_w # Update internal state after successful write
        except OSError as e:
            self._log(ERROR, f"OSError writing TDP to {self.power_limit_uw_file}: {e}")
            # Potentially revert self.current_tdp_w or mark state as uncertain
            raise RuntimeError(f"OSError setting TDP: {e}") # Critical failure
        except Exception as e:
            self._log(ERROR, f"Exception writing TDP: {e}")
            raise RuntimeError(f"Exception setting TDP: {e}") # Critical failure

    def _run_ru_timing_pid_step(self, current_ru_cpu_usage: float):
        if not self.ru_timing_core_indices:
            return # PID is disabled if no RU cores

        error = self.target_ru_cpu_usage - current_ru_cpu_usage  # Positive error means usage is too low
        abs_error = abs(error)
        
        # Sensitivity threshold: Only act if error is significant
        # Example: if target is 99.5% and sensitivity is 0.001 (0.1%), then threshold is ~0.0995%
        sensitivity_threshold = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        
        if abs_error > sensitivity_threshold:
            # Determine step size: larger step if error is much larger than sensitivity threshold
            step_size = self.tdp_adj_step_w_large if abs_error > (sensitivity_threshold * self.adaptive_step_far_thresh_factor) else self.tdp_adj_step_w_small
            
            # If usage is too low (error > 0), we need to DECREASE TDP to make CPU busier (lower freq)
            # If usage is too high (error < 0), we need to INCREASE TDP to make CPU less busy (higher freq headroom)
            tdp_change_w = -step_size if error > 0 else step_size
            
            # Apply the change
            # The base for adjustment is the current actual TDP
            new_target_tdp = self.current_tdp_w + tdp_change_w
            
            ctx = (f"RU_PID: RU CPU {current_ru_cpu_usage:.2f}% (Target {self.target_ru_cpu_usage:.2f}%), "
                   f"Error {error:.2f}%, TDP Change {tdp_change_w:.1f}W")
            self._set_tdp_limit_w(new_target_tdp, context=ctx)
        # else:
            # self._log(DEBUG_ALL, f"RU_PID: RU CPU {current_ru_cpu_usage:.2f}%, Error {error:.2f}% within sensitivity threshold.")


    def _run_bandit_optimizer_step(self, current_efficiency_bits_per_uj: Optional[float]):
        """ Selects an arm (TDP delta), applies it, and updates the bandit based on previous action's reward. """
        
        # 1. Update bandit with the reward from the PREVIOUS action (if any)
        if self.last_selected_bandit_arm is not None:
            if current_efficiency_bits_per_uj is not None and math.isfinite(current_efficiency_bits_per_uj):
                self.thompson_sampler.update_arm(self.last_selected_bandit_arm, current_efficiency_bits_per_uj)
            else:
                self._log(WARN, f"Bandit: Invalid efficiency ({current_efficiency_bits_per_uj}) for arm '{self.last_selected_bandit_arm}'. Skipping update.")
                # Optionally, you could penalize or use a default low reward here.
                # For now, we just skip the update for this arm for this round.
        
        # 2. Select a new arm (TDP delta to try for the NEXT interval)
        selected_arm_key = self.thompson_sampler.select_arm()
        tdp_delta_w = self.bandit_actions[selected_arm_key]
        self.last_selected_bandit_arm = selected_arm_key # Store for next update

        # 3. Determine the new target TDP
        #    Base the delta on the Bandit's *intended* target from the last step,
        #    or current TDP if it's the first run.
        base_tdp_for_bandit_decision = self.optimizer_target_tdp_w
        
        proposed_next_tdp_by_bandit = base_tdp_for_bandit_decision + tdp_delta_w
        
        # Clamp to ensure it's within global min/max (though _set_tdp_limit_w also does this)
        # proposed_next_tdp_by_bandit = max(self.tdp_min_w, min(proposed_next_tdp_by_bandit, self.tdp_max_w))
        
        self._log(INFO, f"Bandit: Arm='{selected_arm_key}', Delta={tdp_delta_w:.1f}W. "
                        f"Base TDP for decision: {base_tdp_for_bandit_decision:.1f}W. "
                        f"Proposed TDP: {proposed_next_tdp_by_bandit:.1f}W.")

        # 4. Apply the new TDP
        #    This TDP will be active during the next efficiency_measurement_interval.
        #    The RU_PID might still adjust it in finer steps during that interval.
        self._set_tdp_limit_w(proposed_next_tdp_by_bandit, context=f"Optimizer Bandit (Arm: {selected_arm_key})")
        
        # 5. Update the bandit's view of its intended target
        #    It should be what was actually set (after clamping by _set_tdp_limit_w, not after RU_PID tweaks yet)
        self.optimizer_target_tdp_w = self._read_current_tdp_limit_w() # Read back what was set by _set_tdp_limit_w

    def _get_pkg_power_w(self) -> Tuple[float, bool]:
        if not os.path.exists(self.energy_uj_file): return 0.0, False
        try:
            with open(self.energy_uj_file, 'r') as f: current_e_uj = int(f.read().strip())
            now = time.monotonic(); pwr_w, ok = 0.0, False
            if self.last_pkg_energy_uj is not None and self.last_energy_read_time is not None:
                dt = now - self.last_energy_read_time
                if dt > 0.001: # Avoid division by zero or too small interval
                    de = current_e_uj - self.last_pkg_energy_uj
                    if de < 0: # RAPL counter wrapped
                        max_r = self.max_energy_val_rapl
                        try:
                            with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"),'r') as f_max_r:
                                max_r_val = int(f_max_r.read().strip());
                                if max_r_val > 0: max_r = max_r_val
                        except Exception: pass # Use configured default
                        de += max_r
                    pwr_w = (de / 1e6) / dt # uJ/s = W
                    if 0 <= pwr_w < 5000: # Sanity check (5000W is very high for a server pkg)
                        ok = True
                    else:
                        self._log(DEBUG_ALL, f"Unrealistic PkgPwr calculated: {pwr_w:.1f}W (dE={de}, dt={dt:.3f}s). Resetting baseline.")
                        ok=False; pwr_w=0.0
                        # Reset baseline to avoid propagation of error
                        self.last_pkg_energy_uj, self.last_energy_read_time = current_e_uj, now
                        return pwr_w, ok # Return immediately after reset
            
            # Update baseline only if calculation was ok or it's the first time
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
        """Calculates energy consumed since the last optimizer interval's end."""
        current_e_uj = self._read_current_energy_uj()
        if current_e_uj is None: return None

        if self.energy_at_last_optimizer_interval_uj is None: # First call or reset
            self.energy_at_last_optimizer_interval_uj = current_e_uj
            return None # Not enough data for a delta yet

        delta_e = float(current_e_uj - self.energy_at_last_optimizer_interval_uj)

        if delta_e < 0: # RAPL counter wrapped
            max_r = self.max_energy_val_rapl
            try:
                with open(os.path.join(self.rapl_base_path,"max_energy_range_uj"),'r') as f_max_r:
                    max_r_val = int(f_max_r.read().strip());
                    if max_r_val > 0: max_r = max_r_val
            except Exception as e: self._log(WARN, f"Could not read max_energy_range_uj: {e}"); pass
            delta_e += max_r
        
        self.energy_at_last_optimizer_interval_uj = current_e_uj # Update baseline for NEXT interval
        return delta_e

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str,
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes,
                                 kpm_report_style: Optional[int] = None, ue_id: Optional[Any] = None):
        # This callback accumulates KPM metrics. The main loop will process them for the optimizer.
        current_kpm_report_time = time.monotonic()
        # No direct energy reading here to avoid contention/delay in callback

        self._log(DEBUG_KPM, f"KPM CB: Agent:{e2_agent_id}, Sub:{subscription_id}, Time:{current_kpm_report_time:.3f}")
        if not self.e2sm_kpm: self._log(WARN, f"KPM from {e2_agent_id}, but e2sm_kpm unavailable."); return

        try:
            meas_report = self.e2sm_kpm.extract_meas_data(indication_msg_bytes)
            if not meas_report: self._log(WARN, f"KPM CB: Failed to extract KPM data from {e2_agent_id}."); return

            dl_kbits_this_period, ul_kbits_this_period = 0.0, 0.0
            measurements = meas_report.get("measData", {})
            if not isinstance(measurements, dict): self._log(WARN, f"KPM CB: Invalid 'measData' from {e2_agent_id}."); return

            for metric_name, value_list in measurements.items():
                if isinstance(value_list, list) and value_list:
                    value = value_list[0]
                    try:
                        if metric_name == 'DRB.RlcSduTransmittedVolumeDL': dl_kbits_this_period = float(value)
                        elif metric_name == 'DRB.RlcSduTransmittedVolumeUL': ul_kbits_this_period = float(value)
                    except (ValueError, TypeError): self._log(WARN, f"KPM CB: Metric '{metric_name}' value '{value}' invalid.")

            bits_dl_this_kpm_interval = dl_kbits_this_period * 1000.0 # Convert kbits to bits
            bits_ul_this_kpm_interval = ul_kbits_this_period * 1000.0 # Convert kbits to bits
            # total_bits_this_kpm_interval = bits_dl_this_kpm_interval + bits_ul_this_kpm_interval # Not used directly here

            with self.kpm_data_lock:
                # Accumulate for the optimizer interval
                if e2_agent_id not in self.accumulated_kpm_metrics_for_optimizer:
                    self.accumulated_kpm_metrics_for_optimizer[e2_agent_id] = {
                        'bits_sum_dl':0.0,
                        'bits_sum_ul':0.0,
                        'num_reports':0
                    }
                acc_data = self.accumulated_kpm_metrics_for_optimizer[e2_agent_id]
                acc_data['bits_sum_dl'] += bits_dl_this_kpm_interval
                acc_data['bits_sum_ul'] += bits_ul_this_kpm_interval
                acc_data['num_reports'] += 1
                
                self._log(DEBUG_KPM, f"KPM CB (for Optimizer): {e2_agent_id}: Added DL_b={bits_dl_this_kpm_interval:.0f}, UL_b={bits_ul_this_kpm_interval:.0f}")

        except Exception as e: self._log(ERROR, f"Error processing KPM indication from {e2_agent_id}: {e}"); import traceback; traceback.print_exc()

    def _get_and_reset_accumulated_kpm_for_optimizer(self) -> Dict[str, Dict[str, Any]]:
        """Gets accumulated KPM data for the current optimizer interval and resets."""
        with self.kpm_data_lock:
            snap = {}
            for gnb_id, data in self.accumulated_kpm_metrics_for_optimizer.items():
                snap[gnb_id] = {
                    'dl_bits': data.get('bits_sum_dl', 0.0),
                    'ul_bits': data.get('bits_sum_ul', 0.0),
                    'reports_in_interval': data.get('num_reports', 0)
                }
                # Reset for the next optimizer interval
                data['bits_sum_dl'] = 0.0
                data['bits_sum_ul'] = 0.0
                data['num_reports'] = 0
        return snap

    def _setup_kpm_subscriptions(self):
        self._log(INFO, "--- Setting up KPM Subscriptions ---")
        if not self.e2sm_kpm: self._log(WARN, "e2sm_kpm module unavailable for KPM subscriptions."); return
        
        nodes = list(self.gnb_ids_map.values()) # These are the E2 Node IDs (e.g., GNB_ID strings)
        if not nodes: self._log(WARN, "No gNB IDs configured for KPM subscriptions."); return

        # Ensure metrics are as per E2SM KPM standard or your specific implementation
        metrics = ['DRB.RlcSduTransmittedVolumeDL', 'DRB.RlcSduTransmittedVolumeUL']
        kpm_report_p = int(self.config.get('kpm_report_period_ms', 1000)) # e.g., 1000ms for 1s reports
        kpm_gran_p = int(self.config.get('kpm_granularity_period_ms', kpm_report_p)) # Often same as report period

        self._log(INFO, f"KPM: Subscribing to metrics: {metrics} for nodes {nodes} with ReportPeriod={kpm_report_p}ms, Granularity={kpm_gran_p}ms")

        style = 1; successes = 0 # Assuming Style 1: Periodic Reporting
        for node_id_str in nodes: # node_id_str is the GNB_ID from config
            # The callback adapter ensures the correct parameters are passed from xAppBase's generic callback mechanism
            # to your specific _kpm_indication_callback.
            # The xAppBase likely calls the provided callback with (agent_id, sub_id, header, message).
            # We need to ensure our _kpm_indication_callback matches this signature or adapt.
            # Assuming xAppBase calls it with these 4 standard args, then add our specifics.
            
            # cb_adapter = lambda e2_node_id, sub_id, ind_hdr, ind_msg: \
            #                self._kpm_indication_callback(e2_node_id, sub_id, ind_hdr, ind_msg, kpm_report_style=style)
            
            # If xAppBase handles the RAN Function ID and UE ID internally, we might not need to pass them explicitly here.
            # Let's assume the xAppBase correctly maps subscriptions to callbacks.

            if self.dry_run:
                self._log(INFO, f"[DRY RUN] KPM Sub: Node {node_id_str}, Metrics {metrics}, Style {style}")
                with self.kpm_data_lock:
                    if node_id_str not in self.accumulated_kpm_metrics_for_optimizer:
                        self.accumulated_kpm_metrics_for_optimizer[node_id_str] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'num_reports':0}
                successes+=1; continue
            
            try:
                self._log(INFO, f"Subscribing KPM: Node {node_id_str}, Metrics {metrics}, Report {kpm_report_p}ms, Granularity {kpm_gran_p}ms, Style {style}")
                # The actual subscribe call depends on your xAppBase implementation.
                # It might look like:
                self.e2sm_kpm.subscribe_report_service_style_1(
                    node_id_str,        # e2_node_id (positional)
                    kpm_report_p,       # report_period (positional)
                    metrics,            # meas_names (positional)
                    kpm_gran_p,         # granularity_period_ms (positional)
                    self._kpm_indication_callback # callback (positional)
                )
                # Or if it needs a subscription ID generated by you:
                # sub_id = f"pm_sub_{node_id_str}_{int(time.time())}"
                # self.e2sm_kpm.subscribe(node_id_str, sub_id, report_params, action_params, self._kpm_indication_callback)
                
                with self.kpm_data_lock: # Initialize accumulator for this node
                    if node_id_str not in self.accumulated_kpm_metrics_for_optimizer:
                        self.accumulated_kpm_metrics_for_optimizer[node_id_str] = {'bits_sum_dl':0.0, 'bits_sum_ul':0.0, 'num_reports':0}
                successes+=1
            except Exception as e:
                self._log(ERROR, f"KPM subscription failed for {node_id_str}: {e}"); import traceback; traceback.print_exc()
        
        if successes > 0: self._log(INFO, f"--- KPM Subscriptions: {successes} successful attempts for {len(nodes)} nodes. ---")
        elif nodes: self._log(WARN, "No KPM subscriptions successfully initiated.")
        else: self._log(INFO, "--- No KPM nodes to subscribe to. ---")


    @xAppBase.start_function # Decorator from your xAppBase
    def run_power_management_xapp(self):
        if os.geteuid() != 0 and not self.dry_run:
            self._log(ERROR, "Must be root for live run to access MSRs and RAPL. Exiting."); sys.exit(1)
        
        try:
            # --- Initializations ---
            self.current_tdp_w = self._read_current_tdp_limit_w() # Read actual HW TDP
            self.optimizer_target_tdp_w = self.current_tdp_w # Bandit's target starts at current actual
            self._log(INFO, f"Initial current TDP read from HW: {self.current_tdp_w:.1f}W. Optimizer target set to this.")

            # Prime MSR data
            if self.ru_timing_core_indices:
                self._log(INFO, "Priming MSR data for RU timing cores...")
                self._update_ru_core_msr_data(); time.sleep(0.1) # Small sleep for delta
                self._update_ru_core_msr_data()
                self._log(INFO, "MSR data primed.")
            
            # Initialize energy baseline for optimizer (and package power)
            self.energy_at_last_optimizer_interval_uj = self._read_current_energy_uj()
            self.last_pkg_energy_uj = self._read_current_energy_uj() # For _get_pkg_power_w
            self.last_energy_read_time = time.monotonic()

            if self.energy_at_last_optimizer_interval_uj is None and not self.dry_run:
                self._log(WARN, "Could not get initial package energy for optimizer baseline.")
            
            # Set initial TDP (optional, could rely on current HW setting or a default)
            # For now, we use the read HW TDP as the starting point. If you want to force an initial TDP:
            # self._set_tdp_limit_w(self.tdp_min_w, context="Initial TDP Set to Min")
            # self.optimizer_target_tdp_w = self.tdp_min_w

            self._setup_intel_sst() # Configure SST-CP (CLOS, frequencies)
            self._setup_kpm_subscriptions() # Subscribe to KPM reports

            # Initialize timestamps for control loops
            now = time.monotonic()
            self.last_ru_pid_run_time = now
            self.last_optimizer_run_time = now # Bandit will run on first valid interval
            self.last_stats_print_time = now

            self._log(INFO, f"\n--- Starting Monitoring & Control Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            self._log(INFO, f"RU Timing PID Interval: {self.ru_timing_pid_interval_s}s | Target RU CPU: {self.target_ru_cpu_usage if self.ru_timing_core_indices else 'N/A'}%")
            self._log(INFO, f"Bandit Optimizer Interval: {self.optimizer_decision_interval_s}s | TDP Range: {self.tdp_min_w}W-{self.tdp_max_w}W")
            self._log(INFO, f"Bandit Actions (TDP Delta W): {self.bandit_actions}")
            self._log(INFO, f"Stats Print Interval: {self.stats_print_interval_s}s")
            self._log(INFO, f"Main Loop Sleep: {self.main_loop_sleep_s}s")
            self._log(INFO, f"KPM metrics from gNBs: {', '.join(self.gnb_ids_map.values()) or 'NONE'}")

            # --- Main Control Loop ---
            while self.running: # Check termination flag from xAppBase
                loop_start_time = time.monotonic()

                # --- 1. RU Timing MSR Update (always, for freshest data if PID runs) ---
                if self.ru_timing_core_indices:
                    self._update_ru_core_msr_data()
                
                current_ru_cpu_usage_control_val = self._get_control_ru_timing_cpu_usage()

                # --- 2. RU Timing PID Controller (Fast Loop) ---
                if loop_start_time - self.last_ru_pid_run_time >= self.ru_timing_pid_interval_s:
                    if self.ru_timing_core_indices:
                         self._run_ru_timing_pid_step(current_ru_cpu_usage_control_val)
                    self.last_ru_pid_run_time = loop_start_time
                
                # --- 3. Bandit Optimizer (Slower Loop) ---
                if loop_start_time - self.last_optimizer_run_time >= self.optimizer_decision_interval_s:
                    # a. Calculate efficiency for the completed interval
                    interval_energy_uj = self._get_interval_energy_uj_for_optimizer() # Also resets baseline for next
                    kpm_data_optimizer = self._get_and_reset_accumulated_kpm_for_optimizer()

                    total_bits_optimizer_interval = sum(
                        d.get('dl_bits', 0.0) + d.get('ul_bits', 0.0)
                        for d in kpm_data_optimizer.values()
                    )
                    num_kpm_reports_in_interval = sum(d.get('reports_in_interval',0) for d in kpm_data_optimizer.values())

                    current_efficiency_for_bandit: Optional[float] = None
                    if num_kpm_reports_in_interval > 0: # Only calculate if we got KPM data
                        if interval_energy_uj is not None and interval_energy_uj > 1e-3: # Min 0.001 uJ
                            current_efficiency_for_bandit = total_bits_optimizer_interval / interval_energy_uj
                        elif total_bits_optimizer_interval > 1e-9 and (interval_energy_uj is None or interval_energy_uj <= 1e-3):
                            current_efficiency_for_bandit = float('inf') # High bits, negligible energy
                            self._log(WARN, "Optimizer: Calculated 'inf' efficiency. Clamping for bandit.")
                            # Bandit might not handle 'inf' well; clamp to a large number or handle in sampler.
                            # For Gaussian Thompson, 'inf' is problematic. Use a very large number or skip update.
                            # Let's skip update if inf, handled in _run_bandit_optimizer_step
                        else: # Low bits and/or low energy
                            current_efficiency_for_bandit = 0.0
                    else:
                        self._log(INFO, "Optimizer: No KPM reports in the last interval. Skipping efficiency calculation.")
                    
                    # b. Run bandit step (updates with old reward, selects new action)
                    self._run_bandit_optimizer_step(current_efficiency_for_bandit)
                    self.last_optimizer_run_time = loop_start_time
                    # Note: energy baseline for next optimizer interval was reset in _get_interval_energy_uj_for_optimizer

                # --- 4. Statistics Printing (Independent Loop) ---
                if loop_start_time - self.last_stats_print_time >= self.stats_print_interval_s:
                    # For stats, we calculate instantaneous power and overall efficiency if possible
                    # (this efficiency might be over a different window than the optimizer's)
                    pkg_pwr_w, pkg_pwr_ok = self._get_pkg_power_w() # This updates its own energy baseline

                    # For logging, use KPM data accumulated since *last print* if different from optimizer interval
                    # For simplicity, let's just log the optimizer's last calculated values or current state
                    
                    ru_usage_str = "N/A"
                    if self.ru_timing_core_indices:
                        ru_usage_parts = []
                        for cid in self.ru_timing_core_indices:
                            data = self.ru_core_msr_prev_data.get(cid)
                            ru_usage_parts.append(f"C{cid}:{data.busy_percent:>6.2f}%" if data else f"C{cid}:N/A")
                        ru_usage_str = f"[{', '.join(ru_usage_parts)}] (AvgMax:{current_ru_cpu_usage_control_val:>6.2f}%)"

                    pkg_pwr_log_str = f"{pkg_pwr_w:.1f}" if pkg_pwr_ok else "N/A"
                    
                    # Log bandit state
                    best_empirical_arm, best_empirical_eff = self.thompson_sampler.get_best_arm_empirically()
                    bandit_log = (f"Bandit(LastArm:{self.last_selected_bandit_arm or 'None'}, "
                                  f"BestEmpArm:{best_empirical_arm or 'None'}@Eff:{best_empirical_eff:.2f})")

                    log_parts = [
                        f"RU:{ru_usage_str}",
                        f"TDP_Act:{self.current_tdp_w:>5.1f}W", # Actual current TDP
                        f"TDP_OptTrg:{self.optimizer_target_tdp_w:>5.1f}W", # Bandit's intended target
                        f"PkgPwr:{pkg_pwr_log_str}W",
                        bandit_log
                    ]
                    if hasattr(self, 'current_efficiency_for_bandit') and self.current_efficiency_for_bandit is not None:
                         log_parts.append(f"LastEffCalc:{self.current_efficiency_for_bandit:.2f}b/uJ")


                    self._log(INFO, " | ".join(log_parts))
                    self.last_stats_print_time = loop_start_time
                
                # --- Loop Sleep ---
                # Calculate time spent in loop and sleep for the remainder of main_loop_sleep_s
                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, self.main_loop_sleep_s - loop_duration)
                if sleep_time > 0 : time.sleep(sleep_time)

        except KeyboardInterrupt:
            self._log(INFO, "\nMonitoring loop interrupted by user (KeyboardInterrupt).")
        except SystemExit as e: # To catch sys.exit calls
            self._log(INFO, f"Application exiting (SystemExit: {e}).")
            raise # Re-raise to allow xAppBase to handle if it has specific logic
        except RuntimeError as e: # Catch critical errors we raised (e.g., SST or TDP set failure)
            self._log(ERROR, f"Critical runtime error in main loop: {e}. xApp will terminate.")
            # self.terminate() # Signal xAppBase to terminate if it doesn't catch RuntimeError
            raise # Re-raise so it's clear the app failed critically
        except Exception as e:
            self._log(ERROR, f"\nUnexpected error in main loop: {e}")
            import traceback
            self._log(ERROR, traceback.format_exc())
            # self.terminate()
            raise # Re-raise for xAppBase or main caller
        finally:
            self._log(INFO, "--- Power Manager xApp run_power_management_xapp finished. ---")
            # Perform any cleanup if necessary (e.g., unsubscribe KPM - though xAppBase might do this)
            # if not self.dry_run and self.e2sm_kpm:
            # for gnb_id in self.gnb_ids_map.values():
            # try: self.e2sm_kpm.unsubscribe(gnb_id, "all") # Fictitious unsubscribe all
            # except: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoRAN Power Manager xApp with Thompson Sampling Optimizer")
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--http_server_port", type=int, default=8090, help="HTTP server port for xAppBase (default: 8090).")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port for xAppBase (default: 4560).")
    args = parser.parse_args()
    manager = None
    try:
        manager = PowerManager(args.config_path, args.http_server_port, args.rmr_port)
        
        # --- MODIFIED SIGNAL HANDLING SETUP ---
        # The xAppBase already has a signal_handler that calls self.stop()
        # self.stop() sets self.running = False
        signal.signal(signal.SIGINT, manager.signal_handler)
        signal.signal(signal.SIGTERM, manager.signal_handler)
        # SIGQUIT might also be relevant if your xAppBase handles it
        if hasattr(signal, 'SIGQUIT'): # SIGQUIT might not be available on all OS (e.g. Windows)
             signal.signal(signal.SIGQUIT, manager.signal_handler)
        manager._log(INFO, "Registered signal handlers from xAppBase to control 'self.running'.")
        # --- END MODIFIED SIGNAL HANDLING SETUP ---

        manager.run_power_management_xapp() # This will now loop on manager.running

    except RuntimeError as e: # Catch critical init/run errors
        print(f"E: Critical error during PowerManager execution: {e}", file=sys.stderr)
        # manager._log might not be available if error is in __init__
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"Critical error: {e}")
        sys.exit(1)
    except SystemExit as e: # Catch sys.exit called by the app
        print(f"Application terminated with exit code: {e.code}", file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, f"Application terminated (SystemExit: {e.code}).")
        sys.exit(e.code if e.code is not None else 0) # Ensure it exits with the right code
    except Exception as e:
        print(f"E: An unexpected error occurred at the top level: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(ERROR, f"TOP LEVEL UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        msg = "Application finished."
        if manager and hasattr(manager, '_log') and manager.logger.hasHandlers(): manager._log(INFO, msg)
        else: print(f"INFO: {msg} (logger may not be available).")
