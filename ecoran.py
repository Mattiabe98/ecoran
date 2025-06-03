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

try:
    from lib.xAppBase import xAppBase # Assumes xAppBase.py is now modified
except ImportError:
    print("E: Failed to import xAppBase from lib.xAppBase. Ensure the library is correctly installed and accessible.")
    sys.exit(1)

# MSR Addresses
MSR_IA32_TSC = 0x10
MSR_IA32_MPERF = 0xE7

def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
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
    def __init__(self, core_id: int):
        self.core_id = core_id
        self.mperf: Optional[int] = None
        self.tsc: Optional[int] = None
        self.busy_percent: float = 0.0

class PowerManager(xAppBase):
    def __init__(self, config_path: str, http_server_port: int, rmr_port: int, kpm_ran_func_id: int = 2):
        self.config_path = config_path
        self.config = self._load_config()

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
        self.max_ru_timing_usage_history: List[float] = []
        self.last_tdp_adjustment_time: float = 0.0
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}

        self.kpm_ran_func_id = kpm_ran_func_id
        if hasattr(self, 'e2sm_kpm') and self.e2sm_kpm is not None:
            self.e2sm_kpm.set_ran_func_id(self.kpm_ran_func_id)
        else:
            print("W: xAppBase.e2sm_kpm module not found or not initialized. KPM functionality will be disabled.")
            self.e2sm_kpm = None

        self.gnb_ids_map = self.config.get('gnb_ids', {}) 
        self.gnb_id_to_du_name_map = {v: k for k, v in self.gnb_ids_map.items()}
        clos_association_config = self.config.get('clos_association', {})
        self.clos_to_du_names_map: Dict[int, List[str]] = {}
        ran_components_in_config = self.config.get('ran_cores', {}).keys()
        for clos_id, components in clos_association_config.items():
            if not isinstance(components, list):
                print(f"W: Components for CLOS {clos_id} is not a list in config. Skipping.")
                continue
            self.clos_to_du_names_map[int(clos_id)] = [
                comp for comp in components if comp in ran_components_in_config and comp.startswith('du')
            ]
        
        self.accumulated_kpm_metrics: Dict[str, Dict[str, Any]] = {} 
        self.kpm_data_lock = threading.Lock()
        self.energy_at_last_log_uj: Optional[int] = None 
        self._validate_config()
        # ... (dry run message)

    def _validate_config(self): # Unchanged
        if not os.path.exists(self.rapl_base_path) or \
           not os.path.exists(self.power_limit_uw_file):
            raise RuntimeError(f"RAPL path {self.rapl_base_path} or power limit file missing.")
        if not os.path.exists(self.energy_uj_file):
             print(f"W: Energy file {self.energy_uj_file} not found. PkgPower/Efficiency readings will be N/A.")
        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'):
            print("W: 'ru_timing_cores' is defined in config but resulted in an empty list. MSR monitoring inactive.")
        elif not self.ru_timing_core_indices:
             print("INFO: No 'ru_timing_cores' defined. MSR-based CPU utilization monitoring inactive.")
        elif self.ru_timing_core_indices:
            test_core = self.ru_timing_core_indices[0]
            msr_path_test = f'/dev/cpu/{test_core}/msr'
            if not os.path.exists(msr_path_test):
                raise RuntimeError(f"MSR device file {msr_path_test} not found for core {test_core}.")
            test_val = read_msr_direct(test_core, MSR_IA32_TSC)
            if test_val is None:
                raise RuntimeError(f"Failed initial MSR read on core {test_core}.")
            print("INFO: MSR access test passed.")
        try: 
            subprocess.run([self.intel_sst_path, "--version"], capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"'{self.intel_sst_path}' command failed or not found: {e}.")
        # ... (other validations)
        print("Configuration and system checks passed.")

    def _load_config(self) -> Dict[str, Any]: # Unchanged
        try:
            with open(self.config_path, 'r') as f: return yaml.safe_load(f)
        except FileNotFoundError: raise RuntimeError(f"Config file '{self.config_path}' not found.")
        except yaml.YAMLError as e: raise RuntimeError(f"Could not parse config file '{self.config_path}': {e}.")

    def _parse_core_list_string(self, core_str: str) -> List[int]: # Unchanged
        cores: Set[int] = set()
        if not core_str: return []
        parts = core_str.split(',')
        for part_raw in parts:
            part = part_raw.strip();
            if not part: continue
            if '-' in part:
                try: 
                    s_str, e_str = part.split('-',1); s, e = int(s_str), int(e_str)
                    if s > e: print(f"W: Invalid core range {s}-{e}. Skipping."); continue
                    cores.update(range(s,e+1))
                except ValueError: print(f"W: Invalid number in core range '{part}'. Skipping.")
            else:
                try: cores.add(int(part))
                except ValueError: print(f"W: Invalid core number '{part}'. Skipping.")
        return sorted(list(cores))

    def _update_ru_core_msr_data(self): # Unchanged
        if not self.ru_timing_core_indices: return
        for core_id in self.ru_timing_core_indices:
            current_mperf = read_msr_direct(core_id, MSR_IA32_MPERF)
            current_tsc = read_msr_direct(core_id, MSR_IA32_TSC)
            current_busy_percent = 0.0 
            if core_id not in self.ru_core_msr_prev_data:
                 self.ru_core_msr_prev_data[core_id] = CoreMSRData(core_id)
                 self.ru_core_msr_prev_data[core_id].mperf, self.ru_core_msr_prev_data[core_id].tsc = current_mperf, current_tsc
                 continue 
            prev_data = self.ru_core_msr_prev_data[core_id]
            if prev_data.mperf is not None and prev_data.tsc is not None and current_mperf is not None and current_tsc is not None:
                delta_mperf = current_mperf - prev_data.mperf; delta_tsc = current_tsc - prev_data.tsc
                if delta_mperf < 0: delta_mperf += (2**64) 
                if delta_tsc < 0: delta_tsc += (2**64)
                if delta_tsc > 0: current_busy_percent = min(100.0, 100.0 * delta_mperf / delta_tsc)
                else: current_busy_percent = prev_data.busy_percent 
            else: current_busy_percent = prev_data.busy_percent
            self.ru_core_msr_prev_data[core_id].mperf, self.ru_core_msr_prev_data[core_id].tsc, self.ru_core_msr_prev_data[core_id].busy_percent = current_mperf, current_tsc, current_busy_percent

    def _get_control_ru_timing_cpu_usage(self) -> float: # Unchanged
        if not self.ru_timing_core_indices: return 0.0
        current_max_busy_percent = 0.0; found_valid_core_data = False
        for core_id in self.ru_timing_core_indices:
            data_point = self.ru_core_msr_prev_data.get(core_id) 
            if data_point and data_point.busy_percent is not None :
                current_max_busy_percent = max(current_max_busy_percent, data_point.busy_percent); found_valid_core_data = True
        if not found_valid_core_data and not self.max_ru_timing_usage_history : return 0.0 
        self.max_ru_timing_usage_history.append(current_max_busy_percent)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg: self.max_ru_timing_usage_history.pop(0)
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str]) -> None: # Unchanged
        actual_cmd_list_or_str: Any = cmd_list; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        if cmd_list[0] == "intel-speed-select": 
            actual_cmd_list_or_str = [self.intel_sst_path] + cmd_list[1:]; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        if self.dry_run: print(f"[DRY RUN] Would execute: {print_cmd_str}"); return
        print(f"Executing: {print_cmd_str}")
        try: subprocess.run(actual_cmd_list_or_str, shell=False, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            msg = f"Command '{e.cmd}' failed: {e.returncode}. STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}"
            print(f"E: {msg}");
            if not self.dry_run: raise RuntimeError(msg)
        except FileNotFoundError:
            msg = f"Command '{actual_cmd_list_or_str[0]}' not found."
            print(f"E: {msg}");
            if not self.dry_run: raise RuntimeError(msg)

    def _setup_intel_sst(self): # Unchanged
        print("--- Configuring Intel SST-CP ---")
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])
            for clos_id_key, min_freq in self.config.get('clos_min_frequency', {}).items():
                self._run_command(["intel-speed-select", "core-power", "config", "-c", str(clos_id_key), "--min", str(min_freq)])
            ran_component_cores = {name: self._parse_core_list_string(str(cs)) for name, cs in self.config.get('ran_cores', {}).items()}
            processed_clos_associations = {int(k): v for k, v in self.config.get('clos_association', {}).items()} 
            for clos_id_int, comps in processed_clos_associations.items():
                cores = set(c for comp_name in comps if isinstance(comps, list) for c in ran_component_cores.get(comp_name, []))
                if clos_id_int == 0 and self.ru_timing_core_indices:
                    if not cores.issuperset(self.ru_timing_core_indices): print(f"INFO: Ensuring RU_Timing cores in CLOS 0.")
                    cores.update(self.ru_timing_core_indices)
                if cores: self._run_command(["intel-speed-select", "-c", ",".join(map(str, sorted(list(cores)))), "core-power", "assoc", "-c", str(clos_id_int)])
                elif clos_id_int == 0 and self.ru_timing_core_indices and not comps :
                    self._run_command(["intel-speed-select", "-c", ",".join(map(str, sorted(self.ru_timing_core_indices))), "core-power", "assoc", "-c", str(clos_id_int)])
                else: print(f"W: No cores for CLOS {clos_id_int}.")
            print("--- Intel SST-CP Configuration Complete ---")
        except Exception as e: 
            msg = f"Intel SST-CP setup error: {e}"
            print(f"E: {msg}");
            if not self.dry_run: raise RuntimeError(msg)

    def _read_current_tdp_limit_w(self) -> float: # Unchanged
        if self.dry_run and self.last_tdp_adjustment_time > 0: return self.current_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f: return int(f.read().strip()) / 1e6
        except Exception: print(f"W: Could not read TDP. Assuming min_tdp."); return float(self.tdp_min_w)

    def _set_tdp_limit_w(self, tdp_watts: float): # Unchanged
        clamped_tdp_uw = max(int(self.tdp_min_w*1e6), min(int(tdp_watts*1e6), int(self.tdp_max_w*1e6)))
        new_tdp_w = clamped_tdp_uw / 1e6
        if self.dry_run:
            if abs(self.current_tdp_w - new_tdp_w) > 0.01 : print(f"[DRY RUN] Set TDP: {new_tdp_w:.1f}W (req {tdp_watts:.1f}W)")
            self.current_tdp_w = new_tdp_w; return
        try: 
            with open(self.power_limit_uw_file, 'r') as f_read:
                if int(f_read.read().strip()) == clamped_tdp_uw: self.current_tdp_w = new_tdp_w; return
        except Exception: pass 
        try: 
            print(f"Setting TDP: {new_tdp_w:.1f}W ({clamped_tdp_uw}uW)")
            with open(self.power_limit_uw_file, 'w') as f_write: f_write.write(str(clamped_tdp_uw))
            self.current_tdp_w = new_tdp_w
        except OSError as e_os: msg = f"OSError writing TDP: {e_os}"; print(f"E: {msg}"); if not self.dry_run: raise RuntimeError(msg)
        except Exception as e_exc: msg = f"Exception writing TDP: {e_exc}"; print(f"E: {msg}"); if not self.dry_run: raise RuntimeError(msg)

    def _adjust_tdp(self, control_ru_cpu_usage: float): # Corrected logic
        error_percent = self.target_ru_cpu_usage - control_ru_cpu_usage 
        abs_error_percent = abs(error_percent) 
        sensitivity_abs_thresh = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_abs_thresh = sensitivity_abs_thresh * self.adaptive_step_far_thresh_factor
        
        tdp_change_w = 0.0
        if abs_error_percent > sensitivity_abs_thresh:
            step = self.tdp_adj_step_w_large if abs_error_percent > far_abs_thresh else self.tdp_adj_step_w_small
            if error_percent > 0: tdp_change_w = -step # Actual usage too LOW -> DECREASE TDP
            else: tdp_change_w = step             # Actual usage too HIGH -> INCREASE TDP
            if tdp_change_w != 0:
                print(f"TDP Adjust: RU CPU {control_ru_cpu_usage:.2f}%, Target {self.target_ru_cpu_usage:.2f}%. Error {error_percent:.2f}%. Action: TDP by {tdp_change_w:.1f}W.")
                self._set_tdp_limit_w(self.current_tdp_w + tdp_change_w)

    def _get_pkg_power_w(self) -> Tuple[float, bool]: # Unchanged
        if not os.path.exists(self.energy_uj_file): return 0.0, False
        try:
            with open(self.energy_uj_file, 'r') as f: current_energy_uj = int(f.read().strip())
            current_time = time.monotonic(); power_w, success = 0.0, False
            if self.last_pkg_energy_uj is not None and self.last_energy_read_time is not None:
                delta_t = current_time - self.last_energy_read_time
                if delta_t > 0.001:
                    delta_e = current_energy_uj - self.last_pkg_energy_uj
                    if delta_e < 0: 
                        max_r = self.max_energy_val_rapl 
                        try: 
                            with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"), 'r') as f_max:
                                max_r_read = int(f_max.read().strip())
                                if max_r_read > 0: self.max_energy_val_rapl = max_r_read; max_r = max_r_read
                        except Exception: pass
                        delta_e += max_r
                    power_w = (delta_e / 1e6) / delta_t 
                    if 0 <= power_w < 5000 : success = True
                    else: success = False; power_w = 0.0 
            if success or self.last_pkg_energy_uj is None: 
                self.last_pkg_energy_uj, self.last_energy_read_time = current_energy_uj, current_time
            return power_w, success
        except Exception: return 0.0, False
        
    def _read_current_energy_uj(self) -> Optional[int]: # Unchanged
        if not os.path.exists(self.energy_uj_file): return None
        try:
            with open(self.energy_uj_file, 'r') as f: return int(f.read().strip())
        except Exception as e: print(f"W: Could not read energy_uj: {e}"); return None

    def _get_interval_energy_uj(self) -> Optional[float]: # Unchanged
        current_energy_uj = self._read_current_energy_uj()
        if current_energy_uj is None: return None
        if self.energy_at_last_log_uj is None: self.energy_at_last_log_uj = current_energy_uj; return None 
        delta_e_uj = float(current_energy_uj - self.energy_at_last_log_uj)
        if delta_e_uj < 0: 
            max_r = self.max_energy_val_rapl
            try:
                with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"), 'r') as f_max:
                    max_r_read = int(f_max.read().strip())
                    if max_r_read > 0: self.max_energy_val_rapl = max_r_read; max_r = max_r_read
            except Exception: pass 
            delta_e_uj += max_r
        self.energy_at_last_log_uj = current_energy_uj
        return delta_e_uj

    def _kpm_indication_callback(self, e2_agent_id: str, subscription_id: str, 
                                 indication_hdr_bytes: bytes, indication_msg_bytes: bytes, 
                                 kpm_report_style: Optional[int] = None, ue_id: Optional[Any] = None):
        # Using the version with extensive DEBUG prints from before
        print(f"DEBUG KPM CB: Received from {e2_agent_id}, SubID {subscription_id}, Style {kpm_report_style}, UE {ue_id}, HDR len {len(indication_hdr_bytes)}, MSG len {len(indication_msg_bytes)}")
        if not self.e2sm_kpm: print(f"W: KPM indication, but e2sm_kpm module unavailable."); return
        try:
            meas_report = self.e2sm_kpm.extract_meas_data(indication_msg_bytes) 
            print(f"DEBUG KPM CB: Extracted meas_report from {e2_agent_id}: {meas_report}")
            if not meas_report: print(f"W: Failed to extract KPM meas_data from {e2_agent_id}."); return

            granul_period_ms = meas_report.get("granulPeriod")
            print(f"DEBUG KPM CB: granulPeriod from {e2_agent_id}: {granul_period_ms}")
            if not isinstance(granul_period_ms, (int, float)) or granul_period_ms <= 0:
                print(f"W: Invalid granulPeriod ({granul_period_ms}) from {e2_agent_id}."); return
            
            granul_period_s = float(granul_period_ms) / 1000.0
            dl_thp_bps, ul_thp_bps = 0.0, 0.0
            measurements = meas_report.get("measData", {})
            print(f"DEBUG KPM CB: measData from {e2_agent_id}: {measurements}")
            if not isinstance(measurements, dict): print(f"W: Invalid 'measData' format from {e2_agent_id}."); return

            for metric_name, value in measurements.items():
                print(f"DEBUG KPM CB: Metric from {e2_agent_id}: '{metric_name}', Value: '{value}'")
                try:
                    if metric_name == 'DRB.UEThpDl': dl_thp_bps = float(value)
                    elif metric_name == 'DRB.UEThpUL': ul_thp_bps = float(value)
                except (ValueError, TypeError): print(f"W: KPM metric '{metric_name}' value '{value}' invalid.")

            dl_bits, ul_bits = dl_thp_bps * granul_period_s, ul_thp_bps * granul_period_s
            print(f"DEBUG KPM CB: Calculated for {e2_agent_id}: dl_bits={dl_bits}, ul_bits={ul_bits}")
            with self.kpm_data_lock:
                if e2_agent_id not in self.accumulated_kpm_metrics:
                    self.accumulated_kpm_metrics[e2_agent_id] = {'dl_bits': 0.0, 'ul_bits': 0.0, 'num_reports': 0, 'total_granularity_s': 0.0}
                acc = self.accumulated_kpm_metrics[e2_agent_id]
                acc['dl_bits'] += dl_bits; acc['ul_bits'] += ul_bits; acc['num_reports'] += 1; acc['total_granularity_s'] += granul_period_s
                print(f"DEBUG KPM CB: Accumulated for {e2_agent_id}: {acc}")
        except Exception as e:
            print(f"E: Error processing KPM indication from {e2_agent_id}: {e}"); import traceback; traceback.print_exc()

    def _get_and_reset_accumulated_kpm_metrics(self) -> Dict[str, Dict[str, Any]]: # Unchanged
        with self.kpm_data_lock:
            metrics_snapshot = dict(self.accumulated_kpm_metrics) 
            for gnb_id in self.accumulated_kpm_metrics:
                 self.accumulated_kpm_metrics[gnb_id] = {'dl_bits': 0.0, 'ul_bits': 0.0, 'num_reports': 0, 'total_granularity_s': 0.0}
        return metrics_snapshot

    def _setup_kpm_subscriptions(self): # Reverted to subscribe to all configured gNBs
        print("--- Setting up KPM Subscriptions ---")
        if not self.e2sm_kpm: print("W: e2sm_kpm module not available."); return

        e2_node_ids_to_subscribe = list(self.gnb_ids_map.values())
        if not e2_node_ids_to_subscribe: print("W: No gNB IDs for KPM."); return

        kpm_metrics = ['DRB.UEThpDl', 'DRB.UEThpUL'] 
        report_period_ms = int(self.config.get('kpm_report_period_ms', 1000))
        granul_period_ms = int(self.config.get('kpm_granularity_period_ms', 1000))
        current_kpm_report_style = 1 # For Style 1
        successful_subscriptions = 0

        for e2_node_id_str in e2_node_ids_to_subscribe:
            # Using the lambda adapter for the callback
            subscription_callback_adapter = lambda agent, sub, hdr, msg, style=current_kpm_report_style: \
                self._kpm_indication_callback(agent, sub, hdr, msg, kpm_report_style=style, ue_id=None)

            if self.dry_run:
                print(f"[DRY RUN] KPM Sub: Node {e2_node_id_str}, Metrics {kpm_metrics}, Style {current_kpm_report_style}")
                with self.kpm_data_lock:
                    if e2_node_id_str not in self.accumulated_kpm_metrics:
                        self.accumulated_kpm_metrics[e2_node_id_str] = {'dl_bits': 0.0, 'ul_bits': 0.0, 'num_reports': 0, 'total_granularity_s': 0.0}
                successful_subscriptions +=1; continue 
            
            try:
                print(f"Subscribing KPM: Node {e2_node_id_str}, Metrics {kpm_metrics}, Report {report_period_ms}ms, Granularity {granul_period_ms}ms, Style {current_kpm_report_style}")
                self.e2sm_kpm.subscribe_report_service_style_1(
                    e2_node_id_str, report_period_ms, kpm_metrics,
                    granul_period_ms, subscription_callback_adapter
                )
                with self.kpm_data_lock: 
                    if e2_node_id_str not in self.accumulated_kpm_metrics:
                         self.accumulated_kpm_metrics[e2_node_id_str] = {'dl_bits': 0.0, 'ul_bits': 0.0, 'num_reports': 0, 'total_granularity_s': 0.0}
                successful_subscriptions +=1
            except Exception as e: 
                print(f"E: KPM subscription failed for {e2_node_id_str}: {e}"); import traceback; traceback.print_exc()
        
        if successful_subscriptions > 0: print(f"--- KPM Subscriptions: {successful_subscriptions} nodes attempted. ---")
        elif e2_node_ids_to_subscribe : print("W: No KPM subscriptions initiated.")
    
    @xAppBase.start_function
    def run_power_management_xapp(self): # Main loop structure unchanged
        if os.geteuid() != 0 and not self.dry_run: raise SystemExit("Root privileges required.")
        try:
            self.energy_at_last_log_uj = self._read_current_energy_uj() 
            if self.ru_timing_core_indices:
                print("Priming MSR..."); self._update_ru_core_msr_data(); time.sleep(0.2); self._update_ru_core_msr_data(); print("MSR primed.")
            
            print("Setting initial TDP..."); initial_pkg_power, pkg_power_ok = self._get_pkg_power_w(); time.sleep(0.2)
            initial_pkg_power, pkg_power_ok = self._get_pkg_power_w()
            if pkg_power_ok and initial_pkg_power > 1.0 : 
                safe_initial_tdp = max(self.tdp_min_w, min(initial_pkg_power, self.tdp_max_w))
                print(f"Initial PkgWatt: {initial_pkg_power:.1f}W. Safe TDP: {safe_initial_tdp:.1f}W.")
                self._set_tdp_limit_w(safe_initial_tdp) 
            else:
                print(f"W: Invalid initial PkgWatt. Using config/current RAPL.");
                if not self.dry_run: self.current_tdp_w = self._read_current_tdp_limit_w() 
                self._set_tdp_limit_w(self.current_tdp_w) 
            print(f"Initial TDP: {self.current_tdp_w:.1f}W.")

            self._setup_intel_sst(); self._setup_kpm_subscriptions() 
            self.last_tdp_adjustment_time = time.monotonic(); last_print_time = time.monotonic()
            print(f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
            # ... (print initial setup info) ...

            while True: # Main loop
                loop_start_time = time.monotonic()
                if self.ru_timing_core_indices: self._update_ru_core_msr_data() 
                control_val_tdp = self._get_control_ru_timing_cpu_usage()
                current_time = time.monotonic()

                if current_time - self.last_tdp_adjustment_time >= self.tdp_update_interval_s:
                    if self.ru_timing_core_indices: self._adjust_tdp(control_val_tdp)
                    self.last_tdp_adjustment_time = current_time
                
                if current_time - last_print_time >= self.print_interval_s:
                    pkg_pwr_w, pkg_pwr_ok = self._get_pkg_power_w()
                    interval_egy_uj = self._get_interval_energy_uj()
                    kpm_data = self._get_and_reset_accumulated_kpm_metrics()
                    
                    tot_dl_b = sum(d.get('dl_bits',0.0) for d in kpm_data.values())
                    tot_ul_b = sum(d.get('ul_bits',0.0) for d in kpm_data.values())
                    tot_b = tot_dl_b + tot_ul_b
                    
                    srv_eff_s = "N/A"
                    if interval_egy_uj is not None:
                        if interval_egy_uj > 1e-9 : srv_eff_s = f"{tot_b / interval_egy_uj:.2f} b/uJ"
                        elif tot_b > 1e-9: srv_eff_s = "inf b/uJ"
                        else: srv_eff_s = "0.00 b/uJ"
                    
                    clos_eff_s_list = []
                    if self.clos_to_du_names_map:
                        for clos_id in sorted(self.clos_to_du_names_map.keys()):
                            clos_b = sum(kpm_data.get(self.gnb_ids_map.get(du_n), {}).get('dl_bits',0.0) + 
                                         kpm_data.get(self.gnb_ids_map.get(du_n), {}).get('ul_bits',0.0)
                                         for du_n in self.clos_to_du_names_map[clos_id] if self.gnb_ids_map.get(du_n))
                            clos_eff_v_s = "N/A"
                            if interval_egy_uj is not None:
                                if interval_egy_uj > 1e-9: clos_eff_v_s = f"{clos_b / interval_egy_uj:.2f} b/uJ"
                                elif clos_b > 1e-9: clos_eff_v_s = "inf b/uJ"
                                else: clos_eff_v_s = "0.00 b/uJ"
                            clos_eff_s_list.append(f"CLOS{clos_id}:{clos_eff_v_s} ({clos_b/1e6:.2f}Mb)")
                    clos_log_s = " | ".join(clos_eff_s_list) if clos_eff_s_list else "No CLoS DUs"

                    ru_usage_s = ", ".join([f"C{i}:{self.ru_core_msr_prev_data.get(i).busy_percent:>6.2f}%" if self.ru_core_msr_prev_data.get(i) else f"C{i}: N/A" for i in self.ru_timing_core_indices]) or "N/A"
                    pkg_pwr_s = f"{pkg_pwr_w:.1f}" if pkg_pwr_ok else "N/A"
                    egy_J_s = f"{interval_egy_uj/1e6:.2f}" if interval_egy_uj is not None else "N/A"
                    
                    log_parts = [f"{time.strftime('%H:%M:%S')}", f"RU:[{ru_usage_s}] (AvgMax:{control_val_tdp:>6.2f}%)",
                                 f"TDP:{self.current_tdp_w:>5.1f}W", f"PkgPwr:{pkg_pwr_s:>5}W", f"IntEgy:{egy_J_s:>5}J",
                                 f"TotBits:{tot_b/1e6:.2f}Mb", f"SrvEff:{srv_eff_s}", f"CLoSEff:[{clos_log_s}]"]
                    print(" | ".join(log_parts)); last_print_time = current_time

                time.sleep(max(0, 1.0 - (time.monotonic() - loop_start_time)))
        except KeyboardInterrupt: print(f"\n--- Monitoring loop interrupted ---")
        except SystemExit as e: print(f"Application exiting: {e}"); raise
        except RuntimeError as e: print(f"E: Critical runtime error: {e}")
        except Exception as e: print(f"\n--- Unexpected error: {e} ---"); import traceback; traceback.print_exc()
        finally: print("--- Power Manager xApp run_power_management_xapp finished. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoRAN Power Manager with KPM xApp")
    parser.add_argument("config_path", type=str, help="Path to YAML config.")
    parser.add_argument("--http_server_port", type=int, default=8090, help="HTTP server port.")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port.")
    args = parser.parse_args()
    try:
        manager = PowerManager(args.config_path, args.http_server_port, args.rmr_port)
        if hasattr(manager, 'signal_handler') and callable(manager.signal_handler):
            signal.signal(signal.SIGINT, manager.signal_handler)
            signal.signal(signal.SIGTERM, manager.signal_handler)
            print("INFO: Registered signal handlers.")
        manager.run_power_management_xapp() 
    except Exception as e: print(f"E: Top-level error: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    finally: print("Application finished.")
