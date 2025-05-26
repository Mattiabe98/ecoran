import yaml
import subprocess
import time
import os
import sys
import struct
from typing import List, Dict, Any, Set, Tuple, Optional

# MSR Addresses
MSR_IA32_TSC = 0x10
MSR_IA32_MPERF = 0xE7

def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
    """Reads a 64-bit MSR value for a specific CPU directly from /dev/cpu/X/msr."""
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            msr_val_bytes = f.read(8) 
            if len(msr_val_bytes) == 8:
                return struct.unpack('<Q', msr_val_bytes)[0]
            else:
                # This should be rare if the MSR exists and is readable
                # print(f"W: Short read ({len(msr_val_bytes)} bytes) from MSR {hex(reg)} on CPU {cpu_id}", file=sys.stderr)
                return None
    except FileNotFoundError: # MSR module likely not loaded or invalid CPU
        return None
    except PermissionError: # Script not run as root
        return None
    except OSError as e: # Other OS errors like I/O error if MSR is invalid
        if e.errno != 2 and e.errno != 13: # Avoid re-printing common FileNotFoundError/PermissionError
             print(f"W: OSError reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None
    except Exception as e: # Catch-all for unexpected issues
        print(f"E: Unexpected error reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None

class CoreMSRData:
    def __init__(self, core_id: int):
        self.core_id = core_id
        self.mperf: Optional[int] = None
        self.tsc: Optional[int] = None
        self.busy_percent: float = 0.0

class PowerManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        self.max_energy_val = self.config.get('rapl_max_energy_uj_override', 2**60 -1) 


        self.print_interval_s = int(self.config.get('print_interval', 5))
        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 50)) 
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 250)) 
        self.target_ru_cpu_usage = float(self.config.get('target_ru_timing_cpu_usage', 95.0))
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_update_interval_s = int(self.config.get('tdp_update_interval_s', 1))

        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.03))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 5))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 2.0))
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3))
        
        self.dry_run = bool(self.config.get('dry_run', False))

        self.current_tdp_w = self.tdp_min_w 
        self.last_pkg_energy_uj: Optional[int] = None
        self.last_energy_read_time: Optional[float] = None
        self.max_ru_timing_usage_history: List[float] = []
        self.last_tdp_adjustment_time: float = 0.0
        
        self.ru_core_msr_prev_data: Dict[int, CoreMSRData] = {}

        self._validate_config()
        if self.dry_run:
            print("!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")
            print("Sensor data (MSRs, PkgWatt) WILL BE READ.")
            print("SST commands and TDP changes WILL BE PRINTED but NOT EXECUTED.")
            print("Internal TDP state WILL BE SIMULATED.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def _validate_config(self):
        if not os.path.exists(self.rapl_base_path) or \
           not os.path.exists(self.power_limit_uw_file):
            print(f"E: RAPL path {self.rapl_base_path} or power limit file missing. Exiting.")
            sys.exit(1)
        if not os.path.exists(self.energy_uj_file):
             print(f"W: Energy file {self.energy_uj_file} not found. PkgPower readings will be N/A.")

        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'): # Check if defined but parsed to empty
            print("W: 'ru_timing_cores' is defined in config but resulted in an empty list after parsing. Check format. MSR monitoring inactive.")
        elif not self.ru_timing_core_indices: # Not defined or empty string
             print("INFO: No 'ru_timing_cores' defined or list is empty. MSR-based CPU utilization monitoring for RU cores will not be active.")
        elif self.ru_timing_core_indices: # Only test MSR access if RU cores are actually configured
            test_core = self.ru_timing_core_indices[0]
            msr_path_test = f'/dev/cpu/{test_core}/msr'
            if not os.path.exists(msr_path_test):
                print(f"E: MSR device file {msr_path_test} not found for core {test_core}. Is 'msr' kernel module loaded? (`sudo modprobe msr`). Exiting.")
                sys.exit(1)
            
            test_val = read_msr_direct(test_core, MSR_IA32_TSC)
            if test_val is None:
                # Try to give a more specific reason
                try:
                    with open(msr_path_test, 'rb') as f_test: # Try opening to check permission
                        f_test.seek(MSR_IA32_TSC) # Try seek
                        f_test.read(8) # Try read
                except PermissionError:
                    print(f"E: Permission denied reading MSR device file {msr_path_test}. Script must be run as root. Exiting.")
                    sys.exit(1)
                except FileNotFoundError: # Should have been caught by os.path.exists
                     print(f"E: MSR device file {msr_path_test} disappeared or invalid core. Exiting.")
                     sys.exit(1)
                except OSError as e_os:
                     print(f"E: OSError during MSR test read on core {test_core} (MSR: {hex(MSR_IA32_TSC)}): {e_os}. 'msr' module might be loaded but access fails. Exiting.")
                     sys.exit(1)
                print(f"E: Failed initial MSR read on core {test_core} (MSR: {hex(MSR_IA32_TSC)}) for unknown reason after path/permission checks. Exiting.")
                sys.exit(1)
            print("INFO: MSR access test passed.")

        try: 
            subprocess.run([self.intel_sst_path, "--version"], capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"E: '{self.intel_sst_path}' command failed or not found: {e}. Exiting.")
            sys.exit(1)
        
        if self.tdp_update_interval_s <= 0: print(f"E: 'tdp_update_interval_s' must be positive. Exiting."); sys.exit(1)
        if not (0 < self.tdp_adj_sensitivity_factor < 1): print(f"E: 'tdp_adjustment_sensitivity' must be > 0 and < 1. Exiting."); sys.exit(1)
        if self.tdp_adj_step_w_small <=0 : print(f"E: 'tdp_adjustment_step_w_small' must be positive. Exiting."); sys.exit(1)
        if self.tdp_adj_step_w_large <=0 : print(f"E: 'tdp_adjustment_step_w_large' must be positive. Exiting."); sys.exit(1)
        if self.adaptive_step_far_thresh_factor <=1.0 : print(f"E: 'adaptive_step_far_threshold_factor' must be > 1.0. Exiting."); sys.exit(1)
        if self.max_samples_cpu_avg <=0 : print(f"E: 'max_cpu_usage_samples' must be positive. Exiting."); sys.exit(1)
        if not (0 < self.target_ru_cpu_usage <= 100): print(f"E: 'target_ru_timing_cpu_usage' must be > 0 and <= 100. Exiting."); sys.exit(1)

        print("Configuration and system checks passed.")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f: return yaml.safe_load(f)
        except FileNotFoundError: print(f"E: Config file '{self.config_path}' not found. Exiting."); sys.exit(1)
        except yaml.YAMLError as e: print(f"E: Could not parse config file '{self.config_path}': {e}. Exiting."); sys.exit(1)

    def _parse_core_list_string(self, core_str: str) -> List[int]:
        cores: Set[int] = set()
        if not core_str: return []
        parts = core_str.split(',')
        for part_raw in parts:
            part = part_raw.strip()
            if not part: continue # Skip empty parts resulting from "1,,2"
            if '-' in part:
                try: 
                    s_str, e_str = part.split('-',1) # Split only on the first hyphen
                    s, e = int(s_str), int(e_str)
                    if s > e: 
                        print(f"W: Invalid core range {s}-{e} in '{part}' (start > end). Skipping.")
                        continue
                    cores.update(range(s,e+1))
                except ValueError: 
                    print(f"W: Invalid number in core range format '{part}'. Skipping.")
                    continue # Skip this malformed part
            else:
                try: 
                    cores.add(int(part))
                except ValueError: 
                    print(f"W: Invalid core number format '{part}'. Skipping.")
                    continue # Skip this malformed part
        return sorted(list(cores))

    def _update_ru_core_msr_data(self):
        if not self.ru_timing_core_indices: return

        for core_id in self.ru_timing_core_indices:
            current_mperf = read_msr_direct(core_id, MSR_IA32_MPERF)
            current_tsc = read_msr_direct(core_id, MSR_IA32_TSC)
            current_busy_percent = 0.0 

            if core_id not in self.ru_core_msr_prev_data:
                 self.ru_core_msr_prev_data[core_id] = CoreMSRData(core_id)
                 self.ru_core_msr_prev_data[core_id].mperf = current_mperf
                 self.ru_core_msr_prev_data[core_id].tsc = current_tsc
                 self.ru_core_msr_prev_data[core_id].busy_percent = 0.0 
                 continue 

            prev_data = self.ru_core_msr_prev_data[core_id]
            
            if prev_data.mperf is not None and prev_data.tsc is not None and \
               current_mperf is not None and current_tsc is not None:
                
                delta_mperf = current_mperf - prev_data.mperf
                if delta_mperf < 0: delta_mperf += (2**64) 
                
                delta_tsc = current_tsc - prev_data.tsc
                if delta_tsc < 0: delta_tsc += (2**64)

                if delta_tsc > 0:
                    calculated_busy = 100.0 * delta_mperf / delta_tsc
                    # Strict cap at 100.0% for C0 residency.
                    # If calculated_busy is e.g. 100.00001, it becomes 100.0.
                    # If it's 99.9999, it remains ~99.9999 (Python float precision).
                    current_busy_percent = min(100.0, calculated_busy) 
                else: 
                    current_busy_percent = prev_data.busy_percent 
            else: 
                current_busy_percent = prev_data.busy_percent
            
            self.ru_core_msr_prev_data[core_id].mperf = current_mperf
            self.ru_core_msr_prev_data[core_id].tsc = current_tsc
            self.ru_core_msr_prev_data[core_id].busy_percent = current_busy_percent

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        current_max_busy_percent = 0.0
        found_valid_core_data = False
        for core_id in self.ru_timing_core_indices:
            data_point = self.ru_core_msr_prev_data.get(core_id) 
            if data_point and data_point.busy_percent is not None : # Ensure busy_percent is not None
                current_max_busy_percent = max(current_max_busy_percent, data_point.busy_percent)
                found_valid_core_data = True
        
        if not found_valid_core_data: 
            # If no valid data and it's the first few cycles, history might be empty.
            # Return 0 to avoid division by zero if history is empty.
            # The TDP algorithm should ideally not react aggressively to initial 0s.
            return 0.0 
            
        self.max_ru_timing_usage_history.append(current_max_busy_percent)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg: 
            self.max_ru_timing_usage_history.pop(0)
        
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str], use_sudo_for_tee: bool = False) -> None:
        actual_cmd_list_or_str: Any
        shell_needed = False
        print_cmd_str: str

        if cmd_list[0] == "intel-speed-select": 
            actual_cmd_list_or_str = [self.intel_sst_path] + cmd_list[1:]
            print_cmd_str = ' '.join(actual_cmd_list_or_str)
        elif use_sudo_for_tee and cmd_list[0] == 'echo' and len(cmd_list) == 3: # echo VAL FILE
            val_to_echo, target_file = cmd_list[1], cmd_list[2]
            # Basic check for safety, allow ':' for intel-rapl:X in path
            if not (all(c.isalnum() or c in ['-', '_', '.', '/', ':'] for c in target_file) and val_to_echo.isdigit()):
                err_msg = f"Invalid characters or format in echo/tee command: echo {val_to_echo} | sudo tee {target_file}"
                print(f"E: {err_msg}")
                if not self.dry_run: raise ValueError(err_msg) # Still raise if not dry_run
                return 
            actual_cmd_list_or_str = f"echo {val_to_echo} | sudo tee {target_file}"
            shell_needed = True
            print_cmd_str = actual_cmd_list_or_str
        else: 
            actual_cmd_list_or_str = cmd_list
            print_cmd_str = ' '.join(actual_cmd_list_or_str)
        
        if self.dry_run: 
            print(f"[DRY RUN] Would execute: {print_cmd_str}")
            return
        
        # For actual execution:
        print(f"Executing: {print_cmd_str}")
        try: 
            subprocess.run(actual_cmd_list_or_str, shell=shell_needed, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            cmd_executed = e.cmd if isinstance(e.cmd, str) else ' '.join(e.cmd)
            print(f"E: Command '{cmd_executed}' failed with exit code {e.returncode}.")
            if e.stdout: print(f"   STDOUT: {e.stdout.strip()}")
            if e.stderr: print(f"   STDERR: {e.stderr.strip()}")
            # Do not raise if dry_run, as it's already handled. If not dry_run, this implies a real failure.
            if not self.dry_run: raise 
        except FileNotFoundError:
            # Extract the command name that was not found
            cmd_name_failed = actual_cmd_list_or_str
            if isinstance(actual_cmd_list_or_str, list):
                cmd_name_failed = actual_cmd_list_or_str[0]
            elif isinstance(actual_cmd_list_or_str, str): # For shell=True commands
                cmd_name_failed = actual_cmd_list_or_str.split()[0]
            print(f"E: Command '{cmd_name_failed}' not found. Is it installed and in PATH?")
            if not self.dry_run: raise 

    def _setup_intel_sst(self):
        print("--- Configuring Intel SST-CP ---")
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])

            clos_min_freqs = self.config.get('clos_min_frequency', {})
            for clos_id_key, min_freq in clos_min_freqs.items(): # Key can be int or str from YAML
                self._run_command(["intel-speed-select", "core-power", "config", "-c", str(clos_id_key), "--min", str(min_freq)])

            ran_component_cores: Dict[str, List[int]] = {
                name: self._parse_core_list_string(str(core_str_val)) # Ensure core_str_val is string
                for name, core_str_val in self.config.get('ran_cores', {}).items()
            }

            clos_associations = self.config.get('clos_association', {})
            # Keys from YAML might be integers if written as numbers, or strings. Standardize to int for logic.
            processed_clos_associations = {int(k): v for k, v in clos_associations.items()} 

            for clos_id_int, ran_components_list in processed_clos_associations.items():
                associated_cores: Set[int] = set()
                if ran_components_list: # Ensure ran_components_list is not None and is iterable
                    for comp_name in ran_components_list:
                        if comp_name in ran_component_cores:
                            associated_cores.update(ran_component_cores[comp_name])
                        else:
                            print(f"W: RAN component '{comp_name}' for CLOS {clos_id_int} not found in 'ran_cores' definitions. Skipping this component.")
                
                # Always add RU_Timing cores to CLOS 0 if it's being configured
                if clos_id_int == 0 and self.ru_timing_core_indices:
                    if not associated_cores.issuperset(self.ru_timing_core_indices): # Avoid duplicate messages
                         print(f"INFO: Ensuring RU_Timing cores {self.ru_timing_core_indices} are included in CLOS 0.")
                    associated_cores.update(self.ru_timing_core_indices)
                
                if associated_cores:
                    core_list_str = ",".join(map(str, sorted(list(associated_cores))))
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                elif clos_id_int == 0 and self.ru_timing_core_indices and (not ran_components_list or not any(ran_components_list)):
                    # This case handles if CLOS 0 is defined in clos_association but with an empty component list,
                    # or if CLOS 0 wasn't in clos_association but we still want to assign RU cores to it.
                    # However, current logic requires CLOS 0 to be in clos_association for this to be hit.
                    # For simplicity, if RU cores exist and CLOS 0 is being processed, they are added above.
                    # This specific 'elif' might be redundant if the above logic for CLOS 0 is comprehensive.
                    # Let's assume if associated_cores is empty for CLOS 0, but RU cores exist, we create the association for RU cores.
                    core_list_str = ",".join(map(str, sorted(list(self.ru_timing_core_indices))))
                    print(f"INFO: Assigning only RU_Timing cores {core_list_str} to CLOS 0 as no other components were specified or resolved for it.")
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                else:
                    # Avoid warning if this is CLOS 0 and it only consisted of RU cores (which are now in associated_cores)
                    if not (clos_id_int == 0 and self.ru_timing_core_indices and not associated_cores):
                         print(f"W: No cores found or specified to associate with CLOS {clos_id_int}.")
            
            print("--- Intel SST-CP Configuration Complete ---")
        except Exception as e: 
            print(f"An error occurred during Intel SST-CP setup: {e}")
            import traceback
            traceback.print_exc() 
            if not self.dry_run: 
                print("E: Halting application due to critical SST-CP failure.")
                sys.exit(1)
            else: 
                print("[DRY RUN] Intel SST-CP setup would have failed due to the above error.")


    def _read_current_tdp_limit_w(self) -> float:
        if self.dry_run and self.last_tdp_adjustment_time > 0: 
            return self.current_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f: 
                return int(f.read().strip()) / 1e6
        except Exception: 
            print(f"W: Could not read {self.power_limit_uw_file}. Assuming current TDP is min_tdp ({self.tdp_min_w}W).")
            return self.tdp_min_w

    def _set_tdp_limit_w(self, tdp_watts: float):
        target_tdp_uw = int(tdp_watts*1e6)
        min_tdp_uw_config=int(self.tdp_min_w*1e6)
        max_tdp_uw_config=int(self.tdp_max_w*1e6)
        
        clamped_tdp_uw = max(min_tdp_uw_config,min(target_tdp_uw,max_tdp_uw_config))
        new_tdp_w=clamped_tdp_uw/1e6

        if self.dry_run:
            if abs(self.current_tdp_w - new_tdp_w) > 0.01 : 
                 print(f"[DRY RUN] Would set TDP to {new_tdp_w:.1f}W (requested {tdp_watts:.1f}W, current sim TDP {self.current_tdp_w:.1f}W)")
            self.current_tdp_w = new_tdp_w 
            return

        try: 
            with open(self.power_limit_uw_file, 'r') as f:
                if int(f.read().strip()) == clamped_tdp_uw:
                    if abs(self.current_tdp_w - new_tdp_w) > 0.01: 
                        self.current_tdp_w = new_tdp_w
                    return 
        except Exception: pass 

        try: 
            self._run_command(["echo",str(clamped_tdp_uw),self.power_limit_uw_file],use_sudo_for_tee=False)
            self.current_tdp_w = new_tdp_w
        except Exception as e: 
            print(f"E: Exception during _set_tdp_limit_w writing to TDP limit file: {e}")
            if not self.dry_run: raise 

    def _adjust_tdp(self, control_ru_cpu_usage: float):
        error_percent = self.target_ru_cpu_usage - control_ru_cpu_usage 
        abs_error_percent = abs(error_percent) 
        
        sensitivity_abs_threshold_percent = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_abs_threshold_percent = sensitivity_abs_threshold_percent * self.adaptive_step_far_thresh_factor
        
        chosen_step_w = 0.0
        if abs_error_percent > sensitivity_abs_threshold_percent: 
            chosen_step_w = self.tdp_adj_step_w_large if abs_error_percent > far_abs_threshold_percent else self.tdp_adj_step_w_small
            tdp_change_w = -chosen_step_w if error_percent > 0 else chosen_step_w
            
            if tdp_change_w != 0:
                if self.dry_run:
                    print(f"[DRY RUN] Control CPU usage {control_ru_cpu_usage:.1f}%, target {self.target_ru_cpu_usage}%. Error {error_percent:.1f}%.")
                    print(f"[DRY RUN] Action: Change TDP by {tdp_change_w:.1f}W from current sim TDP {self.current_tdp_w:.1f}W.")
                new_tdp_w = self.current_tdp_w + tdp_change_w
                self._set_tdp_limit_w(new_tdp_w)

    def _get_pkg_power_w(self) -> Tuple[float, bool]:
        if not os.path.exists(self.energy_uj_file): return 0.0, False
        try:
            with open(self.energy_uj_file, 'r') as f: current_energy_uj = int(f.read().strip())
            current_time = time.monotonic()
            power_w, success = 0.0, False

            if self.last_pkg_energy_uj is not None and self.last_energy_read_time is not None:
                delta_t = current_time - self.last_energy_read_time
                if delta_t > 0.001: 
                    delta_e = current_energy_uj - self.last_pkg_energy_uj
                    if delta_e < 0: 
                        max_r = self.max_energy_val
                        try: # Attempt to read actual max range if not already cached
                            with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"), 'r') as f_max:
                                max_r = int(f_max.read().strip())
                                self.max_energy_val = max_r # Cache it
                        except Exception: pass 
                        delta_e += max_r
                    
                    power_w = (delta_e / 1e6) / delta_t 
                    if 0 <= power_w < 5000 : 
                        success = True
                    else:
                        # print(f"W: Unrealistic PkgPower: {power_w:.1f}W (dE:{delta_e}, dT:{delta_t:.3f})")
                        success = False; power_w = 0.0 
            
            # Update last known values only if calculation was successful or it's the first time
            if success or self.last_pkg_energy_uj is None: 
                self.last_pkg_energy_uj = current_energy_uj
                self.last_energy_read_time = current_time
            # If not successful and not first time, retain old last values for next attempt to get a larger delta
            
            return power_w, success
        except Exception:
            return 0.0, False
        
    def run_monitor(self):
        if os.geteuid() != 0 and not self.dry_run:
            print("E: Script must be run as root for MSR and RAPL access (unless in dry_run mode). Exiting.")
            sys.exit(1)
        
        if self.ru_timing_core_indices:
            print("Priming MSR readings for initial busy % calculation...")
            self._update_ru_core_msr_data() 
            time.sleep(0.2) 
            self._update_ru_core_msr_data() 
            print("MSR readings primed. Initial busy % values calculated.")
        else:
            print("INFO: No RU timing cores defined, skipping MSR priming.")

        print("Attempting to set initial TDP based on current PkgWatt...")
        initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() 
        time.sleep(0.2) 
        initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() 
        
        if pkg_power_ok and initial_pkg_power > 1.0 : 
            safe_initial_tdp = max(self.tdp_min_w, min(initial_pkg_power, self.tdp_max_w))
            print(f"Initial PkgWatt measured: {initial_pkg_power:.1f}W. Clamped safe initial TDP: {safe_initial_tdp:.1f}W.")
            self._set_tdp_limit_w(safe_initial_tdp) 
        else:
            print(f"W: Could not get valid initial PkgWatt (got: {initial_pkg_power if pkg_power_ok else 'N/A'}). Using config min_tdp or current RAPL value.")
            if not self.dry_run: # In dry_run, current_tdp_w is already tdp_min_w
                self.current_tdp_w = self._read_current_tdp_limit_w() 
            self._set_tdp_limit_w(self.current_tdp_w) 
        print(f"Effective Initial TDP after setup: {self.current_tdp_w:.1f}W.")

        self._setup_intel_sst() 
        
        self.last_tdp_adjustment_time = time.monotonic()
        last_print_time = time.monotonic()
        print(f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        print(f"Target MAX RU CPU (MSR-based): {self.target_ru_cpu_usage:.2f}% | RU Cores Monitored: {self.ru_timing_core_indices if self.ru_timing_core_indices else 'NONE'}") # Display target with more precision
        print(f"TDP Update Interval: {self.tdp_update_interval_s}s | Print Interval: {self.print_interval_s}s")
        print(f"TDP Range: {self.tdp_min_w}W - {self.tdp_max_w}W")
        print(f"TDP Adjust: Sens={self.tdp_adj_sensitivity_factor*100:.2f}%, SmallStep={self.tdp_adj_step_w_small}W, LargeStep={self.tdp_adj_step_w_large}W, FarFactor={self.adaptive_step_far_thresh_factor}")
        print(f"CPU Max Samples: {self.max_samples_cpu_avg}")

        try:
            while True:
                # ... (loop_start_time, MSR update, control value calculation remain the same) ...
                loop_start_time = time.monotonic()
                
                if self.ru_timing_core_indices: 
                    self._update_ru_core_msr_data() 

                control_value_for_tdp = self._get_control_ru_timing_cpu_usage()
                
                current_time = time.monotonic()
                if current_time - self.last_tdp_adjustment_time >= self.tdp_update_interval_s:
                    self._adjust_tdp(control_value_for_tdp)
                    self.last_tdp_adjustment_time = current_time
                
                if current_time - last_print_time >= self.print_interval_s:
                    pkg_power_w, pkg_power_ok = self._get_pkg_power_w()
                    
                    ru_core_usage_details_list = []
                    for core_idx in self.ru_timing_core_indices:
                        data = self.ru_core_msr_prev_data.get(core_idx) 
                        # Format busy_percent to 2 decimal places for logging
                        usage_str = f"{data.busy_percent:>6.2f}%" if data and data.busy_percent is not None else " N/A  " # Pad N/A to match width
                        ru_core_usage_details_list.append(f"C{core_idx}:{usage_str}")
                    ru_core_details_str = ", ".join(ru_core_usage_details_list) if ru_core_usage_details_list else "N/A"

                    pkg_power_str = f"{pkg_power_w:.1f}" if pkg_power_ok else "N/A"
                    log_msg = (
                        f"{time.strftime('%H:%M:%S')} | "
                        # Format control_value_for_tdp to 2 decimal places
                        f"RU_Cores(MSR): [{ru_core_details_str}] (S_MaxCtrl:{control_value_for_tdp:>6.2f}%) | " 
                        f"TDP:{self.current_tdp_w:>5.1f}W | "
                        f"PkgPwr:{pkg_power_str:>7}W" 
                    )
                    print(log_msg)
                    last_print_time = current_time

                # ... (loop_duration, sleep, except, finally remain the same) ...
                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, 1.0 - loop_duration) 
                time.sleep(sleep_time)

        except KeyboardInterrupt: print(f"\n--- Monitoring stopped by user ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        except Exception as e: 
            print(f"\n--- An unexpected error occurred in main loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}): {e} ---")
            import traceback; traceback.print_exc()
        finally: print("Exiting application.")

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: sudo python3 ecoran.py <path_to_config.yaml>")
        sys.exit(1)
    
    manager = PowerManager(config_path=sys.argv[1])
    manager.run_monitor()
