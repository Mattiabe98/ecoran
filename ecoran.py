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
    # ... (Function remains the same as previous complete version) ...
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            msr_val_bytes = f.read(8) 
            if len(msr_val_bytes) == 8:
                return struct.unpack('<Q', msr_val_bytes)[0]
            else:
                # print(f"W: Short read ({len(msr_val_bytes)} bytes) from MSR {hex(reg)} on CPU {cpu_id}", file=sys.stderr)
                return None
    except (FileNotFoundError, PermissionError): # Common, less verbose if happens often after initial check
        return None
    except OSError as e:
        if e.errno != 2 and e.errno != 13: 
             print(f"W: OSError reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"E: Unexpected error reading MSR {hex(reg)} on CPU {cpu_id}: {e}", file=sys.stderr)
        return None


class CoreMSRData:
    # ... (Class remains the same) ...
    def __init__(self, core_id: int):
        self.core_id = core_id
        self.mperf: Optional[int] = None
        self.tsc: Optional[int] = None
        self.busy_percent: float = 0.0

class PowerManager:
    def __init__(self, config_path: str):
        # ... (Initialization of paths, config values like previous complete version) ...
        self.config_path = config_path
        self.config = self._load_config()

        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")
        # Max energy_uj before wrap-around, default assumes 64-bit like for energy_uj
        # Some RAPL energy counters are 32-bit (PKG_ENERGY_STATUS MSR), but /sys/class/powercap energy_uj is usually wider.
        self.max_energy_val = self.config.get('rapl_max_energy_uj_override', 2**60 -1) # Heuristic, actual is in max_energy_range_uj


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
        # self.ru_core_msr_curr_data: Dict[int, CoreMSRData] = {} # curr_data is temporary inside _update

        self._validate_config()
        if self.dry_run:
            print("!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")
            print("Sensor data (MSRs, PkgWatt) WILL BE READ.")
            print("SST commands and TDP changes WILL BE PRINTED but NOT EXECUTED.")
            print("Internal TDP state WILL BE SIMULATED.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def _validate_config(self):
        # No changes needed for dry_run here, as we always read sensors now
        if not os.path.exists(self.rapl_base_path) or \
           not os.path.exists(self.power_limit_uw_file):
            print(f"E: RAPL path {self.rapl_base_path} or power limit file missing.")
            sys.exit(1)
        if not os.path.exists(self.energy_uj_file):
             print(f"W: Energy file {self.energy_uj_file} not found. PkgPower will be N/A.")

        if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'):
            print("W: 'ru_timing_cores' defined in config but resulted in an empty list after parsing. Check format.")
        elif not self.ru_timing_core_indices:
             print("INFO: No 'ru_timing_cores' defined or list is empty. MSR-based CPU utilization monitoring for RU cores will not be active.")
        elif self.ru_timing_core_indices:
            test_core = self.ru_timing_core_indices[0]
            msr_path_test = f'/dev/cpu/{test_core}/msr'
            if not os.path.exists(msr_path_test):
                print(f"E: MSR device file {msr_path_test} not found for core {test_core}. Is 'msr' kernel module loaded? (`sudo modprobe msr`)")
                sys.exit(1)
            
            test_val = read_msr_direct(test_core, MSR_IA32_TSC)
            if test_val is None:
                if not os.access(msr_path_test, os.R_OK):
                     print(f"E: Permission denied reading MSR device file {msr_path_test}. Script must be run as root.")
                else:
                     print(f"E: Failed initial MSR read on core {test_core} (MSR: {hex(MSR_IA32_TSC)}). 'msr' module might be loaded but MSR access is failing.")
                sys.exit(1)
            print("INFO: MSR access test passed.")

        try: 
            subprocess.run([self.intel_sst_path, "--version"], capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"E: '{self.intel_sst_path}' command failed or not found: {e}")
            sys.exit(1)
        
        # ... (other parameter validations)
        print("Configuration and system checks passed.")

    def _load_config(self) -> Dict[str, Any]: # ... (same)
        try:
            with open(self.config_path, 'r') as f: return yaml.safe_load(f)
        except FileNotFoundError: print(f"E: Config '{self.config_path}' not found."); sys.exit(1)
        except yaml.YAMLError as e: print(f"E: Parse config '{self.config_path}': {e}"); sys.exit(1)

    def _parse_core_list_string(self, core_str: str) -> List[int]: # ... (same)
        cores: Set[int] = set();
        if not core_str: return []
        for part in core_str.split(','):
            part = part.strip()
            if not part: continue
            if '-' in part:
                try: s_str, e_str = part.split('-',1); s,e = int(s_str), int(e_str)
                if s > e: print(f"W: Invalid core range {s}-{e} in '{part}'."); continue
                cores.update(range(s,e+1))
                except ValueError: print(f"W: Invalid core range format '{part}'."); continue
            else:
                try: cores.add(int(part))
                except ValueError: print(f"W: Invalid core number format '{part}'."); continue
        return sorted(list(cores))

    def _update_ru_core_msr_data(self):
        if not self.ru_timing_core_indices: return

        for core_id in self.ru_timing_core_indices:
            # Read current MSRs regardless of dry_run for useful dry_run output
            current_mperf = read_msr_direct(core_id, MSR_IA32_MPERF)
            current_tsc = read_msr_direct(core_id, MSR_IA32_TSC)

            current_busy_percent = 0.0 # Default if calculation fails

            # Initialize prev_data entry if it's the very first time for this core
            if core_id not in self.ru_core_msr_prev_data:
                 self.ru_core_msr_prev_data[core_id] = CoreMSRData(core_id)
                 # Store current raw MSRs to prev so next iteration can calculate. Busy % is 0.
                 self.ru_core_msr_prev_data[core_id].mperf = current_mperf
                 self.ru_core_msr_prev_data[core_id].tsc = current_tsc
                 self.ru_core_msr_prev_data[core_id].busy_percent = 0.0 # Can't calculate on first sample
                 continue 

            prev_data = self.ru_core_msr_prev_data[core_id]
            
            if prev_data.mperf is not None and prev_data.tsc is not None and \
               current_mperf is not None and current_tsc is not None:
                
                delta_mperf = current_mperf - prev_data.mperf
                if delta_mperf < 0: delta_mperf += (2**64) 
                
                delta_tsc = current_tsc - prev_data.tsc
                if delta_tsc < 0: delta_tsc += (2**64)

                if delta_tsc > 0:
                    current_busy_percent = min(100.0, 100.0 * delta_mperf / delta_tsc)
                else: 
                    current_busy_percent = prev_data.busy_percent 
            else: 
                current_busy_percent = prev_data.busy_percent
            
            # Update prev_data for the next iteration with current values
            self.ru_core_msr_prev_data[core_id].mperf = current_mperf
            self.ru_core_msr_prev_data[core_id].tsc = current_tsc
            self.ru_core_msr_prev_data[core_id].busy_percent = current_busy_percent


    def _get_control_ru_timing_cpu_usage(self) -> float: # ... (same logic, relies on updated prev_data)
        if not self.ru_timing_core_indices: return 0.0
        current_max_busy_percent = 0.0; found_valid_core_data = False
        for core_id in self.ru_timing_core_indices:
            data_point = self.ru_core_msr_prev_data.get(core_id) 
            if data_point:
                current_max_busy_percent = max(current_max_busy_percent, data_point.busy_percent)
                found_valid_core_data = True
        if not found_valid_core_data: return 0.0 
        self.max_ru_timing_usage_history.append(current_max_busy_percent)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg: self.max_ru_timing_usage_history.pop(0)
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str], use_sudo_for_tee: bool = False) -> None: # ... (same)
        actual_cmd_list_or_str: Any; shell_needed = False; print_cmd_str: str
        if cmd_list[0] == "intel-speed-select": actual_cmd_list_or_str = [self.intel_sst_path] + cmd_list[1:]; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        elif use_sudo_for_tee and cmd_list[0] == 'echo' and len(cmd_list) == 3:
            val_to_echo, target_file = cmd_list[1], cmd_list[2]
            if not (all(c.isalnum() or c in ['-', '_', '.', '/', ':'] for c in target_file) and val_to_echo.isdigit()):
                err_msg = f"Invalid chars in echo/tee: echo {val_to_echo} | sudo tee {target_file}"; print(f"E: {err_msg}");
                if not self.dry_run: raise ValueError(err_msg)
                return
            actual_cmd_list_or_str = f"echo {val_to_echo} | sudo tee {target_file}"; shell_needed = True; print_cmd_str = actual_cmd_list_or_str
        else: actual_cmd_list_or_str = cmd_list; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        if self.dry_run: print(f"[DRY RUN] Would execute: {print_cmd_str}"); return
        print(f"Executing: {print_cmd_str}")
        try: subprocess.run(actual_cmd_list_or_str, shell=shell_needed, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            cmd_executed = e.cmd if isinstance(e.cmd, str) else ' '.join(e.cmd); print(f"E: Cmd '{cmd_executed}' failed ({e.returncode}). STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}");
            if not self.dry_run: raise
        except FileNotFoundError:
            cmd_name = actual_cmd_list_or_str[0] if isinstance(actual_cmd_list_or_str, list) else actual_cmd_list_or_str.split()[0]; print(f"E: Cmd '{cmd_name}' not found.");
            if not self.dry_run: raise

    def _setup_intel_sst(self): # ... (same)
        print("--- Configuring Intel SST-CP ---");
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])
            clos_min_freqs = self.config.get('clos_min_frequency', {}); 
            for clos_id, min_freq in clos_min_freqs.items(): self._run_command(["intel-speed-select", "core-power", "config", "-c", str(clos_id), "--min", str(min_freq)])
            ran_component_cores: Dict[str, List[int]] = { name: self._parse_core_list_string(str(cs)) for name, cs in self.config.get('ran_cores', {}).items() }
            clos_associations = self.config.get('clos_association', {}); processed_clos_associations = {int(k): v for k,v in clos_associations.items()}
            for clos_id_int, ran_components in processed_clos_associations.items():
                associated_cores: Set[int] = set()
                if ran_components: 
                    for comp_name in ran_components:
                        if comp_name in ran_component_cores: associated_cores.update(ran_component_cores[comp_name])
                        else: print(f"W: RAN comp '{comp_name}' for CLOS {clos_id_int} not in 'ran_cores'.")
                if clos_id_int == 0 and self.ru_timing_core_indices:
                    if not associated_cores.issuperset(self.ru_timing_core_indices): print(f"INFO: Ensuring RU_Timing cores {self.ru_timing_core_indices} are in CLOS 0.")
                    associated_cores.update(self.ru_timing_core_indices)
                if associated_cores:
                    core_list_str = ",".join(map(str, sorted(list(associated_cores)))); self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                elif clos_id_int == 0 and self.ru_timing_core_indices and (not ran_components or not any(ran_components)):
                    core_list_str = ",".join(map(str, sorted(list(self.ru_timing_core_indices)))); print(f"INFO: Assigning only RU_Timing cores {core_list_str} to CLOS 0."); self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                elif not (clos_id_int == 0 and self.ru_timing_core_indices): print(f"W: No cores to associate with CLOS {clos_id_int}.")
            print("--- Intel SST-CP Configuration Complete ---")
        except Exception as e: print(f"An error during Intel SST-CP setup: {e}"); import traceback; traceback.print_exc();
            if not self.dry_run: print("E: Halting due to critical SST-CP failure."); sys.exit(1)
            else: print("[DRY RUN] SST-CP setup would have failed.")


    def _read_current_tdp_limit_w(self) -> float: # ... (same)
        if self.dry_run and self.last_tdp_adjustment_time > 0: return self.current_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f: return int(f.read().strip()) / 1e6
        except Exception: print(f"W: Could not read {self.power_limit_uw_file}. Assuming current TDP is min_tdp ({self.tdp_min_w}W)."); return self.tdp_min_w

    def _set_tdp_limit_w(self, tdp_watts: float): # ... (same)
        target_tdp_uw = int(tdp_watts*1e6); min_tdp_uw_config=int(self.tdp_min_w*1e6); max_tdp_uw_config=int(self.tdp_max_w*1e6)
        clamped_tdp_uw = max(min_tdp_uw_config,min(target_tdp_uw,max_tdp_uw_config)); new_tdp_w=clamped_tdp_uw/1e6
        if self.dry_run:
            if abs(self.current_tdp_w - new_tdp_w) > 0.01 : print(f"[DRY RUN] Would set TDP to {new_tdp_w:.1f}W (requested {tdp_watts:.1f}W, current sim {self.current_tdp_w:.1f}W)")
            self.current_tdp_w = new_tdp_w; return
        try: 
            with open(self.power_limit_uw_file, 'r') as f:
                if int(f.read().strip()) == clamped_tdp_uw:
                    if abs(self.current_tdp_w - new_tdp_w) > 0.01: self.current_tdp_w = new_tdp_w
                    return 
        except Exception: pass 
        try: self._run_command(["echo",str(clamped_tdp_uw),self.power_limit_uw_file],use_sudo_for_tee=True); self.current_tdp_w = new_tdp_w
        except Exception as e: 
            print(f"E: Exception during _set_tdp_limit_w after _run_command attempt: {e}");
            if not self.dry_run: raise

    def _adjust_tdp(self, control_ru_cpu_usage: float): # ... (same)
        error_percent = self.target_ru_cpu_usage - control_ru_cpu_usage; abs_error_percent = abs(error_percent) 
        sensitivity_abs_threshold_percent = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_abs_threshold_percent = sensitivity_abs_threshold_percent * self.adaptive_step_far_thresh_factor
        chosen_step_w = 0.0
        if abs_error_percent > sensitivity_abs_threshold_percent:
            chosen_step_w = self.tdp_adj_step_w_large if abs_error_percent > far_abs_threshold_percent else self.tdp_adj_step_w_small
            tdp_change_w = -chosen_step_w if error_percent > 0 else chosen_step_w
            if tdp_change_w != 0: self._set_tdp_limit_w(self.current_tdp_w + tdp_change_w)

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
                    # Handle wrap-around: if delta_e is negative, assume wrap.
                    # This assumes the energy counter won't wrap more than once between readings.
                    # A more robust solution would use max_energy_range_uj.
                    if delta_e < 0: 
                        # Try to get max_energy_range_uj if available, else use heuristic
                        max_range = self.max_energy_val
                        try:
                            with open(os.path.join(self.rapl_base_path, "max_energy_range_uj"), 'r') as f_max:
                                max_range = int(f_max.read().strip())
                        except Exception:
                            pass # Use pre-set heuristic
                        delta_e += max_range
                    
                    power_w = (delta_e / 1e6) / delta_t # uJ to J, then J/s = W
                    # Sanity check power value, e.g. not excessively large or negative
                    if 0 <= power_w < 5000 : # Assuming CPU power won't exceed 5000W
                        success = True
                    else:
                        # print(f"W: Unrealistic PkgPower calculated: {power_w:.1f}W (delta_e: {delta_e}, delta_t: {delta_t:.3f}). Using N/A.")
                        success = False; power_w = 0.0 # Reset if unrealistic
            
            if success or self.last_pkg_energy_uj is None: # Update if successful or first read
                self.last_pkg_energy_uj = current_energy_uj
                self.last_energy_read_time = current_time
            
            return power_w, success
        except Exception as e:
            # print(f"W: Could not read package energy: {e}", file=sys.stderr)
            return 0.0, False
        
    def run_monitor(self):
        if os.geteuid() != 0 and not self.dry_run:
            print("E: Script must be run as root for MSR and RAPL access (unless in dry_run mode).")
            sys.exit(1)
        
        if self.ru_timing_core_indices: # Prime MSR only if RU cores are defined
            print("Priming MSR readings for initial busy % calculation...")
            self._update_ru_core_msr_data() 
            time.sleep(0.2) 
            self._update_ru_core_msr_data() 
            print("MSR readings primed. Initial busy % values calculated.")
        else:
            print("INFO: No RU timing cores defined, skipping MSR priming.")


        # Initial TDP setting based on current PkgWatt
        # This happens regardless of dry_run for the PkgWatt reading part.
        # The _set_tdp_limit_w will respect dry_run for actual setting.
        print("Attempting to set initial TDP based on current PkgWatt...")
        initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() 
        time.sleep(0.2) 
        initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() 
        
        if pkg_power_ok and initial_pkg_power > 1.0 : 
            safe_initial_tdp = max(self.tdp_min_w, min(initial_pkg_power, self.tdp_max_w))
            print(f"Initial PkgWatt measured: {initial_pkg_power:.1f}W. Clamped safe initial TDP: {safe_initial_tdp:.1f}W.")
            self._set_tdp_limit_w(safe_initial_tdp) # This will respect dry_run
        else:
            print(f"W: Could not get valid initial PkgWatt (got: {initial_pkg_power if pkg_power_ok else 'N/A'}). Reading current TDP from RAPL or using config min_tdp.")
            # current_tdp_w is already tdp_min_w or simulated. If not dry_run, read and set.
            if not self.dry_run:
                self.current_tdp_w = self._read_current_tdp_limit_w() 
            self._set_tdp_limit_w(self.current_tdp_w) # Apply (and clamp), respects dry_run
        print(f"Effective Initial TDP after setup: {self.current_tdp_w:.1f}W.")

        self._setup_intel_sst() 
        
        self.last_tdp_adjustment_time = time.monotonic()
        last_print_time = time.monotonic()

        print(f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        # ... (other startup print lines) ...

        try:
            while True:
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
                        usage_str = f"{data.busy_percent:>5.1f}%" if data and data.busy_percent is not None else "N/A"
                        ru_core_usage_details_list.append(f"C{core_idx}:{usage_str}")
                    ru_core_details_str = ", ".join(ru_core_usage_details_list) if ru_core_usage_details_list else "N/A"

                    pkg_power_str = f"{pkg_power_w:.1f}" if pkg_power_ok else "N/A"
                    log_msg = (
                        f"{time.strftime('%H:%M:%S')} | "
                        f"RU_Cores(MSR): [{ru_core_details_str}] (S_MaxCtrl:{control_value_for_tdp:>5.1f}%) | "
                        f"TDP:{self.current_tdp_w:>5.1f}W | "
                        f"PkgPwr:{pkg_power_str:>7}W" # Adjusted padding for PkgPwr
                    )
                    print(log_msg)
                    last_print_time = current_time

                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, 1.0 - loop_duration) 
                time.sleep(sleep_time)

        except KeyboardInterrupt: print(f"\n--- Monitoring stopped by user ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        except Exception as e: 
            print(f"\n--- An unexpected error occurred in main loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}): {e} ---")
            import traceback; traceback.print_exc()
        finally: print("Exiting application.")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: sudo python3 ecoran.py <path_to_config.yaml>"); sys.exit(1)
    manager = PowerManager(config_path=sys.argv[1]); manager.run_monitor()
