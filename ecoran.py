import yaml
import subprocess # Still used for intel-speed-select
import time
import os
import sys
import struct # For MSR reading
from typing import List, Dict, Any, Set, Tuple, Optional

# MSR Addresses
MSR_IA32_TSC = 0x10
MSR_IA32_MPERF = 0xE7
# MSR_IA32_APERF = 0xE8 # If you prefer APERF/MPERF for actual frequency utilization

def read_msr_direct(cpu_id: int, reg: int) -> Optional[int]:
    """Reads a 64-bit MSR value for a specific CPU directly from /dev/cpu/X/msr."""
    try:
        with open(f'/dev/cpu/{cpu_id}/msr', 'rb') as f:
            f.seek(reg)
            msr_val_bytes = f.read(8) # Read 8 bytes for a 64-bit MSR
            if len(msr_val_bytes) == 8:
                return struct.unpack('<Q', msr_val_bytes)[0] # '<Q' is little-endian unsigned long long (64-bit)
            else:
                print(f"W: Short read ({len(msr_val_bytes)} bytes) from MSR {hex(reg)} on CPU {cpu_id}", file=sys.stderr)
                return None
    except FileNotFoundError:
        return None
    except PermissionError:
        return None
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

class PowerManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")

        self.print_interval_s = int(self.config.get('print_interval', 5))
        self.tdp_min_w = int(self.config.get('tdp_range', {}).get('min', 50)) # Default if missing
        self.tdp_max_w = int(self.config.get('tdp_range', {}).get('max', 250)) # Default if missing
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
        self.ru_core_msr_curr_data: Dict[int, CoreMSRData] = {}

        self._validate_config()
        if self.dry_run:
            print("!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")
            print("Commands will be printed but NOT executed.")
            print("TDP limit changes will be simulated.")
            print("MSR reads (except initial validation) will be simulated.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def _validate_config(self):
        if not self.dry_run:
            if not os.path.exists(self.rapl_base_path) or \
               not os.path.exists(self.power_limit_uw_file):
                print(f"E: RAPL path {self.rapl_base_path} or power limit file missing.")
                sys.exit(1)
            if not os.path.exists(self.energy_uj_file):
                 print(f"W: Energy file {self.energy_uj_file} not found. PkgPower will be N/A.")

            if not self.ru_timing_core_indices and self.config.get('ru_timing_cores'): # If defined but parsed to empty
                print("W: 'ru_timing_cores' defined in config but resulted in an empty list after parsing. Check format.")
            elif not self.ru_timing_core_indices: # If not defined or empty string
                 print("INFO: No 'ru_timing_cores' defined or list is empty. MSR-based CPU utilization monitoring for RU cores will not be active.")
            elif self.ru_timing_core_indices:
                test_core = self.ru_timing_core_indices[0]
                msr_path_test = f'/dev/cpu/{test_core}/msr'
                if not os.path.exists(msr_path_test):
                    print(f"E: MSR device file {msr_path_test} not found for core {test_core}. Is 'msr' kernel module loaded? (`sudo modprobe msr`)")
                    sys.exit(1)
                
                # Attempt a test read
                test_val = read_msr_direct(test_core, MSR_IA32_TSC)
                if test_val is None:
                    # Check for common reasons if read_msr_direct failed silently before
                    if not os.access(msr_path_test, os.R_OK):
                         print(f"E: Permission denied reading MSR device file {msr_path_test}. Script must be run as root.")
                    else:
                         print(f"E: Failed initial MSR read on core {test_core} (MSR: {hex(MSR_IA32_TSC)}). 'msr' module might be loaded but MSR access is failing for other reasons.")
                    sys.exit(1)
                print("INFO: MSR access test passed.")

        try: 
            subprocess.run([self.intel_sst_path, "--version"], capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"E: '{self.intel_sst_path}' command failed or not found: {e}")
            sys.exit(1)
        
        if self.tdp_update_interval_s <= 0: print(f"E: 'tdp_update_interval_s' must be positive."); sys.exit(1)
        if not (0 < self.tdp_adj_sensitivity_factor < 1): print(f"E: 'tdp_adjustment_sensitivity' must be > 0 and < 1."); sys.exit(1)
        if self.tdp_adj_step_w_small <=0 : print(f"E: 'tdp_adjustment_step_w_small' must be positive."); sys.exit(1)
        if self.tdp_adj_step_w_large <=0 : print(f"E: 'tdp_adjustment_step_w_large' must be positive."); sys.exit(1)
        if self.adaptive_step_far_thresh_factor <=1.0 : print(f"E: 'adaptive_step_far_threshold_factor' must be > 1.0."); sys.exit(1)
        if self.max_samples_cpu_avg <=0 : print(f"E: 'max_cpu_usage_samples' must be positive."); sys.exit(1)
        if self.target_ru_cpu_usage <=0 or self.target_ru_cpu_usage > 100: print(f"E: 'target_ru_timing_cpu_usage' must be between 0 and 100."); sys.exit(1)


        print("Configuration and system checks passed.")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f: return yaml.safe_load(f)
        except FileNotFoundError: print(f"E: Config '{self.config_path}' not found."); sys.exit(1)
        except yaml.YAMLError as e: print(f"E: Parse config '{self.config_path}': {e}"); sys.exit(1)

    def _parse_core_list_string(self, core_str: str) -> List[int]:
        cores: Set[int] = set();
        if not core_str: return []
        for part in core_str.split(','):
            part = part.strip()
            if not part: continue
            if '-' in part:
                try: 
                    s_str, e_str = part.split('-',1)
                    s,e = int(s_str), int(e_str)
                    if s > e: print(f"W: Invalid core range {s}-{e} in '{part}'. Skipping."); continue
                    cores.update(range(s,e+1))
                except ValueError: print(f"W: Invalid core range format '{part}'. Skipping."); continue
            else:
                try: cores.add(int(part))
                except ValueError: print(f"W: Invalid core number format '{part}'. Skipping."); continue
        return sorted(list(cores))

    def _update_ru_core_msr_data(self):
        if not self.ru_timing_core_indices:
            return

        for core_id in self.ru_timing_core_indices:
            if core_id not in self.ru_core_msr_curr_data:
                self.ru_core_msr_curr_data[core_id] = CoreMSRData(core_id)
            
            current_data = self.ru_core_msr_curr_data[core_id]
            
            if self.dry_run:
                # Simulate MSR increments for dry run to get non-zero busy %
                # Use previous values if available, otherwise start from 0
                prev_mperf_sim = self.ru_core_msr_prev_data.get(core_id, CoreMSRData(core_id)).mperf or 0
                prev_tsc_sim = self.ru_core_msr_prev_data.get(core_id, CoreMSRData(core_id)).tsc or 0
                
                # Simulate based on target usage, make it vary a bit
                sim_mperf_inc = int(self.target_ru_cpu_usage * 10000 * (0.9 + (core_id % 3) * 0.05)) # Vary slightly per core
                sim_tsc_inc = 1000000 # Constant TSC increment for simulation
                
                current_data.mperf = prev_mperf_sim + sim_mperf_inc
                current_data.tsc = prev_tsc_sim + sim_tsc_inc
            else:
                current_data.mperf = read_msr_direct(core_id, MSR_IA32_MPERF)
                current_data.tsc = read_msr_direct(core_id, MSR_IA32_TSC)

            # Initialize prev_data entry if it's the very first time for this core
            if core_id not in self.ru_core_msr_prev_data:
                 self.ru_core_msr_prev_data[core_id] = CoreMSRData(core_id)
                 # For the very first sample, we can't calculate delta, so busy_percent remains 0.0
                 # We'll copy current raw MSRs to prev so next iteration can calculate.
                 self.ru_core_msr_prev_data[core_id].mperf = current_data.mperf
                 self.ru_core_msr_prev_data[core_id].tsc = current_data.tsc
                 continue # Skip busy_percent calculation for the very first sample of a core

            prev_data = self.ru_core_msr_prev_data[core_id] # Now this is guaranteed to exist
            
            if prev_data.mperf is not None and prev_data.tsc is not None and \
               current_data.mperf is not None and current_data.tsc is not None:
                
                delta_mperf = current_data.mperf - prev_data.mperf
                if delta_mperf < 0: delta_mperf += (2**64) 
                
                delta_tsc = current_data.tsc - prev_data.tsc
                if delta_tsc < 0: delta_tsc += (2**64)

                if delta_tsc > 0:
                    current_data.busy_percent = min(100.0, 100.0 * delta_mperf / delta_tsc)
                else: # Should be rare; TSC should always increase.
                    current_data.busy_percent = prev_data.busy_percent # Keep previous if delta_tsc is bad
            else: # One of the MSR reads failed for current or previous
                current_data.busy_percent = prev_data.busy_percent # Try to keep previous valid data
            
            # Current becomes previous for next iteration
            self.ru_core_msr_prev_data[core_id].mperf = current_data.mperf
            self.ru_core_msr_prev_data[core_id].tsc = current_data.tsc
            self.ru_core_msr_prev_data[core_id].busy_percent = current_data.busy_percent

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        
        current_max_busy_percent = 0.0
        found_valid_core_data = False
        for core_id in self.ru_timing_core_indices:
            # prev_data now holds the latest calculated busy_percent from _update_ru_core_msr_data
            data_point = self.ru_core_msr_prev_data.get(core_id) 
            if data_point:
                current_max_busy_percent = max(current_max_busy_percent, data_point.busy_percent)
                found_valid_core_data = True
        
        if not found_valid_core_data:
            # If it's dry_run, and we just started, simulate something.
            # Otherwise (live run), if no data, means MSR reads failed or it's literally the first pass.
            if self.dry_run and not self.max_ru_timing_usage_history: # And history is empty (first call)
                return self.target_ru_cpu_usage - (self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor * 1.5)
            return 0.0 # No valid data to control upon

        self.max_ru_timing_usage_history.append(current_max_busy_percent)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg:
            self.max_ru_timing_usage_history.pop(0)
        
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str], use_sudo_for_tee: bool = False) -> None:
        actual_cmd_list_or_str: Any; shell_needed = False; print_cmd_str: str
        if cmd_list[0] == "intel-speed-select": 
            actual_cmd_list_or_str = [self.intel_sst_path] + cmd_list[1:]
            print_cmd_str = ' '.join(actual_cmd_list_or_str)
        elif use_sudo_for_tee and cmd_list[0] == 'echo' and len(cmd_list) == 3:
            val_to_echo, target_file = cmd_list[1], cmd_list[2]
            if not (all(c.isalnum() or c in ['-', '_', '.', '/', ':'] for c in target_file) and val_to_echo.isdigit()): # Allow ':' for intel-rapl:X
                err_msg = f"Invalid chars in echo/tee: echo {val_to_echo} | sudo tee {target_file}"; print(f"E: {err_msg}");
                if not self.dry_run: raise ValueError(err_msg)
                return
            actual_cmd_list_or_str = f"echo {val_to_echo} | sudo tee {target_file}"; shell_needed = True; print_cmd_str = actual_cmd_list_or_str
        else: 
            actual_cmd_list_or_str = cmd_list; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        
        if self.dry_run: print(f"[DRY RUN] Would execute: {print_cmd_str}"); return
        
        print(f"Executing: {print_cmd_str}")
        try: 
            subprocess.run(actual_cmd_list_or_str, shell=shell_needed, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            cmd_executed = e.cmd if isinstance(e.cmd, str) else ' '.join(e.cmd)
            print(f"E: Cmd '{cmd_executed}' failed ({e.returncode}). STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}");
            if not self.dry_run: raise
        except FileNotFoundError:
            cmd_name = actual_cmd_list_or_str[0] if isinstance(actual_cmd_list_or_str, list) else actual_cmd_list_or_str.split()[0]
            print(f"E: Cmd '{cmd_name}' not found.");
            if not self.dry_run: raise

    def _setup_intel_sst(self):
        print("--- Configuring Intel SST-CP ---");
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])

            clos_min_freqs = self.config.get('clos_min_frequency', {})
            for clos_id, min_freq in clos_min_freqs.items():
                self._run_command(["intel-speed-select", "core-power", "config", "-c", str(clos_id), "--min", str(min_freq)])

            ran_component_cores: Dict[str, List[int]] = {
                name: self._parse_core_list_string(str(core_str)) # Ensure core_str is string
                for name, core_str in self.config.get('ran_cores', {}).items()
            }

            clos_associations = self.config.get('clos_association', {})
            processed_clos_associations = {int(k): v for k, v in clos_associations.items()} 

            for clos_id_int, ran_components in processed_clos_associations.items():
                associated_cores: Set[int] = set()
                if ran_components: # Ensure ran_components is not None
                    for comp_name in ran_components:
                        if comp_name in ran_component_cores:
                            associated_cores.update(ran_component_cores[comp_name])
                        else:
                            print(f"W: RAN comp '{comp_name}' for CLOS {clos_id_int} not in 'ran_cores'.")
                
                if clos_id_int == 0 and self.ru_timing_core_indices:
                    if not associated_cores.issuperset(self.ru_timing_core_indices): 
                         print(f"INFO: Ensuring RU_Timing cores {self.ru_timing_core_indices} are in CLOS 0.")
                    associated_cores.update(self.ru_timing_core_indices)
                
                if associated_cores:
                    core_list_str = ",".join(map(str, sorted(list(associated_cores))))
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                elif clos_id_int == 0 and self.ru_timing_core_indices and (not ran_components or not any(ran_components)): # Only RU if no other components
                    core_list_str = ",".join(map(str, sorted(list(self.ru_timing_core_indices))))
                    print(f"INFO: Assigning only RU_Timing cores {core_list_str} to CLOS 0.")
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                else:
                    if not (clos_id_int == 0 and self.ru_timing_core_indices): # Avoid warning if CLOS 0 is just RU cores
                        print(f"W: No cores to associate with CLOS {clos_id_int}.")
            
            print("--- Intel SST-CP Configuration Complete ---")
        except Exception as e: 
            print(f"An error during Intel SST-CP setup: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for SST setup issues
            if not self.dry_run: 
                print("E: Halting due to critical SST-CP failure.")
                sys.exit(1)
            else: 
                print("[DRY RUN] SST-CP setup would have failed.")

    def _read_current_tdp_limit_w(self) -> float:
        # In dry_run, after the first simulated set, this reflects the simulated TDP.
        # On first call (last_tdp_adjustment_time is 0), or if not dry_run, read from system.
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
            if abs(self.current_tdp_w - new_tdp_w) > 0.01 : # Only print if change is significant
                 print(f"[DRY RUN] Would set TDP to {new_tdp_w:.1f}W (requested {tdp_watts:.1f}W, current sim {self.current_tdp_w:.1f}W)")
            self.current_tdp_w = new_tdp_w # Update simulated TDP
            return

        # Actual run:
        try: # Check if already at target to avoid unnecessary writes
            with open(self.power_limit_uw_file, 'r') as f:
                if int(f.read().strip()) == clamped_tdp_uw:
                    if abs(self.current_tdp_w - new_tdp_w) > 0.01: # Update our state if it drifted
                        self.current_tdp_w = new_tdp_w
                    return 
        except Exception: pass # If read fails, proceed to write

        try: 
            self._run_command(["echo",str(clamped_tdp_uw),self.power_limit_uw_file],use_sudo_for_tee=True)
            self.current_tdp_w = new_tdp_w
        except Exception as e: 
            # _run_command now prints errors, re-raise if not dry_run handled there.
            # This specific error might be if tee fails after echo.
            print(f"E: Exception during _set_tdp_limit_w after _run_command attempt: {e}")
            if not self.dry_run: raise

    def _adjust_tdp(self, control_ru_cpu_usage: float):
        error_percent = self.target_ru_cpu_usage - control_ru_cpu_usage 
        abs_error_percent = abs(error_percent) 
        
        sensitivity_abs_threshold_percent = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_abs_threshold_percent = sensitivity_abs_threshold_percent * self.adaptive_step_far_thresh_factor
        
        chosen_step_w = 0.0
        if abs_error_percent > sensitivity_abs_threshold_percent: # Outside the deadband
            chosen_step_w = self.tdp_adj_step_w_large if abs_error_percent > far_abs_threshold_percent else self.tdp_adj_step_w_small
            
            tdp_change_w = -chosen_step_w if error_percent > 0 else chosen_step_w # error > 0 means MAX usage < target -> decrease TDP
            
            if tdp_change_w != 0:
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
                if delta_t > 0.001: # Ensure time has passed to avoid division by zero or huge spikes
                    delta_e = current_energy_uj - self.last_pkg_energy_uj
                    if delta_e < 0 : delta_e += (2**32) # RAPL energy counters are often 32-bit and wrap
                    power_w = (delta_e / 1e6) / delta_t
                    success = True
            self.last_pkg_energy_uj, self.last_energy_read_time = current_energy_uj, current_time
            return power_w, success
        except Exception as e:
            # print(f"W: Could not read package energy: {e}", file=sys.stderr) # Can be noisy
            return 0.0, False
        
    def run_monitor(self):
        if os.geteuid() != 0 and not self.dry_run:
            print("E: Script must be run as root for MSR and RAPL access (unless in dry_run mode).")
            sys.exit(1)
        
        # Prime MSR readings for the first valid busy % calculation
        if not self.dry_run and self.ru_timing_core_indices:
            print("Priming MSR readings for initial busy % calculation...")
            self._update_ru_core_msr_data() # Populates prev_data with first raw MSRs, busy_percent still 0
            time.sleep(0.2) # Wait for MSRs to change for a good delta
            self._update_ru_core_msr_data() # Now busy_percent should be calculated based on the delta
            print("MSR readings primed. Initial busy % values calculated.")
        elif self.dry_run and self.ru_timing_core_indices:
            print("[DRY RUN] Simulating MSR priming.")
            self._update_ru_core_msr_data() # Simulate first set of MSRs
            self._update_ru_core_msr_data() # Simulate second to get a busy %

        # Initial TDP setting based on current PkgWatt
        if not self.dry_run:
            print("Attempting to set initial TDP based on current PkgWatt...")
            initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() 
            time.sleep(0.2) # Allow energy counter to update for a better delta
            initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() 
            
            if pkg_power_ok and initial_pkg_power > 1.0 : # Ensure PkgWatt is somewhat reasonable
                safe_initial_tdp = max(self.tdp_min_w, min(initial_pkg_power, self.tdp_max_w))
                print(f"Initial PkgWatt measured: {initial_pkg_power:.1f}W. Clamped safe initial TDP: {safe_initial_tdp:.1f}W.")
                self._set_tdp_limit_w(safe_initial_tdp)
            else:
                print(f"W: Could not get valid initial PkgWatt (got: {initial_pkg_power if pkg_power_ok else 'N/A'}). Reading current TDP from RAPL or using config min_tdp.")
                self.current_tdp_w = self._read_current_tdp_limit_w() # Read from RAPL
                self._set_tdp_limit_w(self.current_tdp_w) # Apply (and clamp)
            print(f"Effective Initial TDP after setup: {self.current_tdp_w:.1f}W.")
        else:
            # In dry_run, current_tdp_w is self.tdp_min_w from __init__
            print(f"[DRY RUN] Initial simulated TDP: {self.current_tdp_w:.1f}W.")

        self._setup_intel_sst() # Configure SST-CP
        
        self.last_tdp_adjustment_time = time.monotonic() # Initialize for TDP update interval logic
        last_print_time = time.monotonic()

        print(f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        print(f"Target MAX RU CPU (MSR-based): {self.target_ru_cpu_usage}% | RU Cores Monitored: {self.ru_timing_core_indices if self.ru_timing_core_indices else 'NONE'}")
        print(f"TDP Update Interval: {self.tdp_update_interval_s}s | Print Interval: {self.print_interval_s}s")
        print(f"TDP Range: {self.tdp_min_w}W - {self.tdp_max_w}W")
        # ... other startup info

        try:
            while True:
                loop_start_time = time.monotonic()
                
                if self.ru_timing_core_indices: 
                    self._update_ru_core_msr_data() # Update MSRs and calculate busy% for each RU core

                control_value_for_tdp = self._get_control_ru_timing_cpu_usage() # Get (smoothed) max busy%
                
                current_time = time.monotonic()
                # Check if it's time to adjust TDP
                if current_time - self.last_tdp_adjustment_time >= self.tdp_update_interval_s:
                    self._adjust_tdp(control_value_for_tdp)
                    self.last_tdp_adjustment_time = current_time
                
                # Logging
                if current_time - last_print_time >= self.print_interval_s:
                    pkg_power_w, pkg_power_ok = self._get_pkg_power_w()
                    
                    ru_core_usage_details_list = []
                    for core_idx in self.ru_timing_core_indices:
                        # prev_data holds the latest calculated busy% from the most recent _update_ru_core_msr_data call
                        data = self.ru_core_msr_prev_data.get(core_idx) 
                        usage_str = f"{data.busy_percent:>5.1f}%" if data else "N/A"
                        ru_core_usage_details_list.append(f"C{core_idx}:{usage_str}")
                    ru_core_details_str = ", ".join(ru_core_usage_details_list) if ru_core_usage_details_list else "N/A"

                    log_msg = (
                        f"{time.strftime('%H:%M:%S')} | "
                        f"RU_Cores(MSR): [{ru_core_details_str}] (S_MaxCtrl:{control_value_for_tdp:>5.1f}%) | "
                        f"TDP:{self.current_tdp_w:>5.1f}W | "
                        f"PkgPwr: {pkg_power_w if pkg_power_ok else 'N/A':>6s}W"
                    )
                    print(log_msg)
                    last_print_time = current_time

                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, 1.0 - loop_duration) # Aim for roughly 1s main loop for data collection
                time.sleep(sleep_time)

        except KeyboardInterrupt: print(f"\n--- Monitoring stopped by user ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        except Exception as e: 
            print(f"\n--- An unexpected error occurred in main loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}): {e} ---")
            import traceback
            traceback.print_exc()
        finally: 
            print("Exiting application.")

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: sudo python3 ecoran.py <path_to_config.yaml>")
        sys.exit(1)
    
    config_file_path = sys.argv[1]
    manager = PowerManager(config_path=config_file_path)
    manager.run_monitor()
