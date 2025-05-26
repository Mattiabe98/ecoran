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
                # This case should ideally not happen if seek and read are successful on a valid MSR
                print(f"W: Short read ({len(msr_val_bytes)} bytes) from MSR {hex(reg)} on CPU {cpu_id}", file=sys.stderr)
                return None
    except FileNotFoundError:
        # This likely means the msr module is not loaded, or the cpu_id is invalid.
        # Only print once or a few times to avoid flooding logs if it's a persistent issue.
        # For now, let _validate_config handle initial check.
        # print(f"W: MSR device file not found for CPU {cpu_id}. Is 'msr' module loaded?", file=sys.stderr)
        return None
    except PermissionError:
        # This means the script doesn't have root privileges.
        # print(f"W: Permission denied reading MSR {hex(reg)} on CPU {cpu_id}. Run as root.", file=sys.stderr)
        return None
    except OSError as e:
        # Catch other OS-level errors (e.g., I/O error if MSR is invalid for seek/read)
        # EIO (Input/output error) can happen if trying to read an MSR that doesn't exist or is protected.
        if e.errno != 2 and e.errno != 13: # Avoid re-printing FileNotFoundError or PermissionError messages
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

        self.print_interval_s = int(self.config['print_interval'])
        self.tdp_min_w = int(self.config['tdp_range']['min'])
        self.tdp_max_w = int(self.config['tdp_range']['max'])
        self.target_ru_cpu_usage = float(self.config['target_ru_timing_cpu_usage'])
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_update_interval_s = int(self.config.get('tdp_update_interval_s', 1))

        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.05))
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
        if self.dry_run: print("!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")

    def _validate_config(self):
        if not self.dry_run:
            if not os.path.exists(self.rapl_base_path) or \
               not os.path.exists(self.power_limit_uw_file):
                print(f"E: RAPL path {self.rapl_base_path} or power limit file missing.")
                sys.exit(1)
            if not os.path.exists(self.energy_uj_file):
                 print(f"W: Energy file {self.energy_uj_file} not found. PkgPower will be N/A.")

            # Validate MSR access
            if not self.ru_timing_core_indices and not self.config.get('ru_timing_cores'):
                print("INFO: No 'ru_timing_cores' defined. MSR-based CPU utilization will not be used.")
            elif self.ru_timing_core_indices: # Only test if RU cores are defined
                test_core = self.ru_timing_core_indices[0]
                msr_path_test = f'/dev/cpu/{test_core}/msr'
                if not os.path.exists(msr_path_test):
                    print(f"E: MSR device file {msr_path_test} not found. Is 'msr' kernel module loaded? (`sudo modprobe msr`)")
                    sys.exit(1)
                if read_msr_direct(test_core, MSR_IA32_TSC) is None: # Try reading TSC
                    # read_msr_direct prints its own warnings for permission etc.
                    print(f"E: Failed initial MSR read on core {test_core}. Check permissions (run as root?) or 'msr' module state.")
                    sys.exit(1)
                print("INFO: MSR access test passed.")
            else: # ru_timing_cores is empty string or not present in config
                 print("INFO: 'ru_timing_cores' is empty or not defined. MSR-based CPU utilization will not be used.")


        try: # Check intel-speed-select presence
            subprocess.run([self.intel_sst_path, "--version"], capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"E: '{self.intel_sst_path}' command failed or not found: {e}")
            sys.exit(1)
        
        # ... (other parameter validations like tdp_update_interval_s > 0 etc.)
        if self.tdp_update_interval_s <= 0: print(f"E: 'tdp_update_interval_s' must be positive."); sys.exit(1)
        if not (0 < self.tdp_adj_sensitivity_factor < 1): print(f"E: 'tdp_adjustment_sensitivity' must be > 0 and < 1."); sys.exit(1)
        # ...

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
                try: s,e = map(int,part.split('-',1)); cores.update(range(s,e+1))
                except ValueError: print(f"W: Invalid core range format '{part}'"); continue
            else:
                try: cores.add(int(part))
                except ValueError: print(f"W: Invalid core number format '{part}'"); continue
        return sorted(list(cores))

    def _update_ru_core_msr_data(self):
        """Updates MPERF and TSC MSR readings for all RU_Timing cores using direct read."""
        if not self.ru_timing_core_indices:
            return

        for core_id in self.ru_timing_core_indices:
            if core_id not in self.ru_core_msr_curr_data:
                self.ru_core_msr_curr_data[core_id] = CoreMSRData(core_id)
            
            current_data = self.ru_core_msr_curr_data[core_id]
            
            if self.dry_run: # Simulate MSR reads in dry run
                # Increment fake MSRs to get some non-zero busy %
                prev_mperf = self.ru_core_msr_prev_data.get(core_id, CoreMSRData(core_id)).mperf or 0
                prev_tsc = self.ru_core_msr_prev_data.get(core_id, CoreMSRData(core_id)).tsc or 0
                current_data.mperf = prev_mperf + int(self.target_ru_cpu_usage * 1000) # Simulate based on target
                current_data.tsc = prev_tsc + 100000 # Simulate some TSC increment
            else:
                current_data.mperf = read_msr_direct(core_id, MSR_IA32_MPERF)
                current_data.tsc = read_msr_direct(core_id, MSR_IA32_TSC)

            if core_id in self.ru_core_msr_prev_data:
                prev_data = self.ru_core_msr_prev_data[core_id]
                if prev_data.mperf is not None and prev_data.tsc is not None and \
                   current_data.mperf is not None and current_data.tsc is not None:
                    
                    delta_mperf = current_data.mperf - prev_data.mperf
                    if delta_mperf < 0: delta_mperf += (2**64) 
                    
                    delta_tsc = current_data.tsc - prev_data.tsc
                    if delta_tsc < 0: delta_tsc += (2**64)

                    if delta_tsc > 0:
                        current_data.busy_percent = min(100.0, 100.0 * delta_mperf / delta_tsc) # Cap at 100%
                    else: # Should not happen if TSC is always incrementing and no error
                        current_data.busy_percent = prev_data.busy_percent # Keep previous if delta_tsc is bad
                # else: MSR read failed, busy_percent remains from previous or 0.0
            
            # Prepare for next iteration
            if core_id not in self.ru_core_msr_prev_data: # First time init
                 self.ru_core_msr_prev_data[core_id] = CoreMSRData(core_id)

            self.ru_core_msr_prev_data[core_id].mperf = current_data.mperf
            self.ru_core_msr_prev_data[core_id].tsc = current_data.tsc
            self.ru_core_msr_prev_data[core_id].busy_percent = current_data.busy_percent

    def _get_control_ru_timing_cpu_usage(self) -> float:
        if not self.ru_timing_core_indices: return 0.0
        
        current_max_busy_percent = 0.0
        found_valid_core_data = False
        for core_id in self.ru_timing_core_indices:
            data_point = self.ru_core_msr_prev_data.get(core_id) # prev_data has latest calculated busy%
            if data_point:
                current_max_busy_percent = max(current_max_busy_percent, data_point.busy_percent)
                found_valid_core_data = True
        
        if not found_valid_core_data and not self.dry_run: return 0.0 
        if self.dry_run and not found_valid_core_data :
            current_max_busy_percent = self.target_ru_cpu_usage - (self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor * 1.5) # Simulate slightly below target

        self.max_ru_timing_usage_history.append(current_max_busy_percent)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg:
            self.max_ru_timing_usage_history.pop(0)
        
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history) if self.max_ru_timing_usage_history else 0.0

    def _run_command(self, cmd_list: List[str], use_sudo_for_tee: bool = False) -> None:
        # ... (This function remains mostly the same for intel-speed-select and echo|tee) ...
        actual_cmd_list_or_str: Any; shell_needed = False; print_cmd_str: str
        if cmd_list[0] == "intel-speed-select": actual_cmd_list_or_str = [self.intel_sst_path] + cmd_list[1:]; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        elif use_sudo_for_tee and cmd_list[0] == 'echo' and len(cmd_list) == 3:
            val_to_echo, target_file = cmd_list[1], cmd_list[2]
            if not (all(c.isalnum() or c in ['-', '_', '.', '/'] for c in target_file) and val_to_echo.isdigit()):
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
            # ... (rest of SST setup from previous versions) ...
            clos_min_freqs = self.config.get('clos_min_frequency', {}) # Example
            for clos_id, min_freq in clos_min_freqs.items(): self._run_command(["intel-speed-select", "core-power", "config", "-c", str(clos_id), "--min", str(min_freq)])
            # ...
            print("--- Intel SST-CP Configuration Complete ---")
        except Exception as e: print(f"An error during Intel SST-CP setup: {e}");
            if not self.dry_run: print("E: Halting due to SST-CP failure."); sys.exit(1)
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
            if abs(self.current_tdp_w - new_tdp_w) > 0.01 : print(f"[DRY RUN] Would set TDP to {new_tdp_w:.1f}W (req {tdp_watts:.1f}W)")
            self.current_tdp_w = new_tdp_w; return
        try: 
            with open(self.power_limit_uw_file, 'r') as f:
                if int(f.read().strip()) == clamped_tdp_uw:
                    if abs(self.current_tdp_w - new_tdp_w) > 0.01: self.current_tdp_w = new_tdp_w
                    return 
        except Exception: pass 
        try: self._run_command(["echo",str(clamped_tdp_uw),self.power_limit_uw_file],use_sudo_for_tee=True); self.current_tdp_w = new_tdp_w
        except Exception as e: 
            if not self.dry_run: raise e

    def _adjust_tdp(self, control_ru_cpu_usage: float): # ... (same)
        error_percent = self.target_ru_cpu_usage - control_ru_cpu_usage; abs_error_percent = abs(error_percent) 
        sensitivity_abs_threshold_percent = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_abs_threshold_percent = sensitivity_abs_threshold_percent * self.adaptive_step_far_thresh_factor
        chosen_step_w = 0.0
        if abs_error_percent > sensitivity_abs_threshold_percent:
            chosen_step_w = self.tdp_adj_step_w_large if abs_error_percent > far_abs_threshold_percent else self.tdp_adj_step_w_small
            tdp_change_w = -chosen_step_w if error_percent > 0 else chosen_step_w
            if tdp_change_w != 0: self._set_tdp_limit_w(self.current_tdp_w + tdp_change_w)

    def _get_pkg_power_w(self) -> Tuple[float, bool]: # ... (same)
        if not os.path.exists(self.energy_uj_file): return 0.0, False
        try:
            with open(self.energy_uj_file, 'r') as f: current_energy_uj = int(f.read().strip())
            current_time = time.monotonic(); power_w, success = 0.0, False
            if self.last_pkg_energy_uj is not None and self.last_energy_read_time is not None:
                delta_t = current_time - self.last_energy_read_time
                if delta_t > 0.001: power_w = ((current_energy_uj - self.last_pkg_energy_uj) / 1e6) / delta_t; success = True
            self.last_pkg_energy_uj, self.last_energy_read_time = current_energy_uj, current_time
            return power_w, success
        except Exception: return 0.0, False
        
    def run_monitor(self):
        if os.geteuid() != 0 and not self.dry_run:
            print("E: Script must be run as root for MSR and RAPL access (unless in dry_run mode).")
            sys.exit(1)
        
        if not self.dry_run and self.ru_timing_core_indices:
            print("Priming MSR readings for initial busy % calculation...")
            self._update_ru_core_msr_data() # First sample
            time.sleep(0.2) # Increased sleep for a more stable first delta
            self._update_ru_core_msr_data() # Second sample, now busy% is calculated
            print("MSR readings primed.")
        elif self.dry_run:
            print("[DRY RUN] Skipping MSR priming.")

        # Initial TDP setting based on current PkgWatt
        if not self.dry_run:
            initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() # First read
            time.sleep(0.2) 
            initial_pkg_power, pkg_power_ok = self._get_pkg_power_w() # Second read
            
            if pkg_power_ok and initial_pkg_power > 0:
                safe_initial_tdp = max(self.tdp_min_w, min(initial_pkg_power, self.tdp_max_w))
                print(f"Initial PkgWatt: {initial_pkg_power:.1f}W. Setting initial TDP to {safe_initial_tdp:.1f}W.")
                self._set_tdp_limit_w(safe_initial_tdp)
            else:
                print(f"W: Could not get valid initial PkgWatt. Reading current TDP from RAPL or using min_tdp.")
                self.current_tdp_w = self._read_current_tdp_limit_w()
                self._set_tdp_limit_w(self.current_tdp_w) # Apply to ensure it's clamped
            print(f"Effective Initial TDP: {self.current_tdp_w:.1f}W.")
        else:
            # In dry_run, current_tdp_w is self.tdp_min_w from __init__
            print(f"[DRY RUN] Initial simulated TDP: {self.current_tdp_w:.1f}W.")

        self._setup_intel_sst() 
        
        self.last_tdp_adjustment_time = time.monotonic()
        last_print_time = time.monotonic()

        print(f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        # ... (startup log messages) ...

        try:
            while True:
                loop_start_time = time.monotonic()
                
                if self.ru_timing_core_indices: # Only update MSR if RU cores are defined
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
                # Main loop aims for ~1s. MSR reads take time.
                # No psutil call now, so sleep time might need adjustment based on MSR read speed.
                # Assuming direct MSR reads are faster than subprocess calls.
                sleep_time = max(0, 1.0 - loop_duration) 
                time.sleep(sleep_time)

        except KeyboardInterrupt: print(f"\n--- Monitoring stopped ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        except Exception as e: print(f"\n--- An unexpected error: {e} ---"); import traceback; traceback.print_exc()
        finally: print("Exiting application.")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: sudo python3 ecoran.py <config.yaml>"); sys.exit(1)
    manager = PowerManager(config_path=sys.argv[1]); manager.run_monitor()
