import yaml
import psutil
import subprocess
import time
import os
import sys
from typing import List, Dict, Any, Set, Tuple

class PowerManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

        # Paths
        self.intel_sst_path = self.config.get('intel_speed_select_path', 'intel-speed-select')
        self.rapl_base_path = self.config.get('rapl_path_base', '/sys/class/powercap/intel-rapl:0')
        self.power_limit_uw_file = os.path.join(self.rapl_base_path, "constraint_0_power_limit_uw")
        self.energy_uj_file = os.path.join(self.rapl_base_path, "energy_uj")

        # Core settings
        self.print_interval_s = int(self.config['print_interval'])
        self.tdp_min_w = int(self.config['tdp_range']['min'])
        self.tdp_max_w = int(self.config['tdp_range']['max'])
        self.target_ru_cpu_usage = float(self.config['target_ru_timing_cpu_usage']) # Target for the MAX RU core
        self.ru_timing_core_indices = self._parse_core_list_string(self.config.get('ru_timing_cores', ""))
        self.tdp_update_interval_s = int(self.config.get('tdp_update_interval_s', 1))

        # TDP Adjustment Parameters
        self.tdp_adj_sensitivity_factor = float(self.config.get('tdp_adjustment_sensitivity', 0.05))
        self.tdp_adj_step_w_small = float(self.config.get('tdp_adjustment_step_w_small', 1))
        self.tdp_adj_step_w_large = float(self.config.get('tdp_adjustment_step_w_large', 5))
        self.adaptive_step_far_thresh_factor = float(self.config.get('adaptive_step_far_threshold_factor', 2.0))
        self.max_samples_cpu_avg = int(self.config.get('max_cpu_usage_samples', 3)) # Now for MAX usage history
        
        # Operational Settings
        self.dry_run = bool(self.config.get('dry_run', False))

        # Runtime state
        self.current_tdp_w = self.tdp_min_w 
        self.last_pkg_energy_uj = None
        self.last_energy_read_time = None
        self.max_ru_timing_usage_history = [] # History of MAX RU core usage for TDP control
        self.last_tdp_adjustment_time = 0.0

        self._validate_config()
        if self.dry_run:
            print("!!!!!!!!!!!! DRY RUN MODE ENABLED !!!!!!!!!!!!")
            # ... (rest of dry run message)

    def _validate_config(self):
        # ... (previous validations for paths, tools, tdp_update_interval_s) ...
        # Parameter validations (ensure they make sense for MAX core control)
        if not (0 < self.tdp_adj_sensitivity_factor < 1):
            print(f"ERROR: 'tdp_adjustment_sensitivity' must be > 0 and < 1. Found: {self.tdp_adj_sensitivity_factor}")
            sys.exit(1)
        if self.tdp_adj_step_w_small <= 0:
            print(f"ERROR: 'tdp_adjustment_step_w_small' must be positive. Found: {self.tdp_adj_step_w_small}")
            sys.exit(1)
        if self.tdp_adj_step_w_large <= 0:
            print(f"ERROR: 'tdp_adjustment_step_w_large' must be positive. Found: {self.tdp_adj_step_w_large}")
            sys.exit(1)
        # Consider if large < small warning is still relevant, it might be for this strategy
        if self.adaptive_step_far_thresh_factor <= 1.0:
            print(f"ERROR: 'adaptive_step_far_threshold_factor' must be > 1.0. Found: {self.adaptive_step_far_thresh_factor}")
            sys.exit(1)
        if self.max_samples_cpu_avg <= 0: # Renamed in spirit from avg to max but var name kept for less churn
            print(f"ERROR: 'max_cpu_usage_samples' must be positive. Found: {self.max_samples_cpu_avg}")
            sys.exit(1)
        print("Configuration and system checks passed.")

    def _load_config(self) -> Dict[str, Any]:
        # ... (same)
        try:
            with open(self.config_path, 'r') as f: return yaml.safe_load(f)
        except FileNotFoundError: print(f"E: Config '{self.config_path}' not found."); sys.exit(1)
        except yaml.YAMLError as e: print(f"E: Parse config '{self.config_path}': {e}"); sys.exit(1)


    def _parse_core_list_string(self, core_str: str) -> List[int]:
        # ... (same)
        cores: Set[int] = set()
        if not core_str: return []
        parts = core_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                try:
                    start, end = int(start_str), int(end_str)
                    if start > end: print(f"W: Invalid core range '{part}'."); continue
                    cores.update(range(start, end + 1))
                except ValueError: print(f"W: Invalid num in range '{part}'.")
            else:
                try: cores.add(int(part))
                except ValueError: print(f"W: Invalid core num '{part}'.")
        return sorted(list(cores))

    def _run_command(self, cmd_list: List[str], use_sudo_for_tee: bool = False) -> None:
        # ... (same, respects self.dry_run)
        actual_cmd_list_or_str: Any; shell_needed = False; print_cmd_str: str
        if cmd_list[0] == "intel-speed-select":
            actual_cmd_list_or_str = [self.intel_sst_path] + cmd_list[1:]; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        elif use_sudo_for_tee and cmd_list[0] == 'echo' and len(cmd_list) == 3:
            val_to_echo, target_file = cmd_list[1], cmd_list[2]
            if not (all(c.isalnum() or c in ['-', '_', '.', '/'] for c in target_file) and val_to_echo.isdigit()):
                err_msg = f"Invalid chars in echo/tee: echo {val_to_echo} | sudo tee {target_file}"
                print(f"E: {err_msg}");
                if not self.dry_run: raise ValueError(err_msg)
                return
            actual_cmd_list_or_str = f"echo {val_to_echo} | sudo tee {target_file}"; shell_needed = True; print_cmd_str = actual_cmd_list_or_str
        else: actual_cmd_list_or_str = cmd_list; print_cmd_str = ' '.join(actual_cmd_list_or_str)
        if self.dry_run: print(f"[DRY RUN] Would execute: {print_cmd_str}"); return
        print(f"Executing: {print_cmd_str}")
        try: subprocess.run(actual_cmd_list_or_str, shell=shell_needed, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            cmd_executed = e.cmd if isinstance(e.cmd, str) else ' '.join(e.cmd)
            print(f"E: Cmd '{cmd_executed}' failed ({e.returncode}). STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}");
            if not self.dry_run: raise
        except FileNotFoundError:
            cmd_name = actual_cmd_list_or_str[0] if isinstance(actual_cmd_list_or_str, list) else actual_cmd_list_or_str.split()[0]
            print(f"E: Cmd '{cmd_name}' not found.");
            if not self.dry_run: raise


    def _setup_intel_sst(self):
        # ... (same, respects self.dry_run)
        print("--- Configuring Intel SST-CP ---")
        try:
            self._run_command(["intel-speed-select", "core-power", "enable"])
            clos_min_freqs = self.config.get('clos_min_frequency', {})
            for clos_id, min_freq in clos_min_freqs.items(): self._run_command(["intel-speed-select", "core-power", "config", "-c", str(clos_id), "--min", str(min_freq)])
            ran_component_cores: Dict[str, List[int]] = { name: self._parse_core_list_string(cs) for name, cs in self.config.get('ran_cores', {}).items() }
            clos_associations = self.config.get('clos_association', {}); processed_clos_associations = {int(k): v for k,v in clos_associations.items()}
            for clos_id_int, ran_components in processed_clos_associations.items():
                associated_cores: Set[int] = set()
                for comp_name in ran_components:
                    if comp_name in ran_component_cores: associated_cores.update(ran_component_cores[comp_name])
                    else: print(f"W: RAN comp '{comp_name}' for CLOS {clos_id_int} not in 'ran_cores'.")
                if clos_id_int == 0 and self.ru_timing_core_indices:
                    if not associated_cores.issuperset(self.ru_timing_core_indices): print(f"INFO: Ensuring RU_Timing cores {self.ru_timing_core_indices} are in CLOS 0.")
                    associated_cores.update(self.ru_timing_core_indices)
                if associated_cores:
                    core_list_str = ",".join(map(str, sorted(list(associated_cores))))
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                elif clos_id_int == 0 and self.ru_timing_core_indices and not ran_components:
                    core_list_str = ",".join(map(str, sorted(list(self.ru_timing_core_indices))))
                    print(f"INFO: Assigning only RU_Timing cores {core_list_str} to CLOS 0.")
                    self._run_command(["intel-speed-select", "-c", core_list_str, "core-power", "assoc", "-c", str(clos_id_int)])
                else: print(f"W: No cores to associate with CLOS {clos_id_int}.")
            print("--- Intel SST-CP Configuration Complete ---")
        except Exception as e: print(f"An error during Intel SST-CP setup: {e}");
            if not self.dry_run: sys.exit(1)
            else: print("[DRY RUN] SST-CP setup would have failed.")


    def _get_cpu_usages(self) -> List[float]:
        return psutil.cpu_percent(interval=0.1, percpu=True) 

    def _get_control_ru_timing_cpu_usage(self, all_core_usages: List[float]) -> float:
        """
        Determines the RU_Timing CPU usage value to use for TDP control.
        This is now the MAXIMUM utilization among ru_timing_cores, smoothed over history.
        Returns 0.0 if no RU_Timing cores are defined or found.
        """
        if not self.ru_timing_core_indices:
            return 0.0
        
        current_max_usage = 0.0
        found_valid_core = False
        for core_idx in self.ru_timing_core_indices:
            if 0 <= core_idx < len(all_core_usages):
                current_max_usage = max(current_max_usage, all_core_usages[core_idx])
                found_valid_core = True
            # else: Silently ignore out-of-bounds, psutil should give all cores.
        
        if not found_valid_core:
            return 0.0 # No valid RU_Timing core usages found in this sample

        self.max_ru_timing_usage_history.append(current_max_usage)
        if len(self.max_ru_timing_usage_history) > self.max_samples_cpu_avg: # max_samples_cpu_avg is now for max history
            self.max_ru_timing_usage_history.pop(0)
        
        if not self.max_ru_timing_usage_history:
            return 0.0
        
        # Return the average of the recent maximums (smoothed maximum)
        return sum(self.max_ru_timing_usage_history) / len(self.max_ru_timing_usage_history)

    def _read_current_tdp_limit_w(self) -> float:
        # ... (same)
        if self.dry_run and self.last_tdp_adjustment_time > 0: return self.current_tdp_w 
        try:
            with open(self.power_limit_uw_file, 'r') as f: return int(f.read().strip()) / 1_000_000
        except Exception: print(f"W: Could not read {self.power_limit_uw_file}. Assuming current TDP is min_tdp ({self.tdp_min_w}W)."); return self.tdp_min_w

    def _set_tdp_limit_w(self, tdp_watts: float):
        # ... (same)
        target_tdp_uw = int(tdp_watts * 1_000_000); min_tdp_uw_config = int(self.tdp_min_w * 1_000_000); max_tdp_uw_config = int(self.tdp_max_w * 1_000_000)
        clamped_tdp_uw = max(min_tdp_uw_config, min(target_tdp_uw, max_tdp_uw_config)); new_tdp_w = clamped_tdp_uw / 1_000_000
        if self.dry_run:
            if abs(self.current_tdp_w - new_tdp_w) > 0.01 : print(f"[DRY RUN] Would set TDP to {new_tdp_w:.1f}W (req {tdp_watts:.1f}W)")
            self.current_tdp_w = new_tdp_w; return
        try: 
            with open(self.power_limit_uw_file, 'r') as f:
                if int(f.read().strip()) == clamped_tdp_uw:
                    if abs(self.current_tdp_w - new_tdp_w) > 0.01: self.current_tdp_w = new_tdp_w
                    return 
        except Exception: pass 
        try: self._run_command(["echo", str(clamped_tdp_uw), self.power_limit_uw_file], use_sudo_for_tee=True); self.current_tdp_w = new_tdp_w
        except Exception as e: 
            if not self.dry_run: raise e


    def _adjust_tdp(self, control_ru_cpu_usage: float): # control_ru_cpu_usage is now the (smoothed) MAX RU core usage
        error_percent = self.target_ru_cpu_usage - control_ru_cpu_usage # Positive if MAX usage is BELOW target
        abs_error_percent = abs(error_percent) 
        sensitivity_abs_threshold_percent = self.target_ru_cpu_usage * self.tdp_adj_sensitivity_factor
        far_abs_threshold_percent = sensitivity_abs_threshold_percent * self.adaptive_step_far_thresh_factor
        chosen_step_w = 0.0

        if abs_error_percent > sensitivity_abs_threshold_percent: # Outside the deadband
            chosen_step_w = self.tdp_adj_step_w_large if abs_error_percent > far_abs_threshold_percent else self.tdp_adj_step_w_small
            
            # If MAX usage is BELOW target (error_percent > 0), we can try to DECREASE TDP.
            # If MAX usage is ABOVE target (error_percent < 0), we MUST INCREASE TDP.
            tdp_change_w = -chosen_step_w if error_percent > 0 else chosen_step_w
            
            if tdp_change_w != 0:
                new_tdp_w = self.current_tdp_w + tdp_change_w
                self._set_tdp_limit_w(new_tdp_w)

    def _get_pkg_power_w(self) -> Tuple[float, bool]:
        # ... (same)
        if not os.path.exists(self.energy_uj_file): return 0.0, False
        try:
            with open(self.energy_uj_file, 'r') as f: current_energy_uj = int(f.read().strip())
            current_time = time.monotonic(); power_w, success = 0.0, False
            if self.last_pkg_energy_uj is not None and self.last_energy_read_time is not None:
                delta_t = current_time - self.last_energy_read_time
                if delta_t > 0.001: power_w = ((current_energy_uj - self.last_pkg_energy_uj) / 1_000_000) / delta_t; success = True
            self.last_pkg_energy_uj, self.last_energy_read_time = current_energy_uj, current_time
            return power_w, success
        except Exception: return 0.0, False
        
    def run_monitor(self):
        # ... (startup messages and initial TDP setting same)
        if os.geteuid() != 0 and not self.dry_run: print("W: Not root. Critical ops may fail.")
        self._setup_intel_sst() 
        if not self.dry_run:
            self.current_tdp_w = self._read_current_tdp_limit_w(); print(f"Initial TDP from system: {self.current_tdp_w:.1f}W.")
            self._set_tdp_limit_w(self.current_tdp_w); print(f"Initial TDP set (clamped): {self.current_tdp_w:.1f}W.")
        else: print(f"[DRY RUN] Initial simulated TDP: {self.current_tdp_w:.1f}W.")
        self.last_tdp_adjustment_time = time.monotonic(); last_print_time = time.monotonic()
        psutil.cpu_percent(percpu=True); time.sleep(0.1) 

        print(f"\n--- Starting Monitoring Loop ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        print(f"Target MAX RU CPU: {self.target_ru_cpu_usage}% | RU Cores Monitored: {self.ru_timing_core_indices}")
        print(f"TDP Update Interval: {self.tdp_update_interval_s}s | Print Interval: {self.print_interval_s}s")
        # ... (other startup info printing like adaptive steps)

        try:
            while True:
                loop_start_time = time.monotonic()
                all_core_usages = self._get_cpu_usages()
                
                # Get the (smoothed) MAX RU_Timing core usage for control
                control_value_for_tdp = self._get_control_ru_timing_cpu_usage(all_core_usages)
                
                current_time = time.monotonic()
                if current_time - self.last_tdp_adjustment_time >= self.tdp_update_interval_s:
                    self._adjust_tdp(control_value_for_tdp)
                    self.last_tdp_adjustment_time = current_time
                
                if current_time - last_print_time >= self.print_interval_s:
                    pkg_power_w, pkg_power_ok = self._get_pkg_power_w()
                    ru_core_usage_details_list = [
                        f"C{idx}:{all_core_usages[idx]:>5.1f}%" if 0 <= idx < len(all_core_usages) else f"C{idx}:N/A"
                        for idx in self.ru_timing_core_indices
                    ]
                    ru_core_details_str = ", ".join(ru_core_usage_details_list) if ru_core_usage_details_list else "N/A"
                    
                    # Find instantaneous max for logging, if different from smoothed control value
                    instant_max_ru_usage = 0.0
                    if self.ru_timing_core_indices and any(0 <= i < len(all_core_usages) for i in self.ru_timing_core_indices):
                        instant_max_ru_usage = max(all_core_usages[i] for i in self.ru_timing_core_indices if 0 <= i < len(all_core_usages))


                    log_msg = (
                        f"{time.strftime('%H:%M:%S')} | "
                        f"RU_Cores: [{ru_core_details_str}] (I_Max:{instant_max_ru_usage:>5.1f}%, S_MaxCtrl:{control_value_for_tdp:>5.1f}%) | "
                        f"TDP:{self.current_tdp_w:>5.1f}W | "
                        f"PkgPwr: {pkg_power_w if pkg_power_ok else 'N/A':>6s}W"
                    )
                    print(log_msg)
                    last_print_time = current_time

                loop_duration = time.monotonic() - loop_start_time
                sleep_time = max(0, 1.0 - loop_duration) 
                time.sleep(sleep_time)

        except KeyboardInterrupt: print(f"\n--- Monitoring stopped by user ({'DRY RUN' if self.dry_run else 'LIVE RUN'}) ---")
        except Exception as e: print(f"\n--- An unexpected error: {e} ---"); import traceback; traceback.print_exc()
        finally: print("Exiting application.")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: [sudo] python3 power_manager_app.py <config.yaml>"); sys.exit(1)
    manager = PowerManager(config_path=sys.argv[1]); manager.run_monitor()
