# EcoRAN
This script is a starting point for a framework to reduce power consumption in a multi-tenant, neutral host vRAN deployment.

## CPU Power Optimization Parameters

The framework utilizes two main power management parameters for CPU power optimization:

### 1. Dynamic CPU Power Limit (TDP) Adjustment

Modern Intel Xeon processors expose RAPL (Running Average Power Limit) interfaces, allowing software control over package power limits.  
The framework focuses on the `constraint_0_power_limit_uw` parameter (long-term power limit, PL1). Adjusting PL1 dynamically allows the system to cap the CPU’s sustained power consumption.

This method proved more effective and stable than direct frequency scaling, as the CPU’s internal power management can make faster and finer-grained decisions to stay within the given power budget.

### 2. Intel Speed Select Technology - Core Power (SST-CP)

SST-CP enables the creation of up to four Classes of Service (CLOS), to which physical CPU cores can be assigned.  
Cores in higher-priority CLOS (with CLOS 0 being the highest) receive preferential power allocation when the CPU reaches its power limit.  
Additionally, a minimum frequency can be set per CLOS.

This is ideal for a multi-tenant, cost-based deployment scenario:

- Tenants can be mapped to different CLOS based on their service tier (e.g., Platinum, Gold, Silver, Bronze → CLOS 0, 1, 2, 3).
- Critical RAN threads (e.g., `srsRAN`'s `ru_timing`) can be assigned to a high-priority CLOS and pinned to dedicated physical cores to ensure stability and low latency, even under power constraints.


## Requirements:
- A CPU that supports Intel SST-CP
- A srsRAN 7.2 O-RU deployment (or similar 7.2 gNB with dedicated O-RU timing core)
- A Kubernetes cluster
- Python3
- Privileged pod

The script is very dependant on the configuration file, which requires editing to create the different SST-CP classes of service and to import the srsRAN/gNB CPU affinity list. 
The dynamic algorithm is ready to be used and is configured to keep the O-RU timing cores around 99.5%, which should lead to maximum power saving, while keeping throughput acceptable. Configuring the algorithm variables can greatly influence the throughput vs energy consumption mapping.

The config.yaml file is commented to help you get up and running as easy as possible.

Running the script is quite easy: 
Usage: python3 ecoran.py <path_to_config.yaml>

