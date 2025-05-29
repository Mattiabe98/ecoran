# EcoRAN
This script is a starting point for a framework to reduce power consumption in a multi-tenant, neutral host vRAN deployment.

Requirements:
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

