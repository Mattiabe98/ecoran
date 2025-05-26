# Use a lightweight Python base image - Alpine Linux is very small
FROM debian:stable-slim

# Install intel-speed-select and dependencies
RUN apt-get update && \
    apt-get install -y nano libffi-dev libnl-3-dev libnl-genl-3-dev python3-psutil python3-yaml && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /opt/ecoran/


# Copy Intel Speed Select binary
COPY intel-speed-select /usr/bin/

ENTRYPOINT ["python3", "ecoran.py"]
