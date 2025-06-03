FROM python:3.8-slim as builder

RUN apt-get update && apt-get install -y \
    git wget curl less nano \
    nmap mtr net-tools tcpdump apt-utils sudo jq tree iproute2 iputils-ping traceroute \
    libffi-dev libnl-3-dev libnl-genl-3-dev \
    python3-psutil python3-yaml

# Install RMR
ARG RMR_VERSION=4.9.4
ARG RMR_LIB_URL=https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/rmr_${RMR_VERSION}_amd64.deb/download.deb
ARG RMR_DEV_URL=https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/rmr-dev_${RMR_VERSION}_amd64.deb/download.deb
RUN wget --content-disposition ${RMR_LIB_URL} && dpkg -i rmr_${RMR_VERSION}_amd64.deb
RUN wget --content-disposition ${RMR_DEV_URL} && dpkg -i rmr-dev_${RMR_VERSION}_amd64.deb
RUN rm -f rmr_${RMR_VERSION}_amd64.deb rmr-dev_${RMR_VERSION}_amd64.deb

# Install E2AP
ARG E2AP_VERSION=1.1.0
ARG E2AP_LIB_URL=https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/riclibe2ap_${E2AP_VERSION}_amd64.deb/download.deb
ARG E2AP_DEV_URL=https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/riclibe2ap-dev_${E2AP_VERSION}_amd64.deb/download.deb
RUN wget --content-disposition ${E2AP_LIB_URL} && dpkg -i riclibe2ap_${E2AP_VERSION}_amd64.deb
RUN wget --content-disposition ${E2AP_DEV_URL} && dpkg -i riclibe2ap-dev_${E2AP_VERSION}_amd64.deb
RUN rm -f riclibe2ap_${E2AP_VERSION}_amd64.deb riclibe2ap-dev_${E2AP_VERSION}_amd64.deb


# ---- Final image ----
FROM python:3.8-slim

ARG RMR_VERSION=4.9.4
ARG E2AP_VERSION=1.1.0

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget curl less nano \
    nmap mtr net-tools tcpdump apt-utils sudo jq tree iproute2 iputils-ping traceroute \
    libffi-dev libnl-3-dev libnl-genl-3-dev \
    python3-psutil python3-yaml && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy built shared libs and rmr_probe
COPY --from=builder /usr/local/lib/librmr_si.so.${RMR_VERSION} /usr/local/lib/librmr_si.so
COPY --from=builder /usr/local/lib/libriclibe2ap.so.${E2AP_VERSION} /usr/local/lib/libriclibe2ap.so
COPY --from=builder /usr/local/bin/rmr_probe /opt/e2/rmr_probe

RUN chmod -R 755 /usr/local/lib/librmr_si.so
RUN chmod -R 755 /usr/local/lib/libriclibe2ap.so

# Set environment for shared libs
ENV LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install certifi six python_dateutil setuptools urllib3 logger requests \
    inotify_simple mdclogpy google-api-python-client msgpack ricsdl asn1tools

# Create dirs
RUN mkdir -p /opt/xApps && chmod -R 755 /opt/xApps
RUN mkdir -p /opt/ric/config && chmod -R 755 /opt/ric/config
RUN mkdir -p /opt/e2 && chmod -R 755 /opt/e2

# Clone xApp Python Framework
WORKDIR /opt/
ARG SC_RIC_VERSION=i-release
RUN git clone --depth 1 --branch ${SC_RIC_VERSION} https://github.com/o-ran-sc/ric-plt-xapp-frame-py.git

# Patch xApp Python Framework
WORKDIR /opt/ric-plt-xapp-frame-py
COPY ./ric-plt-xapp-frame-py.patch .
RUN git apply ./ric-plt-xapp-frame-py.patch

# Install xApp Framework
WORKDIR /opt/
RUN pip install -e ./ric-plt-xapp-frame-py

# Copy your app and config
WORKDIR /opt/ecoran
COPY ecoran.py .
COPY config.yaml .
COPY intel-speed-select /usr/bin/

# Copy custom library files
COPY lib/ /opt/xApps/lib/
RUN touch /opt/xApps/lib/__init__.py

# Final command
ENTRYPOINT ["python3", "ecoran.py", "config.yaml"]
