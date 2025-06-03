FROM python:3.8-slim as builder

RUN apt-get update -y && \
    apt-get install -y git wget curl nano linux-tools-common \
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

FROM python:3.8-slim

# Copy compiled libs and rmr_probe from builder
ARG RMR_VERSION=4.9.4
ARG E2AP_VERSION=1.1.0
COPY --from=builder /usr/local/lib/librmr_si.so.${RMR_VERSION} /usr/local/lib/librmr_si.so
COPY --from=builder /usr/local/lib/libriclibe2ap.so.${E2AP_VERSION} /usr/local/lib/libriclibe2ap.so
COPY --from=builder /usr/local/bin/rmr_probe /opt/e2/rmr_probe

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libffi-dev libnl-3-dev libnl-genl-3-dev \
    nano linux-tools-common python3-psutil python3-yaml && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip && \
    pip install certifi six python_dateutil setuptools urllib3 logger requests \
    inotify_simple mdclogpy google-api-python-client msgpack ricsdl asn1tools

# Copy application code
WORKDIR /opt/ecoran
COPY ecoran.py .
COPY config.yaml .
COPY lib/ ./lib/
COPY intel-speed-select /usr/bin/

# Optional: make lib a package
RUN touch lib/__init__.py

# Set environment for shared libs
ENV LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib

# Run the app
ENTRYPOINT ["python3", "ecoran.py", "config.yaml"]
