# Use a lightweight Python base image - Alpine Linux is very small
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /opt/ecoran/

# Install system dependencies
# - build-base is needed for psutil to compile C extensions if a wheel isn't available
# - yaml-dev is needed for PyYAML to compile its C extensions
# We'll remove build-base after to keep the image small
RUN apk add --no-cache --virtual .build-deps build-base yaml-dev && \
    apk add --no-cache libffi-dev nano # libffi-dev might be needed by PyYAML's C extensions on some platforms

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    # Clean up build dependencies
    apk del .build-deps

# Copy the application code into the container
COPY ecoran.py .

# Copy Intel Speed Select binary
COPY intel-speed-select /usr/bin/

ENTRYPOINT ["python3", "ecoran.py"]
