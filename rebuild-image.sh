#/bin/bash
podman build -t boing7898/ecoran:latest .
podman tag localhost/boing7898/ecoran:latest docker.io/boing7898/ecoran:latest
podman push boing7898/ecoran:latest
