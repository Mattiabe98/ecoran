#/bin/bash
docker build -t boing7898/ecoran:latest .
podman tag localhost/boing7898/ecoran:latest docker.io/boing7898/ecoran:latest
docker push boing7898/ecoran:latest
