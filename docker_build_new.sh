#!/bin/bash

set -e

# run:
# chmod +x docker_build_new.sh
# ./docker_run.sh

export NAME="ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-nlpss23:0.1.0"

echo "Building the container..."
fastbuildah bud -t ${NAME} --format docker -f Dockerfile .
echo "Login to container registry. Username: stud, Password: studregistry"
fastbuildah login ls6-stud-registry.informatik.uni-wuerzburg.de
echo "Pushing container to registry..."
fastbuildah push ${NAME}
