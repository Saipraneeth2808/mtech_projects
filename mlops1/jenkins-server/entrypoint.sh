#!/bin/bash
set -e

echo "Fixing permissions for /mlruns2..."
chmod -R 777 /mlruns2

exec /usr/bin/tini -- /usr/local/bin/jenkins.sh
