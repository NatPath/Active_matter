#!/usr/bin/env bash

# Copy this file to cluster_scripts/cluster_env.sh and fill in your local values.
# That local file is ignored by git and read automatically by the copy scripts.

export CLUSTER_REMOTE_USER="your_user"
export CLUSTER_REMOTE_HOST="your.cluster.example.edu"

# Where the git-clone lives on the cluster.
export CLUSTER_CODE_ROOT="/path/to/cluster/repo/clone"

# Where heavy run data lives on the cluster.
export CLUSTER_DATA_ROOT="/path/to/cluster/run-data"

# Optional: cluster-specific Julia setup script used before local plotting after fetch.
# Leave empty to use plain `julia`.
export CLUSTER_JULIA_SETUP_SCRIPT=""

# Optional: notification settings for cluster-side POST/FINAL notifier scripts.
# These stay in cluster_env.sh so secrets do not enter git.
#
# ntfy push notifications:
export NOTIFY_NTFY_TOPIC="nativ-cluster-9f4e3b7c2a18"
export NOTIFY_NTFY_SERVER="https://ntfy.sh"
export NOTIFY_NTFY_PRIORITY="default"
export NOTIFY_NTFY_TAGS="white_check_mark"
export NOTIFY_CURL_BIN="/usr/bin/curl"
#
# Optional: ntfy access token for private/self-hosted topics.
#   export NOTIFY_NTFY_TOKEN=""
#
# SMTP with curl:
#   export NOTIFY_SMTP_URL="smtps://smtp.example.com:465"
#   export NOTIFY_SMTP_USER="smtp-user"
#   export NOTIFY_SMTP_PASS="smtp-app-password"
#   export NOTIFY_FROM_EMAIL="sender@example.com"
#
# Optional: override curl binary if needed.
#   export NOTIFY_CURL_BIN="/usr/bin/curl"
