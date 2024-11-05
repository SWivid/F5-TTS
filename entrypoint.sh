#!/bin/bash

# Change ownership of the mounted volume to match host UID and GID
chown -R ${HOST_UID}:${HOST_GID} /tmp

# Execute the main process (the command passed to the container)
exec "$@"
