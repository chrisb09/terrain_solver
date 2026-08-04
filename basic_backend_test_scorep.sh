#!/bin/bash

export USE_SCOREP_ENV=1

exec "$(dirname "$0")/basic_backend_test.sh" "$@"
