#!/bin/bash
# Make `pytest` work on the first try in a fresh session.
#
# The suite needs only pytest, numpy and httpx -- the same three CI installs -- not the
# runtime stack (mediapipe, opencv, ffmpeg), which is heavy, slow and unavailable on
# some platforms. See .github/workflows/tests.yml.
#
# Three rules this script keeps:
#   1. Idempotent. If the imports already work it prints nothing and does nothing, so
#      resuming a session costs milliseconds.
#   2. Never installs into a Python it does not own. A remote session's container is
#      disposable and an active virtualenv is the user's choice; a bare system Python is
#      neither, so there it only prints the command and leaves the decision alone.
#   3. Always exits 0. A session must never be blocked by its own setup, so every
#      failure degrades to a printed hint.

set -uo pipefail

readonly DEPS="pytest numpy httpx"

if python3 -c "import pytest, numpy, httpx" 2>/dev/null; then
    exit 0
fi

if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "session-start: installing test dependencies ($DEPS)"
    # shellcheck disable=SC2086
    if ! python3 -m pip install --quiet --disable-pip-version-check $DEPS; then
        echo "session-start: install failed; run 'python3 -m pip install $DEPS' by hand"
    fi
    exit 0
fi

echo "session-start: tests need '$DEPS'; install them in a venv before running pytest"
exit 0
