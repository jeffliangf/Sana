#!/usr/bin/env bash
# Run on the persistent login host. GPU worker jobs are independent of this service.
set -euo pipefail
umask 077
gateway_root="${H3_DEPLOY_ROOT:-/home/jeff/sol-h3-h200}"
gateway_host="${H3_GATEWAY_HOST:-10.2.201.187}"
gateway_port="${H3_GATEWAY_PORT:-18080}"
gateway_env="$gateway_root/gateway-venv"
gateway_units="$HOME/.config/systemd/user"
gateway_uv="${UV_BIN:-$HOME/.local/bin/uv}"
mkdir -p "$gateway_root/gateway" "$gateway_units"

# Refuse accidental endpoint drift on later updates or GPU topology changes.
python3 - "$gateway_root/gateway/endpoint.json" "$gateway_host" "$gateway_port" <<'PY'
from pathlib import Path
import json,sys
path=Path(sys.argv[1])
expected={'host':sys.argv[2], 'port':int(sys.argv[3]), 'url':f'http://{sys.argv[2]}:{sys.argv[3]}'}
if path.exists():
    if json.loads(path.read_text()) != expected:
        raise RuntimeError('Gateway endpoint is pinned; do not change client URLs during an update')
else:
    path.write_text(json.dumps(expected,indent=2)+'\n')
PY

if [[ ! -x "$gateway_env/bin/python" ]]; then
    "$gateway_uv" venv --python 3.12 "$gateway_env"
fi
"$gateway_uv" pip install --python "$gateway_env/bin/python" -r "$gateway_root/Sol-H3/requirements-gateway.txt"

cat > "$gateway_units/h3-gateway.socket" <<EOF
[Unit]
Description=Stable Sol-H3 API listener

[Socket]
ListenStream=$gateway_host:$gateway_port
NoDelay=true
Backlog=256

[Install]
WantedBy=sockets.target
EOF

cat > "$gateway_units/h3-gateway.service" <<EOF
[Unit]
Description=Sol-H3 worker gateway
Requires=h3-gateway.socket
After=h3-gateway.socket

[Service]
Type=simple
Sockets=h3-gateway.socket
WorkingDirectory=$gateway_root/Sol-H3
ExecStart=$gateway_env/bin/python $gateway_root/Sol-H3/gateway.py --registry $gateway_root/deployment --state-dir $gateway_root/gateway --token-file $gateway_root/deployment/token --host $gateway_host --port $gateway_port --fd 3
Environment=PYTHONUNBUFFERED=1
UMask=0077
Restart=always
RestartSec=3
TimeoutStopSec=45
NoNewPrivileges=true

[Install]
WantedBy=default.target
EOF

loginctl enable-linger "$(id -un)"
systemctl --user daemon-reload
systemctl --user enable --now h3-gateway.socket
systemctl --user enable h3-gateway.service
systemctl --user restart h3-gateway.service
systemctl --user --no-pager status h3-gateway.socket h3-gateway.service
