#!/usr/bin/env bash
# Run on the client machine. This remains in the foreground; tmux is optional.
set -euo pipefail

usage() {
    echo "Usage: $0 user@host [--ssh-port 22] [--local-port 18001] [--remote-port 8001]"
    echo "SSH host aliases are also accepted. Ctrl-C closes the tunnel."
}

fail() { echo "Error: $*" >&2; exit 2; }

ssh_port=22
local_port=18001
remote_port=8001
destination=
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --ssh-port|--local-port|--remote-port)
            (( $# >= 2 )) || fail "Missing value for $1"
            case "$1" in
                --ssh-port) ssh_port=$2 ;;
                --local-port) local_port=$2 ;;
                --remote-port) remote_port=$2 ;;
            esac
            shift 2 ;;
        -*) fail "Unknown option: $1" ;;
        *)
            [[ -z "$destination" ]] || fail "Only one SSH destination is allowed"
            destination=$1
            shift ;;
    esac
done

[[ -n "$destination" ]] || fail "Specify an SSH destination (user@host or host alias)"
[[ "$destination" != *[[:space:]]* ]] || fail "SSH destination must not contain whitespace"
for value in "$ssh_port" "$local_port" "$remote_port"; do
    [[ "$value" =~ ^[0-9]{1,5}$ ]] || fail "Invalid port: $value"
    (( 10#$value >= 1 && 10#$value <= 65535 )) || fail "Port must be between 1 and 65535: $value"
done

ssh_port=$((10#$ssh_port))
local_port=$((10#$local_port))
remote_port=$((10#$remote_port))
args=(-N -T -p "$ssh_port"
      -o ExitOnForwardFailure=yes
      -o ServerAliveInterval=30
      -o ServerAliveCountMax=3
      -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}")
exec ssh "${args[@]}" "$destination"
