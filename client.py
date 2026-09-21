#!/usr/bin/env python3
"""Standalone, dependency-free text/audio client for Qwen3-Omni (Python 3.10+)."""
from __future__ import annotations

import argparse
import base64
from http.client import HTTPException
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen3Omni-Instruct"
_AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
                ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4",
                ".mp4": "audio/mp4", ".aac": "audio/aac", ".aiff": "audio/aiff",
                ".aif": "audio/aiff", ".webm": "audio/webm"}


class ClientError(RuntimeError):
    """An invalid input, failed request, or unusable server response."""


def extract_text(response: dict) -> str:
    """Return the first assistant answer; reject absent or malformed content."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError) as exc:
        raise ClientError("Invalid server response: missing choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        if all(isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
               for part in content):
            return "".join(part["text"] for part in content)
    raise ClientError("Invalid server response: assistant content must contain text")


def _audio_part(audio_path) -> dict:
    path = Path(audio_path).expanduser()
    media_type = _AUDIO_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ClientError(f"Unsupported audio extension: {path.suffix or '(none)'}")
    try:
        if not path.is_file():
            raise ClientError(f"Audio file does not exist or is not a regular file: {path}")
        data = path.read_bytes()
    except OSError as exc:
        raise ClientError(f"Cannot read audio file {path}: {exc}") from exc
    if not data:
        raise ClientError(f"Audio file is empty: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "audio_url", "audio_url": {"url": f"data:{media_type};base64,{encoded}"}}


def chat(prompt, audio_paths=(), *, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
         timeout=300, max_tokens=1024, system=None) -> dict:
    """Send one nonstreaming request, without proxies from the environment or retries.

    Audio paths refer to files on the calling machine. Their exact bytes are sent
    as base64 data URLs. HTTP failures raise ClientError; successful calls return
    the complete JSON response, including usage and finish_reason.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ClientError("prompt must be a nonempty string")
    if system is not None and not isinstance(system, str):
        raise ClientError("system must be a string")
    if not isinstance(model, str) or not model.strip():
        raise ClientError("model must be a nonempty string")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ClientError("timeout must be a finite number greater than zero")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ClientError("max_tokens must be a positive integer")
    if not isinstance(base_url, str):
        raise ClientError("base_url must be an HTTP or HTTPS URL ending in /v1")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise ClientError(f"Invalid base_url: {exc}") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ClientError("base_url must be an HTTP or HTTPS URL without credentials, query, or fragment")
    if port == 0:
        raise ClientError("base_url port must be between 1 and 65535")
    if isinstance(audio_paths, (str, bytes, Path)):
        raise ClientError("audio_paths must be a sequence of paths, for example [path]")
    paths = list(audio_paths)
    if len(paths) > 4:
        raise ClientError("At most 4 audio files may be sent in one request")
    content = prompt
    if paths:
        content = [{"type": "text", "text": prompt}] + [_audio_part(path) for path in paths]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    request = Request(base_url.rstrip("/") + "/chat/completions",
                      data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise ClientError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        raise ClientError(f"Request failed: {exc}") from exc
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ClientError("Invalid server response: expected JSON") from exc
    if not isinstance(result, dict):
        raise ClientError("Invalid server response: expected a JSON object")
    extract_text(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API root including /v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", help="Optional system instruction")
    parser.add_argument("--audio", action="append", default=[], metavar="PATH", help="Local audio file; repeat up to four times")
    parser.add_argument("--timeout", type=float, default=300, help="HTTP timeout in seconds (default: 300)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--json", action="store_true", help="Print the complete JSON response")
    args = parser.parse_args(argv)
    try:
        result = chat(args.prompt, args.audio, base_url=args.base_url, model=args.model,
                      timeout=args.timeout, max_tokens=args.max_tokens, system=args.system)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else extract_text(result))
    except ClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
