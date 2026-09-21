"""Exercise the client against a real local HTTP server without model weights."""
import base64
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPLY = {"id": "chat-test", "object": "chat.completion", "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "你好，声音清晰。"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        cls.client = importlib.import_module("client")

    def setUp(self):
        self.requests = []
        self.status = 200
        self.extra_length = 0
        self.reply = json.dumps(REPLY, ensure_ascii=False).encode()
        case = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                case.requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(case.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(case.reply) + case.extra_length))
                self.end_headers()
                self.wfile.write(case.reply)
                self.close_connection = True

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)

    def audio(self, name="音频.wav", data=b"RIFF\x00\xff\x80exact\r\nbytes"):
        path = Path(self.temp.name) / name
        path.write_bytes(data)
        return path

    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(ROOT / "client.py"), "--base-url", self.base_url, *args],
                              capture_output=True, text=True, env=env)

    def test_text_request_and_complete_response(self):
        result = self.client.chat("你好", base_url=self.base_url + "/", system="简洁回答", max_tokens=81)
        self.assertEqual(result, REPLY)
        path, request = self.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(request["messages"], [{"role": "system", "content": "简洁回答"}, {"role": "user", "content": "你好"}])
        self.assertFalse(request["stream"])
        self.assertEqual(request["max_tokens"], 81)
        self.assertEqual(request["model"], "Qwen3Omni-Instruct")
        self.assertEqual(self.client.extract_text(result), "你好，声音清晰。")

    def test_multiple_audio_preserves_exact_bytes_and_never_sends_paths(self):
        paths = [self.audio(), self.audio("第二条.MP3", b"ID3\xff\x00\x99")]
        self.client.chat("比较这两段", paths, base_url=self.base_url)
        request = self.requests[0][1]
        content = request["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "比较这两段"})
        for item, path, mime in zip(content[1:], paths, ["audio/wav", "audio/mpeg"]):
            self.assertEqual(item["type"], "audio_url")
            prefix, encoded = item["audio_url"]["url"].split(",", 1)
            self.assertEqual(prefix, f"data:{mime};base64")
            self.assertEqual(base64.b64decode(encoded), path.read_bytes())
            self.assertNotIn(str(path), json.dumps(request))

    def test_four_audio_allowed_fifth_rejected_without_request(self):
        path = self.audio()
        self.client.chat("比较", [path] * 4, base_url=self.base_url)
        with self.assertRaises(self.client.ClientError):
            self.client.chat("比较", [path] * 5, base_url=self.base_url)
        self.assertEqual(len(self.requests), 1)

    def test_invalid_audio_rejected_before_network(self):
        for path in [Path(self.temp.name) / "missing.wav", Path(self.temp.name), self.audio("empty.wav", b""), self.audio("video.exe")]:
            with self.subTest(path=path), self.assertRaises(self.client.ClientError):
                self.client.chat("转写", [path], base_url=self.base_url)
        self.assertEqual(self.requests, [])

    def test_http_error_is_clear_and_does_not_retry(self):
        self.status = 500
        self.reply = b'{"error":{"message":"model unavailable"}}'
        with self.assertRaisesRegex(self.client.ClientError, "500.*model unavailable"):
            self.client.chat("hello", base_url=self.base_url)
        self.assertEqual(len(self.requests), 1)

    def test_invalid_reply_fails(self):
        for reply in [b"not json", b"[]", b"{}", b'{"choices":[]}', b'{"choices":[{"message":{}}]}']:
            self.reply = reply
            with self.subTest(reply=reply), self.assertRaises(self.client.ClientError):
                self.client.chat("hello", base_url=self.base_url)

    def test_truncated_http_reply_is_clear_without_retry(self):
        self.extra_length = 20
        with self.assertRaises(self.client.ClientError):
            self.client.chat("hello", base_url=self.base_url)
        self.assertEqual(len(self.requests), 1)
        result = self.cli("--prompt", "你好")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)

    def test_text_blocks_are_extracted(self):
        self.assertEqual(self.client.extract_text({"choices": [{"message": {"content": [
            {"type": "text", "text": "第一"}, {"type": "text", "text": "第二"}]}}]}), "第一第二")

    def test_proxy_environment_ignored_without_modification(self):
        values = {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""}
        with patch.dict(os.environ, values):
            self.client.chat("hello", base_url=self.base_url)
            self.assertTrue(all(os.environ[key] == value for key, value in values.items()))
        self.assertEqual(len(self.requests), 1)

    def test_cli_text_and_json(self):
        result = self.cli("--prompt", "你好", "--audio", str(self.audio()))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "你好，声音清晰。")
        result = self.cli("--prompt", "你好", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), REPLY)

    def test_cli_error_is_nonzero_without_traceback(self):
        for args in [("--prompt", "你好", "--audio", "/missing.wav"), ("--prompt", "你好", "--timeout", "0"), ("--prompt", "你好", "--max-tokens", "0")]:
            result = self.cli(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
