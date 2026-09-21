import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TunnelTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "argv.json"
        ssh = self.directory / "ssh"
        ssh.write_text("#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ['TEST_SSH_ARGV'], 'w') as f: json.dump(sys.argv[1:], f)\n")
        ssh.chmod(0o755)

    def run_tunnel(self, *args):
        env = dict(os.environ, PATH=f"{self.directory}:{os.environ['PATH']}", TEST_SSH_ARGV=str(self.output))
        return subprocess.run(["bash", str(ROOT / "scripts/tunnel.sh"), *args], env=env, capture_output=True, text=True)

    def test_defaults_are_loopback_and_foreground(self):
        result = self.run_tunnel("user@compute")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.output.read_text())
        self.assertEqual(args[-1], "user@compute")
        self.assertIn("127.0.0.1:18001:127.0.0.1:8001", args)
        self.assertEqual(args[args.index("-p") + 1], "22")
        self.assertIn("-N", args)
        self.assertIn("-T", args)
        self.assertNotIn("-f", args)
        self.assertIn("ExitOnForwardFailure=yes", args)
        self.assertIn("ServerAliveInterval=30", args)
        self.assertIn("ServerAliveCountMax=3", args)

    def test_custom_ports_and_host_alias(self):
        result = self.run_tunnel("compute-alias", "--ssh-port", "2222", "--local-port", "18002", "--remote-port", "8002")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.output.read_text())
        self.assertEqual(args[-1], "compute-alias")
        self.assertIn("127.0.0.1:18002:127.0.0.1:8002", args)
        self.assertEqual(args[args.index("-p") + 1], "2222")

    def test_bad_inputs_do_not_run_ssh(self):
        for args in [(), ("-oProxyCommand=touch /tmp/bad",), ("host", "--local-port", "0"), ("host", "--remote-port", "65536"), ("host", "--ssh-port", "no"), ("host", "--ssh-port"), ("host", "another"), ("host", "--local-port", "999999999999999999999999")]:
            with self.subTest(args=args):
                result = self.run_tunnel(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
