import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from tc.diagnostics import observe, supervise
from tc.jobs import Prepare


class DiagnosticsTests(unittest.TestCase):
    def test_exit_and_console_survive(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch('tc.diagnostics.snapshot', return_value={}):
            root = Path(temp) / "diagnostics"
            code = supervise([sys.executable, '-u', '-c', 'print("before failure"); raise RuntimeError("failure")'], root, .01)
            self.assertEqual(code, 1)
            self.assertIn('RuntimeError: failure', (root / 'console.log').read_text())
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['exit_code'], 1)

    def test_signal_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch('tc.diagnostics.snapshot', return_value={}):
            root = Path(temp) / "diagnostics"
            code = supervise([sys.executable, '-c', 'import os,signal; os.kill(os.getpid(),signal.SIGTERM)'], root, .01)
            self.assertEqual(code, -signal.SIGTERM)
            self.assertEqual(json.loads((root / 'events.jsonl').read_text().splitlines()[-1])['signal'], 'SIGTERM')

    @unittest.skipUnless(sys.platform == "linux", "native crash diagnostics target Colab/Linux")
    def test_native_abort_preserves_stack_and_buffered_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch('tc.diagnostics.snapshot', return_value={}):
            root = Path(temp) / 'diagnostics'
            child = 'import os,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); print("before abort"); os.abort()'
            code = supervise([sys.executable, '-c', child], root, .01)
            self.assertEqual(code, -signal.SIGABRT)
            console = (root / 'console.log').read_text()
            self.assertIn('before abort', console)
            self.assertIn('Fatal Python error: Aborted', console)
            self.assertIn('File "<string>", line 1', console)

    def test_worker_preserves_traceback(self) -> None:
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'prepare requires'):
                Prepare(jobs_dir=str(root)).execute()
            traceback = next(root.glob('*/traceback.txt')).read_text()
            self.assertIn('ValueError: prepare requires', traceback)

    def test_completed_or_active_directory_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch('tc.diagnostics.snapshot', return_value={}):
            root = Path(temp) / 'diagnostics'
            self.assertEqual(supervise([sys.executable, '-c', 'pass'], root, .01), 0)
            terminal = (root / 'terminal.json').read_bytes()
            with patch('tc.diagnostics.subprocess.Popen') as start:
                with self.assertRaises(FileExistsError):
                    supervise([sys.executable, '-c', 'pass'], root)
                start.assert_not_called()
            with self.assertRaises(FileExistsError):
                observe(123, root)
            self.assertEqual((root / 'terminal.json').read_bytes(), terminal)
            (root / 'terminal.json').unlink()
            with self.assertRaises(FileExistsError):
                observe(123, root)
