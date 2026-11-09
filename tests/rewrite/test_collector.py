from pathlib import Path
import shutil
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tc.collector import Collector, archive_script, finish
from tc.diagnostics import supervise


class CollectorTests(unittest.TestCase):
    def test_waits_for_exit_before_final_copy(self) -> None:
        fetch, archive, call = Mock(return_value=False), Mock(), Mock()
        self.assertFalse(finish({'state': 'failed', 'job_id': 'test'}, fetch, archive, call, 'job', 'console', 'diag'))
        fetch.assert_called_once_with('diag/terminal.json', 'terminal.json')
        archive.assert_not_called()
        call.assert_not_called()

    def test_failed_final_copy_prevents_release(self) -> None:
        fetch, archive, call = Mock(side_effect=[True, False]), Mock(), Mock()
        self.assertFalse(finish({'state': 'failed', 'job_id': 'test'}, fetch, archive, call, 'job', 'console', 'diag'))
        call.assert_not_called()

    def test_failed_archive_prevents_release(self) -> None:
        call = Mock()
        self.assertFalse(finish({'state': 'failed', 'job_id': 'test'}, Mock(return_value=True), Mock(return_value=False), call, 'job', 'console', 'diag'))
        call.assert_not_called()

    def test_exception_tail_copied_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch('tc.diagnostics.snapshot', return_value={}):
            root = Path(temp)
            remote, local = root/'remote', root/'local'
            local.mkdir()
            supervise([sys.executable, '-u', '-c', 'raise RuntimeError("last line")'], remote, .01)
            def fetch(source: str, name: str) -> bool:
                if not Path(source).exists(): return False
                shutil.copyfile(source, local/name)
                return True
            def release(*args: str) -> bool:
                self.assertEqual(args, ('stop',))
                self.assertIn('RuntimeError: last line', (local/'console.log').read_text())
                return True
            self.assertTrue(finish({'state': 'failed', 'job_id': 'test'}, fetch, Mock(return_value=True), release,
                                   str(root/'job'), str(remote/'console.log'), str(remote)))

    def test_failed_partial_download_keeps_last_recovery_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            collector = Collector('session', 'job', '/job', '/console', '/diag', temp, '/sessions',
                                  snapshot_files=['outputs/records.jsonl'])
            failed = False
            def download(command, **kwargs):
                source, target = command[6:8]
                data = b'{"answer": 42}\n'
                code = 0
                if source.endswith('status.json'):
                    data = json.dumps({'job_id': 'job', 'state': 'running'}).encode()
                elif source.endswith('terminal.json') or source.endswith('log.txt'):
                    code = 1
                elif source.endswith('records.jsonl') and failed:
                    data, code = b'broken partial transfer', 1
                Path(target).write_bytes(data)
                return subprocess.CompletedProcess(command, code, b'', b'')
            with patch('tc.collector.subprocess.run', side_effect=download), patch('tc.collector.time.sleep', side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):
                    collector.run()
                saved = Path(temp)/'partial/job/outputs/records.jsonl'
                self.assertEqual(saved.read_bytes(), b'{"answer": 42}\n')
                failed = True
                with self.assertRaises(InterruptedError):
                    collector.run()
                self.assertEqual(saved.read_bytes(), b'{"answer": 42}\n')

    def test_final_archive_includes_external_console(self) -> None:
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostics = root/'diagnostics'
            diagnostics.mkdir()
            console = root/'external.log'
            console.write_text('final traceback line')
            archive = root/'final.tar.gz'
            exec(archive_script(str(diagnostics), str(archive), str(console)), {})
            with tarfile.open(archive) as saved:
                self.assertEqual(saved.extractfile('console.log').read(), b'final traceback line')
