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
    def test_collection_recovers_after_prolonged_transport_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            collector = Collector('session', 'job', '/job', '/console', '/diag', temp, '/sessions')
            attempts = 0
            def download(command, **kwargs):
                nonlocal attempts
                source, target = command[6:8]
                if source.endswith('status.json'):
                    attempts += 1
                    if attempts > 5:
                        Path(target).write_text(json.dumps(dict(job_id='job', state='running', step=9)))
                        return subprocess.CompletedProcess(command, 0, b'', b'')
                return subprocess.CompletedProcess(command, 1, b'', b'network unavailable')
            def wait(seconds):
                if attempts > 5:
                    raise InterruptedError
            with patch('tc.collector.subprocess.run', side_effect=download), patch('tc.collector.time.sleep', side_effect=wait):
                with self.assertRaises(InterruptedError):
                    collector.run()
            saved = json.loads((Path(temp)/'status.json').read_text())
            self.assertEqual(saved['step'], 9)
            self.assertEqual(json.loads((Path(temp)/'monitor.json').read_text())['state'], 'monitoring')

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

    def test_prebuilt_checkpoint_requires_valid_transfer_without_kernel_execution(self) -> None:
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collector = Collector('session', 'job', '/job', '/console', '/diag', temp, '/sessions',
                                  checkpoint_archives='/exports')
            valid = False
            def download(command, **kwargs):
                self.assertNotIn('exec', command)
                source, target = command[6:8]
                target = Path(target)
                code = 0
                if source.endswith('status.json'):
                    target.write_text(json.dumps(dict(job_id='job', state='running', checkpoint='checkpoints/step-49')))
                elif source.endswith('config.yaml'):
                    target.write_text('arguments:\n  checkpoint_every: 49\n')
                elif source.endswith('checkpoint-49.tar.gz') and valid:
                    with tarfile.open(target, 'w:gz') as archive:
                        entry = tarfile.TarInfo('step-49/state.pt')
                        entry.size = 7
                        archive.addfile(entry, io.BytesIO(b'resume!'))
                elif source.endswith('terminal.json') or source.endswith('log.txt'):
                    code = 1
                else:
                    target.write_bytes(b'incomplete archive')
                return subprocess.CompletedProcess(command, code, b'', b'')
            with patch('tc.collector.subprocess.run', side_effect=download), patch('tc.collector.time.sleep', side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):
                    collector.run()
                self.assertFalse((root/'checkpoint-49.tar.gz').exists())
                valid = True
                with self.assertRaises(InterruptedError):
                    collector.run()
                with tarfile.open(root/'checkpoint-49.tar.gz') as archive:
                    self.assertEqual(archive.extractfile('step-49/state.pt').read(), b'resume!')

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

    def test_collection_keeps_one_safety_copy_until_success(self) -> None:
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collector = Collector('session', 'job', '/job', '/console', '/diag', temp, '/sessions',
                                  checkpoint_archives='/exports')
            (root/'checkpoint-637.tar.gz').write_bytes(b'previous verified copy')
            missing = True
            state = 'running'
            copied = []
            def download(command, **kwargs):
                source, target = command[6:8]
                target = Path(target)
                if source.endswith('status.json'):
                    target.write_text(json.dumps(dict(job_id='job', state=state, step=737,
                                                     checkpoint='checkpoints/step-737')))
                elif source.endswith('config.yaml'):
                    target.write_text('arguments:\n  checkpoint_every: 49\n')
                elif source.startswith('/exports/'):
                    number = int(Path(source).name.split('-')[1].split('.')[0])
                    copied.append(number)
                    if number == 735 and missing:
                        return subprocess.CompletedProcess(command, 1, b'', b'failed transfer')
                    with tarfile.open(target, 'w:gz') as archive:
                        entry = tarfile.TarInfo(f'step-{number}/state.pt')
                        entry.size = 7
                        archive.addfile(entry, io.BytesIO(b'resume!'))
                else:
                    return subprocess.CompletedProcess(command, 1, b'', b'not present')
                return subprocess.CompletedProcess(command, 0, b'', b'')
            with patch('tc.collector.subprocess.run', side_effect=download), patch('tc.collector.finish', return_value=True) as final, patch('tc.collector.time.sleep', side_effect=InterruptedError):
                with self.assertRaises(InterruptedError): collector.run()
                final.assert_not_called()
                self.assertEqual(copied, [735, 737])
                self.assertTrue((root/'checkpoint-637.tar.gz').exists())
                missing = False
                with self.assertRaises(InterruptedError): collector.run()
                self.assertFalse((root/'checkpoint-637.tar.gz').exists())
                self.assertTrue((root/'checkpoint-735.tar.gz').exists())
                state = 'completed'
                collector.run()
                final.assert_called_once()
                self.assertFalse((root/'checkpoint-735.tar.gz').exists())
                for number in (737,):
                    with tarfile.open(root/f'checkpoint-{number}.tar.gz') as archive:
                        self.assertEqual(archive.extractfile(f'step-{number}/state.pt').read(), b'resume!')

    def test_resumed_collection_does_not_require_parent_runtime_checkpoints(self) -> None:
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            collector = Collector('session', 'job', '/job', '/console', '/diag', temp, '/sessions',
                                  resume_step=737, checkpoint_archives='/exports')
            requested = []
            def download(command, **kwargs):
                source, target = command[6:8]
                target = Path(target)
                if source.endswith('status.json'):
                    target.write_text(json.dumps(dict(job_id='job', state='completed', step=784,
                                                     checkpoint='checkpoints/step-784')))
                elif source.endswith('config.yaml'):
                    target.write_text('arguments:\n  checkpoint_every: 49\n')
                elif source.startswith('/exports/'):
                    requested.append(source)
                    self.assertEqual(source, '/exports/checkpoint-784.tar.gz')
                    with tarfile.open(target, 'w:gz') as archive:
                        entry = tarfile.TarInfo('step-784/state.pt'); entry.size = 7
                        archive.addfile(entry, io.BytesIO(b'resume!'))
                else:
                    return subprocess.CompletedProcess(command, 1, b'', b'not present')
                return subprocess.CompletedProcess(command, 0, b'', b'')
            with patch('tc.collector.subprocess.run', side_effect=download), patch('tc.collector.finish', return_value=True) as final:
                collector.run()
                final.assert_called_once()
            self.assertEqual(requested, ['/exports/checkpoint-784.tar.gz'])
