import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from tc.cli import parse_args
from tc.monitor import monitor
from tc.job_utils import check_stop


class MonitorTests(unittest.TestCase):
    def test_stop_only_selected_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = [str(uuid4()), str(uuid4())]
            for identity in ids:
                path = Path(tmp) / identity
                path.mkdir()
                (path / 'status.json').write_text(json.dumps({'state': 'running'}))
            with contextlib.redirect_stdout(io.StringIO()):
                monitor(parse_args(['stop', ids[0], tmp]))
            with self.assertRaises(KeyboardInterrupt):
                check_stop(Path(tmp) / ids[0])
            check_stop(Path(tmp) / ids[1])
            with self.assertRaises(ValueError):
                monitor(parse_args(["stop", "../" + ids[0], tmp]))
            self.assertEqual(json.loads((Path(tmp) / ids[0] / 'status.json').read_text())['state'], 'running')
