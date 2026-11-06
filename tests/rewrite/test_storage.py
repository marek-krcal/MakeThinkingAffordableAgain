from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from tc import colab, drive


class StorageTests(unittest.TestCase):
    def test_pagination_including_empty_page(self):
        service = Mock()
        service.files().list().execute.side_effect = [
            {'files': [{'id': 'one'}], 'nextPageToken': 'next'},
            {'files': [], 'nextPageToken': 'last'}, {'files': [{'id': 'two'}]}]
        self.assertEqual(list(drive.children(service, 'folder')), [{'id': 'one'}, {'id': 'two'}])
        self.assertEqual(service.files().list.call_args.kwargs['pageToken'], 'last')

    def test_ambiguous_create_reuses_reserved_id(self):
        service = Mock()
        service.files().create().execute.side_effect = TimeoutError()
        service.files().get().execute.return_value = {'id': 'reserved', 'name': 'output', 'parents': ['folder']}
        self.assertEqual(drive.create(service, 'reserved', 'output', 'folder')['id'], 'reserved')
        service.files().generateIds.assert_not_called()

    def test_upload_checks_completion(self):
        with tempfile.TemporaryDirectory() as tmp, patch('googleapiclient.http.MediaFileUpload'):
            path = Path(tmp) / 'data'
            path.write_bytes(b'abc')
            service = Mock()
            service.files().update().next_chunk.side_effect = [(None, None), (None, {'id': 'x', 'size': '3', 'md5Checksum': drive.checksum(path)})]
            self.assertEqual(drive.upload(service, 'x', path)['id'], 'x')
            service.files().update().next_chunk.side_effect = [(None, {'id': 'x', 'size': '2', 'md5Checksum': 'bad'})]
            with self.assertRaises(ValueError):
                drive.upload(service, 'x', path)

    def test_stop_targets_only_named_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = str(uuid4()), str(uuid4())
            runner = Mock()
            colab.stop(tmp, first, runner=runner)
            command = runner.call_args.args[0]
            self.assertEqual(command[-2:], ['--session', 'tc-' + first])
            self.assertNotIn(colab.session_name(second), command)
            runner.side_effect = TimeoutError()
            with self.assertRaises(TimeoutError):
                colab.create(tmp, first, gpu='L4', runner=runner)
            self.assertEqual(runner.call_count, 2)
