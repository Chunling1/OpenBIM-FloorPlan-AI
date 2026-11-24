import base64
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import cv2
import numpy as np

from label_tool import export_contribution, server


class ExportContributionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.images = root / 'community_annotations' / 'images'
        self.masks = self.images.parent / 'annotations'
        self.records = self.images.parent / 'contributors.json'
        self.script = root / 'label_tool' / 'export_contribution.py'
        for directory in (self.images, self.masks, self.script.parent):
            directory.mkdir(parents=True)
        shutil.copyfile(export_contribution.__file__, self.script)

    def export(self, name, *args):
        subprocess.run([sys.executable, str(self.script), '--annotator-name', name, *args],
                       check=True, capture_output=True, text=True)
        return json.loads(self.records.read_text(encoding='utf-8'))

    def test_label_save_then_export_in_place(self):
        drawing = np.full((32, 32, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(self.images / 'room.png'), drawing)
        paint = np.zeros((32, 32, 4), dtype=np.uint8)
        paint[8:12] = [60, 76, 231, 160]
        png = cv2.imencode('.png', paint)[1]
        body = json.dumps({'image_name': 'room.png',
                           'mask_data': base64.b64encode(png).decode()})
        with patch.object(server, 'MASK_DIR', self.masks), \
                HTTPServer(('127.0.0.1', 0), server.LabelToolHTTPHandler) as http:
            worker = Thread(target=http.handle_request, daemon=True)
            worker.start()
            connection = HTTPConnection('127.0.0.1', http.server_port, timeout=5)
            try:
                connection.request('POST', '/api/save_mask', body,
                                   {'Content-Type': 'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())['stats']['wall_px'], 128)
            finally:
                connection.close()
                worker.join(timeout=5)
        recorded = self.export('First')
        self.assertEqual(recorded['annotations'][0]['files'],
                         [{'image': 'room.png', 'mask': 'room_mask.npy'}])
        self.assertEqual(self.export('First'), recorded)
        # Running the default command on an existing clone must preserve attribution.
        self.assertEqual(self.export('Second'), recorded)
        shutil.copyfile(self.images / 'room.png', self.images / 'suite.png')
        shutil.copyfile(self.masks / 'room_mask.npy', self.masks / 'suite_mask.npy')
        updated = self.export('Second')
        self.assertEqual(updated['annotations'][0], recorded['annotations'][0])
        self.assertEqual(updated['annotations'][1]['files'],
                         [{'image': 'suite.png', 'mask': 'suite_mask.npy'}])

    def test_external_directory_and_legacy_names(self):
        source = self.script.parent / 'external'
        source.mkdir()
        for image_name, mask_name in [('plan.JPG', 'plan.npy'),
                                      ('old_mask.png', 'old_mask.npy'),
                                      ('new_mask.png', 'new_mask_mask.npy')]:
            (source / image_name).write_bytes(b'source image')
            np.save(source / mask_name, np.ones((6, 6), dtype=np.uint8))
        recorded = self.export('A', '--data-dir', source, '--mask-dir', source)
        self.assertEqual(recorded['stats']['total_annotated'], 3)
        for item in recorded['annotations'][0]['files']:
            self.assertEqual((self.images / item['image']).read_bytes(),
                             (source / item['image']).read_bytes())
            self.assertEqual((self.masks / item['mask']).read_bytes(),
                             (source / item['mask']).read_bytes())


if __name__ == '__main__':
    unittest.main()
