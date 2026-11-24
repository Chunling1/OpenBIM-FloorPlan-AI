import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mask_to_ifc import mask_to_ifc


def opening_widths(entities, kind):
    """Read OverallWidth and the width of the extruded profile from the IFC file."""
    product, = [value for value in entities.values() if value.startswith(kind + '(')]

    def refs(entity):
        return [entities[key] for key in re.findall(r'#\d+', entity)]

    shape = refs(product)[-1]
    solid = refs(refs(shape)[0])[-1]
    profile = refs(solid)[0]
    points = refs(refs(profile)[0])
    xs = [float(point.split('((')[1].split(',')[0]) for point in points]
    overall_width = float(product.rsplit(',', 1)[1].rstrip(')'))
    return overall_width, max(xs) - min(xs)


class OpeningWidthTests(unittest.TestCase):
    def test_door_and_window_at_two_scales(self):
        mask = np.zeros((240, 300), dtype=np.uint8)
        mask[40:46, 10:290] = 1
        mask[40:46, 85:175] = 3
        mask[120:126, 10:290] = 1
        mask[120:126, 70:190] = 2
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'geometry.ifc'
            for scale in (0.01, 0.02):
                with self.subTest(scale=scale):
                    mask_to_ifc(mask, output, scale=scale)
                    entities = dict(re.findall(r'^(#\d+)=(.+);$', output.read_text(), re.M))
                    for kind, pixel_width in [('IFCDOOR', 90), ('IFCWINDOW', 120)]:
                        overall, geometry = opening_widths(entities, kind)
                        self.assertAlmostEqual(overall, pixel_width * scale)
                        self.assertAlmostEqual(geometry, pixel_width * scale)


if __name__ == '__main__':
    unittest.main()
