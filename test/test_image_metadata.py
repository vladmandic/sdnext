import base64
import io
import unittest

from PIL import Image

from modules import images
from modules.api.helpers import encode_pil_to_base64


class TestImageMetadataRoundtrip(unittest.TestCase):
    def test_api_encoded_images_load_through_ui_metadata_reader(self):
        source = Image.new('RGB', (32, 32), 'navy')
        source.info['parameters'] = 'Steps: 12, Sampler: Euler, Seed: 98765, Model: metadata-roundtrip'

        for extension in ('png', 'jpg', 'webp', 'avif', 'heif', 'jxl'):
            if f'.{extension}' not in Image.registered_extensions():
                continue
            with self.subTest(extension=extension):
                encoded = encode_pil_to_base64(source, ext=extension)
                image = Image.open(io.BytesIO(base64.b64decode(encoded)))
                info, items = images.read_info_from_image(image)

                self.assertIn('Steps: 12', info)
                self.assertIn('Seed: 98765', info)
                self.assertEqual(items.get('parameters'), source.info['parameters'])


if __name__ == '__main__':
    unittest.main()
