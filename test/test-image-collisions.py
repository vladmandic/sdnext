"""Offline regression tests for generated image filenames; no server or inference.

Run with: python test/test-image-collisions.py
"""

import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))
os.chdir(script_dir)
os.environ['SD_INSTALL_QUIET'] = '1'

# Match test-namegen.py's standalone bootstrap before importing shared.
import modules.cmd_args # pylint: disable=wrong-import-position
import installer # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from PIL import Image # pylint: disable=wrong-import-position
from modules import shared # pylint: disable=wrong-import-position
from modules.image.namegen import FilenameGenerator # pylint: disable=wrong-import-position
from modules.image.save import save_image # pylint: disable=wrong-import-position


class ImageCollisionTests(unittest.TestCase):
    def setUp(self):
        temp_root = script_dir / 'tmp'
        temp_root.mkdir(exist_ok=True)
        work = tempfile.TemporaryDirectory(prefix='image-collisions-', dir=temp_root)
        self.addCleanup(work.cleanup)
        self.folder = Path(work.name)
        self.namegen = FilenameGenerator()
        options = patch.dict(shared.opts.data, save_images_add_number=False, samples_filename_pattern='same',
                             save_to_dirs=False, save_txt=False, image_metadata=True, save_log_fn='',
                             image_watermark='', image_watermark_image='')
        options.start()
        self.addCleanup(options.stop)
        # Keep parameter history in the temporary directory, not the user's data.
        params_path = patch('modules.image.save.paths.params_path', str(self.folder / 'params.txt'))
        params_path.start()
        self.addCleanup(params_path.stop)

    def save(self, color='red', **kwargs):
        image = Image.new('RGB', (8, 8), color)
        filename, text, info = save_image(image, path=str(self.folder), extension='png', **kwargs)
        self.assertEqual(image.already_saved_as, filename)
        return Path(filename), Path(text) if text else None, info

    def assert_color(self, filename, color):
        with Image.open(filename) as image:
            self.assertEqual(image.getpixel((0, 0)), Image.new('RGB', (1, 1), color).getpixel((0, 0)))

    def test_unused_generated_name_is_unchanged(self):
        filename, text, _info = self.save()
        self.assertEqual(filename.name, 'same.png')
        self.assertIsNone(text)

    def test_repeated_saves_preserve_images_and_sidecars(self):
        colors = ['red', 'green', 'blue', 'yellow']
        with patch.dict(shared.opts.data, save_txt=True):
            for index, color in enumerate(colors):
                filename, text, info = self.save(color, info=f'image {index}')
                expected = 'same.png' if index == 0 else f'same-{index}.png'
                self.assertEqual(filename.name, expected)
                self.assertEqual(text, filename.with_suffix('.txt'))
                self.assertEqual(text.read_text(encoding='utf8').strip(), info)
        for index, color in enumerate(colors):
            filename = self.folder / ('same.png' if index == 0 else f'same-{index}.png')
            self.assert_color(filename, color)

    def test_existing_sidecar_without_image_is_preserved(self):
        sidecar = self.folder / 'same.txt'
        sidecar.write_text('existing metadata', encoding='utf8')
        with patch.dict(shared.opts.data, save_txt=True):
            filename, text, _info = self.save(info='new metadata')
        self.assertEqual(filename.name, 'same-1.png')
        self.assertEqual(text.name, 'same-1.txt')
        self.assertEqual(sidecar.read_text(encoding='utf8'), 'existing metadata')

    def test_sidecars_from_other_image_formats_are_preserved(self):
        with patch.dict(shared.opts.data, save_txt=True):
            save_image(Image.new('RGB', (8, 8), 'red'), path=str(self.folder), extension='jpg', info='JPEG metadata')
            filename, text, _info = self.save('blue', info='PNG metadata')
        self.assertEqual(filename.name, 'same-1.png')
        self.assertEqual(text.read_text(encoding='utf8').strip(), 'PNG metadata')
        self.assertEqual((self.folder / 'same.txt').read_text(encoding='utf8').strip(), 'JPEG metadata')

    def test_sidecar_is_ignored_when_none_will_be_written(self):
        sidecar = self.folder / 'same.txt'
        sidecar.write_text('keep', encoding='utf8')
        with patch.dict(shared.opts.data, save_txt=True):
            filename, text, _info = self.save()
        self.assertEqual(filename.name, 'same.png')
        self.assertIsNone(text)
        self.assertEqual(sidecar.read_text(encoding='utf8'), 'keep')

    def test_forced_filename_can_overwrite_image_and_sidecar(self):
        with patch.dict(shared.opts.data, save_txt=True):
            first, _text, _info = self.save('red', forced_filename='explicit', info='old metadata')
            second, text, _info = self.save('blue', forced_filename='explicit', info='new metadata')
        self.assertEqual(first, second)
        self.assertEqual(second.name, 'explicit.png')
        self.assert_color(second, 'blue')
        self.assertEqual(text.read_text(encoding='utf8').strip(), 'new metadata')

    def test_concurrent_generated_saves_do_not_overwrite(self):
        colors = ['red', 'green', 'blue', 'yellow']
        ready = threading.Barrier(len(colors))

        def save_color(color):
            ready.wait(timeout=10)
            return self.save(color)[0]

        with ThreadPoolExecutor(max_workers=len(colors)) as executor:
            filenames = list(executor.map(save_color, colors))
        self.assertEqual(len(set(filenames)), len(colors))
        for filename, color in zip(filenames, colors):
            self.assert_color(filename, color)

    def test_callback_receives_an_unused_image_filename(self):
        first, _text, _info = self.save()
        seen = []
        with patch('modules.script_callbacks.before_image_saved_callback', side_effect=lambda params: seen.append(params.filename)):
            second, _text, _info = self.save('blue')
        self.assertEqual(seen, [str(second)])
        self.assertNotEqual(first, second)
        self.assert_color(first, 'red')

    def test_callback_destination_is_checked_again(self):
        destination = self.folder / 'redirected.png'

        def redirect(params):
            params.filename = str(destination)

        with patch('modules.script_callbacks.before_image_saved_callback', side_effect=redirect):
            first, _text, _info = self.save('red')
            second, _text, _info = self.save('blue')
        self.assertEqual(first, destination)
        self.assertEqual(second.name, 'redirected-1.png')
        self.assert_color(first, 'red')
        self.assert_color(second, 'blue')

    def test_callback_added_metadata_respects_existing_sidecar(self):
        sidecar = self.folder / 'same.txt'
        sidecar.write_text('keep', encoding='utf8')

        def add_metadata(params):
            params.pnginfo['parameters'] = 'callback metadata'

        with patch.dict(shared.opts.data, save_txt=True), patch('modules.script_callbacks.before_image_saved_callback', side_effect=add_metadata):
            filename, text, _info = self.save()
        self.assertEqual(filename.name, 'same-1.png')
        self.assertEqual(text.read_text(encoding='utf8').strip(), 'callback metadata')
        self.assertEqual(sidecar.read_text(encoding='utf8'), 'keep')

    def test_reentrant_callback_save_preserves_both_images(self):
        nested = []

        def save_nested(_params):
            if not nested:
                nested.append(None) # prevent recursion, but allow the nested save's callback
                nested[0] = self.save('blue')[0]

        with patch('modules.script_callbacks.before_image_saved_callback', side_effect=save_nested):
            outer, _text, _info = self.save('red')
        self.assertNotEqual(outer, nested[0])
        self.assert_color(outer, 'red')
        self.assert_color(nested[0], 'blue')

    def test_existing_numbering_options_are_preserved(self):
        for numbering, pattern in ((True, 'image'), (False, '[seq]-image')):
            with self.subTest(numbering=numbering, pattern=pattern), patch.dict(shared.opts.data, save_images_add_number=numbering, samples_filename_pattern=pattern):
                first, _text, _info = self.save('red')
                second, _text, _info = self.save('blue')
            self.assertRegex(first.name, r'^\d{5,}-image\.png$')
            self.assertRegex(second.name, r'^\d{5,}-image\.png$')
            self.assertNotEqual(first, second)
            self.assert_color(first, 'red')

    def test_helper_preserves_extension_and_skips_occupied_suffixes(self):
        filename = self.folder / 'image.v1.png'
        self.assertEqual(self.namegen.collision(str(filename)), str(filename))
        filename.write_text('keep', encoding='utf8')
        (self.folder / 'image.v1-1.png').touch()
        self.assertEqual(Path(self.namegen.collision(str(filename))).name, 'image.v1-2.png')
        self.assertEqual(filename.read_text(encoding='utf8'), 'keep')

    def test_helper_avoids_directory_and_dangling_symlink(self):
        filename = self.folder / 'directory.png'
        filename.mkdir()
        self.assertEqual(Path(self.namegen.collision(str(filename))).name, 'directory-1.png')
        link = self.folder / 'link.png'
        try:
            link.symlink_to(self.folder / 'missing')
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'Symlinks unavailable: {error}')
        self.assertEqual(Path(self.namegen.collision(str(link))).name, 'link-1.png')

    def test_helper_keeps_long_unicode_names_within_component_limit(self):
        limit = os.statvfs(self.folder).f_namemax if hasattr(os, 'statvfs') else 255
        stem = '猫' * ((limit - 4) // len(os.fsencode('猫')))
        filename = self.folder / (stem + '.png')
        filename.touch()
        result = Path(self.namegen.collision(str(filename)))
        self.assertLessEqual(len(os.fsencode(result.name)), limit)
        self.assertTrue(result.name.endswith('-1.png'))
        self.assertNotEqual(filename, result)
        result.touch() # the second candidate really is a valid filename

    def test_helper_allows_the_suffix_to_grow(self):
        limit = os.statvfs(self.folder).f_namemax if hasattr(os, 'statvfs') else 255
        filename = self.folder / ('a' * (limit - 4) + '.png')
        filename.touch()
        for _index in range(12):
            result = Path(self.namegen.collision(str(filename)))
            self.assertLessEqual(len(os.fsencode(result.name)), limit)
            result.touch()
        self.assertTrue(result.name.endswith('-12.png'))

    def test_helper_rejects_a_limit_too_small_for_the_suffix(self):
        filename = self.folder / 'image.png'
        filename.touch()
        with patch('modules.image.namegen.os.statvfs', return_value=SimpleNamespace(f_namemax=5), create=True):
            with self.assertRaisesRegex(ValueError, 'No room for a collision suffix'):
                self.namegen.collision(str(filename))


if __name__ == '__main__':
    unittest.main()
