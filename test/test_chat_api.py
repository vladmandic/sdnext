import asyncio
import base64
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI, UploadFile
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules.chat import artifacts, commands
from modules.chat import api as chat_api
from modules.chat import db
from modules import progress, shared


class TestChatArtifactService(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory
    allowed_dir: Path
    artifact_dir: Path

    def setUp(self):
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.allowed_dir = Path(self.temp_dir) / 'allowed'
        self.allowed_dir.mkdir()
        self.artifact_dir = Path(self.temp_dir) / 'private'
        self.artifact_dir.mkdir(mode=0o700)

    def test_storage_root_uses_configured_chat_upload_directory(self):
        configured_dir = self.allowed_dir / 'chat-uploads'
        with patch.dict(artifacts.shared.opts.data, {'chat_upload_dir': str(configured_dir)}):
            root = artifacts.storage_root()

        self.assertEqual(root, configured_dir.resolve())
        self.assertTrue(root.is_dir())

    def test_storage_root_defaults_to_chat_artifact_folder_under_temp_dir(self):
        configured_temp_dir = self.allowed_dir / 'temp'
        expected_root = configured_temp_dir / 'sdnext-chat-artifacts'
        with patch.dict(artifacts.shared.opts.data, {
            'chat_upload_dir': '',
            'temp_dir': str(configured_temp_dir),
        }):
            root = artifacts.storage_root()

        self.assertEqual(root, expected_root.resolve())
        self.assertTrue(root.is_dir())

    def test_allowed_image_import_copies_validated_content_to_private_storage(self):
        source = self.allowed_dir / 'source image.jpg'
        Image.new('RGB', (2, 2), 'red').save(source, format='PNG')
        with patch.object(artifacts.shared, 'demo', SimpleNamespace(allowed_paths=[str(self.allowed_dir)])), \
                patch.object(artifacts, 'storage_root', return_value=self.artifact_dir):
            imported = artifacts.copy_allowed_file(str(source))

            self.assertEqual(imported['kind'], 'image')
            self.assertEqual(imported['media_type'], 'image/png')
            self.assertEqual(imported['display_name'], 'source image.jpg')
            self.assertNotEqual(imported['path'], str(source))
            self.assertTrue(Path(imported['path']).is_file())
            self.assertEqual(Path(imported['path']).suffix, '.png')
            with self.assertRaisesRegex(ValueError, 'unavailable or not allowed'):
                artifacts.copy_allowed_file(str(Path(self.temp_dir) / 'outside.png'))

    def test_relative_server_path_rejects_ambiguous_matches(self):
        first = self.allowed_dir / 'first'
        second = self.allowed_dir / 'second'
        first.mkdir()
        second.mkdir()
        Image.new('RGB', (1, 1), 'blue').save(first / 'same.png')
        Image.new('RGB', (1, 1), 'green').save(second / 'same.png')
        with patch.object(artifacts.shared, 'demo', SimpleNamespace(allowed_paths=[str(first), str(second)])):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                artifacts.resolve_allowed_path('same.png')

    def test_text_validation_enforces_size_and_utf8(self):
        text_path = self.allowed_dir / 'commands.md'
        text_path.write_text('set steps=12\n', encoding='utf-8')
        validated = artifacts.inspect_file(text_path, text_path.name)
        self.assertEqual(validated['kind'], 'text')
        self.assertEqual(validated['content'], 'set steps=12\n')

        large_path = self.allowed_dir / 'large.txt'
        large_path.write_bytes(b'x' * (artifacts.TEXT_LIMIT + 1))
        with self.assertRaisesRegex(ValueError, 'must not exceed 1 MB'):
            artifacts.inspect_file(large_path, large_path.name)

    def test_image_metadata_separates_parsed_parameters_from_raw_items(self):
        path = self.allowed_dir / 'metadata.png'
        metadata = PngInfo()
        metadata.add_text('parameters', 'a landscape\nSteps: 20, Seed: 42, Sampler: Euler')
        metadata.add_text('source', 'test fixture')
        Image.new('RGB', (2, 2), 'green').save(path, pnginfo=metadata)

        rendered = artifacts.image_metadata({'storage_ref': str(path)})

        self.assertIn('Parsed:', rendered)
        self.assertIn('"Steps": 20', rendered)
        self.assertIn('Raw JSON:', rendered)
        self.assertIn('"source": "test fixture"', rendered)
        self.assertNotIn('"parameters"', rendered)


class TestChatControlParameterImports(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory
    db_path: str

    def setUp(self):
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.db_path = os.path.join(self.temp_dir, 'chat.db')

    def _create_session(self, context='image'):
        with patch.object(db, '_db_path', return_value=self.db_path):
            return chat_api.create_session(chat_api.ReqSessionCreate(context=context))

    def test_import_persists_only_non_default_values_and_preserves_other_drafts(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.update_session(session['id'], drafts={
                'image': {'cfg_scale': 2.0},
                'video': {'steps': 8},
            })
            result = chat_api.import_control_settings(
                session['id'],
                {'prompt': 'imported prompt', 'steps': 4, 'cfg_scale': 6.0, 'sampler_name': 'Euler'},
                {'prompt': '', 'steps': 20, 'cfg_scale': 6.0, 'sampler_name': 'Default'},
            )

        self.assertEqual(result['imported'], ['prompt', 'sampler_name', 'steps'])
        self.assertEqual(result['session']['drafts']['image'], {
            'cfg_scale': 2.0,
            'prompt': 'imported prompt',
            'steps': 4,
            'sampler_name': 'Euler',
        })
        self.assertEqual(result['session']['drafts']['video'], {'steps': 8})
        self.assertEqual(result['session']['active_settings_revision_id'], result['revision_id'])
        revision = db.get_settings_revision(session['id'], result['revision_id'], db_path=self.db_path)
        self.assertEqual(revision['snapshot']['schema_version'], 1)
        self.assertEqual(revision['snapshot']['context'], 'image')
        self.assertEqual(revision['snapshot']['source'], 'control-ui')
        self.assertEqual(revision['snapshot']['parameters'], {
            'prompt': 'imported prompt', 'steps': 4, 'sampler_name': 'Euler',
        })
        self.assertEqual(revision['snapshot']['extras'], {})

    def test_import_revisions_preserve_chat_edits_and_restore_sparse_defaults(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            first = chat_api.import_control_settings(session['id'], {'steps': 30}, {'steps': 20}, extras={})
            db.update_session(session['id'], drafts={'image': {'steps': 24, 'prompt': 'keep prompt'}})
            self.assertEqual(commands.execute_command(session['id'], 'get steps')['text'], 'steps=24')
            second = chat_api.import_control_settings(session['id'], {'steps': 20}, {'steps': 20}, extras={})

            self.assertEqual(second['session']['drafts']['image'], {'prompt': 'keep prompt'})
            self.assertEqual(db.get_settings_revision(session['id'], second['revision_id'])['snapshot']['parameters'], {})

            restored_first = chat_api.restore_control_settings_revision(session['id'], first['revision_id'])
            self.assertEqual(restored_first['session']['drafts']['image'], {'steps': 30, 'prompt': 'keep prompt'})
            restored_second = chat_api.restore_control_settings_revision(session['id'], second['revision_id'])

        self.assertEqual(restored_second['session']['active_settings_revision_id'], second['revision_id'])
        self.assertEqual(restored_second['session']['drafts']['image'], {'prompt': 'keep prompt'})

    def test_import_persists_sparse_extras_and_clears_previous_snapshot_when_defaulted(self):
        session = self._create_session()
        script_info = SimpleNamespace(
            name='example-extra',
            is_alwayson=True,
            args=[SimpleNamespace(label='Mode', value='fast', choices=['fast', 'quality'])],
        )
        runner = SimpleNamespace(alwayson_scripts=[SimpleNamespace(api_info=script_info)])
        extras = {
            'example-extra': {
                'argument_schema': [
                    {'index': 0, 'label': 'Mode', 'default': 'fast', 'choices': ['fast', 'quality']},
                ],
                'arguments': [{'index': 0, 'label': 'Mode', 'value': 'quality'}],
            },
        }

        with patch.object(__import__('modules.scripts_manager', fromlist=['scripts_control']), 'scripts_control', runner), \
            patch.object(db, '_db_path', return_value=self.db_path):
            result = chat_api.import_control_settings(session['id'], {'steps': 4}, {'steps': 20}, extras=extras)
            self.assertEqual(result['extras'], ['example-extra'])
            self.assertEqual(result['session']['drafts']['image']['control_extras'], extras)
            db.update_session(session['id'], drafts={'image': {'control_extras': extras}})
            cleared = chat_api.import_control_settings(session['id'], {}, {}, extras={})

        self.assertNotIn('control_extras', cleared['session']['drafts']['image'])
        self.assertEqual(cleared['extras'], [])

    def test_import_rejects_changed_extra_schema_without_partial_update(self):
        session = self._create_session()
        script_info = SimpleNamespace(
            name='example-extra',
            is_alwayson=True,
            args=[SimpleNamespace(label='Mode', value='fast', choices=['fast', 'quality'])],
        )
        runner = SimpleNamespace(alwayson_scripts=[SimpleNamespace(api_info=script_info)])
        stale_extras = {
            'example-extra': {
                'argument_schema': [
                    {'index': 0, 'label': 'Mode', 'default': 'slow', 'choices': ['slow', 'quality']},
                ],
                'arguments': [{'index': 0, 'label': 'Mode', 'value': 'quality'}],
            },
        }

        with patch.object(__import__('modules.scripts_manager', fromlist=['scripts_control']), 'scripts_control', runner), \
            patch.object(db, '_db_path', return_value=self.db_path):
            with self.assertRaisesRegex(chat_api.HTTPException, 'schema changed'):
                chat_api.import_control_settings(session['id'], {'steps': 4}, {'steps': 20}, extras=stale_extras)
            unchanged = db.get_session(session['id'])

        self.assertEqual(unchanged['drafts'], {})
        self.assertEqual(db.list_settings_revisions(session['id'], db_path=self.db_path), [])

    def test_import_requires_image_context(self):
        session = self._create_session(context='video')
        with patch.object(db, '_db_path', return_value=self.db_path):
            with self.assertRaisesRegex(chat_api.HTTPException, 'Not yet implemented for video context'):
                chat_api.import_control_settings(session['id'], {'steps': 4}, {'steps': 20})

    def test_save_chat_settings_revision_snapshots_sparse_image_draft(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.update_session(session['id'], drafts={'image': {'steps': 31, 'prompt': 'draft prompt', 'unrelated': True}})
            result = chat_api.save_chat_settings_revision(session['id'])
            revision = db.get_settings_revision(session['id'], result['revision']['id'])

        self.assertEqual(result['session']['active_settings_revision_id'], result['revision']['id'])
        self.assertEqual(result['session']['drafts']['image'], {'steps': 31, 'prompt': 'draft prompt', 'unrelated': True})
        self.assertEqual(revision['snapshot']['source'], 'chat-draft')
        self.assertEqual(revision['snapshot']['parameters'], {'steps': 31, 'prompt': 'draft prompt'})


class TestChatFileImports(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory
    db_path: str
    allowed_dir: Path
    private_dir: Path

    def setUp(self):
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.db_path = os.path.join(self.temp_dir, 'chat.db')
        self.allowed_dir = Path(self.temp_dir) / 'allowed'
        self.allowed_dir.mkdir()
        self.private_dir = Path(self.temp_dir) / 'private'
        self.private_dir.mkdir(mode=0o700)

    def _create_session(self):
        with patch.object(db, '_db_path', return_value=self.db_path):
            return chat_api.create_session(chat_api.ReqSessionCreate())

    def test_image_upload_creates_private_selected_artifact(self):
        session = self._create_session()
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        upload = UploadFile(filename='pasted-image.png', file=io.BytesIO(image_bytes))
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir):
            submission = asyncio.run(chat_api.upload_chat_file(session['id'], upload))
            stored = db.get_artifact(session['id'], submission.session.selected_image_id)
            messages = db.list_messages(session['id'])

        self.assertEqual(stored['display_name'], 'pasted-image.png')
        self.assertEqual(stored['role'], 'input')
        self.assertNotEqual(stored['storage_ref'], str(self.allowed_dir))
        self.assertTrue(Path(stored['storage_ref']).is_file())
        self.assertEqual(Path(stored['storage_ref']).suffix, '.png')
        self.assertEqual(submission.artifacts[0].id, stored['id'])
        self.assertEqual(submission.artifacts[0].message_id, submission.user_message.id)
        self.assertNotIn('storage_ref', submission.model_dump_json())
        self.assertNotIn(image_bytes.hex(), str(messages))

    def test_generated_fallback_artifact_uses_namegen_and_detected_extension(self):
        session = self._create_session()
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'blue').save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode('ascii')
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir), \
            patch.dict(artifacts.shared.opts.data, {'samples_filename_pattern': '[seq]-[prompt_words]'}), \
                patch('modules.image.namegen.seq', 0):
            message = db.create_message(session['id'], 'user', 'generate')
            artifact = artifacts.create_generated_artifact(
                session['id'], message['id'], image_data, 'generated-image.jpg',
            )

        storage_path = Path(artifact['storage_ref'])
        self.assertEqual(storage_path.suffix, '.png')
        self.assertRegex(storage_path.name, r'^\d{5}-generated image\.png$')
        self.assertTrue(storage_path.is_file())

    def test_artifact_content_and_delete_are_session_scoped_and_immediate(self):
        session = self._create_session()
        other_session = self._create_session()
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
        upload = UploadFile(filename='owned.png', file=io.BytesIO(buffer.getvalue()))
        with patch.object(db, '_db_path', return_value=self.db_path), \
            patch.object(artifacts, 'storage_root', return_value=self.private_dir), \
            patch.object(chat_api.log, 'warning') as log_warning, \
            patch.object(chat_api.log, 'error') as log_error:
            submission = asyncio.run(chat_api.upload_chat_file(session['id'], upload))
            artifact_id = submission.artifacts[0].id
            content = chat_api.get_chat_artifact_content(session['id'], artifact_id)
            content_bytes = Path(content.path).read_bytes()
            stored = db.get_artifact(session['id'], artifact_id)
            with self.assertRaises(Exception) as content_error:
                chat_api.get_chat_artifact_content(other_session['id'], artifact_id)
            with self.assertRaises(Exception) as ownership_error:
                chat_api.delete_chat_artifact(other_session['id'], artifact_id)
            deleted = chat_api.delete_chat_artifact(session['id'], artifact_id)
            remaining_artifact = db.get_artifact(session['id'], artifact_id)
            updated_session = db.get_session(session['id'])

        self.assertEqual(content_bytes, buffer.getvalue())
        self.assertEqual(content_error.exception.status_code, 404)
        self.assertEqual(ownership_error.exception.status_code, 404)
        self.assertEqual(deleted, {'deleted': True, 'id': artifact_id})
        log_warning.assert_called_once_with(f'Chat: Delete: image="{stored["storage_ref"]}"')
        self.assertEqual(
            [call.args[0] for call in log_error.call_args_list],
            [
                'Chat: image content request: artifact unavailable',
                'Chat: image deletion: artifact not found',
            ],
        )
        self.assertFalse(Path(stored['storage_ref']).exists())
        self.assertIsNone(remaining_artifact)
        self.assertIsNone(updated_session['selected_image_id'])

    def test_artifact_open_folder_and_save_are_session_scoped(self):
        session = self._create_session()
        other_session = self._create_session()
        path = self.private_dir / 'actions.png'
        Image.new('RGB', (2, 2), 'green').save(path)
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch('platform.system', return_value='Linux'), \
                patch.object(shared.cmd_opts, 'hide_ui_dir_config', False), \
                patch('subprocess.Popen') as open_folder, \
                patch('modules.images.save_image', return_value=('/outputs/actions.png', None, '')) as save_image:
            artifact = db.create_artifact(
                session['id'], 'image/png', path.name, 'result', str(path), db_path=self.db_path,
            )
            opened = chat_api.open_chat_artifact_folder(session['id'], artifact['id'])
            saved = chat_api.save_chat_artifact(session['id'], artifact['id'])
            with self.assertRaises(Exception) as ownership_error:
                chat_api.open_chat_artifact_folder(other_session['id'], artifact['id'])

        open_folder.assert_called_once()
        self.assertIn(str(self.private_dir), open_folder.call_args.args[0])
        save_image.assert_called_once()
        self.assertIsInstance(save_image.call_args.args[0], Image.Image)
        self.assertEqual(opened, {'opened': True})
        self.assertEqual(saved, {'saved': True})
        self.assertNotIn(str(path), str(opened))
        self.assertNotIn(str(path), str(saved))
        self.assertEqual(ownership_error.exception.status_code, 404)

    def test_session_update_selects_only_available_owned_images(self):
        session = self._create_session()
        other_session = self._create_session()
        path = self.private_dir / 'selected.png'
        Image.new('RGB', (2, 2), 'blue').save(path)
        with patch.object(db, '_db_path', return_value=self.db_path):
            artifact = db.create_artifact(
                session['id'], 'image/png', path.name, 'result', str(path), db_path=self.db_path,
            )
            updated = chat_api.update_session(
                session['id'], chat_api.ReqSessionUpdate(selected_image_id=artifact['id']),
            )
            with self.assertRaises(Exception) as ownership_error:
                chat_api.update_session(
                    other_session['id'], chat_api.ReqSessionUpdate(selected_image_id=artifact['id']),
                )
            path.unlink()
            with self.assertRaises(Exception) as unavailable_error:
                chat_api.update_session(
                    session['id'], chat_api.ReqSessionUpdate(selected_image_id=artifact['id']),
                )

        self.assertEqual(updated['selected_image_id'], artifact['id'])
        self.assertEqual(ownership_error.exception.status_code, 400)
        self.assertEqual(unavailable_error.exception.status_code, 400)

    def test_session_update_selects_multiple_available_images(self):
        session = self._create_session()
        paths = [self.private_dir / 'first.png', self.private_dir / 'second.png']
        for path, color in zip(paths, ('red', 'blue')):
            Image.new('RGB', (2, 2), color).save(path)
        with patch.object(db, '_db_path', return_value=self.db_path):
            image_ids = [
                db.create_artifact(session['id'], 'image/png', path.name, 'input', str(path))['id']
                for path in paths
            ]
            updated = chat_api.update_session(
                session['id'], chat_api.ReqSessionUpdate(selected_image_ids=image_ids),
            )

        self.assertEqual(updated['selected_image_ids'], image_ids)
        self.assertEqual(updated['selected_image_id'], image_ids[0])

    def test_artifact_delete_preserves_and_marks_derived_images(self):
        session = self._create_session()
        source_path = self.private_dir / 'source.png'
        derived_path = self.private_dir / 'derived.png'
        Image.new('RGB', (2, 2), 'red').save(source_path)
        Image.new('RGB', (2, 2), 'blue').save(derived_path)
        with patch.object(db, '_db_path', return_value=self.db_path):
            source = db.create_artifact(
                session['id'], 'image/png', source_path.name, 'result', str(source_path), db_path=self.db_path,
            )
            derived = db.create_artifact(
                session['id'], 'image/png', derived_path.name, 'result', str(derived_path),
                source_artifact_id=source['id'], db_path=self.db_path,
            )
            db.set_selected_image(session['id'], source['id'])
            chat_api.delete_chat_artifact(session['id'], source['id'])
            remaining = db.get_artifact(session['id'], derived['id'])
            updated_session = db.get_session(session['id'])

        self.assertFalse(source_path.exists())
        self.assertTrue(derived_path.exists())
        self.assertTrue(remaining['source_deleted'])
        self.assertIsNone(updated_session['selected_image_id'])

    def test_missing_artifact_file_is_reported_as_unavailable(self):
        session = self._create_session()
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'blue').save(buffer, format='PNG')
        upload = UploadFile(filename='missing.png', file=io.BytesIO(buffer.getvalue()))
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir):
            submission = asyncio.run(chat_api.upload_chat_file(session['id'], upload))
            stored = db.get_artifact(session['id'], submission.artifacts[0].id)
            Path(stored['storage_ref']).unlink()
            detail = chat_api.get_session(session['id'])
            restore_errors = chat_api.restore_session(session['id']).restore_errors
            with self.assertRaises(Exception) as content_error:
                chat_api.get_chat_artifact_content(session['id'], stored['id'])

        self.assertFalse(detail.artifacts[0].available)
        self.assertTrue(any('missing.png' in message for message in restore_errors))
        self.assertEqual(content_error.exception.status_code, 404)

    def test_image_commands_list_select_reuse_clear_and_read_metadata(self):
        session = self._create_session()
        message = db.create_message(session['id'], 'user', 'uploaded images', db_path=self.db_path)
        artifacts_by_role = {}
        for role, name in (('input', 'input.png'), ('result', 'result.png')):
            path = self.private_dir / name
            metadata = PngInfo()
            metadata.add_text('parameters', f'{name} prompt')
            Image.new('RGB', (2, 2), 'green').save(path, pnginfo=metadata)
            artifacts_by_role[role] = db.create_artifact(
                session['id'], 'image/png', name, role, str(path), message_id=message['id'],
                db_path=self.db_path,
            )

        with patch.object(db, '_db_path', return_value=self.db_path):
            listed = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='image list'))
            selected = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='image 1'))
            metadata = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='meta'))
            latest_input_path = self.private_dir / 'latest-input.png'
            latest_input_metadata = PngInfo()
            latest_input_metadata.add_text('parameters', 'latest-input.png prompt')
            Image.new('RGB', (2, 2), 'yellow').save(latest_input_path, pnginfo=latest_input_metadata)
            db.create_artifact(
                session['id'], 'image/png', 'latest-input.png', 'input', str(latest_input_path),
                message_id=message['id'], db_path=self.db_path,
            )
            latest_input = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='meta'))
            reused = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='image reuse'))
            cleared = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='image clear'))

        self.assertEqual(len(listed.result.data['artifacts']), 2)
        self.assertEqual(selected.session.selected_image_id, artifacts_by_role['input']['id'])
        self.assertIn('result.png prompt', metadata.assistant_message.content)
        self.assertIn('latest-input.png prompt', latest_input.assistant_message.content)
        self.assertEqual(reused.session.selected_image_id, artifacts_by_role['result']['id'])
        self.assertIsNone(cleared.session.selected_image_id)

    def test_image_save_saves_latest_result_batch(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            old_message = db.create_message(session['id'], 'user', 'old results')
            latest_message = db.create_message(session['id'], 'user', 'latest results')
            input_message = db.create_message(session['id'], 'user', 'input image')

            def add_artifact(name, role, message_id):
                path = self.private_dir / name
                metadata = PngInfo()
                metadata.add_text('parameters', f'{name} prompt')
                Image.new('RGB', (2, 2), 'red').save(path, pnginfo=metadata)
                db.create_artifact(
                    session['id'], 'image/png', name, role, str(path), message_id=message_id,
                )

            add_artifact('old-result.png', 'result', old_message['id'])
            add_artifact('latest-one.png', 'result', latest_message['id'])
            add_artifact('latest-two.png', 'result', latest_message['id'])
            add_artifact('newer-input.png', 'input', input_message['id'])
            with patch('modules.images.save_image', side_effect=[
                ('/saved/latest-one.png', None, ''),
                ('/saved/latest-two.png', None, ''),
            ]) as save_image:
                submission = chat_api.post_message(
                    session['id'], chat_api.ReqChatMessage(content='image save'),
                )

        self.assertEqual(save_image.call_count, 2)
        self.assertEqual(
            [call.kwargs['info'] for call in save_image.call_args_list],
            ['latest-one.png prompt', 'latest-two.png prompt'],
        )
        self.assertIn('Saved 2 image(s)', submission.assistant_message.content)
        self.assertIn('/saved/latest-one.png', submission.assistant_message.content)
        self.assertIn('/saved/latest-two.png', submission.assistant_message.content)

    def test_text_upload_executes_lines_and_stops_at_first_failure(self):
        session = self._create_session()
        upload = UploadFile(
            filename='commands.txt',
            file=io.BytesIO(b'set steps=12\nset steps=999\nset seed=10\n'),
        )
        execute_command = chat_api._execute_chat_command
        progress_texts = []

        def capture_progress(*args, **kwargs):
            if shared.state.textinfo and shared.state.textinfo.startswith('Processing...'):
                progress_texts.append(shared.state.textinfo)
            return execute_command(*args, **kwargs)

        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir), \
                patch.object(shared.state, 'job', ''), \
                patch.object(shared.state, 'textinfo', None), \
                patch.object(chat_api, '_execute_chat_command', side_effect=capture_progress):
            submission = asyncio.run(chat_api.upload_chat_file(session['id'], upload))
            saved = db.get_session(session['id'])

        self.assertEqual(submission.result.kind, 'error')
        self.assertIn('<span class="chat-file-line">commands.txt:1</span>', submission.assistant_message.content)
        self.assertIn('steps set to 12', submission.assistant_message.content)
        self.assertIn('<span class="chat-file-line">commands.txt:2</span>', submission.assistant_message.content)
        self.assertNotIn('<pre>', submission.assistant_message.content)
        self.assertNotIn('<br>', submission.assistant_message.content)
        self.assertNotIn('commands.txt:3', submission.assistant_message.content)
        self.assertEqual(progress_texts, [
            'Processing... line 1/3: set steps=12',
            'Processing... line 1/3: set steps=12\nline 2/3: set steps=999',
        ])
        self.assertEqual(submission.assistant_message.metadata['format'], 'html')
        self.assertEqual(saved['drafts']['image']['steps'], 12)
        self.assertNotIn('seed', saved['drafts']['image'])
        self.assertEqual(list(self.private_dir.iterdir()), [])

    def test_text_upload_renders_hash_and_slash_comment_lines(self):
        session = self._create_session()
        upload = UploadFile(
            filename='comments.txt',
            file=io.BytesIO(b'# first comment\n  // second comment\nset steps=12\n'),
        )
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir):
            submission = asyncio.run(chat_api.upload_chat_file(session['id'], upload))

        self.assertEqual(submission.result.kind, 'text')
        self.assertIn('Comment: first comment', submission.assistant_message.content)
        self.assertIn('Comment: second comment', submission.assistant_message.content)
        self.assertIn('steps set to 12', submission.assistant_message.content)
        self.assertNotIn('</div>\n<div>', submission.assistant_message.content)

    def test_text_upload_executes_endpoint_commands_and_continues(self):
        session = self._create_session()
        upload = UploadFile(
            filename='endpoint.txt',
            file=io.BytesIO(b'memory\nset steps=12\n'),
        )
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir), \
                patch('modules.api.server.get_memory', return_value={'ram': {'used': 123}}):
            submission = asyncio.run(chat_api.upload_chat_file(session['id'], upload))

        self.assertEqual(submission.result.kind, 'text')
        self.assertIn('memory:', submission.assistant_message.content)
        self.assertIn('&quot;used&quot;: 123', submission.assistant_message.content)
        self.assertIn('steps set to 12', submission.assistant_message.content)

    def test_server_path_import_copies_image_and_never_persists_path(self):
        session = self._create_session()
        source = self.allowed_dir / 'private source.png'
        Image.new('RGB', (2, 2), 'blue').save(source)
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir), \
                patch.object(shared, 'demo', SimpleNamespace(allowed_paths=[str(self.allowed_dir)])):
            submission = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content=f'use "{source}"'))
            messages = db.list_messages(session['id'])
            stored = db.get_artifact(session['id'], submission.session.selected_image_id)

        self.assertEqual(submission.result.kind, 'text')
        self.assertEqual(submission.user_message.content, 'Server file import')
        self.assertNotIn(str(source), submission.model_dump_json())
        self.assertNotIn(str(source), str(messages))
        self.assertNotEqual(stored['storage_ref'], str(source))
        self.assertTrue(source.is_file())

    def test_server_text_import_renders_run_html(self):
        session = self._create_session()
        source = self.allowed_dir / 'chat-test.txt'
        source.write_text('prompt a landscape\nrun\n', encoding='utf-8')
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
        encoded_image = base64.b64encode(buffer.getvalue()).decode('ascii')
        control_api = Mock(post_control=Mock(return_value=SimpleNamespace(
            images=[encoded_image], info='generated <b>details</b>', params={},
        )))
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(artifacts, 'storage_root', return_value=self.private_dir), \
                patch.object(shared, 'demo', SimpleNamespace(allowed_paths=[str(self.allowed_dir)])):
            submission = chat_api.post_message(
                session['id'], chat_api.ReqChatMessage(content=f'file "{source}"'),
                SimpleNamespace(control=control_api),
            )

        self.assertEqual(submission.result.kind, 'text')
        self.assertEqual(submission.assistant_message.metadata['format'], 'html')
        self.assertIn('<span class="chat-file-line">chat-test.txt:2</span>', submission.assistant_message.content)
        self.assertIn('generated <b>details</b>', submission.assistant_message.content)
        self.assertNotIn('<pre>', submission.assistant_message.content)
        self.assertNotIn('<br>', submission.assistant_message.content)
        self.assertNotIn('&lt;b&gt;details', submission.assistant_message.content)


class TestChatApiRegistration(unittest.TestCase):
    def test_registers_only_session_routes_and_initializes_database(self):
        app = FastAPI()
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                chat_api.register_api(app)
            self.assertTrue(os.path.isfile(db_path))

        registered = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, 'methods', set())
            if route.path.startswith('/sdapi/v1/chat/')
        }
        self.assertEqual(registered, {
            ('/sdapi/v1/chat/sessions', 'GET'),
            ('/sdapi/v1/chat/sessions', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}', 'GET'),
            ('/sdapi/v1/chat/sessions/{session_id}', 'PATCH'),
            ('/sdapi/v1/chat/sessions/{session_id}', 'DELETE'),
            ('/sdapi/v1/chat/sessions/{session_id}/parameters', 'GET'),
            ('/sdapi/v1/chat/sessions/{session_id}/choices', 'GET'),
            ('/sdapi/v1/chat/sessions/{session_id}/restore', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/messages', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/messages/{message_id}', 'PATCH'),
            ('/sdapi/v1/chat/sessions/{session_id}/files', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}/content', 'GET'),
            ('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}/open-folder', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}/save', 'POST'),
            ('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}', 'DELETE'),
        })

    def test_registration_logs_database_diagnostics_when_requested(self):
        app = FastAPI()
        stats = {'ok': False, 'missing_tables': ['messages']}
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch.object(db, 'diagnostics', return_value=stats) as diagnostics, \
                    patch.object(chat_api.log, 'info') as log_info, \
                    patch.object(chat_api.log, 'error') as log_error:
                self.assertTrue(chat_api.register_api(app, diag=True))

        diagnostics.assert_called_once_with()
        log_info.assert_any_call(f'Chat database diagnostics: {stats}')
        log_error.assert_called_once_with('Chat: database diagnostics found integrity or schema issues')

    def test_session_parameter_catalog_matches_the_active_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                image_session = chat_api.create_session(chat_api.ReqSessionCreate(context='image'))
                video_session = chat_api.create_session(chat_api.ReqSessionCreate(context='video'))
                image_parameters = chat_api.get_session_parameters(image_session['id'])
                video_parameters = chat_api.get_session_parameters(video_session['id'])

        self.assertEqual(image_parameters.context, 'image')
        self.assertEqual(video_parameters.context, 'video')
        self.assertIn('steps', {parameter.name for parameter in image_parameters.parameters})
        self.assertIn('frames', {parameter.name for parameter in video_parameters.parameters})

    def test_parameter_choices_match_the_command_validator_catalog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='image'))
                with patch.object(commands.endpoints, 'get_prompt_styles', return_value=[{'name': 'Cycle Style'}]):
                    styles = chat_api.get_session_parameter_choices(session['id'], name='styles')
                with patch.object(commands.endpoints, 'get_sd_models', return_value=[SimpleNamespace(title='Base Model')]):
                    models = chat_api.get_session_parameter_choices(session['id'], name='model')

        self.assertEqual(styles.name, 'styles')
        self.assertEqual(styles.choices, ['Cycle Style'])
        self.assertEqual(models.name, 'model')
        self.assertEqual(models.choices, ['Base Model'])

    def test_chat_progress_stays_active_while_shared_job_is_cleared(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())

                def execute_command(*_args, **_kwargs):
                    response = progress.api_progress(progress.ProgressRequest(id_task='undefined', id_live_preview=-1))
                    self.assertTrue(response.active)
                    return {'kind': 'text', 'text': 'done', 'data': {}}

                with patch.object(shared.state, 'job', ''), \
                        patch.object(chat_api, '_execute_chat_command', side_effect=execute_command):
                    submission = chat_api.post_message(
                        session['id'], chat_api.ReqChatMessage(content='help'),
                    )
                    response = progress.api_progress(
                        progress.ProgressRequest(id_task='undefined', id_live_preview=-1),
                    )

        self.assertEqual(submission.result.kind, 'text')
        self.assertFalse(response.active)

    def test_message_submission_persists_command_exchange_and_restore_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='set steps=24'),
                )
                detail = chat_api.get_session(session['id'])
                restored = chat_api.restore_session(session['id'])

                self.assertEqual(submission.session.title, 'set steps=24')
                self.assertEqual(submission.user_message.content, 'set steps=24')
                self.assertEqual(submission.assistant_message.content, 'steps set to 24')
                self.assertEqual(submission.result.kind, 'text')
                self.assertEqual(detail.session.drafts['image']['steps'], 24)
                self.assertEqual([message.role for message in detail.messages], ['user', 'assistant'])
                self.assertEqual(restored.restore_errors, [])
                self.assertEqual(restored.session.id, session['id'])

                with patch.object(chat_api.log, 'error') as log_error:
                    invalid = chat_api.post_message(
                        session['id'],
                        chat_api.ReqChatMessage(content='set steps=999'),
                    )
                log_error.assert_called_once_with('Chat: command: steps must be between 1 and 200')
                self.assertEqual(invalid.result.kind, 'error')
                self.assertEqual(invalid.assistant_message.metadata['kind'], 'error')

                endpoint = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='version'))
                self.assertEqual(endpoint.result.kind, 'endpoint')
                self.assertEqual(endpoint.result.data['endpoints'][0]['path'], '/sdapi/v1/version')
                updated = chat_api.patch_message(
                    session['id'],
                    endpoint.assistant_message.id,
                    chat_api.ReqChatMessageUpdate(
                        content='{"version": "test"}',
                        metadata={'kind': 'endpoint', 'alias': 'version', 'status': 200},
                    ),
                )
                self.assertEqual(updated['content'], '{"version": "test"}')
                self.assertEqual(updated['metadata']['status'], 200)

                self.assertTrue(chat_api.delete_session(session['id']).status_code == 204)
                with self.assertRaises(Exception) as context:
                    chat_api.get_session(session['id'])
                self.assertEqual(getattr(context.exception, 'status_code', None), 404)

    def test_multiline_submission_executes_commands_and_keeps_prompt_newlines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                batch = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='set steps 8\nset cfg_scale 1'),
                )
                prompt = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='prompt this is\nmultiline prompt'),
                )
                detail = chat_api.get_session(session['id'])

        self.assertEqual(batch.result.text, 'steps set to 8\ncfg_scale set to 1.0')
        self.assertEqual(prompt.result.text, 'prompt set to this is\nmultiline prompt')
        self.assertEqual(detail.session.drafts['image']['steps'], 8)
        self.assertEqual(detail.session.drafts['image']['cfg_scale'], 1)
        self.assertEqual(detail.session.drafts['image']['prompt'], 'this is\nmultiline prompt')
        self.assertEqual(len(detail.messages), 4)

    def test_help_list_and_args_are_not_persisted_to_session_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                submissions = [
                    chat_api.post_message(session['id'], chat_api.ReqChatMessage(content=command))
                    for command in ('help', 'list', 'args', 'snapshot', 'snap')
                ]
                detail = chat_api.get_session(session['id'])

        self.assertEqual([item.user_message.content for item in submissions], ['help', 'list', 'args', 'snapshot', 'snap'])
        self.assertEqual([item.assistant_message.content for item in submissions][0], submissions[0].result.text)
        self.assertEqual(detail.messages, [])
        self.assertEqual(detail.session.title, 'New chat')

    def test_snapshot_restore_is_persisted_as_a_chat_exchange(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                revision = db.create_settings_revision(
                    session['id'],
                    {
                        'schema_version': 1,
                        'context': 'image',
                        'source': 'control-ui',
                        'captured_at': '2026-11-07T09:00:00+00:00',
                        'parameters': {'steps': 30},
                        'extras': {},
                    },
                    {'image': {'steps': 30}},
                )
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content=f'snap {revision["revision"]["id"]}'),
                )
                detail = chat_api.get_session(session['id'])
                saved_session = db.get_session(session['id'])

        self.assertEqual(submission.assistant_message.content, f'Restored image settings snapshot #{revision["revision"]["id"]}')
        self.assertEqual(len(detail.messages), 2)
        self.assertEqual(detail.session.drafts['image']['steps'], 30)
        self.assertEqual(saved_session['active_settings_revision_id'], revision['revision']['id'])

    def test_title_command_reads_default_name_and_updates_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                current = chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='title'))
                renamed = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='title A useful name'),
                )

        self.assertEqual(current.assistant_message.content, 'Current session name: New chat')
        self.assertEqual(current.session.title, 'New chat')
        self.assertEqual(renamed.assistant_message.content, 'Session name set to A useful name')
        self.assertEqual(renamed.session.title, 'A useful name')

    def test_last_command_returns_pretty_json_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                with patch.object(commands, '_format_last_processing', return_value={'prompt': 'a test', 'steps': 20}):
                    submission = chat_api.post_message(
                        session['id'], chat_api.ReqChatMessage(content='last'),
                    )

        section = submission.assistant_message.metadata['sections'][0]
        self.assertEqual(section['format'], 'json')
        self.assertEqual(json.loads(section['content']), {'prompt': 'a test', 'steps': 20})
        self.assertNotIn('```', submission.assistant_message.content)

    def test_list_commands_expose_styles_sessions_wildcards_process_and_scripts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                with patch.object(commands.endpoints, 'get_prompt_styles', return_value=[{'name': 'Test style'}]), \
                        patch('modules.ui_extra_networks_wildcards.list_wildcard_names', return_value=['animals/cats']), \
                        patch('modules.scripts_manager.scripts_postproc') as postproc, \
                        patch('modules.scripts_manager.scripts_txt2img') as txt2img, \
                        patch('modules.scripts_manager.scripts_img2img') as img2img, \
                        patch('modules.scripts_manager.scripts_control') as control, \
                        patch('modules.scripts_manager.scripts_video') as video:
                    postproc.scripts_in_preferred_order.return_value = [SimpleNamespace(name='Resize')]
                    txt2img.scripts = [SimpleNamespace(title=lambda: 'Txt script')]
                    img2img.scripts = [SimpleNamespace(title=lambda: 'Img script')]
                    control.scripts = [SimpleNamespace(title=lambda: 'Control script')]
                    video.scripts = [SimpleNamespace(title=lambda: 'Video script')]
                    replies = {
                        command: chat_api.post_message(
                            session['id'], chat_api.ReqChatMessage(content=command),
                        ).assistant_message.content
                        for command in ('list styles', 'list sessions', 'list wildcards', 'list process', 'list scripts')
                    }

        self.assertIn('Test style', replies['list styles'])
        self.assertIn(session['id'], replies['list sessions'])
        self.assertIn('animals/cats', replies['list wildcards'])
        self.assertIn('Resize', replies['list process'])
        for label in ('Txt script', 'Img script', 'Control script', 'Video script'):
            self.assertIn(label, replies['list scripts'])

    def test_image_generation_uses_control_api_and_persists_no_image_bytes(self):
        script_info = SimpleNamespace(
            name='nudenet',
            is_alwayson=True,
            args=[SimpleNamespace(label='Enabled', value=False, choices=None)],
        )
        script_runner = SimpleNamespace(alwayson_scripts=[SimpleNamespace(api_info=script_info)])
        extras = {
            'nudenet': {
                'argument_schema': [
                    {'index': 0, 'label': 'Enabled', 'default': False, 'choices': None},
                ],
                'arguments': [{'index': 0, 'label': 'Enabled', 'value': True}],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.scripts_manager.scripts_control', script_runner):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                revision = db.create_settings_revision(
                    session['id'],
                    {
                        'schema_version': 1,
                        'context': 'image',
                        'source': 'control-ui',
                        'captured_at': '2026-11-07T09:00:00+00:00',
                        'parameters': {'steps': 4},
                        'extras': extras,
                    },
                    {'image': {'steps': 4, 'control_extras': extras}},
                )
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a landscape'))
                response = SimpleNamespace(
                    images=['base64-image'],
                    info='generated <b>info</b>',
                    params={'prompt': 'a landscape'},
                )
                control_api = Mock(post_control=Mock(return_value=response))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(control=control_api),
                )
                operations = db.list_operations(session['id'])
                messages = db.list_messages(session['id'])
                self.assertEqual(submission.session.last_generation_snapshot['prompt'], 'a landscape')
                replay = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='again another landscape'),
                    SimpleNamespace(control=control_api),
                )
                operations = db.list_operations(session['id'])

        self.assertEqual(control_api.post_control.call_count, 2)
        request = control_api.post_control.call_args_list[0].args[0]
        self.assertIsInstance(request, chat_api.api_control.ReqControl)
        self.assertEqual(request.prompt, 'a landscape')
        self.assertEqual(request.inputs, [])
        self.assertTrue(request.send_images)
        self.assertEqual(request.save_images, shared.opts.samples_save)
        self.assertEqual(submission.result.kind, 'operation')
        self.assertEqual(submission.result.data['images'], ['base64-image'])
        self.assertEqual(submission.assistant_message.content, 'Generated 1 image(s)\ngenerated <b>info</b>')
        self.assertEqual(submission.assistant_message.metadata['kind'], 'operation')
        self.assertEqual(submission.assistant_message.metadata['format'], 'html')
        self.assertEqual(operations[0]['status'], 'completed')
        self.assertEqual(operations[0]['result']['image_count'], 1)
        operations_by_command = {operation['command']: operation for operation in operations}
        self.assertEqual(operations_by_command['generate']['snapshot']['settings_revision_id'], revision['revision']['id'])
        self.assertEqual(operations_by_command['again']['snapshot']['settings_revision_id'], revision['revision']['id'])
        self.assertEqual(operations_by_command['generate']['snapshot']['alwayson_scripts'], {'nudenet': {'args': [True]}})
        self.assertEqual(operations_by_command['generate']['snapshot']['prompt'], 'a landscape')
        self.assertEqual(operations_by_command['again']['snapshot']['prompt'], 'another landscape')
        self.assertNotIn('images', operations[0]['result'])
        self.assertNotIn('base64-image', str(messages))
        self.assertEqual(control_api.post_control.call_args.args[0].prompt, 'another landscape')
        self.assertIn('again', [operation['command'] for operation in operations])
        self.assertEqual(replay.result.kind, 'operation')

    def test_run_repeats_and_aggregates_image_results(self):
        image_buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(image_buffer, format='PNG')
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode('ascii')
        responses = [
            SimpleNamespace(
                images=[encoded_image], info=f'<b>run {index}</b>', stats=f'<div>stats {index}</div>',
                saved_images=[f'/outputs/run-{index}.png'], params={'prompt': 'test landscape'},
            )
            for index in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            artifact_dir = Path(temp_dir) / 'artifacts'
            artifact_dir.mkdir()
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch.object(chat_api.artifacts, 'storage_root', return_value=artifact_dir):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a test landscape'))
                control_api = Mock(post_control=Mock(side_effect=responses))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='run 3'),
                    SimpleNamespace(control=control_api),
                )
                operation = db.list_operations(session['id'])[0]

        self.assertEqual(control_api.post_control.call_count, 3)
        self.assertEqual(len(submission.result.data['images']), 3)
        self.assertEqual(len(submission.artifacts), 3)
        self.assertEqual([artifact.display_name for artifact in submission.artifacts], [
            'run-1.png', 'run-2.png', 'run-3.png',
        ])
        self.assertIn('Generated 3 image(s)', submission.assistant_message.content)
        self.assertIn('<b>run 1</b>', submission.assistant_message.content)
        self.assertIn('<b>run 3</b>', submission.assistant_message.content)
        self.assertIn('<div>stats 1</div>', submission.assistant_message.content)
        self.assertIn('<div>stats 3</div>', submission.assistant_message.content)
        self.assertIn('<b>run 3</b><div>stats 1</div>', submission.assistant_message.content)
        self.assertIn('<div>stats 1</div>\n<div>stats 2</div>', submission.assistant_message.content)
        self.assertEqual(operation['command'], 'run')
        self.assertEqual(operation['result']['run_count'], 3)
        self.assertEqual(operation['result']['image_count'], 3)

    def test_run_uses_multiple_selected_uploaded_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            artifact_dir = Path(temp_dir) / 'artifacts'
            artifact_dir.mkdir()
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch.object(artifacts, 'storage_root', return_value=artifact_dir):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                uploaded_ids = []
                encoded_image = ''
                for filename, color in (('first.png', 'red'), ('second.png', 'blue')):
                    image_buffer = io.BytesIO()
                    Image.new('RGB', (2, 2), color).save(image_buffer, format='PNG')
                    image_bytes = image_buffer.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode('ascii')
                    upload = UploadFile(filename=filename, file=io.BytesIO(image_bytes))
                    upload_result = asyncio.run(chat_api.upload_chat_file(session['id'], upload))
                    uploaded_ids.append(upload_result.session.selected_image_id)
                response = SimpleNamespace(images=[encoded_image], info='', params={'prompt': 'test landscape'})
                control_api = Mock(post_control=Mock(return_value=response))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt test landscape'))
                selection = chat_api.post_message(
                    session['id'], chat_api.ReqChatMessage(content='image 1,2'),
                )
                selected_images = [
                    artifacts.read_image_base64(db.get_artifact(session['id'], artifact_id))
                    for artifact_id in selection.session.selected_image_ids
                ]
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='run'),
                    SimpleNamespace(control=control_api),
                )

        request = control_api.post_control.call_args.args[0]
        self.assertEqual(selection.session.selected_image_ids, uploaded_ids)
        self.assertEqual(submission.session.selected_image_ids, uploaded_ids)
        self.assertEqual(request.inputs, selected_images)

    def test_run_count_defaults_and_is_bounded(self):
        self.assertEqual(commands.parse_command('run').arguments['count'], 1)
        self.assertEqual(commands.parse_command('run 3').arguments['count'], 3)
        for value in ('0', '11', 'many', '2 3'):
            with self.subTest(value=value), self.assertRaises(commands.CommandError):
                commands.parse_command(f'run {value}')

    def test_image_generation_records_failure_from_control_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a landscape'))
                control_api = Mock(post_control=Mock(side_effect=RuntimeError('control failed')))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(control=control_api),
                )
                operation = db.list_operations(session['id'])[0]

        self.assertEqual(submission.result.kind, 'error')
        self.assertEqual(operation['status'], 'failed')
        self.assertEqual(operation['error'], 'control failed')
        self.assertEqual(submission.assistant_message.metadata['kind'], 'error')
        self.assertEqual(submission.assistant_message.metadata['status'], 'failed')
        self.assertTrue(submission.assistant_message.metadata['failed'])

    def test_image_generation_keeps_saved_output_paths_private(self):
        saved_path = '/outputs/generated image.png'
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                response = SimpleNamespace(
                    images=['base64-image'], saved_images=[saved_path], info='', params={},
                    stats='<div><p>Elapsed time</p></div>',
                )
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(control=Mock(post_control=Mock(return_value=response))),
                )
                operation = db.list_operations(session['id'])[0]

        self.assertNotIn(saved_path, submission.assistant_message.content)
        self.assertIn('Elapsed time', submission.assistant_message.content)
        self.assertEqual(operation['result']['stats'], response.stats)
        self.assertNotIn(saved_path, str(submission.assistant_message.metadata))
        serialized = chat_api.api_control.ResControl(
            saved_images=[saved_path], stats=response.stats,
        ).model_dump()
        self.assertEqual(serialized['stats'], response.stats)
        self.assertNotIn('saved_images', serialized)

    def test_video_generation_uses_video_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='video'))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a test clip'))
                response = SimpleNamespace(
                    video='video-base64', video_path=None, thumbnail='thumbnail-base64',
                    frames=[], frames_count=8, info='video info', params={'prompt': 'a test clip'},
                )
                video_api = Mock(post_video=Mock(return_value=response))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate'),
                    SimpleNamespace(video=video_api),
                )
                operation = db.list_operations(session['id'])[0]

        request = video_api.post_video.call_args.args[0]
        self.assertEqual(request.prompt, 'a test clip')
        self.assertEqual(submission.result.data['video'], 'video-base64')
        self.assertEqual(submission.result.data['thumbnail'], 'thumbnail-base64')
        self.assertEqual(submission.assistant_message.content, 'Generated video\nvideo info')
        self.assertEqual(operation['context'], 'video')

    def test_process_generation_uses_transient_image_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='process'))
                process_api = Mock(extras_single_image_api=Mock(return_value=SimpleNamespace(
                    image='processed-base64', html_info='processed <b>info</b>',
                )))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate', image='input-image-base64'),
                    SimpleNamespace(process=process_api),
                )
                replay = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='reprocess', image='latest-image-base64'),
                    SimpleNamespace(process=process_api),
                )
                operations = db.list_operations(session['id'])
                messages = db.list_messages(session['id'])

        first_request = process_api.extras_single_image_api.call_args_list[0].args[0]
        replay_request = process_api.extras_single_image_api.call_args_list[1].args[0]
        self.assertEqual(first_request.image, 'input-image-base64')
        self.assertEqual(replay_request.image, 'latest-image-base64')
        self.assertCountEqual([operation['command'] for operation in operations], ['again', 'generate'])
        self.assertTrue(all('image' not in operation['snapshot'] for operation in operations))
        self.assertNotIn('input-image-base64', str(messages))
        self.assertEqual(submission.result.data['images'], ['processed-base64'])
        self.assertEqual(submission.assistant_message.content, 'Image processed\nprocessed <b>info</b>')
        self.assertEqual(replay.result.kind, 'operation')

    def test_caption_generation_uses_transient_image_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.api.caption.post_vqa', return_value=SimpleNamespace(
                        answer='A blue bird', annotated_image=None,
                    )) as post_vqa:
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='caption'))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='generate', image='caption-input-base64'),
                )
                operation = db.list_operations(session['id'])[0]

        request = post_vqa.call_args.args[0]
        self.assertEqual(request.image, 'caption-input-base64')
        self.assertNotIn('image', operation['snapshot'])
        self.assertEqual(submission.result.data['caption'], 'A blue bird')
        self.assertEqual(submission.assistant_message.content, 'A blue bird')

    def test_schedule_dispatches_image_request_to_scheduler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.scheduler.enqueue_control_request', return_value=iter(
                        [(None, None, 'Queued scheduler task task-1', None)]
                    )) as enqueue:
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='image'))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a landscape'))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='schedule'),
                )

        self.assertEqual(submission.result.kind, 'schedule')
        self.assertEqual(submission.assistant_message.content, 'Queued scheduler task task-1')
        self.assertEqual(enqueue.call_args.kwargs['prompt'], 'a landscape')
        self.assertNotIn('send_images', enqueue.call_args.kwargs)
        self.assertNotIn('save_images', enqueue.call_args.kwargs)

    def test_schedule_dispatches_video_request_to_scheduler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path), \
                    patch('modules.scheduler.enqueue_video_request', return_value=[
                        None, None, None, None, 'Queued scheduler task task-2',
                    ]) as enqueue:
                session = chat_api.create_session(chat_api.ReqSessionCreate(context='video'))
                chat_api.post_message(session['id'], chat_api.ReqChatMessage(content='prompt a test clip'))
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='schedule'),
                )

        self.assertEqual(submission.result.kind, 'schedule')
        self.assertEqual(submission.assistant_message.content, 'Queued scheduler task task-2')
        self.assertEqual(enqueue.call_args.args[0], 'video')
        self.assertEqual(enqueue.call_args.kwargs['prompt'], 'a test clip')

    def test_version_two_database_migrates_operation_contexts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with sqlite3.connect(db_path) as connection:
                connection.execute('''
                    CREATE TABLE operations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                        command TEXT NOT NULL,
                        context TEXT NOT NULL CHECK (context IN ('image', 'video')),
                        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
                        snapshot TEXT NOT NULL DEFAULT '{}',
                        result TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT
                    )
                ''')
                connection.execute('CREATE INDEX chat_operations_session_idx ON operations(session_id, created_at DESC, id)')
                connection.execute('PRAGMA user_version = 2')

            with patch.object(db, '_db_path', return_value=db_path):
                db.initialize()
                session = db.create_session(context='caption')
                operation = db.create_operation(session['id'], 'generate', 'caption', {})
                diagnostics = db.diagnostics()

        self.assertEqual(operation['context'], 'caption')
        self.assertEqual(diagnostics['user_version'], db.SCHEMA_VERSION)
        self.assertTrue(diagnostics['ok'])

    def test_database_initialization_failure_disables_chat_without_routes(self):
        app = FastAPI()
        with patch.dict(shared.opts.data, {'chat_enabled': True}), \
                patch.object(db, 'initialize', side_effect=sqlite3.OperationalError('database is locked')), \
                patch.object(chat_api.log, 'error'):
            self.assertFalse(chat_api.register_api(app))
            self.assertFalse(shared.opts.data['chat_enabled'])

        self.assertFalse(any(route.path.startswith('/sdapi/v1/chat/') for route in app.routes))

    def test_help_sections_are_persisted_in_assistant_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'chat.db')
            with patch.object(db, '_db_path', return_value=db_path):
                session = chat_api.create_session(chat_api.ReqSessionCreate())
                submission = chat_api.post_message(
                    session['id'],
                    chat_api.ReqChatMessage(content='help'),
                )

        expected_titles = ['Chat', 'Process', 'Params', 'References', 'Helpers', 'Global', 'Server']
        self.assertEqual([section['title'] for section in submission.result.sections], expected_titles)
        self.assertEqual(
            [section['title'] for section in submission.assistant_message.metadata['sections']],
            expected_titles,
        )
