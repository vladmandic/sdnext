import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from modules import processing_diffusers
from modules.chat import commands, db


class TestChatCommands(unittest.TestCase):
    temp_dir: str
    db_path: str

    def setUp(self):
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.db_path = os.path.join(self.temp_dir, 'chat.db')

    def _create_session(self):
        with patch.object(db, '_db_path', return_value=self.db_path):
            return db.create_session()

    def test_set_get_and_context_drafts_persist(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            result = commands.execute_command(session['id'], 'set steps=30')
            self.assertEqual(result['kind'], 'text')
            self.assertIn('steps set to 30', result['text'])

            result = commands.execute_command(session['id'], 'get steps')
            self.assertEqual(result['text'], 'steps=30')
            commands.execute_command(session['id'], 'context video')
            commands.execute_command(session['id'], 'set frames 24')
            commands.execute_command(session['id'], 'context image')
            saved = db.get_session(session['id'])

        self.assertEqual(saved['context'], 'image')
        self.assertEqual(saved['drafts']['image']['steps'], 30)
        self.assertEqual(saved['drafts']['video']['frames'], 24)

    def test_help_returns_titled_sections_with_descriptions(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            result = commands.execute_command(session['id'], 'help')

        sections = result['sections']
        self.assertEqual([section['title'] for section in sections], ['Process', 'Params', 'Special', 'Global', 'Server'])
        self.assertNotIn('Help\n', result['text'])
        for section in sections:
            with self.subTest(section=section['title']):
                description, command_list = section['content'].split('\n\n', maxsplit=1)
                self.assertTrue(description)
                self.assertTrue(command_list)
                self.assertTrue(all(line.startswith('- ') and ' - ' in line for line in command_list.splitlines()))

    def test_rejects_bad_ranges_and_unknown_parameters_as_chat_errors(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            too_many_steps = commands.execute_command(session['id'], 'set steps=201')
            bad_video_frames = commands.execute_command(session['id'], 'context video')
            commands.execute_command(session['id'], 'set frames=0')
            unknown = commands.execute_command(session['id'], 'set arbitrary=1')

        self.assertIn('between 1 and 200', too_many_steps['text'])
        self.assertEqual(too_many_steps['kind'], 'error')
        self.assertIn('Unsupported parameter', unknown['text'])
        self.assertEqual(unknown['kind'], 'error')
        self.assertEqual(bad_video_frames['text'], 'Context set to video')

    def test_prompt_parsing_preserves_spaces_and_quotes(self):
        parsed = commands.parse_command('set prompt a "quoted" phrase')
        self.assertEqual(parsed.arguments, {'name': 'prompt', 'value': 'a "quoted" phrase'})
        parsed = commands.parse_command('set prompt="a rainy evening"')
        self.assertEqual(parsed.arguments['value'], 'a rainy evening')
        parsed = commands.parse_command('PROMPT a full paragraph with spaces')
        self.assertEqual(parsed.arguments, {'name': 'prompt', 'value': 'a full paragraph with spaces'})
        parsed = commands.parse_command('model "base model" "video unet"')
        self.assertEqual(parsed.arguments['values'], ['base model', 'video unet'])
        parsed = commands.parse_command('negative low contrast')
        self.assertEqual(parsed.arguments, {'name': 'negative_prompt', 'value': 'low contrast'})

    def test_prompt_markers_set_the_current_prompt(self):
        session = self._create_session()
        prompt = 'a full prompt\nwith a second line'
        with patch.object(db, '_db_path', return_value=self.db_path):
            for marker in ('`', "'", '"', '>'):
                with self.subTest(marker=marker):
                    parsed = commands.parse_command(f'{marker}{prompt}')
                    self.assertEqual(parsed.arguments, {'name': 'prompt', 'value': prompt})
                    result = commands.execute_command(session['id'], f'{marker}{prompt}')
                    self.assertEqual(result['kind'], 'text')
            saved = db.get_session(session['id'])

        self.assertEqual(saved['drafts']['image']['prompt'], prompt)

    def test_slash_prefix_is_removed_before_command_parsing(self):
        self.assertEqual(commands.parse_command('/help'), commands.parse_command('help'))
        self.assertEqual(commands.parse_command('/set steps=30'), commands.parse_command('set steps=30'))
        self.assertEqual(commands.parse_command('// a note').name, 'comment')
        with self.assertRaises(commands.CommandError):
            commands.parse_command('/')

    def test_bare_prompt_returns_the_current_prompt(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            commands.execute_command(session['id'], 'prompt a landscape')
            result = commands.execute_command(session['id'], 'prompt')

        self.assertEqual(result['text'], 'prompt=a landscape')

    def test_title_command_reads_and_sets_the_current_session_name(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            current = commands.execute_command(session['id'], 'title')
            updated = commands.execute_command(session['id'], 'title "A named session"')
            saved = db.get_session(session['id'])

        self.assertEqual(current['text'], 'Current session name: New chat')
        self.assertEqual(updated['text'], 'Session name set to A named session')
        self.assertEqual(saved['title'], 'A named session')
        self.assertEqual(commands.parse_command('title').arguments, {'value': None})
        with self.assertRaises(commands.CommandError):
            commands.parse_command('title ""')

    def test_db_command_compacts_and_reports_database_statistics(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.create_message(session['id'], 'user', 'test message')
            result = commands.execute_command(session['id'], 'db')

        self.assertEqual(result['kind'], 'text')
        self.assertIn('Chat database compacted', result['text'])
        self.assertIn(self.db_path, result['text'])
        self.assertEqual(result['data']['sessions'], 1)
        self.assertEqual(result['data']['messages'], 1)
        self.assertEqual(result['data']['operations'], 0)
        self.assertEqual(result['data']['artifacts'], 0)
        self.assertGreater(result['data']['size_bytes'], 0)
        self.assertGreater(result['data']['page_count'], 0)

    def test_set_accepts_to_equals_and_colon_assignment_syntax(self):
        expected = {'name': 'steps', 'value': '8'}
        commands_to_test = ('set steps 8', 'set steps to 8', 'set steps=8', 'set steps = 8', 'set steps: 8')
        for command in commands_to_test:
            with self.subTest(command=command):
                self.assertEqual(commands.parse_command(command).arguments, expected)
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            results = [commands.execute_command(session['id'], command) for command in commands_to_test]
            saved = db.get_session(session['id'])
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(saved['drafts']['image']['steps'], 8)

    def test_opts_accepts_the_same_assignment_syntax_as_set(self):
        expected = {'name': 'sample_setting', 'value': '8'}
        for command in (
                'opts sample_setting 8', 'opts sample_setting to 8', 'opts sample_setting=8',
                'opts sample_setting = 8', 'opts sample_setting: 8'):
            with self.subTest(command=command):
                self.assertEqual(commands.parse_command(command).arguments, expected)
        self.assertEqual(
            commands.parse_command('opts sample_setting').arguments,
            {'name': 'sample_setting', 'value': None},
        )

    def test_bare_get_and_set_list_all_supported_session_drafts(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.update_session(session['id'], drafts={
                'image': {'steps': 30},
                'video': {'frames': 17},
                'caption': {'prompt': 'describe this'},
            })
            get_result = commands.execute_command(session['id'], 'get')
            set_result = commands.execute_command(session['id'], 'set')

        for result in (get_result, set_result):
            self.assertIn('image.steps=30', result['text'])
            self.assertIn('video.frames=17', result['text'])
            self.assertIn('caption.prompt=describe this', result['text'])

    def test_styles_are_parsed_as_a_comma_delimited_list(self):
        session = self._create_session()
        styles = [SimpleNamespace(name='Cinematic'), SimpleNamespace(name='Studio')]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_prompt_styles', return_value=styles):
            result = commands.execute_command(session['id'], 'set style Cinematic, Studio')
            saved = db.get_session(session['id'])

        self.assertIn('styles set to', result['text'])
        self.assertEqual(saved['drafts']['image']['styles'], ['Cinematic', 'Studio'])

    def test_list_choices_are_printed_one_per_line(self):
        session = self._create_session()
        models = [SimpleNamespace(title='Model A'), SimpleNamespace(title='Model B')]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_sd_models', return_value=models):
            result = commands.execute_command(session['id'], 'list model')

        self.assertEqual(result['text'], 'model:\n- Model A\n- Model B')

    def test_bare_list_shows_listable_keys_for_current_context(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            image_result = commands.execute_command(session['id'], 'list')
            commands.execute_command(session['id'], 'context process')
            process_result = commands.execute_command(session['id'], 'list')

        self.assertEqual(
            image_result['text'],
            'Listable keys for image:\n- model\n- sampler_name\n- styles\n- unet\n- vae',
        )
        self.assertEqual(process_result['text'], 'Listable keys for process:\n- model\n- unet\n- vae')

    def test_params_lists_keys_for_current_context(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            image_result = commands.execute_command(session['id'], 'params')
            commands.execute_command(session['id'], 'context video')
            video_result = commands.execute_command(session['id'], 'params')
            commands.execute_command(session['id'], 'context caption')
            caption_result = commands.execute_command(session['id'], 'params')

        self.assertIn('prompt str ""\n', image_result['text'])
        self.assertIn('batch_size int 1\n', image_result['text'])
        self.assertIn('denoising_strength float 0.3\n', image_result['text'])
        self.assertIn('detailer_augment bool None\n', image_result['text'])
        self.assertIn('mp4_video bool true\n', video_result['text'])
        self.assertIn('references list[str] []\n', video_result['text'])
        self.assertIn('init_image str None\n', video_result['text'])
        self.assertEqual(
            caption_result['text'],
            'Valid parameter keys for caption:\n'
            'prompt str unspecified\n'
            'negative_prompt str unspecified',
        )
        self.assertEqual(commands.parse_command('PARAMS').name, 'params')
        self.assertEqual(commands.parse_command('ARGS').name, 'params')
        with self.assertRaises(commands.CommandError):
            commands.parse_command('params image')

        with patch.object(db, '_db_path', return_value=self.db_path):
            commands.execute_command(session['id'], 'context image')
            set_result = commands.execute_command(session['id'], 'set denoising_strength=0.5')
            updated_params = commands.execute_command(session['id'], 'params')
            run_result = commands.execute_command(session['id'], 'run')

        self.assertEqual(set_result['text'], 'denoising_strength set to 0.5')
        set_parameter = next(
            entry for entry in updated_params['data']['parameters']
            if entry['name'] == 'denoising_strength'
        )
        self.assertEqual(set_parameter, {
            'name': 'denoising_strength', 'type': 'float', 'value': '0.5',
            'is_set': True, 'is_none': False,
        })
        self.assertEqual(run_result['data']['endpoints'][0]['arguments']['denoising_strength'], 0.5)

    def test_video_params_follow_request_schema(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            commands.execute_command(session['id'], 'context video')
            params_result = commands.execute_command(session['id'], 'params')
            bounds_result = commands.execute_command(session['id'], 'list frames')
            commands.execute_command(session['id'], 'set mp4_sf=true')
            commands.execute_command(session['id'], 'set init_image=upload:frame')
            saved = db.get_session(session['id'])

        parameter_keys = {entry['name'] for entry in params_result['data']['parameters']}
        self.assertEqual(parameter_keys, set(commands.ReqVideo.model_fields))
        self.assertTrue(all(entry['is_set'] is False for entry in params_result['data']['parameters']))
        init_image = next(
            entry for entry in params_result['data']['parameters']
            if entry['name'] == 'init_image'
        )
        self.assertTrue(init_image['is_none'])
        self.assertTrue(all(
            not line.startswith('- ')
            for line in params_result['text'].splitlines()[1:]
        ))
        self.assertEqual(bounds_result['text'], 'frames: <class \'int\'>; range 1..1024')
        self.assertTrue(saved['drafts']['video']['mp4_sf'])
        self.assertEqual(saved['drafts']['video']['init_image'], 'upload:frame')

        with patch.object(db, '_db_path', return_value=self.db_path):
            updated_params = commands.execute_command(session['id'], 'params')
        set_parameter = next(
            entry for entry in updated_params['data']['parameters']
            if entry['name'] == 'mp4_sf'
        )
        self.assertEqual(set_parameter['value'], 'true')
        self.assertTrue(set_parameter['is_set'])
        self.assertFalse(set_parameter['is_none'])

    def test_opts_get_set_and_bare_list(self):
        session = self._create_session()
        values = {'sample_setting': 'default'}
        option_info = SimpleNamespace(
            default='default', component_args={}, validate=lambda _name, _value: True,
        )
        options = SimpleNamespace(
            data_labels={'sample_setting': option_info},
            secrets={},
            get=lambda name: values.get(name, 'default'),
            set=lambda name, value: values.__setitem__(name, value) or True,
            save=lambda silent=True: None,
        )
        with patch.object(db, '_db_path', return_value=self.db_path), patch.object(commands.shared, 'opts', options):
            empty_list = commands.execute_command(session['id'], 'opts')
            set_result = commands.execute_command(session['id'], 'opts sample_setting a multiword value')
            get_result = commands.execute_command(session['id'], 'opts sample_setting')
            list_result = commands.execute_command(session['id'], 'opts')

        self.assertEqual(empty_list['text'], 'No non-default options are set')
        self.assertIn('sample_setting set to a multiword value', set_result['text'])
        self.assertEqual(get_result['text'], 'sample_setting=a multiword value')
        self.assertIn('sample_setting="a multiword value"', list_result['text'])

    def test_opts_choice_fragments_use_validator_choices(self):
        choices = ['North Light', 'South Light']
        values = {'sample_setting': choices[1]}
        session = self._create_session()
        option_info = SimpleNamespace(
            default=choices[0], component_args={'choices': choices},
            validate=lambda _name, value: value in choices,
        )
        options = SimpleNamespace(
            data_labels={'sample_setting': option_info},
            secrets=set(),
            get=lambda name: values[name],
            set=lambda name, value: values.__setitem__(name, value) or True,
            save=lambda silent=True: None,
        )
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.shared, 'opts', options):
            available = commands.get_parameter_choices('image', 'sample_setting')
            result = commands.execute_command(session['id'], 'opts sample_setting north')

        self.assertEqual(available, choices)
        self.assertIn('sample_setting set to North Light', result['text'])

    def test_process_and_caption_contexts_require_an_image_input(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            for context in ('process', 'caption'):
                commands.execute_command(session['id'], f'context {context}')
                result = commands.execute_command(session['id'], 'run')
                self.assertEqual(result['kind'], 'error')
                self.assertIn(f'{context.capitalize()} requires an image', result['text'])

    def test_special_commands_return_the_documented_placeholder(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            for command in ('enhance', 'dlss', 'upscale'):
                self.assertEqual(commands.execute_command(session['id'], command)['text'], 'Not yet implemented')

    def test_model_alias_validates_choices_and_persists_canonical_option(self):
        session = self._create_session()
        available = [SimpleNamespace(title='model one')]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_sd_models', return_value=available), \
                patch.object(commands.shared.opts, 'set', return_value=True), \
                patch.object(commands.shared.opts, 'save'):
            result = commands.execute_command(session['id'], 'set model=model one')
            saved = db.get_session(session['id'])

        self.assertIn('sd_model_checkpoint set', result['text'])
        self.assertEqual(saved['drafts']['image']['sd_model_checkpoint'], 'model one')

    def test_model_command_resolves_unique_fragment_case_insensitively(self):
        session = self._create_session()
        available = [
            SimpleNamespace(title='Meina Unreal XL 2.0'),
            SimpleNamespace(title='Another checkpoint'),
        ]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_sd_models', return_value=available), \
                patch.object(commands.shared.opts, 'set', return_value=True) as set_setting, \
                patch.object(commands.shared.opts, 'save'):
            result = commands.execute_command(session['id'], 'model meinaunreal')

        self.assertIn('sd_model_checkpoint set to Meina Unreal XL 2.0', result['text'])
        set_setting.assert_called_once_with('sd_model_checkpoint', 'Meina Unreal XL 2.0')

    def test_model_command_rejects_ambiguous_fragments_with_candidates(self):
        session = self._create_session()
        available = [
            SimpleNamespace(title='Meina Unreal XL'),
            SimpleNamespace(title='Meina Mix'),
        ]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_sd_models', return_value=available):
            result = commands.execute_command(session['id'], 'model meina')

        self.assertEqual(result['kind'], 'error')
        self.assertIn('Ambiguous model name', result['text'])
        self.assertIn('Meina Unreal XL', result['text'])
        self.assertIn('Meina Mix', result['text'])

    def test_set_styles_resolves_unique_fragment_case_insensitively(self):
        session = self._create_session()
        available = [{'name': 'Cycle Style'}, {'name': 'Vintage Film'}]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_prompt_styles', return_value=available):
            result = commands.execute_command(session['id'], 'set styles=cycle')
            saved = db.get_session(session['id'])

        self.assertIn("styles set to ['Cycle Style']", result['text'])
        self.assertEqual(saved['drafts']['image']['styles'], ['Cycle Style'])

    def test_parameter_choice_lookup_uses_the_set_validator_choices(self):
        available = [{'name': 'Cycle Style'}, {'name': 'Vintage Film'}]
        with patch.object(commands.endpoints, 'get_prompt_styles', return_value=available):
            choices = commands.get_parameter_choices('image', 'styles')

        self.assertEqual(choices, ['Cycle Style', 'Vintage Film'])

    def test_set_styles_rejects_ambiguous_fragment_with_candidates(self):
        session = self._create_session()
        available = [{'name': 'Portrait Lighting'}, {'name': 'Portrait Film'}]
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(commands.endpoints, 'get_prompt_styles', return_value=available):
            result = commands.execute_command(session['id'], 'set styles=portrait')

        self.assertEqual(result['kind'], 'error')
        self.assertIn('Ambiguous styles', result['text'])
        self.assertIn('Portrait Lighting', result['text'])
        self.assertIn('Portrait Film', result['text'])

    def test_restore_continues_after_an_individual_setting_failure(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.update_session(session['id'], drafts={
                'image': {'sd_model_checkpoint': 'missing model', 'sd_vae': 'Automatic', 'steps': 30},
            })
            with patch.object(commands, '_validate_setting', side_effect=[
                    ValueError('model unavailable'), ('sd_vae', 'Automatic')]), \
                    patch.object(commands, '_apply_setting', return_value=True) as apply_setting:
                failures = commands.restore_session_state(session['id'])

        self.assertEqual(failures, ['sd_model_checkpoint: model unavailable'])
        apply_setting.assert_called_once_with('sd_vae', 'Automatic')

    def test_reset_clears_only_the_active_context_draft(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.update_session(session['id'], drafts={
                'image': {'steps': 30, 'sd_vae': 'Automatic'},
                'video': {'frames': 24},
            })
            with patch.object(commands, '_apply_setting', return_value=True):
                result = commands.execute_command(session['id'], 'reset')
            saved = db.get_session(session['id'])

        self.assertEqual(result['text'], 'Current context draft reset to defaults')
        self.assertNotIn('image', saved['drafts'])
        self.assertEqual(saved['drafts']['video']['frames'], 24)

    def test_last_details_print_only_non_default_processing_parameters(self):
        last = SimpleNamespace(
            steps=20, seed=42, prompt='a landscape', styles=[], init_images=[object()],
            negative_prompt=None, cfg_name='Default', detailer_prompt='  ', script_args=['ignored'],
        )
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path), \
                patch.object(processing_diffusers, 'last_p', last):
            result = commands.execute_command(session['id'], 'details')

        self.assertIn('"seed": 42', result['text'])
        self.assertIn('"prompt": "a landscape"', result['text'])
        self.assertIn('```json\n', result['text'])
        self.assertTrue(result['text'].endswith('\n```'))
        self.assertNotIn('"steps"', result['text'])
        self.assertNotIn('"styles"', result['text'])
        self.assertNotIn('"init_images"', result['text'])
        self.assertNotIn('"negative_prompt"', result['text'])
        self.assertNotIn('"cfg_name"', result['text'])
        self.assertNotIn('"detailer_prompt"', result['text'])
        self.assertNotIn('"script_args"', result['text'])
        self.assertEqual(commands.parse_command('last').name, 'last')
        with self.assertRaises(commands.CommandError):
            commands.parse_command('details now')

    def test_reset_keeps_the_loaded_model_checkpoint(self):
        session = self._create_session()
        loaded_model = SimpleNamespace(sd_checkpoint_info=SimpleNamespace(title='Loaded model'))
        with patch.object(db, '_db_path', return_value=self.db_path), \
            patch.object(type(commands.shared), 'sd_loaded', new_callable=PropertyMock, return_value=True), \
            patch.object(type(commands.shared), 'sd_model', new_callable=PropertyMock, return_value=loaded_model), \
                patch.object(commands.shared.opts, 'get_default', side_effect=lambda name: f'{name}-default'), \
                patch.object(commands, '_apply_setting', return_value=True) as apply_setting:
            commands.execute_command(session['id'], 'reset')

        reset_values = {call.args[0]: call.args[1] for call in apply_setting.call_args_list}
        self.assertEqual(reset_values['sd_model_checkpoint'], 'Loaded model')
        self.assertNotEqual(reset_values['sd_model_checkpoint'], 'sd_model_checkpoint-default')

    def test_generation_dispatches_the_current_image_draft(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            commands.execute_command(session['id'], 'prompt a landscape')
            result = commands.execute_command(session['id'], 'generate')

        self.assertEqual(result['kind'], 'operation')
        request = result['data']['endpoints'][0]
        self.assertEqual(request['path'], '/sdapi/v1/control')
        self.assertEqual(request['arguments']['prompt'], 'a landscape')
        self.assertEqual(request['arguments']['inputs'], [])
        self.assertTrue(request['arguments']['send_images'])
        self.assertFalse(request['arguments']['save_images'])

    def test_again_reuses_snapshot_and_accepts_prompt_override(self):
        session = self._create_session()
        snapshot = {
            'prompt': 'original prompt',
            'negative_prompt': 'blur',
            'steps': 18,
            'width_before': 768,
            'height_before': 1024,
            'inputs': [],
            'send_images': True,
            'save_images': False,
        }
        with patch.object(db, '_db_path', return_value=self.db_path):
            db.update_session(session['id'], last_generation_snapshot=snapshot)
            replay = commands.execute_command(session['id'], 'again')
            replacement = commands.execute_command(session['id'], 'again new prompt')

        self.assertEqual(replay['kind'], 'operation')
        self.assertEqual(replay['data']['endpoints'][0]['path'], '/sdapi/v1/control')
        self.assertEqual(replay['data']['endpoints'][0]['arguments']['prompt'], 'original prompt')
        self.assertEqual(replay['data']['endpoints'][0]['arguments']['steps'], 18)
        self.assertEqual(replacement['data']['endpoints'][0]['arguments']['prompt'], 'new prompt')
        self.assertEqual(replacement['data']['command'], 'again')

    def test_history_lists_only_operations_for_the_current_session(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            empty_history = commands.execute_command(session['id'], 'history 5')
            db.create_operation(session['id'], 'generate', 'image', {'prompt': 'a blue landscape'})
            result = commands.execute_command(session['id'], 'history 5')

        self.assertEqual(empty_history['text'], 'No past operations')
        self.assertIn('generate running - a blue landscape', result['text'])
        self.assertNotIn('No past operations', result['text'])

    def test_video_generation_returns_video_api_descriptor(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            commands.execute_command(session['id'], 'context video')
            commands.execute_command(session['id'], 'prompt a moving landscape')
            result = commands.execute_command(session['id'], 'start')

        self.assertEqual(result['kind'], 'operation')
        descriptor = result['data']['endpoints'][0]
        self.assertEqual(descriptor['path'], '/sdapi/v1/video')
        self.assertEqual(descriptor['arguments']['prompt'], 'a moving landscape')

    def test_caption_and_named_unimplemented_commands(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            for command in ('caption', 'captioning', 'enhance prompt text', 'prompt-enhancement prompt text',
                            'upscale', 'upscaling'):
                result = commands.execute_command(session['id'], command)
                self.assertEqual(result['text'], 'Not yet implemented', command)

    def test_process_control_aliases_call_shared_state(self):
        session = self._create_session()
        state = commands.shared.state
        was_paused = state.paused
        try:
            with patch.object(db, '_db_path', return_value=self.db_path), \
                    patch.object(state, 'interrupt') as interrupt, \
                    patch.object(state, 'skip') as skip, \
                    patch.object(state, 'pause', side_effect=lambda: setattr(state, 'paused', not state.paused)) as pause, \
                    patch.object(state, 'paused', False):
                for alias in ('cancel', 'abort', 'stop', 'interrupt'):
                    commands.execute_command(session['id'], alias)
                commands.execute_command(session['id'], 'pause')
                commands.execute_command(session['id'], 'resume')
                commands.execute_command(session['id'], 'skip')

            self.assertEqual(interrupt.call_count, 4)
            self.assertEqual(pause.call_count, 2)
            skip.assert_called_once_with()
        finally:
            state.paused = was_paused

    def test_only_documented_server_aliases_return_fixed_endpoint_groups(self):
        info = commands.parse_command('Ver')
        self.assertEqual(info.arguments['alias'], 'ver')
        self.assertEqual(
            [endpoint['path'] for endpoint in info.arguments['endpoints']],
            ['/sdapi/v1/version', '/sdapi/v1/platform', '/sdapi/v1/torch', '/sdapi/v1/gpu'],
        )
        status = commands.parse_command('status')
        self.assertEqual(
            [endpoint['path'] for endpoint in status.arguments['endpoints']],
            ['/sdapi/v1/status', '/sdapi/v1/progress'],
        )
        self.assertEqual(status.arguments['endpoints'][1]['arguments'], {'skip_current_image': True})
        smi = commands.parse_command('gpu')
        self.assertEqual(smi.arguments['endpoints'][0]['path'], '/sdapi/v1/gpu-smi')

        for command in ('shutdown', 'restart', 'get-log', 'post-log', 'motd', 'cmd-flags'):
            with self.subTest(command=command), self.assertRaises(commands.CommandError):
                commands.parse_command(command)
        self.assertEqual(commands.parse_command('start').name, 'generate')
        self.assertEqual(commands.parse_command('interrupt').name, 'cancel')
        self.assertEqual(commands.parse_command('skip').name, 'skip')
        self.assertEqual(commands.parse_command('history 10').name, 'history')

    def test_endpoint_group_is_not_executed_and_unknown_draft_fields_are_hidden(self):
        session = self._create_session()
        with patch.object(db, '_db_path', return_value=self.db_path):
            result = commands.execute_command(session['id'], 'info')
            db.update_session(session['id'], drafts={'image': {'steps': 20, 'api_key': 'secret'}})
            current = commands.execute_command(session['id'], 'opts')

        self.assertEqual(result['kind'], 'endpoint')
        self.assertEqual(result['data']['alias'], 'info')
        self.assertEqual(len(result['data']['endpoints']), 4)
        self.assertNotIn('secret', current['text'])
        self.assertNotIn('api_key', current['text'])


if __name__ == '__main__':
    unittest.main()
