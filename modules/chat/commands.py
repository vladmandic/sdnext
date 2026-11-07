"""Deterministic chat command parsing, validation, and session draft handling."""

import inspect
import re
import json
import ast
import shlex
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from modules import shared
from modules.api import endpoints, models
from modules.api import control as control_api
from modules.api.video import ReqVideo
from modules.chat import db
from modules.logger import log


class CommandError(ValueError):
    """Invalid or unsupported chat command input."""


@dataclass(frozen=True)
class EndpointSpec:
    method: str
    path: str
    request_model: type[BaseModel] | None = None
    location: str = 'none'
    effects: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    arguments: dict[str, Any]


SERVER_ENDPOINTS = {
    'version': EndpointSpec('GET', '/sdapi/v1/version'),
    'torch': EndpointSpec('GET', '/sdapi/v1/torch'),
    'status': EndpointSpec('GET', '/sdapi/v1/status'),
    'platform': EndpointSpec('GET', '/sdapi/v1/platform'),
    'progress': EndpointSpec('GET', '/sdapi/v1/progress', models.ReqProgress, 'query'),
    'memory': EndpointSpec('GET', '/sdapi/v1/memory'),
    'gpu': EndpointSpec('GET', '/sdapi/v1/gpu'),
    'gpu-smi': EndpointSpec('GET', '/sdapi/v1/gpu-smi'),
    'storage': EndpointSpec('GET', '/sdapi/v1/storage', models.ReqStorage, 'query'),
}
_SERVER_COMMAND_ALIASES = {
    'info': 'info', 'ver': 'info', 'version': 'info',
    'gpu': 'gpu-smi', 'smi': 'gpu-smi',
    'mem': 'memory', 'memory': 'memory',
    'status': 'status', 'storage': 'storage',
}
_SERVER_COMMAND_GROUPS = {
    'info': ('version', 'platform', 'torch', 'gpu'),
    'gpu-smi': ('gpu-smi',),
    'memory': ('memory',),
    'status': ('status', 'progress'),
    'storage': ('storage',),
}

_IMAGE_MODEL = control_api.ReqControl
_IMAGE_MODEL_FIELDS = _IMAGE_MODEL.model_fields
_IMAGE_PARAMETERS = set(_IMAGE_MODEL_FIELDS)
_VIDEO_MODEL_FIELDS = ReqVideo.model_fields
_VIDEO_PARAMETERS = set(_VIDEO_MODEL_FIELDS)
_IMAGE_RANGES = {
    'width_before': (64, 4096), 'height_before': (64, 4096), 'steps': (1, 200),
    'batch_size': (1, 16), 'batch_count': (1, 100), 'cfg_scale': (0, 100),
    'seed': (-1, 4294967295),
}
_SETTING_ALIASES = {
    'sd_model_checkpoint': ('model',),
    'sd_vae': ('vae',),
    'sd_unet': ('unet', 'dit', 'transformer', 'finetune'),
    'sd_text_encoder': ('te', 'text-encoder'),
}
_SETTING_NAMES = {
    name: canonical
    for canonical, aliases in _SETTING_ALIASES.items()
    for name in (canonical, *aliases)
}
_COMMAND_ALIASES = {
    'args': 'params',
    'snap': 'snapshot',
    'start': 'generate',
    'abort': 'cancel', 'stop': 'cancel', 'interrupt': 'cancel',
    'reprocess': 'again',
    'prompt-enhancement': 'enhance',
    'captioning': 'caption',
    'details': 'last',
    'p': 'last',
    'exif': 'meta',
    'metadata': 'meta',
}
_PARAMETER_ALIASES = {
    'sampler': 'sampler_name', 'prompt': 'prompt', 'negative': 'negative_prompt', 'style': 'styles',
    'width': 'width_before', 'height': 'height_before', 'n_iter': 'batch_count',
}
_OPERATION_COMMANDS = {
    'again', 'caption', 'generate', 'run', 'schedule', 'cancel', 'pause', 'resume', 'skip', 'history', 'reset', 'last',
}
_BUILTIN_COMMANDS = {'context', 'clear', 'set', 'get', 'list', 'params', 'help', 'model', 'opts', 'db', 'snapshot'}


def _command_result(text: str, kind='text', data=None, sections=None):
    result = {'kind': kind, 'text': text, 'data': data or {}}
    if sections is not None:
        result['sections'] = sections
    return result


def _command_error(text: str):
    return _command_result(text, kind='error')


def _split_assignment(text: str) -> tuple[str, str]:
    extra_assignment = re.match(r'^(\S+:\S+?)(?:\s*=\s*|\s+)(.*)$', text, flags=re.DOTALL)
    if extra_assignment:
        name, raw_value = extra_assignment.groups()
        if raw_value.lower() == 'to':
            raw_value = ''
        elif raw_value.lower().startswith('to '):
            raw_value = raw_value[3:]
        return name.strip(), _normalize_value(raw_value.strip())
    assignment = re.match(r'^(\S+?)\s*[=:]\s*(.*)$', text, flags=re.DOTALL)
    if assignment:
        name, raw_value = assignment.groups()
    else:
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            raise CommandError('Use: set <parameter> <value>, set <parameter> to <value>, or set <parameter>=<value>')
        name, raw_value = parts
        if raw_value.lower() == 'to':
            raw_value = ''
        elif raw_value.lower().startswith('to '):
            raw_value = raw_value[3:]
    return name.strip(), _normalize_value(raw_value.strip())


def _normalize_value(raw_value: str) -> str:
    if raw_value[:1] in {'"', "'"}:
        try:
            value_parts = shlex.split(raw_value)
        except ValueError as exc:
            raise CommandError(f'Invalid quoted value: {exc}') from exc
        if len(value_parts) == 1:
            return value_parts[0]
    return raw_value


def _parse_value(raw_value: str, annotation: Any, example: Any = None):
    list_type = None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        list_type = args[0] if args else str
    elif annotation is list or isinstance(example, list):
        list_type = type(example[0]) if example else str
    elif origin is not None and args:
        for arg in args:
            if get_origin(arg) is list:
                list_args = get_args(arg)
                list_type = list_args[0] if list_args else str
                break

    if list_type is not None:
        try:
            values = json.loads(raw_value)
        except json.JSONDecodeError:
            try:
                values = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                values = None
        if isinstance(values, tuple):
            values = list(values)
        if not isinstance(values, list):
            list_text = raw_value.strip()
            if list_text.startswith('[') and list_text.endswith(']'):
                list_text = list_text[1:-1]
            values = [item.strip() for item in list_text.split(',') if item.strip()]
        item_adapter = TypeAdapter(list_type)
        return [item_adapter.validate_python(item) for item in values]
    if annotation is Any or annotation is None:
        return raw_value
    return TypeAdapter(annotation).validate_python(raw_value)


def _model_request(spec: EndpointSpec, raw_arguments: str) -> dict[str, Any]:
    if not raw_arguments.strip():
        payload = {}
    else:
        try:
            payload = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise CommandError(f'Endpoint arguments must be a JSON object: {exc.msg}') from exc
        if not isinstance(payload, dict):
            raise CommandError('Endpoint arguments must be a JSON object')
    if spec.request_model is None:
        if payload:
            raise CommandError('This endpoint does not accept arguments')
        return {}

    fields = spec.request_model.model_fields
    allowed = set(fields)
    allowed.update(field.alias for field in fields.values() if field.alias)
    unknown = set(payload) - allowed
    if unknown:
        raise CommandError(f'Unsupported endpoint argument(s): {", ".join(sorted(unknown))}')
    for name, field in fields.items():
        if name in payload and field.alias and field.alias != name:
            payload[field.alias] = payload.pop(name)
    try:
        request = spec.request_model.model_validate(payload)
    except ValidationError as exc:
        raise CommandError(f'Invalid endpoint arguments: {exc.errors()[0]["msg"]}') from exc
    return request.model_dump(exclude_unset=True, by_alias=True)


def parse_command(text: str) -> ParsedCommand:
    if not isinstance(text, str) or not text.strip():
        raise CommandError('Enter a command. Use help to see available commands.')
    stripped = text.strip()
    if stripped.startswith('#'):
        return ParsedCommand('comment', {'text': stripped[1:].strip()})
    if stripped.startswith('//'):
        return ParsedCommand('comment', {'text': stripped[2:].strip()})
    if stripped[0] in {'`', "'", '"', '>'}:
        value = stripped[1:].strip()
        if not value:
            raise CommandError('Enter prompt text after the prompt marker.')
        return ParsedCommand('set', {'name': 'prompt', 'value': value})
    if stripped.startswith('/'):
        stripped = stripped[1:].lstrip()
        if not stripped:
            raise CommandError('Enter a command. Use help to see available commands.')
    match = re.match(r'^(\S+)(?:\s+(.*))?$', stripped, flags=re.DOTALL)
    command, raw_arguments = match.groups()
    command = command.lower()
    raw_arguments = raw_arguments or ''

    server_command = _SERVER_COMMAND_ALIASES.get(command)
    if server_command is not None:
        if raw_arguments.strip():
            raise CommandError(f'{command} does not accept arguments')
        descriptors = []
        for endpoint_alias in _SERVER_COMMAND_GROUPS[server_command]:
            spec = SERVER_ENDPOINTS[endpoint_alias]
            arguments = _model_request(spec, '{"skip_current_image": true}' if endpoint_alias == 'progress' else '')
            descriptors.append({
                'alias': endpoint_alias,
                'method': spec.method,
                'path': spec.path,
                'location': spec.location,
                'arguments': arguments,
                'effects': list(spec.effects),
            })
        return ParsedCommand('endpoint', {'alias': command, 'endpoints': descriptors})

    command = _COMMAND_ALIASES.get(command, command)
    if command in {'use', 'file'}:
        try:
            paths = shlex.split(raw_arguments)
        except ValueError as exc:
            raise CommandError(f'Invalid file path: {exc}') from exc
        if len(paths) != 1:
            raise CommandError('Use: use <path>')
        return ParsedCommand('import', {'path': paths[0]})
    if command == 'image':
        value = raw_arguments.strip().lower()
        action = value
        if action in {'list', 'save', 'reuse', 'clear'}:
            return ParsedCommand(command, {'action': action})
        if not re.fullmatch(r'\d+(?:\s*,\s*\d+)*', value):
            raise CommandError('Use: image list, image save, image reuse, image clear, or image <number[,number...]>')
        indices = list(dict.fromkeys(int(item.strip()) for item in value.split(',')))
        if any(index < 1 for index in indices):
            raise CommandError('Image number must be positive')
        return ParsedCommand(command, {'action': 'select', 'indices': indices})
    if command == 'meta':
        if raw_arguments.strip():
            raise CommandError('meta does not accept arguments')
        return ParsedCommand(command, {})
    if command == 'db':
        if raw_arguments.strip():
            raise CommandError('db does not accept arguments')
        return ParsedCommand(command, {})
    if command == 'snapshot':
        value = raw_arguments.strip()
        if not value:
            return ParsedCommand(command, {'action': 'list', 'revision_id': None})
        if value.casefold() == 'save':
            return ParsedCommand(command, {'action': 'save', 'revision_id': None})
        if not value.isdecimal() or int(value) < 1:
            raise CommandError('Use: snapshot, snapshot save, or snapshot <positive snapshot ID>')
        return ParsedCommand(command, {'action': 'restore', 'revision_id': int(value)})
    if command == 'context':
        parts = raw_arguments.split()
        if len(parts) > 1 or (parts and parts[0].lower() not in db.CONTEXTS):
            raise CommandError('Use: context image, video, process, or caption')
        return ParsedCommand(command, {'context': parts[0].lower() if parts else None})
    if command == 'title':
        if not raw_arguments.strip():
            return ParsedCommand(command, {'value': None})
        value = _normalize_value(raw_arguments.strip())
        if not value.strip():
            raise CommandError('A session name is required')
        return ParsedCommand(command, {'value': value})
    if command == 'clear':
        parts = raw_arguments.split()
        if len(parts) != 1:
            raise CommandError('Use: clear <parameter>')
        name = parts[0].lower()
        return ParsedCommand(command, {'name': _PARAMETER_ALIASES.get(name, name)})
    if command in {'prompt', 'negative'}:
        value = _normalize_value(raw_arguments.strip())
        if not value:
            return ParsedCommand('get', {'name': 'prompt'})
        name = _PARAMETER_ALIASES[command]
        return ParsedCommand('set', {'name': name, 'value': value})
    if command == 'set':
        if not raw_arguments.strip():
            return ParsedCommand(command, {'name': None, 'value': None})
        name, value = _split_assignment(raw_arguments.strip())
        if not value:
            raise CommandError('A value is required')
        name = name.lower()
        return ParsedCommand(command, {'name': _PARAMETER_ALIASES.get(name, name), 'value': value})
    if command in {'get', 'list'}:
        parts = raw_arguments.split()
        if len(parts) > 1:
            raise CommandError(f'Use: {command} [parameter]')
        name = parts[0].lower() if parts else None
        return ParsedCommand(command, {'name': _PARAMETER_ALIASES.get(name, name) if name else None})
    if command == 'params':
        if raw_arguments.strip():
            raise CommandError('params does not accept arguments')
        return ParsedCommand(command, {})
    if command == 'model':
        try:
            values = shlex.split(raw_arguments)
        except ValueError as exc:
            raise CommandError(f'Invalid model value: {exc}') from exc
        if len(values) > 2:
            raise CommandError('Use: model <model> [unet]')
        return ParsedCommand(command, {'values': values})
    if command == 'opts':
        arguments = raw_arguments.strip()
        if not arguments:
            return ParsedCommand(command, {'name': None, 'value': None})
        parts = arguments.split(maxsplit=1)
        if len(parts) == 1 and not re.search(r'[=:]', arguments):
            return ParsedCommand(command, {'name': parts[0].lower(), 'value': None})
        name, value = _split_assignment(arguments)
        return ParsedCommand(command, {
            'name': name.lower(),
            'value': value,
        })
    if command == 'help':
        if raw_arguments.strip():
            raise CommandError('help does not accept arguments')
        return ParsedCommand(command, {})
    if command in _OPERATION_COMMANDS:
        if command == 'run':
            if not raw_arguments.strip():
                return ParsedCommand(command, {'count': 1})
            try:
                count = int(raw_arguments.strip())
            except ValueError as exc:
                raise CommandError('Use: run [count]') from exc
            if not 1 <= count <= 10:
                raise CommandError('Run count must be between 1 and 10')
            return ParsedCommand(command, {'count': count})
        if command in {'generate', 'schedule', 'cancel', 'pause', 'resume', 'skip', 'reset', 'last'} and raw_arguments.strip():
            raise CommandError(f'{command} does not accept arguments')
        if command == 'history' and raw_arguments.strip():
            try:
                limit = int(raw_arguments.strip())
            except ValueError as exc:
                raise CommandError('Use: history [limit]') from exc
            if not 1 <= limit <= 100:
                raise CommandError('History limit must be between 1 and 100')
            return ParsedCommand(command, {'limit': limit})
        return ParsedCommand(command, {'text': raw_arguments.strip()} if raw_arguments.strip() else {})
    raise CommandError(f'Unknown command: {command}')


def split_commands(text: str) -> list[str]:
    try:
        parsed = parse_command(text)
    except CommandError:
        parsed = None
    if parsed is not None and parsed.name == 'set' and parsed.arguments.get('name') in {'prompt', 'negative_prompt'}:
        return [text.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_transient_command(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith('/') and not stripped.startswith('//'):
        stripped = stripped[1:].lstrip()
    match = re.match(r'^(\S+)', stripped)
    if match is None:
        return False
    command = _COMMAND_ALIASES.get(match.group(1).casefold(), match.group(1).casefold())
    if command == 'snapshot':
        try:
            parsed = parse_command(stripped)
        except CommandError:
            return False
        return parsed.arguments['action'] == 'list'
    return command in {'help', 'list', 'params'}


def _parameter_spec(context: str, name: str):
    if name in _SETTING_NAMES:
        return 'setting', _SETTING_NAMES[name]
    if context in {'process', 'caption'} and name in {'prompt', 'negative_prompt'}:
        return 'parameter', (None, name)
    if context == 'image' and name in _IMAGE_PARAMETERS and name in _IMAGE_MODEL_FIELDS:
        return 'parameter', (_IMAGE_MODEL, name)
    if context == 'video' and name in _VIDEO_PARAMETERS and name in _VIDEO_MODEL_FIELDS.keys():
        return 'parameter', (ReqVideo, name)
    if context == 'image' and ':' in name:
        extra = _control_extra_field(name)
        if extra is not None:
            return 'extra', extra
    raise CommandError(f'Unsupported parameter for {context}: {name}')


def _choice_values(name: str):
    if name in {'sampler_name', 'sampler'}:
        return [item['name'] for item in endpoints.get_samplers()]
    if name == 'scheduler':
        return [item['name'] for item in endpoints.get_schedulers()]
    if name == 'hr_upscaler':
        return [item['name'] for item in endpoints.get_upscalers()]
    if name == 'styles':
        return [item['name'] if isinstance(item, dict) else item.name for item in endpoints.get_prompt_styles()]
    if name in {'model', 'sd_model_checkpoint'}:
        values = [item.title for item in endpoints.get_sd_models()]
        return values
    if name in {'vae', 'sd_vae'}:
        return ['Automatic', 'None', *[item['model_name'] for item in endpoints.get_sd_vaes()]]
    if name in {'unet', 'dit', 'transformer', 'finetune', 'sd_unet'}:
        return ['Default', *[item['name'] for item in endpoints.get_unets()]]
    canonical = _SETTING_NAMES.get(name, name)
    info = shared.opts.data_labels.get(canonical)
    component_args = info.component_args if info else None
    if callable(component_args):
        component_args = component_args()
    choices = component_args.get('choices', []) if isinstance(component_args, dict) else []
    return [item[0] if isinstance(item, (list, tuple)) else item for item in choices]


def get_parameter_choices(context: str, name: str) -> list[str]:
    name = _PARAMETER_ALIASES.get(name, name)
    try:
        kind, spec = _parameter_spec(context, name)
    except CommandError:
        canonical = _SETTING_NAMES.get(name, name)
        if canonical in shared.opts.secrets or canonical.endswith(('_version', '_token', '_key', '_secret', '_password')):
            raise CommandError(f'Setting is not available through chat: {canonical}') from None
        if canonical not in shared.opts.data_labels:
            raise CommandError(f'Unsupported parameter: {name}') from None
        name = canonical
    else:
        if kind == 'setting':
            name = spec
        elif kind == 'extra':
            return [str(choice) for choice in (spec[1].get('choices') or [])]
    return [str(choice) for choice in _choice_values(name)]


def _resolve_choice_fragment(name: str, value: Any, choices: list[Any]):
    if isinstance(value, list):
        return [_resolve_choice_fragment(name, item, choices) for item in value]
    if not isinstance(value, str) or value in choices:
        return value
    fragment = ''.join(character for character in value.casefold() if character.isalnum())
    if not fragment:
        return value
    matches = [
        choice for choice in choices
        if fragment in ''.join(character for character in str(choice).casefold() if character.isalnum())
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        candidates = '\n'.join(str(choice) for choice in matches[:5])
        if len(matches) > 5:
            candidates += ', ...'
        label = 'model name' if name in {'model', 'sd_model_checkpoint'} else name
        raise CommandError(f'Ambiguous {label}:\n{candidates}')
    return value


def _validate_setting(name: str, raw_value: Any) -> tuple[str, Any]:
    canonical = _SETTING_NAMES.get(name)
    if canonical is None:
        raise CommandError(f'Unsupported setting: {name}')
    info = shared.opts.data_labels.get(canonical)
    if info is None:
        raise CommandError(f'Setting is unavailable: {canonical}')
    try:
        annotation = type(info.default) if info.default is not None else type(shared.opts.get(canonical))
        value = _parse_value(raw_value, annotation, info.default)
    except ValidationError as exc:
        raise CommandError(f'Invalid value for {name}: {exc.errors()[0]["msg"]}') from exc
    choices = _choice_values(canonical)
    if choices:
        value = _resolve_choice_fragment(canonical, value, choices)
    current = shared.opts.get(canonical)
    if choices and value != current:
        values = value if isinstance(value, list) else [value]
        if any(item not in choices for item in values):
            raise CommandError(f'Invalid value for {name}\nUse list {name} to see valid choices')
    if not choices and value != current:
        raise CommandError(f'No valid choices are available for {name}')
    return canonical, value


def _validate_parameter(context: str, name: str, raw_value: Any):
    _, (request_model, field_name) = _parameter_spec(context, name)
    if request_model is None:
        return raw_value
    field = request_model.model_fields[field_name]
    try:
        value = _parse_value(raw_value, field.annotation, field.default)
        request_model.model_validate({field_name: value})
    except ValidationError as exc:
        raise CommandError(f'Invalid value for {name}: {exc.errors()[0]["msg"]}') from exc
    if context == 'image' and name in _IMAGE_RANGES:
        minimum, maximum = _IMAGE_RANGES[name]
        if not minimum <= value <= maximum:
            raise CommandError(f'{name} must be between {minimum} and {maximum}')
    choices = _choice_values(name)
    if choices:
        value = _resolve_choice_fragment(name, value, choices)
        values = value if isinstance(value, list) else [value]
        if any(item not in choices for item in values):
            raise CommandError(f'Invalid value for {name}\nUse list {name} to see valid choices')
    return value


def _apply_setting(name: str, value: Any) -> bool:
    current = shared.opts.get(name)
    if current == value:
        return True
    if not shared.opts.set(name, value):
        return False
    shared.opts.save(silent=True)
    return True


def _apply_saved_settings(draft: dict[str, Any]) -> list[str]:
    failures = []
    for name, raw_value in draft.items():
        if name not in _SETTING_NAMES:
            continue
        try:
            canonical, value = _validate_setting(name, raw_value)
            if not _apply_setting(canonical, value):
                failure = f'{name}: setting callback rejected the value'
                failures.append(failure)
                log.error(f'Chat: setting restore failed: {failure}')
        except Exception as exc:
            failure = f'{name}: {exc}'
            failures.append(failure)
            log.error(f'Chat: setting restore failed: {failure}')
    return failures


def restore_session_state(session_id: str) -> list[str]:
    """Reapply supported model settings from the session's active context draft."""
    session = db.get_session(session_id)
    if session is None:
        raise CommandError('Chat session not found')
    draft = (session.get('drafts') or {}).get(session['context'], {})
    return _apply_saved_settings(draft)


def _format_session(session: dict[str, Any]) -> str:
    model = shared.opts.get('sd_model_checkpoint')
    values = [f'context={session["context"]}', f'model={model}']
    allowed_by_context = {
        'image': _IMAGE_PARAMETERS,
        'video': _VIDEO_PARAMETERS,
        'process': {'prompt', 'negative_prompt'},
        'caption': {'prompt', 'negative_prompt'},
    }
    for context, draft in sorted((session.get('drafts') or {}).items()):
        allowed = allowed_by_context.get(context, set())
        values.extend(
            f'{context}.{name}={value}'
            for name, value in sorted(draft.items())
            if name in allowed or name in _SETTING_NAMES
        )
        if context == 'image':
            for script_name, snapshot in (draft.get('control_extras') or {}).items():
                schema = snapshot.get('argument_schema', [])
                keys = _control_extra_keys(script_name, schema)
                for argument in snapshot.get('arguments', []):
                    index = argument.get('index')
                    if isinstance(index, int) and 0 <= index < len(keys):
                        values.append(f'image.{keys[index]}={argument.get("value")}')
    return '\n'.join(values)


def _format_options() -> str:
    values = []
    secret_suffixes = ('_version', '_token', '_key', '_secret', '_password')
    for name, info in shared.opts.data_labels.items():
        if name in shared.opts.secrets or name.endswith(secret_suffixes):
            continue
        value = shared.opts.get(name)
        if value == info.default:
            continue
        values.append(f'{name}={json.dumps(value, ensure_ascii=False, default=str)}')
    return '\n'.join(values) if values else 'No non-default options are set'


def _validate_option(name: str, raw_value: str) -> tuple[str, Any]:
    canonical = _SETTING_NAMES.get(name, name)
    if canonical in _SETTING_NAMES.values():
        return _validate_setting(name, raw_value)
    if canonical in shared.opts.secrets or canonical.endswith(('_version', '_token', '_key', '_secret', '_password')):
        raise CommandError(f'Setting is not available through chat: {canonical}')
    info = shared.opts.data_labels.get(canonical)
    if info is None:
        raise CommandError(f'Unsupported setting: {canonical}')
    current = shared.opts.get(canonical)
    example = info.default if info.default is not None else current
    annotation = type(example) if example is not None else str
    try:
        value = _parse_value(raw_value, annotation, example)
    except ValidationError as exc:
        raise CommandError(f'Invalid value for {canonical}: {exc.errors()[0]["msg"]}') from exc
    choices = _choice_values(canonical)
    if choices:
        value = _resolve_choice_fragment(canonical, value, choices)
    if not info.validate(canonical, value):
        raise CommandError(f'Invalid value for {canonical}')
    if choices:
        values = value if isinstance(value, list) else [value]
        if any(item not in choices for item in values):
            raise CommandError(f'Invalid value for {canonical}')
    return canonical, value


def _help_sections() -> list[dict[str, str]]:
    return [
        {
            'title': 'Chat',
            'content': '<a href="https://vladmandic.github.io/sdnext-docs/Chat/" target="_blank">Online Documentation</a>\n',
            'format': 'html',
        },
        {
            'title': 'Process',
            'content': 'Choose a context and control active work\n'
            '- title [name]: Read or set the current session name\n'
            '- context <value>: Select the active context\n'
            '- clear: Clear displayed messages without changing session history\n'
            '- run [count]: Run image generation 1-10 times (default: 1)\n'
            '- generate/start: Start the current process once\n'
            '- schedule: Add the current image or video generation to the scheduler\n'
            '- cancel/abort/stop/interrupt: Stop the current process\n'
            '- pause: Pause the current process\n'
            '- resume: Resume a paused process\n'
            '- skip: Skip the current item\n'
            '- history [limit]: Show recent process history\n'
            '- snapshot: List saved image settings snapshots\n'
            '- snapshot save: Save the current image draft as a snapshot and list snapshots\n'
            '- snapshot <id>: Restore a saved image settings snapshot\n'
            '- last/details: Show active parameters from the last processing run\n',
        },
        {
            'title': 'Params',
            'content': 'Set, inspect, and reset context parameters\n'
            '- prompt <value>: Set the prompt\n'
            '- negative <value>: Set the negative prompt\n'
            '- set [key] [value]: Set a value or list session values\n'
            '- get [key]: Get a value or list session values\n'
            '- clear <key>: Reset one parameter or setting to its default\n'
            '- params/args: List valid parameter keys for the current context\n'
            '- reset: Restore parameter defaults\n'
            '- Control Extras use script:field keys, such as dlss:nr\n',
        },
        {
            'title': 'References',
            'content': 'Upload and manage reference images and command text\n'
            '- use/file <path>: Import a server-local text or image file\n'
            '- image list: List available session images\n'
            '- image save: Save all images from the latest result batch\n'
            '- image <number[,number...]>: Select one or more images as active inputs\n'
            '- image reuse: Select results image as next input image\n'
            '- image clear: Clear active image inputs\n'
            '- meta/exif/metadata: Inspect metadata for the last image\n',
        },
        {
            'title': 'Helpers',
            'content': 'List available options\n'
            '- list <key>: Show valid choices\n',
        },
        {
            'title': 'Global',
            'content': 'Read or change settings shared across sessions\n'
            '- model [model] [unet]: Show or change the model and optional UNet\n'
            '- opts [key] [value]: Read or update global options\n',
        },
        {
            'title': 'Server',
            'content': 'Query runtime, device, memory, and storage details\n'
            '- info/ver/version: Show runtime version details\n'
            '- gpu/smi: Show GPU metrics\n'
            '- mem/memory: Show RAM, CUDA, and model usage\n'
            '- status: Show server status and progress\n'
            '- storage: Show storage usage\n'
            '- db: Compact the Chat database and show database statistics\n',
        },
    ]

def _format_help_sections(sections: list[dict[str, str]]) -> str:
    return '\n'.join(f'{section["title"]}\n{section["content"]}' for section in sections)


def _listable_keys(context: str) -> list[str]:
    keys = ['model', 'unet', 'vae']
    if context in {'image', 'video'}:
        keys.extend(['sampler_name', 'styles'])
    if context == 'image':
        try:
            keys.extend(item['name'] for item in _control_extra_parameter_specs() if item['field']['choices'])
        except CommandError:
            pass
    return sorted(keys)


def _context_parameter_keys(context: str) -> list[str]:
    if context == 'image':
        return sorted(_IMAGE_PARAMETERS)
    if context == 'video':
        return sorted(_VIDEO_PARAMETERS)
    return ['prompt', 'negative_prompt']


def _format_type(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        arguments = tuple(argument for argument in arguments if argument is not type(None))
        return ' | '.join(_format_type(argument) for argument in arguments)
    if origin is not None and arguments:
        name = getattr(origin, '__name__', str(origin).replace('typing.', ''))
        return f'{name}[{", ".join(_format_type(argument) for argument in arguments)}]'
    return str(annotation).replace('typing.', '')


def _parameter_field(context: str, name: str):
    if context == 'image':
        return _IMAGE_MODEL_FIELDS[name]
    if context == 'video':
        return _VIDEO_MODEL_FIELDS[name]
    return None


def _parameter_type(context: str, name: str) -> str:
    field = _parameter_field(context, name)
    return _format_type(field.annotation) if field is not None else 'str'


def _parameter_default(context: str, name: str) -> tuple[Any, str]:
    field = _parameter_field(context, name)
    if field is None:
        return None, 'unspecified'
    if field.is_required():
        return None, 'required'
    if field.default_factory is not None:
        if field.default_factory_takes_validated_data:
            return None, 'context-dependent'
        return field.get_default(call_default_factory=True), 'default'
    return field.default, 'default'


def _format_parameter_value(value: Any, state: str) -> str:
    if state != 'default' and state != 'set':
        return state
    if value is None:
        formatted = 'None'
    elif isinstance(value, bool):
        formatted = 'true' if value else 'false'
    elif isinstance(value, str):
        formatted = value.replace('\n', '\\n') if value else '""'
    elif isinstance(value, (int, float)):
        formatted = str(value)
    else:
        formatted = json.dumps(value, ensure_ascii=False, default=str)
    return formatted if len(formatted) <= 160 else f'{formatted[:157]}...'


def _context_parameter_entries(context: str, keys: list[str], draft: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for name in keys:
        is_set = name in draft
        if is_set:
            value, state = draft[name], 'set'
        else:
            value, state = _parameter_default(context, name)
        entries.append({
            'name': name,
            'type': _parameter_type(context, name),
            'value': _format_parameter_value(value, state),
            'is_set': is_set,
            'is_none': value is None,
        })
    return entries


def get_parameter_entries(context: str, draft: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    draft = draft or {}
    entries = _context_parameter_entries(context, _context_parameter_keys(context), draft)
    if context == 'image':
        try:
            for item in _control_extra_parameter_specs():
                field = item['field']
                value = _control_extra_value(draft, item['script'], field)
                entries.append({
                    'name': item['name'],
                    'type': type(field['default']).__name__ if field['default'] is not None else 'Any',
                    'value': _format_parameter_value(value, 'set' if value != field['default'] else 'default'),
                    'is_set': value != field['default'],
                    'is_none': value is None,
                })
        except CommandError:
            pass
    return entries


def _list_choices(context: str, name: str | None) -> str:
    if name is None:
        return f'Listable keys for {context}:\n' + '\n'.join(
            f'- {key}' for key in _listable_keys(context)
        )
    kind, spec = _parameter_spec(context, name)
    if kind == 'extra':
        choices = spec[1].get('choices') or []
        if not choices:
            raise CommandError(f'No valid choices are currently available for {name}')
        return f'{name}:\n' + '\n'.join(f'- {choice}' for choice in choices)
    choices = _choice_values(name)
    if kind == 'setting' and not choices:
        raise CommandError(f'No valid choices are currently available for {name}')
    if choices:
        return f'{name}:\n' + '\n'.join(f'- {choice}' for choice in choices)
    request_model, field_name = spec
    field = request_model.model_fields[field_name]
    image_range = _IMAGE_RANGES.get(name) if context == 'image' else None
    parameter_range = _field_range(field) or image_range
    if parameter_range:
        return f'{name}: {field.annotation}; range {parameter_range[0]}..{parameter_range[1]}'
    return f'{name}: {field.annotation}'


def _list_sessions() -> str:
    sessions = []
    offset = 0
    while True:
        batch = db.list_sessions(limit=100, offset=offset)
        sessions.extend(batch)
        if len(batch) < 100:
            break
        offset += len(batch)
    if not sessions:
        return 'No chat sessions'
    return 'Sessions:\n' + '\n'.join(
        f'- {item["title"]} ({item["context"]}) [{item["id"]}] updated {item["updated_at"]}'
        for item in sessions
    )


def _list_catalog(name: str) -> str:
    if name == 'sessions':
        return _list_sessions()
    if name == 'wildcards':
        from modules import ui_extra_networks_wildcards
        items = ui_extra_networks_wildcards.list_wildcard_names()
        return 'Wildcards:\n' + '\n'.join(f'- {item}' for item in items) if items else 'No wildcards available'
    if name in {'process', 'scripts'}:
        from modules import scripts_manager
        if name == 'process':
            runner = scripts_manager.scripts_postproc
            items = runner.scripts_in_preferred_order() if runner is not None else []
            labels = [item.name for item in items]
            return 'Postprocessing tasks:\n' + '\n'.join(f'- {item}' for item in labels) if labels else 'No postprocessing tasks available'
        groups = (
            ('txt2img', scripts_manager.scripts_txt2img),
            ('img2img', scripts_manager.scripts_img2img),
            ('control', scripts_manager.scripts_control),
            ('video', scripts_manager.scripts_video),
        )
        sections = []
        for group, runner in groups:
            labels = [script.title() for script in runner.scripts] if runner is not None else []
            if labels:
                sections.append(f'{group}:\n' + '\n'.join(f'- {label}' for label in labels))
        return '\n'.join(sections) if sections else 'No scripts available'
    return ''


def _field_range(field) -> tuple[Any, Any] | None:
    minimum = maximum = None
    for constraint in field.metadata:
        minimum = getattr(constraint, 'ge', getattr(constraint, 'gt', minimum))
        maximum = getattr(constraint, 'le', getattr(constraint, 'lt', maximum))
    if minimum is not None and maximum is not None:
        return minimum, maximum
    return None


def _reset_context(session_id: str, session: dict[str, Any]) -> str:
    names = set(_SETTING_NAMES.values())
    failures = []
    for name in names:
        if name == 'sd_model_checkpoint':
            loaded_model = shared.sd_model if shared.sd_loaded else None
            checkpoint_info = getattr(loaded_model, 'sd_checkpoint_info', None)
            default = getattr(checkpoint_info, 'title', None) or shared.opts.get(name)
        else:
            default = shared.opts.get_default(name)
        if default is not None and not _apply_setting(name, default):
            failure = f'{name}: setting callback rejected the default'
            failures.append(failure)
            log.error(f'Chat: setting reset failed: {failure}')
    drafts = dict(session.get('drafts') or {})
    drafts.pop(session['context'], None)
    db.update_session(session_id, drafts=drafts)
    return 'Current context parameters reset to defaults' + (('\n' + '\n'.join(failures)) if failures else '')


def _clear_parameter(session_id: str, session: dict[str, Any], name: str) -> str:
    context = session['context']
    kind, spec = _parameter_spec(context, name)
    if kind == 'setting':
        name = spec
        if name == 'sd_model_checkpoint':
            loaded_model = shared.sd_model if shared.sd_loaded else None
            checkpoint_info = getattr(loaded_model, 'sd_checkpoint_info', None)
            default = getattr(checkpoint_info, 'title', None) or shared.opts.get(name)
        else:
            default = shared.opts.get_default(name)
        if default is None:
            raise CommandError(f'No default value is available for {name}')
        if not _apply_setting(name, default):
            raise CommandError(f'Setting callback rejected the default for {name}')
    elif kind == 'extra':
        default = spec[1]['default']
    else:
        request_model, name = spec
        if request_model is None:
            default = ''
        else:
            default, state = _parameter_default(context, name)
            if state != 'default':
                raise CommandError(f'No default value is available for {name}')

    drafts = dict(session.get('drafts') or {})
    context_draft = dict(drafts.get(context) or {})
    if kind == 'extra':
        _set_control_extra_value(context_draft, spec[0], spec[1], default)
    else:
        context_draft.pop(name, None)
    if context_draft:
        drafts[context] = context_draft
    else:
        drafts.pop(context, None)
    db.update_session(session_id, drafts=drafts)
    return f'{name} cleared to default {_format_parameter_value(default, "default")}'


_UNSUPPORTED = object()
_OMIT = object()


def _simple_value(value: Any):
    if value is None:
        return _OMIT
    if isinstance(value, str):
        if not value.strip() or value.strip().casefold() == 'default':
            return _OMIT
        return value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        values = [_simple_value(item) for item in value]
        if any(item is _UNSUPPORTED for item in values):
            return _UNSUPPORTED
        values = [item for item in values if item is not _OMIT]
        return values if values else _OMIT
    if isinstance(value, dict):
        values = {str(key): _simple_value(item) for key, item in value.items()}
        if any(item is _UNSUPPORTED for item in values.values()):
            return _UNSUPPORTED
        values = {key: item for key, item in values.items() if item is not _OMIT}
        return values if values else _OMIT
    return _UNSUPPORTED


def _format_last_processing() -> dict[str, Any] | str:
    from modules import processing_class, processing_diffusers
    last = processing_diffusers.last_p
    if last is None:
        return 'No previous processing run is available'

    signature = inspect.signature(processing_class.StableDiffusionProcessing.__init__)
    parameters = {}
    for name, parameter in signature.parameters.items():
        if name in {'self', 'script_args'} or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter.default is inspect.Parameter.empty or not hasattr(last, name):
            continue
        value = _simple_value(getattr(last, name))
        if value is _OMIT or value is _UNSUPPORTED:
            continue
        default = _simple_value(parameter.default)
        if default is not _OMIT and default is not _UNSUPPORTED and value == default:
            continue
        parameters[name] = value
    if not parameters:
        return 'No non-default processing parameters'
    return parameters


def _control_script_specs():
    from modules import scripts_manager
    runner = scripts_manager.scripts_control
    if runner is None:
        raise CommandError('Control Extras are unavailable')
    specs = {}
    for script in runner.alwayson_scripts:
        info = script.api_info
        if info is None or not info.is_alwayson or not info.name:
            continue
        name = info.name.casefold()
        schema = [
            {
                'index': index,
                'label': argument.label or '',
                'default': _json_control_value(argument.value, f'{name} default'),
                'choices': _json_control_value(argument.choices, f'{name} choices'),
            }
            for index, argument in enumerate(info.args)
        ]
        specs[name] = {'script': script, 'schema': schema}
    return specs


def _control_extra_keys(script_name: str, schema: list[dict[str, Any]]) -> list[str]:
    script_key = re.sub(r'[^a-z0-9]+', '_', script_name.casefold()).strip('_') or script_name.casefold()
    labels = [
        re.sub(r'[^a-z0-9]+', '_', str(field['label']).casefold()).strip('_') or f'arg{field["index"]}'
        for field in schema
    ]
    counts = {label: labels.count(label) for label in labels}
    keys = []
    for field, label in zip(schema, labels):
        suffix = label if counts[label] == 1 else f'{label}_{field["index"]}'
        keys.append(f'{script_key}:{suffix}')
    return keys


def _control_extra_parameter_specs() -> list[dict[str, Any]]:
    parameters = []
    for script_name, spec in sorted(_control_script_specs().items()):
        keys = _control_extra_keys(script_name, spec['schema'])
        parameters.extend(
            {'name': key, 'script': script_name, 'field': field}
            for key, field in zip(keys, spec['schema'])
        )
    return parameters


def _control_extra_field(name: str):
    for item in _control_extra_parameter_specs():
        if item['name'] == name.casefold():
            return item['script'], item['field']
    return None


def _json_control_value(value: Any, label: str):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise CommandError(f'Control Extras value is not serializable: {label}') from exc


def _control_value_matches_default(value: Any, field: dict[str, Any]) -> bool:
    default = field['default']
    if value == default or (value is None and default == []) or (value == [] and default is None):
        return True
    choices = field.get('choices') or []
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(choices):
        return choices[value] == default
    return False


def capture_control_extras(script_values: list[Any]) -> dict[str, dict[str, Any]]:
    """Capture sparse always-on values from the Control script UI input vector."""
    extras = {}
    for name, spec in _control_script_specs().items():
        script = spec['script']
        schema = spec['schema']
        values = script_values[script.args_from:script.args_to]
        if len(values) != len(schema):
            raise CommandError(f'Control Extras values are unavailable for {name}')
        arguments = []
        for field, value in zip(schema, values):
            value = _json_control_value(value, f'{name}.{field["label"]}')
            if not _control_value_matches_default(value, field):
                arguments.append({'index': field['index'], 'label': field['label'], 'value': value})
        if arguments:
            extras[name] = {'argument_schema': schema, 'arguments': arguments}
    return extras


def validate_control_extras(extras: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(extras, dict):
        raise CommandError('Control Extras must be an object')
    if not extras:
        return {}
    specs = _control_script_specs()
    normalized = {}
    for raw_name, snapshot in extras.items():
        name = str(raw_name).casefold()
        spec = specs.get(name)
        if spec is None:
            raise CommandError(f'Control Extras script is unavailable: {raw_name}; re-import Control settings')
        schema = spec['schema']
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get('argument_schema'), list):
            raise CommandError(f'Invalid Control Extras schema for {raw_name}')
        saved_schema = _json_control_value(snapshot['argument_schema'], f'{name} schema')
        if saved_schema != schema:
            raise CommandError(f'Control Extras schema changed for {raw_name}; re-import Control settings')
        saved_arguments = snapshot.get('arguments')
        if not isinstance(saved_arguments, list):
            raise CommandError(f'Invalid Control Extras arguments for {raw_name}')
        arguments = []
        used_indexes = set()
        for item in saved_arguments:
            if not isinstance(item, dict):
                raise CommandError(f'Invalid Control Extras argument for {raw_name}')
            index = item.get('index')
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(schema):
                raise CommandError(f'Invalid Control Extras argument index for {raw_name}')
            if index in used_indexes or item.get('label') != schema[index]['label'] or 'value' not in item:
                raise CommandError(f'Control Extras argument order changed for {raw_name}; re-import Control settings')
            value = _json_control_value(item['value'], f'{name}.{schema[index]["label"]}')
            used_indexes.add(index)
            if value != schema[index]['default']:
                arguments.append({'index': index, 'label': schema[index]['label'], 'value': value})
        arguments.sort(key=lambda argument: argument['index'])
        if arguments:
            normalized[name] = {'argument_schema': schema, 'arguments': arguments}
    return normalized


def _control_extra_value(draft: dict[str, Any], script_name: str, field: dict[str, Any]):
    snapshot = validate_control_extras(draft.get('control_extras', {})).get(script_name)
    if snapshot is None:
        return field['default']
    for argument in snapshot['arguments']:
        if argument['index'] == field['index']:
            return argument['value']
    return field['default']


def _set_control_extra_value(draft: dict[str, Any], script_name: str, field: dict[str, Any], value: Any):
    extras = validate_control_extras(draft.get('control_extras', {}))
    schema = _control_script_specs()[script_name]['schema']
    snapshot = extras.get(script_name, {'argument_schema': schema, 'arguments': []})
    arguments = {item['index']: item for item in snapshot['arguments']}
    if value == field['default']:
        arguments.pop(field['index'], None)
    else:
        arguments[field['index']] = {
            'index': field['index'],
            'label': field['label'],
            'value': _json_control_value(value, f'{script_name}.{field["label"]}'),
        }
    if arguments:
        extras[script_name] = {
            'argument_schema': schema,
            'arguments': [arguments[index] for index in sorted(arguments)],
        }
    else:
        extras.pop(script_name, None)
    if extras:
        draft['control_extras'] = validate_control_extras(extras)
    else:
        draft.pop('control_extras', None)


def _validate_control_extra_value(name: str, field: dict[str, Any], raw_value: str):
    default = field['default']
    annotation = type(default) if default is not None else Any
    try:
        value = _parse_value(raw_value, annotation, default)
    except (ValidationError, ValueError) as exc:
        message = exc.errors()[0]['msg'] if isinstance(exc, ValidationError) else str(exc)
        raise CommandError(f'Invalid value for {name}: {message}') from exc
    choices = field.get('choices') or []
    if choices:
        value = _resolve_choice_fragment(name, value, choices)
        values = value if isinstance(value, list) else [value]
        if any(item not in choices for item in values):
            raise CommandError(f'Invalid value for {name}\nUse list {name} to see valid choices')
    return _json_control_value(value, name)


def control_extras_to_api(extras: Any) -> dict[str, dict[str, list[Any]]]:
    normalized = validate_control_extras(extras)
    payload = {}
    for name, snapshot in normalized.items():
        arguments = [entry['default'] for entry in snapshot['argument_schema']]
        for override in snapshot['arguments']:
            arguments[override['index']] = override['value']
        payload[name] = {'args': arguments}
    return payload


def _generation_endpoint(
    session: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    prompt: str | None = None,
    image: str | list[str] | None = None,
) -> dict[str, Any]:
    context = session['context']
    draft = (session.get('drafts') or {}).get(context, {})
    source = snapshot if snapshot is not None else draft
    if context == 'image':
        payload = {name: value for name, value in source.items() if name in _IMAGE_PARAMETERS}
        imported_extras = control_extras_to_api(source.get('control_extras', {}))
        if imported_extras:
            payload['alwayson_scripts'] = {**(payload.get('alwayson_scripts') or {}), **imported_extras}
        if prompt is not None:
            payload['prompt'] = prompt
        inputs = image if isinstance(image, list) else [image] if image else []
        payload.update({'inputs': inputs, 'send_images': True, 'save_images': shared.opts.samples_save})
        request_model = _IMAGE_MODEL
        error_prefix = 'Cannot run image generation'
    elif context == 'video':
        payload = {name: value for name, value in source.items() if name in _VIDEO_PARAMETERS}
        if prompt is not None:
            payload['prompt'] = prompt
        payload.update({'send_video': True, 'send_frames': False, 'send_thumbnail': True})
        request_model = ReqVideo
        error_prefix = 'Cannot run video generation'
    elif context == 'process':
        if not image:
            raise CommandError('Process requires an image generated in this chat')
        payload = {'image': image}
        request_model = models.ReqProcessImage
        error_prefix = 'Cannot process the image'
    elif context == 'caption':
        if not image:
            raise CommandError('Caption requires an image generated in this chat')
        from modules.api.caption import ReqVQA
        payload = {'image': image, 'question': 'describe the image'}
        request_model = ReqVQA
        error_prefix = 'Cannot caption the image'
    else:
        raise CommandError(f'Run is not supported for context: {context}')
    try:
        request = request_model.model_validate(payload)
    except ValidationError as exc:
        raise CommandError(f'{error_prefix}: {exc.errors()[0]["msg"]}') from exc
    descriptor = {
        'alias': 'generate-image',
        'method': 'POST',
        'path': {
            'image': '/sdapi/v1/control',
            'video': '/sdapi/v1/video',
            'process': '/sdapi/v1/process',
            'caption': '/sdapi/v1/vqa',
        }[context],
        'location': 'body',
        'arguments': request.model_dump(exclude_unset=True),
        'effects': [],
    }
    return {'alias': 'generate', 'endpoints': [descriptor]}


def _format_history(session_id: str, limit: int) -> str:
    operations = db.list_operations(session_id, limit=limit)
    if not operations:
        return 'No past operations'
    lines = []
    for operation in operations:
        snapshot = operation.get('snapshot') or {}
        prompt = ' '.join(str(snapshot.get('prompt', '')).split())
        summary = f'{operation["created_at"]}: {operation["command"]} {operation["status"]}'
        if prompt:
            summary += f' - {prompt[:100]}'
        lines.append(summary)
    return '\n'.join(lines)


def _format_settings_snapshots(session: dict[str, Any]) -> str:
    revisions = db.list_settings_revisions(session['id'])
    if not revisions:
        return 'No image settings snapshots have been saved'
    lines = []
    active_revision_id = session.get('active_settings_revision_id')
    for revision in revisions:
        snapshot = revision.get('snapshot') or {}
        parameters = snapshot.get('parameters') or {}
        extras = snapshot.get('extras') or {}
        details = []
        for name, value in sorted(parameters.items())[:3]:
            text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            details.append(f'{name}={text[:40] + "..." if len(text) > 40 else text}')
        if len(parameters) > 3:
            details.append(f'+{len(parameters) - 3} parameters')
        if extras:
            details.append(f'Extras: {", ".join(sorted(extras))}')
        summary = '; '.join(details) if details else 'no non-default settings'
        timestamp = snapshot.get('captured_at') or revision.get('created_at', '')
        active = ' [active]' if revision['id'] == active_revision_id else ''
        lines.append(f'#{revision["id"]}{active} {timestamp}: {summary}')
    return '\n'.join(lines)


def _execute_command(session_id: str, text: str, image: str | list[str] | None = None) -> dict[str, Any]:
    """Execute deterministic commands; endpoint aliases return validated descriptors only."""
    try:
        command = parse_command(text)
    except CommandError as exc:
        return _command_error(str(exc))

    if command.name == 'endpoint':
        endpoints_text = '\n'.join(
            f'{endpoint["method"]} {endpoint["path"]}'
            for endpoint in command.arguments['endpoints']
        )
        return _command_result(
            endpoints_text,
            kind='endpoint',
            data=command.arguments,
        )
    if command.name == 'comment':
        return _command_result(f'Comment: {command.arguments["text"]}')

    session = db.get_session(session_id)
    if session is None:
        return _command_error('Chat session not found')
    context = session['context']

    if command.name == 'help':
        sections = _help_sections()
        return _command_result(_format_help_sections(sections), sections=sections)
    if command.name == 'db':
        stats = db.compact_and_stats()
        size_kib = stats['size_bytes'] / 1024
        text = (
            'Chat database compacted\n'
            f'Path: {stats["path"]}\n'
            f'Size: {size_kib:.1f} KiB ({stats["size_bytes"]:,} bytes)\n'
            f'Pages: {stats["page_count"]} x {stats["page_size"]} bytes; free pages: {stats["free_pages"]}\n'
            f'Sessions: {stats["sessions"]}; messages: {stats["messages"]}; '
            f'operations: {stats["operations"]}; artifacts: {stats["artifacts"]}'
        )
        return _command_result(text, data=stats)
    if command.name == 'context':
        target = command.arguments['context']
        if target is None:
            return _command_result(f'Current context: {context}')
        failures = _apply_saved_settings((session.get('drafts') or {}).get(target, {}))
        db.update_session(session_id, context=target)
        message = f'Context set to {target}'
        if failures:
            message += '\n' + '\n'.join(failures)
        return _command_result(message)
    if command.name == 'title':
        title = command.arguments['value']
        if title is None:
            return _command_result(f'Current session name: {session["title"]}')
        try:
            updated = db.update_session(session_id, title=title)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        return _command_result(f'Session name set to {updated["title"]}')
    if command.name == 'import':
        return _command_result('', kind='import', data={'path': command.arguments['path']})
    if command.name == 'image':
        from modules.chat import artifacts
        images = [item for item in db.list_artifacts(session_id) if item['media_type'].startswith('image/')]
        action = command.arguments['action']
        if action == 'list':
            if not images:
                return _command_result('No session images')
            selected_ids = set(session.get('selected_image_ids') or ([session['selected_image_id']] if session.get('selected_image_id') else []))
            lines = []
            for index, artifact in enumerate(images, start=1):
                public = artifacts.public_artifact(artifact)
                status = ' (unavailable)' if not public['available'] else ''
                selected = ' [selected]' if artifact['id'] in selected_ids else ''
                lines.append(f'{index}. {public["display_name"]} ({artifact["created_at"]}){selected}{status}')
            return _command_result('\n'.join(lines), data={'format': 'images', 'artifacts': [artifacts.public_artifact(item) for item in images]})
        if action == 'clear':
            db.set_selected_images(session_id, [])
            return _command_result('Selected images cleared')
        if action == 'save':
            result_images = [item for item in images if item['role'] == 'result']
            if not result_images:
                return _command_error('No result images to save')
            latest_message_id = result_images[-1].get('message_id')
            latest_results = [item for item in result_images if item.get('message_id') == latest_message_id] if latest_message_id else [result_images[-1]]
            from PIL import Image
            from modules import images as image_utils
            saved_files = []
            failures = []
            for artifact in latest_results:
                if not artifacts.public_artifact(artifact)['available']:
                    failures.append(f'{artifact["display_name"]}: unavailable')
                    continue
                try:
                    with Image.open(artifact['storage_ref']) as result_image:
                        result_image.load()
                        filename, _, _ = image_utils.save_image(result_image, info=result_image.info.get('parameters'))
                    if filename:
                        saved_files.append(filename)
                    else:
                        failures.append(f'{artifact["display_name"]}: could not be saved')
                except Exception as exc:
                    failures.append(f'{artifact["display_name"]}: {type(exc).__name__}')
            if not saved_files:
                detail = '; '.join(failures) or 'No available result images'
                return _command_error(f'Could not save result images: {detail}')
            message = f'Saved {len(saved_files)} image(s):\n' + '\n'.join(saved_files)
            if failures:
                message += '\nFailed: ' + '; '.join(failures)
            return _command_result(message, kind='error' if failures else 'text', data={'files': saved_files})
        if action == 'reuse':
            available = [
                item for item in reversed(images)
                if item['role'] == 'result' and artifacts.public_artifact(item)['available']
            ]
            if not available:
                available = [item for item in reversed(images) if artifacts.public_artifact(item)['available']]
            if not available:
                return _command_error('No available image to reuse')
            selected = available[0]
        else:
            indices = command.arguments['indices']
            if any(index > len(images) for index in indices):
                return _command_error('Image number is not in the current image list')
            selected = [images[index - 1] for index in indices]
            if any(not artifacts.public_artifact(item)['available'] for item in selected):
                return _command_error('Selected image is no longer available')
        selected_images = selected if isinstance(selected, list) else [selected]
        db.set_selected_images(session_id, [item['id'] for item in selected_images])
        if len(selected_images) == 1:
            return _command_result(f'Selected image: {selected_images[0]["display_name"]}')
        names = ', '.join(item['display_name'] for item in selected_images)
        return _command_result(f'Selected images: {names}')
    if command.name == 'meta':
        from modules.chat import artifacts
        artifact = next((
            item for item in reversed(db.list_artifacts(session_id))
            if item['media_type'].startswith('image/') and artifacts.public_artifact(item)['available']
        ), None)
        if artifact is None:
            return _command_error('No available image to inspect')
        return _command_result(artifacts.image_metadata(artifact))
    if command.name == 'params':
        draft = (session.get('drafts') or {}).get(context, {})
        entries = get_parameter_entries(context, draft)
        return _command_result(
            f'Valid parameter keys for {context}:\n'
            + '\n'.join(
                f'{entry["name"]} {entry["type"]} {entry["value"]}'
                for entry in entries
            ),
            data={'format': 'params', 'parameters': entries},
        )
    if command.name == 'set':
        name = command.arguments['name']
        if name is None:
            return _command_result(_format_session(session))
        kind, spec = _parameter_spec(context, name)
        if kind == 'setting':
            name, value = _validate_setting(name, command.arguments['value'])
            if not _apply_setting(name, value):
                return _command_error(f'Setting callback rejected {name}; the session draft was not changed.')
        elif kind == 'extra':
            value = _validate_control_extra_value(name, spec[1], command.arguments['value'])
        else:
            value = _validate_parameter(context, name, command.arguments['value'])
        drafts = dict(session.get('drafts') or {})
        context_draft = dict(drafts.get(context) or {})
        if kind == 'extra':
            _set_control_extra_value(context_draft, spec[0], spec[1], value)
        else:
            context_draft[name] = value
        drafts[context] = context_draft
        db.update_session(session_id, drafts=drafts)
        return _command_result(f'{name} set to {value}')
    if command.name == 'get':
        name = command.arguments['name']
        if name is None:
            return _command_result(_format_session(session))
        kind, spec = _parameter_spec(context, name)
        if kind == 'setting':
            canonical = spec
            value = shared.opts.get(canonical)
        elif kind == 'extra':
            draft = (session.get('drafts') or {}).get(context, {})
            value = _control_extra_value(draft, spec[0], spec[1])
        else:
            draft = (session.get('drafts') or {}).get(context, {})
            request_model, field_name = spec
            default = request_model.model_fields[field_name].default if request_model is not None else ''
            value = draft.get(name, default)
        return _command_result(f'{name}={value}')
    if command.name == 'list':
        name = command.arguments['name']
        catalog = _list_catalog(name) if name in {'sessions', 'wildcards', 'process', 'scripts'} else ''
        if catalog:
            return _command_result(catalog)
        try:
            return _command_result(_list_choices(context, name))
        except CommandError as exc:
            return _command_error(str(exc))
    if command.name == 'model':
        values = command.arguments['values']
        if not values:
            return _command_result(
                f'model={shared.opts.get("sd_model_checkpoint")}\nunet={shared.opts.get("sd_unet")}')
        updates = []
        for name, raw_value in zip(('model', 'unet'), values):
            canonical, value = _validate_setting(name, raw_value)
            updates.append((canonical, value))
        messages = []
        for name, value in updates:
            if not _apply_setting(name, value):
                return _command_error(f'Setting callback rejected {name}; no further settings were changed.')
            messages.append(f'{name} set to {value}')
        return _command_result('\n'.join(messages))
    if command.name == 'opts':
        name = command.arguments['name']
        raw_value = command.arguments['value']
        if name is None:
            return _command_result(_format_options())
        name = _SETTING_NAMES.get(name, name)
        if raw_value is None:
            if name not in shared.opts.data_labels:
                raise CommandError(f'Unsupported setting: {name}')
            if name in shared.opts.secrets or name.endswith(('_version', '_token', '_key', '_secret', '_password')):
                raise CommandError(f'Setting is not available through chat: {name}')
            return _command_result(f'{name}={shared.opts.get(name)}')
        canonical, value = _validate_option(name, raw_value)
        if not _apply_setting(canonical, value):
            return _command_error(f'Setting callback rejected {canonical}; the option was not changed.')
        return _command_result(f'{canonical} set to {value}')
    if command.name == 'generate':
        data = _generation_endpoint(session, image=image)
        return _command_result('Generation started', kind='operation', data=data)
    if command.name == 'run':
        data = _generation_endpoint(session, image=image)
        data['command'] = 'run'
        data['run_count'] = command.arguments['count']
        return _command_result('Generation started', kind='operation', data=data)
    if command.name == 'schedule':
        if context not in {'image', 'video'}:
            raise CommandError(f'Scheduling is not supported for context: {context}')
        data = _generation_endpoint(session)
        return _command_result('Task scheduled', kind='schedule', data=data)
    if command.name == 'again':
        if context in {'process', 'caption'}:
            data = _generation_endpoint(session, image=image)
        else:
            snapshot = session.get('last_generation_snapshot')
            if not snapshot:
                return _command_result('No previous successful image generation')
            data = _generation_endpoint(session, snapshot, command.arguments.get('text'), image=image)
        data['command'] = 'again'
        return _command_result('Generation started', kind='operation', data=data)
    if command.name == 'cancel':
        shared.state.interrupt()
        return _command_result('Current process cancelled')
    if command.name == 'pause':
        if not shared.state.paused:
            shared.state.pause()
        return _command_result('Current process paused')
    if command.name == 'resume':
        if shared.state.paused:
            shared.state.pause()
        return _command_result('Current process resumed')
    if command.name == 'skip':
        shared.state.skip()
        return _command_result('Current process skipped')
    if command.name == 'history':
        return _command_result(_format_history(session_id, command.arguments.get('limit', 20)))
    if command.name == 'snapshot':
        if context != 'image':
            return _command_result(f'Not yet implemented for {context} context')
        revision_id = command.arguments['revision_id']
        action = command.arguments['action']
        if action == 'list':
            return _command_result(_format_settings_snapshots(session))
        if action == 'save':
            from modules.chat.api import save_chat_settings_revision
            saved = save_chat_settings_revision(session_id)
            revision_id = saved['revision']['id']
            return _command_result(
                f'Saved image settings snapshot #{revision_id}\n{_format_settings_snapshots(saved["session"])}',
                data={'revision_id': revision_id, 'active_revision_id': saved['session']['active_settings_revision_id']},
            )
        from modules.chat.api import restore_control_settings_revision
        restored = restore_control_settings_revision(session_id, revision_id)
        return _command_result(
            f'Restored image settings snapshot #{revision_id}',
            data={'revision_id': revision_id, 'active_revision_id': restored['session']['active_settings_revision_id']},
        )
    if command.name == 'last':
        last_processing = _format_last_processing()
        if isinstance(last_processing, str):
            return _command_result(last_processing)
        return _command_result(
            'Last processing parameters',
            sections=[{
                'title': 'Parameters',
                'content': json.dumps(last_processing, indent=2, ensure_ascii=False),
                'format': 'json',
            }],
        )
    if command.name == 'reset':
        return _command_result(_reset_context(session_id, session))
    if command.name == 'clear':
        return _command_result(_clear_parameter(session_id, session, command.arguments['name']))
    return _command_error(f'Unsupported command: {command.name}')


def execute_command(session_id: str, text: str, image: str | list[str] | None = None) -> dict[str, Any]:
    try:
        return _execute_command(session_id, text, image=image)
    except CommandError as exc:
        return _command_error(str(exc))
