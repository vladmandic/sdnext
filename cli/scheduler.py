#!/usr/bin/env python
"""Inspect and manage tasks already recorded by the SD.Next scheduler."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import scheduler


def _json_argument(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f'invalid JSON: {exc}') from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError('JSON value must be an object')
    return result


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def create_parser():
    parser = argparse.ArgumentParser(description='Manage existing SD.Next scheduler tasks')
    parser.add_argument('--db', help='scheduler SQLite database path')
    commands = parser.add_subparsers(dest='command', required=True)

    listing = commands.add_parser('list', help='list tasks')
    listing.add_argument('--status', choices=('scheduled', 'running', 'paused', 'completed', 'failed'))
    listing.add_argument('--limit', type=int)

    show = commands.add_parser('show', help='show one task')
    show.add_argument('task_id', type=int)

    search = commands.add_parser('search', help='search tasks')
    search.add_argument('--start')
    search.add_argument('--end')
    search.add_argument('--status', choices=('scheduled', 'running', 'paused', 'completed', 'failed'))
    search.add_argument('--parameters', type=_json_argument)
    search.add_argument('--limit', type=int)

    run = commands.add_parser('run', help='queue an existing task for playback')
    run.add_argument('task_id', type=int)

    update = commands.add_parser('update', help='override named generation parameters with JSON values')
    update.add_argument('task_id', type=int)
    update.add_argument('--parameters', required=True, type=_json_argument)

    for name, help_text in (
        ('delete', 'delete an existing task'),
        ('pause', 'pause a scheduled task'),
        ('resume', 'resume a paused task'),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument('task_id', type=int)

    reschedule = commands.add_parser('reschedule', help='change a task scheduled time')
    reschedule.add_argument('task_id', type=int)
    reschedule.add_argument('--at', required=True, help='UTC ISO-8601 timestamp')

    prune = commands.add_parser('prune', help='delete tasks created before a timestamp')
    prune.add_argument('--before', required=True, help='UTC ISO-8601 timestamp')

    commands.add_parser('pause-all', help='pause scheduler execution globally')
    commands.add_parser('resume-all', help='resume scheduler execution globally')
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.db:
        os.environ['SD_SCHEDULER_DB'] = args.db

    if args.command == 'list':
        _print(scheduler.list_tasks(status=args.status, limit=args.limit))
    elif args.command == 'show':
        task = scheduler.get_task(args.task_id)
        if task is None:
            print(f'Task not found: {args.task_id}', file=sys.stderr)
            return 1
        _print(task)
    elif args.command == 'search':
        _print(scheduler.search_tasks(start=args.start, end=args.end, status=args.status, parameters=args.parameters, limit=args.limit))
    elif args.command == 'run':
        if not scheduler.queue_existing_task(args.task_id):
            print(f'Task cannot be queued: {args.task_id}', file=sys.stderr)
            return 1
        print(f'Queued existing task {args.task_id} for application-worker playback')
    elif args.command == 'update':
        if not scheduler.update_task_parameters(args.task_id, args.parameters):
            print(f'Task parameters could not be updated: {args.task_id}', file=sys.stderr)
            return 1
        print(f'Updated task {args.task_id}')
    elif args.command == 'delete':
        if not scheduler.delete_task(args.task_id):
            print(f'Task could not be deleted: {args.task_id}', file=sys.stderr)
            return 1
        print(f'Deleted task {args.task_id}')
    elif args.command == 'pause':
        if not scheduler.pause_task(args.task_id):
            print(f'Task could not be paused: {args.task_id}', file=sys.stderr)
            return 1
        print(f'Paused task {args.task_id}')
    elif args.command == 'resume':
        if not scheduler.resume_task(args.task_id):
            print(f'Task could not be resumed: {args.task_id}', file=sys.stderr)
            return 1
        print(f'Resumed task {args.task_id}')
    elif args.command == 'reschedule':
        if not scheduler.reschedule_task(args.task_id, args.at):
            print(f'Task could not be rescheduled: {args.task_id}', file=sys.stderr)
            return 1
        print(f'Rescheduled task {args.task_id} for {args.at}')
    elif args.command == 'prune':
        print(f'Pruned {scheduler.prune_tasks(args.before)} task(s)')
    elif args.command == 'pause-all':
        scheduler.pause_all()
        print('Scheduler paused')
    elif args.command == 'resume-all':
        scheduler.resume_all()
        print('Scheduler resumed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
