"""Workspace-owned sleep policy: defer GPU jobs and automatic HDD maintenance.

This never changes GPU precision, fan curves, deployed Product or other processes.
An enabled or malformed existing policy blocks new GPU work and scheduled archive
maintenance until user steering; CPU checks and capacity admission continue.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


POLICY = dev_state_root() / 'dev_quiet_mode.json'


class GPUDeferred(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def require_gpu_allowed():
    try:
        raw = POLICY.read_text(encoding='utf-8')
    except FileNotFoundError:
        return
    except (OSError, UnicodeError) as exc:
        raise GPUDeferred('Cannot verify sleep policy; GPU development deferred') from exc
    try:
        value = json.loads(raw)
        if (not isinstance(value, dict) or type(value.get('schema')) is not int
                or value.get('schema') != 1 or type(value.get('enabled')) is not bool):
            raise ValueError('Invalid policy')
    except (ValueError, TypeError) as exc:
        raise GPUDeferred('Invalid sleep policy; GPU development deferred') from exc
    if value['enabled']:
        until = value.get('gpu_pause_until_utc')
        if until is not None:
            try:
                if not isinstance(until, str):
                    raise ValueError('Invalid GPU deadline')
                deadline = datetime.fromisoformat(until.replace('Z', '+00:00'))
                if deadline.tzinfo is None or deadline.utcoffset() is None:
                    raise ValueError('GPU deadline requires a timezone')
            except (ValueError, TypeError) as exc:
                raise GPUDeferred('Invalid explicit GPU pause deadline; development deferred') from exc
            if utc_now() >= deadline.astimezone(timezone.utc):
                return
        raise GPUDeferred('Sleep quiet mode: new GPU development is deferred; CPU checks remain available')


def quiet_requested():
    """Defer automatic archive work independently of explicit GPU permission.

    This never forbids explicit HDD work. Existing sleep policies preserve their
    behavior; a low-load awake policy can allow GPU while postponing background
    archive scans and storage-task nudges. Invalid flags fail closed for storage.
    """
    try:
        require_gpu_allowed()
    except GPUDeferred:
        return True
    try:
        value = json.loads(POLICY.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return False
    except (OSError, UnicodeError, ValueError):
        return True
    if not isinstance(value, dict):
        return True
    flag = value.get('defer_automatic_maintenance', False)
    return flag if type(flag) is bool else True
