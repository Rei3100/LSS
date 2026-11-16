"""Development Strata rejects unsafe acquisition and startup before ownership."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('strata_dev', SCRIPTS / 'strata_dev.py')
strata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strata)


def test_existing_unexpected_model_is_retained_without_download(tmp_path, monkeypatch):
    target = tmp_path / 'model.gguf'
    target.write_bytes(b'user bytes')
    run = Mock()
    monkeypatch.setattr(strata.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='retained'):
        strata.download('https://example.invalid/model', target, '0' * 64)
    assert target.read_bytes() == b'user bytes'
    run.assert_not_called()


def test_failed_checksum_never_promotes_partial(tmp_path, monkeypatch):
    target = tmp_path / 'model.gguf'
    partial = tmp_path / 'model.gguf.partial'
    partial.write_bytes(b'truncated')
    monkeypatch.setattr(strata.subprocess, 'run', Mock())
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        strata.download('https://example.invalid/model', target, '0' * 64)
    assert not target.exists()
    assert partial.read_bytes() == b'truncated'


def test_completed_download_is_verified_before_promotion(tmp_path, monkeypatch):
    target = tmp_path / 'model.gguf'
    (tmp_path / 'model.gguf.partial').write_bytes(b'complete')
    monkeypatch.setattr(strata.subprocess, 'run', Mock())
    strata.download('https://example.invalid/model', target, hashlib.sha256(b'complete').hexdigest())
    assert target.read_bytes() == b'complete'
    assert not target.with_name('model.gguf.partial').exists()


def test_insufficient_ram_never_starts_or_stops_server(tmp_path, monkeypatch):
    import gpu_run
    monkeypatch.setattr(gpu_run, 'memory_headroom', lambda: (80, 13))
    cli = Mock()
    monkeypatch.setattr(strata, 'cli', cli)
    with pytest.raises(RuntimeError, match='14 GB'):
        strata.worker(tmp_path, tmp_path / 'model', argparse.Namespace(command='smoke', profile='qwen36'))
    cli.assert_not_called()


def test_occupied_port_never_starts_or_stops_server(tmp_path, monkeypatch):
    import gpu_run
    monkeypatch.setattr(gpu_run, 'memory_headroom', lambda: (80, 28))
    cli = Mock()
    monkeypatch.setattr(strata, 'cli', cli)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        monkeypatch.setattr(strata, 'PORT', occupied.getsockname()[1])
        with pytest.raises(OSError):
            strata.worker(tmp_path, tmp_path / 'model', argparse.Namespace(command='smoke', profile='qwen36'))
    cli.assert_not_called()


def test_failed_launch_stops_owned_partial_start(tmp_path, monkeypatch):
    import gpu_run
    monkeypatch.setattr(gpu_run, 'memory_headroom', lambda: (80, 28))
    with socket.socket() as unused:
        unused.bind(('127.0.0.1', 0))
        monkeypatch.setattr(strata, 'PORT', unused.getsockname()[1])
    cli = Mock(side_effect=[RuntimeError('launch failed'), None])
    monkeypatch.setattr(strata, 'cli', cli)
    with pytest.raises(RuntimeError, match='launch failed'):
        strata.worker(tmp_path, tmp_path / 'model', argparse.Namespace(command='smoke', profile='qwen36'))
    assert cli.call_args_list[-1].args == (tmp_path, 'launch', '--stop')


def test_environment_does_not_inherit_python_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'unrelated-site-packages')
    monkeypatch.setenv('STRATA_PROFILE_DIR', 'unrelated-profiles')
    env = strata.environment(tmp_path)
    assert env['PYTHONPATH'] == ''
    assert env['STRATA_PROFILE_DIR'] == str(tmp_path / 'profiles')
    assert env['STRATA_RUNTIME_DIR'] == str(tmp_path / 'runtime')
    assert env['STRATA_INSTALL_DIR'] == str(tmp_path / 'install')
    assert env['LOCALAPPDATA'] == str(tmp_path / 'user-state')
    assert env['STRATA_ROUTE_TAIL_SKIP'] == '0'


def test_quiet_variant_preserves_context_and_isolates_state(tmp_path, monkeypatch):
    monkeypatch.setattr(strata, 'ACTIVE_VARIANT', 'quiet-exact')
    env = strata.environment(tmp_path)
    assert env['STRATA_PROFILE_DIR'] == str(tmp_path / 'variants/quiet-exact/profiles')
    assert env['STRATA_COMPONENTS_DIR'] == str(tmp_path / 'components')
    args = strata.launch_args(tmp_path / 'model.gguf')
    assert args[args.index('--context') + 1] == '8k'
    assert args[args.index('--kv') + 1] == 'fp16'
    assert 'resident-budget-gib=6' in args
    assert 'prefill=512' in args
    assert 'vram-reserve-mib=2048' in args


def test_profile_reuse_rejects_a_different_model(tmp_path):
    source = tmp_path / 'user-state' / 'Strata' / 'launch' / 'iq1_m-8k.json'
    source.parent.mkdir(parents=True)
    model = tmp_path / 'wanted.gguf'
    source.write_text(json.dumps({'args': ['--native', str(tmp_path / 'other.gguf'), '--max-context', '8192'],
                                  'mtp': False}))
    profile = tmp_path / 'profiles' / 'launch-iq1_m-8k.json'
    profile.parent.mkdir()
    profile.write_text(json.dumps({'source_config': str(source),
                                   'overrides': {'host': '127.0.0.1', 'parallel': 1, 'port': strata.PORT}}))
    assert strata.prepared_profile(tmp_path, model, 'coder') is None


def test_async_start_avoids_synchronous_launch_for_valid_profile(tmp_path, monkeypatch):
    cli = Mock()
    monkeypatch.setattr(strata, 'prepared_profile', lambda *args: 'launch-iq1_m-8k')
    monkeypatch.setattr(strata, 'cli', cli)
    monkeypatch.setattr(strata, 'request', lambda route: {'status': 'ok'})
    strata.start_ready(tmp_path, tmp_path / 'wanted.gguf', 'coder')
    cli.assert_called_once_with(tmp_path, 'runtime', 'start', '--profile', 'launch-iq1_m-8k', '--wait', '0')


def test_first_launch_rpc_timeout_recovers_only_with_valid_profile(tmp_path, monkeypatch):
    profiles = iter([None, 'launch-iq1_m-8k'])
    monkeypatch.setattr(strata, 'prepared_profile', lambda *args: next(profiles))
    cli = Mock(side_effect=[argparse.Namespace(returncode=6), argparse.Namespace(returncode=0)])
    monkeypatch.setattr(strata, 'cli', cli)
    monkeypatch.setattr(strata, 'request', lambda route: {'status': 'ok'})
    strata.start_ready(tmp_path, tmp_path / 'wanted.gguf', 'coder')
    assert cli.call_args_list[0].kwargs == {'check': False}
    assert cli.call_args_list[1].args == (tmp_path, 'runtime', 'start', '--profile', 'launch-iq1_m-8k', '--wait', '0')


def test_bounded_variant_limits_memory_without_reducing_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(strata, 'ACTIVE_VARIANT', 'quiet-bounded')
    args = strata.launch_args(tmp_path / 'model.gguf')
    assert args[args.index('--context') + 1] == '8k'
    assert args[args.index('--kv') + 1] == 'fp16'
    assert args[args.index('--mtp') + 1] == 'off'
    assert 'resident-budget-gib=4' in args
    assert 'conversation-cache-mib=0' in args
    assert 'prompt-cache=2' in args
    assert strata.environment(tmp_path)['STRATA_ROUTE_TAIL_SKIP'] == '0'
    assert 'quiet-bounded' in strata.environment(tmp_path)['STRATA_RUNTIME_DIR']


def test_full_speed_policy_requires_reason_and_safe_output_tag(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(strata, 'ROOT', tmp_path)
    assert strata.study_duty() == .1
    state = tmp_path / '.state'
    state.mkdir()
    path = state / 'strata-development.json'
    path.write_text(json.dumps({'process_duty': 1, 'run_tag': 'r2-full'}))
    with pytest.raises(ValueError, match='authorization'):
        strata.development_policy()
    path.write_text(json.dumps({'process_duty': 1, 'run_tag': 'r2-full', 'reason': 'User lifts quiet mode'}))
    assert strata.study_duty() == 1
    assert strata.study_output('protocol').name.endswith('r2-full')
    path.write_text(json.dumps({'process_duty': 1, 'run_tag': '../escape', 'reason': 'User lifts quiet mode'}))
    with pytest.raises(ValueError, match='run tag'):
        strata.development_policy()


def test_official_profile_without_overrides_uses_verified_source_settings(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(strata, 'ACTIVE_VARIANT', 'standard')
    monkeypatch.setattr(strata, 'development_policy', lambda: {'process_duty': .1, 'run_tag': 'r1'})  # no machine overrides
    source = tmp_path / 'user-state/Strata/launch/iq1_m-8k.json'
    source.parent.mkdir(parents=True)
    model = tmp_path / 'model.gguf'
    source.write_text(json.dumps({'args': ['--native', str(model), '--max-context', '8192'],
                                 'mtp': False, 'host': '127.0.0.1', 'port': strata.PORT, 'parallel': 1}))
    profile = tmp_path / 'profiles/launch-iq1_m-8k.json'
    profile.parent.mkdir()
    profile.write_text(json.dumps({'source_config': str(source), 'overrides': {}}))
    assert strata.prepared_profile(tmp_path, model, 'coder') == 'launch-iq1_m-8k'
    profile.write_text(json.dumps({'source_config': str(source), 'overrides': {'port': 9999}}))
    assert strata.prepared_profile(tmp_path, model, 'coder') is None


def test_hardware_variant_cannot_replace_model_or_precision_flags(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(strata, 'ROOT', tmp_path)
    monkeypatch.setattr(strata, 'ACTIVE_VARIANT', 'hardware-max')
    (tmp_path / '.state').mkdir()
    path = tmp_path / '.state/strata-development.json'
    path.write_text(json.dumps({'hardware_args': {'--prefill': 'auto', '--resident-budget-gib': '8', '--vram-reserve-mib': '512'}}))
    assert strata.variant_args()['--prefill'] == 'auto'
    assert strata.variant_args()['--conversation-cache-mib'] == '0'
    path.write_text(json.dumps({'hardware_args': {'--native': 'another-model.gguf'}}))
    with pytest.raises(ValueError, match='only measured'):
        strata.variant_args()


def test_image_encoder_threads_use_explicit_pc_capacity_and_reject_invalid_count(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(strata, 'ROOT', tmp_path)
    monkeypatch.setattr(strata.os, 'cpu_count', lambda: 16)
    (tmp_path / '.state').mkdir()
    path = tmp_path / '.state/strata-development.json'
    path.write_text(json.dumps({'process_duty': 1, 'reason': 'User requests PC maximum', 'vision_threads': 8}))
    assert strata.vision_thread_count() == 8
    path.write_text(json.dumps({'vision_threads': 0}))
    with pytest.raises(ValueError, match='Vision CPU threads'):
        strata.vision_thread_count()


def test_hf_request_accepts_only_pinned_huggingface_file_urls():
    rev = 'a' * 40
    assert strata.hf_request(f'https://huggingface.co/o/r/resolve/{rev}/IQ1_M/x.gguf') == ('o/r', rev, 'IQ1_M/x.gguf')
    assert strata.hf_request('https://huggingface.co/o/r/resolve/main/x.gguf') is None
    assert strata.hf_request('https://github.com/o/r/releases/download/v1/x.tar.gz') is None


def test_active_profile_env_selects_named_serving_profile(monkeypatch):
    monkeypatch.setenv('LSS_STRATA_PROFILE', 'qwen36')
    assert strata.active_profile() == 'qwen36'


def test_active_profile_rejects_unknown_env_profile(monkeypatch):
    monkeypatch.setenv('LSS_STRATA_PROFILE', 'nope')
    with pytest.raises(ValueError, match='Unknown Strata profile'):
        strata.active_profile()


def test_active_profile_without_env_or_policy_serves_the_coder(monkeypatch):
    monkeypatch.delenv('LSS_STRATA_PROFILE', raising=False)
    monkeypatch.setattr(strata, 'development_policy', lambda: {'process_duty': .1, 'run_tag': 'r1'})
    assert strata.active_profile() == 'coder'


def test_profile_engine_args_engine_profiles_sets_only_the_listed_model(monkeypatch):
    monkeypatch.setattr(strata, 'development_policy', lambda: {
        'process_duty': .1, 'run_tag': 'r1',
        'engine_profiles': {'qwen36': {'--prefill': '512'}}})
    assert strata.profile_engine_args('qwen36') == {'--prefill': '512'}
    assert strata.profile_engine_args('coder') == {}


def test_profile_engine_args_rejects_engine_profiles_outside_measured_settings(monkeypatch):
    monkeypatch.setattr(strata, 'development_policy', lambda: {
        'process_duty': .1, 'run_tag': 'r1',
        'engine_profiles': {'qwen36': {'--max-context': '9'}}})
    with pytest.raises(ValueError, match='only measured'):
        strata.profile_engine_args('qwen36')
