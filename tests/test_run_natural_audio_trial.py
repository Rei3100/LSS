import pytest
from scripts import run_natural_audio_trial as runner


def test_blackout_refuses_before_sampling_or_child_launch(monkeypatch):
    calls = []
    def refuse():
        raise RuntimeError('gpu_blackout_active')
    monkeypatch.setattr(runner, '_enforce_gpu_blackout', refuse)
    monkeypatch.setattr(runner.local_llm, '_vram_mib', lambda: calls.append('sample'))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: calls.append('launch'))
    with pytest.raises(RuntimeError, match='gpu_blackout_active'):
        runner.main([])
    assert calls == []


def test_label_validation_rejects_paths_before_gpu(monkeypatch):
    monkeypatch.setattr(runner, '_enforce_gpu_blackout', lambda: pytest.fail('GPU check'))
    with pytest.raises(SystemExit):
        runner.main(['--label', 'r5-../escape'])


def test_existing_run_directory_refuses_before_gpu_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, "dev_evidence_root", lambda checkout_root: checkout_root)
    monkeypatch.setattr(runner, '_enforce_gpu_blackout', lambda: None)
    out = tmp_path / 'release-evidence/local-ai-audio-r1/natural-comparison-r5-safe'
    out.mkdir(parents=True)
    monkeypatch.setattr(runner.local_llm, '_vram_mib', lambda: pytest.fail('GPU sample'))
    with pytest.raises(RuntimeError, match='immutable'):
        runner.main(['--label', 'r5-safe', '--constrained-captions'])


@pytest.mark.parametrize('duty', ['0.25', '1.0'])
def test_unique_label_and_constraint_forwarded(tmp_path, monkeypatch, duty):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, "dev_evidence_root", lambda checkout_root: checkout_root)
    evidence = tmp_path / 'release-evidence/local-ai-audio-r1'
    evidence.mkdir(parents=True)
    probe = tmp_path / 'scripts/probe_audio_understanding.py'
    probe.parent.mkdir()
    probe.write_text('fake')
    monkeypatch.setattr(runner, '_enforce_gpu_blackout', lambda: None)
    monkeypatch.setattr(runner.local_llm, '_vram_mib', lambda: 0)
    monkeypatch.setattr(runner.local_llm, '_vram_release', lambda *_: (0, 0))
    class Sampler:
        vram_peak = 0
        ram_min = 10
        def __init__(self, **kwargs): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(runner.local_llm, 'Sampler', Sampler)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(runner.subprocess, 'run', run)
    assert runner.main(['--label', 'r5-trie', '--constrained-captions', '--duty', duty]) == 0
    assert '--constrained-captions' in commands[0]
    assert commands[0][commands[0].index('--duty') + 1] == duty
    assert 'local-ai-natural-r5-trie' in commands[0]
    assert (evidence / 'natural-comparison-r5-trie/resources.json').is_file()
