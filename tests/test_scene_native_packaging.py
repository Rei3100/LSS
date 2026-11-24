"""Frozen native scene naming does not require the optional source-only Python ML stack."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENE = ('scene_job', 'scene_runtime', 'scene_native', 'scene_models', 'scene_names')
OPTIONAL = ('transformers', 'bitsandbytes', 'accelerate', 'huggingface_hub', 'tokenizers', 'safetensors')


def test_spec_packages_native_bridge_without_forcing_source_only_stack():
    tree = ast.parse((ROOT / 'LSS.spec').read_text(encoding='utf-8'))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('datas', 'hiddenimports', 'excludes'):
                    values[target.id] = ast.literal_eval(node.value)
    for name in SCENE:
        assert (f'lss/bgm/{name}.py', 'lss/bgm') in values['datas']
        assert f'lss.bgm.{name}' in values['hiddenimports']
    assert not (set(OPTIONAL) & set(values['hiddenimports']))
    # Installed in the development venv for the source route, but never packaged (2405 files, about 170 MB).
    assert set(OPTIONAL) - {'huggingface_hub'} <= set(values['excludes'])
    build = (ROOT / 'build.ps1').read_text(encoding='utf-8-sig')
    assert 'Scene naming dependencies are missing' not in build
    for name in SCENE:
        assert f'_internal\\lss\\bgm\\{name}.py' in build


def test_native_import_and_revision_do_not_import_optional_python_stack(tmp_path):
    code = '''
import builtins
original = builtins.__import__
forbidden = {"transformers", "bitsandbytes", "accelerate", "huggingface_hub", "tokenizers", "safetensors"}
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise AssertionError("optional dependency imported: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from lss.bgm import scene_native, scene_runtime
from lss import paths
paths.is_frozen = lambda: True
assert not scene_runtime.transformers_available()
assert scene_native.revision()["backend"] == scene_native.BACKEND
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stderr
