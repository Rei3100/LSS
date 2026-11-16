from scripts import check_packaged_scene_bridge as bridge


def test_same_source_with_different_build_filename_is_same_code():
    source=b'def prepare():\n    return "verified"\n'
    assert bridge.matches(source, compile(source, 'build/lss/bgm/scene_native.py', 'exec'))


def test_old_embedded_logic_is_not_certified_by_correct_physical_source(tmp_path):
    source=tmp_path/'source'; package=tmp_path/'package'
    current=b'def prepare():\n    raise RuntimeError("unverified")\n'
    previous=b'def prepare():\n    return "ready"\n'
    for name in bridge.MODULES:
        for root in (source,package/'_internal'):
            target=root/'lss/bgm'/(name+'.py');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(current)
    class Archive:
        def extract(self, name): return compile(previous, name, 'exec')
    report=bridge.check(source,package,Archive())
    assert not report['passed']
    assert all(r['physical_source_matches'] for r in report['modules'])
    assert not any(r['embedded_bytecode_matches'] for r in report['modules'])


def test_non_code_and_nested_function_drift_are_rejected():
    assert not bridge.matches(b'x=1',None)
    source=b'def prepare():\n    def verify(): return True\n    return verify()\n'
    altered=source.replace(b'return True',b'return False')
    assert not bridge.matches(source,compile(altered,'native.py','exec'))
