"""Existing development commands preserve attempts and report current process truth."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import batch
import hidden_run
import process_state


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.delenv('CODEX_SESSION_ID',raising=False)
    monkeypatch.delenv('CODEX_THREAD_ID',raising=False)
    path=tmp_path/'work.json'
    path.write_text(json.dumps({'done':'DONE','steps':[{'name':'one','cmd':[sys.executable,'-c','pass'],'duty':1}]}),encoding='utf-8')
    return path


def test_finished_attempt_cannot_be_overwritten_or_dispatched_twice(queue,monkeypatch):
    calls=[]
    monkeypatch.setattr(batch,'run_command',lambda *a,**k:calls.append(a) or 0)
    raw=queue.read_bytes()
    assert batch.run_queue(queue)==0
    receipt=queue.with_suffix('.receipt.json');before=receipt.read_bytes()
    assert json.loads(before)['queue_sha256']==hashlib.sha256(raw).hexdigest()
    assert batch.run_queue(queue)==1
    assert batch.start_queue(queue)==1
    assert receipt.read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('error',[ValueError('provider failed'),KeyboardInterrupt()])
def test_unexpected_step_failure_is_terminal_and_blocks_downstream(queue,monkeypatch,error):
    def fail(*a,**k):raise error
    monkeypatch.setattr(batch,'run_command',fail)
    assert batch.run_queue(queue)==1
    record=json.loads(queue.with_suffix('.receipt.json').read_text())
    assert record['ended'] and record['status'].startswith('failed')
    assert record['steps'][0]['error_type']==type(error).__name__
    assert batch.marker_failure(queue.parent/'DONE')


def test_interrupt_while_waiting_for_upstream_is_terminal(queue,monkeypatch):
    data=json.loads(queue.read_text());data['steps'][0]['after']=['MISSING']
    queue.write_text(json.dumps(data))
    monkeypatch.setattr(batch.time,'sleep',lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    assert batch.run_queue(queue)==1
    record=json.loads(queue.with_suffix('.receipt.json').read_text())
    assert record['error_type']=='KeyboardInterrupt' and record['steps'][0]['rc'] is None
    assert batch.marker_failure(queue.parent/'DONE')


@pytest.mark.parametrize('value',[0,-1,2,float('nan'),'0.3',True])
def test_bad_duty_is_rejected_before_any_command(queue,value):
    data=json.loads(queue.read_text());data['steps'][0]['duty']=value;queue.write_text(json.dumps(data))
    with pytest.raises(ValueError):batch.load_queue(queue)
    assert not queue.with_suffix('.receipt.json').exists()


def test_colliding_logs_and_old_done_marker_are_rejected(queue,monkeypatch):
    data=json.loads(queue.read_text());data['steps']=[{'name':n,'cmd':['python']} for n in ['x y','x_y']]
    queue.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='collide'):batch.load_queue(queue)
    data['steps']=data['steps'][:1];queue.write_text(json.dumps(data));(queue.parent/'DONE').write_text('old evidence')
    with pytest.raises(ValueError,match='Existing done'):batch.run_queue(queue)
    assert (queue.parent/'DONE').read_text()=='old evidence'


def test_current_process_identity_and_recycled_or_legacy_pid(monkeypatch):
    own=process_state.identity()
    assert own['created'] and process_state.observe(own)=='RUNNING'
    assert process_state.observe({'pid':os.getpid()})=='UNKNOWN'
    monkeypatch.setattr(process_state,'probe',lambda pid:{'state':'RUNNING','created':'different-birth'})
    assert process_state.observe(own)=='REPLACED'
    monkeypatch.setattr(process_state,'probe',lambda pid:{'state':'UNKNOWN','created':None})
    assert process_state.observe(own)=='UNKNOWN'


def test_status_does_not_call_a_dead_receipt_running(queue,monkeypatch,capsys):
    queue.with_suffix('.receipt.json').write_text(json.dumps({'queue':str(queue),'started':'then','ended':None,'steps':[], 'owner':{'pid':12,'created':'old'}}))
    monkeypatch.setattr(process_state,'probe',lambda pid:{'state':'EXITED','created':None})
    assert batch.status(queue,as_json=True)==3
    assert json.loads(capsys.readouterr().out)['process_state']=='EXITED'


def test_two_real_queue_processes_execute_the_step_only_once(queue):
    data=json.loads(queue.read_text());data['steps'][0]['cmd']=[sys.executable,'-c',"from pathlib import Path; import time; p=Path('COUNT');p.write_text(p.read_text()+'x' if p.exists() else 'x');time.sleep(.3)"]
    queue.write_text(json.dumps(data))
    env={k:v for k,v in os.environ.items() if k not in ('CODEX_SESSION_ID','CODEX_THREAD_ID','LSS_STORAGE_RESERVATION')}
    env.update(LSS_MIN_FREE_RAM_GB='0',LSS_MIN_COMMIT_HEADROOM_GB='0')
    command=[sys.executable,str(ROOT/'scripts/batch.py'),'run',str(queue)]
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    first=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=flags)
    second=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=flags)
    try:
        first.communicate(timeout=30);second.communicate(timeout=30)
        assert sorted([first.returncode,second.returncode])==[0,1]
        assert (queue.parent/'COUNT').read_text()=='x'
    finally:
        for child in [first,second]:
            if child.poll() is None:child.kill();child.wait()


def test_hidden_launcher_receipt_observes_real_child_without_arguments(tmp_path,monkeypatch,capsys):
    children=[];real=hidden_run.subprocess.Popen
    def spawn(*a,**k):
        child=real(*a,**k);children.append(child);return child
    monkeypatch.setattr(hidden_run.subprocess,'Popen',spawn)
    receipt=tmp_path/'launch.json';log=tmp_path/'job.log'
    try:
        assert hidden_run.main(['--log',str(log),'--receipt',str(receipt),'--',sys.executable,'-c','import time;time.sleep(1)'],storage_supervisor=True)==0
        capsys.readouterr()
        assert 'time.sleep' not in receipt.read_text()
        assert hidden_run.main(['--status',str(receipt)])==0
        assert json.loads(capsys.readouterr().out)['state']=='RUNNING'
        children[0].wait(timeout=5)
        assert hidden_run.main(['--status',str(receipt)])==3
        assert json.loads(capsys.readouterr().out)['state']=='EXITED'
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait()


@pytest.mark.skipif(os.name!='nt',reason='PowerShell build entry')
def test_candidate_preflight_has_no_worktree_or_config_side_effect(tmp_path):
    target=tmp_path/'uncreated-build'
    before=subprocess.check_output(['git','config','--local','--list'],cwd=ROOT)
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/build_candidate_offdrive.ps1'),'-Commit','HEAD','-Worktree',str(target),'-Plan'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert len(plan['commit'])==40 and plan['reserves_before_preparation'] and not plan['mutates_install']
    assert not target.exists() and not Path(str(target)+'.build-receipt.json').exists()
    assert subprocess.check_output(['git','config','--local','--list'],cwd=ROOT)==before


def test_controller_exception_stops_owned_real_step_before_terminal_receipt(queue,monkeypatch):
    import gpu_run
    monkeypatch.delenv('LSS_STORAGE_RESERVATION',raising=False)
    monkeypatch.setattr(gpu_run,'wait_for_memory',lambda *a,**k:None)
    data=json.loads(queue.read_text());data['steps'][0]['cmd']=[sys.executable,'-c',"from pathlib import Path;import time;Path('READY').write_text('x');time.sleep(.5);Path('LATE').write_text('should not run')"]
    queue.write_text(json.dumps(data))
    def interrupt(child,*a):
        deadline=time.monotonic()+10
        while not (queue.parent/'READY').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert (queue.parent/'READY').exists()
        raise KeyboardInterrupt()
    monkeypatch.setattr(gpu_run,'_run_capped',interrupt)
    assert batch.run_queue(queue)==1
    assert json.loads(queue.with_suffix('.receipt.json').read_text())['ended']
    time.sleep(.7)
    assert not (queue.parent/'LATE').exists()


def test_completion_publication_failure_cannot_leave_ok_receipt(queue,monkeypatch):
    monkeypatch.setattr(batch,'run_command',lambda *a,**k:0)
    def write(*a,**k):
        assert batch.dependency_state(queue)=='waiting'  # consumers cannot start before publication
        raise PermissionError('fixture marker refusal')
    monkeypatch.setattr(batch,'publish_marker',write)
    assert batch.run_queue(queue)==1
    record=json.loads(queue.with_suffix('.receipt.json').read_text())
    assert record['status'].startswith('failed completion publication')


def test_completion_marker_is_never_allowed_to_overwrite_another_attempt(tmp_path):
    marker=tmp_path/'DONE';marker.write_text('prior evidence')
    with pytest.raises(FileExistsError):batch.publish_marker(marker,'replacement')
    assert marker.read_text()=='prior evidence'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('suffix',['-one.log','.launch.json','.queue.lock'])
def test_done_cannot_overwrite_diagnostic_or_ownership_files(queue,suffix):
    data=json.loads(queue.read_text());data['done']=queue.stem+suffix;queue.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='collide'):batch.load_queue(queue)


def test_failed_hidden_launch_is_terminal_and_does_not_spawn_again(tmp_path,monkeypatch,capsys):
    def fail(*a,**k):raise OSError('fixture spawn refusal')
    monkeypatch.setattr(hidden_run.subprocess,'Popen',fail)
    receipt=tmp_path/'refused.json'
    with pytest.raises(OSError):hidden_run.main(['--log',str(tmp_path/'log'),'--receipt',str(receipt),'--',sys.executable,'-c','pass'],storage_supervisor=True)
    assert json.loads(receipt.read_text())['stage']=='failed'
    assert hidden_run.main(['--status',str(receipt)])==1
    assert json.loads(capsys.readouterr().out)['state']=='FAILED'


def test_hidden_receipt_publication_failure_stops_the_spawned_child(tmp_path,monkeypatch):
    children=[];real=hidden_run.subprocess.Popen
    def spawn(*a,**k):
        child=real(*a,**k);children.append(child);return child
    monkeypatch.setattr(hidden_run.subprocess,'Popen',spawn)
    monkeypatch.setattr(hidden_run.os,'replace',lambda *a,**k:(_ for _ in ()).throw(PermissionError('fixture publication refusal')))
    receipt=tmp_path/'launch.json'
    with pytest.raises(PermissionError):hidden_run.main(['--log',str(tmp_path/'log'),'--receipt',str(receipt),'--',sys.executable,'-c','import time;time.sleep(30)'],storage_supervisor=True)
    assert children[0].poll() is not None
    assert json.loads(receipt.read_text())['stage']=='failed'


def test_short_hidden_child_records_observed_exit_instead_of_unknown(tmp_path,monkeypatch):
    real=hidden_run.subprocess.Popen
    def completed(*a,**k):
        child=real(*a,**k);child.wait(timeout=5);return child
    monkeypatch.setattr(hidden_run.subprocess,'Popen',completed)
    receipt=tmp_path/'short.json'
    assert hidden_run.main(['--log',str(tmp_path/'log'),'--receipt',str(receipt),'--',sys.executable,'-c','pass'],storage_supervisor=True)==0
    data=json.loads(receipt.read_text())
    assert data['stage']=='exited' and data['returncode']==0


@pytest.mark.skipif(os.name!='nt',reason='Windows Job kill-on-close contract')
def test_killed_queue_controller_cannot_leave_provider_children(queue):
    data=json.loads(queue.read_text());data['steps'][0]['cmd']=[sys.executable,'-c',"from pathlib import Path;import time;Path('READY').write_text('x');time.sleep(.6);Path('LATE').write_text('should not run')"]
    queue.write_text(json.dumps(data))
    env={k:v for k,v in os.environ.items() if k not in ('CODEX_SESSION_ID','CODEX_THREAD_ID','LSS_STORAGE_RESERVATION')}
    env.update(LSS_MIN_FREE_RAM_GB='0',LSS_MIN_COMMIT_HEADROOM_GB='0')
    child=subprocess.Popen([sys.executable,str(ROOT/'scripts/batch.py'),'run',str(queue)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline=time.monotonic()+10
        while not (queue.parent/'READY').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert (queue.parent/'READY').exists()
        child.kill();child.communicate(timeout=5)
        time.sleep(.8)
        assert not (queue.parent/'LATE').exists()
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_upstream_crash_fails_dependent_without_waiting_for_a_missing_marker(queue,monkeypatch):
    upstream=queue.parent/'upstream.json';upstream.write_text('{}')
    upstream.with_suffix('.receipt.json').write_text(json.dumps({'owner':{'pid':4,'created':'old'},'ended':None,'queue_sha256':hashlib.sha256(upstream.read_bytes()).hexdigest()}))
    data=json.loads(queue.read_text());data['steps'][0]['after_queues']=[upstream.name];queue.write_text(json.dumps(data))
    monkeypatch.setattr(process_state,'probe',lambda pid:{'state':'EXITED','created':None})
    monkeypatch.setattr(batch,'run_command',lambda *a,**k:(_ for _ in ()).throw(AssertionError('dependent must not run')))
    assert batch.run_queue(queue)==1
    assert 'stopped upstream' in json.loads(queue.with_suffix('.receipt.json').read_text())['status']


def test_completed_upstream_requires_the_same_queue_bytes(queue):
    upstream=queue.parent/'upstream.json';upstream.write_text('{}')
    upstream.with_suffix('.receipt.json').write_text(json.dumps({'ended':'now','status':'ok','queue_sha256':hashlib.sha256(upstream.read_bytes()).hexdigest()}))
    assert batch.dependency_state(upstream)=='ok'
    upstream.write_text('{"new":"settings"}')
    assert 'changed' in batch.dependency_state(upstream)


@pytest.mark.skipif(os.name!='nt',reason='Windows candidate preparation')
def test_candidate_preparation_records_exact_failure_without_rebuilding_product(tmp_path):
    import shutil
    import storage_tiers
    source=tmp_path/'fixture';(source/'scripts').mkdir(parents=True)
    for name in ['build_candidate_offdrive.ps1','storage_budget.py','storage_tiers.py','storage_paths.py','dev_hot_paths.py','machine_config.py','dev_state.py']:
        shutil.copyfile(ROOT/'scripts'/name,source/'scripts'/name)
    policy=json.loads((ROOT/'scripts/storage_policy.json').read_text())
    policy.update(build_root=str(tmp_path/'output'),local_state_root=str(tmp_path/'budget-state'),temp_root=str(tmp_path/'child-temp'))
    (source/'scripts/storage_policy.json').write_text(json.dumps(policy))
    (source/'build.ps1').write_text("throw 'fixture build stopped before packaging'\n")
    storage_tiers.make_junction(source/'.venv',ROOT/'.venv')
    subprocess.run(['git','init','-q',str(source)],check=True)
    for key,value in [('user.name','fixture'),('user.email','fixture@example.invalid')]:subprocess.run(['git','config',key,value],cwd=source,check=True)
    (source/'.gitignore').write_text('bin/\nvendor/\n')
    for name in ['bin/ffmpeg.exe','vendor/upstream/source.py']:
        (source/name).parent.mkdir(parents=True);(source/name).write_text(name)
    subprocess.run(['git','add','scripts','build.ps1','.gitignore'],cwd=source,check=True)
    subprocess.run(['git','commit','-qm','Fixture'],cwd=source,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    config_before=subprocess.check_output(['git','config','--local','--list'],cwd=source)
    target=tmp_path/'candidate'
    env={k:v for k,v in os.environ.items() if k not in ('LSS_STORAGE_RESERVATION','LSS_STORAGE_POLICY_JSON','LSS_STORAGE_POLICY')}
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(source/'scripts/build_candidate_offdrive.ps1'),'-Commit',sha,'-Worktree',str(target)],env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode!=0
    record=json.loads(Path(str(target)+'.build-receipt.json').read_text(encoding='utf-8'))
    assert record['commit']==sha and record['phase']=='build' and record['status']=='failed',result.stdout+result.stderr
    assert (target/'.venv/Scripts/python.exe').samefile(ROOT/'.venv/Scripts/python.exe')
    assert (target/'build').resolve().is_relative_to(tmp_path/'output')
    # Fetched files the package needs are ignored by git; the candidate worktree receives them.
    assert [(target/name).read_text() for name in ['bin/ffmpeg.exe','vendor/upstream/source.py']]==['bin/ffmpeg.exe','vendor/upstream/source.py']
    assert subprocess.check_output(['git','config','--local','--list'],cwd=source)==config_before
