from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from lss.separation import intro_cleanup as cleanup


def signal(rate=1000,seconds=8):
    t = np.arange(rate*seconds)/rate
    return np.column_stack((np.sin(2*np.pi*37*t),np.sin(2*np.pi*43*t))).astype('float32')


def test_extremely_quiet_music_and_natural_crescendo_are_never_nominated():
    x = signal()
    x *= np.linspace(1e-6,1e-3,len(x),dtype='float32')[:,None]
    assert cleanup.nomination(x,x,1000) is None


def test_quietness_alone_never_zeroes_music_under_strong_voice():
    voice = signal()*.2
    target = signal()*.002
    region = cleanup.nomination(voice+target,target,1000)
    refined,receipt = cleanup.refine(target,target.copy(),region,1000)
    assert not receipt['adopted']
    assert np.array_equal(refined,target)


def test_second_estimate_has_to_reduce_residue_and_never_changes_later_pcm():
    x = signal()
    first = x*.0005
    first[1200:] = x[1200:]*.2
    region = cleanup.nomination(x*.2,first,1000)
    assert region['end'] == 1200
    rejected,receipt = cleanup.refine(first,first*.9,region,1000)
    assert not receipt['adopted'] and rejected is first
    second = first*.05
    refined,receipt = cleanup.refine(first,second,region,1000)
    assert receipt['adopted']
    assert np.array_equal(refined[1200:],first[1200:])
    assert np.array_equal(refined[0],first[0])
    assert np.array_equal(refined[1199],first[1199])


def test_full_stem_staging_retains_length_stereo_and_samples_after_prefix(tmp_path):
    rate = 1000
    source = signal(rate,14)*.2
    estimate = source*.001
    estimate[1200:] = source[1200:]
    original = tmp_path/'raw.wav'
    mix = tmp_path/'mix.wav'
    sf.write(mix,source,rate,subtype='FLOAT')
    sf.write(original,estimate,rate,subtype='FLOAT')
    before = original.read_bytes()
    def separate(crop,directory):
        directory.mkdir()
        data,sr = sf.read(crop,dtype='float32',always_2d=True)
        data *= .01
        output = directory/'Instrumental.wav'
        sf.write(output,data,sr,subtype='FLOAT')
        return output
    receipt = cleanup.process(mix,original,tmp_path/'intro',separate=separate)
    data,sr = sf.read(receipt['refined_path'],dtype='float32',always_2d=True)
    assert receipt['adopted'] and sr == rate and data.shape == estimate.shape
    assert np.array_equal(data[1200:],estimate[1200:])
    assert original.read_bytes() == before


def test_invalid_second_result_preserves_original_and_publishes_nothing(tmp_path):
    rate = 1000
    source = signal(rate,8)*.2
    estimate = source*.001
    mix,original = tmp_path/'mix.wav',tmp_path/'raw.wav'
    sf.write(mix,source,rate,subtype='FLOAT')
    sf.write(original,estimate,rate,subtype='FLOAT')
    before = original.read_bytes()
    def truncated(crop,directory):
        directory.mkdir()
        output = directory/'Instrumental.wav'
        sf.write(output,np.zeros((3,2),dtype='float32'),rate,subtype='FLOAT')
        return output
    with pytest.raises(ValueError,match='complete finite'):
        cleanup.process(mix,original,tmp_path/'intro',separate=truncated)
    assert original.read_bytes() == before
    assert not (tmp_path/'intro/refined.wav').exists()
