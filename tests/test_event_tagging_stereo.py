"""Stereo frontend invariants independent of learned class probabilities."""
import numpy as np
import pytest
import soundfile as sf
from lss.bgm import event_tagging as et


def test_legacy_input_and_stereo_timeline(tmp_path):
    from scipy.signal import resample_poly
    x=np.random.default_rng(9).normal(0,.1,(44100,2)).astype('float32')
    p=tmp_path/'input.wav';sf.write(p,x,44100,subtype='FLOAT')
    legacy,frames=et.model_input(p)
    stereo,n=et.model_input(p,channel_policy=et.STEREO_POWER)
    assert frames==n==len(x)
    np.testing.assert_array_equal(legacy,resample_poly(x.mean(axis=1),320,441).astype('float32'))
    assert stereo.shape==(2,32000)
    np.testing.assert_array_equal(stereo,resample_poly(x,320,441,axis=0).astype('float32').T)
    with pytest.raises(et.EventTaggingError):et.model_input(p,channel_policy='unknown')


def test_spectral_frontend_keeps_antiphase_and_channel_symmetry():
    import torch
    model=et.build_model().eval()
    # Exercise actual checkpoint-compatible STFT parameter structure without
    # allocating a second CNN or depending on random classifier decisions.
    spec=model.spectrogram_extractor
    with torch.no_grad():
        x=torch.randn(1,3200)
        original=spec(x)
        dual=spec(torch.cat([x,x],dim=0)).mean(dim=0,keepdim=True)
        anti=spec(torch.cat([x,-x],dim=0)).mean(dim=0,keepdim=True)
        # CPU convolution can choose a different accumulation kernel for batch2.
        torch.testing.assert_close(dual,original,rtol=3e-5,atol=4e-6)
        torch.testing.assert_close(anti,original,rtol=3e-5,atol=4e-6)
        assert spec((x-x)/2).count_nonzero()==0


def test_channel_policy_changes_recipe_identity(monkeypatch):
    before=et.analyzer_revision()['analyzer_id']
    monkeypatch.setitem(et.INFERENCE,'downmix',et.STEREO_POWER)
    assert before!=et.analyzer_revision()['analyzer_id']


def test_cancellation_guard_is_physical_and_silence_never_nominates():
    x=np.random.default_rng(11).normal(0,.1,(1,3200))
    dual=np.stack([x,x],axis=1)
    anti=np.stack([x,-x],axis=1)
    noise=np.random.default_rng(12).normal(0,.1,dual.shape)
    np.testing.assert_array_equal(et.cancelled_windows(np.concatenate([dual,anti,noise,dual*0])),[False,True,False,False])
    assert et.cancelled_windows(anti[:,:,::-1])[0]
    assert et.cancelled_windows(anti[:,::-1])[0]
    with pytest.raises(et.EventTaggingError):et.cancelled_windows(np.full((1,2,10),np.nan))
