"""参照だけの制御変換が左右と時刻を明示した規則に従い、入力を変更しないことを確認。"""
import numpy as np
from scipy.signal import resample_poly

from scripts.probe_bgm_reference_master_variants import SCENARIOS, construct_reference


def test_crossfeed_reference_is_authored_without_touching_the_source_channels():
    source = np.zeros((2, 4410));source[0,1470] = 0.4;source[1,2940] = 0.8
    before = source.copy()
    result = construct_reference(source, SCENARIOS[1])
    expected = resample_poly(source * 0.85 + source[::-1] * 0.15,160,147,axis=1,
                             window=("kaiser",5.0),padtype="constant")
    assert np.array_equal(result,expected) and np.array_equal(source,before)
    assert result.shape == (2,4800)


def test_authored_speed_variant_has_a_distinct_duration_and_common_stereo_time_axis():
    source = np.zeros((2,44100));source[:,22050] = [0.4,0.8]
    result = construct_reference(source, SCENARIOS[2])
    # 各段のresample_polyは端のフレーム数をceilするため、二段変換では1フレーム増える。
    assert result.shape == (2,48049)
    assert np.argmax(result[0]) == np.argmax(result[1]) == 24024
