import math

import numpy as np

import pytest

from signalsaurus import Spectrum


def test_Spectrum_basic_init():
    df = 0.1  # Hz
    f_sample = 100  # Hz
    spectrum = Spectrum(f_sample, df)
    N_samples = math.ceil(f_sample / df)
    ts = np.arange(N_samples) / f_sample

    assert math.isclose(spectrum.T_sample, 1 / f_sample)
    assert math.isclose(spectrum.T_max, 1 / df)
    assert math.isclose(spectrum.f_Ny, f_sample / 2.0)
    assert spectrum.N_samples == int(f_sample / df)
    assert np.all(np.isclose(spectrum.ts, ts))

    with pytest.raises(ValueError):
        spectrum.Xs

    print(spectrum)


def test_Spectrum_noninteger_N_samples():
    df = 0.15  # Hz
    f_sample = 100  # Hz
    spectrum = Spectrum(f_sample, df)
    N_samples = math.ceil(f_sample / df)
    df_actual = f_sample / N_samples

    assert math.isclose(spectrum.df, df_actual)
    assert math.isclose(spectrum.T_sample, 1 / f_sample)
    assert math.isclose(spectrum.T_max, 1 / df_actual)
    assert math.isclose(spectrum.f_Ny, f_sample / 2.0)
    assert spectrum.N_samples == N_samples


def test_Spectrum_update():
    df = 0.1  # Hz
    f_sample = 100  # Hz
    spectrum = Spectrum(f_sample, df)

    df2 = 0.15  # Hz
    spectrum.df = df2

    N_samples = math.ceil(f_sample / df2)
    df_actual = f_sample / N_samples
    ts = np.arange(N_samples) / f_sample

    assert math.isclose(spectrum.df, df_actual)
    assert math.isclose(spectrum.T_sample, 1 / f_sample)
    assert math.isclose(spectrum.T_max, 1 / df_actual)
    assert math.isclose(spectrum.f_Ny, f_sample / 2.0)
    assert spectrum.N_samples == N_samples
    assert np.all(np.isclose(spectrum.ts, ts))

    print(spectrum)

    f_sample2 = 125  # Hz
    N_samples2 = math.ceil(f_sample2 / df2)
    ts2 = np.arange(N_samples2) / f_sample2
    spectrum.f_sample = f_sample2

    N_samples2 = math.ceil(f_sample2 / df2)
    df_actual = f_sample2 / N_samples2

    assert math.isclose(spectrum.df, df_actual)
    assert math.isclose(spectrum.T_sample, 1 / f_sample2)
    assert math.isclose(spectrum.T_max, 1 / df_actual)
    assert math.isclose(spectrum.f_Ny, f_sample2 / 2.0)
    assert spectrum.N_samples == N_samples2
    assert np.all(np.isclose(spectrum.ts, ts2))

    print(spectrum)
