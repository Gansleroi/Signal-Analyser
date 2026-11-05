import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

path = Path(__file__).resolve().parent.parent
if str(path) not in sys.path:
    sys.path.append(str(path))

from filters import IIRFilter, FIRFilter

@pytest.fixture
def IIRFilter_param():

    return {
        'n': 4,
        'cut_off1': 420,
        'cut_off2': 460,
        'btype': 'bandpass',
        'fs': 44100,
        'ftype': 'butter',
        'rp': None,
        'rs': None
    }

@pytest.fixture
def FIRFilter_param():

    return {
        'n': 4,
        'cut_off1': 420,
        'cut_off2': 460,
        'btype': 'bandpass',
        'fs': 44100,
        'beta': 8.6
    }
    

def test_IIRFilter_create(IIRFilter_param):

    filter = IIRFilter(**IIRFilter_param)

    filter.design()
    
    assert filter.sos is not None
    assert filter.fs == 44100

def test_IIRFilter_plot(IIRFilter_param):

    filter = IIRFilter(**IIRFilter_param)

    try:
        filter.plot(show=False)

    except Exception as e:
        pytest.fail(f"Метод plot() вызвал ошибку: {e}")

    plt.close('all')

def test_FIRFilter_create(FIRFilter_param):

    filter = FIRFilter(**FIRFilter_param)

    filter.design()

    assert filter.b is not None
    assert filter.fs == 44100

def test_FIRFrilter_plot(FIRFilter_param):

    filter = FIRFilter(**FIRFilter_param)

    try:
        filter.plot(show=False)

    except Exception as e:
        pytest.fail(f"Метод plot() вызвал ошибку: {e}")

    plt.close('all')

