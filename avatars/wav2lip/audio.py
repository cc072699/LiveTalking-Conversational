import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from .hparams import hparams as hp

# ── GPU 加速 Mel 频谱缓存 ──
_gpu_mel_basis = None
_gpu_mel_device = None
_hanning_window = None

def _get_gpu_mel_basis(device):
    global _gpu_mel_basis, _gpu_mel_device
    if _gpu_mel_basis is None or _gpu_mel_device != device:
        mel_basis = librosa.filters.mel(sr=float(hp.sample_rate), n_fft=hp.n_fft, n_mels=hp.num_mels,
                                         fmin=hp.fmin, fmax=hp.fmax)
        _gpu_mel_basis = torch.from_numpy(mel_basis.astype(np.float32)).to(device)
        _gpu_mel_device = device
    return _gpu_mel_basis


import torch

def melspectrogram_gpu(wav):
    """
    GPU 加速 Mel 频谱 — 用 torch.stft 替代 librosa.stft，适合弱 CPU 强 GPU 场景。
    输入输出同 melspectrogram()，可无缝替换。
    """
    global _hanning_window
    device = torch.device('cuda')

    wav_t = torch.from_numpy(wav).to(device, non_blocking=True)

    # Pre-emphasis
    if hp.preemphasize:
        wav_t = torch.cat([wav_t[:1], wav_t[1:] - hp.preemphasis * wav_t[:-1]])

    # Hanning window（缓存避免重复分配）
    if _hanning_window is None or _hanning_window.device != device or _hanning_window.shape[0] != hp.n_fft:
        _hanning_window = torch.hann_window(hp.n_fft, device=device)

    D = torch.stft(
        wav_t,
        n_fft=hp.n_fft,
        hop_length=get_hop_size(),
        win_length=hp.win_size,
        window=_hanning_window,
        return_complex=True,
    )

    mag = torch.abs(D)

    # Mel filter bank（缓存）
    mel_basis = _get_gpu_mel_basis(device)
    S = torch.mm(mel_basis, mag)

    # amp_to_db
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    S = 20 * torch.log10(torch.clamp(S, min=float(min_level)))
    S = S - hp.ref_level_db

    # Normalize
    if hp.signal_normalization:
        if hp.allow_clipping_in_normalization:
            if hp.symmetric_mels:
                S = torch.clamp((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                                -hp.max_abs_value, hp.max_abs_value)
            else:
                S = torch.clamp(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
        else:
            if hp.symmetric_mels:
                S = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
            else:
                S = hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

    return S.cpu().numpy()

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=float(hp.sample_rate), n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
