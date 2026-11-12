"""CPU-only, timestamped signal measurements for the opt-in audio-math harness.

The public entry points are probe_audio, decode_window and analyze. No model or
API is loaded. Event scores describe the detector, not semantic probabilities.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal, ndimage

MAX_WINDOW_SECONDS = 120.0
MAX_DECODE_FLOATS = 32_000_000


def _finite(value, name, low=None, high=None):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number')
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a number') from None
    if not math.isfinite(x) or (low is not None and x < low) or (high is not None and x > high):
        raise ValueError(f'{name} is outside its supported range')
    return x


def clean(value):
    """Convert scientific scalars/arrays to strict JSON values."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def probe_audio(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('audio source does not exist')
    proc = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                          capture_output=True, check=True, timeout=30)
    meta = json.loads(proc.stdout)
    stream = next((s for s in meta.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if stream is None:
        raise ValueError('source has no audio track')
    fmt = meta.get('format', {})
    def number(v, default):
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default
    origin = number(fmt.get('start_time'), 0.0)
    start = number(stream.get('start_time'), origin) - origin
    duration = number(fmt.get('duration'), None)
    audio_duration = number(stream.get('duration'), None)
    if duration is None and audio_duration is not None:
        duration = audio_duration + max(0, start)
    if duration is None or duration <= 0:
        raise ValueError('source requires a known finite duration')
    sr, channels = int(stream.get('sample_rate', 0)), int(stream.get('channels', 0))
    if sr <= 0 or channels <= 0:
        raise ValueError('source has invalid audio format')
    return {'has_audio': True, 'sample_rate': sr, 'channels': channels, 'duration': duration,
            'audio_start_s': start, 'audio_duration_s': audio_duration,
            'container_start_pts_s': origin, 'codec': stream.get('codec_name'),
            'time_reference': 'seconds from container start (same timeline as video tools)'}


def decode_window(path, t0, t1):
    """Decode original channels/rate; row zero corresponds to requested t0.

    FFmpeg accurate seeking is used on the container timeline. Leading/trailing
    time outside the audio stream is padded with zeros, retaining A/V alignment.
    """
    t0, t1 = _finite(t0, 't0', 0), _finite(t1, 't1', 0)
    if not 0 < t1 - t0 <= MAX_WINDOW_SECONDS:
        raise ValueError(f'window must be positive and at most {MAX_WINDOW_SECONDS:g} seconds')
    info = probe_audio(path)
    if t1 > info['duration'] + 0.001:
        raise ValueError(f't1 exceeds source duration ({info["duration"]:.3f}s)')
    sr, channels = info['sample_rate'], info['channels']
    n = int(round((t1 - t0) * sr))
    if n < 1 or n * channels > MAX_DECODE_FLOATS:
        raise ValueError('decoded window is too large; request a shorter interval')
    start = max(t0, info['audio_start_s'])
    end = t1
    if info['audio_duration_s'] is not None:
        end = min(end, info['audio_start_s'] + info['audio_duration_s'])
    out = np.zeros((n, channels), dtype=np.float32)
    if end <= start:
        return out, sr
    proc = subprocess.run([
        'ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-ss', f'{start:.9f}', '-i', str(path),
        '-t', f'{end-start:.9f}', '-map', '0:a:0', '-vn', '-sn', '-dn',
        '-ac', str(channels), '-ar', str(sr), '-f', 'f32le', '-acodec', 'pcm_f32le', '-'
    ], capture_output=True, check=True, timeout=90)
    raw = np.frombuffer(proc.stdout, dtype='<f4')
    if not raw.size:
        raise ValueError('audio decoder returned no samples')
    raw = raw[:raw.size // channels * channels].reshape(-1, channels)
    offset = max(0, int(round((start - t0) * sr)))
    count = min(len(raw), max(0, n-offset))
    out[offset:offset+count] = raw[:count]
    if not np.isfinite(out).all():
        raise ValueError('decoded audio contains nonfinite samples')
    return out, sr


def _db_power(p):
    return 10 * np.log10(np.maximum(p, 1e-16))


def _db_amp(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-8))


def _mono(x, channel):
    if channel in (None, 'mix'):
        return x.mean(axis=1, dtype=np.float64), 'mean of original channels'
    if channel == 'left':
        channel = 0
    elif channel == 'right':
        channel = 1
    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < x.shape[1]:
        raise ValueError('channel must be mix, left, right, or a valid zero-based integer')
    return x[:, channel].astype(np.float64), f'channel {channel}'


def _band(params, sr):
    band = params.get('band_hz')
    if band is None:
        return None
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        raise ValueError('band_hz must contain [low_hz, high_hz]')
    lo, hi = (_finite(v, 'band_hz', 0) for v in band)
    if not 0 <= lo < hi < sr / 2:
        raise ValueError(f'band_hz must satisfy 0 <= low < high < {sr/2:g} Hz')
    return lo, hi


def _filter(y, sr, band):
    if band is None:
        return y.copy()
    lo, hi = band
    sos = signal.butter(4, [lo, hi] if lo > 0 else hi,
                        btype='bandpass' if lo > 0 else 'lowpass', fs=sr, output='sos')
    if len(y) < 32:
        raise ValueError('window is too short for band filtering')
    return signal.sosfiltfilt(sos, y)


def _csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(row.get(k)) for k in fields})


def _features(y, sr, t0, params):
    # Preserve high frequencies if requested; ordinary overview is capped at 24 kHz.
    band = _band(params, sr)
    if band is not None:
        y = _filter(y, sr, band)
    target = min(sr, max(24000, int(math.ceil(band[1]*2.1))) if band else 24000)
    if target != sr:
        g = math.gcd(target, sr)
        y = signal.resample_poly(y, target//g, sr//g)
        sr = target
    window_ms = _finite(params.get('window_ms', 42.7), 'window_ms', 5, 1000)
    hop_ms = _finite(params.get('hop_ms', 10), 'hop_ms', 1, 1000)
    win = min(len(y), max(8, round(sr*window_ms/1000)))
    if win < 8:
        raise ValueError('window is too short for spectral analysis')
    hop = min(win, max(1, round(sr*hop_ms/1000)))
    cells = (win//2+1) * (1+(len(y)-win)//hop)
    if cells > 8_000_000:
        raise ValueError('time-frequency table is too large; shorten the interval or increase hop_ms')
    f, t, z = signal.stft(y, sr, window='hann', nperseg=win, noverlap=win-hop,
                          boundary=None, padded=False)
    power = np.abs(z)**2
    # RMS uses identical windows and centers, avoiding a false 1ms accuracy claim.
    starts = np.arange(len(t))*hop
    squared = np.r_[0.0, np.cumsum(y*y)]
    rms = np.sqrt(np.maximum((squared[starts+win]-squared[starts])/win, 0))
    flux = np.r_[0.0, np.maximum(np.diff(np.log1p(1000*np.abs(z)), axis=1), 0).mean(axis=0)]
    denom = power.sum(axis=0)
    centroid = (power*f[:, None]).sum(axis=0) / np.maximum(denom, 1e-20)
    centroid[denom < 1e-16] = np.nan
    novelty = np.r_[0.0, np.linalg.norm(np.diff(power / np.maximum(denom, 1e-20), axis=1), axis=0)]
    return {'frequency_hz': f, 'time_s': t+t0, 'power': power, 'rms_dbfs': _db_amp(rms),
            'centroid_hz': centroid, 'onset_strength': flux, 'spectral_change': novelty,
            'sample_rate': sr, 'window_ms': win/sr*1000, 'hop_ms': hop/sr*1000,
            'input_band_hz': band}


def _events(y, sr, t0, params):
    band = _band(params, sr)
    filtered = _filter(y, sr, band)
    envelope = np.abs(signal.hilbert(filtered))
    smooth_hz = min(_finite(params.get('envelope_smooth_hz', 120), 'envelope_smooth_hz', 1, 1000), sr/4)
    if len(y) >= 32:
        envelope = signal.sosfiltfilt(signal.butter(2, smooth_hz, fs=sr, output='sos'), envelope)
    envelope = np.maximum(envelope, 0)
    stride = max(1, sr//1000)
    env = envelope[::stride]
    env_sr = sr/stride
    level = _db_amp(env)
    floor_n = max(3, int(round(0.5*env_sr)) | 1)
    floor = ndimage.median_filter(level, size=floor_n, mode='nearest')
    gap_ms = _finite(params.get('min_gap_ms', 150), 'min_gap_ms', 1, 10000)
    threshold = _finite(params.get('threshold_above_local_floor_db', 12), 'threshold_above_local_floor_db', 0, 120)
    min_level = _finite(params.get('min_peak_dbfs', -60), 'min_peak_dbfs', -140, 20)
    pk, _ = signal.find_peaks(env, distance=max(1, round(gap_ms/1000*env_sr)))
    events = []
    candidates = []
    for p in pk:
        above = float(level[p]-floor[p])
        accepted = bool(above >= threshold and level[p] >= min_level)
        left = max(0, p-round(0.04*env_sr))
        below = np.where(env[left:p+1] <= env[p]*0.1)[0]
        onset = left+int(below[-1]) if len(below) else left
        a = max(0, int((p/env_sr-0.01)*sr))
        b = min(len(y), a+max(16, round(0.04*sr)))
        block = y[a:b]
        freq = None
        if len(block) >= 8 and np.max(np.abs(block)) > 1e-8:
            spec = np.abs(np.fft.rfft(block*np.hanning(len(block))))
            bins = np.fft.rfftfreq(len(block), 1/sr)
            select = (bins >= (band[0] if band else 20)) & (bins <= (band[1] if band else sr/2))
            if select.any():
                freq = float(bins[select][np.argmax(spec[select])])
        row = {'candidate_id': len(candidates), 'onset_s': t0+onset/env_sr,
               'peak_s': t0+p/env_sr, 'peak_dbfs': float(level[p]),
               'above_local_floor_db': above, 'dominant_frequency_hz': freq,
               'accepted': accepted, 'event_index': len(events) if accepted else None,
               'near_window_edge': bool(onset/env_sr < .05 or (len(y)/sr-p/env_sr) < .05),
               'reason': 'passed amplitude/local-floor thresholds' if accepted else 'below amplitude or local-floor threshold'}
        candidates.append(row)
        if accepted:
            events.append(row)
    return candidates, events, {'time_s': t0+np.arange(len(env))/env_sr,
                                'envelope_dbfs': level, 'local_floor_dbfs': floor,
                                'step_ms': stride/sr*1000,
                                'band_hz': band, 'min_gap_ms': gap_ms,
                                'threshold_above_local_floor_db': threshold, 'min_peak_dbfs': min_level}


def _rhythm(features, events):
    curve = features['onset_strength']
    rate = 1000/features['hop_ms']
    output = {'period_candidates': [], 'detected_event_count': len(events),
              'note': 'Periodicity candidates are not additional detected sound events.'}
    if len(curve) > rate*2 and np.max(curve) > 1e-8:
        centered = curve-curve.mean()
        acf = signal.correlate(centered, centered, mode='full', method='fft')[len(curve)-1:]
        if acf[0] > 1e-16:
            acf /= acf[0]
            lo, hi = max(1, int(rate*.25)), min(len(acf)-1, int(rate*1.5))
            peaks, _ = signal.find_peaks(acf[lo:hi+1])
            peaks = peaks+lo
            peaks = sorted(peaks, key=lambda p: acf[p], reverse=True)[:5]
            output['period_candidates'] = [{'period_s': p/rate, 'bpm': 60*rate/p,
                                             'autocorrelation': float(acf[p])} for p in peaks if acf[p] > 0]
    if len(events) >= 3:
        times = np.array([r['onset_s'] for r in events])
        intervals = np.diff(times)
        output['observed_intervals'] = {'median_s': np.median(intervals), 'min_s': intervals.min(),
                                        'max_s': intervals.max(), 'std_s': intervals.std()}
    return output


def _spectrum(features, params):
    f, p = features['frequency_hz'], features['power'].mean(axis=1)
    band = params.get('band_hz', [20, float(f[-1])])
    sel = (f >= band[0]) & (f <= band[1])
    idx = np.where(sel)[0]
    if not len(idx) or p.max() < 1e-16:
        return {'peaks': [], 'note': 'No measurable spectral peaks in this window.'}
    peaks, _ = signal.find_peaks(p[idx])
    peaks = sorted(idx[peaks], key=lambda i: p[i], reverse=True)[:12]
    reference = max(float(p.max()), 1e-16)
    result = {'peaks': [{'frequency_hz': float(f[i]), 'relative_power_db': float(10*np.log10(p[i]/reference))} for i in peaks],
            'frequency_bin_spacing_hz': float(f[1]-f[0]),
            'level_reference': '0 dB is the strongest bin of the window-averaged spectrum',
            'note': 'Multiple spectral peaks can be harmonics of one source; peaks do not count instruments or people.'}
    # A descriptive fit, not an estimator of the number of acoustic sources.
    fit_candidates = []
    for i in peaks:
        f0 = float(f[i])
        if not 40 <= f0 <= 1000:
            continue
        harmonic = np.maximum(1, np.rint(f/f0))
        match = np.abs(f-harmonic*f0) <= max(float(f[1]-f[0]), f0*.03)
        fit = float(p[sel & match].sum()/max(p[sel].sum(), 1e-16))
        fit_candidates.append({'fundamental_hz': f0, 'explained_power_fraction': fit})
    result['harmonic_fit_candidates'] = sorted(fit_candidates, key=lambda r:r['explained_power_fraction'], reverse=True)[:5]
    result['harmonic_fit_note'] = 'Candidate fundamentals must be present spectral peaks. Missing fundamentals and mixtures can make this fit ambiguous.'
    return result


def _plot(out, features, events, envelope=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 9})
    f, t, power = features['frequency_hz'], features['time_s'], features['power']
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, features['rms_dbfs'], lw=1, label='RMS dBFS')
    axes[0].set_ylabel('RMS dBFS'); axes[0].grid(alpha=.2)
    axes[1].plot(t, features['onset_strength'], lw=.8, label='Spectral flux')
    for event in events:
        axes[1].axvline(event['onset_s'], c='tomato', alpha=.45, lw=.6)
    axes[1].set_ylabel('Onset strength')
    reference = max(float(power.max()), 1e-16)
    # Bound rendered pixels, retaining full precision in CSV rather than a huge image.
    stride = max(1, len(t)//1600)
    db = 10*np.log10(np.maximum(power[:, ::stride]/reference, 1e-8))
    im = axes[2].pcolormesh(t[::stride], f, db, shading='auto', vmin=-80, vmax=0, cmap='magma')
    axes[2].set_ylabel('Frequency Hz'); axes[2].set_xlabel('Original media time (seconds)')
    fig.colorbar(im, ax=axes[2], label='dB relative to strongest STFT bin in this analysis')
    fig.savefig(out/'overview.png', dpi=130); plt.close(fig)
    artifacts = {'overview_plot': 'overview.png'}
    if envelope is not None:
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        step = max(1, len(envelope['time_s'])//5000)
        ax.plot(envelope['time_s'][::step], envelope['envelope_dbfs'][::step], lw=.8, label='Envelope')
        ax.plot(envelope['time_s'][::step], envelope['local_floor_dbfs'][::step], lw=.7, label='Local median')
        for event in events:
            ax.axvline(event['onset_s'], c='tomato', alpha=.5, lw=.7)
        ax.set(xlabel='Original media time (seconds)', ylabel='Band envelope dBFS')
        ax.legend(); fig.savefig(out/'events.png', dpi=130); plt.close(fig)
        artifacts['event_plot'] = 'events.png'
    return artifacts


def analyze(path, t0, t1, mode='overview', params=None, output_dir=None):
    """Write full data files and return a compact, strict-JSON-compatible result.

    Supported params: band_hz, channel, window_ms, hop_ms, min_gap_ms,
    threshold_above_local_floor_db, min_peak_dbfs, envelope_smooth_hz,
    comparison_window. Method-specific values are recorded in every result.
    """
    if mode not in {'overview', 'events', 'spectrum', 'rhythm', 'compare'}:
        raise ValueError('mode must be overview, events, spectrum, rhythm or compare')
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError('params must be an object')
    params = dict(params)
    allowed = {'band_hz', 'channel', 'window_ms', 'hop_ms', 'min_gap_ms',
               'threshold_above_local_floor_db', 'min_peak_dbfs', 'envelope_smooth_hz',
               'comparison_window'}
    unknown = set(params)-allowed
    if unknown:
        raise ValueError('unsupported analysis parameters: '+', '.join(sorted(unknown)))
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    info = probe_audio(path)
    x, sr = decode_window(path, t0, t1)
    t0, t1 = float(t0), float(t1)
    y, channel = _mono(x, params.get('channel', 'mix'))
    features = _features(y, sr, t0, params)
    candidates, events, envelope = _events(y, sr, t0, params)
    rows = [dict(zip(('time_s', 'rms_dbfs', 'centroid_hz', 'onset_strength', 'spectral_change'), vals))
            for vals in zip(*(features[k] for k in ('time_s', 'rms_dbfs', 'centroid_hz', 'onset_strength', 'spectral_change')))]
    _csv(out/'features.csv', rows, ['time_s', 'rms_dbfs', 'centroid_hz', 'onset_strength', 'spectral_change'])
    event_fields = ['candidate_id', 'event_index', 'onset_s', 'peak_s', 'peak_dbfs', 'above_local_floor_db',
                    'dominant_frequency_hz', 'accepted', 'near_window_edge', 'reason']
    _csv(out/'events.csv', candidates, event_fields)
    (out/'events.json').write_text(json.dumps(clean({'candidates': candidates, 'events': events}), indent=2), encoding='utf-8')
    level = float(_db_amp(np.sqrt(np.mean(y*y))))
    result = {'mode': mode, 'window': [t0, t1], 'time_reference': info['time_reference'],
              'source': info, 'channel_analysis': channel, 'params': params,
              'analysis': {k: features[k] for k in ('sample_rate', 'window_ms', 'hop_ms', 'input_band_hz')},
              'rms_dbfs': level, 'peak_dbfs': float(_db_amp(np.max(np.abs(y)))),
              'candidate_count': len(candidates), 'detected_event_count': len(events),
              'events_preview': events[:20], 'events_preview_truncated': len(events)>20,
              'detector': {k: envelope[k] for k in ('step_ms', 'band_hz', 'min_gap_ms', 'threshold_above_local_floor_db', 'min_peak_dbfs')},
              'summary': [f'Window {t0:.3f}–{t1:.3f}s; RMS {level:.1f} dBFS.',
                          f'{len(events)} amplitude/transient candidates passed the configured thresholds.'],
              'interpretation': ['Events are detections under this method, not named sound sources.',
                                 'Analysis hop size is not event-localization accuracy; window edges can affect filtering.'],
              'artifacts': {'features': 'features.csv', 'events': 'events.json', 'event_table': 'events.csv'}}
    # Ratios of STFT-bin power, with the same denominator for all bands.
    # They describe frequency allocation and are not calibrated band RMS levels.
    frequency, power = features['frequency_hz'], features['power']
    total_power = float(power.sum())
    result['band_power_fractions'] = [
        {'band_hz': [lo, min(hi, float(frequency[-1]))],
         'fraction': float(power[(frequency>=lo)&(frequency<hi)].sum()/max(total_power,1e-16))}
        for lo,hi in [(20,60),(60,150),(150,400),(400,1600),(1600,5000),(5000,12000)]
        if lo < frequency[-1]
    ]
    if x.shape[1] == 2:
        variance = x.var(axis=0)
        result['stereo'] = {'correlation': float(np.corrcoef(x.T)[0, 1]) if np.all(variance>1e-16) else None,
                            'channel_rms_dbfs': _db_amp(np.sqrt(np.mean(x.astype(float)**2, axis=0)))}
    if mode in {'overview', 'rhythm'}:
        result['rhythm'] = _rhythm(features, events)
    if mode in {'overview', 'spectrum', 'compare'}:
        result['spectrum'] = _spectrum(features, params)
    if mode in {'overview', 'compare'}:
        novelty = features['spectral_change']
        spacing = max(1, round(500/features['hop_ms']))
        change_idx, _ = signal.find_peaks(novelty, distance=spacing)
        selected = sorted(change_idx, key=lambda i: novelty[i], reverse=True)[:10]
        result['change_candidates'] = [{'time_s': float(features['time_s'][i]), 'spectral_change': float(novelty[i])}
                                       for i in sorted(selected) if novelty[i] > 1e-8]
    if mode == 'compare':
        other = params.get('comparison_window')
        if not isinstance(other, (list, tuple)) or len(other) != 2:
            raise ValueError('compare requires comparison_window: [start, end]')
        ox, osr = decode_window(path, other[0], other[1])
        oy, _ = _mono(ox, params.get('channel', 'mix'))
        of = _features(oy, osr, float(other[0]), params)
        other_level = float(_db_amp(np.sqrt(np.mean(oy*oy))))
        result['comparison'] = {'window': list(other), 'rms_dbfs': other_level,
                                'rms_change_db_other_minus_this': other_level-level,
                                'spectrum': _spectrum(of, params),
                                'note': 'Differences are recorded-level measurements; gain/distance/source identity are not inferred.'}
    result['artifacts'].update(_plot(out, features, events, envelope if mode in {'events', 'rhythm'} else None))
    result['artifacts']['analysis'] = 'analysis.json'
    result = clean(result)
    (out/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result
