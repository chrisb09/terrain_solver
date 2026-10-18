"""Ambient soundtrack for the edited Terrain Solver demo video.

Built from CC0 field recordings downloaded from freesound.org
(assets/audio/, decoded to 48 kHz stereo WAV by convert_audio_assets.sbatch):

  bed_storm_waves_867645  Powerful Storm Waves - Rough Sea and Surf (bassimat)
  crash_shore_546862      waves crashing into shore, parkdale beach
  splash_small_646568     WaterSplash (Ryanz-Official)
  splash_splatter_869973  Water Splatter 3 (TDaddy22)
  whoosh_001_812636       Sound Design Elements Whoosh SFX 001 (AudioPapkin)
  bell_ding_333695        Thin bell ding 2 (Khrinx)

Layers: storm-surf bed + a sub drone, a wave-crash boom on the collapse,
whooshes on the four cross-section moves, bell cues on the annotation
cards and a swell + crash + deep bell for the finale. Writes
OUT_DIR/audio.wav (48 kHz stereo 16-bit).
"""

import math
import os
import wave

import numpy as np

import edit_storyboard as SB

SR = 48000
FPS = SB.FPS

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "audio", "wav")


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pan_gains(pan):
    ang = (np.clip(pan, -1.0, 1.0) + 1.0) * (math.pi / 4.0)
    return math.cos(ang), math.sin(ang)


def read_wav(name):
    path = os.path.join(ASSET_DIR, name)
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SR, path
        assert wf.getsampwidth() == 2, path
        n = wf.getnframes()
        ch = wf.getnchannels()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16)
    x = data.astype(np.float64) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2)
    else:
        x = np.stack([x, x], axis=1)
    return x


def add(buf_l, buf_r, t0, sig_l, sig_r):
    i0 = int(round(t0 * SR))
    if i0 >= len(buf_l) or i0 + len(sig_l) <= 0:
        return
    j0 = max(0, -i0)
    i0 = max(0, i0)
    i1 = min(len(buf_l), i0 + len(sig_l) - j0)
    if i1 <= i0:
        return
    buf_l[i0:i1] += sig_l[j0:j0 + (i1 - i0)]
    buf_r[i0:i1] += sig_r[j0:j0 + (i1 - i0)]


def loudest_window(x, sec):
    m = int(round(sec * SR))
    mono = x.mean(axis=1)
    cs = np.concatenate([[0.0], np.cumsum(mono * mono)])
    n_win = len(mono) - m
    if n_win <= 0:
        return 0
    idx = np.arange(0, n_win, max(1, SR // 4))
    energy = cs[idx + m] - cs[idx]
    return int(idx[int(np.argmax(energy))])


def resample_pitch(sig, rate):
    """Re-pitch a stereo signal by resampling (rate>1 = higher+shorter)."""
    n = sig.shape[0]
    new_n = max(2, int(round(n / rate)))
    src = np.arange(n, dtype=np.float64)
    dst = np.arange(new_n, dtype=np.float64) * rate
    out = np.stack([np.interp(dst, src, sig[:, 0]),
                    np.interp(dst, src, sig[:, 1])], axis=1)
    return out


def main():
    T = SB.build_once()
    dur = len(T.frames) / float(FPS)
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)

    # -------- bed: rough sea (field recording) --------
    bed = read_wav("bed_storm_waves_867645.wav")
    off = int(round(15.0 * SR))
    seg = bed[off:off + n]
    if len(seg) < n:
        seg = np.concatenate([seg, np.zeros((n - len(seg), 2))])
    bed_env = np.ones(n)
    fi = int(round(3.0 * SR))
    bed_env[:fi] *= smooth(np.linspace(0, 1, fi))
    fo = int(round(4.5 * SR))
    bed_env[-fo:] *= 1.0 - smooth(np.linspace(0, 1, fo))
    lfo = 0.82 + 0.18 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.047 * t + 1.0))
    L += seg[:, 0] * bed_env * lfo * 0.62
    R += seg[:, 1] * bed_env * lfo * 0.62
    print("bed rms %.3f" % np.sqrt((seg[:, 0] ** 2).mean()))

    # -------- subtle sub drone for depth --------
    def drone_channel(seed_phase):
        out = np.zeros(n)
        for f, a in [(55.0, 1.0), (55.15, 0.55), (110.0, 0.3),
                     (110.25, 0.18)]:
            out += a * np.sin(2 * np.pi * f * t + seed_phase * a * 7.0)
        return out

    lfo_d = 0.5 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.051 * t + 1.0))
    env_d = lfo_d * smooth(t / 6.0) * \
        (1.0 - smooth((t - (dur - 3.5)) / 3.5)) * 0.026
    L += drone_channel(0.0) * env_d
    R += drone_channel(1.7) * env_d

    crash = read_wav("crash_shore_546862.wav")
    booms = {}
    used_starts = []
    for tag, sec, start_skip in (("boom", 3.2, 0), ("finale_wave", 3.0, 0)):
        best, best_i = -1.0, 0
        for cand in range(0, len(crash) - int(sec * SR), SR):
            if any(abs(cand - u) < 30 * SR for u in used_starts):
                continue
            win = crash[cand:cand + int(sec * SR), 0]
            v = float(np.sqrt((win ** 2).mean()))
            if v > best:
                best, best_i = v, cand
        used_starts.append(best_i)
        booms[tag] = crash[best_i:best_i + int(sec * SR)]
        print("%s window at %.1fs rms %.3f" % (tag, best_i / SR, best))

    # -------- collapse: wave-crash boom + splash --------
    t_collapse = SB.events()["collapse"] / float(FPS)
    boom = booms["boom"]
    bl = int(round(len(boom) * 0.65))
    b = boom[:bl]
    env_b = np.concatenate([smooth(np.linspace(0, 1, int(0.02 * SR))),
                            np.ones(bl - int(0.02 * SR))])
    env_b *= (1.0 - smooth((np.arange(bl) - 0.3 * SR) / (2.6 * SR)))
    gain_b = 1.35 / max(1e-9, np.abs(b).max())
    add(L, R, t_collapse - 0.05, b[:, 0] * env_b * gain_b * 0.55,
        b[:, 1] * env_b * gain_b * 0.55)
    spl = read_wav("splash_small_646568.wav")
    gain_s = 1.0 / max(1e-9, np.abs(spl).max())
    add(L, R, t_collapse + 0.02, spl[:, 0] * gain_s * 0.35,
        spl[:, 1] * gain_s * 0.35)

    # -------- whooshes on the four cross-section moves --------
    wh = read_wav("whoosh_001_812636.wav")
    wc = loudest_window(wh, 1.5)
    wseg = wh[wc:wc + int(1.6 * SR)]
    wenv = np.concatenate([smooth(np.linspace(0, 1, int(0.25 * SR))),
                           np.ones(len(wseg) - int(0.25 * SR))])
    wenv *= 1.0 - smooth((np.arange(len(wseg)) - 0.7 * SR) / (0.9 * SR))
    wgain = 1.0 / max(1e-9, np.abs(wseg).max())
    wh_in = wseg * wenv[:, None] * wgain * 0.30
    wh_out = wh_in[::-1]
    ev = SB.events()
    for key, osc in (("zoom1_in", True), ("zoom1_out", False),
                     ("zoom2_in", True), ("zoom2_out", False)):
        sig = wh_in if osc else wh_out
        p = -0.35 if osc else 0.35
        gl, gr = pan_gains(p)
        add(L, R, ev[key] / float(FPS) - 0.1,
            sig[:, 0] * gl * 1.4, sig[:, 1] * gr * 1.4)

    # -------- bell cues on the annotation cards --------
    bell = read_wav("bell_ding_333695.wav")
    bell = bell[:int(1.6 * SR)]
    bell /= max(1e-9, np.abs(bell).max())
    rates = [1.0, 0.84, 1.19, 0.9, 1.06, 0.8]
    bi = 0
    finale_t = None
    for t_cue, kind in SB.cues():
        if kind == "beat":
            if bi == 0:
                bi += 1
                continue  # collapse uses the boom instead
            sig = resample_pitch(bell, rates[(bi - 1) % len(rates)]) * 0.24
            gl, gr = pan_gains(-0.25 if bi % 2 else 0.25)
            add(L, R, t_cue + 0.02, sig[:, 0] * gl, sig[:, 1] * gr)
            bi += 1
        elif kind == "finale":
            finale_t = t_cue

    # -------- finale: swell + crash + deep bell --------
    if finale_t is not None:
        lead = 6.0
        m = int(round((lead + 6.5) * SR))
        tt = np.arange(m) / SR
        env = smooth(tt / lead)
        env *= 1.0 - smooth((tt - lead) / 6.0)
        low = (np.sin(2 * np.pi * 55.0 * tt) +
               0.7 * np.sin(2 * np.pi * 82.41 * tt + 0.4))
        sig = 0.075 * low * env
        add(L, R, finale_t - lead, sig, sig)
        fw = booms["finale_wave"]
        fl = int(round(len(fw) * 0.9))
        fb = fw[:fl]
        fenv = np.concatenate([smooth(np.linspace(0, 1, int(0.03 * SR))),
                               np.ones(fl - int(0.03 * SR))])
        fenv *= 1.0 - smooth((np.arange(fl) - 0.9 * SR) / (1.9 * SR))
        fgain = 1.3 / max(1e-9, np.abs(fb).max())
        add(L, R, finale_t - 0.1, fb[:, 0] * fenv * fgain * 0.42,
            fb[:, 1] * fenv * fgain * 0.42)
        dbel = resample_pitch(bell, 0.7) * 0.30
        add(L, R, finale_t + 0.15, dbel[:, 0], dbel[:, 1])

    # -------- master fade-out with the final fade to black --------
    fade_last = int(2.2 * SR)
    fade = 1.0 - smooth(np.linspace(0, 1, fade_last))
    L[-fade_last:] *= fade
    R[-fade_last:] *= fade

    peak = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))))
    if peak > 0.0:
        gain = 0.92 / peak
        L *= gain
        R *= gain

    out = np.empty(2 * n, dtype=np.int16)
    out[0::2] = np.clip(np.round(L * 32767.0), -32768, 32767)
    out[1::2] = np.clip(np.round(R * 32767.0), -32768, 32767)

    path = os.path.join(SB.OUT_DIR, "audio.wav")
    os.makedirs(SB.OUT_DIR, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(out.tobytes())
    print("wrote %s (%.1f s, peak before gain %.3f)" % (path, dur, peak))


if __name__ == "__main__":
    main()
