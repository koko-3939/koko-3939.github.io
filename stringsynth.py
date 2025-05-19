# stringsynth.py — physically detailed Karplus-Strong
# ---------------------------------------------------
#  * 完全物理減衰（ADSR 不使用）
#  * Sullivan 2-tap FIR + 1-pole IIR による高調波ダンピング
#  * RT60 から loop-gain をニュートン法で求める
#  * 分数遅延は 3rd-order Thiran AP
#  * 任意次数の分散 AP（奇数項のみ）をサポート
#  * オーバーサンプリング (×2) 対応
# ---------------------------------------------------

from __future__ import annotations
import csv, os, datetime
import math

_DEBUG_FILE = "ks_debug.csv"
_DEBUG_HEADER_WRITTEN = True      # 1 回だけヘッダを書くフラグ


def _log(stage:str, note:int, vel:int, gate:float, **vals):
    """デバッグ用に 1 行追記。debug=False なら呼ばないこと!"""
    global _DEBUG_HEADER_WRITTEN
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage": stage, "note": note, "vel": vel, "gate": gate, **vals
    }
    # ファイルが無ければ新規、あれば追記モード
    write_header = not _DEBUG_HEADER_WRITTEN and not os.path.exists(_DEBUG_FILE)
    with open(_DEBUG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
            _DEBUG_HEADER_WRITTEN = True
        w.writerow(row)
# ========================================================================

'''デバック用↑'''






from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from modules.hammer import Hammer  

# ---------- 定数 ----------
_FS: int = 44_100

# ==== stringsynth.py の冒頭 ====
# --- default pick-position parameters ---------------------------------
_PICK_BASE      = 0.14   # note60 での基準位置 (14 %)
_PICK_SLOPE     = 0.0007 # 1ノート上がるごとに位置を減らす量

'''固定推奨'''
_PICK_MIN       = 0.04   # 4 % までに下限クリップ 
_PICK_MAX       = 0.26   # 26 % までに上限クリップ

# ==================================================
#   FIR / IIR 減衰フィルタ係数のダイナミック生成
# ==================================================
# 学習パラメータ候補を一箇所にまとめて管理
_FIR_SIG_K     = 2.4   # ← シグモイドの勾配   （学習候補）
_IIR_SIG_K     = 4.0   # ← 同上（IIR 用）      （学習候補）
_IIR_SIG_OFFS  = 0.25  # ← シグモイド中心位置  （学習候補）
# ＊fir_base, fir_note_k, … は dataclass フィールドで既に外部チューニング可






# ---------- Thiran AP 係数 ----------
def thiran_coeff(frac: float, order: int = 3) -> np.ndarray:
    """N 次 Thiran オールパスの分子係数 a[0..N] を返す."""
    assert 0.0 < frac < 1.0
    N = order
    a = np.zeros(N + 1, np.float64)
    for k in range(N + 1):
        num = (-1) ** k
        for m in range(k):
            num *= frac - N + m
        for m in range(k, N):
            num *= frac - m
        den = 1.0
        for j in range(N):
            den *= frac - N + j
        a[k] = num / den
    return a


# ---------- Kaiser half-band FIR (odd length) ----------
def kaiser_halfband(length: int = 31, beta: float = 6.5) -> np.ndarray:
    """0.5 で −80 dB 以上落ちる半帯域フィルタ."""
    assert length % 2 == 1, "length must be odd"
    n = np.arange(length) - (length - 1) / 2
    taps = np.sinc(n / 2) * np.kaiser(length, beta)
    taps /= np.sum(taps)                  # DC 正規化
    taps[length // 2] /= 2.0              # half-band 条件
    return taps.astype(np.float32)


# ======================================================
@dataclass(slots=True)
class KSPhysical:

    # ===== 追加パラメータ ====================================
    unison_break1: int = 40        # 1→2本弦へ切替ノート
    unison_break2: int = 80        # 2→3本弦へ切替ノート
    detune_cents_2: tuple[float,float] = (-0.5, +0.5)     # 2本弦の detune
    detune_cents_3: tuple[float,float,float] = (0.0, -2.0, +2.0)     # 3本弦
    # =========================================================
    
    # ----- global -----
    fs: int = _FS
    oversample: bool = False

    # ----- Sullivan FIR/IIR -----
    fir_base: float = 0.50
    fir_note_k: float = 0.15
    fir_vel_k: float = 0.10
    iir_base: float = 0.45
    iir_note_k: float = 0.30
    iir_vel_k: float = 0.20

    # ----- RT60 -----
    rt60_ref: float = 4.0
    rt60_oct_scal: float = 0.5
    rt60_vel_maxf: float = 1.4

    # ----- loop-gain clip -----
    c_clip_base: float = 0.99995
    c_clip_drop: float = 0.0005

    # ----- dispersion odd-order coeffs (k2, k3, …) -----
    disp_coeffs: Sequence[float] = (0.12, -0.07)   # k2, k3

    # ===== マルチリップル減衰用追加パラメータ ======================
    ripple_fir_note_k: float = 0.04   # 4-tap FIR 用 note 依存勾配
    ripple_fir_vel_k : float = 0.03   # 〃 velocity 依存勾配
    ripple_base      : float = 0.06   # 〃 基準係数
    ripple_gain      : float = 0.0    # ループ内 ripple 係数の重み  ★FIX
    global_gain_db   : float = 6.0    # ← 追加：全体ゲイン調整（dB）

    # ===== 弦間エネルギー結合（ユニゾン干渉） ======================
    coupling_gain: float = 0.05       # 他弦からの受け渡し係数 (0～0.2 程度)
    

    # ----- Note-OFF -----
    off_tau_mult: float = 0.25
    off_interp_ms: float = 5.0
    off_shape: str = "cos"   # "cos" or "exp"

    # ----- init-noise HPF -----
    hp_cut: float = 20.0
    # -------- Attack / Release 動的パラメータ -----------------
    attack_ms_base : float = 3.0   # note60, vel64 の基準 (≈3 ms)
    attack_vel_k   : float = -0.35 # v↑ で接触時間↓
    attack_note_k  : float =  0.25 # 低音ほど接触時間↑
    release_ms_base: float = 40.0  # note60 基準のダンパーフェード (≈40 ms)
    release_note_k : float = -0.5  # 高音ほど速く消える
    
    

    # ==================================================
    #                  internal helpers
    # ==================================================
    def _rt60(self, note: int, vel: int) -> float:
        base = self.rt60_ref * self.rt60_oct_scal ** ((note - 60) / 12)
        v_fac = np.interp(vel, [40, 64, 127], [0.7, 1.0, self.rt60_vel_maxf])
        return max(0.05, base * v_fac)



    def _fir_coef(self, note: int, vel: int) -> float:
        '''
        ▸ 目的 : 2-tap FIR の係数 a_fir (0〜0.5) を
                「音高 note」＋「打鍵強さ vel」から滑らかに算出。
        ▸ 数式
            x = base
                + note_slope * (note-60)/12   # ±オクターブ単位の線形
                + vel_slope  * (vel -64)/63   # velocity を −1〜+1 へ正規化
            a_fir = 0.5 * sigmoid( k * x )    # シグモイドで 0〜0.5 に制限
        '''
        x = ( self.fir_base
            + self.fir_note_k * (note - 60) / 12.0
            + self.fir_vel_k  * (vel  - 64) / 63.0 )
        s = 1 / (1 + np.exp(-_FIR_SIG_K * x))   # sigmoid
    
        return float(0.18*s)      # ★PATCH (高域を殺しすぎない)
    
     # ----------- 新: マルチリップル FIR 4-tap 係数 -----------------
    def _ripple_coef(self, note:int, vel:int) -> tuple[float,float]:
        '''
        2 個のゼロを追加し “低次・高次倍音を選択的に抑える” 4-tap FIR (1 + b1 z⁻¹ + b2 z⁻² + z⁻³)
        b1・b2 を note と velocity で滑らかに変化させる。
        低音・弱打   → 小さくしてローパス緩め
        高音・強打   → 大きくして倍音を速く減衰
        '''
        x  = (self.ripple_base
             + self.ripple_fir_note_k * (note - 60)/12.0
             + self.ripple_fir_vel_k  * (vel  - 64)/63.0)
        b1 =  x * 0.4        # ★半分に引き下げ
        b2 = -x * 0.3        # ★同上
        return float(b1), float(b2)

    def _iir_coef(self, note: int, vel: int) -> float:
        '''
        ▸ 目的 : 1-pole IIR の極位置 d_iir (≈0.3〜0.95) を決定。
        極が 1 に近いほどローパスが強く → 高調波が早く減衰。
        ▸ 数式
            x = base + note_slope·… + vel_slope·…
            d_iir = 0.98 / (1 + exp( -K·( x - offset ) ) )
        ※ 0.98 は「極が 1 を超えない」安全上限。
        '''
        x = ( self.iir_base
            + self.iir_note_k * (note - 60) / 12.0
            + self.iir_vel_k  * (vel  - 64) / 63.0 )
        return float(
            0.95 / (1.0 + np.exp(-_IIR_SIG_K * (x - (_IIR_SIG_OFFS + 0.08))))
        )


    # ---------- 新しい公開 API ----------
    def generate(
        self,
        f0: float,
        duration: float,
        *,
        note: int,
        velocity: int,
        gate: float,
        pick_pos: Optional[float] = None,
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> np.ndarray:
        '''ユニゾン 1〜3 弦を内部生成して合成するラッパー'''

        # ---- 1. 何本弦か決定（ピアノ実機の張弦数に倣う） ----
        if note < self.unison_break1:
            detune_cents = (0.0,)                       # 1 本弦
        elif note < self.unison_break2:
            detune_cents = self.detune_cents_2          # 2 本弦
        else:
            detune_cents = self.detune_cents_3          # 3 本弦

        # ---- 2. 各弦を個別生成 -------------------------------
        outs: list[np.ndarray] = []
        for idx, cents in enumerate(detune_cents):
            f_det   = f0 * 2.0 ** (cents / 1200.0)      # 周波数オフセット
            seed_i  = None if seed is None else (seed + idx * 101) & 0xFFFFFFFF
            out_i   = self._generate_single(
                f_det,
                duration,
                note     = note,
                velocity = velocity,
                gate     = gate,
                pick_pos = pick_pos,
                seed     = seed_i,
                debug    = debug and idx == 0,          # デバッグは 1 本目だけ
            )
            outs.append(out_i.astype(np.float64))

        # ---- 3. 長さを揃えて平均し音量を規格化 ---------------
        max_len = max(len(o) for o in outs)
        mix = np.zeros(max_len, dtype=np.float64)
        for o in outs:
            mix[:len(o)] += o
        # ---- 4. 弦間弱結合 (ブリッジ共振の簡易近似) -------------
        if len(outs) >= 2 and self.coupling_gain > 1e-6:
            avg = np.mean(np.vstack([o[:max_len] for o in outs]), axis=0)
            mix = (1-self.coupling_gain)*mix + self.coupling_gain*avg

        # 3 本ユニゾンでも実機と同じ音圧になるよう 1/√N で正規化
        mix /= math.sqrt(len(outs))
        # 出力ゲイン（ハンマー‐>弦エネルギ較正）
        mix *= 10 ** (self.global_gain_db / 20)  

        return mix.astype(np.float32)






    # ---------- 旧 1 本弦ロジックをそのまま移動 ---------------
    def _generate_single(          # ★旧 generate の本体★
        self,
        f0: float,
        duration: float,
        *,
        note: int,
        velocity: int,
        gate: float,
        pick_pos: Optional[float],
        seed: Optional[int],
        debug: bool,
    ) -> np.ndarray:
        
        # ---------- derived constants ----------
        '''
        1.	内部サンプリング周波数を決める。
        2.	目標 合成長（サンプル数） を算出。
        3.	基本周波数から ディレイライン長（整数＋分数） を求め、分数部は後でオールパスで補正。
        4.	遅延が物理モデル最小長に達しない場合は 即エラーで中断。
        '''

        # 物理計算を行う内部サンプリング周波数 [Hz]。オーバーサンプリングが有効なら2倍にして高域ノイズを後段で除去する。44.1kHz→88.2kHz。
        fs_int = self.fs * (2 if self.oversample else 1)  
        
        # 合成全体のサンプル数。duration[秒]にfs_int[Hz]を掛け整数化することでループ回数を決定する。
        N = int(duration * fs_int)  
        
        # 基本周波数f0での一周期が何サンプルかを浮動小数で計算。例: 88,200/440≈200.5サンプル。
        delay = fs_int / f0  
        
        # 遅延ラインの整数部（ディジタル波形メモリ長）。最低でも3サンプル必要。
        D_int = int(delay)  
        
        # 遅延の小数部（0≤frac<1）。後でThiranオールパスで“分数遅延”を再現し音程を微調整。
        frac = delay - D_int  
        
        # 遅延ラインが短すぎるとKarplus–Strongループが成立せず、フィルタを挿入する余地もない
        # 実質的にf0が高すぎる場合のガード。このモデルでは超高音はサポートしない。
        if D_int < 3:  
            raise ValueError("delay too short")  
        
        
        if debug:                                                   # ★①
            _log("derived", note, velocity, gate,
                fs_int=fs_int, N=N, delay=delay, D_int=D_int, frac=frac)


        # ---------- initial buffer (white noise) ------------------------------------
        '''
        目的
            1. 遅延ライン長 D_int に合わせてホワイトノイズを生成し、弦全体をランダムに初期変位させる。
            2. `numpy.random.default_rng(seed)` は PCG64 系列を使う現代的な乱数生成器。
            同じ seed を与えると常に同じ乱数系列が得られるため、機械学習データセットの再現性が担保できる。
            3. `rng.uniform(-1, 1, D_int)` は区間 [-1, 1) の一様分布。
            周波数特性がフラット なホワイトノイズが得られる。
            4. `astype(np.float64)` により 64bit 浮動小数へ昇格。 長時間減衰計算でも丸め誤差で振幅がゼロ潰れしにくい。
        '''
        rng = np.random.default_rng(seed)                       # 再現性付き RNG (PCG64)
        # --- 初期励振: フェルト接触形状で帯域コントロール ----------
        hlen  = int( max(1, 0.6 * D_int / (velocity/64)**0.3) )  # 弱打で長く・強打で短い
        win   = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, hlen))
        noise = rng.uniform(-1.0, 1.0, D_int).astype(np.float64)
        noise[:hlen] *= win                                       # フェルト接触包絡
        # --- ripple FIR（1 回だけ掛ける）-------------------
        if self.ripple_gain > 1e-6:
            b1, b2 = self._ripple_coef(note, velocity)
            rf = np.array([1.0, b1, b2, 1.0], np.float64)
            buf = np.convolve(noise, rf, "same") * self.ripple_gain \
                  + noise * (1.0 - self.ripple_gain)
        else:
            buf = noise
        
        
        
        if debug:                                                   # ★②
            _log("hpf", note, velocity, gate,
                rms_hp=np.sqrt(np.mean(buf**2)))




        # ---------- high-pass filter (1-pole) ---------------------------------------
        '''
        ＜なぜ DC / 低周波 を取り除くのか？＞
            ▸ **DC 成分**  ＝ 信号の平均値（0 Hz）。波形全体が上下どちらかへ偏ること。
            ▸ **低周波成分** ＝ 20 Hz 以下など、人間が周期として感じないほど遅い揺れ。
            ▶ 例：バネをゆ～っくり押し続ければ「揺れ」というより傾きになる。

            ▸ 弦モデルの理想状態 … 静止位置（平均）が 0。
            ● DC が残ると「弦が片側に押し出されてから振動開始」→
                ・シミュレーションでは遅延ラインにオフセットが溜まり発振しやすい
                ・実際の音にも「ボコッ」とした低域ノイズが乗る

        ＜1-pole ハイパスの動作（デジタル版 RC フィルタ）＞
            * 差分方程式
                y[n] = a * y[n-1] + x[n] - x[n-1]
                x[n] - x[n-1] : ほぼ「微分」→ 低い周波数ほど値が小さくなるので減衰
                a * y[n-1]    : 微分だけだとチリチリするので少し戻して安定させる
            * 係数 a = exp(-2π f_c / f_s)
                f_c = 20 Hz（切り替え周波数）  f_s = サンプリング周波数
                → ここでは a ≈ 0.997。−3 dB 点が約 20 Hz になるよう調整。
            * 配列の最初 (n=0) は x[n-1] が無いので、最後のサンプル buf[-1] を
            参照して「輪っか」にし、クリックを防ぐ。

        ＜結果＞
            1. DC と 20 Hz 以下の「ゆっくり傾き」をほぼゼロに。
            2. 100 Hz 以上の音声帯域はほぼそのまま通過。
            ◇ 弦の平均位置が 0 に戻り、数値的にも耳にも安定した初期励振を得られる。
            
            
        参考文献　https://www.musicdsp.org/en/latest/Filters/237-one-pole-filter-lp-and-hp.html
        '''

        alpha_hp = np.exp(-2.0 * np.pi * self.hp_cut / fs_int)  # ハイパス係数 a
        hp_prev  = 0.0                                          # y[n-1] 初期値

        for i in range(D_int):
            x_cur  = buf[i]                                    # 現在サンプル
            x_prev = buf[i-1] if i else buf[-1]                # n=0 だけ末尾を参照
            y_hp   = alpha_hp * hp_prev + x_cur - x_prev       # ハイパス差分方程式
            hp_prev = y_hp
            buf[i]  = y_hp                                     # DC 除去後の値で上書き
            
        
        # ---- Attack 窓：note / vel 依存の接触時間 ----------------
        att_ms  = ( self.attack_ms_base *
                    (velocity / 64)**self.attack_vel_k *
                    (note     / 60)**self.attack_note_k )
        att_len = int(att_ms * 1e-3 * fs_int)
        # 遅延ライン長を超えないよう安全クリップ
        att_len = max(1, min(att_len, D_int))

        if att_len > 1:                     # 長さ1ならフェード不要
            win = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, att_len))
            buf[:att_len] *= win
            


        # --- ピック位置ノッチ（特定倍音の削減）------------------------------------
        '''
        【なにをしたい？（目的）】
            ● ギターやピアノの弦は「弾く場所」で音色が変わる。
            ・端（ブリッジ近く）で弾く → キラキラした高音が目立つ  
            ・真ん中寄りで弾く        → まろやかな低音が残る  
            ● これを数式でまねしておくと、打鍵位置を変えるだけで
            “明るい／丸い” 音色差が作れる。

        【どうやって？（やり方）】
            1) まず「どこを弾いたか」を 0～1 の割合で決める  
                pick_pos = 0.14 − 0.0007 × (note − 60)
                - 中音（note60）は 0.14 ≒ 弦長の14%  
                - 高音ほど値が小さく（端寄り）、低音ほど大きく（中央寄り）  
                - 最小4%、最大26%にクリップして安全確保
            2) 弦の長さ D_int サンプルに合わせて整数化  
                k_pp = round(pick_pos × D_int)
            3) 2-tap フィルタ「1 + z⁻ᵏ」をかける  
                → ちょうど波長が k_pp の倍音だけ “打ち消しあい” ＝ ノッチ

        【なにが起こる？（結果）】
            ● pick_pos が端寄り(小)  → 高次倍音が残り、キラッとした音  
            ● pick_pos が中央寄り(大)→ 低次倍音中心、ふくよかな音  
            ─ 実際に弦を弾く場所の違いを、乱数バッファに一手間かけるだけで再現できる。
        '''
        if pick_pos is None:
            pick_pos = np.clip(
                _PICK_BASE - _PICK_SLOPE * (note - 60),
                _PICK_MIN, _PICK_MAX
            )
        k_pp = int(np.clip(round(pick_pos * D_int),            # ノッチ遅延長 [sample]
                        1, D_int - 1))
        buf[: D_int - k_pp] += buf[k_pp:]                      # 1+z^-k FIR で倍音削減
        
        
        if debug:                                                   # ★③
            _log("notch", note, velocity, gate,
                pick_pos=pick_pos, k_pp=k_pp, rms_notch=np.sqrt(np.mean(buf**2)))



        # --- 初期振幅スケーリング ---------------------------------------------------
        '''
        目的
            ▶ 打鍵の強さ (velocity) だけを “ハンマーの運動エネルギー” に基づいて反映。
            ▶ 音高による主観ラウドネス補正は行わず、純粋な物理挙動を評価する。

        アルゴリズム
            hammer_amp = √E_ratio
                • ハンマー質量 m = 3 g，衝突速度 v による運動エネルギー
                E = ½ m v² を最大値で正規化し、その平方根を取る。
                • 平方根を使うことで「打鍵強さと感覚的音圧 (≈dB) がほぼ比例」。
            buf *= hammer_amp
                • 生成したホワイトノイズを hammer_amp 倍して、
                “弦へ入力される初期変位量” を決める。

        結果
            - velocity↑ → 振幅↑ (運動エネルギーに比例)  
            - note による補正 0 → 音高による主観バランスを排除  
            これにより、以降の Karplus-Strong ループへ **純粋な物理エネルギー量** が渡る。
        '''
        
        
        hammer_amp   = 0.75 * Hammer().velocity_to_amplitude(velocity)
        buf *= hammer_amp 
                



        # ---------- damping filter coeffs -------------------------------------------
        '''
        【なにをしている？】
        ▸ 弦ループの中には 2 つの “音を弱めるフィルタ” が入っている
                ● 2-tap FIR（フィルタその1） … 「いまの音」と「1つ前の音」を混ぜて
                                                高い音ほど小さくする弱いローパス
                ● 1-pole IIR（フィルタその2） … 1 回遅れをフィードバックして
                                                なだらかなローパスを作る
        ▸ 打鍵した音の高さ(note)や強さ(velocity)によって
            「どのくらい弱めるか」を毎回計算し、
            ループ全体の減衰量 H_mag を求める。

        【手順をざっくり】
        1️まずフィルタ係数 a_fir, d_iir を
            ▸ _fir_coef(note, vel)
            ▸ _iir_coef(note, vel)
            で取り出す。  
            → 高音・強打になるほど “たくさん弱める” 値が返るようになっている。

        2️そのノートの基本周波数 f0 に対応する角周波数
                w0 = 2π·f0 / fs_int
            を計算。（1 周期のスピードをラジアンで表したもの）

        3️「今のフィルタが f0 の音をどれだけ通すか」
            を数式で求める。  
                H_FIR = |1 + a·e^{-jw0}|   ← 2-tap FIR の振幅  
                H_IIR = |(1-d)/(1-d·e^{-jw0})| ← 1-pole IIR の振幅

        4️これらを掛け合わせて
                H_mag = H_FIR × H_IIR
            とする。  
            → 1 より小さければ “その音が弱まる” 倍率になる。

        5️数値誤差でゼロ割りが起きないよう
                H_mag が 1e-6 より小さくならないようにクリップ。

        【どう役立つ？】
        • H_mag がわかると「ループの中で基本音が何 % 減衰するか」が判る。  
        • このあと計算する ループゲイン c を
                c = 目標の減衰量 ÷ H_mag
            に設定すれば **RT60（余韻時間）を狙いどおりに合わせられる**。
        • つまり H_mag は “現状フィルタによる損失” を測るリファレンスで、
            これを元にゲインを補正してやれば
            高音でも低音でも望みの長さで鳴って消えてくれる。
        '''

        a_fir = self._fir_coef(note, velocity)         # 2-tap FIR 係数 a
        b1,b2 = self._ripple_coef(note, velocity)  # ★追加 4-tap 用係数
        d_iir = self._iir_coef(note, velocity)         # 1-pole IIR 係数 d

        w0 = 2.0 * np.pi * f0 / fs_int                 # 基本角周波数 [rad/sample]
        
        
        H2 = abs(1 + b1*np.exp(-1j*w0) + b2*np.exp(-2j*w0) + np.exp(-3j*w0))
        H_fir = abs(1 + a_fir*np.exp(-1j*w0))
        H_iir = abs((1-d_iir) / (1-d_iir*np.exp(-1j*w0)))

        H2 = np.clip(H2, 0.25, 1.0)           # ★下限を 0.25 (-12 dB) に
        H_mag = max(H_fir * H_iir * H2, 1e-6)
        


        # ---------- loop gain --------------------------------------------------
        '''
        【目的】  
            ループを 1 周（＝ 1 基本周期）するごとに  
            「目標 RT60 で −60 dB になる」ようなゲイン c を決める。

        【ステップ詳細】

        1. 目標 RT60 を取得
                target_rt60 = _rt60(note, velocity)
                ─ 打鍵の音高と強さで決まる余韻時間 [秒]

        2. RT60 から 1 周期あたりの損失 g_loop を計算  
                g_loop = 10 ** ( -3 / ( target_rt60 · f0 ) )
                ─ −60 dB = 10^(−3) ≈ 0.001  
                ─ 周期数 N = RT60 · f0  なので  
                1 周期あたり倍率 = 0.001^(1/N)

        3. フィルタで失われる分を補正  
                c = g_loop / H_mag
                ─ H_mag は「FIR×IIR」の損失。  
                乗じて g_loop になるよう c を設定。

        4. 誤差を 1 ステップだけニュートン補正  
                delta = (H_mag·c − g_loop) / H_mag
                if |delta| < 0.2:
                    c -= delta
                ─ 近似誤差が 20 % 未満なら 1 回だけ修正。  
                H_mag が実数なので 1 ステップで十分収束。

        【結果】  
            • (フィルタ損失) × c = g_loop となり、  
            基本波は 1 周で “ちょうど必要な減衰量” に整う。  
            • この後に安全クリップが掛かり  
            発振しない範囲 0 < c < c_max(≈0.93) に制限される。
        '''
        target_rt60 = self._rt60(note, velocity)            # 1.
        g_cycle     = 10.0 ** (-3.0 / (target_rt60 * f0))   # 周期あたり
        g_per       = g_cycle ** (1.0 / D_int)              # 1 sample
        c           = g_per / H_mag

        # --- Newton correction (fixed) ---
        for _ in range(3):
            delta = (H_mag*c - g_per) / H_mag
            c    -= delta * 0.7               # 収束を緩め安定化
            if abs(delta) < 1e-3:
                break
            
        
        
        
        if debug:                                                   # ★④
            _log("damping", note, velocity, gate,
                H_fir=H_fir, H_iir=H_iir, H_mag=H_mag,
                c_preclip=c, g_loop=g_loop)
            
        
        # ---------- 安全クリップ ------------------------------------------------
        '''
        【目的】
            求めたゲイン c が
            ●  0 未満にならない（負ゲインは位相反転＆発振）
            ●  1.0 を超えて発振しない
            よう **音域に応じた上限 c_max** を設定し、c を丸める。

        【式の中身】

        1) oct_shift = (note − 60) / 12
            ─ 基準ノート 60 (C4) から何オクターブ上下か
                例: note 25 → −35/12 ≈ −2.92 oct
                    note100 →  40/12 ≈  3.33 oct

        2) k = clip(oct_shift, −2, +3)
            ─ 丸めて 低音は −2 oct、高音は +3 oct までに制限  
                （それ以上は弦モデル対象外なので固定値扱い）

        3) c_max = (c_clip_base − c_clip_drop · k) · 0.94
            ─ ベース値 0.9995 から  
                音域が 1 oct 上がるごとに 0.0025 ずつ下げる。  
                さらに 0.94 を掛け、最終上限を **約 0.93–0.94** に固定。  
                * 理由：高音ほどループが短く、わずかなオーバーゲインで発振しやすい。  
                低音は少し余裕を持たせても安全。*

        4) c = clip(c, 0.0, c_max)
            ─ 実数部だけ取り、
                0 ≤ c ≤ c_max に丸める（ハードクリップ）。

        【結果】
            * 低音 (note25, k=−2) → c_max ≈ 0.944  
            * 中音 (note60, k=0)  → c_max ≈ 0.939  
            * 高音 (note100,k=+3) → c_max ≈ 0.932  

            これにより **全音域で |c·H| < 1** が保証され、
            丸め誤差や突発的ピークによる発振を確実に防止します。
        '''
        oct_shift = (note - 60) / 12                     # 1)
        k         = np.clip(oct_shift, -2, 3)            # 2)
        # ① 周期ベースの上限値
        c_max_period = (self.c_clip_base - self.c_clip_drop * k) * 0.94
        if note >= 96:                      # 17 kHz 以上は緩めに
            c_max_period = 3.0
        # ② 1 sample あたりへ変換
        c_max_sample  = c_max_period ** (1.0 / D_int)
        c = float(np.clip(c.real, 0.0, c_max_sample))

        
        
        if debug:                                                   # ★⑤
            _log("clip", note, velocity, gate,
                c_final=c, c_max=c_max_period)
        
        
        # -------- フラクショナル遅延 (Thiran) と 分散オールパス ---------
        # ----------------------------------------------------------------------
        # ❶ 「frac サンプル」だけ音を遅らせたい   ( 0 ≤ frac < 1 )
        # ----------------------------------------------------------------------
        #    ───── どうして必要？ ───────────────────────────────────
        #      • ディレイライン長 D_int は整数サンプルしか置けない。
        #      • 例えば 200.37 サンプル 欲しいときは
        #          → 200 サンプル + 0.37 サンプル（分数部）が不足。
        #      • 分数部は “位相だけ回すオールパス” で擬似的に実現するのが
        #        Karplus-Strong の定番手法。
        #
        #    ───── Thiran 3rd-order AP とは？ ────────────────────────
        #      • 全帯域で **ゲイン=1.0（音量変化ゼロ）** のまま、
        #        位相を滑らかに回せる IIR フィルタ。
        #      • 群遅延をほぼフラットに保てるので、弦モードの
        #        周波数間バランスが崩れない＝音色が変わらない。
        #
        #    ───── 実装 ───────────────────────────────────────
        a_th = thiran_coeff(frac, order=3) if frac > 1e-6 else np.array([1, 0, 0, 0])
        #              ↑       ↑
        #         分数遅延      3次AP (4 個係数)
        #      frac≃0 なら係数 [1,0,0,0] → つまり何もしない

        # ----------------------------------------------------------------------
        # ❷ 「弦の剛性で高音ほどピッチが上ずる」現象にちょっとだけ対策
        # ----------------------------------------------------------------------
        #    ───── 分散（dispersion）って？ ────────────────────────
        #      • 理想弦 = 張力 Only → 倍音は 1f, 2f, 3f… と完全整数倍。
        #      • ピアノ弦は **曲げ剛性** が無視できず、周波数が高いほど
        #        位相速度が少し速くなり「整数倍より上にズレ」= 分散。
        #      • 実機の音質(“キラッ”)を真面目に出すには必要。
        #
        #    ───── どう補正する？ ────────────────────────────────
        #      • 奇数次数 (k₂, k₃, …) のオールパスを **直列に** 重ねると
        #        「周波数が高いほど遅延時間を足す」性質を作れる。
        #      • 係数の符号と大きさで “足す / 引く” や強さを微調整。
        #
        #    ───── 安全クリップ ────────────────────────────────
        #      • 0.3 を超えると位相が暴れ、発振 or 音質破綻の恐れあり。
        #      • 論文や実測から k≃0.05～0.15 程度が一般的なので、
        #        万一設定ミスをしても ±0.3 でガードする。
        disp_k  = [np.clip(k, -0.3, 0.3) for k in self.disp_coeffs]   # 奇数次数AP係数
        n_disp  = len(disp_k)                                         # 段数を保存
        #      例: disp_coeffs = (0.12, -0.07) → 2 段カスケード
        #
        #   ループ本体では
        #      for j, k_ap in enumerate(disp_k):
        #          y_d, z[j] = y_d + k_ap * (y_d - z[j]), y_d
        #   という 1 行 IIR を n_disp 回まわし、
        #   ほんの僅かだけ高域を遅らせ “整数倍より上ずれ” を軽減する。
        
        
        
        # ---------- loop state ----------
        # -------------------------------------------------------------------------
        # ここから先は「Karplus–Strong ループ」を 1 サンプルずつ回すための
        # **内部状態 (state variables)** をゼロ初期化し、
        # ノート OFF 時にゲインを滑らかに下げるフェード区間を算出するブロック。
        # ─────────────────────────────────────────────
        # 1.  out      : 出力波形バッファ  (= 完成品) を先に確保
        # 2.  head     : リングバッファ buf の読み書き位置ポインタ
        # 3.  y_prev   : 2-tap FIR の「1 サンプル遅れ」メモリ   (z⁻¹)
        # 4.  lp_state : 1-pole IIR (ローパス) の前回出力       (極位置 d_iir 用)
        # 5.  disp_z   : 分散オールパス各段の z⁻¹ メモリ        (弦剛性補正)
        # 6.  th_z     : Thiran 3 次 AP の z⁻¹,z⁻²,z⁻³ メモリ   (分数遅延)
        # 7.  off_start: gate 秒経過時に「鍵盤を離した」扱いでゲインを下げ始める位置
        # 8.  off_len  : ゲインを c → c·(1-off_tau_mult) へ補間するサンプル長
        #                 - self.off_interp_ms [ms] を使うが 30 ms 未満なら 30 ms に切り上げ
        # -------------------------------------------------------------------------

        out      = np.empty(N, np.float64)      # (1) 出力バッファ確保
        head     = 0                            # (2) 遅延線の現在位置インデックス
        y_prev   = 0.0                          # (3) 2-tap FIR 用の z⁻¹
        y2_prev  = 0.0                          # z⁻² (4-tap FIR 用)
        y3_prev  = 0.0                          # z⁻³ (4-tap FIR 用)
        lp_state = 0.0                          # (4) 1-pole IIR の前回出力
        disp_z   = np.zeros(n_disp, np.float64) # (5) 分散 AP 各段の z⁻¹
        th_z     = np.zeros(3, np.float64)      # (6) Thiran 3 次 AP の z⁻¹,z⁻²,z⁻³

        # ―――― ノート OFF（ダンパー接触）に向けたゲイン補間設定 ――――
        off_start = int(gate * fs_int)          # (7) ノート ON 区間の終了位置 [sample]
        rel_ms  = ( self.release_ms_base *
                    (note / 60)**self.release_note_k )
        off_len = max(1, int(rel_ms * 1e-3 * fs_int))
        # 出力残りフレームより長過ぎないよう制限
        off_len = min(off_len, max(1, N - off_start))
        
        
        for n in range(N):
            '''
            ---------------------------------------------------------------
            【ループ本体：1 サンプル進行の流れ】
            1) 遅延線 buf から現在サンプル y を読み出す
            2) Sullivan 2-tap FIR でピック位置に応じた高域減衰
            3) 1-pole IIR で周波数依存のローパス減衰
                └ soft-clip (tanh) を挟み飽和歪み＆数値暴走を防止
            4) 分散オールパス (奇数次数 k₂,k₃,…) をカスケード
                └ ピアノ弦の剛性による「高次倍音のピッチ上ずり」を補正
            5) ここで出来た y_d が「遅延線へ戻す直前」の最新波形
            ---------------------------------------------------------------
            '''

            y = buf[head]                              # 遅延線の現フレーム値を取得

            # --- 2-tap FIR --------------------------------------------------
            y0     = y                               # x[n]
            y      = y0 + a_fir * y_prev             # x[n] + a·x[n-1]


            # ------- シフトレジスタ更新（z⁻¹, z⁻², z⁻³） -------
            y3_prev = y2_prev
            y2_prev = y_prev
            y_prev  = y0

            # ------ シフトレジスタ更新 ---------------------------------------
            y3_prev = y2_prev
            y2_prev = y_prev
            y_prev  = y0

            # --- 1-pole IIR (ローパス) ＋ soft-clip ------------------------
            y_lp   = (1 - d_iir) * y + d_iir * lp_state  # y と前回出力の加重平均        
            lp_state = y_lp                              # IIR の状態 z⁻¹ を更新

            # --- 分散オールパス (odd-order APs) ---------------------------
            y_d = y_lp                                   # 入力を y_d にセット
            amp = np.clip(abs(y_lp), 0.0, 1.0)
            disp_scale = 1.0 + 0.4*amp
            y_d = y_lp
            for j, k_ap in enumerate(disp_k):
                k_dyn = k_ap * disp_scale
                y_d, disp_z[j] = (y_d + k_dyn * (y_d - disp_z[j]), y_d)

            # ---- 軽い soft-clip (非線形倍音生成) ------------------
            y_d = np.tanh(y_d * 0.5)




            # ───────────────────────────────────────────────────────────────────────────
            #  Thiran 3-rd-order fractional delay  +  安全ソフトクリップ
            #  --------------------------------------------------------------------------
            #  ❚ 目的
            #     ▸ 遅延線の「分数サンプル不足 (0 ≤ frac < 1)」を *完全ゲイン 1.0* のまま補う。
            #     ▸ Thiran AP は群遅延がほぼ定数 ➜ 弦モード間の位相関係（＝音色）を崩さない。
            #     ▸ ループ内に突発的ピークが入っても発振しにくいよう、出力と内部状態を
            #       tanh でソフトクリップして数値発散を防ぐ。
            #
            #  ❚ アルゴリズム
            #       H(z) =  (a0 + a1 z⁻¹ + a2 z⁻² + a3 z⁻³)
            #              ---------------------------------    （3次 Thiran オールパス）
            #               1 + a1 z⁻¹ + a2 z⁻² + a3 z⁻³
            #
            #     係数 a0..a3 は  thiran_coeff(frac, order=3) で事前計算。
            #     転置ダイレクトフォーム-II を展開し、掛け算4・加算4・
            #     メモリ3個 (th_z[0..2]) で 1サンプル更新を実装している。
            #
            #  ❚ 変数
            #     x_ap      : フィルタ入力（ここでは分散 AP 後の y_d）
            #     y_ap      : フィルタ出力 = 分数遅延が補正されたサンプル
            #     th_z[0..2]: z⁻¹, z⁻², z⁻³ の内部状態（次サイクルまで保持）
            #
            #  ❚ ソフトクリップ
            #     y_ap, th_z をまとめて
            #         tanh(0.25·x) · 4.0   （≈ ±4 で滑らかに頭打ち）
            #     に通すことで、
            #       • 数値オーバーフロー／NaN 防止
            #       • ラウドな発音時の過度なひずみを回避
            #     Karplus-Strong ループ固有の「暴発」を実用範囲で抑えるテクニック。
            # ───────────────────────────────────────────────────────────────────────────
            # -------- Thiran 3-rd-order fractional delay + soft-clip ---------------
            x_ap   = y_d                    # フィルタ入力
            y_ap   = a_th[0] * x_ap + th_z[0]

            th_z[0] = a_th[1] * x_ap - a_th[0] * y_ap + th_z[1]
            th_z[1] = a_th[2] * x_ap - a_th[1] * y_ap + th_z[2]
            th_z[2] = a_th[3] * x_ap - a_th[2] * y_ap           # z⁻³ 更新

            # ── 安全ソフトクリップ（出力 & 内部状態を同時に抑制）
            y_ap    = np.tanh(y_ap * 0.15) * 2.0
            th_z[:] = np.tanh(th_z * 0.15) * 2.0
            
            
            

            # ───────────────────────────────────────────────────────────────────────────
            #  ⧉ Variable-gain section ― ノート OFF 後の減衰を物理らしく制御する
            #  --------------------------------------------------------------------------
            #  ❚ 目的
            #     • ループゲイン c を「ノートが押されている間」と
            #       「ダンパーが当たった後」で段階的に変化させる。
            #     • ピアノのダンパーを模擬：key-off 直後に数十 ms で
            #       ゲインを c → c·(1-off_tau_mult) へ滑らかに落とす。
            #
            #  ❚ 主要パラメータ
            #     off_start       : gate 秒が経過したサンプル位置。
            #                       ここを境に key-off フェーズへ移行。
            #     off_len         : key-off フェーズの長さ（release_ms_base 由来）。
            #     off_tau_mult    : ゲインをどこまで下げるかの割合 (例 0.25 → 75 % に低下)。
            #     off_shape       : フェード曲線  "exp"=指数  / "cos"=raised-cosine。
            #
            #  ❚ g の決定ロジック
            #        n < off_start            : g = c                      （演奏中）
            #        off_start ≤ n < off_end  : g が c → c·(1-off_tau) へ補間
            #        n ≥ off_end              : g = c·(1-off_tau)          （完全ダンパー）
            #
            #        w =  { exp(-5t)                  （指数）      
            #            or 0.5·(1+cos πt) }         （コサイン）
            #        t = 0…1 で時間正規化
            #
            #  ❚ 出力の安全処理
            #     • next_samp = g * y_ap                     : ループに可変ゲインを掛ける
            #     • 非有限値 (NaN/Inf) をゼロに置換
            #     • np.clip(-4, +4) で最終的な振幅暴走を防ぐ
            #     • buf[head] に戻し out[n] に記録、head を 1 進めリングバッファ更新
            # ───────────────────────────────────────────────────────────────────────────
            # -------- variable gain (key-off damping) -------------------------------
            if n < off_start:                           # 鍵が押されている間
                g = c
            elif n < off_start + off_len:               # key-off フェード区間
                t = (n - off_start) / max(1, off_len)   # 0 → 1
                if self.off_shape == "exp":
                    w = np.exp(-5.0 * t)                # 指数カーブ
                else:                                   # "cos"
                    w = 0.5 * (1 + np.cos(np.pi * t))   # raised-cosine
                g = c * (1 - self.off_tau_mult * (1 - w))
            else:                                       # ダンパー完全接触後
                g = c * (1 - self.off_tau_mult)

            next_samp = g * y_ap                        # 可変ゲインを適用

            # ---- 数値安全ガード ----------------------------------------------------
            if not np.isfinite(next_samp):              # NaN / Inf なら無音化
                next_samp = 0.0

            buf[head] = next_samp                       # *** フィードバック ***
            out[n]    = next_samp                       # 出力バッファへ書き出し
            head      = (head + 1) % D_int              # リングバッファを 1 サンプル進める



        # ───────────────────────────────────────────────────────────────────────────
        #  ⧉ Oversampling → Decimation (×2 → ÷2) で折り返しノイズを抑制
        #  --------------------------------------------------------------------------
        #  ❚ 目的
        #     • 物理計算フェーズでは fs_int = 2·fs (= 88.2 kHz) で演算し、
        #       高域に生じる量子化ノイズや非線形クリップのエイリアシングを回避。
        #     • 出力段で 44.1 kHz へ戻す際、**half-band FIR** を通してから
        #       2 サンプルごとに間引き (↓2) することで折り返しノイズを −80 dB 以上抑制。
        #
        #  ❚ 実装詳細
        #     hb = kaiser_halfband(63, 6.5)
        #         • 長さ 63 tap、β=6.5 の Kaiser 窓 half-band フィルタを動的生成。
        #         • 「half-band」特性: 通過帯・阻止帯が 0.5 fs で対称、
        #           タップの半分がゼロになるため計算量を約 ½ に削減できる。
        #
        #     out = np.convolve(out, hb, "same")[::2]
        #         1) np.convolve(…, "same") で インライン FIR フィルタリング
        #            - 配列長を変えず中央合わせ。  
        #         2) [::2] で偶数インデックスを抜き取り、サンプリング周波数を ½ に。
        #            → fs_int (= 88.2 kHz)  →  fs (= 44.1 kHz)
        #
        #  ❚ 効果
        #     • 0.22 fs 付近から先の折り返し成分を十分減衰させた上で間引くため、
        #       最終波形に高域エイリアシングがほぼ残らない。
        #     • 63 tap × half-band により CPU コストと音質のバランス良好。
        # ───────────────────────────────────────────────────────────────────────────
        if self.oversample:
            hb = kaiser_halfband(63, 6.5)
            out = np.convolve(out, hb, "same")[::2]
            
        
        

        # --- 出力段で軽くノーマライズ (-3 dBFS) -----------
        peak = np.max(np.abs(out)) + 1e-12
        out  = 0.7071 * out / peak      # √0.5 ≈ -3 dB
        return out.astype(np.float32)
