# conv_bench — RBW ⊗ Gaussian convolution: TensorFlow FFT vs direct numerical integral

Benchmark of two ways to smear a relativistic Breit–Wigner (J/ψ in dimuon q²) with a Gaussian q²
resolution of σ = 0.04 GeV², on **gpu01.dice.priv** (Tesla T4). Started 2026-10-16.

## Where things live

| What | Path |
|---|---|
| Project on gpu01 (canonical) | `/storage/kp14102/conv_bench` (node-local 1.8 TB disk; symlink `~/conv_bench`) |
| Local mirror (this dir) | `/Users/kp14102/Work/claude/conv_bench` |
| Notebook | `conv_bench.ipynb` (executed in place; generated from `make_notebook.py`, **never edit the .ipynb directly**) |
| Env | `venv/` (Python 3.12, TensorFlow 2.21 + CUDA 12 pip wheels, scipy 1.18, numpy 2.5, pandas, tabulate, jupyter); kernel "Python 3.12 (conv_bench)"; rebuild with `bash setup.sh` |
| Outputs (current notebook) | `plots/lineshape_N18_{f64,f32}.png`, `plots/timing_scan_{f64,f32}.png`, `results/timings_{f64,f32}.csv`, `results/summary_{f64,f32}.md`, `plots/cost_model.png` (analytic n_out×N vs N log N, dtype-independent) |
| Legacy outputs | `plots/lineshape_N18.png`, `plots/timing_scan.png`, `results/timings.csv`, `results/summary.md` — from the original 10-config version (CPU + numpy backends), kept as the only record of those numbers |

Run: `ssh gpu01.dice.priv` (via seis jump host, see `~/.ssh/config`), then
`cd ~/conv_bench && ./venv/bin/jupyter nbconvert --to notebook --execute --inplace conv_bench.ipynb`
(≈ 1 min). Edit `make_notebook.py`, regenerate with `python3 make_notebook.py`, and copy to gpu01.

### Notebook structure (simplified 2026-10-16 afternoon)
GPU only, TensorFlow only: FFT vs direct integral. The **Settings** cell at the top holds `DTYPE = "f64"` or `"f32"`
(applies to every calculation), `N_LIST` for the timing scan, `N_PLOT` for the lineshape figure and `N_OUT_FIT` (30 000) for the cost-model plot. All output files
are suffixed with the dtype so an f64 and an f32 run coexist. The notebook asserts that a GPU is visible.
Cells: settings → GPU pick + TF import → physics/Voigt reference → `conv_fft` → `conv_direct` → `bench`/`run_config`
→ lineshape plot → N scan + tables → scan plot → cost model (n_out×N vs N log N) → bin-integrated vs point-sampled study → notes.
Original 10-config generator (numpy + TF-CPU + TF-GPU) is not kept in the repo; its results are in the tables below.

### Machine notes
- gpu01: Rocky 9.4, 48 cores, 376 GB RAM, 6× Tesla T4 (15 GB, compute 7.5, driver 595 / CUDA 13.2).
  GPUs are **shared** (other users hold 10–14 GB on each); the notebook picks the least-loaded GPU via
  `nvidia-smi` and sets memory growth. Timings on a shared GPU have a few-ms jitter at small N.
- Home `/users/kp14102` is NFS with a 50 GB soft quota and was **over quota** on 2026-10-16
  (`mkdir` failed). Purged `~/.cache/pip` (1.6 GB) and `~/.cache/uv` (5.9 GB). `~/.cache/vscode-cpptools`
  (5 GB) was left. The venv therefore lives on `/storage` (not backed up; `setup.sh` recreates it).

## Physics / numerics conventions
- M = 3.0969 GeV, Γ = 92.6 keV (PDG). Constant-width RBW in q² is a Lorentzian in x = q² − M² with
  half-width γ = MΓ = 2.87×10⁻⁴ GeV²; σ/γ ≈ 139.
- Exact reference: Lorentzian ⊗ Gaussian = Voigt profile (`scipy.special.voigt_profile`).
- Grid: N bin centres over x ∈ [−1, 1] GeV². Direct integral evaluated at 1001 points in |x| ≤ 0.4 GeV².
- Everything is computed in the centred variable x to keep float32 usable (q² ≈ 9.6 would eat the mantissa).
- FFT: zero-padded to 2N (linear convolution), kernel stored with zero lag at index 0.
- Direct: dense N_out × N Gaussian kernel matrix, built and multiplied in ≤ 256 MiB chunks.
- Accuracy metric: max |method − Voigt| / peak over the output window.

## Outcomes

### Current notebook (GPU only), runs of 2026-10-16 afternoon, GPU 1 of gpu01 — median ms / max rel err

| method / dtype |   N=2^12 |   N=2^14 |   N=2^16 |   N=2^18 |   N=2^20 |
|:---------------|---------:|---------:|---------:|---------:|---------:|
| FFT f64        | 1.16 / 0.049 | 1.01 / 7.8e-7 | 0.79 / 9e-16 | 1.65 / 7e-16 | 5.5 / 7e-16 |
| Direct f64     | 4.6 / 0.049  | 10.4 / 7.8e-7 | 33.8 / 5e-16 | 143 / 5e-16  | 575 / 5e-16 |
| FFT f32        | 0.79 / 0.049 | 0.78 / 1.2e-6 | 0.79 / 3.9e-7 | 0.76 / 5.8e-7 | 1.75 / 6.8e-7 |
| Direct f32     | 1.8 / 0.049  | 4.3 / 8.9e-7  | 13.4 / 1.9e-7 | 49 / 2.1e-7  | 582* / 2.8e-6 |

\* 582 ms is a shared-node outlier; the earlier run gave 209 ms. Bin-integrated table is unchanged from below
(f32 floors at ~2–5×10⁻⁷ instead of 2.4×10⁻⁸ at N = 2¹⁶).

### Original 10-config run (2026-10-16 morning, TF 2.21, GPU 2 of gpu01) — legacy, CPU/numpy no longer in notebook

### Median wall time [ms] (compute only, inputs pre-placed on device; shared node, run-to-run jitter up to ~2×)

| label                    |   N=2^12 |   N=2^14 |   N=2^16 |   N=2^18 |   N=2^20 |
|:-------------------------|---------:|---------:|---------:|---------:|---------:|
| FFT / tf GPU / f32       |    0.811 |    1.285 |    0.801 |    0.847 |    1.671 |
| FFT / tf GPU / f64       |    1.25  |    1.06  |    1.193 |    3.302 |   12.715 |
| FFT / tf CPU / f32       |    0.845 |    3.312 |    3.001 |    7.207 |   46.711 |
| FFT / tf CPU / f64       |    1.176 |    7.804 |    3.969 |   12.313 |  132.402 |
| FFT / numpy CPU / f64    |    0.295 |    3     |    7.063 |   32.63  |  383.721 |
| DIRECT / tf GPU / f32    |    2.547 |    4.112 |   13.435 |   52.747 |  208.632 |
| DIRECT / tf GPU / f64    |    5.03  |   14.223 |   34.809 |  150.926 |  576.41  |
| DIRECT / tf CPU / f32    |    4.733 |   14.83  |   52.517 |  257.112 |  851.282 |
| DIRECT / tf CPU / f64    |    6.841 |   28.417 |  109.067 |  490.264 | 2647     |
| DIRECT / numpy CPU / f64 |   67.615 |  193.496 |  848.642 | 4211.75  |  –       |

### max |method − Voigt| / peak

| label                    |   N=2^12 |   N=2^14 |   N=2^16 |   N=2^18 |    N=2^20 |
|:-------------------------|---------:|---------:|---------:|---------:|----------:|
| FFT / tf GPU / f64       |    0.049 |  7.8e-07 |  9e-16   |  7.2e-16 |   7.2e-16 |
| FFT / tf GPU / f32       |    0.049 |  1.2e-06 |  3.9e-07 |  5.8e-07 |   6.8e-07 |
| DIRECT / tf GPU / f64    |    0.049 |  7.8e-07 |  5.4e-16 |  5.4e-16 |   5.4e-16 |
| DIRECT / tf GPU / f32    |    0.049 |  8.9e-07 |  1.9e-07 |  2.1e-07 |   2.8e-06 |
| DIRECT / tf CPU / f32    |    0.049 |  1.2e-06 |  1.1e-06 |  4.4e-06 |   2e-05   |
| (all other f64 variants) |    0.049 |  7.8e-07 |  ≤4e-15  |  ≤1e-14  |   ≤4e-14  |

### Bin-integrated vs point-sampled RBW (TF GPU f64), max rel err

|                       |   N=2^8 |   N=2^10 |   N=2^12 |   N=2^14 |   N=2^16 |
|:----------------------|--------:|---------:|---------:|---------:|---------:|
| point-sampled (either)|  0.89   |  0.57    |  0.049   |  7.8e-07 |  ~1e-15  |
| bin-integrated (either)|  0.0043 |  0.00021 |  7.1e-06 |  3.9e-07 |  2.4e-08 |

### Conclusions
1. **FFT wins by 1–2 orders of magnitude at every N.** At the "safe" grid N = 2¹⁸ (dx ≈ 0.1γ):
   FFT TF-GPU f64 ≈ 3 ms vs direct TF-GPU f64 ≈ 150 ms (f32: 0.8 ms vs 53 ms). The direct method
   scales as N_out × N (here N_out = 1001); the FFT as N log N and sits at the launch-overhead floor
   (~1 ms, dominated by TF eager dispatch + host sync) up to N ≈ 2¹⁸ on the GPU.
2. **Both methods converge to the exact Voigt profile** (f64: ~10⁻¹⁵) once the grid resolves γ.
   Point-sampled RBW needs dx ≲ 0.1γ, i.e. N ≥ 2¹⁶ over ±1 GeV² for machine precision, N = 2¹⁴ for 10⁻⁶;
   N = 2¹² (dx ≈ 1.7γ) is 5 % wrong. The bottleneck is the J/ψ width, not the 0.04 GeV² resolution.
3. **Bin-integrated RBW** (analytic arctan bin average) makes coarse grids usable: N = 2¹² gives 10⁻⁵,
   N = 2¹⁴ gives 4×10⁻⁷, at identical cost. Use this if N must be small (e.g. many PDFs per fit step).
4. **float32 on the T4**: FFT f32 accuracy is ~3–7×10⁻⁷ of peak and 3× faster than f64 at large N;
   direct f32 degrades with N (up to 2×10⁻⁵ on CPU at 2²⁰ from summation error). T4 f64 is 1/32 rate,
   so f64 FFT on GPU is ~10× faster than TF-CPU f64 at N = 2²⁰ (13 ms vs 130 ms), f32 ~30×.
5. CPU baselines: numpy pocketfft is fastest for tiny N (< 2¹⁴) because TF eager dispatch costs ~1 ms;
   TF-CPU (48 threads) beats numpy above that. numpy direct is 4.2 s at 2¹⁸ and was skipped at 2²⁰.
6. Domain truncation of the Lorentzian tails outside ±1 GeV² (~10⁻⁴ of the norm) is invisible in the
   ±0.4 GeV² window because it sits > 15σ away; it would matter if the output window reached the grid edge.

## Physics conclusions (what matters for a B → K*μμ / J/ψ fit)

1. **The J/ψ width, not the detector resolution, sets the grid.** γ = MΓ = 2.87×10⁻⁴ GeV² must be resolved:
   dx ≲ 0.1γ (N ≥ 2¹⁶ over ±1 GeV²) for machine precision, dx ≈ 0.4γ (N = 2¹⁴) for 10⁻⁶, dx ≈ 1.7γ (N = 2¹²)
   is 5 % wrong at the peak. σ = 0.04 GeV² is 139γ and is trivially resolved on any of these grids.
2. **Use the bin-integrated RBW** (analytic arctan average over each bin) whenever N must be small: it is exact
   in the limit dx ≫ γ as well as dx ≪ γ, so N = 2¹² gives 10⁻⁵ and N = 2¹⁴ gives 4×10⁻⁷ at zero extra cost.
   Point-sampling a Lorentzian whose half-width is below the bin size is the dominant numerical error.
3. **The smeared lineshape is a Voigt profile**, i.e. a Gaussian core of width σ with Lorentzian tails that take
   over beyond ~4σ (≈ 0.16 GeV² from the pole). A pure Gaussian underestimates the pdf by orders of magnitude
   there; the tails carry the γ/(π x²) dependence and matter for any sideband or interference study.
4. **Domain truncation**: the ±1 GeV² grid loses ~γ/(π·1 GeV²) ≈ 10⁻⁴ of the Lorentzian normalisation, which
   lies > 15σ from the ±0.4 GeV² window and is invisible there. Extend the grid if the fit window grows.
5. **Precision**: f64 reproduces the Voigt profile to ~10⁻¹⁵ of peak; f32 floors at ~5×10⁻⁷ (FFT) and degrades
   with N for the direct sum (2×10⁻⁶ at 2²⁰). f32 is adequate for a likelihood but not for validating the method.
6. **Evaluation points** (discussion 2026-10-16): the FFT returns values on the N-point grid, the direct integral
   at arbitrary user-chosen points (here 1001 in ±0.4 GeV²). Nothing is interpolated; each is compared to the
   Voigt profile at its own points. In a fit the FFT result must be interpolated onto event q² values (linear
   interpolation error ~10⁻¹¹ of peak at N = 2¹⁸), whereas the direct sum can be evaluated at the events directly.
7. **Cost model**: direct ∝ n_out × N, FFT ∝ N log₂N, ratio n_out / log₂N. For n_out = 30 000 events the direct
   sum is 1500–2500× more operations for N = 2²⁰ … 2¹²; it never catches up. Measured gap at n_out = 1001 is
   30–100× because the FFT constant (2N padding, three transforms) is larger. Recommendation: FFT on an
   N = 2¹⁶–2¹⁸ grid with bin-integrated RBW, then interpolate.

## Possible next steps
- Interpolate the FFT output onto `X_OUT` inside the timed call so both methods report on identical points.
- Mass-dependent width / Blatt–Weisskopf RBW (no analytic Voigt reference; compare FFT vs direct only).
- ψ(2S) or the full B → K*μμ q² spectrum with interference; q²-dependent σ (then FFT needs binning in σ).
- Benchmark inside a `tf.function` fit loop (gradients through `tf.signal.rfft`) rather than eager calls.
