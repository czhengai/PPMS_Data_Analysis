###
# COPYRIGHT @ PETER CE ZHENG
# 03 OCT 2025
###

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

def sg_smooth(y, window_length, polyorder):
    #優化 叠加法
    window_length = int(window_length)
    polyorder = int(polyorder)
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 1
        if window_length % 2 == 0:
            window_length += 1
    return savgol_filter(y, window_length=window_length, polyorder=polyorder, mode="interp")

def fft_1overB(B, dR, min_B, max_B, oversample=1.0):
    # SG法
    b1, b2 = (min_B, max_B) if min_B <= max_B else (max_B, min_B)
    mask = (B >= b1) & (B <= b2)
    if mask.sum() < 8:
        raise ValueError("Too few data points in selected B range.")
    B_sel = B[mask]
    dR_sel = dR[mask]
    x = 1.0 / B_sel
    order = np.argsort(x)
    x = x[order]
    y = dR_sel[order]
    dx_raw = np.diff(x)
    dx = np.median(dx_raw)
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError("Invalid step size in 1/B.")
    dx = dx / float(oversample)
    x_min, x_max = x.min(), x.max()
    n_est = int(np.floor((x_max - x_min) / dx)) + 1
    if n_est < 8:
        raise ValueError("Too few points after interpolation.")
    x_grid = np.linspace(x_min, x_max, n_est)
    y_grid = np.interp(x_grid, x, y)
    y_detrend = y_grid - np.mean(y_grid)
    window = np.hanning(len(y_detrend))
    y_win = y_detrend * window
    Y = np.fft.rfft(y_win)
    F = np.fft.rfftfreq(len(x_grid), d=(x_grid[1] - x_grid[0]))
    amp = np.abs(Y) / (len(x_grid) * np.mean(window))
    return F, amp, x_grid, y_grid

def main():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not filepath:
        messagebox.showinfo("Info", "No file selected.")
        return
    try:
        try_delims = [',', '\t', ';', ' ']
        df = None
        for d in try_delims:
            try:
                df = pd.read_csv(filepath, header=None, sep=d, engine='python')
                if df.shape[1] >= 2:
                    break
            except Exception:
                continue
        if df is None or df.shape[1] < 2:
            raise ValueError("Invalid file format.")
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df.iloc[:, :2]
        df.columns = ['B', 'R']
        df = df.sort_values('B').reset_index(drop=True)
        B = df['B'].to_numpy()
        R = df['R'].to_numpy()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file:\n{e}")
        return
    window_length = simpledialog.askinteger("SG Parameters", "window_length (odd, e.g. 101~401):", initialvalue=151, minvalue=5)
    if window_length is None:
        return
    polyorder = simpledialog.askinteger("SG Parameters", "polyorder (e.g. 2~4):", initialvalue=3, minvalue=1, maxvalue=9)
    if polyorder is None:
        return
    try:
        R_bg = sg_smooth(R, window_length, polyorder)
    except Exception as e:
        messagebox.showerror("Error", f"SG smoothing failed:\n{e}")
        return
    dR = R - R_bg
    B_min_all = float(np.nanmin(B))
    B_max_all = float(np.nanmax(B))
    b_low = simpledialog.askfloat("B Range", f"Lower bound of B (range {B_min_all:.4g} ~ {B_max_all:.4g}):", initialvalue=B_min_all)
    if b_low is None:
        return
    b_high = simpledialog.askfloat("B Range", f"Upper bound of B (range {B_min_all:.4g} ~ {B_max_all:.4g}):", initialvalue=B_max_all)
    if b_high is None:
        return
    oversample = 1.0
    try:
        F, Amp, x_grid, dR_grid = fft_1overB(B, dR, b_low, b_high, oversample=oversample)
    except Exception as e:
        messagebox.showerror("Error", f"FFT failed:\n{e}")
        return
    outdir = Path(filepath).parent
    stem = Path(filepath).stem
    plt.figure(figsize=(7,4.5), dpi=140)
    plt.plot(B, R, label="Raw R(B)", linewidth=1)
    plt.plot(B, R_bg, label=f"SG Background (win={window_length}, poly={polyorder})", linewidth=1.5)
    plt.xlabel("B")
    plt.ylabel("R")
    plt.title("Raw data and SG background")
    plt.legend()
    plt.tight_layout()
    fig1_path = outdir / f"{stem}_sg_smooth.png"
    plt.savefig(fig1_path)
    plt.figure(figsize=(7,4.5), dpi=140)
    plt.plot(B, dR, linewidth=1)
    plt.xlabel("B")
    plt.ylabel("ΔR")
    plt.title("Oscillation after background removal (SG)")
    plt.tight_layout()
    fig2_path = outdir / f"{stem}_sg_oscillation.png"
    plt.savefig(fig2_path)
    plt.figure(figsize=(7,4.5), dpi=140)
    plt.plot(F, Amp, linewidth=1)
    plt.xlabel("Frequency (Oe)")
    plt.ylabel("Amplitude (arb. u.)")
    lo, hi = (b_low, b_high) if b_low <= b_high else (b_high, b_low)
    plt.title(f"FFT of ΔR vs 1/B, B ∈ [{lo:.3g}, {hi:.3g}]")
    plt.tight_layout()
    fig3_path = outdir / f"{stem}_fft.png"
    plt.savefig(fig3_path)
    ones_over_B = np.where(B != 0, 1.0 / B, np.nan)
    default_txt1 = filedialog.asksaveasfilename(
        title="Save processed TXT (R & ΔR)",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialdir=str(outdir),
        initialfile=f"{stem}_processed.txt"
    )
    if default_txt1:
        arr1 = np.column_stack([B, R, R_bg, dR, ones_over_B])
        header1 = "B\tR\tR_bg\tDeltaR\tinvB"
        np.savetxt(default_txt1, arr1, delimiter="\t", header=header1, comments="", fmt="%.10g")
    default_txt2 = filedialog.asksaveasfilename(
        title="Save FFT TXT",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialdir=str(outdir),
        initialfile=f"{stem}_fft.txt"
    )
    if default_txt2:
        header2 = f"FFT of DeltaR (B range [{lo:.6g}, {hi:.6g}] Oe)\nFrequency(Oe)\tAmplitude"
        np.savetxt(default_txt2, np.column_stack([F, Amp]), delimiter="\t", header=header2, comments="", fmt="%.10g")
    plt.show()
    messagebox.showinfo(
        "Done",
        "Generated figures and TXT files."
    )

if __name__ == "__main__":
    main()
