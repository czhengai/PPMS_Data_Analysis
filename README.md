# PPMS_Data_Analysis
This is a folder containing the program which dealt with the data 
# Quantum Oscillation Data Analysis (SdH)

This Python script provides a GUI-based workflow to process magnetoresistance data and extract quantum oscillation signals (Shubnikov–de Haas).  
It allows you to load a CSV file, remove background using Savitzky–Golay smoothing, select a B-field range, and perform FFT on ΔR vs 1/B to obtain oscillation frequencies.

---

## Features
- Select CSV file via dialog (two columns: **B** [magnetic field] and **R** [resistance]).
- Background removal using **Savitzky–Golay (SG) filter**.
- Interactive input for:
  - SG filter parameters (`window_length`, `polyorder`).
  - B-field range (`lower bound`, `upper bound`).
  - FFT settings: resampling points, zero-padding factor.
- Outputs:
  - ΔR (oscillatory component after background subtraction).
  - FFT spectrum of ΔR vs 1/B.
- Saves figures automatically:
  - Raw data + background (`*_sg_smooth.png`)
  - ΔR vs B (`*_sg_oscillation.png`)
  - FFT spectrum (`*_fft.png`)
- Exports TXT files via "Save As" dialog:
  - Processed data: `B`, `R_raw`, `R_bg`, `ΔR`, `1/B`.
  - FFT data: `Frequency`, `Amplitude`.

---

## Requirements
- Python 3.8+
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `tkinter` (standard in Python)

Install missing packages with:
```bash
pip install numpy pandas matplotlib scipy
