# Interactive Contour Explorer

Explore **surface enrichment** (a.k.a. segregation strength) across two simulation parameters with an interactive contour heatmap and linked cross-sections. The tool reads your simulation summary CSV, interpolates onto a grid, and lets you scrub through **Volume fraction PS (%)** and **Size ratio (−)** to inspect trends.

---

## Features

- **CSV → interactive plots**: loads `summary_all.csv` and parses X, Y, Z from specific columns.
- **Grid interpolation** with `scipy.interpolate.griddata` and a refinement step that **overwrites** interpolated nodes with real measurements at the nearest grid points for fidelity.
- **Linked views**:
  - Main **contour heatmap** (Z over X–Y) with colorbar (“Surface enrichment / −”).
  - **Z vs Y** at selected X (red curve).
  - **Z vs X** at selected Y (blue curve).
- **Two sliders** to scrub X and Y (ranges X: 1–30 step 1, Y: 1–7 step 0.2) updating the cross-sections in real time.

---

## Quick start

### 1) Requirements
- Python 3.9+
- Packages: `numpy`, `matplotlib`, `scipy`, `mpld3` (imported; not strictly used in the current figure)

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install numpy matplotlib scipy mpld3
```


### 2) Project structure

├─ interactive_tool.py
├─ interactive_tool.ipynb
├─ requirements.txt
├─ README.md
└─ summary_all.csv

### 3) Run it
```bash
python interactive_tool.py
```

## Documentation

The simulations and theory behind the results presented in this tool can be found in the following publication:

tbd.