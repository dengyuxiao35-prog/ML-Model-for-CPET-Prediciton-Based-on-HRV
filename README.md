# AI-CPET Assessment System (Streamlit)

This app predicts **Ventilatory Threshold stages (0/1/2)** over time using a trained Random Forest model (`rf_vts_model.pkl`) and a matching scaler (`scaler.pkl`), then extracts **VT1/VT2** from the predicted stage transitions and reports a **VO2peak** estimate (linear formula).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input data requirements

Upload **5s-interpolated** time-series data (CSV/Excel). Required signals (aliases supported):

- Time: `Time` or `TIME`
- `HR`
- Respiratory rate: `RR` (also accepts `RF`)
- `RMSSD`
- DFA α1: `DFA_alpha1` or `DFAα1`
- `MeanRRi`
- `SD1`, `SD2`
- Spectral powers: `HF_power`/`HF power`, `VLF_power`/`VLF power`

Participant demographics are provided in the sidebar (Gender/Age/Height/Weight/Resting HR).

## Example data + expected results

Files:
- De-identified input: `examples/example_101_input.csv`
- De-identified expected output (TrueStage + PredStage): `examples/example_101_output_cpet_results.csv`
- De-identified expected summary: `examples/example_101_output_summary.json`

To reproduce the expected results, set the sidebar as:
- Gender: **Male**
- Age: **25**
- Height: **176.3 cm**
- Weight: **72.2 kg**
- Resting HR: **61 bpm**

Notes:
- `example_101_input.csv` is **de-identified** (identifiers removed).
