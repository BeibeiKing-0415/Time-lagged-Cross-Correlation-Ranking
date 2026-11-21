# Time-lagged cross-correlation (TCC) ranking

Run the synthetic example:

    python -m pip install -r requirements.txt
    cd code
    python synthetic_example.py

The example selects the three target groups at angular frequencies 1.40,
2.20 and 2.90, with a target-subfield error of approximately 2.15%.
It saves synthetic_result.png.

Processed data for DMD/POD:

    File            State shape       dt
    synthetic.npz   (2688, 1200)       0.05
    cylinder.npz    (14600, 310)       0.25
    cavity.npz      (15708, 3148)      0.00025

Each NPZ includes state, time, dt, weight and observable. State columns are
snapshots. Remove row means before decomposition; cavity is already centered.
For weighted DMD/POD, use np.sqrt(weight)[:, None] * state.

Data sources:

- Cylinder: https://doi.org/10.57745/6LG9AS
- Cavity: https://doi.org/10.2514/1.J066928
