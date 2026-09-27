# Local Mesh Edit — geometry baseline

An interactive baseline for curve-guided local mesh deformation. The red line shows the current curve; the dashed line shows the requested curve. An optimizer adjusts vertex positions while penalizing curve error, boundary drift, and rough displacement.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Drag the sliders and rotate the 3D views.

## Scope

This is per-instance differentiable geometry optimization, not a trained neural network. The flat grid and fixed curve indices keep topology simple. Next steps: load a real mesh, map a user stroke to surface vertices, choose a local region by mesh geodesic distance, evaluate preservation and latency, then train a model to predict displacement and compare it to this baseline.
