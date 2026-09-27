import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

st.set_page_config(page_title="Local Mesh Edit", layout="wide")
st.title("Local mesh editing from a 3D curve")
st.caption("A differentiable geometry baseline: edit a curve, constrain the untouched region, and inspect the resulting mesh.")

@st.cache_data
def make_grid(n=17):
    coords = np.linspace(-1, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    zz = 0.08 * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    faces = []
    edges = []
    for r in range(n):
        for c in range(n):
            a = r * n + c
            if c < n - 1:
                edges.append((a, a + 1))
            if r < n - 1:
                edges.append((a, a + n))
            if r < n - 1 and c < n - 1:
                faces.extend([(a, a + 1, a + n), (a + 1, a + n + 1, a + n)])
    return vertices, np.asarray(faces), np.asarray(edges)


def edit_mesh(original, edges, curve_ids, fixed_ids, height, smooth_weight):
    base = torch.tensor(original)
    e = torch.tensor(edges, dtype=torch.long)
    target = base[curve_ids].clone()
    x = target[:, 0]
    target[:, 2] += height * (1 - x.square())  # lift center more than endpoints
    offset = torch.nn.Parameter(torch.zeros_like(base))
    opt = torch.optim.Adam([offset], lr=0.04)

    for _ in range(180):
        edited = base + offset
        curve_loss = F.mse_loss(edited[curve_ids], target)
        fixed_loss = F.mse_loss(edited[fixed_ids], base[fixed_ids])
        smooth_loss = ((offset[e[:, 0]] - offset[e[:, 1]]) ** 2).mean()
        loss = 40 * curve_loss + 40 * fixed_loss + smooth_weight * smooth_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        result = (base + offset).numpy()
        metrics = {
            "curve error": float(torch.linalg.vector_norm((base + offset)[curve_ids] - target, dim=1).mean()),
            "fixed drift": float(torch.linalg.vector_norm((base + offset)[fixed_ids] - base[fixed_ids], dim=1).mean()),
            "max displacement": float(torch.linalg.vector_norm(offset, dim=1).max()),
        }
    return result, target.numpy(), metrics


def draw_mesh(vertices, faces, curve_ids, title, target=None):
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#63a6d8", opacity=0.82, flatshading=False, name="mesh"
    ))
    fig.add_trace(go.Scatter3d(
        x=vertices[curve_ids, 0], y=vertices[curve_ids, 1], z=vertices[curve_ids, 2],
        mode="lines+markers", line=dict(color="#e1553e", width=8),
        marker=dict(size=3), name="surface curve"
    ))
    if target is not None:
        fig.add_trace(go.Scatter3d(
            x=target[:, 0], y=target[:, 1], z=target[:, 2],
            mode="lines", line=dict(color="#273b8a", width=5, dash="dash"),
            name="target curve"
        ))
    fig.update_layout(
        title=title, height=530, margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(aspectmode="cube", xaxis=dict(range=[-1.2, 1.2]),
                   yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-0.7, 0.7]))
    )
    return fig

height = st.slider("Curve lift", 0.0, 0.6, 0.3, 0.05)
smooth_weight = st.slider("Smoothness weight", 0.1, 20.0, 3.0, 0.1)
original, faces, edges = make_grid()
n = 17
curve_ids = np.arange((n // 2) * n, (n // 2 + 1) * n)
fixed_ids = np.array([r * n + c for r in range(n) for c in range(n)
                      if (r == 0 or r == n - 1 or c == 0 or c == n - 1)
                      and (r * n + c) not in set(curve_ids)])
edited, target, metrics = edit_mesh(original, edges, curve_ids, fixed_ids, height, smooth_weight)
left, right = st.columns(2)
with left:
    st.plotly_chart(draw_mesh(original, faces, curve_ids, "Original"), use_container_width=True)
with right:
    st.plotly_chart(draw_mesh(edited, faces, curve_ids, "Edited", target), use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.metric("Mean curve error", f"{metrics['curve error']:.4f}")
c2.metric("Mean fixed drift", f"{metrics['fixed drift']:.4f}")
c3.metric("Max vertex displacement", f"{metrics['max displacement']:.4f}")
st.caption("Prototype only: each slider change optimizes this single mesh; it does not train a model that generalizes to new meshes.")
