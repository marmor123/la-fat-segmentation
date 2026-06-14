# PyVista + Panel replaces matplotlib for the QA Dashboard

The static `dashboard.html` rendered 2D slices and a 4-angle matplotlib 3D GIF. Colleagues couldn't rotate, zoom, toggle individual masks, or inspect intermediate pipeline steps — only the final LA Fat mesh. We replaced it with an interactive browser application using PyVista (VTK/WebGL) and Panel.

PyVista renders medical meshes at full resolution with free rotation and per-mask toggles, where Plotly Mesh3d would stutter on the triangle counts produced by marching cubes. The three key pipeline steps (Anchors, Partition, Final LA Fat) each get their own viewport in a vertical scroll layout, with a collapsible sidebar for patient selection, key numbers, and quality flags.

## Considered Options

- **Plotly Mesh3d** — lighter dependency, but struggles with >10k triangles per surface; toggling is fragile. Rejected because medical marching-cubes output exceeds this threshold.
- **Dash + VTK** — similar capability, but Dash's callback model adds boilerplate. Panel's reactive architecture is simpler for a single-user local app.
- **Keep matplotlib** — no interactivity, no per-mask toggles, no intermediate step inspection. Rejected because it doesn't meet the "interactive 3D views of every step" requirement.

## Consequences

- PyVista pulls VTK as a dependency (~200MB), but it's install-once and the app remains local.
- Meshes are pre-computed during the pipeline and saved as `.ply` files, so the dashboard loads instantly without re-running marching cubes.
- The old matplotlib 3D code in `qa_dashboard.py` is removed; 2D slices and numeric summaries remain.
