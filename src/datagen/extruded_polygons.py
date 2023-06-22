import pyvista
import polygenerator
import shutil
import os
import numpy as np
import random
import time

OUTPUT_DIR = 'extruded_polygons'
NUM_IMAGES = 10000


dirpath = os.path.join(OUTPUT_DIR)
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

os.mkdir(OUTPUT_DIR)


def render_polygon(N, i):
    rng = np.random.default_rng()
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = rng.uniform(0.5, 1.5, N)
    coords = np.array([np.cos(angles), np.sin(angles)]) * radii
    points_2d = coords.T 
    points_3d = np.pad(points_2d, [(0, 0), (0, 1)])
    face = [N + 1] + list(range(N)) + [0]
    poly = pyvista.PolyData(points_3d, faces=face)
    for j in range(0, 5):
        pl = pyvista.Plotter(off_screen=True, window_size=[224,224])
        mesh = poly.extrude([0, 0, 1], capping=True)
        pl.add_mesh(mesh, smooth_shading=False)
        pl.camera.azimuth = random.random() * 180
        pl.camera.roll = random.random() * 180
        pl.camera.elevation = random.random() * 180
        pl.screenshot(os.path.join(OUTPUT_DIR, 'poly-{i}-{j}.png'.format(i=i, j=j)))

for i in range(0,5):
    render_polygon(N=random.choice(range(3,10)), i=i)