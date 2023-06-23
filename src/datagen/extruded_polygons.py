import pyvista
from PIL import Image
import shutil
import os
import numpy as np
import random

OUTPUT_DIR = 'extruded_polygons'
NUM_IMAGES = 10000


dirpath = os.path.join(OUTPUT_DIR)
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

os.mkdir(OUTPUT_DIR)

for i in range(3,10):
    os.mkdir(os.path.join(OUTPUT_DIR,str(i)))

def render_polygon(pl, N, i):
    rng = np.random.default_rng()
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = rng.uniform(0.5, 1.5, N)
    coords = np.array([np.cos(angles), np.sin(angles)]) * radii
    points_2d = coords.T 
    points_3d = np.pad(points_2d, [(0, 0), (0, 1)])
    face = [N + 1] + list(range(N)) + [0]
    poly = pyvista.PolyData(points_3d, faces=face)
    imgs = []
    for j in range(0, 5):
        mesh = poly.extrude([0, 0, 1], capping=True)
        actor = pl.add_mesh(mesh, smooth_shading=False)
        pl.camera.azimuth = random.random() * 180
        pl.camera.roll = random.random() * 180
        pl.camera.elevation = random.random() * 180
        imgs.append(pl.screenshot(return_img=True))
        pl.remove_actor(actor)
    print(np.array(imgs))
    final = np.concatenate(np.array(imgs),1)
    im = Image.fromarray(final)
    path = os.path.join(os.path.join(OUTPUT_DIR, str(N)), "example-{i}.png".format(i=i))
    print(path)
    im.save(path)

pl = pyvista.Plotter(off_screen=True, window_size=[224,224])
pl.set_background("red")
for i in range(0,10000):
    render_polygon(pl, N=random.choice(range(3,10)), i=i)