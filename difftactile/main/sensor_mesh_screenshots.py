"""Three screenshots of the ViTacTip tetrahedral mesh, as Gmsh draws it.

Renders the sensor mesh written by ``generate_vitactip_mesh_gmsh``
(``files.gmsh_debug_msh``) from the +x side - level with the sensor (line of
sight along -x), from 45 degrees above and from 45 degrees below - and saves
each view as a small WebP for the project page (``docs/images/sensor_mesh/``).
The sensor's base lies in the xy plane and its dome points along +z.  Nothing
here touches the mesh pickles the simulator reads - this is a pure viewer.

Gmsh can only rasterise through its FLTK front end, so a Gmsh window opens
briefly on the display (there is no off-screen path); it is closed again as
soon as the three frames are written.  Run through ``docker/sensor_mesh_screenshots.sh``.

``VIEWS`` records the Gmsh Euler angles that produce each view; they were
verified against Gmsh's own axis triad (which stays in every screenshot
precisely so the reader can check too).
"""

import os

import gmsh
from PIL import Image

from difftactile.main.constants import *
from difftactile.main.paths import repo_path

# view name -> (General.RotationX, RotationY, RotationZ in degrees; caption).
# Gmsh applies these only with General.Trackball = 0, as RotationZ (a spin
# about the sensor's own axis) followed by RotationX (a tilt about the screen's
# horizontal axis).  The default view (0, 0, 0) has x right, y up, z towards the
# viewer, i.e. line of sight -z; RotationX = -90 stands the sensor up (z up), so
# -45 / -135 raise / lower the camera by 45 degrees from that side view.
VIEWS = {
    "side":       (-90,  0, -90, "line of sight along -x (camera on the +x side, level with the sensor): y right, z up"),
    "from_above": (-45,  0, -90, "camera on the +x side raised to look DOWN at the sensor at 45 degrees"),
    "from_below": (-135, 0, -90, "camera on the +x side lowered to look UP at the sensor at 45 degrees"),
}
# Rendered frame size in pixels (square). The model is scaled up a little so
# it fills the frame instead of cropping afterwards, which would either lose
# Gmsh's axis triad or keep the whitespace between it and the model.
FRAME_PX = int(os.environ.get("DIFFTACTILE_SENSOR_MESH_FRAME_PX", "800"))
MODEL_SCALE = 1.25


def screenshot_dir():
    """Where the WebP screenshots go (default: the project page's image folder)."""
    return os.environ.get(
        "DIFFTACTILE_SENSOR_MESH_SCREENSHOT_DIR",
        repo_path("docs/images/sensor_mesh"),
    )


def configure_gmsh_view():
    """Static Gmsh display options shared by all six views: shaded surface
    faces with mesh edges, entity colouring, no ruler axes, the small
    orientation triad kept, and Euler-angle (non-trackball) rotation."""
    gmsh.option.setNumber("General.Terminal", 1)
    # GraphicsWidth is the whole window including the side menu, so add the
    # menu's width to get a FRAME_PX-wide drawing area (and hence screenshot).
    menu_width = gmsh.option.getNumber("General.MenuWidth")
    gmsh.option.setNumber("General.GraphicsWidth", FRAME_PX + menu_width)
    gmsh.option.setNumber("General.GraphicsHeight", FRAME_PX)
    gmsh.option.setNumber("General.Trackball", 0)
    gmsh.option.setNumber("General.Axes", 0)
    gmsh.option.setNumber("General.SmallAxes", 1)
    for axis in "XYZ":
        gmsh.option.setNumber(f"General.Scale{axis}", MODEL_SCALE)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.VolumeEdges", 0)


def render_views(msh_path, png_dir):
    """Open ``msh_path`` in Gmsh, write one PNG per entry of ``VIEWS`` into
    ``png_dir`` and return {view name: png path}."""
    os.makedirs(png_dir, exist_ok=True)
    gmsh.initialize()
    configure_gmsh_view()
    gmsh.open(msh_path)
    gmsh.fltk.initialize()  # opens the Gmsh window - the only way Gmsh rasterises
    pngs = {}
    for name, (rx, ry, rz, _caption) in VIEWS.items():
        gmsh.option.setNumber("General.RotationX", rx)
        gmsh.option.setNumber("General.RotationY", ry)
        gmsh.option.setNumber("General.RotationZ", rz)
        gmsh.graphics.draw()
        gmsh.fltk.update()
        path = os.path.join(png_dir, f"vitactip_mesh_{name}.png")
        gmsh.write(path)
        pngs[name] = path
    gmsh.fltk.finalize()
    gmsh.finalize()
    return pngs


def compress_to_webp(pngs, out_dir):
    """Re-encode each PNG as a LOSSLESS WebP in ``out_dir``. Flat colours and
    sharp mesh edges compress far better losslessly than lossily (measured:
    21 kB lossless vs 33 kB optimised PNG vs 110-145 kB lossy WebP), so this
    is both the smallest and the exact image."""
    os.makedirs(out_dir, exist_ok=True)
    for name, png in pngs.items():
        webp = os.path.join(out_dir, f"vitactip_mesh_{name}.webp")
        Image.open(png).convert("RGB").save(webp, "WEBP", lossless=True, method=6)
        print(f"{webp}: {os.path.getsize(webp) / 1024:.1f} kB  ({VIEWS[name][3]})")


def main():
    msh_path = repo_path(SYSTEM_PARAMS.files.gmsh_debug_msh)
    if not os.path.exists(msh_path):
        raise FileNotFoundError(
            f"{msh_path} not found - run script_generate_vitactip_mesh_gmsh first"
        )
    png_dir = repo_path("difftactile/output/sensor_mesh_screenshots")
    pngs = render_views(msh_path, png_dir)
    compress_to_webp(pngs, screenshot_dir())


if __name__ == "__main__":
    main()
