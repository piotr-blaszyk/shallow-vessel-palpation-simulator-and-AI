"""Central policy for anything that could block a script on user input.

Scripts in this project used to stop dead on `plt.show()`, `cv2.waitKey(0)`,
`input()`, `gmsh.fltk.run()` and tkinter's `mainloop()`, each of which waits for
somebody to click the window's close button. That is fine at a desk and fatal in
a container, over SSH, in CI, or in any batch run of the pipeline.

The policy implemented here is: **nothing blocks by default.** Every figure and
every visualisation is written to disk regardless, so the artifacts in
`difftactile/output/` (and the other configured output folders) are the real
deliverable; the on-screen window is only ever a convenience. A user who wants
the interactive browsers back opts in explicitly:

    DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_visualise

Environment variables:

    DIFFTACTILE_INTERACTIVE=1   Restore blocking behaviour: `plt.show()` waits,
                                `wait_key(0)` waits for a real key press, the
                                Gmsh viewer and tkinter GUIs run their event
                                loops, and `prompt()` reads stdin.
    DIFFTACTILE_HEADLESS=1      Stronger still: do not even create windows.
                                Implies non-interactive.

With neither set, windows are still drawn (so you can watch a run go past) but
they are never waited on: `show_plots()` returns False, `wait_key()` returns
"no key pressed" after a short delay, and the viewer loops advance on their own
instead of asking for a keystroke.

Note that this module is the policy for the project's **OpenCV** windows. The
two interactive annotation viewers behind `docker/annotate_data_bare_metal.sh` were moved
to Qt (`difftactile/main/qt_viewer.py`) so they are native Wayland clients; they
still honour `is_interactive()` for the "should this open at all" decision, but
their event loop, frame scaling and key handling live there. The frame-browser
loop and display-downscaling helpers that used to live here went with them - Qt
fits the view to the window itself, and its native presentation removed the need
for the double-present workaround the OpenCV loop carried.
"""

import os


def is_headless():
    """True when no window should be created at all.

    A display is "reachable" if either X11 (`DISPLAY`) or Wayland
    (`WAYLAND_DISPLAY`) is available. Checking only `DISPLAY` used to be enough
    because every window in the project was an OpenCV/Xwayland one, but the Qt
    annotation viewers are native Wayland clients: on a Wayland session that has
    no Xwayland running, `DISPLAY` is unset while windows open perfectly well.
    Treating that as headless would refuse to start the very tools that work
    best there.
    """
    if os.environ.get("DIFFTACTILE_HEADLESS", "0") == "1":
        return True
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def is_interactive():
    """True when the user has explicitly opted in to blocking GUI windows.

    Requires both `DIFFTACTILE_INTERACTIVE=1` and an actual display; asking for
    interactivity on a machine with no X server is a mistake, not a request to
    hang.
    """
    if os.environ.get("DIFFTACTILE_INTERACTIVE", "0") != "1":
        return False
    return not is_headless()


def show_plots():
    """True when `plt.show()` may be called and allowed to block.

    Callers must `plt.savefig(...)` before consulting this: the saved file is
    the output, the window is optional.
    """
    return is_interactive()


def finish_plot(plt, save_path=None, **savefig_kwargs):
    """Standard end-of-figure handling: save, optionally show, always close.

    `save_path` is saved (creating parent directories) when given, passing any
    extra keyword arguments straight through to `plt.savefig`. The figure is
    closed either way so a long loop of figures does not leak them.
    """
    if save_path:
        parent = os.path.dirname(str(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, **savefig_kwargs)
    if show_plots():
        plt.show()
    plt.close()


def wait_key(cv2, delay=0):
    """Non-blocking replacement for `cv2.waitKey(delay)`.

    In interactive mode this is exactly `cv2.waitKey(delay)`. Otherwise a
    `delay` of 0 ("wait forever") is replaced by a brief 1 ms poll so the window
    still paints, and any other delay is honoured but capped, so a script that
    displays hundreds of frames does not spend minutes sleeping. The return
    value is whatever key was pressed, or -1 (& 0xFF -> 255) for none — which is
    what the existing `if key == ord('q')` branches already treat as "keep
    going".
    """
    if is_headless():
        # No window exists; there is nothing to pump and nothing to press.
        return -1
    if is_interactive():
        return _guard_gui(lambda: cv2.waitKey(delay), -1)
    capped = 1 if delay == 0 else min(delay, MAX_NONBLOCKING_DELAY_MS)
    return _guard_gui(lambda: cv2.waitKey(capped), -1)


# Longest a non-interactive run will pause on a single displayed frame.
MAX_NONBLOCKING_DELAY_MS = 30

# Set once the OpenCV build turns out to have no GUI support, so the warning is
# printed a single time rather than once per displayed frame.
_gui_unavailable = False


def _guard_gui(call, fallback):
    """Run an OpenCV GUI call, tolerating a build without GUI support.

    `opencv-python-headless` (what the container and many CI images install)
    raises cv2.error from imshow/waitKey because it is compiled without GTK.
    That is not a reason to kill a run whose real output is the files on disk,
    so the failure is reported once and then windows are quietly skipped.
    """
    global _gui_unavailable
    if _gui_unavailable:
        return fallback
    try:
        return call()
    except Exception as exc:  # cv2.error, and anything else the backend raises
        _gui_unavailable = True
        print(
            f"WARNING: OpenCV cannot open windows ({exc.__class__.__name__}); "
            "continuing without them. Output files are still written."
        )
        return fallback


def imshow(cv2, window_name, image):
    """`cv2.imshow` that becomes a no-op when no window can be shown."""
    if is_headless():
        return
    _guard_gui(lambda: cv2.imshow(window_name, image), None)


def move_window(cv2, window_name, x, y):
    """`cv2.moveWindow` that becomes a no-op when no window can be shown.

    Positioning a window that was never created raises "NULL guiReceiver", so
    tiling calls need the same guard as imshow() rather than being left bare.
    """
    if is_headless():
        return
    _guard_gui(lambda: cv2.moveWindow(window_name, x, y), None)


def destroy_windows(cv2):
    """`cv2.destroyAllWindows` plus the event-pump calls it needs to take effect."""
    if is_headless():
        return

    def _destroy():
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    _guard_gui(_destroy, None)


def show_plotter(plotter, screenshot_path=None):
    """Show a PyVista plotter without blocking.

    Interactively this is a normal blocking `plotter.show()`. Otherwise the
    scene is rendered off-screen and (optionally) written to `screenshot_path`,
    so the 3-D view still produces an artifact you can look at afterwards
    instead of a window nobody can close.
    """
    if is_interactive():
        return plotter.show()
    if screenshot_path:
        parent = os.path.dirname(str(screenshot_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # interactive=False + auto_close=True renders, writes the PNG and tears
        # the render window down again without ever waiting for a close event.
        result = plotter.show(
            screenshot=screenshot_path, auto_close=True, interactive=False,
        )
        print(f"3-D view written to: {screenshot_path}")
        return result
    plotter.close()
    return None


def prompt(message, default=""):
    """Non-blocking replacement for `input()`.

    Returns `default` (empty string unless overridden) without reading stdin
    unless the user opted in to interactive mode, so an unattended run never
    stalls on a prompt that nobody will answer.
    """
    if not is_interactive():
        return default
    try:
        return input(message)
    except EOFError:
        # stdin closed mid-run (piped input exhausted, detached process).
        return default


def iteration_limit(env_var, default):
    """Bound on a would-be-infinite `while True:` viewer loop.

    Frame browsers loop until the user presses 'q'. With nobody to press it,
    they must stop on their own; this returns how many iterations they may run.
    None means unbounded (interactive mode, where the user is in control).
    """
    if is_interactive():
        return None
    override = os.environ.get(env_var)
    if override is not None:
        return int(override)
    return default
