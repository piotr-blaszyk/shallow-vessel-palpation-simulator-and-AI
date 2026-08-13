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
    DIFFTACTILE_VIEW_WIDTH      Width in pixels that the frame browsers scale
                                their display image down to (default 960, `0`
                                disables scaling). The source videos are 1080p,
                                which is both bigger than most screens and slow
                                to push over an X connection; see
                                `fit_to_view()`.

With neither set, windows are still drawn (so you can watch a run go past) but
they are never waited on: `show_plots()` returns False, `wait_key()` returns
"no key pressed" after a short delay, and the frame-browser loops advance on
their own instead of asking for a keystroke.
"""

import os


def is_headless():
    """True when no window should be created at all."""
    if os.environ.get("DIFFTACTILE_HEADLESS", "0") == "1":
        return True
    return not os.environ.get("DISPLAY")


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


def view_width():
    """Width the frame browsers scale their display image down to, or 0 for none.

    Override with DIFFTACTILE_VIEW_WIDTH; `0` keeps the frames at full size.
    """
    try:
        return max(0, int(os.environ.get("DIFFTACTILE_VIEW_WIDTH", "960")))
    except ValueError:
        return 960


def fit_to_view(cv2, image):
    """Scale a frame down to `view_width()` for display, preserving aspect ratio.

    The source videos are 1080p, so every `cv2.imshow` hands the GUI a 6.2 MB
    uncompressed BGR buffer. That is the dominant per-keypress cost in the
    interactive browsers: it is larger than most screens (so the window is
    downscaled for display anyway), and when the window is served over a
    forwarded X connection the whole buffer crosses a socket on every repaint,
    which is what makes stepping through frames feel choppy. Halving each
    dimension cuts the data pushed per repaint by ~4x.

    Only the *displayed* copy is resized. Annotation coordinates are unaffected:
    callers scale clicks back up themselves (see the silicone annotator's mouse
    callback), so what is stored on disk stays in full-resolution pixels.

    Returns `(scaled_image, scale)`, where `scale` is the factor applied - 1.0
    when the frame was already small enough or scaling is disabled.
    """
    width = view_width()
    if not width:
        return image, 1.0
    h, w = image.shape[:2]
    if w <= width:
        return image, 1.0
    scale = width / float(w)
    resized = cv2.resize(
        image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
    )
    return resized, scale


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


def run_frame_browser(cv2, window, render, on_key):
    """Shared event loop for the frame-stepping annotation viewers.

    Both real-world datasets are browsed the same way - step through frames,
    hop between videos/trials, quit on `q` - but their data models are not
    interchangeable (the silicone set is up to four hand-clicked points per
    frame in a pickle; the meat set is a fixed 127-marker label vector per frame
    in an npz, derived from kinematics and not editable). So the *shell* is
    shared here and each viewer keeps its own rendering and key handling.

    Centralising it also means the X11 paint-ordering fix lives in exactly one
    place. `imshow()` only queues an image; it is `waitKey()` that pumps the GUI
    event loop and actually paints. Blocking in `waitKey(0)` straight after an
    `imshow()` lets an already-pending keystroke be delivered *before* the paint
    is serviced, so the window lingers on the previous frame and then snaps to
    the requested one - visible as one wrong frame followed by the right one.
    Under X11 forwarding the round trip is slow enough to make that obvious,
    while on bare metal both updates usually land within one refresh. The loop
    below pumps once after drawing, and carries over any key that arrives during
    the pump instead of dropping it.

    Args:
        cv2: the OpenCV module (passed in so this file needs no cv2 import).
        window: window name.
        render: `() -> image | None` for the current state. Returning None ends
            the loop. Called only when `on_key` reports the state changed.
        on_key: `(key) -> "redraw" | "quit" | None`. Return "redraw" when the
            key changed what should be displayed, "quit" to exit.

    The loop always BLOCKS on a keypress, like any ordinary GUI - it never wakes
    on a timer. A viewer whose state can also change without a keypress (the
    silicone annotator, via a mouse callback) repaints from inside its own
    callback: `cv2.waitKey(0)` is what pumps the GUI event loop, so callbacks
    run while this loop is blocked and can draw straight to the window.

    Returns when the user quits, or immediately when not interactive.
    """
    if not is_interactive():
        return

    needs_redraw = True
    pending_key = None
    while True:
        if needs_redraw:
            image = render()
            if image is None:
                break
            imshow(cv2, window, image)
            needs_redraw = False
            # Present the frame before blocking; keep any key typed meanwhile.
            pumped = wait_key(cv2, 1) & 0xFF
            if pumped != 255:
                pending_key = pumped
            # Present the same frame a second time. Under rootless Xwayland the
            # window is composited from a small pool of buffers, and a keypress
            # can briefly re-present a stale one still holding the frame from
            # two redraws ago (seen as a flash of the opposite-direction frame).
            # After a second present every buffer holds the current frame, so a
            # stale re-present shows the identical image and the flash vanishes.
            imshow(cv2, window, image)
            pumped = wait_key(cv2, 1) & 0xFF
            if pumped != 255:
                pending_key = pumped

        if pending_key is not None:
            key, pending_key = pending_key, None
        else:
            # Block until a key is actually pressed, like any ordinary GUI: no
            # timer, no polling, and no CPU burned while idle. An earlier version
            # woke every 30 ms for viewers with a mouse callback, to re-check a
            # dirty flag. That put a 30 ms floor on how fast frames could be
            # stepped and up to 30 ms of dead latency in front of every keypress
            # - about 4.5x the cost of an actual redraw (~7 ms), and the reason
            # stepping through frames felt choppy. Mouse callbacks now repaint
            # themselves, so there is nothing left to poll for.
            key = wait_key(cv2, 0) & 0xFF
            if key == 255:
                continue

        outcome = on_key(key)
        if outcome == "quit":
            break
        if outcome == "redraw":
            needs_redraw = True
            # Collapse whatever else is already queued into the same redraw, so
            # a burst of presses costs one frame over the wire and lands exactly
            # N steps away rather than falling behind the keyboard.
            while True:
                extra = wait_key(cv2, 1) & 0xFF
                if extra == 255:
                    break
                outcome = on_key(extra)
                if outcome == "quit":
                    return destroy_windows(cv2)
    destroy_windows(cv2)


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
