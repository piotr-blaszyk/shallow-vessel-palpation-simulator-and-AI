"""Qt 6 (PySide6) frame browser shared by the two annotation viewers.

This is the display layer for `docker/annotate_data.sh` and nothing else. The
rest of the project still draws its windows with OpenCV; only the two
hand-driven annotation tools were moved, because they are the only ones whose
responsiveness a human actually feels.

Why Qt rather than OpenCV
-------------------------
The `opencv-python` wheel ships exactly one Qt platform plugin - `xcb` - so on a
Wayland desktop every `cv2.imshow` window is an Xwayland client. That works, but
it is a compatibility path: it costs an extra copy per repaint, and it is why the
OpenCV frame-browser loop this replaces had grown a "double present" workaround
for stale compositor buffers. The PySide6 wheels bundle `libqwayland-generic.so`
and `libqwayland-egl.so`, so Qt registers a real `wayland` platform and the
viewers become native Wayland clients with no Xwayland in the loop. Nothing here
needs `DISPLAY`.

What this module provides
-------------------------
`FrameBrowser`, a `QGraphicsView` showing one video frame as a
`QGraphicsPixmapItem` with a `QLabel` status bar underneath, plus:

* `set_frame(bgr)` - swap the displayed frame (a numpy BGR array, as OpenCV and
  PyAV both produce).
* `set_status(lines)` - replace the status bar text.
* `set_overlay_points(points)` - draw annotation dots as real
  `QGraphicsEllipseItem` scene objects rather than pixels burned into the image.
  They are hit-tested by Qt, so a click can select the dot under the cursor and
  Delete removes exactly that one - which is what the `cv2.circle` version could
  not do at all.

The scene is kept in **full video-resolution coordinates** and the *view* is
scaled to fit the window (`fitInView`). That is the important structural
difference from the OpenCV code, which downscaled the image itself and then had
to scale every annotation on the way in and every click on the way back out.
Here `mapToScene()` does that conversion, correctly, for free - including when
the user resizes the window or zooms. Annotations therefore stay in the video's
own pixels on disk with no arithmetic in the callers.

Keyboard handling stays deliberately close to the OpenCV loop it replaces: the
viewers' existing `on_key(key)` callbacks take an integer key code and return
`"redraw"`, `"quit"` or `None`, and this widget calls them with exactly that
contract, so their key-handling logic did not have to be rewritten.
"""

import numpy as np

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QLabel, QVBoxLayout, QWidget,
)


def _qimage_from_bgr(frame):
    """Wrap a numpy BGR array in a QImage, copying so Qt owns the memory.

    `Format_BGR888` matches OpenCV's and PyAV's channel order directly, so no
    colour conversion is needed. `.copy()` is not optional: a QImage built on a
    numpy buffer does not keep that buffer alive, and the frame arrays here are
    owned by per-video caches that the caller is free to drop.
    """
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_BGR888)
    return image.copy()


class FrameBrowser(QWidget):
    """A frame view with a status bar, driven by `render`/`on_key` callbacks.

    Args:
        title: window title.
        render: `() -> numpy BGR array | None`, the frame for the current state.
            Returning None closes the window, matching the old loop's contract.
        on_key: `(key_code) -> "redraw" | "quit" | None`. Key codes are the
            integer ordinals the OpenCV viewers already switch on, e.g.
            `ord("k")`, so existing handlers work unchanged.
        on_click: optional `(x, y) -> "redraw" | None`, called with a left click
            in full-resolution video coordinates.
        on_delete: optional `(index) -> "redraw" | None`, called with the index
            of the overlay point the user clicked on and then deleted. This is
            what the scene-graph overlay buys over burned-in circles.
        status: optional `() -> list[str]` for the status bar, re-read on every
            redraw.
        overlay_points: optional `() -> [(x, y), ...]` in full-resolution video
            coordinates, re-read on every redraw and drawn as scene items.
        point_colours: BGR tuples cycled over the overlay points, so the palette
            stays defined by the caller rather than duplicated here.
    """

    def __init__(self, title, render, on_key, on_click=None, on_delete=None,
                 status=None, overlay_points=None, point_colours=None):
        super().__init__()
        self._render = render
        self._on_key = on_key
        self._on_click = on_click
        self._on_delete = on_delete
        self._status = status
        self._overlay_points = overlay_points
        self._point_colours = point_colours
        self._points = []          # scene-space annotation dots
        self._point_items = []     # their QGraphicsEllipseItems, same order
        self._selected = None      # index of the dot the user clicked, if any

        self.setWindowTitle(title)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._view = _FrameView(self._scene, self)
        self._label = QLabel("")
        # Monospace keeps the counters from jittering as their widths change.
        self._label.setStyleSheet(
            "font-family: monospace; font-size: 13px; padding: 6px;"
        )
        self._label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(self._label)
        self.resize(1000, 760)

    # --- display ----------------------------------------------------------

    def set_frame(self, frame):
        """Show `frame` (numpy BGR), refitting the view if the size changed."""
        pixmap = QPixmap.fromImage(_qimage_from_bgr(frame))
        size_changed = pixmap.size() != self._pixmap_item.pixmap().size()
        self._pixmap_item.setPixmap(pixmap)
        # The scene rect is the video's own pixel grid, so mapToScene() yields
        # full-resolution coordinates regardless of how the view is scaled.
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if size_changed:
            self._view.fit()

    def set_status(self, lines):
        self._label.setText("\n".join(lines))

    def set_overlay_points(self, points, radius=14, colours=None):
        """Draw annotation dots as scene items in full-resolution coordinates.

        Each point becomes a `QGraphicsEllipseItem`, so Qt can hit-test it and
        the user can select and delete an individual dot. `radius` is in scene
        (video) pixels, so a dot keeps its size relative to the image as the
        window is resized.
        """
        for item in self._point_items:
            self._scene.removeItem(item)
        self._point_items = []
        self._points = list(points)
        if self._selected is not None and self._selected >= len(self._points):
            self._selected = None

        default = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        colours = colours or default
        for idx, (x, y) in enumerate(self._points):
            b, g, r = colours[idx % len(colours)]
            item = QGraphicsEllipseItem(
                QRectF(x - radius, y - radius, radius * 2, radius * 2)
            )
            item.setBrush(QBrush(QColor(r, g, b)))
            # The selected dot gets a thick white ring so it is obvious which
            # one Delete will remove.
            if idx == self._selected:
                item.setPen(QPen(QColor(255, 255, 255), 4))
            else:
                item.setPen(QPen(QColor(0, 0, 0), 2))
            item.setZValue(1)
            self._scene.addItem(item)
            self._point_items.append(item)

    def selected_point(self):
        return self._selected

    def redraw(self):
        """Re-run `render` (and `status`), closing the window if it returns None."""
        frame = self._render()
        if frame is None:
            self.close()
            return
        self.set_frame(frame)
        if self._overlay_points is not None:
            self.set_overlay_points(
                self._overlay_points(), colours=self._point_colours
            )
        if self._status is not None:
            self.set_status(self._status())

    # --- input ------------------------------------------------------------

    def keyPressEvent(self, event):
        """Translate a Qt key event into the viewers' integer-key contract."""
        key = event.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._on_delete is not None and self._selected is not None:
                outcome = self._on_delete(self._selected)
                self._selected = None
                self._apply(outcome)
            return
        if key == Qt.Key_Escape:
            self._selected = None
            self.redraw()
            return

        text = event.text()
        if not text:
            return super().keyPressEvent(event)
        outcome = self._on_key(ord(text[0].lower()))
        self._apply(outcome)

    def _apply(self, outcome):
        if outcome == "quit":
            self.close()
        elif outcome == "redraw":
            self.redraw()

    def handle_click(self, scene_pos, button):
        """A click in the view, already mapped to full-resolution coordinates."""
        x, y = scene_pos.x(), scene_pos.y()
        hit = self._hit_test(scene_pos)
        if hit is not None:
            # Clicking an existing dot selects it rather than adding another on
            # top of it; Delete then removes that one specifically.
            self._selected = None if self._selected == hit else hit
            self.redraw()
            return
        self._selected = None
        if button == Qt.LeftButton and self._on_click is not None:
            self._apply(self._on_click(x, y))

    def _hit_test(self, scene_pos):
        """Index of the overlay dot under `scene_pos`, or None."""
        for idx, item in enumerate(self._point_items):
            if item.contains(item.mapFromScene(scene_pos)):
                return idx
        return None


class _FrameView(QGraphicsView):
    """QGraphicsView that keeps the frame fitted and forwards clicks upward.

    Kept private: `FrameBrowser` owns the behaviour, this only handles the
    view-space concerns (fitting on resize, mapping clicks into scene space).
    """

    def __init__(self, scene, browser):
        super().__init__(scene)
        self._browser = browser
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Focus lives on the parent FrameBrowser so keyPressEvent lands there.
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet("background: #202020; border: none;")

    def fit(self):
        rect = self.scene().sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def mousePressEvent(self, event):
        """Map the click into scene (video) pixels and hand it to the browser.

        `mapToScene` undoes the view's fit-to-window scaling exactly, which is
        what removes the manual click-rescaling the OpenCV annotator needed.
        """
        self._browser.handle_click(
            self.mapToScene(event.position().toPoint()), event.button()
        )
        super().mousePressEvent(event)


def run_browser(title, render, on_key, on_click=None, on_delete=None, status=None,
                overlay_points=None, point_colours=None, on_close=None):
    """Open a `FrameBrowser` and run the Qt event loop until it is closed.

    Blocks, like any ordinary GUI. The callers are the two interactive
    annotation tools, which already refuse to run without
    DIFFTACTILE_INTERACTIVE=1, so there is no non-blocking mode to provide here.

    `on_close` is called once the window has gone, which is where the silicone
    annotator saves.
    """
    app = QApplication.instance() or QApplication([])
    browser = FrameBrowser(
        title, render, on_key, on_click, on_delete, status, overlay_points,
        point_colours,
    )
    browser.redraw()
    browser.show()
    browser.setFocus()
    app.exec()
    if on_close is not None:
        on_close()
    return browser
