"""Video decoding for the interactive annotation viewers, via PyAV.

The viewers used `cv2.VideoCapture` to read whole clips into a RAM cache. That
worked, but it ties the annotator environment to a GUI-capable OpenCV build and
gives no frame-accurate seeking if the caching strategy ever changes.
`cv2.VideoCapture` seeks by keyframe and its `CAP_PROP_POS_FRAMES` is unreliable
on the H.264 clips here; PyAV decodes the real frames, so frame `i` of the
returned list is genuinely frame `i` of the file - which matters because the
meat viewer indexes its per-frame label arrays by exactly that number.

Only the two annotation tools use this. Everything else in the project still
decodes with OpenCV.
"""

import av
import numpy as np


def decode_frames(path, max_frames=None):
    """Decode a video to a list of numpy BGR frames, in order.

    BGR because every drawing and preprocessing path in this project is written
    against OpenCV's channel order, and `Format_BGR888` lets Qt display it with
    no conversion either.

    Returns an empty list if the file cannot be opened or holds no video
    stream - the callers treat "no frames" as "nothing to show" rather than an
    error, matching the previous `cv2.VideoCapture` behaviour.
    """
    frames = []
    try:
        container = av.open(str(path))
    except Exception as exc:
        print(f"WARNING: could not open {path}: {exc.__class__.__name__}: {exc}")
        return frames
    try:
        if not container.streams.video:
            return frames
        stream = container.streams.video[0]
        # Let PyAV use all cores for H.264; these are 1080p clips and decoding
        # is the dominant cost of opening a video.
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(np.ascontiguousarray(frame.to_ndarray(format="bgr24")))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        container.close()
    return frames


def count_frames(path):
    """Number of decodable frames, used to size a fresh annotation list.

    The container's advertised frame count is metadata and can be wrong or
    absent, and an annotation list that is shorter than the video would raise
    IndexError mid-session - so this decodes and counts. Decoded frames are
    discarded; callers that need the pixels use `decode_frames`.
    """
    return len(decode_frames(path))
