# =============================================================================
# shallow-vessel-palpation-simulator-and-AI — single-container environment
#
#   Taichi FEM simulation of a ViTacTip optical tactile sensor
#   + a PyTorch-Geometric GNN for subsurface feature localisation.
#
# Base image decision:
#   Unlike the AMBF/ROS setup this was adapted from, nothing here needs ROS, so
#   we are NOT pinned to Ubuntu 20.04. The two hard requirements are:
#     * A CUDA runtime new enough for the torch cu126 wheels (CUDA >= 12.6).
#     * An OpenGL-capable image, because Taichi's GGUI and Gmsh's FLTK viewer
#       are genuine GUI applications with no headless mode.
#   nvidia/cuda:12.6.x-devel-ubuntu22.04 gives CUDA 12.6 + Python 3.10, and the
#   `devel` (rather than `runtime`) flavour ships the headers Taichi needs to
#   JIT-compile its CUDA kernels.
#
#   ONE interpreter only: the whole project (simulator + ML) runs on a single
#   Python 3.10, so there is no micromamba/conda/venv layering here. Taichi,
#   torch and torch-geometric are co-installable, which is what makes this
#   possible.
# =============================================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# 1. Base tools + locale
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git sudo gnupg2 lsb-release \
        build-essential cmake pkg-config locales vim less bash-completion \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# 2. Python 3.10 (system interpreter) + GUI / OpenGL / video libraries.
#    - libgl1, libglib2.0-0, libsm6, libxext6, libxrender1: OpenCV + Taichi GGUI
#    - libx11/libxcursor/libxinerama/libxi/libxrandr: GLFW window creation
#      (Taichi GGUI uses GLFW under the hood)
#    - libgomp1: OpenMP runtime used by Taichi's CPU backend and scikit-learn
#    - python3-tk: the tkinter marker-labelling GUI in the preprocessing chain
#    - libgl1-mesa-dri + mesa-utils: a software GL fallback so the GUI still
#      comes up if the NVIDIA GL stack is not passed through
#    - ffmpeg: OpenCV video I/O for the real-sensor .mkv/.avi recordings
#    - libglu1-mesa / libxcb-*: Gmsh's FLTK viewer
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-tk \
        libgl1 libglu1-mesa libgl1-mesa-dri mesa-utils \
        libglib2.0-0 libsm6 libxext6 libxrender1 libxfixes3 \
        libx11-6 libxcursor1 libxinerama1 libxi6 libxrandr2 libxss1 \
        libxcb1 libxcb-render0 libxcb-shm0 libxkbcommon-x11-0 \
        libgomp1 ffmpeg x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Make `python`/`pip` resolve to python3 so the docs can say `python -m ...`.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# 3. PyTorch (CUDA 12.6 wheels) — installed first and alone, because it is the
#    largest layer and the one we least want to invalidate.
#    torch 2.8.0+cu126 matches requirements/requirements_ml.in, which pins the
#    torch-geometric companion wheels against exactly that build string.
# -----------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.8.0" "torchvision==0.23.0"

# -----------------------------------------------------------------------------
# 4. torch-geometric + its compiled companion wheels.
#    pyg_lib/torch_scatter/torch_sparse/torch_cluster/torch_spline_conv are
#    built per (torch, CUDA) pair and must come from the matching PyG index.
# -----------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir torch_geometric \
    && python -m pip install --no-cache-dir \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# -----------------------------------------------------------------------------
# 5. Simulator + ML + analysis stack.
#    opencv-python-headless is deliberately NOT used: the preprocessing chain
#    opens real cv2 GUI windows (the annotation tool), so we need the full build.
# -----------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        taichi \
        numpy \
        scipy \
        opencv-python \
        matplotlib \
        seaborn \
        pandas \
        scikit-learn \
        lightning \
        torchmetrics \
        tensorboard \
        tensorboardX \
        tqdm \
        shapely \
        sympy \
        trimesh \
        meshio \
        pyvista \
        vedo \
        open3d \
        gmsh \
        mesh-to-sdf \
        bayesian-optimization \
        Pillow \
        ipython

# -----------------------------------------------------------------------------
# 6. Environment.
#    DIFFTACTILE_ROOT is read by difftactile/main/paths.py, which resolves every
#    config/data path against it instead of the current working directory. That
#    is what allows `python -m difftactile.scripts.*` to run from anywhere.
#    PYTHONPATH makes the bind-mounted source importable without `pip install`,
#    so host edits take effect with no rebuild.
# -----------------------------------------------------------------------------
ENV DIFFTACTILE_ROOT=/workspace/shallow-vessel-palpation-simulator-and-AI
ENV PYTHONPATH=/workspace/shallow-vessel-palpation-simulator-and-AI
ENV PYTHONUNBUFFERED=1
# Taichi picks its GPU memory budget from here; the host GPU this targets is a
# 10 GB RTX 3080, and 9 GB leaves headroom for the display server.
ENV TI_DEVICE_MEMORY_GB=9
# Let the NVIDIA runtime expose graphics (not just compute) so GGUI can render.
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
# Qt/OpenCV under X11 forwarding: MIT-SHM is not available across the container
# boundary, and disabling it avoids intermittent BadAccess crashes.
ENV QT_X11_NO_MITSHM=1

WORKDIR /workspace/shallow-vessel-palpation-simulator-and-AI

RUN printf '%s\n' \
        'export DIFFTACTILE_ROOT=/workspace/shallow-vessel-palpation-simulator-and-AI' \
        'export PYTHONPATH=/workspace/shallow-vessel-palpation-simulator-and-AI' \
        'cd /workspace/shallow-vessel-palpation-simulator-and-AI 2>/dev/null || true' \
        "PS1='[difftactile \W] '" \
        >> /root/.bashrc

CMD ["bash"]

# -----------------------------------------------------------------------------
# 7. Vulkan — required by Taichi GGUI.
#    Taichi's GGUI backend is Vulkan-only ("RuntimeError: Vulkan must be
#    available for GGUI"); OpenGL alone is not enough, so the loader and headers
#    are needed even though the CUDA base image already provides GL.
#    The NVIDIA container runtime injects the driver's Vulkan ICD when
#    NVIDIA_DRIVER_CAPABILITIES includes `graphics` (we set `all`), but the ICD
#    file it writes references the driver library by name, so a JSON manifest
#    pointing at it is written here as a fallback for runtimes that do not.
#    Headless runs (DIFFTACTILE_HEADLESS=1) never touch this path.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 vulkan-tools libxrandr2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/vulkan/icd.d \
    && printf '%s\n' \
        '{' \
        '    "file_format_version": "1.0.0",' \
        '    "ICD": {' \
        '        "library_path": "libGLX_nvidia.so.0",' \
        '        "api_version": "1.3.0"' \
        '    }' \
        '}' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json

# -----------------------------------------------------------------------------
# 8. Interactive-shell conveniences: a short prompt + arrow-key history search
#    and no terminal bell (inputrc).
#    NOTE: docker-connect.sh shells run as the invoking host user with
#    HOME=/tmp, so the /root/ copies below are only seen by root shells
#    (docker exec -u 0). The /etc/ copies are what those non-root shells
#    actually read: /etc/inputrc is readline's system-wide default, and
#    /etc/bash.bashrc is sourced by both login and interactive shells (the
#    PS1 line is appended, so it wins over the file's default prompt).
# -----------------------------------------------------------------------------
RUN echo "PS1='[docker \W] '" >> ~/.bashrc

COPY inputrc /root/.inputrc

COPY inputrc /etc/inputrc
RUN echo "PS1='[docker \W] '" >> /etc/bash.bashrc

# -----------------------------------------------------------------------------
# 9. The annotation viewers: PySide6 (Qt 6) + PyAV, and a Qt 6 runtime that can
#    drive BOTH backends.
#
#    Why this is here at all. `docker/annotate_data_bare_metal.sh` runs the two interactive
#    annotation viewers on bare metal, in the tiny `vessel-palpation-annotator`
#    micromamba env, because a hand-driven frame stepper is the one tool whose
#    latency a human actually feels. `docker/annotate_data_docker.sh` is its
#    in-container twin: same modules, same libraries, so the two can be run
#    side by side and compared. That comparison is only meaningful if the
#    container reaches the compositor the same way the host does - natively over
#    the Wayland socket, with no Xwayland in the path - which is what the
#    packages below and the socket mounts in docker-run.sh are for.
#
#    Backend selection is PER PROCESS, not per container: each app picks its
#    transport from its own environment (QT_QPA_PLATFORM / DISPLAY /
#    WAYLAND_DISPLAY). So installing both backends costs nothing at runtime and
#    lets the Qt viewers run on Wayland while the project's OpenCV/Taichi/Gmsh
#    windows keep using X11 in the same container.
#
#    Package notes:
#      * libxcb-cursor0 / -icccm4 / -image0 / -keysyms1 / -render-util0 / -util1
#        are the well-known missing links behind Qt 6's "could not load the Qt
#        platform plugin xcb" error. The base image already has the other
#        libxcb-* pieces, and all the WAYLAND libraries (libwayland-client,
#        -cursor, -egl, libxkbcommon, libEGL) are present too, so only the xcb
#        side needs topping up.
#      * libxkbcommon0 is what the Wayland plugin uses for the keymap the
#        compositor hands it over an fd - the annotator is keyboard-driven, so
#        this is load-bearing rather than cosmetic.
#      * libdecor / GTK are NOT installed. Qt draws its own client-side window
#        decorations via the bundled `libqwayland-generic` plugin.
#      * PySide6/av versions match requirements/annotator-env.yml so the two
#        scripts really do run the same code. OpenCV stays the GUI-capable wheel
#        here (unlike the annotator env) because the rest of this image's
#        pipeline still opens cv2 windows.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-util1 libxcb-shape0 libxcb-xkb1 \
        libxkbcommon0 libxkbcommon-x11-0 \
        libwayland-client0 libwayland-cursor0 libwayland-egl1 \
        libegl1 libdbus-1-3 libfontconfig1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        "PySide6==6.9.1" \
        "av==15.1.0"

# ---------------------------------------------------------------------------
# 10. Scientific colourmaps (Crameri) for the manuscript's curve figures.
#     cnn/curve_plots.py colour-codes the decision threshold with `batlow`,
#     a perceptually uniform, colour-blind-safe map with no washed-out end on
#     a white page. Pure Python, no system deps; appended as its own layer so
#     the layers above are not rebuilt.
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir "cmcrameri==1.9"
