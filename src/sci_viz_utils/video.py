"""Generic frame iteration and video-writing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .arrays import to_uint8_frame


def iter_video_frames(path, every=1, max_frames=None, gray=True):
    """Yield frames from a video using imageio first, then OpenCV as fallback."""

    path = Path(path)
    every = max(int(every), 1)
    count = 0
    try:
        import imageio.v3 as iio

        for i, frame in enumerate(iio.imiter(path)):
            if i % every:
                continue
            frame = np.asarray(frame)
            if gray and frame.ndim == 3:
                frame = frame[..., :3].mean(axis=2)
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                return
        return
    except Exception:
        pass

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install sci-viz-utils[video], imageio, or opencv-python to read videos.") from exc

    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield frame
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def write_video(frames, output, fps: int = 10, cmap: str = "gray") -> Path:
    """Write a 2D grayscale or RGB frame sequence to a video file."""

    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install sci-viz-utils[video] to write videos.") from exc

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmap_obj = plt.get_cmap(cmap)
    prepared = []
    for frame in np.asarray(frames):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            rgb = (cmap_obj(to_uint8_frame(arr) / 255.0)[..., :3] * 255).astype(np.uint8)
        else:
            rgb = to_uint8_frame(arr) if arr.dtype != np.uint8 else arr
        prepared.append(rgb)
    imageio.mimsave(output, prepared, fps=fps)
    return output


def make_video(image_sequences, titles=None, output="video.mp4", fps=5, cmap="viridis", clim="auto"):
    """Create an MP4 from one or more image sequences using Matplotlib animation."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    sequences = [np.asarray(seq) for seq in image_sequences]
    if not sequences:
        raise ValueError("image_sequences must contain at least one sequence")
    num_sequences = len(sequences)
    num_frames = sequences[0].shape[0]

    fig, axes = plt.subplots(1, num_sequences, figsize=(4 * num_sequences + 1, 3), squeeze=False)
    axes = axes.ravel()
    ims = []
    vlims = []
    for seq in sequences:
        if clim == "global":
            mean, std = np.mean(seq), np.std(seq)
            vlims.append((mean - 3 * std, mean + 3 * std))
        elif isinstance(clim, tuple):
            vlims.append(clim)
        else:
            vlims.append(None)

    for ax, seq, vlim in zip(axes, sequences, vlims):
        im = ax.imshow(seq[0], cmap=cmap if seq.ndim == 3 else None)
        if vlim is not None:
            im.set_clim(*vlim)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.axis("off")
        ims.append(im)

    title_obj = fig.suptitle(titles[0] if titles else "", fontsize=12)

    def update(frame):
        for im, seq, vlim in zip(ims, sequences, vlims):
            im.set_data(seq[frame])
            if vlim is None:
                im.set_clim(np.min(seq[frame]), np.max(seq[frame]))
        if titles:
            title_obj.set_text(titles[frame])
        return ims + [title_obj]

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    ani.save(output, writer=writer)
    plt.close(fig)
    return Path(output)


__all__ = ["iter_video_frames", "make_video", "write_video"]
