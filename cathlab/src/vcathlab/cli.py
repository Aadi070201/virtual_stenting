"""``vcathlab``: the virtual cath lab's command line. Research use only."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from vcathlab import DISCLAIMER

app = typer.Typer(help=f"Virtual cath lab. {DISCLAIMER}", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Virtual cath lab: simulated coronary angiography with full ground truth."""
    for stream in (sys.stdout, sys.stderr):  # consoles without UTF-8 (Windows) replace, not fail
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


@app.command()
def frame(
    seed: Annotated[int, typer.Option("--seed", help="Random patient and scene.")] = 0,
    scene_file: Annotated[
        Path | None, typer.Option("--scene", help="A scene YAML instead of a random one.")
    ] = None,
    out: Annotated[Path, typer.Option("--out")] = Path("data/frame"),
) -> None:
    """One frame with every label → image.png, overlay.png, vessel.png, segments.png,
    overlap.png, catheter.png, scene.yaml and truth.json in ``--out``."""
    import cv2
    import yaml

    from vcathlab.config import git_info
    from vcathlab.scene import Scene, SceneConfig, overlay, sample_scene, seg_colours, simulate

    cfg = SceneConfig()
    if scene_file is not None:
        scene = Scene.model_validate(yaml.safe_load(scene_file.read_text(encoding="utf-8")))
    else:
        scene = sample_scene(seed, cfg)
    f = simulate(scene, cfg)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "image.png"), f.image)
    cv2.imwrite(str(out / "overlay.png"), cv2.cvtColor(overlay(f), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "segments.png"), cv2.cvtColor(seg_colours(f), cv2.COLOR_RGB2BGR))
    for name in ("vessel", "overlap", "catheter"):
        cv2.imwrite(str(out / f"{name}.png"), getattr(f, name) * 255)
    (out / "scene.yaml").write_text(
        yaml.safe_dump(scene.model_dump(), sort_keys=False), encoding="utf-8"
    )
    truth = {"disclaimer": DISCLAIMER, "truth": f.truth, "lesions": f.lesions, **git_info()}
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    typer.echo(DISCLAIMER)
    typer.echo(
        f"{scene.artery} coronary, {f.truth['view']}, {f.truth['iso_pixel_mm']:.3f} mm/px "
        f"at the isocentre; {len(f.lesions)} lesions → {out}"
    )
