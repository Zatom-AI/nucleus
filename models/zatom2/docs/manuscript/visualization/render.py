#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.63.0", "pillow==12.3.0", "pymupdf==1.28.2"]
# ///
"""Render the Appendix C structures with Mol*, then assemble labeled PDFs.

See README.md for browser installation and the pinned Mol* viewer bundle.
No inference, relaxation, refolding, or benchmark scoring is performed here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

INITIALIZE = """async () => {
    window.viewer = await molstar.Viewer.create('app', {
        layoutIsExpanded: false, layoutShowControls: false,
        layoutShowRemoteState: false, layoutShowSequence: false,
        layoutShowLog: false, viewportShowExpand: false,
        viewportShowControls: false, viewportShowSelectionMode: false,
        viewportShowAnimation: false, viewportBackgroundColor: 'white'
    });
    viewer.plugin.canvas3d.setProps({
        renderer: { backgroundColor: 0xffffff },
        camera: { mode: 'orthographic' },
        helper: { axes: { name: 'off', params: {} } }
    });
} """

RENDER = """async ({text, domain, atomCount}) => {
    const p = viewer.plugin;
    await p.clear();
    const data = await p.builders.data.rawData({data: text});
    const trajectory = await p.builders.structure.parseTrajectory(
        data, domain === 'proteins' ? 'pdb' : 'mmcif');
    const model = await p.builders.structure.createModel(trajectory);
    const structure = await p.builders.structure.createStructure(model,
        domain === 'materials'
        ? {name: 'symmetry', params: {ijkMin: [0,0,0], ijkMax: [1,1,1]}}
        : {name: 'model', params: {}});
    if (domain !== 'proteins') {
        const expected = atomCount * (domain === 'materials' ? 8 : 1);
        if (structure.obj.data.elementCount !== expected)
            throw new Error(`Expected ${expected} atoms; found ${structure.obj.data.elementCount}`);
    }
    const component = await p.builders.structure.tryCreateComponentStatic(
        structure, domain === 'proteins' ? 'polymer' : 'all');
    if (!component || !component.obj.data.elementCount)
        throw new Error('Empty structure component');
    const props = domain === 'proteins'
        ? {type: 'cartoon', typeParams: {quality: 'highest'},
           color: 'sequence-id', colorParams: {list: {kind: 'interpolate',
               colors: [0x3155a4, 0x26a6c7, 0x58b86b, 0xf2d34f, 0xd63b36]}}}
        : {type: domain === 'materials' ? 'spacefill' : 'ball-and-stick',
           typeParams: {sizeFactor: domain === 'materials' ? 0.38 : 0.24,
                        quality: 'highest'},
           color: 'element-symbol',
           colorParams: {carbonColor: {name: 'element-symbol', params: {}}}};
    await p.builders.structure.representation.addRepresentation(component, props);
    if (domain === 'materials') {
        const cell = await p.builders.structure.tryCreateUnitcell(model,
            {cellColor: 0x515967, cellScale: 1.5, ref: 'origin', attachment: 'corner'});
        if (!cell) throw new Error('Missing material unit cell');
    }
    p.managers.camera.reset(undefined, 0);
    await new Promise(resolve => setTimeout(resolve, 500));
    if (domain !== 'materials') {
        p.managers.camera.orientAxes([structure.obj.data], 0);
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (domain === 'materials') {
        const camera = p.canvas3d.camera.getSnapshot();
        const t = camera.target;
        const distance = Math.hypot(...camera.position.map((x, i) => x-t[i]));
        const norm = Math.hypot(1.2, 0.8, 1.6);
        p.canvas3d.camera.setState({position: t.map((x, i) =>
            x + distance * [1.2,0.8,1.6][i] / norm), up: [0,1,0]}, 0);
    }
    const screenshot = p.helpers.viewportScreenshot;
    screenshot.behaviors.values.next({...screenshot.values,
        resolution: {name: 'custom', params: {width: 1400, height: 1100}},
        transparent: false});
    const uri = await screenshot.getImageDataUri();
    return {uri, camera: p.canvas3d.camera.getSnapshot(),
            elementCount: structure.obj.data.elementCount};
} """


def crop_image(data: bytes) -> bytes:
    """Trim white margins only, retaining all rendered geometry and padding."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    bounds = difference.convert("L").point(lambda value: 255 if value > 12 else 0)
    box = bounds.getbbox()
    if box is None:
        raise ValueError("Mol* produced a blank image")
    if box[0] == 0 or box[1] == 0 or box[2] == image.width or box[3] == image.height:
        raise ValueError("Rendered geometry touches the image edge")
    padding = 35
    image = image.crop(
        (
            max(0, box[0] - padding),
            max(0, box[1] - padding),
            min(image.width, box[2] + padding),
            min(image.height, box[3] + padding),
        )
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_gallery(gallery: str, samples: list[dict], images: dict, output: Path) -> None:
    """Assemble raster Mol* views and vector labels on a three-column grid."""
    expected = 12 if gallery == "omol25" else 9
    if len(samples) != expected:
        raise ValueError(f"Expected {expected} {gallery} panels, found {len(samples)}")
    doc = pymupdf.open()
    page = doc.new_page(width=504, height=152 * (expected // 3))
    for index, sample in enumerate(samples):
        x, y = (index % 3) * 168, (index // 3) * 152
        page.insert_image(
            pymupdf.Rect(x + 5, y + 2, x + 163, y + 124),
            stream=images[sample["sample_id"]],
        )
        letter = chr(ord("a") + index)
        if sample["domain"] == "proteins":
            title = f"({letter}) {sample['length']} residues"
            detail = f"scRMSD {sample['min_scRMSD']:.2f} Å"
            short_id = (
                sample["sample_id"].removeprefix("scope_").removesuffix("_0_model_0")
            )
            detail += f"  |  {short_id}"
        elif "category_label" in sample:
            title = f"({letter}) {sample['category_label']}"
            detail = f"{sample['display_id']} | {sample['atom_count']} atoms"
        else:
            title = f"({letter}) {sample['formula']}"
            detail = sample.get("display_id", sample["sample_id"])
        for text, baseline, size, font, color in (
            (title, y + 135, 10, "hebo", (0, 0, 0)),
            (detail, y + 147, 7.5, "helv", (0.25, 0.28, 0.32)),
        ):
            width = pymupdf.get_text_length(text, fontname=font, fontsize=size)
            if width > 164:
                raise ValueError(f"Panel label exceeds available width: {text}")
            page.insert_text(
                (x + (168 - width) / 2, baseline),
                text,
                fontsize=size,
                fontname=font,
                color=color,
            )
    doc.set_metadata(
        {
            "title": f"Zatom-2 generated {gallery}",
            "creator": "Mol* 5.11.0; visualization/render.py",
        }
    )
    doc.save(output / f"generated_{gallery}_molstar.pdf", deflate=True, garbage=4)
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molstar-viewer",
        type=Path,
        required=True,
        help="Directory containing pinned molstar.js and molstar.css",
    )
    parser.add_argument("--output", type=Path, default=HERE.parent / "figs")
    parser.add_argument(
        "--galleries",
        nargs="+",
        choices=("molecules", "materials", "proteins", "omol25", "omat24"),
        help="Render only these galleries (default: all); retain other audit entries",
    )
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    samples = [
        s
        for s in manifest["samples"]
        if args.galleries is None or s.get("gallery", s["domain"]) in args.galleries
    ]
    bundle = args.molstar_viewer / "molstar.js"
    if (
        hashlib.sha256(bundle.read_bytes()).hexdigest()
        != manifest["molstar_bundle_sha256"]
    ):
        raise ValueError("Mol* bundle does not match the pinned version/checksum")
    args.output.mkdir(parents=True, exist_ok=True)
    images, audit = {}, {"molstar_version": manifest["molstar_version"], "samples": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        audit["chromium_version"] = browser.version
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        # Inputs and scripts are local; block all network requests during rendering.
        page.route("**/*", lambda route: route.abort())
        page.set_content('<div id="app" style="position:absolute;inset:0"></div>')
        page.add_style_tag(path=str(args.molstar_viewer / "molstar.css"))
        page.add_script_tag(path=str(bundle))
        page.evaluate(INITIALIZE)
        for sample in samples:
            data = (HERE / sample["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != sample["sha256"]:
                raise ValueError(f"Structure checksum mismatch: {sample['file']}")
            result = page.evaluate(
                RENDER,
                {
                    "text": data.decode(),
                    "domain": sample["domain"],
                    "atomCount": sample.get("atom_count"),
                },
            )
            images[sample["sample_id"]] = crop_image(
                base64.b64decode(result.pop("uri").split(",", 1)[1])
            )
            audit["samples"].append(
                {
                    "sample_id": sample["sample_id"],
                    "chromium_version": browser.version,
                    **result,
                }
            )
            print(f"Rendered {sample['sample_id']}", flush=True)
        browser.close()
    for gallery in dict.fromkeys(s.get("gallery", s["domain"]) for s in samples):
        make_gallery(
            gallery,
            [s for s in samples if s.get("gallery", s["domain"]) == gallery],
            images,
            args.output,
        )
    audit_path = args.output / "molstar_render_audit.json"
    if args.galleries and audit_path.exists():
        previous = json.loads(audit_path.read_text())
        rendered_ids = {s["sample_id"] for s in samples}
        current_ids = {s["sample_id"] for s in manifest["samples"]}
        retained = [
            s
            for s in previous["samples"]
            if s["sample_id"] in current_ids - rendered_ids
        ]
        audit["samples"] = retained + audit["samples"]
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
