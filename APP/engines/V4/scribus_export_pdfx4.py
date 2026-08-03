"""Run inside Scribus' embedded Python to export one SVG as PDF/X-4.

The parent V4 process passes a JSON job file through RESIZE_V4_SCRIBUS_JOB.
Keeping the exchange file-based avoids command-line quoting problems on Windows.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import scribus


def main() -> None:
    config_path = Path(os.environ["RESIZE_V4_SCRIBUS_JOB"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    svg = Path(config["svg"])
    sla = Path(config["sla"])
    pdf_path = Path(config["pdf"])
    report = Path(config["report"])

    try:
        page_width_mm = float(config["page_width_mm"])
        page_height_mm = float(config["page_height_mm"])
        created = scribus.newDocument(
            (page_width_mm, page_height_mm),
            (0.0, 0.0, 0.0, 0.0),
            scribus.PORTRAIT,
            1,
            scribus.UNIT_MILLIMETERS,
            scribus.PAGE_1,
            0,
            1,
        )
        if not created:
            raise RuntimeError("Scribus could not create the print document")

        bleed = float(config.get("bleed_mm", 0.0))
        scribus.setBleeds(bleed, bleed, bleed, bleed)
        scribus.setInfo(
            str(config.get("author", "Local Print Image Upscaler")),
            str(config["title"]),
            str(config.get("description", "V4 full-vector PDF/X-4 print master")),
        )
        scribus.placeSVG(str(svg), 0.0, 0.0)
        selected_count = scribus.selectionCount()
        if selected_count < 1:
            raise RuntimeError("Scribus imported the SVG without a selectable artwork object")
        selected_names = [scribus.getSelectedObject(index) for index in range(selected_count)]
        if selected_count > 1:
            scribus.groupObjects(selected_names)
            if scribus.selectionCount() < 1:
                raise RuntimeError("Scribus lost the imported SVG selection after grouping")
        artwork_name = scribus.getSelectedObject(0)
        initial_width, initial_height = scribus.getSize(artwork_name)
        if initial_width <= 0.0 or initial_height <= 0.0:
            raise RuntimeError(
                f"Imported SVG has invalid size {initial_width} x {initial_height} mm"
            )

        # placeSVG imports the artwork at its natural bitmap size on some Scribus
        # builds, even when the SVG root declares the full physical page. Scale the
        # imported group explicitly so it covers the trim plus bleed, preserving its
        # aspect ratio and centring any tiny bleed crop.
        outer_width = page_width_mm + 2.0 * bleed
        outer_height = page_height_mm + 2.0 * bleed
        scale_factor = max(outer_width / initial_width, outer_height / initial_height)
        placed_width = initial_width * scale_factor
        placed_height = initial_height * scale_factor
        scribus.sizeObject(placed_width, placed_height, artwork_name)
        placed_width, placed_height = scribus.getSize(artwork_name)
        placed_x = (page_width_mm - placed_width) / 2.0
        placed_y = (page_height_mm - placed_height) / 2.0
        scribus.moveObjectAbs(placed_x, placed_y, artwork_name)
        actual_x, actual_y = scribus.getPosition(artwork_name)
        if (
            placed_width + 0.5 < outer_width
            or placed_height + 0.5 < outer_height
            or abs((actual_x + placed_width / 2.0) - page_width_mm / 2.0) > 0.5
            or abs((actual_y + placed_height / 2.0) - page_height_mm / 2.0) > 0.5
        ):
            raise RuntimeError(
                "Imported SVG did not cover and centre on the requested print page: "
                f"position={actual_x:.3f},{actual_y:.3f} mm, "
                f"size={placed_width:.3f}x{placed_height:.3f} mm, "
                f"outer={outer_width:.3f}x{outer_height:.3f} mm"
            )
        scribus.saveDocAs(str(sla))

        export = scribus.PDFfile()
        export.file = str(pdf_path)
        export.version = 10  # Scribus enum for ISO 15930-7 PDF/X-4.
        export.outdst = 1
        export.profiles = 1
        export.profilei = 1
        export.solidpr = str(config["rgb_profile_name"])
        export.imagepr = str(config["rgb_profile_name"])
        export.printprofc = str(config["output_profile_name"])
        export.intents = 1
        export.intenti = 0
        export.info = str(config["title"])
        export.encrypt = 0
        export.presentation = 0
        export.compress = 1
        export.compressmtd = 2
        export.quality = 0
        export.resolution = 300
        export.downsample = 0
        export.useDocBleeds = 1
        export.cropMarks = 0
        export.bleedMarks = 0
        export.registrationMarks = 0
        export.colorMarks = 0
        export.docInfoMarks = 0
        export.save()

        report.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "scribus_version": getattr(scribus, "SCRIBUS_VERSION", "unknown"),
                    "object_count": len(scribus.getAllObjects()),
                    "placement": {
                        "object": artwork_name,
                        "imported_size_mm": [initial_width, initial_height],
                        "scale_factor": scale_factor,
                        "position_mm": [actual_x, actual_y],
                        "placed_size_mm": [placed_width, placed_height],
                        "page_size_mm": [page_width_mm, page_height_mm],
                        "bleed_mm": bleed,
                    },
                    "svg": str(svg),
                    "sla": str(sla),
                    "pdf": str(pdf_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        report.write_text(
            json.dumps({"status": "error", "traceback": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        if scribus.haveDoc():
            scribus.closeDoc()


main()
