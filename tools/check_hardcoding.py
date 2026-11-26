#!/usr/bin/env python3
"""
CI Hardcoding Enforcement Script (README Section 19.1)
Scans engine source code (app/src/main/java/com/example/engine) for known filter IDs,
package node names, and package asset names.
Fails with exit code 1 if any filter-specific hardcoding is detected.
"""

import os
import sys
import re

FORBIDDEN_IDENTIFIERS = [
    # Filter IDs
    "sunflower_forehead",
    "periwinkle_forehead",
    "ref_sunflower",
    "round_glasses",
    "ref_glasses",
    "multi_prop",
    "ref_multi_prop",
    "star_cheek",
    "ref_sticker_2d",
    "ref_face_paint",
    "ref_beauty",
    # Specific filter node names
    "sunflower_pivot",
    "sunflower_petals",
    "sunflower_disk",
    "sunflower_foliage",
    "periwinkle_pivot",
    "periwinkle_petals",
    "prop_top_hat",
    "prop_mustache",
    "prop_monocle",
    "glasses_frame",
    "furry_glasses_filter",
    "furry_glasses",
    "no_makeup_beauty",
    "light_lens",
    "face_subtle_morph",
    "skin_bilateral_smooth",
    "natural_lip_tint",
    "soft_cheek_blush",
    "light_soft_focus",
    "light_tone_grade",
    "dreamy_soft_focus",
    "natural_warm_grade",
    "portrait_vignette",
    # Specific filter asset names
    "sunflower_forehead.glb",
    "periwinkle_forehead.glb",
    "furry_plush_pink_srgb.png",
    "glasses.glb",
    "hat.glb",
    "mustache.png",
    "monocle.png",
    "paint.png"
]

def scan_engine_sources(engine_dir):
    if not os.path.exists(engine_dir):
        print(f"Error: Engine directory not found: {engine_dir}")
        return 1

    violations = []
    pattern = re.compile(r'\b(' + '|'.join(re.escape(i) for i in FORBIDDEN_IDENTIFIERS) + r')\b', re.IGNORECASE)

    for root, _, files in os.walk(engine_dir):
        for file in files:
            if not file.endswith((".kt", ".java")):
                continue
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    # Ignore comments if any, or match raw line
                    match = pattern.search(line)
                    if match:
                        violations.append({
                            "file": filepath,
                            "line": line_num,
                            "token": match.group(0),
                            "text": line.strip()
                        })

    if violations:
        print("=" * 60)
        print("CI HARDCODING CHECK FAILED: Found filter-specific identifiers in engine sources!")
        print("Engine sources must remain general and data-driven.")
        print("=" * 60)
        for v in violations:
            print(f"  {v['file']}:{v['line']} -> Found '{v['token']}': {v['text']}")
        print("=" * 60)
        return 1

    print("CI Hardcoding Check: PASSED. Zero filter-specific identifiers found in engine sources.")
    return 0

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "app/src/main/java/com/example/engine"
    sys.exit(scan_engine_sources(target))
