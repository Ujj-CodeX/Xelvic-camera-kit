#!/usr/bin/env python3
"""
Generates reference packages for Step 5 Generality Gate:
- round_glasses (ref_glasses)
- multi_prop (ref_multi_prop)
- ref_sticker_2d
- ref_face_paint
Generates valid glTF 2.0 binary (.glb) files, valid PNG images, and valid JPEG thumbnails.
Calculates byte counts and SHA-256 digests matching manifest definitions.
"""

import os
import json
import struct
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def make_minimal_png(width=16, height=16, color=(255, 200, 50, 255)) -> bytes:
    """Create a minimal uncompressed valid PNG file."""
    import zlib
    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data))
    ihdr = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc

    # IDAT
    raw_scanlines = bytearray()
    for _ in range(height):
        raw_scanlines.append(0) # filter type none
        for _ in range(width):
            raw_scanlines.extend(color)
    compressed = zlib.compress(bytes(raw_scanlines))
    idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed))
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + idat_crc

    # IEND
    iend_crc = struct.pack('>I', zlib.crc32(b'IEND'))
    iend = struct.pack('>I', 0) + b'IEND' + iend_crc

    return sig + ihdr + idat + iend

def make_minimal_jpeg() -> bytes:
    """Create a minimal valid 1x1 JPEG image."""
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F,
        0x00, 0x7F, 0x00, 0xBF, 0xFF, 0xD9
    ])

def make_minimal_glb(node_names=["mesh_node"], has_normals=True, has_texcoords=True) -> bytes:
    """Create a valid glTF 2.0 binary (.glb) container with cube/quad geometry."""
    # 4 vertices: positions (vec3 float), normals (vec3 float), texcoords (vec2 float)
    # 2 triangles (6 indices, uint16)
    positions = [
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.5,  0.5, 0.0,
        -0.5,  0.5, 0.0
    ]
    normals = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0
    ]
    uvs = [
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    ]
    indices = [
        0, 1, 2,
        0, 2, 3
    ]

    pos_bytes = struct.pack(f'<{len(positions)}f', *positions)
    norm_bytes = struct.pack(f'<{len(normals)}f', *normals)
    uv_bytes = struct.pack(f'<{len(uvs)}f', *uvs)
    idx_bytes = struct.pack(f'<{len(indices)}H', *indices)

    # Pad buffers to 4 bytes
    def pad4(b: bytes) -> bytes:
        pad = (4 - (len(b) % 4)) % 4
        return b + (b'\x00' * pad)

    pos_bytes = pad4(pos_bytes)
    norm_bytes = pad4(norm_bytes)
    uv_bytes = pad4(uv_bytes)
    idx_bytes = pad4(idx_bytes)

    bin_data = pos_bytes + norm_bytes + uv_bytes + idx_bytes

    offset_pos = 0
    len_pos = len(pos_bytes)
    offset_norm = offset_pos + len_pos
    len_norm = len(norm_bytes)
    offset_uv = offset_norm + len_norm
    len_uv = len(uv_bytes)
    offset_idx = offset_uv + len_uv
    len_idx = len(idx_bytes)

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "Xelvic GLB Builder"},
        "scenes": [{"nodes": list(range(len(node_names)))}],
        "scene": 0,
        "nodes": [],
        "meshes": [
            {
                "name": "Mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2
                        },
                        "indices": 3,
                        "mode": 4
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126, # FLOAT
                "count": 4,
                "type": "VEC3",
                "max": [0.5, 0.5, 0.0],
                "min": [-0.5, -0.5, 0.0]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "max": [0.0, 0.0, 1.0],
                "min": [0.0, 0.0, 1.0]
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC2",
                "max": [1.0, 1.0],
                "min": [0.0, 0.0]
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5123, # UNSIGNED_SHORT
                "count": 6,
                "type": "SCALAR",
                "max": [3],
                "min": [0]
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_pos, "byteLength": len_pos, "target": 34962},
            {"buffer": 0, "byteOffset": offset_norm, "byteLength": len_norm, "target": 34962},
            {"buffer": 0, "byteOffset": offset_uv, "byteLength": len_uv, "target": 34962},
            {"buffer": 0, "byteOffset": offset_idx, "byteLength": len_idx, "target": 34963}
        ],
        "buffers": [
            {"byteLength": len(bin_data)}
        ]
    }

    for i, name in enumerate(node_names):
        gltf_dict["nodes"].append({
            "name": name,
            "mesh": 0,
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0]
        })

    json_str = json.dumps(gltf_dict, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    # Pad json_bytes to 4 bytes with spaces (0x20)
    pad_json = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b' ' * pad_json

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

    header = struct.pack('<III', 0x46546C67, 2, total_len)
    chunk0_header = struct.pack('<II', len(json_bytes), 0x4E4F534A) # JSON
    chunk1_header = struct.pack('<II', len(bin_data), 0x004E4942) # BIN

    return header + chunk0_header + json_bytes + chunk1_header + bin_data

def build_package(dest_dir, manifest_dict, assets_dict, preview_bytes):
    os.makedirs(os.path.join(dest_dir, "assets"), exist_ok=True)
    
    # Write preview
    preview_path = os.path.join(dest_dir, "preview.jpg")
    with open(preview_path, "wb") as f:
        f.write(preview_bytes)

    # Write assets and record hashes/bytes
    assets_manifest = []
    for asset_rel_path, (asset_id, asset_type, asset_data) in assets_dict.items():
        full_asset_path = os.path.join(dest_dir, asset_rel_path)
        os.makedirs(os.path.dirname(full_asset_path), exist_ok=True)
        with open(full_asset_path, "wb") as f:
            f.write(asset_data)
        
        sha = sha256_bytes(asset_data)
        size = len(asset_data)
        assets_manifest.append({
            "id": asset_id,
            "type": asset_type,
            "path": asset_rel_path,
            "sha256": sha,
            "bytes": size
        })

    manifest_dict["assets"] = assets_manifest
    manifest_path = os.path.join(dest_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    print(f"Built package at {dest_dir} (id: {manifest_dict['id']})")

def main():
    base_filters_dir = "app/src/main/assets/filters"
    preview = make_minimal_jpeg()

    # 1. ref_glasses (round_glasses)
    glasses_glb = make_minimal_glb(node_names=["glasses_frame", "lens_left", "lens_right"])
    glasses_manifest = {
        "schema": "xfilter/3",
        "id": "round_glasses",
        "name": "Round Glasses",
        "version": "1.0.0",
        "description": "Reference 3D glasses with two anchors, alpha-blended lenses, and head occluder",
        "requires": {
            "runtime": 3,
            "features": [
                "node.model@1",
                "material.lambert@1",
                "material.unlit@1",
                "blend.alpha@1",
                "blend.opaque@1",
                "occluder.ellipsoid@1"
            ],
            "faces": 1
        },
        "materials": {
            "frame": {
                "template": "lambert@1",
                "params": {"color": [0.1, 0.1, 0.1]},
                "blend": "opaque",
                "depthTest": True,
                "depthWrite": True
            },
            "lens": {
                "template": "unlit@1",
                "params": {"color": [0.6, 0.8, 1.0], "opacity": 0.25},
                "blend": "alpha",
                "depthTest": True,
                "depthWrite": False
            }
        },
        "lighting": {
            "space": "camera",
            "ambient": [0.4, 0.4, 0.4],
            "lights": [
                {
                    "dirToLight": [0.0, 0.6, 0.8],
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 0.7
                }
            ]
        },
        "nodes": [
            {
                "id": "head",
                "type": "occluder",
                "anchor": "head_center",
                "shape": "ellipsoid",
                "radiiFW": [0.56, 0.66, 0.68],
                "segments": [16, 24]
            },
            {
                "id": "glasses",
                "type": "model",
                "anchor": "eye_center",
                "asset": "g",
                "transform": {
                    "position": [0.0, 0.0, 0.1],
                    "scale": 1.0,
                    "rotationFollow": "full"
                },
                "tags": ["frame", "lens"]
            }
        ],
        "preview": {
            "path": "preview.jpg",
            "role": "thumbnail"
        }
    }
    build_package(
        os.path.join(base_filters_dir, "round_glasses"),
        glasses_manifest,
        {"assets/glasses.glb": ("g", "model", glasses_glb)},
        preview
    )

    # 2. ref_multi_prop
    hat_glb = make_minimal_glb(node_names=["top_hat_mesh"])
    mustache_png = make_minimal_png(width=32, height=16, color=(50, 40, 30, 255))
    monocle_png = make_minimal_png(width=16, height=16, color=(200, 220, 255, 200))
    multi_prop_manifest = {
        "schema": "xfilter/3",
        "id": "multi_prop",
        "name": "Multi Prop",
        "version": "1.0.0",
        "description": "Reference multi-prop filter demonstrating multiple nodes, custom anchors, tap triggers, and event emission",
        "requires": {
            "runtime": 3,
            "features": [
                "node.group@1",
                "node.sprite@1",
                "node.model@1",
                "material.unlit@1",
                "material.lambert@1",
                "blend.alpha@1",
                "behavior.trigger@1"
            ],
            "faces": 1
        },
        "anchors": {
            "brow_custom": {
                "landmarks": [
                    {"id": 70, "weight": 0.5},
                    {"id": 105, "weight": 0.5}
                ],
                "offsetFW": [0.0, 0.05, 0.02]
            }
        },
        "materials": {
            "mat_hat": {
                "template": "lambert@1",
                "params": {"color": [0.15, 0.15, 0.2]},
                "blend": "opaque"
            },
            "mat_mustache": {
                "template": "unlit@1",
                "params": {"map": "mustache_tex"},
                "blend": "alpha"
            },
            "mat_monocle": {
                "template": "unlit@1",
                "params": {"map": "monocle_tex"},
                "blend": "alpha"
            }
        },
        "lighting": {
            "space": "camera",
            "ambient": [0.35, 0.35, 0.35],
            "lights": [
                {
                    "dirToLight": [0.0, 0.8, 0.6],
                    "color": [1.0, 0.95, 0.9],
                    "intensity": 0.8
                }
            ]
        },
        "nodes": [
            {
                "id": "prop_top_hat",
                "type": "model",
                "anchor": "forehead",
                "asset": "hat_model",
                "transform": {
                    "position": [0.0, 0.25, -0.05],
                    "scale": 0.8,
                    "rotationFollow": "full"
                },
                "tags": ["prop", "headwear"]
            },
            {
                "id": "prop_mustache",
                "type": "sprite",
                "anchor": "mouth_center",
                "material": "mat_mustache",
                "asset": "mustache_tex",
                "size": {
                    "widthFW": 0.35,
                    "heightFW": 0.12
                },
                "transform": {
                    "position": [0.0, 0.08, 0.02],
                    "rotationFollow": "full"
                },
                "tags": ["prop", "facial_hair"]
            },
            {
                "id": "prop_monocle",
                "type": "sprite",
                "anchor": "brow_custom",
                "material": "mat_monocle",
                "asset": "monocle_tex",
                "size": {
                    "widthFW": 0.18,
                    "heightFW": 0.18
                },
                "transform": {
                    "position": [0.0, -0.02, 0.02],
                    "rotationFollow": "full"
                },
                "visible": True,
                "tags": ["prop", "eyewear"]
            }
        ],
        "behaviors": [
            {
                "id": "tap_toggle_monocle",
                "type": "trigger",
                "on": "tap",
                "cooldownMs": 250,
                "onFire": [
                    {
                        "action": "toggleVisible",
                        "target": "node:prop_monocle"
                    },
                    {
                        "action": "emit",
                        "clip": "custom_multi_prop_tapped"
                    }
                ]
            }
        ],
        "preview": {
            "path": "preview.jpg",
            "role": "thumbnail"
        }
    }
    build_package(
        os.path.join(base_filters_dir, "multi_prop"),
        multi_prop_manifest,
        {
            "assets/hat.glb": ("hat_model", "model", hat_glb),
            "assets/mustache.png": ("mustache_tex", "image", mustache_png),
            "assets/monocle.png": ("monocle_tex", "image", monocle_png)
        },
        preview
    )

    # 3. ref_sticker_2d
    star_png = make_minimal_png(width=32, height=32, color=(255, 230, 40, 255))
    sticker_manifest = {
        "schema": "xfilter/3",
        "id": "ref_sticker_2d",
        "name": "Star Cheek",
        "version": "1.0.0",
        "requires": {
            "runtime": 3,
            "features": ["node.sprite@1", "material.unlit@1", "blend.alpha@1"],
            "faces": 1
        },
        "materials": {
            "mat_star": {
                "template": "unlit@1",
                "params": {"map": "star"},
                "blend": "alpha"
            }
        },
        "nodes": [
            {
                "id": "star_node",
                "type": "sprite",
                "anchor": "cheek_left",
                "material": "mat_star",
                "asset": "star",
                "size": {"widthFW": 0.18, "heightFW": 0.18},
                "transform": {"position": [0.0, 0.0, 0.02], "rotationFollow": "full"}
            }
        ],
        "preview": {
            "path": "preview.jpg",
            "role": "thumbnail"
        }
    }
    build_package(
        os.path.join(base_filters_dir, "ref_sticker_2d"),
        sticker_manifest,
        {"assets/star.png": ("star", "image", star_png)},
        preview
    )

    # 4. ref_face_paint (Step 8 Face-Mesh Texture Conformance Package)
    import math, zlib

    def make_face_paint_png(width=128, height=128) -> bytes:
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
        ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data))
        ihdr = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc
        raw_scanlines = bytearray()
        for y in range(height):
            raw_scanlines.append(0)
            v = (y + 0.5) / height
            for x in range(width):
                u = (x + 0.5) / width
                # Forehead diamond/crown motif around (0.50, 0.16)
                d_forehead = math.hypot((u - 0.50) / 0.14, (v - 0.17) / 0.06)
                # Right cheek warpaint stripes around (0.24, 0.54)
                d_rcheek = math.hypot((u - 0.25) / 0.11, (v - 0.54) / 0.08)
                # Left cheek warpaint stripes around (0.76, 0.54)
                d_lcheek = math.hypot((u - 0.75) / 0.11, (v - 0.54) / 0.08)
                if d_forehead < 1.0:
                    alpha = int(230 * (1.0 - d_forehead * 0.4))
                    raw_scanlines.extend((255, 195, 40, alpha))
                elif d_rcheek < 1.0 or d_lcheek < 1.0:
                    stripe = 0.5 + 0.5 * math.sin(v * 95.0)
                    alpha = int(225 * (0.45 + 0.55 * stripe))
                    raw_scanlines.extend((235, 55, 95, alpha))
                else:
                    # Subtle base tint across face oval
                    raw_scanlines.extend((220, 60, 100, 96))
        compressed = zlib.compress(bytes(raw_scanlines))
        idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed))
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + idat_crc
        iend_crc = struct.pack('>I', zlib.crc32(b'IEND'))
        iend = struct.pack('>I', 0) + b'IEND' + iend_crc
        return sig + ihdr + idat + iend

    def make_face_mask_png(width=128, height=128) -> bytes:
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
        ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data))
        ihdr = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc
        raw_scanlines = bytearray()

        def smoothstep(edge0, edge1, val):
            t = max(0.0, min(1.0, (val - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        for y in range(height):
            raw_scanlines.append(0)
            v = (y + 0.5) / height
            for x in range(width):
                u = (x + 0.5) / width
                # Outer face oval feathering
                d_oval = math.hypot((u - 0.50) / 0.42, (v - 0.50) / 0.45)
                oval_mask = 1.0 - smoothstep(0.85, 1.02, d_oval)
                # Right eye hole (0.3496, 0.3780)
                d_reye = math.hypot((u - 0.3496) / 0.072, (v - 0.3780) / 0.036)
                reye_mask = smoothstep(0.82, 1.15, d_reye)
                # Left eye hole (0.6504, 0.3780)
                d_leye = math.hypot((u - 0.6504) / 0.072, (v - 0.3780) / 0.036)
                leye_mask = smoothstep(0.82, 1.15, d_leye)
                # Lips and mouth interior hole (0.5000, 0.6938)
                d_mouth = math.hypot((u - 0.5000) / 0.092, (v - 0.6938) / 0.050)
                mouth_mask = smoothstep(0.82, 1.15, d_mouth)

                m = int(round(255.0 * oval_mask * min(reye_mask, leye_mask, mouth_mask)))
                raw_scanlines.extend((m, m, m, 255))
        compressed = zlib.compress(bytes(raw_scanlines))
        idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed))
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + idat_crc
        iend_crc = struct.pack('>I', zlib.crc32(b'IEND'))
        iend = struct.pack('>I', 0) + b'IEND' + iend_crc
        return sig + ihdr + idat + iend

    paint_png = make_face_paint_png(width=128, height=128)
    paint_mask_png = make_face_mask_png(width=128, height=128)
    face_paint_manifest = {
        "schema": "xfilter/3",
        "id": "ref_face_paint",
        "name": "Face Paint Reference",
        "version": "1.0.0",
        "description": "Reference 468-vertex canonical UV face paint with eye, lip, and mouth-interior protection mask",
        "requires": {
            "runtime": 3,
            "features": ["node.faceMesh@1", "material.faceTexture@1", "blend.alpha@1"],
            "faces": 1
        },
        "materials": {
            "mat_paint": {
                "template": "faceTexture@1",
                "params": {
                    "map": "paint_texture",
                    "mask": "paint_mask",
                    "uvLayout": "mediapipe_canonical_468_v1",
                    "opacity": 0.9
                },
                "blend": "alpha",
                "depthTest": True,
                "depthWrite": False
            }
        },
        "nodes": [
            {
                "id": "face_paint_node",
                "type": "faceMesh",
                "material": "mat_paint",
                "region": "face_oval",
                "renderOrder": 1
            }
        ],
        "tiers": {
            "A": {"textureMaxSize": 1024},
            "B": {"textureMaxSize": 512},
            "C": {"textureMaxSize": 256}
        },
        "preview": {
            "path": "preview.jpg",
            "role": "thumbnail"
        }
    }
    build_package(
        os.path.join(base_filters_dir, "ref_face_paint"),
        face_paint_manifest,
        {
            "assets/paint.png": ("paint_texture", "texture", paint_png),
            "assets/paint_mask.png": ("paint_mask", "mask", paint_mask_png)
        },
        preview
    )

    print("All reference packages successfully built.")

if __name__ == "__main__":
    main()

