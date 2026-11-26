#!/usr/bin/env python3
"""
Xelvic Camera Engine - Package Validator CLI
README Section 6, 21 Step 3, & Section 25.
Validates package structure, manifest schema, asset hashes, file sizes, and signatures.
"""

import sys
import os
import json
import hashlib
import struct

try:
    import jsonschema
except ImportError:
    # Fallback to manual verification if jsonschema is not installed in the current environment
    jsonschema = None

XFILTER3_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "XFilter3Manifest",
    "type": "object",
    "required": ["id", "name"],
    "properties": {
        "schema": {"type": "string"},
        "id": {"type": "string"},
        "name": {"type": "string"},
        "version": {"type": "string"},
        "description": {"type": "string"},
        "preview": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "role": {"type": "string"}
            }
        },
        "requires": {
            "type": "object",
            "properties": {
                "runtime": {"type": "integer"},
                "features": {"type": "array", "items": {"type": "string"}},
                "faces": {"type": "integer"}
            }
        },
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "path"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["image", "texture", "model", "glb", "mask", "lut"]},
                    "path": {"type": "string"},
                    "sha256": {"type": "string"},
                    "bytes": {"type": "integer"},
                    "colorSpace": {"type": "string"},
                    "mipmaps": {"type": "boolean"},
                    "maxSize": {"type": "integer"}
                }
            }
        },
        "params": {"type": "object"},
        "anchors": {"type": "object"},
        "materials": {"type": "object"},
        "lighting": {"type": "object"},
        "nodes": {"type": "array"},
        "signals": {"type": "object"},
        "behaviors": {"type": "array"},
        "lifecycle": {"type": "object"},
        "tiers": {"type": ["array", "object"]},
        "minTier": {"type": "string"},
        "photo": {"type": "object"},
        "budgets": {"type": "object"}
    },
    "additionalProperties": True
}

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def validate_asset_signature(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    with open(filepath, "rb") as f:
        header = f.read(12)
    
    if ext == ".png":
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return False, f"Invalid PNG magic signature: {filepath}"
    elif ext in [".jpg", ".jpeg"]:
        if header[:3] != b"\xff\xd8\xff":
            return False, f"Invalid JPEG magic signature: {filepath}"
    elif ext == ".glb":
        if len(header) < 12:
            return False, f"GLB file too small for header: {filepath}"
        magic, version, length = struct.unpack("<III", header[:12])
        if magic != 0x46546C67: # 'glTF'
            return False, f"Invalid GLB magic (expected 0x46546C67): {filepath}"
        if version != 2:
            return False, f"Unsupported GLB version {version}: {filepath}"
    return True, ""

def validate_package(package_dir):
    if not os.path.isdir(package_dir):
        print(f"Error: {package_dir} is not a valid directory")
        return 1

    manifest_path = os.path.join(package_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"Error: Missing manifest.json in {package_dir}")
        return 1

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse manifest.json: {e}")
        return 1

    # Schema validation
    if jsonschema:
        try:
            jsonschema.validate(instance=manifest, schema=XFILTER3_SCHEMA)
        except jsonschema.ValidationError as e:
            print(f"Schema Validation Error: {e.message}")
            return 1
    else:
        # Fallback check
        if "id" not in manifest or "name" not in manifest:
            print("Error: manifest.json missing required 'id' or 'name'")
            return 1

    # Validate assets
    assets = manifest.get("assets", [])
    total_bytes = 0
    for a in assets:
        asset_path = a.get("path")
        if not asset_path:
            print("Error: Asset entry missing 'path'")
            return 1

        full_path = os.path.join(package_dir, asset_path)
        if not os.path.isfile(full_path):
            print(f"Error: Declared asset file does not exist: {full_path}")
            return 1

        actual_size = os.path.getsize(full_path)
        total_bytes += actual_size

        declared_bytes = a.get("bytes")
        if declared_bytes is not None and declared_bytes > 0 and actual_size != declared_bytes:
            print(f"Error: Asset byte count mismatch for {asset_path}: expected {declared_bytes}, got {actual_size}")
            return 1

        declared_sha = a.get("sha256")
        if declared_sha:
            actual_sha = sha256_file(full_path)
            if actual_sha.lower() != declared_sha.lower():
                print(f"Error: SHA-256 hash mismatch for {asset_path}: expected {declared_sha}, got {actual_sha}")
                return 1

        ok, sig_err = validate_asset_signature(full_path)
        if not ok:
            print(f"Error: {sig_err}")
            return 1

    # Validate post nodes (README Section 5.9)
    allowlisted_post_effects = {"colorGrade", "blur", "skinSoften"}
    forbidden_post_keys = {"shader", "vertexShader", "fragmentShader", "glsl", "script", "code", "source"}
    for n in manifest.get("nodes", []):
        if n.get("type") == "post":
            for fk in forbidden_post_keys:
                if fk in n or fk in (n.get("params") or {}):
                    print(f"Error: Post node '{n.get('id')}' contains forbidden custom shader/script key '{fk}'")
                    return 1
            effect = n.get("effect")
            if effect not in allowlisted_post_effects:
                print(f"Error: Post node '{n.get('id')}' uses unsupported effect '{effect}'")
                return 1

    print(f"Package '{manifest.get('id')}' ({manifest.get('name')}) validated successfully.")
    print(f"Validated {len(assets)} assets, total uncompressed size: {total_bytes} bytes.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_package.py <package_dir>")
        sys.exit(1)

    sys.exit(validate_package(sys.argv[1]))
