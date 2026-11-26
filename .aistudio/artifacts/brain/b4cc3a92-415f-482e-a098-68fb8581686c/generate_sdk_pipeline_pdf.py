#!/usr/bin/env python3
"""
Generates a comprehensive, publication-grade multi-page PDF architecture & pipeline manual
for the Xelvic AR/Beauty SDK Engine (`Xelvic_SDK_Engine_Architecture_and_Pipeline_Guide.pdf`).
Uses pure Python 3 standard library to emit a valid, multi-page vector PDF 1.4 document
with embedded Type1 fonts (Helvetica, Helvetica-Bold, Courier, Courier-Bold), vector diagrams,
tables, callout boxes, mathematical equations, and end-to-end use-case walkthroughs.
"""
import os
import textwrap

PAGE_W = 595.28  # A4 width in points
PAGE_H = 841.89  # A4 height in points
MARGIN_L = 42.0
MARGIN_R = 42.0
MARGIN_T = 52.0
MARGIN_B = 48.0
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R


def pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


class PdfCanvas:
    def __init__(self):
        self.pages = []
        self.current_cmds = []
        self.y = PAGE_H - MARGIN_T
        self.page_num = 0
        self.section_title = "Xelvic AR & Beauty SDK Engine — Deep Architecture & Pipeline Manual"
        self.new_page()

    def new_page(self):
        if self.current_cmds:
            self._draw_footer()
            self.pages.append("\n".join(self.current_cmds))
        self.current_cmds = []
        self.page_num += 1
        self.y = PAGE_H - MARGIN_T
        if self.page_num > 1:
            self._draw_header()

    def ensure_space(self, needed_h: float):
        if self.y - needed_h < MARGIN_B + 14.0:
            self.new_page()

    def _draw_header(self):
        cmds = self.current_cmds
        # Top header bar
        cmds.append("0.09 0.14 0.24 rg")
        cmds.append(f"{MARGIN_L:.2f} {PAGE_H - 36:.2f} {CONTENT_W:.2f} 18.00 re f")
        cmds.append("1 1 1 rg")
        cmds.append(f"BT /F2 8.5 Tf {MARGIN_L + 8:.2f} {PAGE_H - 30:.2f} Td ({pdf_escape(self.section_title)}) Tj ET")
        cmds.append(f"BT /F1 8.0 Tf {PAGE_W - MARGIN_R - 78:.2f} {PAGE_H - 30:.2f} Td (Snap Parity v2.5) Tj ET")
        self.y = PAGE_H - MARGIN_T - 6.0

    def _draw_footer(self):
        cmds = self.current_cmds
        cmds.append("0.78 0.81 0.86 RG 0.6 w")
        cmds.append(f"{MARGIN_L:.2f} {MARGIN_B - 10:.2f} m {PAGE_W - MARGIN_R:.2f} {MARGIN_B - 10:.2f} l S")
        cmds.append("0.35 0.38 0.44 rg")
        cmds.append(
            f"BT /F1 8.0 Tf {MARGIN_L:.2f} {MARGIN_B - 22:.2f} Td "
            f"(Xelvic AR Filter & Beauty Engine SDK — Deep Pipeline Reference & Use-Case Guide) Tj ET"
        )
        cmds.append(
            f"BT /F2 8.5 Tf {PAGE_W - MARGIN_R - 45:.2f} {MARGIN_B - 22:.2f} Td "
            f"(Page {self.page_num}) Tj ET"
        )

    def draw_cover_banner(self, title: str, subtitle: str, meta_lines: list):
        self.ensure_space(132.0)
        cmds = self.current_cmds
        box_h = 122.0
        top_y = self.y
        bottom_y = top_y - box_h
        # Deep navy banner
        cmds.append("0.07 0.11 0.21 rg")
        cmds.append(f"{MARGIN_L:.2f} {bottom_y:.2f} {CONTENT_W:.2f} {box_h:.2f} re f")
        # Accent cyan left stripe
        cmds.append("0.00 0.72 0.84 rg")
        cmds.append(f"{MARGIN_L:.2f} {bottom_y:.2f} 6.00 {box_h:.2f} re f")
        # Title
        cmds.append("1 1 1 rg")
        cmds.append(f"BT /F2 17.5 Tf {MARGIN_L + 18:.2f} {top_y - 28:.2f} Td ({pdf_escape(title)}) Tj ET")
        cmds.append("0.62 0.88 0.96 rg")
        cmds.append(f"BT /F2 10.5 Tf {MARGIN_L + 18:.2f} {top_y - 46:.2f} Td ({pdf_escape(subtitle)}) Tj ET")
        cmds.append("0.88 0.91 0.96 rg")
        cy = top_y - 66.0
        for line in meta_lines:
            cmds.append(f"BT /F1 8.8 Tf {MARGIN_L + 18:.2f} {cy:.2f} Td ({pdf_escape(line)}) Tj ET")
            cy -= 13.5
        self.y = bottom_y - 14.0

    def section_heading(self, num_str: str, title: str):
        self.section_title = f"Section {num_str}: {title}"
        self.ensure_space(34.0)
        self.y -= 4.0
        cmds = self.current_cmds
        box_h = 22.0
        by = self.y - box_h
        cmds.append("0.11 0.19 0.34 rg")
        cmds.append(f"{MARGIN_L:.2f} {by:.2f} {CONTENT_W:.2f} {box_h:.2f} re f")
        cmds.append("0.00 0.75 0.85 rg")
        cmds.append(f"{MARGIN_L:.2f} {by:.2f} 4.50 {box_h:.2f} re f")
        cmds.append("1 1 1 rg")
        cmds.append(
            f"BT /F2 11.2 Tf {MARGIN_L + 12:.2f} {by + 6.5:.2f} Td "
            f"({pdf_escape(num_str + '  ' + title)}) Tj ET"
        )
        self.y = by - 9.0

    def sub_heading(self, title: str):
        self.ensure_space(22.0)
        self.y -= 2.0
        cmds = self.current_cmds
        cmds.append("0.09 0.28 0.48 rg")
        cmds.append(f"BT /F2 10.0 Tf {MARGIN_L:.2f} {self.y - 10:.2f} Td ({pdf_escape(title)}) Tj ET")
        cmds.append("0.75 0.82 0.90 RG 0.7 w")
        cmds.append(f"{MARGIN_L:.2f} {self.y - 13.5:.2f} m {PAGE_W - MARGIN_R:.2f} {self.y - 13.5:.2f} l S")
        self.y -= 20.0

    def paragraph(self, text: str, font="F1", size=9.0, leading=12.6, indent=0.0, color=(0.14, 0.16, 0.20)):
        max_chars = int((CONTENT_W - indent - 6.0) / (size * 0.49))
        lines = textwrap.wrap(text, width=max_chars)
        self.ensure_space(len(lines) * leading + 4.0)
        cmds = self.current_cmds
        cmds.append(f"{color[0]:.2f} {color[1]:.2f} {color[2]:.2f} rg")
        for line in lines:
            self.ensure_space(leading + 2.0)
            cmds.append(f"BT /{font} {size:.1f} Tf {MARGIN_L + indent:.2f} {self.y - size:.2f} Td ({pdf_escape(line)}) Tj ET")
            self.y -= leading
        self.y -= 3.0

    def bullet(self, label: str, body: str, size=8.8, leading=12.2, indent=8.0):
        full = f"{label}: {body}" if label else body
        max_chars = int((CONTENT_W - indent - 14.0) / (size * 0.485))
        lines = textwrap.wrap(full, width=max_chars)
        self.ensure_space(len(lines) * leading + 2.0)
        cmds = self.current_cmds
        for i, line in enumerate(lines):
            self.ensure_space(leading + 2.0)
            if i == 0:
                cmds.append("0.00 0.55 0.72 rg")
                cmds.append(f"BT /F2 {size:.1f} Tf {MARGIN_L + indent:.2f} {self.y - size:.2f} Td (*) Tj ET")
            cmds.append("0.14 0.16 0.20 rg")
            font = "F2" if (i == 0 and label and len(line) <= len(label) + 2) else "F1"
            cmds.append(f"BT /{font} {size:.1f} Tf {MARGIN_L + indent + 10:.2f} {self.y - size:.2f} Td ({pdf_escape(line)}) Tj ET")
            self.y -= leading
        self.y -= 1.5

    def code_block(self, title: str, code_lines: list, size=7.8, leading=10.6):
        box_h = len(code_lines) * leading + (22.0 if title else 10.0)
        self.ensure_space(box_h + 8.0)
        cmds = self.current_cmds
        top_y = self.y
        bottom_y = top_y - box_h
        # Light slate background + border
        cmds.append("0.95 0.96 0.98 rg")
        cmds.append("0.76 0.80 0.87 RG 0.7 w")
        cmds.append(f"{MARGIN_L:.2f} {bottom_y:.2f} {CONTENT_W:.2f} {box_h:.2f} re B")
        cy = top_y - 12.0
        if title:
            cmds.append("0.18 0.25 0.38 rg")
            cmds.append(f"{MARGIN_L:.2f} {top_y - 15.0:.2f} {CONTENT_W:.2f} 15.00 re f")
            cmds.append("1 1 1 rg")
            cmds.append(f"BT /F2 8.2 Tf {MARGIN_L + 8:.2f} {top_y - 11.0:.2f} Td ({pdf_escape(title)}) Tj ET")
            cy = top_y - 25.5
        cmds.append("0.10 0.13 0.18 rg")
        for raw_line in code_lines:
            line = raw_line[:104]
            cmds.append(f"BT /F3 {size:.1f} Tf {MARGIN_L + 8:.2f} {cy:.2f} Td ({pdf_escape(line)}) Tj ET")
            cy -= leading
        self.y = bottom_y - 7.0

    def table(self, headers: list, col_widths: list, rows: list, row_h=16.5, font_size=8.0):
        total_h = (len(rows) + 1) * row_h + 8.0
        self.ensure_space(min(total_h, 120.0))
        cmds = self.current_cmds
        # Draw header row
        hy = self.y - row_h
        cmds.append("0.14 0.22 0.36 rg")
        cmds.append(f"{MARGIN_L:.2f} {hy:.2f} {CONTENT_W:.2f} {row_h:.2f} re f")
        cmds.append("1 1 1 rg")
        cx = MARGIN_L
        for h_text, w in zip(headers, col_widths):
            cmds.append(f"BT /F2 {font_size:.1f} Tf {cx + 5:.2f} {hy + 4.8:.2f} Td ({pdf_escape(h_text)}) Tj ET")
            cx += w
        self.y = hy

        for r_idx, row in enumerate(rows):
            if self.y - row_h < MARGIN_B + 14.0:
                self.new_page()
                # Redraw header on continuation page
                hy = self.y - row_h
                cmds = self.current_cmds
                cmds.append("0.14 0.22 0.36 rg")
                cmds.append(f"{MARGIN_L:.2f} {hy:.2f} {CONTENT_W:.2f} {row_h:.2f} re f")
                cmds.append("1 1 1 rg")
                cx = MARGIN_L
                for h_text, w in zip(headers, col_widths):
                    cmds.append(f"BT /F2 {font_size:.1f} Tf {cx + 5:.2f} {hy + 4.8:.2f} Td ({pdf_escape(h_text)}) Tj ET")
                    cx += w
                self.y = hy

            ry = self.y - row_h
            cmds = self.current_cmds
            if r_idx % 2 == 0:
                cmds.append("0.96 0.97 0.99 rg")
            else:
                cmds.append("1.00 1.00 1.00 rg")
            cmds.append(f"{MARGIN_L:.2f} {ry:.2f} {CONTENT_W:.2f} {row_h:.2f} re f")
            cmds.append("0.82 0.85 0.90 RG 0.4 w")
            cmds.append(f"{MARGIN_L:.2f} {ry:.2f} {CONTENT_W:.2f} {row_h:.2f} re S")
            cx = MARGIN_L
            for c_idx, (cell, w) in enumerate(zip(row, col_widths)):
                max_c = max(6, int((w - 8.0) / (font_size * 0.48)))
                clipped = cell if len(cell) <= max_c else (cell[: max_c - 2] + "..")
                font = "F2" if c_idx == 0 else "F1"
                cmds.append("0.12 0.15 0.20 rg")
                cmds.append(f"BT /{font} {font_size:.1f} Tf {cx + 5:.2f} {ry + 4.8:.2f} Td ({pdf_escape(clipped)}) Tj ET")
                cx += w
            self.y = ry
        self.y -= 7.0

    def draw_pipeline_flow_diagram(self):
        """Draws a crisp vector block diagram of the 5-Thread / 6-Stage SDK Engine Pipeline."""
        box_h = 158.0
        self.ensure_space(box_h + 10.0)
        cmds = self.current_cmds
        top_y = self.y
        bottom_y = top_y - box_h

        # Outer container
        cmds.append("0.97 0.98 1.00 rg")
        cmds.append("0.70 0.77 0.88 RG 0.8 w")
        cmds.append(f"{MARGIN_L:.2f} {bottom_y:.2f} {CONTENT_W:.2f} {box_h:.2f} re B")
        cmds.append("0.11 0.19 0.34 rg")
        cmds.append(
            f"BT /F2 9.0 Tf {MARGIN_L + 10:.2f} {top_y - 14:.2f} Td "
            f"(Figure 1.1 — End-to-End Dual-Path Zero-Allocation SDK Engine Architecture (Capture -> Perception -> Signal -> Render -> Output)) Tj ET"
        )

        # Row 1: CameraX Capture -> Perception / MediaPipe -> SignalPipeline -> LatencyCompensator
        stages_row1 = [
            ("1. CameraX & Clock", "ImageAnalysis + OES\nClockOffsetCalibrator", (0.12, 0.34, 0.56)),
            ("2. Perception Stage", "FastYuvToRgb + Crop\nMediaPipe 478 + 52 BS", (0.14, 0.44, 0.52)),
            ("3. SignalPipeline", "PoseDecomposer + FW\n1-Euro Pose & BS Filter", (0.16, 0.48, 0.38)),
            ("4. LatencyCompensator", "35-50ms Extrapolation\nT_pred * T_lm^-1 Delta", (0.48, 0.32, 0.14)),
        ]
        bw = 112.0
        bh = 44.0
        gap = 14.0
        start_x = MARGIN_L + 12.0
        r1_y = top_y - 68.0

        for idx, (st_title, st_sub, rgb) in enumerate(stages_row1):
            bx = start_x + idx * (bw + gap)
            cmds.append(f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} rg")
            cmds.append(f"{bx:.2f} {r1_y:.2f} {bw:.2f} {bh:.2f} re f")
            cmds.append("1 1 1 rg")
            cmds.append(f"BT /F2 8.2 Tf {bx + 6:.2f} {r1_y + 30:.2f} Td ({pdf_escape(st_title)}) Tj ET")
            sub_lines = st_sub.split("\n")
            cmds.append(f"BT /F1 7.3 Tf {bx + 6:.2f} {r1_y + 18:.2f} Td ({pdf_escape(sub_lines[0])}) Tj ET")
            if len(sub_lines) > 1:
                cmds.append(f"BT /F1 7.3 Tf {bx + 6:.2f} {r1_y + 8:.2f} Td ({pdf_escape(sub_lines[1])}) Tj ET")
            if idx < len(stages_row1) - 1:
                ax1 = bx + bw + 1.5
                ax2 = bx + bw + gap - 1.5
                ay = r1_y + bh * 0.5
                cmds.append("0.20 0.25 0.35 RG 1.2 w")
                cmds.append(f"{ax1:.2f} {ay:.2f} m {ax2:.2f} {ay:.2f} l S")
                cmds.append(f"{ax2 - 4:.2f} {ay + 3:.2f} m {ax2:.2f} {ay:.2f} l {ax2 - 4:.2f} {ay - 3:.2f} l S")

        # Row 2: GL Render Thread 6-Pass Pipeline & Capture Outputs
        stages_row2 = [
            ("5A. Camera InputRing", "SingleOwner TexMatrix\nFrameRing TimestampSync", (0.26, 0.24, 0.52)),
            ("5B. 2D Masks & Retouch", "ControlPointSmoother\nRegionMask R8 + Recolor", (0.44, 0.20, 0.46)),
            ("5C. 3D AR & Occluder", "Depth Occluder + PBR\nCook-Torrance + StudioIBL", (0.52, 0.18, 0.30)),
            ("6. Post & Outputs", "Warp/Skin/LUT/Grain\nDisplay + Still + MP4/AAC", (0.18, 0.36, 0.26)),
        ]
        r2_y = bottom_y + 14.0
        for idx, (st_title, st_sub, rgb) in enumerate(stages_row2):
            bx = start_x + idx * (bw + gap)
            cmds.append(f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} rg")
            cmds.append(f"{bx:.2f} {r2_y:.2f} {bw:.2f} {bh:.2f} re f")
            cmds.append("1 1 1 rg")
            cmds.append(f"BT /F2 8.2 Tf {bx + 6:.2f} {r2_y + 30:.2f} Td ({pdf_escape(st_title)}) Tj ET")
            sub_lines = st_sub.split("\n")
            cmds.append(f"BT /F1 7.3 Tf {bx + 6:.2f} {r2_y + 18:.2f} Td ({pdf_escape(sub_lines[0])}) Tj ET")
            if len(sub_lines) > 1:
                cmds.append(f"BT /F1 7.3 Tf {bx + 6:.2f} {r2_y + 8:.2f} Td ({pdf_escape(sub_lines[1])}) Tj ET")
            if idx < len(stages_row2) - 1:
                ax1 = bx + bw + 1.5
                ax2 = bx + bw + gap - 1.5
                ay = r2_y + bh * 0.5
                cmds.append("0.20 0.25 0.35 RG 1.2 w")
                cmds.append(f"{ax1:.2f} {ay:.2f} m {ax2:.2f} {ay:.2f} l S")
                cmds.append(f"{ax2 - 4:.2f} {ay + 3:.2f} m {ax2:.2f} {ay:.2f} l {ax2 - 4:.2f} {ay - 3:.2f} l S")

        # Connector from Row 1 (LatencyCompensator) to Row 2 (GL Thread stages)
        cmds.append("0.30 0.35 0.45 RG 1.0 w")
        mid_y = (r1_y + (r2_y + bh)) * 0.5
        cmds.append(f"{start_x + 3.5 * (bw + gap) - gap * 0.5:.2f} {r1_y:.2f} m {start_x + 3.5 * (bw + gap) - gap * 0.5:.2f} {mid_y:.2f} l S")
        cmds.append(f"{start_x + bw * 0.5:.2f} {mid_y:.2f} m {start_x + 3.5 * (bw + gap) - gap * 0.5:.2f} {mid_y:.2f} l S")
        cmds.append(f"{start_x + bw * 0.5:.2f} {mid_y:.2f} m {start_x + bw * 0.5:.2f} {r2_y + bh:.2f} l S")

        self.y = bottom_y - 10.0

    def save(self, output_path: str):
        if self.current_cmds:
            self._draw_footer()
            self.pages.append("\n".join(self.current_cmds))
            self.current_cmds = []

        objects = []

        # Object 1: Catalog
        objects.append("<< /Type /Catalog /Pages 2 0 R >>")

        # Object 2: Pages placeholder (filled after knowing page object IDs)
        objects.append("")

        # Objects 3..6: Standard Type1 Fonts (F1=Helvetica, F2=Helvetica-Bold, F3=Courier, F4=Courier-Bold)
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>")

        page_obj_ids = []
        for page_stream in self.pages:
            stream_bytes = page_stream.encode("latin-1", errors="replace")
            content_obj_id = len(objects) + 1
            objects.append(
                f"<< /Length {len(stream_bytes)} >>\nstream\n{page_stream}\nendstream"
            )
            page_obj_id = len(objects) + 1
            page_obj_ids.append(page_obj_id)
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.2f} {PAGE_H:.2f}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R /F4 6 0 R >> >> "
                f"/Contents {content_obj_id} 0 R >>"
            )

        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>"

        # Assemble PDF 1.4 binary with exact byte offsets
        out = bytearray()
        out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for idx, obj_body in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{idx} 0 obj\n".encode("latin-1"))
            out.extend(obj_body.encode("latin-1", errors="replace"))
            out.extend(b"\nendobj\n")

        xref_offset = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

        out.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
                "latin-1"
            )
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(out)


def build_pdf(output_paths: list):
    pdf = PdfCanvas()

    # =========================================================================
    # COVER & SECTION 1: EXECUTIVE ARCHITECTURE OVERVIEW
    # =========================================================================
    pdf.draw_cover_banner(
        title="Xelvic AR & Beauty SDK Engine — Architecture Manual",
        subtitle="Deep Code-Trace Pipeline Analysis, Mathematical Contracts & Real-World Use Cases",
        meta_lines=[
            "Engine Scope: CameraX Capture -> MediaPipe 478/52 Perception -> 1-Euro Signal & Latency Compensation",
            "Render Scope: Dual-Path OpenGL ES 3.0 RenderGraph (PBR 3D + Half-Res R8 Masks + Retouch + LUT/Grain/Warp)",
            "Output Scope: 60fps Live Preview + High-Res Offscreen Still Photo + Zero-Copy GPU MP4 + AAC Video Recording",
        ],
    )

    pdf.section_heading("1.", "Executive Summary & Core Architectural Principles")
    pdf.paragraph(
        "The Xelvic AR & Beauty SDK Engine is a production-grade, data-driven real-time augmented reality and beauty "
        "rendering engine built in Kotlin and OpenGL ES 3.0. It is architected to achieve Snap Camera Kit parity across "
        "3D PBR accessory rendering, sub-half-pixel 2D facial retouching masks, zero-allocation vsync signal extrapolation, "
        "adaptive thermal/FPS governance, and hardware-accelerated photo/video capture."
    )
    pdf.bullet(
        "1. Track Once, Render Many (Zero Camera/Tracker Restarts)",
        "Switching between a 3D glasses filter, a beauty skin/lip retouch filter, or a multi-pass LUT/grain filter never "
        "restarts CameraX or re-initializes the MediaPipe FaceLandmarker. CameraSession dynamically updates only the "
        "requested tracker output flags (requiresPose, requiresLandmarks, requestedBlendshapes) and preloads GL assets via PackagePreloader.",
    )
    pdf.bullet(
        "2. Unified Monotonic Clock Domain (SystemClock.elapsedRealtimeNanos)",
        "Every timestamp across CameraX ImageProxy, OES SurfaceTexture FrameRing, SignalPipeline, LatencyCompensator, and "
        "VideoRecorderPipeline is calibrated into the monotonic CLOCK_BOOTTIME (FrameClock.elapsedRealtimeNs) domain via "
        "ClockOffsetCalibrator using the lowest 5th-percentile arrival-minus-sensor offset.",
    )
    pdf.bullet(
        "3. Single-Owner Input Transform & Single-Stage Mirroring",
        "CameraInputSurface is the single authoritative owner of front-camera horizontal mirror (mirrorH), FILL_CENTER "
        "aspect-ratio crop (cropCenter), and display rotation for 2D camera textures. For 3D projections, "
        "FacePoseAdapter.createCameraProjection applies the front-camera X flip exactly once (m[0] < 0).",
    )
    pdf.bullet(
        "4. Unified 3D + 2D Pose-Prediction Latency Compensation (< 0.5px Fast-Motion Error)",
        "Both 3D meshes (FaceMeshRenderer.computePoseCorrectedMvp) and 2D region masks/warps "
        "(FacePoseAdapter.mapLandmarkToViewWithPoseDelta) fold in the rigid 3D prediction delta "
        "T = M_pred(t_render) * M_landmark(t_capture)^(-1), combined with coherent raw-step high-velocity 1-Euro beta boost.",
    )

    pdf.draw_pipeline_flow_diagram()

    # =========================================================================
    # SECTION 2: THREADING & CONCURRENCY TOPOLOGY
    # =========================================================================
    pdf.section_heading("2.", "Multi-Threaded Runtime Topology & Zero-Allocation Contracts")
    pdf.table(
        headers=["Thread / Executor", "Primary Classes", "Cadence", "Sync Mechanism & Memory Contract"],
        col_widths=[115.0, 135.0, 62.0, 199.0],
        rows=[
            [
                "1. CameraAnalysisExec",
                "CameraController, CameraSession",
                "15-30 Hz",
                "Single-close ImageProxy; cadence throttle & ClockOffsetCalibrator",
            ],
            [
                "2. DedicatedDispatch",
                "MediaPipeFaceGeometryProvider",
                "15-30 Hz",
                "Latest-slot PendingSlot (capacity 1); FastYuvToRgbConverter into reusable Bitmap",
            ],
            [
                "3. MediaPipe Async",
                "FaceLandmarker LIVE_STREAM",
                "15-30 Hz",
                "Emits FaceGeometrySnapshot -> SignalPipeline.processSnapshot -> AtomicReference",
            ],
            [
                "4. GL Render Thread",
                "XelvicGlSurfaceView, RenderGraph",
                "60-120 Hz",
                "Zero-alloc consumeLatestPredictedState; GlResourceOwner thread guard",
            ],
            [
                "5. AudioRecord Thread",
                "VideoRecorderPipeline",
                "~44.1 kHz",
                "PCM 16-bit mic -> MediaCodec AAC + synchronized GL EGLSurface H.264 muxing",
            ],
        ],
    )

    # =========================================================================
    # SECTION 3: STAGE-BY-STAGE DEEP CODE-TRACE PIPELINE
    # =========================================================================
    pdf.section_heading("3.", "Stage-by-Stage Code-Trace Pipeline Deep Dive")

    pdf.sub_heading("Stage 1: Camera Capture, ClockOffsetCalibrator & Single-Owner FrameRing")
    pdf.paragraph(
        "Files: CameraController.kt, CameraSession.kt, AnalysisFrame.kt (ClockOffsetCalibrator), CameraInputSurface.kt. "
        "CameraX binds Preview and ImageAnalysis (YUV_420_888 or RGBA_8888, Strategy.KEEP_ONLY_LATEST). On devices where "
        "SENSOR_INFO_TIMESTAMP_SOURCE is UNKNOWN (0) rather than REALTIME (1), raw sensor timestamps can differ from "
        "SystemClock.elapsedRealtimeNanos() by an arbitrary boot offset."
    )
    pdf.bullet(
        "ClockOffsetCalibrator (5th-Percentile Lower-Bound Alignment)",
        "Over a 30-frame warmup window, ClockOffsetCalibrator records diff = arrivalElapsedRealtimeNs - sensorTimestampNs "
        "and selects the minimum (or lowest 5th percentile for >= 20 samples), never the median. Using the minimum isolates "
        "the true clock-base offset without baking camera delivery delay into the capture timestamp.",
    )
    pdf.bullet(
        "CameraInputSurface & DefaultFrameRing (Path B SINGLE_SCENE_GL)",
        "SurfaceTextureCameraInputSurface attaches a detached SurfaceTexture to an OES texture exclusively on the GL thread. "
        "WhenreadFrame() runs, it computes the single-owner 4x4 UV matrix T(0.5,0.5)*R(rot)*S(signU*scaleU, signV*scaleV)*T(-0.5,-0.5), "
        "records (textureId, sensorTimestampNs, alignedElapsedRealtimeNs, texMatrix) into DefaultFrameRing (8 slots), and supports "
        "tracker-timestamp-synchronized lookup via findFrameClosestToTimestampNs(targetTimestampNs, useAlignedClock = true).",
    )

    pdf.sub_heading("Stage 2: Perception Stage — Fast YUV Conversion, Crop & MediaPipe 478+52")
    pdf.paragraph(
        "Files: MediaPipeFaceGeometryProvider.kt, FastYuvToRgbConverter.kt, FaceGeometrySnapshot.kt. "
        "CameraSession.onImageProxyAvailable enforces governor cadence (15..30 Hz) and wraps the ImageProxy into an AnalysisFrame "
        "carrying per-frame ProjectionParams (derived from CameraCharacteristics LENS_INTRINSIC_CALIBRATION or physical sensor size + "
        "focal length) and the 3x3 normalized imageToViewTransform."
    )
    pdf.bullet(
        "Latest-Slot Dedicated Dispatch (Zero Queue Buildup)",
        "If an older frame is still waiting in PendingSlot when a newer camera frame arrives, the older frame is atomically swapped "
        "out and closed immediately (recording queueOverwriteDrop). If MediaPipe is still busy with in-flight inference and "
        "skipConversionIfPendingFresh is enabled, YUV conversion is deferred until inference finishes.",
    )
    pdf.bullet(
        "Metric 3D Pose Calibration (calibratePoseMatrix)",
        "MediaPipe's raw facial transformation matrix is calibrated against the analysis-crop vertical FOV (fovYCropDeg) and "
        "inter-cheek metric width (#234 right cheek to #454 left cheek) so translation (tx, ty, tz in meters) and rotation "
        "form a true orthonormal rigid pose in metric camera space (+X right, +Y up, -Z into screen).",
    )

    pdf.sub_heading("Stage 3: Signal Layer — Pose Decomposition, 1-Euro Filtering & LatencyCompensator")
    pdf.paragraph(
        "Files: SignalPipeline.kt, PoseDecomposer.kt, PoseFilter.kt, OneEuroFilter.kt, LatencyCompensator.kt, FaceLifecycle.kt. "
        "Raw 4x4 matrices are NEVER low-pass filtered element-wise (which would destroy rotation orthogonality). Instead:"
    )
    pdf.code_block(
        "SignalPipeline & LatencyCompensator Mathematical Pipeline",
        [
            "1. PoseDecomposer.decompose(M_raw) -> translation t in R^3, unit quaternion q in S^3, scale s in R^3",
            "2. PoseFilter.filter(t, q, s, poseTimestampNs):",
            "   - Translation 1-Euro: minCutoff=1.35Hz, beta=38.0, dCutoff=2.8Hz, highVelBeta=1800 (thresh=0.04m/s)",
            "   - Rotation Quat 1-Euro (hemisphere dot >= 0 sign continuity): minCutoff=1.5Hz, beta=14.0, highVelBeta=260",
            "   - Coherent Raw-Step Gate: highSpeedBoost activates only after >= 4 consecutive same-direction raw steps",
            "3. LatencyCompensator.predict(renderElapsedNs, poseTimestampNs):",
            "   - deltaSec = clamp((renderElapsedNs + extraLeadNs) - poseTimestampNs, 0, maxHorizonSec=50ms)",
            "   - taperGain = 1.0 - 0.008 * (deltaSec / horizon)^2",
            "   - t_pred = t_filt + [vx, vy, vz * zGain(0.25)] * (deltaSec * taperGain * velocityBlendFactor)",
            "   - q_pred = exp(0.5 * omega * effectiveDeltaSec) * q_filt  (Stop-after-motion deadzone = 8 mm/s)",
            "4. FaceState stores BOTH M_pred (poseMatrix) and M_raw_capture (landmarkPoseMatrix) with zero GL allocation.",
        ],
    )

    pdf.sub_heading("Stage 4: Phase 3 2D Landmark Latency Compensation & ControlPointSmoother")
    pdf.paragraph(
        "Files: ControlPointSmoother.kt, FaceRegions.kt, FacePoseAdapter.kt. "
        "To prevent 2D beauty masks ($mask:lips, $mask:skin, $mask:eyes, $mask:teeth, $mask:sclera) and post.warp@1 from lagging "
        "behind 3D meshes during rapid head motion while eliminating stationary pixel shimmer:"
    )
    pdf.code_block(
        "Phase 3 Unified 2D+3D Landmark Projection (FacePoseAdapter.mapLandmarkToViewWithPoseDelta)",
        [
            "1. Selective 1-Euro Smoother: ControlPointSmoother filters FaceRegions.CONTROL_POINT_INDICES (121 indices,",
            "   including contour loops, irises 468..477, and warp anchors #4, #6, #129, #358) ONCE per frame timestamp.",
            "2. Unproject 2D smoothed landmark (u, v, z) into 3D camera space at capture depth absZ = |M_lm[14]|:",
            "   vx = (u - 0.5) * (2 * absZ * cropAspect / fCrop),  vy = -(v - 0.5) * (2 * absZ / fCrop),  vz = -absZ - z*wCam",
            "3. Apply rigid 3D prediction delta T = M_pred * M_lm^(-1):",
            "   p_local = R_lm^T * (p_cam - t_lm)   -->   p_pred = R_pred * p_local + t_pred",
            "4. Reproject p_pred into normalized analysis-crop coordinates (u_pred, v_pred) using (fCrop, cropAspect):",
            "   duCrop = (px / (-pz) - vx / (-vz)) * (fCrop / (2 * cropAspect))",
            "   dvCrop = -(py / (-pz) - vy / (-vz)) * (fCrop / 2)",
            "5. Map (u + duCrop, v + dvCrop) through the 3x3 imageToViewTransform -> < 0.5px error at 35ms latency!",
        ],
    )

    pdf.sub_heading("Stage 5: Data-Driven Package Compiler, PassGraph & 6-Pass GLES30 RenderGraph")
    pdf.paragraph(
        "Files: Compiler.kt, PassGraph.kt, RtPool.kt, RenderGraph.kt, CompositionSurface.kt, RegionMaskPass.kt, RecolorPass.kt, "
        "FaceMeshRenderer.kt, ModelRenderer.kt, MaterialRegistry.kt, PostRenderer.kt."
    )
    pdf.table(
        headers=["Pass Order", "RenderGraph Stage", "FBO Target", "Core Operations & Shaders"],
        col_widths=[68.0, 118.0, 95.0, 230.0],
        rows=[
            [
                "Pass 1",
                "BACKGROUND_PASS",
                "sceneFbo (Full RGBA8)",
                "Draws OES camera texture with single-owner uTexMatrix (uMirrorX=0 in Path B)",
            ],
            [
                "Pass 2",
                "CAMERA_PIXEL_PASSES",
                "$mask:<id> (Half-Res R8)",
                "RegionMaskPass centroid-fan + feathered skirt + MaskExpr algebra + RecolorPass",
            ],
            [
                "Pass 3",
                "OCCLUDER_PASS",
                "sceneFbo Depth24",
                "Color mask (F,F,F,F), Depth write ON; renders 468-vtx faceMesh with pushBackFW",
            ],
            [
                "Pass 4",
                "OPAQUE_3D_PASS",
                "sceneFbo Color+Depth",
                "ModelRenderer glTF 2.0 meshes: material.pbr@1 (Cook-Torrance GGX + Studio IBL)",
            ],
            [
                "Pass 5",
                "TRANSPARENT_3D_PASS",
                "sceneFbo (Sorted)",
                "Back-to-front faceMesh paint, PBR glass lenses (Fresnel alpha), 2D flipbook sprites",
            ],
            [
                "Pass 6",
                "POST_PASSES -> DISPLAY",
                "Ping-Pong -> FBO 0",
                "PostRenderer: warp, skinSoften/bilateral, colorGrade, colorMatrix, 3D LUT, grain, sharpen",
            ],
        ],
    )

    pdf.sub_heading("Stage 6: Dual-Mode Composition (Path A vs Path B) & Capture Output Pipelines")
    pdf.bullet(
        "Path A (TRANSPARENT_OVERLAY) vs Path B (SINGLE_SCENE_GL)",
        "In Path A, PreviewView renders the camera feed on the view hierarchy while TransparentOverlayCompositionSurface renders "
        "3D/2D overlays with premultiplied alpha. Whenever a filter requires camera pixels (requiresCameraPixels == true, e.g., "
        "beauty skin smoothing, recolor, warp, LUT, grain, sharpen) or video recording starts, XelvicGlSurfaceView automatically "
        "activates Path B (SingleSceneCompositionSurface), compositing camera OES + masks + 3D + post into sceneFbo.",
    )
    pdf.bullet(
        "StillPipeline & OffscreenRenderer (High-Resolution Photo Capture)",
        "Captures full-resolution photos via an isolated EGL14 PBuffer context (OffscreenRenderer) or deterministic CPU fallback "
        "using the exact same CompiledEffect, RegionMaskPass (with expanded cranial forehead hairline feather), RecolorPass, and PostRenderer.",
    )
    pdf.bullet(
        "VideoRecorderPipeline (Real-Time GPU MP4 + AAC Microphone Muxing)",
        "Creates an EGLSurface backed by MediaCodec.createInputSurface() (video/avc H.264) sharing the live GL context, blits "
        "SingleSceneCompositionSurface.sceneColorTex at 30/60fps using monotonic sensor timestamps via EGLExt.eglPresentationTimeANDROID, "
        "and simultaneously encodes AudioRecord PCM audio into AAC (audio/mp4a-latm) via a single MediaMuxer.",
    )

    # =========================================================================
    # SECTION 4: THREE CONCRETE END-TO-END USE-CASE DEMONSTRATIONS
    # =========================================================================
    pdf.section_heading("4.", "Concrete End-to-End Use-Case Walkthroughs")

    pdf.sub_heading("Use Case 1: 3D PBR Aviator Sunglasses + Head Occluder + Smile Behavior")
    pdf.paragraph(
        "Demonstrates how a 3D glTF accessory filter with metallic frames, semi-transparent specular glass lenses, head/nose "
        "depth occlusion, and an expression-driven behavior runs from manifest compilation to 60fps GL rendering."
    )
    pdf.code_block(
        "Use Case 1 Manifest & Runtime Execution Trace (3D PBR + Occluder)",
        [
            "Manifest Capabilities: [\"pose\", \"landmarks\", \"blendshapes\"]  (requiresCameraPixels = false -> Path A or B)",
            "1. Compiler.kt & GlbLoader.kt:",
            "   - Parses glTF 2.0 binary (.glb), extracting pbrMetallicRoughness (baseColorFactor, metallicFactor=0.92,",
            "     roughnessFactor=0.18, baseColorTexture, metallicRoughnessTexture, normalTexture, emissiveTexture).",
            "   - Compiles 2 scene nodes: 'face_occ' (type='occluder', shape='faceMesh', pushBackFW=0.02) at renderOrder=0,",
            "     and 'aviator_glasses' (type='model3d', anchor='nose_bridge', shader='material.pbr@1') at renderOrder=10.",
            "2. Frame Execution on GL Thread (60fps):",
            "   - SignalPipeline.consumeLatestPredictedState(renderElapsedNs) predicts 3D head pose 35ms ahead.",
            "   - Pass 3 (OCCLUDER_PASS): FaceMeshRenderer unprojects 468 landmarks at |landmarkPoseMatrix[14]|, pushes",
            "     vertices back by 0.02*FW along view rays, and draws into Depth24 buffer with glColorMask(false,false,false,false)",
            "     using pose-corrected MVP = P_view * (M_pred * M_landmark^-1).",
            "   - Pass 4/5 (3D PBR PASS): AnchorResolver places glasses at 'nose_bridge' in FW-scaled metric units.",
            "     MaterialRegistry 'material.pbr@1' evaluates Cook-Torrance GGX D*G*F specular, Schlick Fresnel glass",
            "     alpha boost (alpha = max(baseAlpha, fresnelLuminance)), procedural 2-light + studio horizon IBL, and ACES",
            "     tonemapping. Temples behind the head/ears are cleanly rejected by the Depth24 faceMesh occluder!",
        ],
    )

    pdf.sub_heading("Use Case 2: Snap Retouch & Beauty Makeup Filter (Full Forehead Skin, Lips, Teeth, Sclera, Warp, 3D LUT)")
    pdf.paragraph(
        "Demonstrates a multi-pass camera-pixel beauty filter combining cranial-extrapolated skin smoothing, lip tinting, "
        "teeth/sclera brightening, facialliquify warp, and 3D LUT color grading during a fast head turn."
    )
    pdf.code_block(
        "Use Case 2 Execution Trace (Beauty Retouch + Phase 3 Zero-Lag 2D Masks + PassGraph)",
        [
            "Manifest Capabilities: [\"pose\", \"landmarks\", \"regionMasks\", \"recolor\", \"post\"] (requiresCameraPixels = true -> Path B)",
            "1. Pass 1 (BACKGROUND_PASS): SingleSceneCompositionSurface reads OES camera frame into sceneFbo.",
            "2. Pass 2A (RegionMaskPass -> Half-Res GL_R8 FBOs in RtPool):",
            "   - ControlPointSmoother steps 121 control-point 1-Euro filters ONCE for the frame's poseTimestampNs.",
            "   - FacePoseAdapter.mapLandmarkToViewWithPoseDelta projects each smoothed landmark with 3D pose delta",
            "     (M_pred * M_lm^-1) through imageToViewTransform (< 0.5px lag even during 525 px/s fast head turn).",
            "   - '$mask:skin': Triangulates FACE_OVAL with cranial forehead extrapolation (+22% of chin-to-forehead vector",
            "     above landmark #10 and 1.77x expanded hairline feather) minus eyes, brows, and lips.",
            "   - '$mask:lips': LIPS_OUTER minus MOUTH_INNER (auto-attenuated when mouth opening ratio > threshold).",
            "   - '$mask:teeth' & '$mask:sclera': MOUTH_INNER luminance-keyed for teeth; EYES minus unclamped IRIS for sclera.",
            "3. Pass 2B (RecolorPass): Applies softLight/multiply lip tinting + HSV yellow-desaturating teeth/sclera whitening.",
            "4. Pass 6 (PostRenderer Ping-Pong Chain):",
            "   - 'post.skinSoften@1': Dual-radius bilateral surface blur masked by '$mask:skin' with high-frequency pore",
            "     detail restoration (detailPreserve) so skin looks airbrushed yet natural.",
            "   - 'post.warp@1': populateWarpAnchors hoists smoothed #4, #6, #10, #152, #234, #454, #172, #397, #129, #358",
            "     anchors to apply jawSlim, eyeEnlarge, noseSlim, and chinVLine strictly inside the face_oval envelope.",
            "   - 'post.lut@1' & 'post.grain@1': Applies trilinear 3D Hald/Strip LUT grading + 24Hz luminance-weighted film grain.",
        ],
    )

    pdf.sub_heading("Use Case 3: Real-Time GPU Video Recording (MP4 + AAC Audio) & Adaptive Governor")
    pdf.paragraph(
        "Demonstrates how the user records a 1080p60 MP4 video with live beauty + 3D filters while PerformanceGovernor "
        "monitors frame budgets and thermal state."
    )
    pdf.code_block(
        "Use Case 3 Execution Trace (Zero-Copy GPU Video Recording & PerformanceGovernor)",
        [
            "1. VideoRecorderPipeline.startRecording(config: 1080x1920, 30/60fps, 12Mbps H.264, 128kbps AAC):",
            "   - Configures MediaCodec 'video/avc' (COLOR_FormatSurface) + creates encoderInputSurface.",
            "   - On GL thread, creates shared-context EGLSurface bound to encoderInputSurface.",
            "   - Starts AudioRecord background thread feeding MediaCodec 'audio/mp4a-latm' with monotonic timestamps.",
            "2. Per-Frame GL Hook (XelvicGlSurfaceView.onDrawFrame -> onAfterRenderGraph):",
            "   - After SingleSceneCompositionSurface finishes rendering all 6 passes into sceneFbo (sceneColorTex),",
            "     VideoRecorderPipeline switches eglMakeCurrent to the encoder EGLSurface, draws sceneColorTex with",
            "     an upright fullscreen quad, sets EGLExt.eglPresentationTimeANDROID(sensorTimestampNs), and swaps buffers.",
            "   - Drains encoded H.264 & AAC packets into MediaMuxer without a single CPU glReadPixels stall!",
            "3. PerformanceGovernor Closed-Loop Guard:",
            "   - Evaluates rolling 5s window from LatencyProbe (drawCpuMs, inferenceMs, ageAtRenderMs, thermal status).",
            "   - Scales analysis resolution (360p -> 256p -> 192p), tracker cadence (30 -> 24 -> 15 Hz), or disables",
            "     optional post passes if frame time exceeds budget, respecting warmup and cooldown windows.",
        ],
    )

    # =========================================================================
    # SECTION 5: COMPLETE SOURCE MODULE REFERENCE TABLE
    # =========================================================================
    pdf.section_heading("5.", "Complete SDK Engine Module & File Reference")
    pdf.table(
        headers=["Layer / Package", "Key Source Files", "Core Responsibility in the Pipeline"],
        col_widths=[108.0, 155.0, 248.0],
        rows=[
            [
                "capture/",
                "CameraSession.kt, AnalysisFrame.kt",
                "Cadence governor, ClockOffsetCalibrator (5th-pct), intrinsics/FOV, ImageProxy lifecycle",
            ],
            [
                "capture/",
                "CameraInputSurface.kt",
                "Single-owner OES SurfaceTexture transform (mirror/crop/rot) & 8-slot DefaultFrameRing",
            ],
            [
                "capture/",
                "VideoRecorderPipeline.kt",
                "Zero-copy GPU EGLSurface H.264 encoder + AudioRecord AAC encoder + MediaMuxer",
            ],
            [
                "perception/",
                "MediaPipeFaceGeometryProvider.kt",
                "DedicatedDispatchThread, FastYuvToRgbConverter, 478 landmarks + 52 blendshapes + pose",
            ],
            [
                "signal/",
                "SignalPipeline.kt, PoseFilter.kt",
                "PoseDecomposer, 1-Euro translation/quat/scale/BS filtering with raw-step coherence boost",
            ],
            [
                "signal/",
                "LatencyCompensator.kt",
                "Bounded 50ms horizon pose extrapolation, velocity deadzone, stop-after-motion braking",
            ],
            [
                "signal/",
                "FacePoseAdapter.kt, ControlPointSmoother",
                "Canonical FW scale, single-stage 3D mirror, 121-pt 1-Euro smoother, 3D->2D pose delta",
            ],
            [
                "pkg/",
                "Compiler.kt, PassGraph.kt, FaceRegions",
                "Manifest validation, DAG pass compiler, pinned 478 topology loops & warp anchors",
            ],
            [
                "render/",
                "RenderGraph.kt, CompositionSurface.kt",
                "Dual-path (Path A Overlay / Path B SingleScene) 6-pass GLES30 orchestrator & RtPool",
            ],
            [
                "render/",
                "RegionMaskPass.kt, RecolorPass.kt",
                "Half-res R8 feathered masks ($mask:skin/lips/eyes/teeth/sclera) & HSV/blend recolor",
            ],
            [
                "render/",
                "ModelRenderer.kt, MaterialRegistry.kt",
                "glTF 2.0 loader, Cook-Torrance GGX PBR shader, procedural studio IBL, FaceMeshRenderer",
            ],
            [
                "render/",
                "PostRenderer.kt, StillPipeline.kt",
                "Warp, bilateral skinSoften, 3D LUT, film grain, sharpen, colorMatrix, PBuffer still photo",
            ],
            [
                "governor/ & debug/",
                "PerformanceGovernor.kt, LatencyProbe",
                "Closed-loop resolution/FPS scaling, mask-offset HUD telemetry, ReplayHarness verification",
            ],
        ],
    )

    for path in output_paths:
        pdf.save(path)
        print(f"Saved PDF ({len(pdf.pages)} pages, {os.path.getsize(path)} bytes) -> {path}")


if __name__ == "__main__":
    build_pdf([
        "/.aistudio/artifacts/brain/b4cc3a92-415f-482e-a098-68fb8581686c/Xelvic_SDK_Engine_Architecture_and_Pipeline_Guide.pdf",
        "/app/applet/Xelvic_SDK_Engine_Architecture_and_Pipeline_Guide.pdf",
        "/app/applet/public/assets/Xelvic_SDK_Engine_Architecture_and_Pipeline_Guide.pdf",
    ])
