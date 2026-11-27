"use strict";

const dims = require("D:/Minor/presentation_rebuild/assets/kit_dims.json");

const pptx = new PptxGenJS();
const W = 20.0;
const H = 11.25;
pptx.defineLayout({ name: "CANVA_WIDE_20", width: W, height: H });
pptx.layout = "CANVA_WIDE_20";
pptx.title = "Extracted Elements Kit — Blue & Red Color Schemes";
pptx.author = "PPT Maker";
pptx.theme = { headFontFace: "Poppins SemiBold", bodyFontFace: "Open Sans" };

const SCHEMES = {
  blue: {
    name: "Blue Scheme (Original)",
    prefix: "blue_",
    ink: "1E293B",
    subInk: "475569",
    accent: "60A5FA",
    border: "65A1E7",
    tableHeader: "A2CBFF",
    boxTint: "C5E0FD",
    cardBody: "9DC4F2",
    cardBodySoft: "BAD9FC",
    cardHeader: "F4F9FD",
    grid: "93C5FD",
    swatchHexes: ["1E293B", "60A5FA", "65A1E7", "A2CBFF", "C5E0FD", "F4F9FD"],
  },
  red: {
    name: "Red Scheme (Duplicated Variant)",
    prefix: "red_",
    ink: "1E293B",
    subInk: "6B3A3E",
    accent: "EF4444",
    border: "DC2626",
    tableHeader: "FCA5A5",
    boxTint: "FECACA",
    cardBody: "F8A5A5",
    cardBodySoft: "FECACA",
    cardHeader: "FFF5F5",
    grid: "F87171",
    swatchHexes: ["1E293B", "EF4444", "DC2626", "FCA5A5", "FECACA", "FFF5F5"],
  },
};

const ELEMENTS = [
  ["01_starburst_12pt.png", "01. 12-Pt Starburst"],
  ["02_starburst_8pt.png", "02. 8-Pt Starburst"],
  ["03_sparkle_4pt.png", "03. 4-Pt Sparkle"],
  ["04_stacked_pills_3.png", "04. 3 Stacked Pills"],
  ["05_bauhaus_grid_2x2.png", "05. Bauhaus 2x2 Grid"],
  ["06_clover_4leaf.png", "06. 4-Leaf Clover"],
  ["07_gradient_capsule.png", "07. Gradient Capsule"],
  ["08_double_corner_ribbon.png", "08. Double Corner Ribbon"],
  ["09_nested_arches_3.png", "09. Nested Arches"],
  ["10_sliced_sphere.png", "10. Sliced Sphere"],
  ["11_split_arch.png", "11. Split Arch"],
  ["12_stacked_crescents.png", "12. Stacked Crescents"],
  ["13_quarter_ring_bold.png", "13. Thick Quarter Arc"],
  ["14_corner_spheres.png", "14. Twin Corner Spheres"],
  ["15_corner_arc.png", "15. Corner Ribbon Arc"],
  ["16_left_accent_bar.png", "16. Left Fade Bar"],
  ["17_cover_hero_circles.png", "17. Hero Cover Circles"],
  ["18_faded_ring.png", "18. Soft Faded Ring"],
];

function fitContain(boxX, boxY, boxW, boxH, natW, natH) {
  const ratioImg = natW / Math.max(natH, 1);
  const ratioBox = boxW / Math.max(boxH, 1);
  let drawW = boxW;
  let drawH = boxH;
  if (ratioImg > ratioBox) {
    drawH = boxW / ratioImg;
  } else {
    drawW = boxH * ratioImg;
  }
  return {
    x: boxX + (boxW - drawW) / 2,
    y: boxY + (boxH - drawH) / 2,
    w: drawW,
    h: drawH,
  };
}

function addGraphicKitSlide(schemeKey) {
  const s = SCHEMES[schemeKey];
  const slide = pptx.addSlide();
  slide.background = { color: "FFFFFF" };

  // Left gradient edge bar + header starburst
  slide.addImage({ path: images[`${s.prefix}16_left_accent_bar.png`], x: 0, y: 0, w: 0.402, h: 11.25 });
  slide.addImage({ path: images[`${s.prefix}01_starburst_12pt.png`], x: 0.95, y: 0.55, w: 0.72, h: 0.72 });

  slide.addText(`Extracted Graphic Elements — ${s.name}`, {
    x: 1.85,
    y: 0.48,
    w: 14.3,
    h: 0.82,
    fontFace: "Poppins SemiBold",
    fontSize: 34,
    bold: true,
    color: s.ink,
    valign: "middle",
    margin: 0,
    isTextBox: true,
    fit: "shrink",
  });

  // Scheme badge pill top-right
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 16.4,
    y: 0.62,
    w: 2.85,
    h: 0.58,
    rectRadius: 0.12,
    fill: { color: s.boxTint },
    line: { color: s.border, width: 1.5 },
  });
  slide.addText(schemeKey.toUpperCase() + " KIT (18 ASSETS)", {
    x: 16.4,
    y: 0.62,
    w: 2.85,
    h: 0.58,
    fontFace: "Open Sans Bold",
    fontSize: 15,
    bold: true,
    color: s.ink,
    align: "center",
    valign: "middle",
    margin: 0,
    isTextBox: true,
  });

  // 6 cols x 3 rows gallery grid
  const cols = 6;
  const startX = 0.95;
  const startY = 1.65;
  const gapX = 0.28;
  const gapY = 0.28;
  const cellW = (19.25 - startX - (cols - 1) * gapX) / cols; // ~2.82"
  const cellH = 2.88;

  ELEMENTS.forEach(([fname, label], idx) => {
    const r = Math.floor(idx / cols);
    const c = idx % cols;
    const cx = startX + c * (cellW + gapX);
    const cy = startY + r * (cellH + gapY);

    // Subtle card tile so faint white/gradient edges stay clearly visible
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: cx,
      y: cy,
      w: cellW,
      h: cellH,
      rectRadius: 0.06,
      fill: { color: "F8FAFC" },
      line: { color: "E2E8F0", width: 1.0 },
    });

    const [nw, nh] = dims[fname] || [500, 500];
    const placed = fitContain(cx + 0.22, cy + 0.16, cellW - 0.44, cellH - 0.78, nw, nh);
    slide.addImage({ path: images[`${s.prefix}${fname}`], ...placed });

    slide.addText(label, {
      x: cx + 0.1,
      y: cy + cellH - 0.46,
      w: cellW - 0.2,
      h: 0.36,
      fontFace: "Open Sans Bold",
      fontSize: 13.5,
      bold: true,
      color: s.ink,
      align: "center",
      valign: "middle",
      margin: 0,
      isTextBox: true,
      fit: "shrink",
    });
  });
}

function addUiComponentsSlide(schemeKey) {
  const s = SCHEMES[schemeKey];
  const slide = pptx.addSlide();
  slide.background = { color: "FFFFFF" };

  slide.addImage({ path: images[`${s.prefix}16_left_accent_bar.png`], x: 0, y: 0, w: 0.402, h: 11.25 });
  slide.addImage({ path: images[`${s.prefix}01_starburst_12pt.png`], x: 0.95, y: 0.55, w: 0.72, h: 0.72 });

  slide.addText(`Native UI & Layout Components — ${s.name}`, {
    x: 1.85,
    y: 0.48,
    w: 16.5,
    h: 0.82,
    fontFace: "Poppins SemiBold",
    fontSize: 34,
    bold: true,
    color: s.ink,
    valign: "middle",
    margin: 0,
    isTextBox: true,
    fit: "shrink",
  });

  // ── 1. Palette Swatches Strip ──
  slide.addText("COLOR PALETTE TOKENS", {
    x: 0.95, y: 1.50, w: 5.0, h: 0.36,
    fontFace: "Open Sans Bold", fontSize: 15, bold: true, color: s.subInk, margin: 0, isTextBox: true,
  });
  s.swatchHexes.forEach((hex, idx) => {
    const sx = 0.95 + idx * 1.48;
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: sx, y: 1.94, w: 1.32, h: 0.86, rectRadius: 0.08,
      fill: { color: hex }, line: { color: "CBD5E1", width: 1.0 },
    });
    slide.addText("#" + hex, {
      x: sx, y: 2.86, w: 1.32, h: 0.32,
      fontFace: "Open Sans Bold", fontSize: 13, bold: true, color: s.ink, align: "center", margin: 0, isTextBox: true,
    });
  });

  // ── 2. Vertical Accent Bar + Presenter Block (from Slide 1) ──
  slide.addText("VERTICAL ACCENT CALLOUT (SLIDE 1)", {
    x: 0.95, y: 3.48, w: 4.5, h: 0.36,
    fontFace: "Open Sans Bold", fontSize: 15, bold: true, color: s.subInk, margin: 0, isTextBox: true,
  });
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0.95, y: 3.96, w: 0.08, h: 1.85, fill: { color: s.accent }, line: { type: "none" },
  });
  slide.addText("Metadata Header :", {
    x: 1.22, y: 3.94, w: 4.1, h: 0.40,
    fontFace: "Open Sans Bold", fontSize: 19, bold: true, color: s.ink, valign: "top", margin: 0, isTextBox: true,
  });
  slide.addText("Primary label item 01\nPrimary label item 02\nPrimary label item 03", {
    x: 1.22, y: 4.40, w: 4.1, h: 1.40,
    fontFace: "Open Sans", fontSize: 17, color: s.ink, lineSpacing: 25, valign: "top", margin: 0, isTextBox: true,
  });

  // ── 3. Split Pointer Card (from Slide 8) ──
  slide.addText("SPLIT POINTER CARD (SLIDE 8)", {
    x: 5.50, y: 3.48, w: 4.3, h: 0.36,
    fontFace: "Open Sans Bold", fontSize: 15, bold: true, color: s.subInk, margin: 0, isTextBox: true,
  });
  // Bottom body card first so top header sits cleanly over it (ends at y = 6.45")
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 5.50, y: 4.62, w: 4.25, h: 1.83, rectRadius: 0.08,
    fill: { color: s.cardBody }, line: { type: "none" },
  });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 5.50, y: 3.94, w: 4.25, h: 0.82, rectRadius: 0.08,
    fill: { color: s.cardHeader }, line: { color: "E2E8F0", width: 0.75 },
  });
  slide.addShape(pptx.shapes.ISOSCELES_TRIANGLE, {
    x: 7.40, y: 4.72, w: 0.45, h: 0.22, rotate: 180,
    fill: { color: s.cardHeader }, line: { type: "none" },
  });
  slide.addText("Dataset Card Header", {
    x: 5.65, y: 3.98, w: 3.95, h: 0.72,
    fontFace: "Open Sans Bold", fontSize: 19, bold: true, color: s.ink, align: "center", valign: "middle", margin: 0, isTextBox: true,
  });
  slide.addText("Metric Summary", {
    x: 5.65, y: 5.02, w: 3.95, h: 0.38,
    fontFace: "Open Sans Bold", fontSize: 18, bold: true, color: s.ink, align: "center", margin: 0, isTextBox: true,
  });
  slide.addText("• Attribute breakdown item\n• Supporting dataset detail", {
    x: 5.80, y: 5.48, w: 3.68, h: 0.88,
    fontFace: "Open Sans", fontSize: 15.5, color: s.ink, lineSpacing: 22, valign: "top", margin: 0, isTextBox: true,
  });

  // ── 4. Numbered Conclusion Card with Side Pointer (from Slide 15) ──
  slide.addText("NUMBERED CALLOUT CARD (SLIDE 15)", {
    x: 0.95, y: 6.72, w: 5.5, h: 0.36,
    fontFace: "Open Sans Bold", fontSize: 15, bold: true, color: s.subInk, margin: 0, isTextBox: true,
  });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 0.95, y: 7.15, w: 8.65, h: 3.15, rectRadius: 0.08,
    fill: { color: s.boxTint }, line: { type: "none" },
  });
  slide.addShape(pptx.shapes.ISOSCELES_TRIANGLE, {
    x: 9.52, y: 8.48, w: 0.38, h: 0.48, rotate: 90,
    fill: { color: s.boxTint }, line: { type: "none" },
  });
  slide.addText("01", {
    x: 1.05, y: 7.90, w: 1.35, h: 1.1,
    fontFace: "Poppins SemiBold", fontSize: 56, bold: true, color: s.ink, align: "center", valign: "middle", margin: 0, isTextBox: true,
  });
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 2.48, y: 7.45, w: 0.05, h: 2.55, fill: { color: s.accent }, line: { type: "none" },
  });
  slide.addText("Insight Headline:", {
    x: 2.75, y: 7.45, w: 6.5, h: 0.42,
    fontFace: "Open Sans Bold", fontSize: 19, bold: true, color: s.ink, valign: "top", margin: 0, isTextBox: true,
  });
  slide.addText(
    "Structured two-column summary card preserving number badge, vertical accent rule, and right chevron pointer.",
    {
      x: 2.75, y: 7.96, w: 6.5, h: 2.05,
      fontFace: "Open Sans", fontSize: 17.5, color: s.ink, lineSpacing: 25, valign: "top", margin: 0, isTextBox: true,
    }
  );

  // ── 5. Diagram Flowchart Nodes + Connectors (from Slide 10) ──
  slide.addText("DIAGRAM NODES & ARROWS (SLIDE 10)", {
    x: 10.45, y: 1.55, w: 8.6, h: 0.38,
    fontFace: "Open Sans Bold", fontSize: 16, bold: true, color: s.subInk, margin: 0, isTextBox: true,
  });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 10.45, y: 2.05, w: 8.75, h: 0.58, rectRadius: 0.05,
    fill: { color: "FFFFFF" }, line: { color: s.border, width: 1.5 },
  });
  slide.addText("STANDARD PROCESS NODE", {
    x: 10.45, y: 2.05, w: 8.75, h: 0.58,
    fontFace: "Open Sans Bold", fontSize: 18, bold: true, color: s.ink, align: "center", valign: "middle", margin: 0, isTextBox: true,
  });
  slide.addShape(pptx.shapes.LINE, {
    x: 14.825, y: 2.63, w: 0, h: 0.44,
    line: { color: s.border, width: 2.0, endArrowType: "triangle" },
  });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 10.45, y: 3.07, w: 8.75, h: 0.62, rectRadius: 0.05,
    fill: { color: s.boxTint }, line: { color: s.border, width: 1.5 },
  });
  slide.addText("DECISION / HIGHLIGHTED NODE", {
    x: 10.45, y: 3.07, w: 8.75, h: 0.62,
    fontFace: "Open Sans Bold", fontSize: 18, bold: true, color: s.ink, align: "center", valign: "middle", margin: 0, isTextBox: true,
  });

  // ── 6. Native Data Table Component (from Slides 5 & 12) ──
  slide.addText("NATIVE DATA TABLE COMPONENT (SLIDES 5, 11–14)", {
    x: 10.45, y: 4.15, w: 8.6, h: 0.38,
    fontFace: "Open Sans Bold", fontSize: 16, bold: true, color: s.subInk, margin: 0, isTextBox: true,
  });
  slide.addTable(
    [
      [
        { text: "Model", options: { bold: true, fill: { color: s.tableHeader }, color: s.ink, align: "center" } },
        { text: "R@50", options: { bold: true, fill: { color: s.tableHeader }, color: s.ink, align: "center" } },
        { text: "mAP@50", options: { bold: true, fill: { color: s.tableHeader }, color: s.ink, align: "center" } },
        { text: "Speed", options: { bold: true, fill: { color: s.tableHeader }, color: s.ink, align: "center" } },
      ],
      [
        { text: "REACT", options: { bold: true, fill: { color: "FFFFFF" }, color: s.ink, align: "left" } },
        { text: "24.55%", options: { fill: { color: "FFFFFF" }, color: s.ink, align: "center" } },
        { text: "37.58%", options: { fill: { color: "FFFFFF" }, color: s.ink, align: "center" } },
        { text: "1.42 FPS", options: { fill: { color: "FFFFFF" }, color: s.ink, align: "center" } },
      ],
      [
        { text: "REACT++", options: { bold: true, fill: { color: s.cardHeader }, color: s.ink, align: "left" } },
        { text: "23.54%", options: { fill: { color: s.cardHeader }, color: s.ink, align: "center" } },
        { text: "37.43%", options: { fill: { color: s.cardHeader }, color: s.ink, align: "center" } },
        { text: "1.34 FPS", options: { fill: { color: s.cardHeader }, color: s.ink, align: "center" } },
      ],
      [
        { text: "C-SGG", options: { bold: true, fill: { color: "FFFFFF" }, color: s.ink, align: "left" } },
        { text: "18.78%", options: { fill: { color: "FFFFFF" }, color: s.ink, align: "center" } },
        { text: "34.19%", options: { fill: { color: "FFFFFF" }, color: s.ink, align: "center" } },
        { text: "2.85 FPS", options: { fill: { color: "FFFFFF" }, color: s.ink, align: "center" } },
      ],
    ],
    {
      x: 10.45,
      y: 4.65,
      w: 8.75,
      colW: [2.25, 2.0, 2.25, 2.25],
      rowH: 0.72,
      fontFace: "Open Sans",
      fontSize: 17,
      valign: "middle",
      border: { type: "solid", color: s.grid, pt: 1.0 },
    }
  );

  // Decorative corner spheres matching current scheme
  slide.addImage({
    path: images[`${s.prefix}14_corner_spheres.png`],
    x: 18.163, y: 9.413, w: 3.673, h: 3.673,
  });
}

// Slide 1 & 2: Blue Scheme | Slide 3 & 4: Red Scheme
addGraphicKitSlide("blue");
addUiComponentsSlide("blue");
addGraphicKitSlide("red");
addUiComponentsSlide("red");

pptx.writeFile({ fileName: "elements_blue_red.pptx" });
