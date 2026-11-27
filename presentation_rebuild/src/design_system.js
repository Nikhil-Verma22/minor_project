"use strict";

/**
 * Reusable Design System derived from "Object Retrieval" Canva PPTX & PDF.
 * Canvas: 20.0" x 11.25" (16:9 high-res Canva widescreen, 1920x1080 @ 96dpi).
 */
const CANVAS = {
  layoutName: "CANVA_WIDE_20",
  width: 20.0,
  height: 11.25,
};

const PALETTE = {
  bg: "FFFFFF",
  ink: "1E293B",       // Deep Navy Slate (headings & primary text)
  inkBlack: "000000",  // Used inside dense tables & card bullets
  accent: "60A5FA",    // Signature sky blue (vertical bars, accents)
  borderBlue: "65A1E7",
  tableHeader: "A2CBFF",
  boxTint: "C5E0FD",
  cardBody: "9DC4F2",
  cardHeader: "F4F9FD",
  tableGrid: "93C5FD",
  mutedGray: "475569",
};

const FONTS = {
  display: "Poppins SemiBold",
  body: "Open Sans",
  bodyBold: "Open Sans Bold",
  alt: "Roboto",
  altBold: "Roboto Bold",
};

const POSITIONS = {
  leftBar: { x: 0.0, y: 0.0, w: 0.402, h: 11.25 },
  starTopLeft: { x: 1.125, y: 0.975, w: 0.926, h: 0.926 },
  starIntro: { x: 1.125, y: 1.125, w: 0.926, h: 0.926 },
  starTopRight: { x: 17.949, y: 0.863, w: 0.926, h: 0.926 },
  spheresBottomRight: { x: 18.163, y: 9.413, w: 3.673, h: 3.673 },
  spheresTopRight: { x: 18.163, y: -1.581, w: 3.673, h: 3.673 },
  arcTopRight: { x: 17.258, y: -1.735, w: 4.5, h: 4.5 },
};

/**
 * Phase-based graphic taxonomy using the selected 6 Blue Canva elements:
 * - headings: 01_starburst_12pt (Phase A) / 02_starburst_8pt (Phase B)
 * - bullets:  03_sparkle_4pt across all bullet lists
 * - notes:    07_gradient_capsule across bottom notes & pointer banners
 * - topDecor: 15_corner_arc (Phase A) / 08_double_corner_ribbon (Phase B)
 * - botDecor: 14_corner_spheres (Phase A) / 11_split_arch (Phase B)
 */
const ELEMENT_TAXONOMY = {
  phaseA: { heading: "star12", topDecor: "arc", botDecor: "spheres" },
  phaseB: { heading: "star8", topDecor: "ribbon", botDecor: "splitArch" },
  bullets: "sparkle",
  notes: "capsule",
};

function initPresentation(pptx) {
  pptx.defineLayout({ name: CANVAS.layoutName, width: CANVAS.width, height: CANVAS.height });
  pptx.layout = CANVAS.layoutName;
  pptx.theme = { headFontFace: FONTS.display, bodyFontFace: FONTS.body };

  pptx.defineSlideMaster({
    title: "CANVA_CLEAN",
    background: { color: PALETTE.bg },
  });
}

module.exports = {
  CANVAS,
  PALETTE,
  FONTS,
  POSITIONS,
  ELEMENT_TAXONOMY,
  initPresentation,
};
