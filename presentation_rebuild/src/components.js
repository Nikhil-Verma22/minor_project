"use strict";

const { PALETTE, FONTS, POSITIONS } = require("D:/Minor/presentation_rebuild/src/design_system.js");

const K = {
  star12: "01_starburst_12pt.png",
  star8: "02_starburst_8pt.png",
  sparkle: "03_sparkle_4pt.png",
  pills: "04_stacked_pills_3.png",
  bauhaus: "05_bauhaus_grid_2x2.png",
  clover: "06_clover_4leaf.png",
  capsule: "07_gradient_capsule.png",
  ribbon: "08_double_corner_ribbon.png",
  arches: "09_nested_arches_3.png",
  slicedSphere: "10_sliced_sphere.png",
  splitArch: "11_split_arch.png",
  crescents: "12_stacked_crescents.png",
  quarterRing: "13_quarter_ring_bold.png",
  spheres: "14_corner_spheres.png",
  arc: "15_corner_arc.png",
  leftBar: "16_left_accent_bar.png",
  heroCircles: "17_cover_hero_circles.png",
  fadedRing: "18_faded_ring.png",
  introPhoto: "intro_workspace_rounded.png",
};

/**
 * Resolve short icon role names cleanly to asset filenames.
 */
function resolveIconPath(images, key) {
  if (!images || !key) return null;
  if (images[key]) return images[key];
  if (K[key] && images[K[key]]) return images[K[key]];
  return null;
}

/**
 * Unified slide chrome: left blue gradient accent strip + diversified heading mark & corner decor.
 */
function addContentChrome(slide, images, {
  kicker = "",
  title = "",
  titleSize = 44,
  subtitle = "",
  slideNum = null,
  phase = "A",
  headingIcon = null,
  starburst = "topLeft",
  corner = null,
  watermarkRing = false,
} = {}) {
  slide.background = { color: PALETTE.bg };

  if (watermarkRing && images[K.fadedRing]) {
    slide.addImage({ path: images[K.fadedRing], x: 15.2, y: -2.2, w: 6.4, h: 6.4 });
  }
  if (images[K.leftBar]) {
    slide.addImage({ path: images[K.leftBar], ...POSITIONS.leftBar });
  }

  // Phase-alternating Top & Bottom corner decor (safe margins, zero content obstruction)
  const resolvedCorner = corner || (phase === "B" ? "phaseB" : "phaseA");
  if (resolvedCorner === "phaseA") {
    if (images[K.arc]) slide.addImage({ path: images[K.arc], x: 18.25, y: -1.98, w: 3.45, h: 3.45 });
    if (images[K.spheres]) slide.addImage({ path: images[K.spheres], x: 18.45, y: 9.65, w: 3.15, h: 3.15 });
  } else if (resolvedCorner === "phaseB") {
    if (images[K.ribbon]) slide.addImage({ path: images[K.ribbon], x: 18.40, y: -1.15, w: 2.40, h: 2.40 });
    if (images[K.splitArch]) slide.addImage({ path: images[K.splitArch], x: 18.78, y: 10.00, w: 1.70, h: 1.70 });
  } else if (resolvedCorner === "spheresBottom" && images[K.spheres]) {
    slide.addImage({ path: images[K.spheres], x: 18.45, y: 9.65, w: 3.15, h: 3.15 });
  } else if (resolvedCorner === "arcTop" && images[K.arc]) {
    slide.addImage({ path: images[K.arc], x: 18.25, y: -1.98, w: 3.45, h: 3.45 });
  } else if (resolvedCorner === "both") {
    if (images[K.arc]) slide.addImage({ path: images[K.arc], x: 18.25, y: -1.98, w: 3.45, h: 3.45 });
    if (images[K.spheres]) slide.addImage({ path: images[K.spheres], x: 18.45, y: 9.65, w: 3.15, h: 3.15 });
  }

  // Phase-alternating Heading Mark (01_starburst_12pt vs 02_starburst_8pt)
  const effHeadingIcon = headingIcon || (phase === "B" ? "star8" : "star12");
  const hdrPath = resolveIconPath(images, effHeadingIcon) || images[K.star12];
  let titleX = 1.05;
  if (starburst === "topLeft" && hdrPath) {
    slide.addImage({ path: hdrPath, x: 0.98, y: 0.56, w: 0.72, h: 0.72 });
    titleX = 1.90;
  } else if (starburst === "topRight" && hdrPath) {
    slide.addImage({ path: hdrPath, x: 17.95, y: 0.62, w: 0.76, h: 0.76 });
  }

  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: titleX,
      y: 0.38,
      w: 15.0,
      h: 0.30,
      fontFace: FONTS.bodyBold,
      fontSize: 13.5,
      bold: true,
      charSpacing: 1.5,
      color: "2563EB",
      valign: "top",
      margin: 0,
      isTextBox: true,
    });
  }

  if (title) {
    slide.addText(title, {
      x: titleX,
      y: kicker ? 0.68 : 0.48,
      w: starburst === "topRight" ? 16.4 : 17.20,
      h: 0.85,
      fontFace: FONTS.display,
      fontSize: titleSize,
      bold: true,
      color: PALETTE.ink,
      valign: "top",
      margin: 0,
      isTextBox: true,
      fit: "shrink",
    });
  }

  if (subtitle) {
    slide.addText(subtitle, {
      x: 1.05,
      y: 1.52,
      w: 17.4,
      h: 0.46,
      fontFace: FONTS.body,
      fontSize: 19,
      color: "334155",
      valign: "top",
      margin: 0,
      isTextBox: true,
      fit: "shrink",
    });
  }

  if (slideNum !== null) {
    // Subtle translucent white badge ensures high contrast even over saturated bottom-right accents
    slide.addShape("roundRect", {
      x: 18.78,
      y: 10.63,
      w: 0.56,
      h: 0.34,
      rectRadius: 0.05,
      fill: { color: "FFFFFF", transparency: 14 },
      line: { color: "DBEAFE", width: 0.75 },
    });
    slide.addText(String(slideNum).padStart(2, "0"), {
      x: 18.78,
      y: 10.63,
      w: 0.56,
      h: 0.34,
      fontFace: FONTS.bodyBold,
      fontSize: 13,
      bold: true,
      color: "334155",
      align: "center",
      valign: "middle",
      margin: 0,
      isTextBox: true,
    });
  }
}

/**
 * Numbered callout card in mid-sem Conclusion style: tinted rounded box, huge number left, accent rule, title + body.
 */
function addNumberedCard(slide, pptx, {
  x, y, w, h, num, heading, body,
  fillColor = "DCEBFF",
  accentColor = "60A5FA",
  numSize = 44,
  headSize = 19,
  bodySize = 16.5,
  chevron = false,
}) {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.07,
    fill: { color: fillColor },
    line: { color: "BFD9FC", width: 0.75 },
  });
  if (chevron) {
    slide.addShape(pptx.shapes.ISOSCELES_TRIANGLE, {
      x: x + w - 0.04, y: y + h / 2 - 0.18, w: 0.24, h: 0.36, rotate: 90,
      fill: { color: fillColor }, line: { type: "none" },
    });
  }
  const numW = Math.min(1.15, w * 0.22);
  slide.addText(num, {
    x: x + 0.12, y: y + 0.15, w: numW, h: h - 0.30,
    fontFace: FONTS.display, fontSize: numSize, bold: true,
    color: PALETTE.ink, align: "center", valign: "middle", margin: 0, isTextBox: true,
  });
  const barX = x + numW + 0.18;
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: barX, y: y + 0.26, w: 0.055, h: h - 0.52,
    fill: { color: accentColor }, line: { type: "none" },
  });
  const txtX = barX + 0.22;
  const txtW = x + w - txtX - 0.22;
  const compact = h < 1.8;
  const topPad = compact ? 0.14 : 0.24;
  const effectiveHeadSize = heading.length > (txtW < 4.5 ? 25 : 38) ? Math.min(headSize, 16.5) : headSize;
  const twoLineHead = txtW < 4.5 ? heading.length > 34 : heading.length > 48;
  const headH = twoLineHead ? (compact ? 0.56 : 0.66) : (compact ? 0.34 : 0.44);
  const bodyTop = y + topPad + headH + (compact ? 0.04 : 0.10);
  slide.addText(heading, {
    x: txtX, y: y + topPad, w: txtW, h: headH,
    fontFace: FONTS.bodyBold, fontSize: effectiveHeadSize, bold: true,
    color: PALETTE.ink, valign: "top", margin: 0, isTextBox: true, fit: "shrink",
  });
  slide.addText(body, {
    x: txtX, y: bodyTop, w: txtW, h: y + h - bodyTop - (compact ? 0.10 : 0.18),
    fontFace: FONTS.body, fontSize: bodySize,
    color: PALETTE.ink, lineSpacing: Math.round(bodySize * (compact ? 1.28 : 1.38)),
    valign: "top", margin: 0, isTextBox: true, fit: "shrink",
  });
}

/**
 * Split-header pointer card (Slide 8 Canva dataset card pattern).
 */
function addSplitCard(slide, pptx, {
  x, y, w, h, headerText, bodyTitle = "", bullets = [], bodyText = "",
  headerFill = "F4F9FD", bodyFill = "A9CDF8", bodyFontSize = 16.5,
  bulletIcon = null, images = null,
}) {
  const hdrH = Math.min(1.08, h * 0.29);
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y: y + hdrH - 0.12, w, h: h - hdrH + 0.12, rectRadius: 0.07,
    fill: { color: bodyFill }, line: { type: "none" },
  });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h: hdrH, rectRadius: 0.07,
    fill: { color: headerFill }, line: { color: "DCE7F5", width: 0.75 },
  });
  slide.addShape(pptx.shapes.ISOSCELES_TRIANGLE, {
    x: x + w / 2 - 0.22, y: y + hdrH - 0.03, w: 0.44, h: 0.22, rotate: 180,
    fill: { color: headerFill }, line: { type: "none" },
  });
  slide.addText(headerText, {
    x: x + 0.14, y: y + 0.05, w: w - 0.28, h: hdrH - 0.10,
    fontFace: FONTS.bodyBold, fontSize: 19, bold: true,
    color: PALETTE.ink, align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
  });

  let curY = y + hdrH + 0.24;
  if (bodyTitle) {
    slide.addText(bodyTitle, {
      x: x + 0.14, y: curY, w: w - 0.28, h: 0.42,
      fontFace: FONTS.bodyBold, fontSize: 19, bold: true,
      color: PALETTE.ink, align: "center", valign: "top", margin: 0, isTextBox: true, fit: "shrink",
    });
    curY += 0.46;
  }
  if (bullets && bullets.length) {
    if (bulletIcon && images) {
      addIconBullets(slide, images, {
        x: x + 0.20,
        y: curY,
        w: w - 0.36,
        h: y + h - curY - 0.15,
        items: bullets,
        bulletIcon,
        fontSize: bodyFontSize,
        lineHeight: Math.max(0.28, (bodyFontSize * 1.44) / 72),
        paraGap: 0.10,
      });
    } else {
      const runs = bullets.map((b, i) => ({
        text: b,
        options: { fontFace: FONTS.body, bullet: { indent: 18 }, breakLine: i < bullets.length - 1 },
      }));
      slide.addText(runs, {
        x: x + 0.20, y: curY, w: w - 0.36, h: y + h - curY - 0.15,
        fontSize: bodyFontSize, color: PALETTE.ink, lineSpacing: Math.round(bodyFontSize * 1.38),
        paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true,
      });
    }
  } else if (bodyText) {
    slide.addText(bodyText, {
      x: x + 0.22, y: curY, w: w - 0.44, h: y + h - curY - 0.15,
      fontFace: FONTS.body, fontSize: bodyFontSize, color: PALETTE.ink,
      lineSpacing: Math.round(bodyFontSize * 1.38), valign: "top", margin: 0, isTextBox: true,
    });
  }
}

/**
 * Rounded rectangle process / diagram node.
 */
function addFlowNode(slide, pptx, { x, y, w, h, text, highlighted = false, fontSize = 18, bold = true, fill = null, borderColor = null }) {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: fill || (highlighted ? PALETTE.boxTint : PALETTE.bg) },
    line: { color: borderColor || PALETTE.borderBlue, width: 1.5 },
  });
  slide.addText(text, {
    x: x + 0.08, y: y + 0.04, w: w - 0.16, h: h - 0.08,
    fontFace: bold ? FONTS.bodyBold : FONTS.body,
    fontSize, bold, color: PALETTE.ink,
    align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
  });
}

function addArrow(slide, pptx, x1, y1, x2, y2, color = null, width = 2.0) {
  slide.addShape(pptx.shapes.LINE, {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
    flipH: x2 < x1,
    flipV: y2 < y1,
    line: { color: color || PALETTE.borderBlue, width, endArrowType: "triangle" },
  });
}

/**
 * Greedy word-wrap line estimator for PowerPoint sans-serif fallback text boxes.
 */
function estimateWrappedLines(str, textW, fontSize) {
  const maxChars = Math.max(9, Math.floor(textW / (fontSize * 0.0074)));
  const words = str.trim().split(/\s+/);
  let lines = 1;
  let curLen = 0;
  for (const word of words) {
    if (curLen === 0) {
      curLen = word.length;
    } else if (curLen + 1 + word.length <= maxChars) {
      curLen += 1 + word.length;
    } else {
      lines++;
      curLen = word.length;
    }
  }
  return lines;
}

/**
 * Point/bullet list with crisp geometric Canva point markers instead of plain black dots.
 */
function addIconBullets(slide, images, {
  x, y, w, h, items,
  bulletIcon = "sparkle",
  fontSize = 18.5,
  lineHeight = 0.34,
  paraGap = 0.14,
}) {
  const iconPath = resolveIconPath(images, bulletIcon);
  const isCapsule = bulletIcon === "capsule";
  const iw = isCapsule ? 0.28 : 0.22;
  const ih = isCapsule ? 0.11 : 0.22;
  const textX = iconPath ? x + (isCapsule ? 0.40 : 0.34) : x;
  const textW = w - (textX - x);

  const plainStrings = items.map((it) =>
    Array.isArray(it) ? it.map((s) => (typeof s === "string" ? s : s.text)).join("") : String(it)
  );
  const naturalLineH = Math.max(lineHeight, (fontSize * 1.44) / 72);
  let effFont = fontSize;
  let effLineH = naturalLineH;
  let effGap = paraGap;

  let lineCounts = plainStrings.map((s) => estimateWrappedLines(s, textW, effFont));
  let totalH = lineCounts.reduce((a, b) => a + b * effLineH, 0) + Math.max(0, items.length - 1) * effGap;
  if (totalH > h && totalH > 0) {
    const ratio = h / totalH;
    effFont = Math.max(12, Math.floor(fontSize * Math.sqrt(ratio) * 2) / 2);
    lineCounts = plainStrings.map((s) => estimateWrappedLines(s, textW, effFont));
    effLineH = Math.max((effFont * 1.38) / 72, naturalLineH * ratio);
    effGap = Math.max(0.05, paraGap * ratio);
  }

  let curY = y;
  items.forEach((it, idx) => {
    const itemH = Math.max(effLineH, lineCounts[idx] * effLineH + 0.03);
    if (iconPath) {
      slide.addImage({
        path: iconPath,
        x,
        y: curY + (isCapsule ? 0.10 : 0.055),
        w: iw,
        h: ih,
      });
    }
    const content = Array.isArray(it)
      ? it.map((seg) =>
          typeof seg === "string"
            ? { text: seg, options: { fontFace: FONTS.body } }
            : { text: seg.text, options: { fontFace: seg.bold ? FONTS.bodyBold : FONTS.body, bold: !!seg.bold } }
        )
      : String(it);

    slide.addText(content, {
      x: textX,
      y: curY,
      w: textW,
      h: Math.min(itemH + 0.04, Math.max(0.32, y + h - curY)),
      fontFace: FONTS.body,
      fontSize: effFont,
      color: PALETTE.ink,
      lineSpacing: Math.round(effFont * 1.36),
      valign: "top",
      margin: 0,
      isTextBox: true,
      fit: "shrink",
    });
    curY += itemH + effGap;
  });
}

/**
 * Bottom full-width callout / note banner with left accent stripe + optional point icon.
 */
function addBanner(slide, pptx, { x = 1.05, y = 9.55, w = 17.9, h = 0.85, label = "", text = "", fill = "EFF6FF", accent = "60A5FA", fontSize = 17.5, icon = "capsule", images = null }) {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: fill }, line: { color: "BFDBFE", width: 1.0 },
  });
  slide.addShape(pptx.shapes.RECTANGLE, {
    x, y: y + 0.12, w: 0.08, h: h - 0.24, fill: { color: accent }, line: { type: "none" },
  });
  const iconPath = resolveIconPath(images, icon);
  const isCapsule = icon === "capsule";
  let textStart = x + 0.26;
  if (iconPath) {
    const iw = isCapsule ? 0.34 : 0.28;
    const ih = isCapsule ? 0.14 : 0.28;
    slide.addImage({ path: iconPath, x: x + 0.22, y: y + (h - ih) / 2, w: iw, h: ih });
    textStart = x + 0.22 + iw + 0.16;
  }
  const runs = [];
  if (label) runs.push({ text: label + " ", options: { fontFace: FONTS.bodyBold, bold: true, color: PALETTE.ink } });
  runs.push({ text: text, options: { fontFace: FONTS.body, bold: false, color: PALETTE.ink } });
  slide.addText(runs, {
    x: textStart, y: y + 0.08, w: x + w - textStart - 0.18, h: h - 0.16,
    fontSize, valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
  });
}

/**
 * Panel card with clean pale-blue header bar, optional header point icon, and border.
 */
function addPanelCard(slide, pptx, { x, y, w, h, title, titleFill = "EBF4FF", bodyFill = "FFFFFF", borderColor = "BFD9FC", titleSize = 20, icon = null, images = null }) {
  const th = 0.62;
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.07,
    fill: { color: bodyFill }, line: { color: borderColor, width: 1.25 },
  });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h: th, rectRadius: 0.07,
    fill: { color: titleFill }, line: { color: borderColor, width: 1.0 },
  });
  const iconPath = resolveIconPath(images, icon);
  const isCapsule = icon === "capsule";
  let titleX = x + 0.24;
  if (iconPath) {
    const iw = isCapsule ? 0.36 : 0.30;
    const ih = isCapsule ? 0.14 : 0.30;
    slide.addImage({ path: iconPath, x: x + 0.18, y: y + (th - ih) / 2, w: iw, h: ih });
    titleX = x + 0.18 + iw + 0.14;
  }
  slide.addText(title, {
    x: titleX, y: y + 0.05, w: x + w - titleX - 0.22, h: th - 0.10,
    fontFace: FONTS.bodyBold, fontSize: titleSize, bold: true,
    color: PALETTE.ink, valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
  });
  return { contentX: x + 0.28, contentY: y + th + 0.18, contentW: w - 0.56, contentH: h - th - 0.34 };
}

/**
 * Styled native PowerPoint table matching Canva's blue header palette.
 */
function addStyledTable(slide, { x, y, w, colW, rowH, headers, rows, fontSize = 16, headerFontSize = 17, boldRows = [], colAligns = null }) {
  const hdrCells = headers.map((h) => ({
    text: h,
    options: { bold: true, fontFace: FONTS.bodyBold, fontSize: headerFontSize, fill: { color: PALETTE.tableHeader }, color: PALETTE.ink, align: "center", valign: "middle" },
  }));
  const bodyRows = rows.map((row, rIdx) => {
    const bg = rIdx % 2 === 1 ? "F4F9FD" : "FFFFFF";
    const isBoldRow = boldRows.includes(rIdx);
    return row.map((val, cIdx) => {
      const align = colAligns ? colAligns[cIdx] : (cIdx === 0 ? "left" : "center");
      return {
        text: String(val),
        options: {
          bold: isBoldRow || cIdx === 0,
          fontFace: (isBoldRow || cIdx === 0) ? FONTS.bodyBold : FONTS.body,
          fontSize,
          fill: { color: bg },
          color: PALETTE.ink,
          align,
          valign: "middle",
        },
      };
    });
  });

  slide.addTable([hdrCells, ...bodyRows], {
    x, y, w, colW, rowH,
    border: { type: "solid", color: PALETTE.tableGrid, pt: 1.0 },
    margin: 0.08,
  });
}

/**
 * Clean workspace schematic frame (Laptop, Mug, Notebook, Phone) for illustrating motion/occlusion across frames!
 */
function drawWorkspaceFrame(slide, pptx, { x, y, w, h, label, phonePos = "table", occluded = false, highlightPhone = true }) {
  // Frame container
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: "F8FBFF" }, line: { color: "93C5FD", width: 1.5 },
  });
  // Frame pill badge
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 0.14, y: y + 0.12, w: 1.45, h: 0.38, rectRadius: 0.05,
    fill: { color: "DCEBFF" }, line: { color: "65A1E7", width: 1.0 },
  });
  slide.addText(label, {
    x: x + 0.14, y: y + 0.12, w: 1.45, h: 0.38,
    fontFace: FONTS.bodyBold, fontSize: 13.5, bold: true, color: PALETTE.ink, align: "center", valign: "middle", margin: 0, isTextBox: true,
  });

  // Desk surface
  const deskY = y + h * 0.44;
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 0.20, y: deskY, w: w - 0.40, h: h * 0.46, rectRadius: 0.05,
    fill: { color: "E2E8F0" }, line: { color: "CBD5E1", width: 1.0 },
  });
  slide.addText("Desk / Table", {
    x: x + 0.25, y: deskY + h * 0.33, w: 1.6, h: 0.26,
    fontFace: FONTS.body, fontSize: 11, color: "64748B", margin: 0, isTextBox: true,
  });

  // Stationary Laptop
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 0.34, y: y + h * 0.24, w: w * 0.28, h: h * 0.34, rectRadius: 0.04,
    fill: { color: "CBD5E1" }, line: { color: "64748B", width: 1.2 },
  });
  slide.addText("Laptop", {
    x: x + 0.34, y: y + h * 0.28, w: w * 0.28, h: 0.32,
    fontFace: FONTS.bodyBold, fontSize: 12, bold: true, color: PALETTE.ink, align: "center", margin: 0, isTextBox: true,
  });

  // Stationary Notebook (kept left of occluded phone spot)
  const nbW = Math.max(0.88, w * 0.28);
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x + w * 0.32, y: deskY + h * 0.18, w: nbW, h: h * 0.15,
    fill: { color: "FFFFFF" }, line: { color: "94A3B8", width: 1.0 },
  });
  slide.addText("Notebook", {
    x: x + w * 0.32, y: deskY + h * 0.18, w: nbW, h: h * 0.15,
    fontFace: FONTS.body, fontSize: 9.5, color: "334155", align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
  });

  // Phone bbox depending on state
  let px = x + w * 0.72;
  let py = deskY + h * 0.11;
  let pw = Math.max(0.72, w * 0.23);
  let ph = h * 0.18;
  if (phonePos === "near_notebook") {
    px = x + w * 0.65;
    py = deskY + h * 0.05;
  } else if (phonePos === "drawer") {
    px = x + w * 0.70;
    py = y + h * 0.77;
  }

  if (occluded) {
    // Folder / Book occluding phone
    const ocW = Math.max(pw + 0.24, 0.96);
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: px - 0.10, y: py - 0.06, w: ocW, h: ph + 0.16, rectRadius: 0.04,
      fill: { color: "94A3B8" }, line: { color: "475569", width: 1.2 },
    });
    slide.addText("Occluded\n(Phone)", {
      x: px - 0.10, y: py - 0.06, w: ocW, h: ph + 0.16,
      fontFace: FONTS.bodyBold, fontSize: 9.5, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
    });
  } else {
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: px, y: py, w: pw, h: ph, rectRadius: 0.04,
      fill: { color: highlightPhone ? "60A5FA" : "93C5FD" },
      line: { color: highlightPhone ? "1D4ED8" : "64748B", width: highlightPhone ? 2.0 : 1.0 },
    });
    slide.addText("Phone", {
      x: px, y: py, w: pw, h: ph,
      fontFace: FONTS.bodyBold, fontSize: 10.5, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
    });
  }
}

module.exports = {
  K,
  resolveIconPath,
  addContentChrome,
  addNumberedCard,
  addSplitCard,
  addFlowNode,
  addArrow,
  addIconBullets,
  addBanner,
  addPanelCard,
  addStyledTable,
  drawWorkspaceFrame,
};
