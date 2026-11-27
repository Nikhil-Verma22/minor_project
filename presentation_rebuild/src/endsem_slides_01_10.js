"use strict";

const { PALETTE, FONTS } = require("D:/Minor/presentation_rebuild/src/design_system.js");
const {
  K,
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
} = require("D:/Minor/presentation_rebuild/src/components.js");

function addSlides01to10(pptx, images) {
  // ── SLIDE 01 — TITLE SLIDE ────────────────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    s.background = { color: PALETTE.bg };
    s.addImage({ path: images[K.leftBar], x: 0, y: 0, w: 0.402, h: 11.25 });
    s.addImage({ path: images[K.heroCircles], x: 10.8, y: 0.0, w: 8.6, h: 11.25 });
    s.addImage({ path: images[K.star12], x: 1.125, y: 0.90, w: 0.926, h: 0.926 });

    // End-Semester badge
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: 1.125, y: 2.02, w: 4.15, h: 0.52, rectRadius: 0.08,
      fill: { color: "DCEBFF" }, line: { color: "65A1E7", width: 1.25 },
    });
    s.addText("End-Semester Evaluation", {
      x: 1.125, y: 2.02, w: 4.15, h: 0.52,
      fontFace: FONTS.bodyBold, fontSize: 18, bold: true, color: "1E40AF",
      align: "center", valign: "middle", margin: 0, isTextBox: true,
    });

    // Main Cover Title
    s.addText(
      [
        { text: "Object Retrieval in", options: { breakLine: true } },
        { text: "Dynamic Workspace", options: { breakLine: true } },
        { text: "Environment" },
      ],
      {
        x: 1.125, y: 2.75, w: 14.5, h: 4.35,
        fontFace: FONTS.display, fontSize: 74, bold: true,
        color: PALETTE.ink, lineSpacing: 98, valign: "top", margin: 0, isTextBox: true,
      }
    );

    // Presented by
    s.addShape(pptx.shapes.RECTANGLE, {
      x: 1.125, y: 7.72, w: 0.08, h: 2.38, fill: { color: PALETTE.accent }, line: { type: "none" },
    });
    s.addText("Presented by:", {
      x: 1.50, y: 7.66, w: 4.5, h: 0.46,
      fontFace: FONTS.bodyBold, fontSize: 24, bold: true, color: PALETTE.ink, valign: "top", margin: 0, isTextBox: true,
    });
    const presenters = [
      ["Shivansh Kaushik", "24103052"],
      ["Nikhil Verma", "24103048"],
      ["Harman Singh", "24103056"],
      ["Mannan Bharti", "24103046"],
    ];
    s.addText(
      presenters.map(([n], i) => ({ text: n, options: { fontFace: FONTS.body, breakLine: i < presenters.length - 1 } })),
      { x: 1.50, y: 8.20, w: 3.15, h: 1.95, fontSize: 23, color: PALETTE.ink, lineSpacing: 33, valign: "top", margin: 0, isTextBox: true }
    );
    s.addText(
      presenters.flatMap(([, r], i) => [
        { text: "–  ", options: { fontFace: FONTS.body } },
        { text: r, options: { fontFace: FONTS.bodyBold, bold: true, breakLine: i < presenters.length - 1 } },
      ]),
      { x: 4.55, y: 8.20, w: 2.35, h: 1.95, fontSize: 23, color: PALETTE.ink, lineSpacing: 33, valign: "top", margin: 0, isTextBox: true }
    );

    // Mentors
    s.addShape(pptx.shapes.RECTANGLE, {
      x: 7.20, y: 7.72, w: 0.08, h: 1.42, fill: { color: PALETTE.accent }, line: { type: "none" },
    });
    s.addText("Mentors:", {
      x: 7.58, y: 7.66, w: 4.3, h: 0.46,
      fontFace: FONTS.bodyBold, fontSize: 24, bold: true, color: PALETTE.ink, valign: "top", margin: 0, isTextBox: true,
    });
    s.addText(
      [{ text: "Dr. Sudesh Rani", options: { breakLine: true } }, { text: "Mayank Gupta" }],
      { x: 7.58, y: 8.20, w: 4.8, h: 0.96, fontFace: FONTS.body, fontSize: 23, color: PALETTE.ink, lineSpacing: 33, valign: "top", margin: 0, isTextBox: true }
    );
  }

  // ── SLIDE 02 — PROJECT OVERVIEW ───────────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Project Overview",
      title: "Object Retrieval in Dynamic Workspaces",
      slideNum: 2,
      phase: "A",
    });

    // Left column: intro text + workspace illustration + research direction
    s.addText(
      "Our project investigates how an AI system can identify, represent, remember, and retrieve objects in environments where their positions and relationships change over time.",
      {
        x: 1.05, y: 1.80, w: 7.35, h: 1.65,
        fontFace: FONTS.body, fontSize: 20, color: PALETTE.ink, lineSpacing: 29,
        valign: "top", margin: 0, isTextBox: true,
      }
    );
    s.addImage({ path: images[K.introPhoto], x: 1.05, y: 3.65, w: 7.35, h: 4.01 });

    addBanner(s, pptx, {
      x: 1.05, y: 7.95, w: 7.35, h: 2.30,
      label: "Research direction:",
      text: "Moving from independent image-based scene understanding towards a persistent, temporally updated representation of the workspace.",
      fill: "EBF4FF", fontSize: 18.5, icon: "capsule", images,
    });

    // Right column: 5 connected research component cards
    const pillars = [
      ["Scene Graph Generation (SGG)", "Representing objects and their semantic relationships."],
      ["Temporal Understanding", "Maintaining consistent information across successive frames."],
      ["Persistent Graph Memory", "Retaining current and historical object information."],
      ["Natural Language Querying (NLQ)", "Retrieving objects through human-language questions."],
      ["Computational Efficiency", "Reducing unnecessary relationship inference through selective updates."],
    ];
    const rx = 8.95;
    const rw = 9.90;
    const startY = 1.80;
    const cardH = 1.48;
    const gapY = 0.24;

    pillars.forEach(([head, desc], idx) => {
      const cy = startY + idx * (cardH + gapY);
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: rx, y: cy, w: rw, h: cardH, rectRadius: 0.07,
        fill: { color: idx === 2 ? "DCEBFF" : "F4F9FD" },
        line: { color: idx === 2 ? "65A1E7" : "BFD9FC", width: 1.25 },
      });
      s.addShape(pptx.shapes.OVAL, {
        x: rx + 0.24, y: cy + 0.42, w: 0.62, h: 0.62,
        fill: { color: "60A5FA" }, line: { type: "none" },
      });
      s.addText(String(idx + 1), {
        x: rx + 0.24, y: cy + 0.42, w: 0.62, h: 0.62,
        fontFace: FONTS.display, fontSize: 20, bold: true, color: "FFFFFF",
        align: "center", valign: "middle", margin: 0, isTextBox: true,
      });
      s.addText(head, {
        x: rx + 1.08, y: cy + 0.22, w: rw - 1.32, h: 0.42,
        fontFace: FONTS.bodyBold, fontSize: 20, bold: true, color: PALETTE.ink,
        valign: "top", margin: 0, isTextBox: true,
      });
      s.addText(desc, {
        x: rx + 1.08, y: cy + 0.68, w: rw - 1.32, h: 0.66,
        fontFace: FONTS.body, fontSize: 17.5, color: "334155", lineSpacing: 23,
        valign: "top", margin: 0, isTextBox: true,
      });
      if (idx < pillars.length - 1) {
        addArrow(s, pptx, rx + 0.55, cy + cardH, rx + 0.55, cy + cardH + gapY, "60A5FA", 1.75);
      }
    });
  }

  // ── SLIDE 03 — REFINED PROBLEM STATEMENT ──────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Refined Problem Statement",
      title: "Problem Statement",
      subtitle: "A conventional SGG model represents objects and relationships in an individual image. A dynamic workspace introduces additional challenges:",
      slideNum: 3,
      phase: "A",
    });

    // Left panel: 5 challenges
    const box = addPanelCard(s, pptx, {
      x: 1.05, y: 2.25, w: 8.65, h: 6.45,
      title: "Limitations of Single-Frame Scene Graphs", titleSize: 20, icon: "star12", images,
    });
    const challenges = [
      ["Objects may ", { text: "move", bold: true }, " between successive observations."],
      ["Objects may become ", { text: "temporarily occluded or disappear", bold: true }, " from the camera's view."],
      ["Relationships between objects may change while other relationships remain ", { text: "unaffected", bold: true }, "."],
      ["Conventional frame-wise processing does not inherently maintain ", { text: "persistent object identities or historical scene information", bold: true }, "."],
      ["Recomputing the complete scene graph for every frame introduces ", { text: "unnecessary computational overhead", bold: true }, "."],
    ];
    addIconBullets(s, images, {
      x: box.contentX, y: box.contentY, w: box.contentW, h: box.contentH,
      items: challenges, bulletIcon: "sparkle", fontSize: 18.5, lineHeight: 0.35, paraGap: 0.16,
    });

    // Right panel: 3-frame desk progression diagram
    const diag = addPanelCard(s, pptx, {
      x: 10.05, y: 2.25, w: 8.90, h: 6.45,
      title: "Dynamic Workspace Across Consecutive Frames", titleSize: 20, icon: "star12", images,
    });
    const fw = 2.55;
    const fh = 3.25;
    const fy = diag.contentY + 0.15;
    drawWorkspaceFrame(s, pptx, { x: diag.contentX, y: fy, w: fw, h: fh, label: "Frame t₁", phonePos: "table" });
    addArrow(s, pptx, diag.contentX + fw, fy + fh / 2, diag.contentX + fw + 0.34, fy + fh / 2);
    drawWorkspaceFrame(s, pptx, { x: diag.contentX + fw + 0.34, y: fy, w: fw, h: fh, label: "Frame t₂", phonePos: "near_notebook" });
    addArrow(s, pptx, diag.contentX + fw * 2 + 0.34, fy + fh / 2, diag.contentX + fw * 2 + 0.68, fy + fh / 2);
    drawWorkspaceFrame(s, pptx, { x: diag.contentX + fw * 2 + 0.68, y: fy, w: fw, h: fh, label: "Frame t₃", phonePos: "near_notebook", occluded: true });

    // Edge summary pills under the 3 frames
    const tags = [
      "Phone — On — Table\nLaptop & Notebook unchanged",
      "Phone — Near — Notebook\nOnly Phone edges changed",
      "Phone — Occluded\nPrior states retained in memory",
    ];
    tags.forEach((t, idx) => {
      const tx = diag.contentX + idx * (fw + 0.34);
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: tx, y: fy + fh + 0.25, w: fw, h: 1.40, rectRadius: 0.06,
        fill: { color: idx === 0 ? "F4F9FD" : "DCEBFF" },
        line: { color: "BFD9FC", width: 1.0 },
      });
      s.addText(t, {
        x: tx + 0.10, y: fy + fh + 0.32, w: fw - 0.20, h: 1.25,
        fontFace: FONTS.bodyBold, fontSize: 13.5, bold: true, color: PALETTE.ink,
        align: "center", valign: "middle", margin: 0, isTextBox: true,
      });
    });

    // Bottom Challenge Banner
    addBanner(s, pptx, {
      x: 1.05, y: 9.05, w: 17.90, h: 1.25,
      label: "Core Challenge:",
      text: "Maintain a temporally consistent scene representation, update only affected relationships wherever possible, and support natural-language retrieval over both current and historical observations.",
      fill: "DCEBFF", fontSize: 19, icon: "capsule", images,
    });
  }

  // ── SLIDE 04 — RESEARCH OBJECTIVES ────────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Research Objectives",
      title: "Objectives",
      slideNum: 4,
      phase: "A",
    });

    const objs = [
      ["01", "Static SGG Benchmarking", "Evaluate REACT, REACT++, FCSGG, and Fast Contextual SGG across multiple datasets."],
      ["02", "Evaluation Standardisation", "Develop a common evaluation pipeline with canonical predicate mapping and model-specific adapters."],
      ["03", "Temporal Scene Understanding", "Study methods for maintaining object and relationship information across successive frames."],
      ["04", "Persistent Visual Memory", "Design a graph-based memory capable of storing current and historical scene states."],
      ["05", "Selective Relationship Updates", "Investigate recomputing only relationships involving objects whose states have changed."],
      ["06", "Natural Language Retrieval", "Enable questions about objects and their last-observed locations to be answered using stored scene information."],
    ];
    const cw = 5.75;
    const ch = 3.70;
    const gx = 0.32;
    const gy = 0.42;
    objs.forEach(([num, head, body], idx) => {
      const r = Math.floor(idx / 3);
      const c = idx % 3;
      addNumberedCard(s, pptx, {
        x: 1.05 + c * (cw + gx),
        y: 1.90 + r * (ch + gy),
        w: cw,
        h: ch,
        num,
        heading: head,
        body,
        fillColor: idx % 2 === 0 ? "DCEBFF" : "EBF4FF",
        numSize: 48,
        headSize: 20,
        bodySize: 18,
      });
    });
  }

  // ── SLIDE 05 — PROJECT SCOPE AND RESEARCH DIRECTION ───────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Project Scope & Research Direction",
      title: "From Static Scene Graphs to Dynamic Object Retrieval",
      slideNum: 5,
      phase: "A",
    });

    // Left focus panel
    const p1 = addPanelCard(s, pptx, {
      x: 1.05, y: 1.95, w: 8.35, h: 5.55,
      title: "Initial Focus — Static Scene Understanding", titleFill: "F4F9FD", titleSize: 21, icon: "star12", images,
    });
    const f1 = [
      "Object detection and relationship prediction.",
      "Comparison of SGG model architectures (REACT, REACT++, FCSGG, Fast Contextual SGG).",
      "Dataset compatibility and standardised evaluation across four benchmarks.",
    ];
    addIconBullets(s, images, {
      x: p1.contentX, y: p1.contentY + 0.14, w: p1.contentW, h: p1.contentH - 0.24,
      items: f1, bulletIcon: "sparkle", fontSize: 20.5, lineHeight: 0.42, paraGap: 0.26,
    });

    addArrow(s, pptx, 9.48, 4.72, 10.42, 4.72, "60A5FA", 3.5);

    // Right focus panel
    const p2 = addPanelCard(s, pptx, {
      x: 10.50, y: 1.95, w: 8.45, h: 5.55,
      title: "Extended Focus — Temporal Scene Understanding", titleFill: "A2CBFF", bodyFill: "F8FBFF", titleSize: 21, icon: "star12", images,
    });
    const f2 = [
      "Persistent object association across successive frames.",
      "Identification of changed vs. stationary objects.",
      "Selective relationship recomputation on affected pairs.",
      "Historical scene graph memory with timestamps.",
      "Natural-language retrieval over current & past states.",
    ];
    addIconBullets(s, images, {
      x: p2.contentX, y: p2.contentY + 0.10, w: p2.contentW, h: p2.contentH - 0.18,
      items: f2, bulletIcon: "sparkle", fontSize: 19.5, lineHeight: 0.38, paraGap: 0.18,
    });

    // Target outcome + Important note
    addBanner(s, pptx, {
      x: 1.05, y: 7.80, w: 17.90, h: 1.25,
      label: "Target outcome:",
      text: "A general architecture connecting temporal SGG, persistent object-level graph memory, computationally meaningful selective updates, and natural-language querying.",
      fill: "DCEBFF", fontSize: 19, icon: "capsule", images,
    });
    addBanner(s, pptx, {
      x: 1.05, y: 9.25, w: 17.90, h: 0.95,
      label: "Scope Note:",
      text: "The current phase establishes the benchmark and architecture design. The complete integrated architecture remains a proposed research direction.",
      fill: "F4F9FD", accent: "93C5FD", fontSize: 17.5, icon: "capsule", images,
    });
  }

  // ── SLIDE 06 — WORK COMPLETED SO FAR ──────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Work Completed So Far",
      title: "Summary of Completed Work",
      subtitle: "Systematic progression from model analysis and dataset curation through 16 full-test benchmark runs and two-pipeline design.",
      slideNum: 6,
      phase: "B",
    });

    const steps = [
      ["01", "SGG Model Investigation", ["Selected four SGG models for systematic evaluation.", "Studied detection, proposal and relationship-prediction pathways."]],
      ["02", "Dataset Preparation", ["Prepared evaluation cohorts from four selected datasets.", "Established compatible test subsets across vocabularies."]],
      ["03", "Evaluation Framework", ["Developed a common evaluation pipeline.", "Conducted 200-image smoke tests & standardised outputs."]],
      ["04", "Benchmark Execution", ["Completed 16 full-test model–dataset evaluations.", "Verified metrics across PSG, VG150, IndoorVG & Open Images V6."]],
      ["05", "Temporal Research", ["Studied temporal SGG, graph memory and NLQ retrieval literature.", "Identified the research gap for selective relationship updates."]],
      ["06", "Architecture Design", ["Developed a proposed two-pipeline architecture for scene graph updating and natural-language retrieval."]],
    ];

    const cw = 2.70;
    const ch = 6.00;
    const gx = 0.34;
    steps.forEach(([num, title, bullets], idx) => {
      const sx = 1.05 + idx * (cw + gx);
      const sy = 2.30;

      addSplitCard(s, pptx, {
        x: sx, y: sy, w: cw, h: ch,
        headerText: `${num}\n${title}`,
        bullets,
        headerFill: "F4F9FD",
        bodyFill: idx >= 3 ? "B2D4FC" : "C5E0FD",
        bodyFontSize: 16.5,
        bulletIcon: "sparkle",
        images,
      });

      if (idx < steps.length - 1) {
        addArrow(s, pptx, sx + cw + 0.04, sy + ch / 2, sx + cw + gx - 0.04, sy + ch / 2, "2563EB", 2.75);
      }
    });

    addBanner(s, pptx, {
      x: 1.05, y: 8.68, w: 17.90, h: 1.38,
      label: "Progression:",
      text: "Model Study → Dataset Preparation → Evaluation Pipeline → 16-Run Benchmark → Temporal Research → Two-Pipeline Architecture Design.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 19, icon: "capsule", images,
    });
  }

  // ── SLIDE 07 — STATIC SGG BENCHMARKING ────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Static SGG Benchmarking",
      title: "Models Selected for Evaluation",
      subtitle: "Four distinct Scene Graph Generation paradigms were benchmarked to establish baseline detection, recall, and efficiency trade-offs.",
      slideNum: 7,
      phase: "B",
    });

    const models = [
      ["REACT", "Real-Time Two-Stage", "Real-time scene graph generation with decoupled detection & dynamic candidate selection."],
      ["REACT++", "Cross-Attention Fusion", "Efficient feature extraction (DAMP/AIFI) and asymmetric relationship prediction (CARPE)."],
      ["FCSGG", "One-Stage Center-Based", "Fully convolutional, center-based detection using Relation Affinity Fields (RAFs)."],
      ["Fast Contextual SGG", "Prior-Guided Filtering", "Contextual reasoning filters candidate relationship pairs before visual verification."],
    ];
    const cw = 4.25;
    const gx = 0.30;
    models.forEach(([name, tag, desc], idx) => {
      const mx = 1.05 + idx * (cw + gx);
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: mx, y: 2.30, w: cw, h: 4.10, rectRadius: 0.08,
        fill: { color: "FFFFFF" }, line: { color: "93C5FD", width: 1.5 },
      });
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: mx, y: 2.30, w: cw, h: 1.02, rectRadius: 0.08,
        fill: { color: PALETTE.tableHeader }, line: { color: "93C5FD", width: 1.0 },
      });
      s.addText(name, {
        x: mx + 0.15, y: 2.40, w: cw - 0.30, h: 0.82,
        fontFace: FONTS.display, fontSize: 24, bold: true, color: PALETTE.ink,
        align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink",
      });
      // Mini architecture badge
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: mx + 0.35, y: 3.55, w: cw - 0.70, h: 0.56, rectRadius: 0.06,
        fill: { color: "EBF4FF" }, line: { color: "65A1E7", width: 1.0 },
      });
      s.addText(tag, {
        x: mx + 0.35, y: 3.55, w: cw - 0.70, h: 0.56,
        fontFace: FONTS.bodyBold, fontSize: 15.5, bold: true, color: "1E40AF",
        align: "center", valign: "middle", margin: 0, isTextBox: true,
      });
      s.addText(desc, {
        x: mx + 0.25, y: 4.35, w: cw - 0.50, h: 1.85,
        fontFace: FONTS.body, fontSize: 18, color: PALETTE.ink, lineSpacing: 26,
        align: "center", valign: "top", margin: 0, isTextBox: true,
      });
    });

    // Bottom normalized panels: Benchmarking Dimensions + Purpose
    const leftP = addPanelCard(s, pptx, {
      x: 1.05, y: 6.75, w: 10.60, h: 3.45,
      title: "Benchmarking Dimensions", titleFill: "EBF4FF", titleSize: 21, icon: "star8", images,
    });
    const dimsList = [
      "Object detection quality (Det mAP across comparable detectors).",
      "Relationship prediction coverage (R@50 and R@100).",
      "Predicate-category fairness (Mean Recall mR@50 and harmonic mean Project H).",
      "Cross-dataset generalisation across four scene distributions.",
    ];
    addIconBullets(s, images, {
      x: leftP.contentX, y: leftP.contentY, w: leftP.contentW, h: leftP.contentH,
      items: dimsList, bulletIcon: "sparkle", fontSize: 18, lineHeight: 0.34, paraGap: 0.12,
    });

    const rightP = addPanelCard(s, pptx, {
      x: 11.95, y: 6.75, w: 7.00, h: 3.45,
      title: "Benchmarking Purpose", titleFill: "DCEBFF", bodyFill: "F4F9FD", titleSize: 21, icon: "star8", images,
    });
    s.addText(
      "Identify the strongest, most compatible SGG backbone components and understand failure modes before designing the temporal update and visual memory architecture.",
      { x: rightP.contentX, y: rightP.contentY + 0.12, w: rightP.contentW, h: rightP.contentH - 0.20, fontFace: FONTS.body, fontSize: 20, color: PALETTE.ink, lineSpacing: 30, valign: "top", margin: 0, isTextBox: true }
    );
  }

  // ── SLIDE 08 — DATASETS AND EVALUATION SCOPE ──────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Datasets & Evaluation Scope",
      title: "Datasets Used",
      subtitle: "Four datasets were selected to evaluate SGG performance across open-world, panoptic, indoor-specific, and large-scale scenes.",
      slideNum: 8,
      phase: "B",
    });

    addStyledTable(s, {
      x: 1.05, y: 2.08, w: 17.90,
      colW: [7.50, 5.20, 5.20],
      rowH: 0.48,
      headers: ["Dataset", "Full Test Split", "Eligible Cohort"],
      rows: [
        ["Visual Genome VG150", "31,876", "26,446"],
        ["Panoptic Scene Graph (PSG)", "2,186", "2,177"],
        ["IndoorVG", "4,403", "4,403"],
        ["Open Images V6", "20,803", "5,123"],
      ],
      fontSize: 16.5,
      headerFontSize: 17,
      colAligns: ["left", "right", "right"],
    });

    const dsCards = [
      ["Visual Genome VG150", "26,446 eligible images", ["150 object classes", "50 relationship predicates", "Object attributes & relations"]],
      ["Panoptic Scene Graph", "2,177 eligible images", ["Objects & background regions", "Mask-grounded relationships", "Reduced label ambiguity"]],
      ["IndoorVG", "4,403 eligible images", ["Indoor scene focus", "Furniture & room-object relations", "Aligned with desk workspaces"]],
      ["Open Images V6", "5,123 eligible images", ["Large-scale real-world images", "Object boxes, masks & relations", "Challenging open distribution"]],
    ];
    const dw = 4.25;
    const dgap = 0.30;
    dsCards.forEach(([hdr, cnt, buls], idx) => {
      addSplitCard(s, pptx, {
        x: 1.05 + idx * (dw + dgap),
        y: 4.92,
        w: dw,
        h: 4.18,
        headerText: hdr,
        bodyTitle: cnt,
        bullets: buls,
        bodyFill: "A9CDF8",
        bodyFontSize: 16.5,
        bulletIcon: "sparkle",
        images,
      });
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.35, w: 17.90, h: 0.92,
      label: "Note:",
      text: "The eligible cohort contains images with annotations suitable for relationship evaluation. Therefore, it may be smaller than the complete test split.",
      fill: "EFF6FF", fontSize: 17, icon: "capsule", images,
    });
  }

  // ── SLIDE 09 — COMMON EVALUATION PIPELINE ─────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Common Evaluation Pipeline",
      title: "Establishing a Reliable Benchmark",
      subtitle: "Before executing full-scale comparisons, every model pathway was verified through a staged validation pipeline.",
      slideNum: 9,
      phase: "B",
    });

    // Stage 1 card
    const st1 = addPanelCard(s, pptx, {
      x: 1.05, y: 2.25, w: 7.60, h: 4.45,
      title: "Stage 1 — 200-Image Smoke Tests", titleFill: "F4F9FD", titleSize: 21, icon: "star8", images,
    });
    s.addText("Fixed 200-image subsets verified:", {
      x: st1.contentX, y: st1.contentY, w: st1.contentW, h: 0.42,
      fontFace: FONTS.bodyBold, fontSize: 19, bold: true, color: PALETTE.ink, margin: 0, isTextBox: true,
    });
    const smokeItems = [
      "Model adapters and prediction decoding.",
      "Bounding-box coordinate conventions.",
      "Object-class and predicate mapping.",
      "Evaluation code and metric calculations.",
    ];
    addIconBullets(s, images, {
      x: st1.contentX, y: st1.contentY + 0.56, w: st1.contentW, h: st1.contentH - 0.60,
      items: smokeItems, bulletIcon: "sparkle", fontSize: 18.5, lineHeight: 0.34, paraGap: 0.14,
    });

    // Validation checkpoint in center
    addArrow(s, pptx, 8.68, 4.45, 9.15, 4.45, "60A5FA", 2.5);
    addFlowNode(s, pptx, {
      x: 9.15, y: 3.72, w: 1.70, h: 1.46,
      text: "Validation\nCheckpoint", highlighted: true, fontSize: 16.5,
    });
    addArrow(s, pptx, 10.85, 4.45, 11.32, 4.45, "60A5FA", 2.5);

    // Stage 2 card
    const st2 = addPanelCard(s, pptx, {
      x: 11.35, y: 2.25, w: 7.60, h: 4.45,
      title: "Stage 2 — Full-Test Evaluation", titleFill: "A2CBFF", bodyFill: "F8FBFF", titleSize: 21, icon: "star8", images,
    });
    s.addText("Complete eligible test cohorts executed:", {
      x: st2.contentX, y: st2.contentY, w: st2.contentW, h: 0.42,
      fontFace: FONTS.bodyBold, fontSize: 19, bold: true, color: PALETTE.ink, margin: 0, isTextBox: true,
    });
    const fullItems = [
      "4 validated SGG model adapters.",
      "4 dataset cohorts (VG150, PSG, IndoorVG, Open Images V6).",
      "Consistent Recall, mRecall, Det mAP & Project H metrics.",
      "Forms the authoritative static-SGG benchmark.",
    ];
    addIconBullets(s, images, {
      x: st2.contentX, y: st2.contentY + 0.56, w: st2.contentW, h: st2.contentH - 0.60,
      items: fullItems, bulletIcon: "sparkle", fontSize: 18.5, lineHeight: 0.34, paraGap: 0.14,
    });

    // Important distinction + Final outcome
    addBanner(s, pptx, {
      x: 1.05, y: 7.05, w: 17.90, h: 1.85,
      label: "Important Distinction:",
      text: "Smoke tests were strictly diagnostic validation runs — full-test evaluations form the actual static-SGG benchmark. Smoke-test metrics were never substituted for full-test results.",
      fill: "EFF6FF", fontSize: 20, icon: "capsule", images,
    });
    addBanner(s, pptx, {
      x: 1.05, y: 9.15, w: 17.90, h: 1.05,
      label: "Final Outcome:",
      text: "A complete evaluation matrix consisting of 4 models × 4 datasets = 16 full-test runs.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 20, icon: "capsule", images,
    });
  }

  // ── SLIDE 10 — PREDICATE MAPPING AND MODEL ADAPTERS ───────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Predicate Mapping & Model Adapters",
      title: "Standardising Different SGG Outputs",
      subtitle: "Different models and datasets use incompatible vocabularies, box formats, and decoding outputs. A shared layer normalises all predictions.",
      slideNum: 10,
      phase: "B",
    });

    // Top standardisation flow diagram (y = 2.15..4.95) with orthogonal merge bus
    const modelLabels = ["REACT Output", "REACT++ Output", "FCSGG Output", "Fast Contextual SGG"];
    modelLabels.forEach((lbl, idx) => {
      const my = 2.15 + idx * 0.72;
      addFlowNode(s, pptx, { x: 1.05, y: my, w: 3.95, h: 0.56, text: lbl, fontSize: 16 });
      s.addShape(pptx.shapes.LINE, {
        x: 5.00, y: my + 0.28, w: 0.65, h: 0,
        line: { color: "60A5FA", width: 2.0 },
      });
    });
    // Vertical collector bus line + single horizontal arrow into Standardisation Layer
    s.addShape(pptx.shapes.LINE, {
      x: 5.65, y: 2.43, w: 0, h: 2.16,
      line: { color: "60A5FA", width: 2.0 },
    });
    addArrow(s, pptx, 5.65, 3.51, 6.28, 3.51, "2563EB", 2.5);

    addFlowNode(s, pptx, {
      x: 6.30, y: 2.22, w: 5.45, h: 2.60,
      text: "Standardisation Layer\n\nCanonical Predicate Mapping\n+\nModel-Specific Adapters",
      highlighted: true, fontSize: 18.5,
    });
    addArrow(s, pptx, 11.75, 3.52, 12.60, 3.52, "2563EB", 2.75);

    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: 12.65, y: 2.22, w: 6.30, h: 2.60, rectRadius: 0.06,
      fill: { color: "F4F9FD" }, line: { color: "2563EB", width: 1.5 },
    });
    s.addText("Unified Prediction Representation", {
      x: 12.85, y: 2.38, w: 5.90, h: 0.44,
      fontFace: FONTS.bodyBold, fontSize: 18.5, bold: true, color: PALETTE.ink,
      align: "center", valign: "middle", margin: 0, isTextBox: true,
    });
    const outFlds = [
      "Subject & Object Classes",
      "Standardised Bounding Boxes",
      "Canonical Relationship Label",
      "Prediction Confidence Score",
    ];
    addIconBullets(s, images, {
      x: 13.05, y: 2.95, w: 5.55, h: 1.72,
      items: outFlds, bulletIcon: "sparkle", fontSize: 16, lineHeight: 0.28, paraGap: 0.08,
    });

    // Bottom 2 detail cards
    const p1 = addPanelCard(s, pptx, {
      x: 1.05, y: 5.30, w: 8.75, h: 3.85,
      title: "1. Canonical Predicate Mapping", titleFill: "DCEBFF", titleSize: 20, icon: "star8", images,
    });
    const mItems = [
      "Extracted predicate vocabularies from all four datasets.",
      "Aligned semantically equivalent labels to a shared canonical vocabulary.",
      "Reviewed ambiguous predicates instead of naive string matching.",
    ];
    addIconBullets(s, images, {
      x: p1.contentX, y: p1.contentY, w: p1.contentW, h: 1.95,
      items: mItems, bulletIcon: "sparkle", fontSize: 17.5, lineHeight: 0.30, paraGap: 0.10,
    });
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: p1.contentX, y: 8.18, w: p1.contentW, h: 0.72, rectRadius: 0.06,
      fill: { color: "EFF6FF" }, line: { color: "65A1E7", width: 1.2 },
    });
    s.addText("Example:  wearing  →  has_on  →  Common canonical concept", {
      x: p1.contentX + 0.15, y: 8.18, w: p1.contentW - 0.30, h: 0.72,
      fontFace: FONTS.bodyBold, fontSize: 16.5, bold: true, color: "1E40AF", align: "center", valign: "middle", margin: 0, isTextBox: true,
    });

    const p2 = addPanelCard(s, pptx, {
      x: 10.20, y: 5.30, w: 8.75, h: 3.85,
      title: "2. Model-Specific Adapters", titleFill: "DCEBFF", titleSize: 20, icon: "star8", images,
    });
    const aItems = [
      "Converted raw outputs into subject/object classes, boxes, relations & confidence.",
      "Handled class index offsets and predicate vocabulary mappings cleanly.",
      "Normalised bounding-box coordinate formats and model decoding differences.",
    ];
    addIconBullets(s, images, {
      x: p2.contentX, y: p2.contentY, w: p2.contentW, h: p2.contentH,
      items: aItems, bulletIcon: "sparkle", fontSize: 18, lineHeight: 0.34, paraGap: 0.14,
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.40, w: 17.90, h: 0.88,
      label: "Outcome:",
      text: "A standardised evaluation interface while preserving differences in the underlying model architectures and proposal pathways.",
      fill: "EFF6FF", fontSize: 17.5, icon: "capsule", images,
    });
  }
}

module.exports = { addSlides01to10 };
