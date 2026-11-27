"use strict";

const { PALETTE, FONTS } = require("D:/Minor/presentation_rebuild/src/design_system.js");
const {
  K,
  addContentChrome,
  addNumberedCard,
  addFlowNode,
  addArrow,
  addIconBullets,
  addBanner,
  addPanelCard,
  addStyledTable,
  drawWorkspaceFrame,
} = require("D:/Minor/presentation_rebuild/src/components.js");

function addSlides11to20(pptx, images) {
  // ── SLIDE 11 — BENCHMARK RESULTS: REACT AND REACT++ ───────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Full-Test Benchmark Results (1/2)",
      title: "Comparative Results — REACT & REACT++",
      slideNum: 11,
      phase: "A",
    });

    addStyledTable(s, {
      x: 1.05, y: 1.80, w: 17.90,
      colW: [3.0, 2.9, 2.2, 2.4, 2.4, 2.4, 2.6],
      rowH: 0.56,
      headers: ["Model", "Dataset", "Det mAP", "R@50", "R@100", "mR@50", "Project H"],
      rows: [
        ["REACT", "PSG", "38.16", "31.72", "33.26", "21.71", "25.78"],
        ["REACT", "VG150", "19.93", "29.60", "32.88", "14.04", "19.05"],
        ["REACT", "IndoorVG", "23.69", "31.13", "35.93", "21.79", "25.64"],
        ["REACT", "OpenImages", "3.05", "7.86", "8.90", "9.29", "8.51"],
        ["REACT++", "PSG", "41.20", "36.29", "37.91", "25.09", "29.67"],
        ["REACT++", "VG150", "20.39", "30.56", "34.58", "16.36", "21.31"],
        ["REACT++", "IndoorVG", "23.69", "31.11", "35.93", "22.89", "26.38"],
        ["REACT++", "OpenImages", "3.89", "0.47", "0.55", "1.32", "0.70"],
      ],
      fontSize: 17,
      headerFontSize: 18,
      boldRows: [4, 5, 6],
      colAligns: ["left", "left", "right", "right", "right", "right", "right"],
    });

    const obs = addPanelCard(s, pptx, {
      x: 1.05, y: 7.05, w: 17.90, h: 2.30,
      title: "Key Observations", titleFill: "EBF4FF", titleSize: 19.5, icon: "star12", images,
    });
    const obsList = [
      "REACT++ achieves higher relationship recall on PSG (36.29% R@50) and VG150 (30.56% R@50) and the highest Project H across PSG, VG150, and IndoorVG.",
      "Both models achieve identical detection mAP (23.69%) and similar recall on IndoorVG, while REACT++ improves mR@50 (22.89% vs 21.79%).",
      "REACT++ drops sharply on Open Images V6 relationship prediction (0.70 Project H) despite a slightly higher detection mAP (3.89% vs 3.05%).",
    ];
    addIconBullets(s, images, {
      x: obs.contentX, y: obs.contentY - 0.04, w: obs.contentW, h: obs.contentH + 0.04,
      items: obsList, bulletIcon: "sparkle", fontSize: 16.5, lineHeight: 0.30, paraGap: 0.10,
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.55, w: 17.90, h: 0.80,
      label: "Metric Note:",
      text: "Values are percentages. Det mAP is reported where a comparable detection result is available. Project H is the harmonic mean of R@50 and mR@50.",
      fill: "EFF6FF", fontSize: 16, icon: "capsule", images,
    });
  }

  // ── SLIDE 12 — BENCHMARK RESULTS: FCSGG AND FAST CONTEXTUAL SGG ───────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Full-Test Benchmark Results (2/2)",
      title: "Comparative Results — FCSGG & Fast Contextual SGG",
      slideNum: 12,
      phase: "A",
    });

    addStyledTable(s, {
      x: 1.05, y: 1.80, w: 17.90,
      colW: [3.6, 2.7, 2.2, 2.3, 2.3, 2.3, 2.5],
      rowH: 0.56,
      headers: ["Model", "Dataset", "Det mAP", "R@50", "R@100", "mR@50", "Project H"],
      rows: [
        ["FCSGG", "PSG", "8.72", "2.21", "2.60", "1.58", "1.84"],
        ["FCSGG", "VG150", "9.88", "13.22", "15.83", "2.48", "4.18"],
        ["FCSGG", "IndoorVG", "3.78", "4.49", "5.40", "2.21", "2.97"],
        ["FCSGG", "OpenImages", "1.57", "1.50", "1.70", "1.96", "1.70"],
        ["Fast Contextual SGG", "PSG", "32.53", "30.10", "31.31", "21.92", "25.37"],
        ["Fast Contextual SGG", "VG150", "18.74", "26.05", "29.88", "14.36", "18.52"],
        ["Fast Contextual SGG", "IndoorVG", "7.97", "11.66", "13.80", "9.32", "10.36"],
        ["Fast Contextual SGG", "OpenImages", "N/A", "12.75", "12.95", "13.01", "12.88"],
      ],
      fontSize: 17,
      headerFontSize: 18,
      boldRows: [4, 7],
      colAligns: ["left", "left", "right", "right", "right", "right", "right"],
    });

    const obs = addPanelCard(s, pptx, {
      x: 1.05, y: 7.05, w: 17.90, h: 2.30,
      title: "Key Observations", titleFill: "EBF4FF", titleSize: 19.5, icon: "star12", images,
    });
    const obsList = [
      "FCSGG produced lower relationship-recall results across the evaluated configurations (1.70%–4.18% Project H).",
      "Fast Contextual SGG demonstrated competitive relationship prediction on PSG (25.37% Project H) and achieved the strongest Project H on Open Images V6 (12.88%).",
      "Det AP is marked N/A for Fast Contextual SGG on Open Images V6 because evaluation used a restricted cached proposal set without a full-detector score.",
    ];
    addIconBullets(s, images, {
      x: obs.contentX, y: obs.contentY - 0.04, w: obs.contentW, h: obs.contentH + 0.04,
      items: obsList, bulletIcon: "sparkle", fontSize: 16.5, lineHeight: 0.30, paraGap: 0.10,
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.55, w: 17.90, h: 0.80,
      label: "Metric Note:",
      text: "Values are percentages. N/A indicates that a comparable detection result was unavailable, not a zero score.",
      fill: "EFF6FF", fontSize: 16, icon: "capsule", images,
    });
  }

  // ── SLIDE 13 — OVERALL BENCHMARK FINDINGS ─────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Overall Benchmark Findings",
      title: "What Did the Benchmark Reveal?",
      slideNum: 13,
      phase: "A",
    });

    const findings = [
      ["01", "No Universal Best Model", "Model performance varies significantly across datasets and evaluation metrics."],
      ["02", "REACT++ Shows Strong Overall Results", "REACT++ achieved the highest Project H on PSG, VG150 and IndoorVG among the evaluated models."],
      ["03", "Fast Contextual SGG Dataset Strength", "Achieved the highest Project H on Open Images V6 (12.88%) and competitive results on PSG."],
      ["04", "FCSGG Shows Lower Performance", "Relationship-prediction recall and Project H were comparatively low across all evaluated datasets."],
      ["05", "Evaluation Compatibility Matters", "Dataset vocabularies, class mappings and proposal pathways influence model comparisons."],
    ];
    const cw = 8.75;
    const ch = 1.98;
    findings.slice(0, 4).forEach(([n, h, b], i) => {
      const r = Math.floor(i / 2);
      const c = i % 2;
      addNumberedCard(s, pptx, {
        x: 1.05 + c * (cw + 0.40),
        y: 1.85 + r * (ch + 0.25),
        w: cw, h: ch, num: n, heading: h, body: b,
        fillColor: i === 1 ? "DCEBFF" : "F4F9FD",
        numSize: 40, headSize: 19, bodySize: 17,
      });
    });

    // Bottom left #05 card + bottom right Decision Card
    addNumberedCard(s, pptx, {
      x: 1.05, y: 6.31, w: 8.75, h: 2.15,
      num: findings[4][0], heading: findings[4][1], body: findings[4][2],
      fillColor: "F4F9FD", numSize: 40, headSize: 19, bodySize: 17,
    });

    const decCard = addPanelCard(s, pptx, {
      x: 10.20, y: 6.31, w: 8.75, h: 2.15,
      title: "Research Insight", titleFill: "A2CBFF", bodyFill: "EFF6FF", titleSize: 20, icon: "star12", images,
    });
    s.addText(
      "Selecting an SGG model requires balancing relationship prediction quality, category-wise fairness, detection quality, and computational requirements.",
      { x: decCard.contentX, y: decCard.contentY - 0.04, w: decCard.contentW, h: decCard.contentH, fontFace: FONTS.body, fontSize: 17.5, color: PALETTE.ink, lineSpacing: 25, valign: "top", margin: 0, isTextBox: true }
    );

    addBanner(s, pptx, {
      x: 1.05, y: 8.78, w: 17.90, h: 1.40,
      label: "Current Research Decision:",
      text: "REACT++ is a promising candidate for further investigation, but model selection must remain subject to dataset compatibility and evaluation consistency.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 19.5, icon: "capsule", images,
    });
  }

  // ── SLIDE 14 — FROM STATIC TO TEMPORAL SCENE GRAPH GENERATION ─────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Transition to Dynamic Workspaces",
      title: "Why Do We Need Temporal SGG?",
      slideNum: 14,
      phase: "B",
    });

    // Left: Static vs Limitation
    const stP = addPanelCard(s, pptx, {
      x: 1.05, y: 1.85, w: 6.10, h: 5.15,
      title: "Static Scene Graph (Single Frame)", titleFill: "F4F9FD", titleSize: 20, icon: "star8", images,
    });
    s.addText("Represents objects and relationships inside one isolated image.", {
      x: stP.contentX, y: stP.contentY, w: stP.contentW, h: 0.78,
      fontFace: FONTS.body, fontSize: 18, color: PALETTE.ink, lineSpacing: 25, margin: 0, isTextBox: true,
    });
    // Single graph triple example
    addFlowNode(s, pptx, { x: stP.contentX, y: stP.contentY + 0.95, w: 1.65, h: 0.65, text: "Phone", highlighted: true, fontSize: 16 });
    addArrow(s, pptx, stP.contentX + 1.65, stP.contentY + 1.27, stP.contentX + 2.15, stP.contentY + 1.27);
    addFlowNode(s, pptx, { x: stP.contentX + 2.15, y: stP.contentY + 0.95, w: 1.25, h: 0.65, text: "On", fontSize: 16 });
    addArrow(s, pptx, stP.contentX + 3.40, stP.contentY + 1.27, stP.contentX + 3.90, stP.contentY + 1.27);
    addFlowNode(s, pptx, { x: stP.contentX + 3.90, y: stP.contentY + 0.95, w: 1.65, h: 0.65, text: "Table", fontSize: 16 });

    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: stP.contentX, y: stP.contentY + 1.95, w: stP.contentW, h: 2.05, rectRadius: 0.06,
      fill: { color: "FFF1F2" }, line: { color: "FECDD3", width: 1.0 },
    });
    s.addText(
      [
        { text: "Limitation of Independent Processing:\n", options: { fontFace: FONTS.bodyBold, bold: true } },
        { text: "Processing every frame independently does not inherently preserve object identity or historical relationships across time.", options: { fontFace: FONTS.body } },
      ],
      { x: stP.contentX + 0.18, y: stP.contentY + 2.12, w: stP.contentW - 0.36, h: 1.75, fontSize: 17, color: PALETTE.ink, lineSpacing: 25, valign: "top", margin: 0, isTextBox: true }
    );

    // Right: Dynamic 3-frame illustration + edges
    const dyP = addPanelCard(s, pptx, {
      x: 7.45, y: 1.85, w: 11.50, h: 5.15,
      title: "Dynamic Scene Graph — Maintaining State Across Frames", titleFill: "DCEBFF", titleSize: 20, icon: "star8", images,
    });
    const fW = 3.25;
    const fH = 2.85;
    const fx0 = dyP.contentX;
    const fy0 = dyP.contentY;
    drawWorkspaceFrame(s, pptx, { x: fx0, y: fy0, w: fW, h: fH, label: "Frame 1", phonePos: "table" });
    addArrow(s, pptx, fx0 + fW, fy0 + fH / 2, fx0 + fW + 0.56, fy0 + fH / 2);
    drawWorkspaceFrame(s, pptx, { x: fx0 + fW + 0.56, y: fy0, w: fW, h: fH, label: "Frame 2", phonePos: "near_notebook" });
    addArrow(s, pptx, fx0 + fW * 2 + 0.56, fy0 + fH / 2, fx0 + fW * 2 + 1.12, fy0 + fH / 2);
    drawWorkspaceFrame(s, pptx, { x: fx0 + fW * 2 + 1.12, y: fy0, w: fW, h: fH, label: "Frame 3", phonePos: "drawer" });

    const edgeLabels = [
      "Phone — On — Table",
      "Phone — Near — Notebook",
      "Phone — In — Drawer",
    ];
    edgeLabels.forEach((lbl, idx) => {
      addFlowNode(s, pptx, {
        x: fx0 + idx * (fW + 0.56), y: fy0 + fH + 0.28, w: fW, h: 0.72,
        text: lbl, highlighted: true, fontSize: 16,
      });
    });

    // Bottom Research Direction strip (5 pills)
    const dirP = addPanelCard(s, pptx, {
      x: 1.05, y: 7.30, w: 17.90, h: 2.95,
      title: "Our Temporal Research Direction", titleFill: "EBF4FF", titleSize: 20, icon: "star8", images,
    });
    const dirs = [
      "Maintain persistent object identities",
      "Detect changes across frames",
      "Update affected relationships",
      "Preserve valid unchanged info",
      "Retain historical observations",
    ];
    const pw = 3.30;
    dirs.forEach((d, idx) => {
      addFlowNode(s, pptx, {
        x: dirP.contentX + idx * (pw + 0.22),
        y: dirP.contentY + 0.15,
        w: pw,
        h: 1.45,
        text: d,
        highlighted: idx === 2 || idx === 4,
        fontSize: 17.5,
      });
    });
  }

  // ── SLIDE 15 — LITERATURE REVIEW: TEMPORAL SCENE GRAPH GENERATION ─────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Literature Review — Temporal SGG",
      title: "Existing Approaches to Dynamic Scene Understanding",
      slideNum: 15,
      phase: "B",
    });

    // Generously sized Literature Table so canvas is normalized cleanly
    addStyledTable(s, {
      x: 1.05, y: 1.88, w: 17.90,
      colW: [4.20, 9.40, 4.30],
      rowH: 1.02,
      headers: ["Paper", "Technique / Contribution", "Dataset"],
      rows: [
        ["STTran (ICCV 2021)", "Spatial encoding and temporal decoding for dynamic scene graph generation from video sequences.", "Action Genome"],
        ["TEMPURA (CVPR 2023)", "Temporal consistency modelling, memory prototypes, and uncertainty attenuation across frames.", "Action Genome"],
        ["TCDSG (2024)", "Temporal matching and feedback mechanisms to maintain subject–object relationship tracklets.", "Action Genome,\nOpenPVSG, MEVA"],
      ],
      fontSize: 18.5,
      headerFontSize: 20,
      colAligns: ["left", "left", "center"],
    });

    const ins = addPanelCard(s, pptx, {
      x: 1.05, y: 6.25, w: 10.30, h: 3.95,
      title: "Research Insights", titleFill: "EBF4FF", titleSize: 21, icon: "star8", images,
    });
    const insBullets = [
      "Temporal context improves relationship continuity across video frames.",
      "Memory prototypes help retain useful relational information.",
      "Relationship tracklets maintain subject–object links over time.",
      "Uncertainty handling is vital when visual observations are incomplete.",
    ];
    addIconBullets(s, images, {
      x: ins.contentX, y: ins.contentY + 0.08, w: ins.contentW, h: ins.contentH - 0.14,
      items: insBullets, bulletIcon: "sparkle", fontSize: 18.5, lineHeight: 0.36, paraGap: 0.16,
    });

    const lim = addPanelCard(s, pptx, {
      x: 11.65, y: 6.25, w: 7.30, h: 3.95,
      title: "Identified Limitation for Our Direction", titleFill: "DCEBFF", bodyFill: "F4F9FD", titleSize: 21, icon: "star8", images,
    });
    s.addText(
      "These approaches address important temporal-SGG components, but our target additionally requires computationally meaningful selective relationship updates and persistent retrieval over historical workspace information.",
      { x: lim.contentX, y: lim.contentY + 0.14, w: lim.contentW, h: lim.contentH - 0.20, fontFace: FONTS.body, fontSize: 20, color: PALETTE.ink, lineSpacing: 31, valign: "top", margin: 0, isTextBox: true }
    );
  }

  // ── SLIDE 16 — LITERATURE REVIEW: GRAPH MEMORY & RETRIEVAL ────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Literature Review — Graph Memory & NLQ",
      title: "Persistent Scene Understanding & Query-Based Reasoning",
      titleSize: 38,
      slideNum: 16,
      phase: "B",
    });

    addStyledTable(s, {
      x: 1.05, y: 1.98, w: 17.90,
      colW: [5.20, 8.50, 4.20],
      rowH: 1.02,
      headers: ["Paper", "Technique / Contribution", "Dataset / Environment"],
      rows: [
        ["Modeling Dynamic Environments with Scene Graph Memory (ICML 2023)", "Maintains an accumulated representation in a partially observable dynamic graph and predicts object-location links.", "Dynamic House Simulator"],
        ["VL-KnG (2024 preprint)", "Builds persistent spatiotemporal knowledge graphs and supports question answering through graph retrieval.", "OpenEQA, NaVQA,\nWalkieKnowledge"],
        ["GraphPad (CVPR Workshop 2025)", "Allows inference-time scene graph and memory updates through targeted operations.", "OpenEQA; HM3D &\nScanNet scene data"],
      ],
      fontSize: 18,
      headerFontSize: 20,
      colAligns: ["left", "left", "center"],
    });

    const ins = addPanelCard(s, pptx, {
      x: 1.05, y: 6.35, w: 10.30, h: 3.85,
      title: "Research Insights", titleFill: "EBF4FF", titleSize: 21, icon: "star8", images,
    });
    const insBullets = [
      "Scene graph memory retains observations beyond the current camera frame.",
      "Spatiotemporal knowledge graphs support persistent object state tracking and language-based retrieval.",
      "Query-driven graph updates provide a principled way to handle missing or uncertain information.",
    ];
    addIconBullets(s, images, {
      x: ins.contentX, y: ins.contentY + 0.12, w: ins.contentW, h: ins.contentH - 0.18,
      items: insBullets, bulletIcon: "sparkle", fontSize: 19, lineHeight: 0.36, paraGap: 0.20,
    });

    const foc = addPanelCard(s, pptx, {
      x: 11.65, y: 6.35, w: 7.30, h: 3.85,
      title: "Our Research Focus", titleFill: "DCEBFF", bodyFill: "F4F9FD", titleSize: 21, icon: "star8", images,
    });
    s.addText(
      "Connecting persistent scene graph memory and spatiotemporal query answering with object-level change detection and selective relationship recomputation inside one unified architecture.",
      { x: foc.contentX, y: foc.contentY + 0.18, w: foc.contentW, h: foc.contentH - 0.25, fontFace: FONTS.body, fontSize: 20.5, color: PALETTE.ink, lineSpacing: 31, valign: "top", margin: 0, isTextBox: true }
    );
  }

  // ── SLIDE 17 — KEY RESEARCH GAP ───────────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Key Research Gap",
      title: "What Is Still Missing?",
      subtitle: "Existing work addresses individual pieces, leaving a clear gap in unifying these six capabilities efficiently:",
      slideNum: 17,
      phase: "A",
    });

    const gaps = [
      ["01", "Persistent Object Identity", "Maintaining the identity of general objects across changing visual observations."],
      ["02", "Historical Graph Memory", "Storing current and previous scene states with timestamps and validity information."],
      ["03", "Selective Relation Recalculation", "Identifying changed objects and recomputing only the affected relationships."],
      ["04", "Actual Computational Savings", "Reducing relationship-model inference rather than filtering after complete inference."],
      ["05", "Historical NL Retrieval", "Answering questions using both current and previously observed relationships."],
      ["06", "Uncertainty & Occlusion Handling", "Distinguishing temporary visibility loss from genuine object disappearance."],
    ];
    const cw = 5.75;
    const ch = 2.95;
    gaps.forEach(([num, head, body], idx) => {
      const r = Math.floor(idx / 3);
      const c = idx % 3;
      addNumberedCard(s, pptx, {
        x: 1.05 + c * (cw + 0.32),
        y: 2.20 + r * (ch + 0.32),
        w: cw, h: ch, num, heading: head, body,
        fillColor: r === 0 ? "F4F9FD" : "DCEBFF",
        numSize: 44, headSize: 19, bodySize: 17.5,
      });
    });

    addBanner(s, pptx, {
      x: 1.05, y: 8.85, w: 17.90, h: 1.35,
      label: "Identified Research Opportunity:",
      text: "An architecture integrating temporal SGG, persistent object-level graph memory, selective updates and natural-language retrieval, evaluated for both retrieval quality and computational efficiency.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 19, icon: "capsule", images,
    });
  }

  // ── SLIDE 18 — PROPOSED INTEGRATED ARCHITECTURE ───────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Proposed Integrated Architecture",
      title: "System Architecture",
      subtitle: "Two connected pipelines sharing a persistent scene graph memory: continuous graph updates (Pipeline A) and language retrieval (Pipeline B).",
      slideNum: 18,
      phase: "A",
    });

    // PIPELINE A header strip & 6 horizontal nodes
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: 1.05, y: 2.15, w: 17.90, h: 2.30, rectRadius: 0.08,
      fill: { color: "F8FBFF" }, line: { color: "93C5FD", width: 1.25 },
    });
    s.addText("PIPELINE A — Scene Graph Construction and Updating", {
      x: 1.30, y: 2.25, w: 10.0, h: 0.42,
      fontFace: FONTS.bodyBold, fontSize: 18, bold: true, color: "1E40AF", margin: 0, isTextBox: true,
    });
    const pipeA = [
      "1. RGB Video\nFrames",
      "2. Object Detection\n& Tracking",
      "3. Initial Scene Graph\nGeneration",
      "4. Changed Object\nIdentification",
      "5. Selective Relation\nGeneration",
      "6. Scene Graph\nMemory Update",
    ];
    const nwA = 2.65;
    pipeA.forEach((txt, idx) => {
      const nx = 1.30 + idx * (nwA + 0.30);
      addFlowNode(s, pptx, { x: nx, y: 2.82, w: nwA, h: 1.32, text: txt, highlighted: idx >= 3, fontSize: 15.5 });
      if (idx < pipeA.length - 1) {
        addArrow(s, pptx, nx + nwA, 3.48, nx + nwA + 0.30, 3.48);
      }
    });

    // Bi-directional arrows between Pipeline A & Shared Memory
    addArrow(s, pptx, 10.0, 4.45, 10.0, 5.05, "2563EB", 3.0);

    // SHARED MEMORY HUB
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: 1.05, y: 5.08, w: 17.90, h: 1.85, rectRadius: 0.08,
      fill: { color: "DCEBFF" }, line: { color: "2563EB", width: 2.0 },
    });
    s.addText("SHARED COMPONENT — Persistent Scene Graph Memory", {
      x: 1.30, y: 5.18, w: 17.40, h: 0.44,
      fontFace: FONTS.display, fontSize: 21, bold: true, color: PALETTE.ink, align: "center", margin: 0, isTextBox: true,
    });
    const memTags = [
      "Current scene representation",
      "Historical object observations",
      "Relationship information",
      "Timestamps & confidence",
      "Last-known object locations",
    ];
    memTags.forEach((m, idx) => {
      addFlowNode(s, pptx, {
        x: 1.35 + idx * 3.48, y: 5.78, w: 3.32, h: 0.90,
        text: m, fill: "FFFFFF", borderColor: "65A1E7", fontSize: 15.5,
      });
    });

    // Arrow from Shared Memory down to Pipeline B
    addArrow(s, pptx, 10.0, 6.93, 10.0, 7.50, "2563EB", 3.0);

    // PIPELINE B
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: 1.05, y: 7.52, w: 17.90, h: 1.85, rectRadius: 0.08,
      fill: { color: "F4F9FD" }, line: { color: "93C5FD", width: 1.25 },
    });
    s.addText("PIPELINE B — Natural Language Querying", {
      x: 1.30, y: 7.60, w: 10.0, h: 0.40,
      fontFace: FONTS.bodyBold, fontSize: 18, bold: true, color: "1E40AF", margin: 0, isTextBox: true,
    });
    const pipeB = [
      "1. User Question",
      "2. Query Processing",
      "3. Graph Retrieval & Reasoning",
      "4. Natural-Language Answer",
    ];
    const nwB = 3.95;
    pipeB.forEach((txt, idx) => {
      const nx = 1.35 + idx * (nwB + 0.50);
      addFlowNode(s, pptx, { x: nx, y: 8.12, w: nwB, h: 1.02, text: txt, highlighted: idx === 3, fontSize: 17 });
      if (idx < pipeB.length - 1) {
        addArrow(s, pptx, nx + nwB, 8.63, nx + nwB + 0.50, 8.63);
      }
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.55, w: 17.90, h: 0.80,
      label: "Central Idea:",
      text: "Pipeline A continuously updates the scene representation, while Pipeline B retrieves relevant information from the same persistent memory.",
      fill: "EFF6FF", fontSize: 17, icon: "capsule", images,
    });
  }

  // ── SLIDE 19 — PIPELINE A: INITIAL SCENE GRAPH CONSTRUCTION ───────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Pipeline A — First Frame (t = 1)",
      title: "Understanding the First Frame",
      subtitle: "The first frame establishes the baseline persistent representation of the workspace from which subsequent frame updates are performed.",
      slideNum: 19,
      phase: "A",
    });

    // Top horizontal flow: RGB Image -> Detected Objects -> Graph
    addFlowNode(s, pptx, {
      x: 1.05, y: 2.25, w: 5.20, h: 2.00,
      text: "RGB Workspace Image\n(t = 1 Conventional Camera)", fill: "F4F9FD", fontSize: 19.5,
    });
    addArrow(s, pptx, 6.25, 3.25, 7.10, 3.25, "2563EB", 2.75);
    addFlowNode(s, pptx, {
      x: 7.12, y: 2.25, w: 5.55, h: 2.00,
      text: "Detected Objects + BBoxes\n[Laptop, Pot, Plant, Phone (ID: 05)]", highlighted: true, fontSize: 19.5,
    });
    addArrow(s, pptx, 12.67, 3.25, 13.52, 3.25, "2563EB", 2.75);
    addFlowNode(s, pptx, {
      x: 13.55, y: 2.25, w: 5.40, h: 2.00,
      text: "Initial Scene Graph\n(Subject — Predicate — Object Edges)", fill: "DCEBFF", borderColor: "2563EB", fontSize: 19.5,
    });

    // 5 Detailed Step cards
    const steps19 = [
      ["Step 1", "RGB Frame Acquisition", "Capture an image from the workspace using a conventional RGB camera."],
      ["Step 2", "Object Detection", "Identify visible objects with object classes, bounding boxes & confidence scores."],
      ["Step 3", "Object Association", "Assign persistent object identifiers to all detected objects."],
      ["Step 4", "Initial Relation Gen", "Evaluate relevant object pairs and generate semantic relationships."],
      ["Step 5", "Graph Initialisation", "Store object nodes, relation edges, attributes, spatial boxes & confidence values."],
    ];
    const sw = 3.30;
    const sg = 0.35;
    steps19.forEach(([num, hdr, desc], idx) => {
      const sx = 1.05 + idx * (sw + sg);
      const p = addPanelCard(s, pptx, {
        x: sx, y: 4.65, w: sw, h: 3.95,
        title: `${num} — ${hdr}`, titleFill: idx === 4 ? "A2CBFF" : "EBF4FF", titleSize: 16.5,
      });
      s.addText(desc, {
        x: p.contentX, y: p.contentY + 0.16, w: p.contentW, h: p.contentH - 0.24,
        fontFace: FONTS.body, fontSize: 19.5, color: PALETTE.ink, lineSpacing: 30,
        valign: "top", margin: 0, isTextBox: true,
      });
      if (idx < steps19.length - 1) {
        addArrow(s, pptx, sx + sw + 0.04, 6.62, sx + sw + sg - 0.04, 6.62, "2563EB", 2.5);
      }
    });

    addBanner(s, pptx, {
      x: 1.05, y: 8.95, w: 17.90, h: 1.15,
      label: "Outcome:",
      text: "An initial graph representation of the workspace from which subsequent frame updates can be performed incrementally.",
      fill: "EFF6FF", fontSize: 19.5, icon: "capsule", images,
    });
  }

  // ── SLIDE 20 — PIPELINE A: SELECTIVE RELATIONSHIP UPDATES ─────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Pipeline A — Selective Updating (t ≥ 2)",
      title: "Updating Only What Has Changed",
      subtitle: "Avoid regenerating the complete relationship graph whenever a new frame arrives by recomputing only pairs linked to changed objects.",
      slideNum: 20,
      phase: "A",
    });

    // Left 5 numbered vertical steps
    const steps20 = [
      ["01", "Detect and Associate", "Match newly observed objects with existing persistent object identities."],
      ["02", "Identify Changes", "Detect objects that are newly appeared, relocated, occluded, or missing."],
      ["03", "Identify Affected Pairs", "Determine strictly which subject–object pairs involve the changed objects."],
      ["04", "Selective Relation Gen", "Recompute relationships only for affected pairs wherever technically supported."],
      ["05", "Update Persistent Graph", "Add new relations, update changed edges, invalidate stale links, keep unchanged ones."],
    ];
    steps20.forEach(([num, hdr, desc], idx) => {
      addNumberedCard(s, pptx, {
        x: 1.05, y: 2.12 + idx * 1.41, w: 9.65, h: 1.30,
        num, heading: hdr, body: desc,
        fillColor: idx === 3 ? "DCEBFF" : "F4F9FD",
        numSize: 32, headSize: 17.5, bodySize: 15.2,
      });
    });

    // Right: Graph diff schematic (moved node + affected edges vs preserved edges)
    const gBox = addPanelCard(s, pptx, {
      x: 11.05, y: 2.15, w: 7.90, h: 6.92,
      title: "Selective Graph Update Illustration (t ≥ 2)", titleFill: "DCEBFF", bodyFill: "F8FBFF", titleSize: 19.5, icon: "star12", images,
    });
    const gx0 = gBox.contentX + 0.15;
    const gy0 = gBox.contentY + 0.15;

    // Stationary nodes (Laptop, Table, Notebook, Mug)
    addFlowNode(s, pptx, { x: gx0, y: gy0, w: 2.2, h: 0.78, text: "Laptop (ID: 01)\nUnchanged", fill: "FFFFFF", borderColor: "94A3B8", fontSize: 14.5 });
    addFlowNode(s, pptx, { x: gx0 + 4.8, y: gy0, w: 2.2, h: 0.78, text: "Mug (ID: 03)\nUnchanged", fill: "FFFFFF", borderColor: "94A3B8", fontSize: 14.5 });
    addFlowNode(s, pptx, { x: gx0, y: gy0 + 2.8, w: 2.2, h: 0.78, text: "Table (ID: 02)\nUnchanged", fill: "FFFFFF", borderColor: "94A3B8", fontSize: 14.5 });
    addFlowNode(s, pptx, { x: gx0 + 4.8, y: gy0 + 2.8, w: 2.2, h: 0.78, text: "Notebook (ID: 04)\nUnchanged", fill: "FFFFFF", borderColor: "94A3B8", fontSize: 14.5 });

    // Preserved static edges (subtle gray)
    addArrow(s, pptx, gx0 + 2.2, gy0 + 0.39, gx0 + 4.8, gy0 + 0.39, "94A3B8", 1.5);
    addArrow(s, pptx, gx0 + 1.1, gy0 + 0.78, gx0 + 1.1, gy0 + 2.8, "94A3B8", 1.5);
    addArrow(s, pptx, gx0 + 2.2, gy0 + 3.19, gx0 + 4.8, gy0 + 3.19, "94A3B8", 1.5);

    // Moved center node highlighted in deep blue
    addFlowNode(s, pptx, {
      x: gx0 + 2.35, y: gy0 + 1.38, w: 2.35, h: 0.90,
      text: "Phone (ID: 05)\nRELOCATED", fill: "60A5FA", borderColor: "1D4ED8", fontSize: 15.5,
    });
    // Affected bright blue arrows
    addArrow(s, pptx, gx0 + 2.55, gy0 + 2.28, gx0 + 1.75, gy0 + 2.80, "2563EB", 2.75);
    addArrow(s, pptx, gx0 + 4.50, gy0 + 2.28, gx0 + 5.25, gy0 + 2.80, "2563EB", 2.75);

    // Legend at bottom of diagram card
    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: gx0, y: gy0 + 4.05, w: 7.05, h: 1.55, rectRadius: 0.06,
      fill: { color: "FFFFFF" }, line: { color: "BFD9FC", width: 1.0 },
    });
    s.addText(
      [
        { text: "Blue Highlighted Edges: ", options: { fontFace: FONTS.bodyBold, bold: true, color: "1D4ED8" } },
        { text: "Recomputed because Phone (ID: 05) moved.\n", options: { fontFace: FONTS.body, color: PALETTE.ink } },
        { text: "Gray Preserved Edges: ", options: { fontFace: FONTS.bodyBold, bold: true, color: "475569" } },
        { text: "Retained directly without redundant SGG inference.", options: { fontFace: FONTS.body, color: PALETTE.ink } },
      ],
      { x: gx0 + 0.18, y: gy0 + 4.20, w: 6.70, h: 1.25, fontSize: 15.5, lineSpacing: 24, valign: "middle", margin: 0, isTextBox: true }
    );

    addBanner(s, pptx, {
      x: 1.05, y: 9.32, w: 17.90, h: 0.90,
      label: "Core Research Contribution:",
      text: "Reduce unnecessary relationship-model computation while retaining a temporally consistent scene representation.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 18.5, icon: "capsule", images,
    });
  }
}

module.exports = { addSlides11to20 };
