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
  drawWorkspaceFrame,
} = require("D:/Minor/presentation_rebuild/src/components.js");

function addSlides21to30(pptx, images) {
  // ── SLIDE 21 — PERSISTENT SCENE GRAPH MEMORY ──────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Shared Component — Persistent Scene Graph Memory",
      title: "Maintaining Current and Historical Information",
      subtitle: "Structured graph memory storing current object states, semantic relations, validity windows, and timestamped observation history.",
      slideNum: 21,
      phase: "B",
    });

    // 3 structured schema columns
    const cols21 = [
      [
        "Object Information (Nodes)", "F4F9FD",
        [
          "Persistent object ID",
          "Object class and attributes",
          "Last-observed bounding box",
          "Detection confidence",
          "Observation timestamp",
          "Current visibility / status",
        ],
      ],
      [
        "Relationship Info (Edges)", "EBF4FF",
        [
          "Subject object ID",
          "Relationship predicate",
          "Target object ID",
          "Relationship confidence",
          "Observation timestamp",
          "Validity interval [t_start, t_end]",
        ],
      ],
      [
        "Historical Information", "DCEBFF",
        [
          "Previous object locations",
          "Previous relationship states",
          "Time of last observation",
          "State transitions across frames",
        ],
      ],
    ];
    const cw = 5.75;
    cols21.forEach(([title, hdrFill, bullets], idx) => {
      const card = addPanelCard(s, pptx, {
        x: 1.05 + idx * (cw + 0.32), y: 2.15, w: cw, h: 4.55,
        title, titleFill: hdrFill, titleSize: 19.5, icon: "star8", images,
      });
      addIconBullets(s, images, {
        x: card.contentX, y: card.contentY, w: card.contentW, h: card.contentH,
        items: bullets, bulletIcon: "sparkle", fontSize: 18, lineHeight: 0.34, paraGap: 0.14,
      });
    });

    // Live Example Timeline strip (Phone ID: 05)
    const ex = addPanelCard(s, pptx, {
      x: 1.05, y: 6.95, w: 17.90, h: 2.15,
      title: "Example Memory Trace — Phone (ID: 05)", titleFill: "A2CBFF", bodyFill: "F8FBFF", titleSize: 19, icon: "star8", images,
    });
    addFlowNode(s, pptx, { x: ex.contentX, y: ex.contentY + 0.08, w: 4.35, h: 0.92, text: "Previously:  On Table\n(Interval: t₁ → t₂)", fontSize: 16.5 });
    addArrow(s, pptx, ex.contentX + 4.35, ex.contentY + 0.54, ex.contentX + 5.15, ex.contentY + 0.54, "2563EB", 2.5);
    addFlowNode(s, pptx, { x: ex.contentX + 5.18, y: ex.contentY + 0.08, w: 4.55, h: 0.92, text: "Currently:  In Drawer\n(Updated at frame t₃)", highlighted: true, fontSize: 16.5 });
    addArrow(s, pptx, ex.contentX + 9.73, ex.contentY + 0.54, ex.contentX + 10.53, ex.contentY + 0.54, "2563EB", 2.5);
    addFlowNode(s, pptx, { x: ex.contentX + 10.56, y: ex.contentY + 0.08, w: 6.75, h: 0.92, text: "Last Observed:  Current Timestamp\nHistory Preserved: Yes (retrievable via NLQ)", fill: "DCEBFF", borderColor: "2563EB", fontSize: 16.5 });

    addBanner(s, pptx, {
      x: 1.05, y: 9.35, w: 17.90, h: 0.90,
      label: "Objective:",
      text: "Preserve useful historical evidence without discarding information whenever the current camera frame changes.",
      fill: "EFF6FF", fontSize: 18, icon: "capsule", images,
    });
  }

  // ── SLIDE 22 — HANDLING OCCLUSION AND UNCERTAINTY ─────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Occlusion & Confidence Logic",
      title: "Avoiding Incorrect Scene Updates",
      subtitle: "A missing detection does not necessarily mean an object has left the workspace.",
      slideNum: 22,
      phase: "B",
    });

    // Top 3-frame visual occlusion progression
    const fW = 4.95;
    const fH = 2.95;
    drawWorkspaceFrame(s, pptx, { x: 1.05, y: 2.12, w: fW, h: fH, label: "t₁ — Visible", phonePos: "near_notebook" });
    addArrow(s, pptx, 6.00, 3.60, 7.50, 3.60, "2563EB", 2.5);
    drawWorkspaceFrame(s, pptx, { x: 7.52, y: 2.12, w: fW, h: fH, label: "t₂ — Occluded", phonePos: "near_notebook", occluded: true });
    addArrow(s, pptx, 12.47, 3.60, 13.97, 3.60, "2563EB", 2.5);
    drawWorkspaceFrame(s, pptx, { x: 14.00, y: 2.12, w: fW, h: fH, label: "t₃ — Reappears", phonePos: "near_notebook" });

    // 4 Object State cards across middle row
    const states22 = [
      ["Visible Object", ["Object is detected in current frame.", "Current location is updated."], "DCEBFF"],
      ["Temporarily Occluded", ["Object not currently visible.", "Previous observations are retained."], "B2D4FC"],
      ["Uncertain Observation", ["Detection/relation confidence too low.", "Avoids treating as confirmed fact."], "A9CDF8"],
      ["Confirmed Change", ["Strong new visual evidence observed.", "Safely updates state & relations."], "9DC4F2"],
    ];
    const sw = 4.25;
    states22.forEach(([hdr, buls, fill], idx) => {
      addSplitCard(s, pptx, {
        x: 1.05 + idx * (sw + 0.30), y: 5.32, w: sw, h: 2.88,
        headerText: hdr, bullets: buls, bodyFill: fill, bodyFontSize: 16,
        bulletIcon: "sparkle", images,
      });
    });

    addBanner(s, pptx, {
      x: 1.05, y: 8.42, w: 17.90, h: 0.90,
      label: "Uncertainty and Feedback:",
      text: "When a query reveals missing or uncertain information, the system can trigger targeted re-observation and update only the affected part of the graph.",
      fill: "F4F9FD", fontSize: 17, icon: "capsule", images,
    });
    addBanner(s, pptx, {
      x: 1.05, y: 9.48, w: 17.90, h: 0.80,
      label: "Key Principle:",
      text: "Preserve historical evidence and avoid confusing temporary invisibility with genuine disappearance.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 17.5, icon: "capsule", images,
    });
  }

  // ── SLIDE 23 — PIPELINE B: NATURAL LANGUAGE QUERY AND RETRIEVAL ───────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Pipeline B — Natural Language Querying",
      title: "Retrieving Objects Through Natural Language",
      subtitle: "Enables users to query both current workspace observations and persistent historical memory in plain language.",
      slideNum: 23,
      phase: "B",
    });

    // Example User Query Highlight Pill
    addBanner(s, pptx, {
      x: 1.05, y: 2.12, w: 17.90, h: 0.92,
      label: "Example User Query:",
      text: "\"Where was my phone last seen?\"",
      fill: "DCEBFF", accent: "2563EB", fontSize: 21, icon: "capsule", images,
    });

    // 4 horizontal pipeline steps
    const steps23 = [
      ["Step 1", "Query Processing", ["Identify required target object.", "Extract retrieval intent and temporal constraint."]],
      ["Step 2", "Graph Retrieval", ["Search persistent graph memory for matching object IDs.", "Retrieve current & historical locations and relations."]],
      ["Step 3", "Temporal Reasoning", ["Compare timestamps & validity intervals.", "Select most recent supported observation."]],
      ["Step 4", "Answer Generation", ["Produce a response grounded in retrieved graph evidence.", "Never fabricate unobserved claims."]],
    ];
    const sw = 4.10;
    const sg = 0.50;
    steps23.forEach(([num, head, buls], idx) => {
      const sx = 1.05 + idx * (sw + sg);
      addSplitCard(s, pptx, {
        x: sx, y: 3.30, w: sw, h: 3.75,
        headerText: `${num} — ${head}`,
        bullets: buls,
        bodyFill: idx === 2 ? "A2CBFF" : "B8D8FC",
        bodyFontSize: 17,
        bulletIcon: "sparkle",
        images,
      });
      if (idx < steps23.length - 1) {
        addArrow(s, pptx, sx + sw + 0.06, 5.18, sx + sw + sg - 0.06, 5.18, "2563EB", 2.75);
      }
    });

    addBanner(s, pptx, {
      x: 1.05, y: 7.42, w: 17.90, h: 1.25,
      label: "Example Grounded Response:",
      text: "\"Your phone was last observed on the study table at the most recent recorded observation.\"",
      fill: "F4F9FD", accent: "60A5FA", fontSize: 19.5, icon: "capsule", images,
    });
    addBanner(s, pptx, {
      x: 1.05, y: 8.92, w: 17.90, h: 1.15,
      label: "Objective:",
      text: "Provide object retrieval using accumulated visual information rather than relying exclusively on the current frame.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 19, icon: "capsule", images,
    });
  }

  // ── SLIDE 24 — PROPOSED EVALUATION FRAMEWORK ──────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Proposed Evaluation Framework",
      title: "How Will the Integrated Architecture Be Evaluated?",
      subtitle: "The static-SGG benchmark establishes our baseline; the proposed temporal architecture requires five evaluation dimensions.",
      slideNum: 24,
      phase: "A",
    });

    const dims24 = [
      ["A. Static Scene Understanding", ["Recall@20, Recall@50, Recall@100", "Mean Recall@50 (mR@50)", "Detection mAP & AP50 where comparable"], "F4F9FD"],
      ["B. Temporal Consistency", ["Consistency of relationships across frames", "Stability of persistent object identities", "Accuracy of object-state updates"], "EBF4FF"],
      ["C. Memory Correctness", ["Accuracy of stored object locations", "Correctness of historical relationships", "Handling of temporarily occluded objects"], "EBF4FF"],
      ["D. Retrieval Quality", ["Natural-language retrieval accuracy", "Correctness of last-observed locations", "Retrieval from historical scene states"], "DCEBFF"],
      ["E. Computational Efficiency", ["Relationship-model inference count", "Number of object pairs evaluated", "Update latency, FPS & peak RAM"], "C5E0FD"],
    ];

    // Top row: 3 cards (A, B, C) | Bottom row: 2 wide cards (D, E)
    const wTop = 5.75;
    dims24.slice(0, 3).forEach(([t, buls, fill], idx) => {
      const card = addPanelCard(s, pptx, {
        x: 1.05 + idx * (wTop + 0.32), y: 2.18, w: wTop, h: 3.30,
        title: t, titleFill: fill, titleSize: 19.5, icon: "star12", images,
      });
      addIconBullets(s, images, {
        x: card.contentX, y: card.contentY + 0.08, w: card.contentW, h: card.contentH - 0.12,
        items: buls, bulletIcon: "sparkle", fontSize: 19, lineHeight: 0.35, paraGap: 0.16,
      });
    });

    const wBot = 8.75;
    dims24.slice(3).forEach(([t, buls, fill], idx) => {
      const card = addPanelCard(s, pptx, {
        x: 1.05 + idx * (wBot + 0.40), y: 5.75, w: wBot, h: 3.25,
        title: t, titleFill: fill, titleSize: 20, icon: "star12", images,
      });
      addIconBullets(s, images, {
        x: card.contentX, y: card.contentY + 0.08, w: card.contentW, h: card.contentH - 0.12,
        items: buls, bulletIcon: "sparkle", fontSize: 19.5, lineHeight: 0.36, paraGap: 0.18,
      });
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.25, w: 17.90, h: 1.00,
      label: "Important:",
      text: "These temporal evaluation metrics are proposed for the next implementation stage; they are not reported as completed experimental results.",
      fill: "EFF6FF", accent: "2563EB", fontSize: 18, icon: "capsule", images,
    });
  }

  // ── SLIDE 25 — CURRENT PROJECT STATUS ─────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Current Project Status",
      title: "Completed Work vs Proposed Development",
      slideNum: 25,
      phase: "A",
    });

    const compCard = addPanelCard(s, pptx, {
      x: 1.05, y: 1.85, w: 8.75, h: 7.20,
      title: "COMPLETED  (11 Milestones Verified)", titleFill: "A2CBFF", bodyFill: "F4F9FD", titleSize: 20, icon: "star12", images,
    });
    const completed = [
      "Investigation of four static SGG models.",
      "Selection and preparation of four evaluation datasets.",
      "Common evaluation pipeline.",
      "200-image diagnostic smoke testing.",
      "Canonical predicate mapping.",
      "Model-specific prediction adapters.",
      "16 full-test model–dataset evaluations.",
      "Comparative benchmark analysis.",
      "Study of temporal SGG and persistent graph memory literature.",
      "Identification of the research gap.",
      "Proposed two-pipeline architecture.",
    ];
    addIconBullets(s, images, {
      x: compCard.contentX, y: compCard.contentY, w: compCard.contentW, h: compCard.contentH,
      items: completed, bulletIcon: "sparkle", fontSize: 17, lineHeight: 0.30, paraGap: 0.12,
    });

    const propCard = addPanelCard(s, pptx, {
      x: 10.20, y: 1.85, w: 8.75, h: 7.20,
      title: "PROPOSED / YET TO BE IMPLEMENTED  (8 Milestones)", titleFill: "E2E8F0", bodyFill: "FFFFFF", borderColor: "94A3B8", titleSize: 20, icon: "star12", images,
    });
    const proposed = [
      "Video-frame-based object association and persistent tracking.",
      "Changed-object identification.",
      "Actual selective relationship recomputation.",
      "Persistent temporal graph memory implementation.",
      "Uncertainty and occlusion handling.",
      "Natural-language retrieval integration.",
      "Temporal sequence-level evaluation.",
      "Quantitative demonstration of computational savings.",
    ];
    addIconBullets(s, images, {
      x: propCard.contentX, y: propCard.contentY + 0.08, w: propCard.contentW, h: propCard.contentH - 0.10,
      items: proposed, bulletIcon: "sparkle", fontSize: 18, lineHeight: 0.34, paraGap: 0.22,
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.28, w: 17.90, h: 0.95,
      label: "Current Stage:",
      text: "Static benchmark completed; temporal architecture established at the design and research level.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 18.5, icon: "capsule", images,
    });
  }

  // ── SLIDE 26 — CHALLENGES AND OPEN RESEARCH QUESTIONS ─────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Challenges & Open Questions",
      title: "Challenges Identified",
      slideNum: 26,
      phase: "A",
    });

    const chals = [
      ["01", "Object Identity Across Frames", "How can the system reliably identify the same object after movement, partial occlusion or reappearance?"],
      ["02", "Relationship Validity", "When should a stored semantic relationship be updated, preserved as still valid, or invalidated?"],
      ["03", "Actual Computational Savings", "How can selective recomputation avoid evaluating the complete set of subject–object pairs?"],
      ["04", "Historical Memory Management", "How should the system balance retaining previous observations with keeping a compact current representation?"],
      ["05", "Dataset & Eval Compatibility", "How can temporal performance be evaluated consistently across different datasets and prediction pipelines?"],
      ["06", "Uncertainty Handling", "How can the system avoid producing confident answers from incomplete or ambiguous visual evidence?"],
    ];
    const cw = 5.75;
    const ch = 3.25;
    chals.forEach(([num, head, body], idx) => {
      const r = Math.floor(idx / 3);
      const c = idx % 3;
      addNumberedCard(s, pptx, {
        x: 1.05 + c * (cw + 0.32),
        y: 1.90 + r * (ch + 0.32),
        w: cw, h: ch, num, heading: head, body,
        fillColor: r === 0 ? "F4F9FD" : "DCEBFF",
        numSize: 42, headSize: 19, bodySize: 17.5,
      });
    });

    addBanner(s, pptx, {
      x: 1.05, y: 9.05, w: 17.90, h: 1.15,
      label: "Research Priority:",
      text: "Ensure that the proposed efficiency improvement is measurable and does not compromise retrieval reliability.",
      fill: "DCEBFF", accent: "2563EB", fontSize: 19.5, icon: "capsule", images,
    });
  }

  // ── SLIDE 27 — FUTURE WORK ────────────────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "Future Work",
      title: "Roadmap for Further Development",
      slideNum: 27,
      phase: "A",
    });

    const road = [
      ["01", "Temporal Dataset Preparation", "Select a suitable video-based dataset and prepare sequential frame annotations."],
      ["02", "Object Association", "Implement persistent object identities and changed-object detection."],
      ["03", "Selective Relationship Gen", "Implement targeted relationship recomputation strictly for affected object pairs."],
      ["04", "Persistent Graph Memory", "Maintain structured current and historical object and relationship state."],
      ["05", "Natural Language Retrieval", "Connect query processing with graph-based retrieval and temporal reasoning."],
      ["06", "Experimental Validation", "Compare full vs selective updates on prediction quality, temporal consistency, retrieval accuracy, latency & compute saved."],
    ];
    const cw = 5.65;
    const ch = 2.95;
    const gx27 = 0.475;
    road.forEach(([num, head, body], idx) => {
      const r = Math.floor(idx / 3);
      const c = idx % 3;
      const rx = 1.05 + c * (cw + gx27);
      const ry = 1.88 + r * (ch + 0.30);
      addNumberedCard(s, pptx, {
        x: rx,
        y: ry,
        w: cw, h: ch, num, heading: head, body,
        fillColor: idx === 5 ? "C5E0FD" : (r === 0 ? "F4F9FD" : "EBF4FF"),
        numSize: 42, headSize: 18.5, bodySize: 16.5,
      });
      if (c < 2) {
        addArrow(s, pptx, rx + cw + 0.04, ry + ch / 2, rx + cw + gx27 - 0.04, ry + ch / 2, "60A5FA", 2.25);
      }
    });

    addBanner(s, pptx, {
      x: 1.05, y: 8.32, w: 17.90, h: 0.92,
      label: "Potential Sequence Resources (from Literature):",
      text: "Action Genome, OpenPVSG, MEVA, and the Dynamic House Simulator.",
      fill: "EFF6FF", fontSize: 18.5, icon: "capsule", images,
    });
    addBanner(s, pptx, {
      x: 1.05, y: 9.40, w: 17.90, h: 0.88,
      label: "Clarification:",
      text: "These are candidate resources identified in the reviewed literature, not a declaration that the final temporal dataset has already been selected.",
      fill: "F4F9FD", accent: "93C5FD", fontSize: 17, icon: "capsule", images,
    });
  }

  // ── SLIDE 28 — REFERENCES: SGG MODELS AND DATASETS ────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "References — SGG Models & Datasets (1/2)",
      title: "Research References",
      slideNum: 28,
      phase: "B",
    });

    const leftRef = addPanelCard(s, pptx, {
      x: 1.05, y: 1.95, w: 10.75, h: 8.15,
      title: "Static Scene Graph Generation Papers", titleFill: "A2CBFF", bodyFill: "FFFFFF", titleSize: 22, icon: "star8", images,
    });
    const staticPapers = [
      ["1", "REACT — Real-time Efficiency and Accuracy Compromise", "2025 · arXiv"],
      ["2", "REACT++ — Efficient Cross-Attention for Real-Time Scene Graph Generation", "2026 · arXiv"],
      ["3", "Fully Convolutional Scene Graph Generation (FCSGG)", "2021 · IEEE Xplore"],
      ["4", "Fast Contextual Scene Graph Generation with Unbiased Predicate Learning", "2023 · IEEE Xplore"],
    ];
    staticPapers.forEach(([num, txt, venue], idx) => {
      const ry = leftRef.contentY + 0.10 + idx * 1.72;
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: leftRef.contentX, y: ry, w: leftRef.contentW, h: 1.48, rectRadius: 0.06,
        fill: { color: "F4F9FD" }, line: { color: "BFD9FC", width: 1.0 },
      });
      s.addShape(pptx.shapes.OVAL, {
        x: leftRef.contentX + 0.20, y: ry + 0.42, w: 0.62, h: 0.62,
        fill: { color: "60A5FA" }, line: { type: "none" },
      });
      s.addText(num, {
        x: leftRef.contentX + 0.20, y: ry + 0.42, w: 0.62, h: 0.62,
        fontFace: FONTS.bodyBold, fontSize: 19, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true,
      });
      s.addText(
        [
          { text: txt + "\n", options: { fontFace: FONTS.bodyBold, bold: true, fontSize: 19 } },
          { text: venue, options: { fontFace: FONTS.body, fontSize: 16.5, color: "2563EB" } },
        ],
        { x: leftRef.contentX + 1.00, y: ry + 0.14, w: leftRef.contentW - 1.20, h: 1.22, color: PALETTE.ink, lineSpacing: 26, valign: "middle", margin: 0, isTextBox: true }
      );
    });

    const rightRef = addPanelCard(s, pptx, {
      x: 12.15, y: 1.95, w: 6.80, h: 8.15,
      title: "Datasets Evaluated", titleFill: "DCEBFF", bodyFill: "FFFFFF", titleSize: 22, icon: "star8", images,
    });
    const dsRefs = [
      ["5", "Visual Genome 150 (VG150)"],
      ["6", "Panoptic Scene Graph (PSG)"],
      ["7", "IndoorVG"],
      ["8", "Open Images V6"],
    ];
    dsRefs.forEach(([num, txt], idx) => {
      const ry = rightRef.contentY + 0.10 + idx * 1.72;
      s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
        x: rightRef.contentX, y: ry, w: rightRef.contentW, h: 1.48, rectRadius: 0.06,
        fill: { color: "EFF6FF" }, line: { color: "BFD9FC", width: 1.0 },
      });
      s.addShape(pptx.shapes.OVAL, {
        x: rightRef.contentX + 0.20, y: ry + 0.42, w: 0.62, h: 0.62,
        fill: { color: "2563EB" }, line: { type: "none" },
      });
      s.addText(num, {
        x: rightRef.contentX + 0.20, y: ry + 0.42, w: 0.62, h: 0.62,
        fontFace: FONTS.bodyBold, fontSize: 19, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true,
      });
      s.addText(txt, {
        x: rightRef.contentX + 1.00, y: ry + 0.15, w: rightRef.contentW - 1.20, h: 1.18,
        fontFace: FONTS.bodyBold, fontSize: 20, bold: true, color: PALETTE.ink, valign: "middle", margin: 0, isTextBox: true,
      });
    });
  }

  // ── SLIDE 29 — REFERENCES: TEMPORAL SCENE UNDERSTANDING ───────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    addContentChrome(s, images, {
      kicker: "References — Temporal Scene Understanding (2/2)",
      title: "Research References",
      subtitle: "Key literature on video-based dynamic scene graphs, temporal consistency, persistent graph memory, and embodied QA retrieval.",
      slideNum: 29,
      phase: "B",
    });

    const tempRefs = [
      ["01", "STTran (ICCV 2021)", "Spatial-Temporal Transformer for Dynamic Scene Graph Generation."],
      ["02", "TEMPURA (CVPR 2023)", "Unbiased Scene Graph Generation in Videos with memory prototypes & uncertainty attenuation."],
      ["03", "TCDSG (2024 preprint)", "Temporally Consistent Dynamic Scene Graphs via temporal matching & feedback."],
      ["04", "Scene Graph Memory (ICML 2023)", "Modeling Dynamic Environments with Scene Graph Memory for partially observable object-location prediction."],
      ["05", "VL-KnG (2024 preprint)", "Visual Scene Understanding for Navigation Goal Identification using Spatiotemporal Knowledge Graphs."],
      ["06", "GraphPad (CVPR Workshop 2025)", "Inference-Time 3D Scene Graph Updates for Embodied Question Answering."],
    ];
    const cw = 8.75;
    const ch = 2.40;
    tempRefs.forEach(([num, head, body], idx) => {
      const r = Math.floor(idx / 2);
      const c = idx % 2;
      addNumberedCard(s, pptx, {
        x: 1.05 + c * (cw + 0.40),
        y: 2.25 + r * (ch + 0.30),
        w: cw, h: ch, num, heading: head, body,
        fillColor: r % 2 === 0 ? "F4F9FD" : "DCEBFF",
        numSize: 44, headSize: 21, bodySize: 18.5,
      });
    });
  }

  // ── SLIDE 30 — THANK YOU ──────────────────────────────────────────────────
  {
    const s = pptx.addSlide({ masterName: "CANVA_CLEAN" });
    s.background = { color: PALETTE.bg };
    s.addImage({ path: images[K.leftBar], x: 0, y: 0, w: 0.402, h: 11.25 });
    s.addImage({ path: images[K.heroCircles], x: 10.8, y: 0.0, w: 8.6, h: 11.25 });
    s.addImage({ path: images[K.star12], x: 1.125, y: 0.90, w: 0.926, h: 0.926 });

    s.addText(
      [
        { text: "Thank", options: { breakLine: true } },
        { text: "You" },
      ],
      {
        x: 1.125, y: 2.35, w: 9.5, h: 3.15,
        fontFace: FONTS.display, fontSize: 104, bold: true,
        color: PALETTE.ink, lineSpacing: 118, valign: "top", margin: 0, isTextBox: true,
      }
    );

    s.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
      x: 1.125, y: 5.75, w: 4.55, h: 0.64, rectRadius: 0.08,
      fill: { color: "DCEBFF" }, line: { color: "65A1E7", width: 1.25 },
    });
    s.addText("Questions & Discussion", {
      x: 1.125, y: 5.75, w: 4.55, h: 0.64,
      fontFace: FONTS.bodyBold, fontSize: 22, bold: true, color: "1E40AF",
      align: "center", valign: "middle", margin: 0, isTextBox: true,
    });

    s.addShape(pptx.shapes.RECTANGLE, {
      x: 1.125, y: 7.45, w: 0.08, h: 1.85, fill: { color: PALETTE.accent }, line: { type: "none" },
    });
    s.addText("Object Retrieval in Dynamic Workspace Environment", {
      x: 1.50, y: 7.42, w: 9.5, h: 0.50,
      fontFace: FONTS.bodyBold, fontSize: 24, bold: true, color: PALETTE.ink, margin: 0, isTextBox: true,
    });
    s.addText(
      [
        { text: "Presented by:\n", options: { fontFace: FONTS.bodyBold, bold: true, fontSize: 20 } },
        { text: "Shivansh Kaushik  |  Nikhil Verma  |  Harman Singh  |  Mannan Bharti", options: { fontFace: FONTS.body, fontSize: 21 } },
      ],
      {
        x: 1.50, y: 8.15, w: 9.5, h: 1.15, color: PALETTE.ink, lineSpacing: 32, valign: "top", margin: 0, isTextBox: true,
      }
    );
  }
}

module.exports = { addSlides21to30 };
