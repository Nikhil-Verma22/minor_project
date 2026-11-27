# End-Semester Evaluation Presentation & Design System

This directory contains the **End-Semester Evaluation** presentation for *Object Retrieval in Dynamic Workspace Environments*, along with the reusable Canva-derived Blue & Red geometric element kits and modular `PptxGenJS` source code.

## Quick Access

| File | Description |
| :--- | :--- |
| [`out/End_Semester_Evaluation.pdf`](out/End_Semester_Evaluation.pdf) | Final 30-slide End-Semester Evaluation presentation (PDF, ready for viewing) |
| [`out/End_Semester_Evaluation.pptx`](out/End_Semester_Evaluation.pptx) | Final 30-slide editable PowerPoint deck (`20.0" × 11.25"` widescreen) |
| [`out/elements_blue_red.pdf`](out/elements_blue_red.pdf) | Blue (original) and Red (recolored) 18-asset element showcase & UI components (PDF) |
| [`out/elements_blue_red.pptx`](out/elements_blue_red.pptx) | Editable element showcase deck (PPTX) |
| [`originals/Object Retrieval (3).pdf`](originals/Object%20Retrieval%20%283%29.pdf) | Mid-Semester original reference export (PDF) |
| [`originals/Object Retrieval.pptx`](originals/Object%20Retrieval.pptx) | Mid-Semester original reference export (PPTX) |

## Directory Layout

```plaintext
presentation_rebuild/
├── README.md                  # This index
├── out/                       # Final presentation and element showcase exports (.pptx & .pdf)
├── originals/                 # Reference Mid-Semester deck exports
├── assets/                    # Reusable graphic kits (kit_blue/, kit_red/, kit_all/, kit_dims.json)
└── src/                       # Modular slide generation scripts and theme system
    ├── build_endsem_deck.js   # Main entrypoint generating End_Semester_Evaluation.pptx
    ├── build_elements_kit.js  # Generates elements_blue_red.pptx
    ├── design_system.js       # Canvas, palette, typography, and phase-based element taxonomy
    ├── components.js          # Reusable layout primitives (chrome, cards, bullets, banners, tables)
    ├── endsem_slides_01_10.js # Slides 01–10 (Title, Overview, Problem, Scope, Benchmarking Setup)
    ├── endsem_slides_11_20.js # Slides 11–20 (Results, Findings, Literature Review, Pipeline A)
    ├── endsem_slides_21_30.js # Slides 21–30 (Memory, Occlusion, Pipeline B, Status, Refs, Closing)
    └── theme.json             # Palette & font metadata passed to OOXML build scripts
```
