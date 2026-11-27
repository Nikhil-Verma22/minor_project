"use strict";

const { initPresentation } = require("D:/Minor/presentation_rebuild/src/design_system.js");
const { addSlides01to10 } = require("D:/Minor/presentation_rebuild/src/endsem_slides_01_10.js");
const { addSlides11to20 } = require("D:/Minor/presentation_rebuild/src/endsem_slides_11_20.js");
const { addSlides21to30 } = require("D:/Minor/presentation_rebuild/src/endsem_slides_21_30.js");

const pptx = new PptxGenJS();
initPresentation(pptx);
pptx.title = "Object Retrieval in Dynamic Workspace Environment — End-Semester Evaluation";
pptx.author = "Shivansh Kaushik, Nikhil Verma, Harman Singh, Mannan Bharti";
pptx.subject = "End-Semester Evaluation Presentation";

addSlides01to10(pptx, images);
addSlides11to20(pptx, images);
addSlides21to30(pptx, images);

pptx.writeFile({ fileName: "End_Semester_Evaluation.pptx" });
