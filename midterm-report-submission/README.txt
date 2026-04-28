Pacific 1940 Battle Analyzer - Midterm Report
==============================================

FILES IN THIS FOLDER
--------------------
  main.tex                Main LaTeX source (CVPR format)
  cvpr.sty                CVPR 2026 author-kit style (eso-pic inlined)
  ieeenat_fullname.bst    Bibliography style (CVPR 2026 author kit)
  references.bib          BibTeX bibliography (5 entries)
  figures/                Image assets used by main.tex
    battlefield_detection.png    End-to-end perception output
                                 (detection boxes + ViT classes +
                                 JP/US faction colours) on a held-out
                                 board photo. Introduction, Fig 1.
    crops_classification.png     Twelve per-piece crops with ViT
                                 confidence and HSV pixel counts.
                                 Data section, Fig 3.
    owl_raw_labels.png           OWL-ViT raw output with noisy
                                 per-prompt labels. Methods, Fig 4
                                 (left panel).
    annotated_cv_boxes_green.jpg Class-agnostic detection boxes
                                 after filtering. Methods, Fig 4
                                 (right panel).
    application_screenshot.png   Deployed FastAPI backend in
                                 Best-Composition mode. Experiments,
                                 Fig 5.
    ui_states.png                (unused) Earlier UI composite,
                                 retained for reference but not
                                 included in main.tex.
  README.txt              This file.

The Introduction summary figure (Figure 2) is drawn inline with TikZ
inside main.tex, so no extra image file is needed for it.

HOW TO COMPILE
--------------
All required style files (cvpr.sty, ieeenat_fullname.bst) and image
assets (figures/) are already in this folder. Just run:

  pdflatex main.tex
  bibtex   main
  pdflatex main.tex
  pdflatex main.tex

On Windows you can use MiKTeX (https://miktex.org/) or TeX Live.

Overleaf (https://overleaf.com) also works: zip the contents of this
folder (not the folder itself) and upload via "New Project ->
Upload Project". Compiler must be set to pdfLaTeX (not XeLaTeX or
LuaLaTeX - cvpr.sty is written for pdfLaTeX).

If MiKTeX prompts for a missing package on first run, the only likely
candidate is `silence` (used internally by cvpr.sty to suppress
cosmetic font warnings). It is not strictly required - the document
will still compile without it, just with a noisier log.

REMAINING WORK FOR FINAL REPORT (May 8)
---------------------------------------
The midterm reflects an MVP scope-down. Remaining work is now
limited to two evaluations and a brief dataset disclosure (all
flagged in red in main.tex):

  1. Perception accuracy on the 13 held-out board photos.
     One hand-labelling pass yields detection precision/recall
     against OWL-ViT and per-class top-1 plus a confusion matrix
     on the predicted ViT classes.

  2. Recommendation quality on the same 13 boards.
     Simulate 1000 battles for the originally fielded mix and the
     system's recommended mix at the same IPC budget; report the
     mean win-rate delta and the fraction of boards on which the
     recommendation strictly improves. This experiment also
     discharges the end-to-end evaluation promised in the
     feasibility report.

  3. Dataset disclosure paragraph.
     Per-class crop counts, collection protocol, and train/val
     split for the cropped-units dataset.

Cut from earlier planned-work lists (no longer red-promised):
- Detector ablation (OWL-ViT base/large/OWLv2)
- CNN-vs-ViT classifier ablation (reframed as deliberate
  architectural pivot in section 4 scope-refinement)
- Reliability diagram and per-stratum MAE
- Inference-optimisation before/after numbers
- MLP batched scoring as a separate deliverable
- BLR production integration as a final-report deliverable
  (kept as future-work mention only)
- User study

QUICK REFERENCE - HEADLINE NUMBERS
------------------------------------
  Unit vocabulary:     8  (7 attacker-eligible + 1 defender-only AA)
  Detector:            OWL-ViT large  (google/owlvit-large-patch14,
                       used as a class-agnostic localiser; per-prompt
                       labels are discarded)
  Classifier:          ViT-Small/16  (vit_small_patch16_224,
                       fully fine-tuned)
  MLP architecture:    15 -> 256 -> 256 -> 128 -> 64 -> 1, BN+ReLU
  MLP training:        N=100k stratified samples, AdamW lr=1e-4
                       wd=0.05, 70k iterations, 2000 rollouts/sample
  MLP MAE:             0.0034 over 500 held-out scenarios at
                       5e4 rollouts each
  BLR (prototype):     28-d hand-crafted features, online posterior,
                       <1 ms per query, 8000 candidates in <1 s
  Held-out photos:     13  (app/backend/test/)
  ViT weights:         86.7 MB  (vit_classifier.pth, fully fine-tuned)
  MLP weights:          1.4 MB  (winrate_model.pt)
  /analyze latency:    ~5 s CPU  (OWL-ViT detection bound)
  /recommend latency:  ~5.6 s CPU with sequential MLP scoring;
                       batched forward pass measured at ~3.8 ms
                       (~700x kernel, ~112x wall-clock) but not
                       wired into the production endpoint yet.
  Citations:           5  (owlvit, vit, timm, livechess2fen,
                          alphazero)
