Pacific 1940 Battle Analyzer - Midterm Report
==============================================

FILES IN THIS FOLDER
--------------------
  main.tex                Main LaTeX source (CVPR format)
  cvpr.sty                CVPR 2026 author-kit style (eso-pic inlined)
  ieeenat_fullname.bst    natbib-compatible bibliography style
  references.bib          BibTeX bibliography
  figures/                Image assets used by main.tex
    battlefield_detection.png   End-to-end perception output
                                (detection boxes + ViT classes +
                                JP/US faction colours) on a held-out
                                board photo. Methods section, Fig 2.
    crops_classification.png    Twelve per-piece crops from the same
                                photo, with ViT confidence and HSV
                                pixel counts. Data section, Fig 3.
    ui_states.png               Composite of all dynamic UI states.
                                Experiments section, Fig 4.
  README.txt              This file.

The Introduction summary figure (Figure 1) is drawn inline with TikZ
inside main.tex, so no extra file is needed for it.

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
Upload Project". Compiler must be set to pdfLaTeX (not XeLaTeX/LuaLaTeX
- cvpr.sty is written for pdfLaTeX). On free Overleaf plans the
git-remote workflow also works; see the chat log for instructions.

If MiKTeX prompts for a missing package on first run, the only likely
candidate is `silence` (used internally by cvpr.sty to suppress
cosmetic font warnings). It is not strictly required - the document
will still compile without it, just with a noisier log.

WHAT STILL NEEDS THE TEAM'S INPUT BEFORE SUBMISSION
---------------------------------------------------
Search main.tex for the following markers:

  [TEAM TODO]   - facts only the team knows. Currently only one
                  remains in main.tex (the affiliation block has been
                  filled with NYU / @nyu.edu). The remaining items
                  live in references.bib:
                  - citation for AAOddsCalc / TripleA battle calculator
                  - citation for the A&A Pacific 1940 rulebook

  \red{...}     - work not yet done; shown in red per rubric. Major
                  remaining red blocks are:
                  - Abstract: detection / classifier evaluation
                  - Section 3 Data: per-class crop counts + protocol
                  - Section 4.6 BLR: production wiring into /recommend
                  - Section 5.2 Planned Evaluations (consolidated):
                      detection accuracy, classifier accuracy, CNN-vs-ViT
                      ablation, recommendation quality
                  - Section 5.3 Win-Rate Calibration: reliability
                      diagram and per-stratum breakdown
                  - Section 6 Conclusion: list of (i)-(iv) future work

QUICK REFERENCE - HEADLINE NUMBERS
------------------------------------
  Unit vocabulary:    8  (7 attacker-eligible + 1 defender-only AA)
  Detector:           OWL-ViT large  (google/owlvit-large-patch14)
  Classifier:         ViT-Small/16   (vit_small_patch16_224, fine-tuned)
  MLP architecture:   15 -> 256 -> 256 -> 128 -> 64 -> 1, BN+ReLU
  MLP training:       N=100k stratified samples, AdamW lr=1e-4 wd=0.05,
                      70k iterations, 2000 rollouts/sample
  MLP MAE:            0.0034 over 500 held-out scenarios at 5e4 rollouts
  Held-out photos:    13  (app/backend/test/)
  ViT weights:        86.7 MB  (vit_classifier.pth, fully fine-tuned)
  MLP weights:         1.4 MB  (winrate_model.pt)
  /analyze latency:  ~5 s CPU  (OWL-ViT detection bound)
  /recommend latency ~5.6 s CPU now; ~50 ms after batching the MLP loop
                                   (~112x user-visible speed-up)
