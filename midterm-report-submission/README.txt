Pacific 1940 Battle Analyzer — Midterm Report
==============================================

FILES IN THIS FOLDER
--------------------
  main.tex        Main LaTeX source (CVPR format, 4-page midterm)
  references.bib  BibTeX bibliography
  figures/        Drop figure files here (PDF or PNG recommended)
  README.txt      This file

HOW TO COMPILE
--------------
Step 1 — Get the CVPR style file
  Download the CVPR 2024 author kit from:
    https://github.com/cvpr-org/author-kit
  Copy these files into this folder (same directory as main.tex):
    cvpr.sty
    eso-pic.sty   (if present in the kit)

  Without cvpr.sty the document will still compile as a plain
  two-column article — comment out the \usepackage{cvpr} line in
  main.tex to do so.

Step 2 — Compile
  Run the following four commands in order:
    pdflatex main.tex
    bibtex   main
    pdflatex main.tex
    pdflatex main.tex

  On Windows you can use MiKTeX (https://miktex.org/) or TeX Live.
  Overleaf (https://overleaf.com) also works — upload all files
  including cvpr.sty, set compiler to pdfLaTeX.

WHAT NEEDS FILLING IN BEFORE SUBMISSION
----------------------------------------
Search main.tex for the following markers:

  [TEAM TODO]   — facts only the team knows:
                  • Author names and affiliations (title block)
                  • ViT training details: crops/class, augmentation,
                    optimiser, LR, epochs
                  • MLP training details: # battles, sampling
                    distribution, train/val split, hidden widths
                  • references.bib: AAOddsCalc / TripleA citation
                  • references.bib: A&A Pacific 1940 rulebook citation

  \red{...}     — work not yet done (shown in red per rubric):
                  • Detection accuracy section + figure
                  • Classifier accuracy section + confusion matrix
                  • Win-rate calibration section + reliability plot
                  • Recommendation quality section
                  • Experiments 2×2 figure panel

FIGURES TO CREATE
-----------------
  figures/pipeline_overview.pdf
      Pipeline diagram: photo → OWL-ViT → ViT → HSV → counts →
      MLP / Recommender / Simulator.
      Tip: the right-hand side already exists as
      docs/images/ui-states.png; draw the perception half and
      stitch them together.

  figures/data_overview.pdf
      Left panel: annotated board photo (use
      assets/annotated_cv_boxes_green.jpg).
      Right panel: grid of representative unit crops (8 classes).

  figures/experiments.pdf  [placeholder, fill in when eval is done]
      2×2 panel: detection P/R curve, ViT confusion matrix,
      MLP reliability diagram, latency bar chart.

Uncomment the relevant \includegraphics lines in main.tex once
the figures are ready and drop the files into the figures/ folder.

QUICK REFERENCE — HEADLINE NUMBERS
------------------------------------
  Unit classes:       8  (7 attacker + 1 defender-only AA)
  Detector:           OWL-ViT large  (google/owlvit-large-patch14)
  Classifier:         ViT-Small/16   (vit_small_patch16_224)
  MLP input dim:      16             (8 unit slots × 2 factions)
  Held-out photos:    13
  ViT weights:        86.7 MB  (vit_classifier.pth)
  MLP weights:         1.4 MB  (winrate_model.pt)
  /analyze latency:  ~5 s CPU  (OWL-ViT bound)
  /recommend latency ~5.6 s CPU (MLP loop; planned <50 ms post-fix)
