# EDA Summary — Waste Classification using CNN

**Dataset:** Garbage Classification (12 classes), Kaggle (mostafaabla/garbage-classification) — 15,150 images across battery, biological, brown/green/white-glass, cardboard, clothes, metal, paper, plastic, shoes, and trash. License is unclear (merged from an older 6-class dataset plus web-scraped images), so it's treated as research-use only.

**Data quality:** No tabular metadata (just image folders), but a small number of corrupt/zero-byte files and exact-duplicate images were found, some within the same class. These are dropped before splitting so counts aren't inflated.

**Class distribution:** Moderately imbalanced — classes from the original studio-photo dataset (cardboard, glass, plastic, paper) are better represented than newer web-scraped classes (battery, clothes, biological). This biases a naive CNN toward majority classes, so we'll use class-weighted loss and per-class precision/recall instead of relying on overall accuracy.

**Split plan:** Stratified 70/15/15 train/val/test split at the image level, done after de-duplication, since there's no patient/user/time dimension to split on. Duplicate removal is our main leakage control given the dataset's mixed provenance.

**Three findings shaping model design:**
1. Images come from two very different sources (clean studio backgrounds vs. cluttered web photos), so the model could learn background style as a shortcut — mitigated with background/lighting augmentation.
2. Glass classes and metal-vs-plastic containers look visually near-identical in color/shape, so we expect most confusion-matrix errors there; may warrant a two-stage classifier (material family → glass color).
3. "Trash" is a minority, heterogeneous catch-all class, so we'll evaluate its recall separately rather than expect it to match visually coherent classes.

**Next step:** resize all images to 224×224 RGB (matching pretrained CNN backbones) and begin baseline transfer-learning training.
