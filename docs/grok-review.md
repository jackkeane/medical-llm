# Review: Finetuning, visualized

Review of all six HTML pages plus `docs/assets/style.css` and `docs/assets/charts.js`.
Issues ordered by severity within each category. Numeric training values were left unchanged.

---

## (a) Factual / technical errors

### A1 — High · Wrong definition of cross-entropy vs. perplexity
- **File:** `docs/04-overfitting.html` · “An intuition for the numbers”
- **What was wrong:** The text said “cross-entropy loss is the exponent of the model's confusion.” That inverts the relationship. Perplexity is \(e^{\text{loss}}\) (for natural-log cross-entropy); the loss itself is the log of that measure, not the exponent.
- **Changed:** Rewrote to define perplexity as \(e\) raised to the power of the loss, kept the same numeric examples (0.71 → ≈2, 1.43 → ≈4.2), and softened “twice as fluent” to “about twice as sure.”

### A2 — Medium · Overview overstated how many samples were trained on
- **File:** `docs/index.html` · lede + first tile
- **What was wrong:** Lede said the model was “trained on 800” samples and the tile was labeled “Training samples / 800.” Chapter 1 correctly explains that `val_size: 0.1` carved validation out of the train file, so the run trained on **720** and evaluated on **80**. The overview contradicted that without caveat.
- **Changed:** Lede now says “adapted on a biomedical Q&A dataset of 800 prepared samples.” Tile label is “Prepared train samples” (value **800** unchanged). No numeric edits.

### A3 — Low · LoRA diagram stated NF4 storage before quantization was introduced
- **File:** `docs/02-qlora.html` · LoRA SVG under “Idea 1”
- **What was wrong:** The frozen-W box said “original weights, stored in 4-bit NF4” in the LoRA-only section. NF4 is the quantization half of QLoRA (Idea 2). Stating it under Idea 1 is not false for this run, but it attributes a QLoRA detail to LoRA alone and uses undefined jargon.
- **Changed:** Diagram caption is now “original weights — not updated.” NF4 is defined in Idea 2.

### A4 — Low · Left as-is · “≈84M / 1.2% of the 7.24B model”
- **Files:** `docs/index.html`, `docs/02-qlora.html`
- **What is slightly loose:** 84M / 7.24B ≈ 1.16%; the pages round to ≈1.2% and treat the share as a fraction of the base rather than of base+adapter. Directionally fine for this audience.
- **Left:** Numbers and rounding match the case-study sources; changing them would violate the numeric freeze.

### A5 — Low · Left as-is · Full-finetune “~16 bytes per weight”
- **File:** `docs/02-qlora.html` · lede
- **Note:** Mixed-precision Adam often lands nearer ~12–16 bytes/param depending on what is stored in fp32. “Roughly 16 bytes… on the order of a hundred gigabytes” is a standard teaching estimate for a 7B model.
- **Left:** Approximate and labeled as such.

---

## (b) Pedagogical weaknesses

### B1 — High · Rank \(r\) and scale \(\alpha\) used before definition
- **File:** `docs/02-qlora.html` · LoRA formula paragraph
- **What was wrong:** Formula \(Wx + (\alpha/r)·BAx\) appeared before either symbol was defined; \(r\) was only explained after the diagram, \(\alpha\) only in the config dump.
- **Changed:** Introduced rank \(r\) and scale \(\alpha\) in the same paragraph as the formula, with this run’s values (r=32, α=64, α/r=2). Shortened the post-diagram sentence so it does not re-define rank from scratch.

### B2 — Medium · QLoRA and “adapter” on first sight without a gloss
- **File:** `docs/index.html` · lede + pipeline intro
- **What was wrong:** Beginners hit “QLoRA” and “adapter” immediately with no one-line meaning.
- **Changed:** Parenthetical gloss for QLoRA (“quantized low-rank adaptation — the memory-saving method Chapter 2 unpacks”) and a short definition of adapter (“a thin set of extra weights that steer the base model without rewriting it”).

### B3 — Medium · bf16 / NF4 used as opaque labels
- **File:** `docs/02-qlora.html` · quantization section
- **What was wrong:** “NF4 code” and “decompressed back to bf16” assumed familiarity with NormalFloat4 and bfloat16.
- **Changed:** Expanded NF4 as NormalFloat4 (16 levels…) and bf16 as bfloat16 used for the actual math.

### B4 — Medium · Cross-entropy introduced too thinly in the training chapter
- **File:** `docs/03-training.html` · “The loss curve”
- **What was wrong:** “Surprise at the correct answer (cross-entropy)” was easy to skim past; Chapter 4 later builds on the same idea.
- **Changed:** One sentence now states that cross-entropy measures how surprised the model is by the correct next token (lower = higher probability on the right answer).

### B5 — Low · Validation vs. eval terminology not bridged
- **File:** `docs/01-data.html` · “Why three files?”
- **What was wrong:** Chapter 1 says “validation”; later chapters say “eval loss” without connecting the terms.
- **Changed:** Added “(sometimes called the eval set)” next to Validation.

### B6 — Low · Tokenization appears after the chat-template example
- **File:** `docs/01-data.html`
- **Note:** The template block is shown before the “How the model reads text: tokens” section. A strict beginner might not know why “token sequence” was already mentioned.
- **Left:** Reordering sections would be a larger restructure than the edit budget allows; the later section still covers the concept.

### B7 — Low · Left as-is · Epoch marked only as vertical lines
- **Files:** `docs/03-training.html`, `docs/04-overfitting.html`
- **Note:** “epoch 2 / epoch 3” lines are clear once Chapter 3 defines an epoch; the overview chart has no epoch lines. Acceptable scaffolding.

---

## (c) HTML / CSS / JS, accessibility, layout

### C1 — Medium · Weak keyboard focus affordance
- **File:** `docs/assets/style.css`
- **What was wrong:** Nav links, theme toggle, chapter cards, pager links, and chart summary elements had hover styles but no `:focus-visible` outline — keyboard users got little feedback.
- **Changed:** Added `:focus-visible` outlines (series-1) for `.site-nav a`, `.theme-toggle`, `.pager a`, `.chapters a`, `.brand`, `.chart-table summary`, and `.fig svg`.

### C2 — Medium · Theme toggle accessibility and init race
- **File:** `docs/assets/charts.js` · `initTheme`; all six HTML pages · theme button
- **What was wrong:** (1) Button had no accessible name beyond ambiguous “Dark mode” text that flips meaning. (2) Theme init only listened for `DOMContentLoaded`, so a late or already-complete document could leave the control uninitialized.
- **Changed:** Dynamic `aria-label` (“Switch to dark/light mode”); HTML buttons given an initial `aria-label`. `initTheme` runs immediately when `document.readyState !== "loading"`, otherwise on `DOMContentLoaded`.

### C3 — Medium · Epoch labels could clip at the top of line charts
- **File:** `docs/assets/charts.js` · `lineChart`
- **What was wrong:** Vertical-line labels were drawn at `m.top - 2` with default top margin 14px, so “epoch 2/3” text could clip against the SVG top edge.
- **Changed:** When `cfg.vLines` is present, top margin is 22px instead of 14px. No API change.

### C4 — Medium · Dark-mode contrast on the frozen-parameter bar label
- **File:** `docs/02-qlora.html` · stacked bar config
- **What was wrong:** Segment used `ink: "#1a1a19"` on `--de-emphasis`. In dark mode `--de-emphasis` is a mid gray (#52514e); near-black label text fails contrast.
- **Changed:** `ink: "var(--text-primary)"` so the label follows the theme.

### C5 — Low · Hard-coded highlight colors ignored dark series tokens
- **File:** `docs/01-data.html` · chat-template card
- **What was wrong:** Input/output token washes used fixed `rgba(232,123,164,…)` and `rgba(0,131,0,…)`, which do not track `--series-3` / `--series-2` in dark mode.
- **Changed:** Prefer `color-mix(..., var(--series-N), transparent)` with the old rgba values as fallback for browsers without `color-mix`.

### C6 — Low · Left as-is · Bar / stacked charts lack keyboard exploration
- **File:** `docs/assets/charts.js`
- **Note:** Line charts support focus + arrow keys; bar and stacked-bar charts are pointer-only (tooltips on hover). Tables under each figure still expose the values.
- **Left:** Adding keyboard parity would be a larger renderer change than “targeted edits” allows; data tables remain the accessible fallback.

### C7 — Low · Left as-is · Data tables without `<thead>` / `scope`
- **Files:** all pages with `table.data`
- **Note:** Header cells are bare `<th>` in a single row without `scope="col"`. Screen readers usually still treat the first row as headers, but explicit scope would be cleaner.
- **Left:** Cosmetic markup sweep across many tables; low user impact given simple two-axis tables.

### C8 — Low · Left as-is · No `<main>` landmark
- **Files:** all HTML pages
- **Note:** Content sits in `.wrap` without a main landmark for skip navigation.
- **Left:** Wrapping every page in `<main>` is structural churn without a content fix.

### C9 — Info · Left as-is · Responsive pipeline SVG
- **File:** `docs/index.html`
- **Note:** Wide pipeline diagram is inside `overflow-x: auto` — correct pattern for `file://` static pages. No change needed.

---

## (d) Writing quality / consistency

### D1 — Medium · “tune-time checking”
- **File:** `docs/01-data.html` · split table
- **What was wrong:** Awkward, non-standard phrase for the validation set’s purpose.
- **Changed:** “checked during training.”

### D2 — Medium · “165× smaller”
- **File:** `docs/03-training.html` · learning-rate list
- **What was wrong:** English “N× smaller” is ambiguous (subtractive vs. fractional).
- **Changed:** “about 1/165 of its peak” (same ratio, clearer).

### D3 — Low · Reward language for supervised loss
- **File:** `docs/01-data.html` · loss masking
- **What was wrong:** “isn't rewarded for parroting” mixes RL vocabulary into cross-entropy training.
- **Changed:** “is not trained to parrot questions — only to produce answers,” and dropped scare-quotes around “scored.”

### D4 — Low · “BioMistral came pretrained”
- **File:** `docs/05-inference.html`
- **What was wrong:** Slightly off idiom.
- **Changed:** “BioMistral was pretrained.”

### D5 — Low · “The skill being taught here”
- **File:** `docs/04-overfitting.html` · Perspective callout
- **What was wrong:** “Skill” is slightly training-course voice and less precise than the intended takeaway.
- **Changed:** “The habit to build is…”

### D6 — Low · Left as-is · “eval” vs “evaluation” mix
- **Files:** throughout
- **Note:** Both forms appear (“eval loss”, “evaluation loss”). Tolerable once B5 bridges validation/eval; full homogenization would touch many sentences for little gain.

### D7 — Low · Left as-is · “finetuning” closed form
- **Files:** throughout
- **Note:** Consistently “finetuning” / “finetuned” rather than “fine-tuning.” Internal consistency is good; left unchanged.

---

## Edit log (every change made)

| # | File | Change |
|---|------|--------|
| 1 | `docs/index.html` | Glossed QLoRA and adapter; “800 prepared samples” wording; tile label “Prepared train samples”; theme `aria-label` |
| 2 | `docs/01-data.html` | Loss-masking wording; validation/eval bridge; table purpose cell; theme-aware token washes + fallback; theme `aria-label` |
| 3 | `docs/02-qlora.html` | Defined r/α with formula; LoRA diagram no longer says NF4; expanded NF4/bf16; frozen-bar ink; theme `aria-label` |
| 4 | `docs/03-training.html` | Clearer cross-entropy sentence; LR “1/165 of its peak”; theme `aria-label` |
| 5 | `docs/04-overfitting.html` | Fixed perplexity explanation; “habit to build”; theme `aria-label` |
| 6 | `docs/05-inference.html` | “was pretrained”; theme `aria-label` |
| 7 | `docs/assets/style.css` | `:focus-visible` styles for nav, toggle, pager, chapters, brand, chart summary, SVG |
| 8 | `docs/assets/charts.js` | Theme init when DOM already ready; dynamic theme `aria-label`; larger top margin when `vLines` present |

**Numeric values:** No loss figures, learning rates, step counts, sizes, percentages, or parameter counts were altered.

**Constraints honored:** No external dependencies, no chart API restructure, CSS custom-property theming preserved, tone kept plain and precise.
