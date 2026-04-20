# REVISE Deck QA Summary

## Render
- `node presentations/revise_reading_group/build_deck.js` succeeded.
- Final deck regenerated at `presentations/revise_reading_group/output/revise_reading_group.pptx`.

## Text QA
- `python -m markitdown presentations/revise_reading_group/output/revise_reading_group.pptx > presentations/revise_reading_group/review/deck_text.txt` succeeded.
- Extracted deck contains `38` slide markers and `38` notes sections.
- Placeholder scan found no matches for:
  - `EDITORIAL READ`
  - `SPEAKER CUE`
  - `Why this example matters`
  - `Visual or evidence panel`
  - `xxxx`
  - `lorem`
  - `ipsum`

## Visual QA
- `python /home/cxk2993/.agents/skills/pptx/scripts/thumbnail.py ...` failed because `soffice` is not installed in this environment.
- `pdftoppm` is also unavailable here, so PDF-to-image slide inspection could not be run locally.

## Interpretation
- Text-level QA passed.
- Visual QA remains partially blocked by missing system binaries, not by deck generation errors.
