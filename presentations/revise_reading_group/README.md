# `revise_reading_group`

Workspace for the REVISE reading-group deck. The deck is generated with `PptxGenJS` from the local JavaScript data files in this directory.

## Layout

- `build_deck.js` renders the PPTX into `output/revise_reading_group.pptx`
- `theme.js` defines the shared styling tokens
- `slide_data.js` stores the slide content model
- `slide_manifest.md` tracks slide intent and ownership
- `speaker_notes.md` stores the working notes for each slide
- `citations.md` collects source references for the deck

## Entrypoints

- Render deck: `node presentations/revise_reading_group/build_deck.js`
- QA check: `node presentations/revise_reading_group/build_deck.js qa`

## Dependencies

This workspace expects a local install of `pptxgenjs` under `presentations/revise_reading_group/node_modules/`.

