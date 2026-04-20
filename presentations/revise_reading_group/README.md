# `revise_reading_group`

## Entrypoints

Render:

`node presentations/revise_reading_group/build_deck.js`

QA:

`python -m markitdown presentations/revise_reading_group/output/revise_reading_group.pptx`

`python /home/cxk2993/.agents/skills/pptx/scripts/thumbnail.py presentations/revise_reading_group/output/revise_reading_group.pptx presentations/revise_reading_group/review/thumb`

Thumbnail and PDF QA depend on `soffice` and `pdftoppm`; both are currently missing in this environment.
