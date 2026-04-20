const fs = require('fs');
const path = require('path');

const {
  createPresentation,
  renderComparisonSlide,
  renderFigureSlide,
  renderGapSlide,
  renderPreviewSlide,
  renderTakeawaySlide,
  renderTaxonomySlide,
  renderTitleHeroSlide,
} = require('./theme');
const { slides } = require('./slide_data');

const baseDir = __dirname;
const outputDir = path.join(baseDir, 'output');
const outputPath = path.join(outputDir, 'revise_reading_group.pptx');
const speakerNotesPath = path.join(baseDir, 'speaker_notes.md');

const layoutRenderers = {
  title_hero: renderTitleHeroSlide,
  preview: renderPreviewSlide,
  taxonomy: renderTaxonomySlide,
  benchmark_compare: renderTakeawaySlide,
  comparison: renderComparisonSlide,
  agenda: renderTakeawaySlide,
  context: renderTakeawaySlide,
  problem: renderTakeawaySlide,
  claim: renderFigureSlide,
  failure_case: renderFigureSlide,
  trend: renderTakeawaySlide,
  gap: renderGapSlide,
};

function getRenderer(layout) {
  const renderer = layoutRenderers[layout];
  if (!renderer) {
    throw new Error(`No renderer configured for layout "${layout}"`);
  }
  return renderer;
}

function parseSpeakerNotes(markdown) {
  const notesByKey = {};
  let currentKey = null;
  let currentLines = [];

  function commitCurrentSection() {
    if (!currentKey) {
      return;
    }

    const body = currentLines.join('\n').trim();
    if (body) {
      notesByKey[currentKey] = body;
    }
  }

  markdown.split(/\r?\n/).forEach((line) => {
    const headingMatch = line.match(/^##\s+(.+?)\s*$/);
    if (headingMatch) {
      commitCurrentSection();

      const heading = headingMatch[1].trim();
      currentKey = /^s\d+_[\w-]+$/.test(heading) ? heading : null;
      currentLines = [];
      return;
    }

    if (currentKey) {
      currentLines.push(line);
    }
  });

  commitCurrentSection();
  return notesByKey;
}

function loadSpeakerNotes(filePath = speakerNotesPath) {
  return parseSpeakerNotes(fs.readFileSync(filePath, 'utf8'));
}

async function renderDeck() {
  fs.mkdirSync(outputDir, { recursive: true });

  const pptx = createPresentation();
  const notesByKey = loadSpeakerNotes();

  slides.forEach((slideData) => {
    const slide = pptx.addSlide();
    const renderSlide = getRenderer(slideData.layout);
    renderSlide(slide, slideData);

    if (slideData.notesKey) {
      const notesBody = notesByKey[slideData.notesKey];
      if (!notesBody) {
        throw new Error(
          `Slide "${slideData.id}" declares notesKey "${slideData.notesKey}" but no notes body was found in ${path.basename(
            speakerNotesPath,
          )}`,
        );
      }
      slide.addNotes(notesBody);
    }
  });

  await pptx.writeFile({ fileName: outputPath });
  return outputPath;
}

if (require.main === module) {
  renderDeck().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}

module.exports = {
  getRenderer,
  layoutRenderers,
  loadSpeakerNotes,
  parseSpeakerNotes,
  renderDeck,
};
