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

async function renderDeck() {
  fs.mkdirSync(outputDir, { recursive: true });

  const pptx = createPresentation();

  slides.forEach((slideData) => {
    const slide = pptx.addSlide();
    const renderSlide = getRenderer(slideData.layout);
    renderSlide(slide, slideData);
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
  renderDeck,
};
