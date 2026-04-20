const fs = require('fs');
const path = require('path');

const { createPresentation } = require('./theme');
const { slides } = require('./slide_data');

const baseDir = __dirname;
const outputDir = path.join(baseDir, 'output');
const outputPath = path.join(outputDir, 'revise_reading_group.pptx');

async function renderDeck() {
  fs.mkdirSync(outputDir, { recursive: true });

  const pptx = createPresentation();

  slides.forEach((slideData) => {
    const slide = pptx.addSlide();
    slide.addText(slideData.title, {
      x: 0.8,
      y: 1.0,
      w: 11.0,
      h: 0.5,
    });
  });

  await pptx.writeFile({ fileName: outputPath });
}

renderDeck().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
