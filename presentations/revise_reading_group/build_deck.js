const fs = require('fs');
const path = require('path');
const PptxGenJS = require('pptxgenjs');

const { applyTheme, theme } = require('./theme');
const slideData = require('./slide_data');

const baseDir = __dirname;
const outputDir = path.join(baseDir, 'output');
const reviewDir = path.join(baseDir, 'review');
const outputPath = path.join(outputDir, 'revise_reading_group.pptx');

function ensureDirectory(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function addSlideHeader(slide, title, subtitle, isTitleSlide = false) {
  slide.background = { color: theme.colors.background };

  slide.addText(title, {
    x: theme.margins.contentLeft,
    y: isTitleSlide ? 1.35 : 0.55,
    w: 11.6,
    h: isTitleSlide ? 0.5 : 0.35,
    fontFace: theme.fonts.heading,
    fontSize: isTitleSlide ? theme.titleSlide.titleSize : theme.contentSlide.titleSize,
    bold: true,
    color: theme.colors.title,
    margin: 0,
  });

  if (subtitle) {
    slide.addText(subtitle, {
      x: theme.margins.contentLeft,
      y: isTitleSlide ? 1.95 : 0.98,
      w: 11.0,
      h: isTitleSlide ? 0.35 : 0.28,
      fontFace: theme.fonts.body,
      fontSize: isTitleSlide ? theme.titleSlide.subtitleSize : 11,
      color: theme.colors.muted,
      margin: 0,
    });
  }
}

function addBullets(slide, bullets) {
  slide.addText(
    bullets.map((bullet) => ({ text: bullet, options: { bullet: { indent: 14 } } })),
    {
      x: theme.margins.contentLeft,
      y: 1.6,
      w: 11.1,
      h: 4.5,
      fontFace: theme.fonts.body,
      fontSize: theme.contentSlide.bodySize,
      color: theme.colors.text,
      breakLine: false,
      paraSpaceAfterPt: 10,
      margin: 0,
      valign: 'top',
    }
  );
}

function addFooter(slide, text) {
  slide.addText(text, {
    x: theme.margins.contentLeft,
    y: 6.95,
    w: 11.0,
    h: 0.2,
    fontFace: theme.fonts.body,
    fontSize: 9,
    color: theme.colors.muted,
    margin: 0,
  });
}

async function renderDeck() {
  ensureDirectory(outputDir);
  ensureDirectory(reviewDir);

  const pptx = applyTheme(new PptxGenJS());

  slideData.slides.forEach((entry, index) => {
    const slide = pptx.addSlide();

    if (entry.layout === 'title') {
      addSlideHeader(slide, entry.title, entry.subtitle, true);
      slide.addShape(pptx.ShapeType.rect, {
        x: theme.margins.contentLeft,
        y: 2.6,
        w: 4.0,
        h: 0.08,
        line: { color: theme.colors.accent, transparency: 100 },
        fill: { color: theme.colors.accent },
      });
      slide.addText(`Presenter: ${slideData.deck.presenter}`, {
        x: theme.margins.contentLeft,
        y: 3.0,
        w: 5.5,
        h: 0.3,
        fontFace: theme.fonts.body,
        fontSize: 12,
        color: theme.colors.text,
        margin: 0,
      });
      slide.addText(slideData.deck.date, {
        x: theme.margins.contentLeft,
        y: 3.32,
        w: 5.0,
        h: 0.25,
        fontFace: theme.fonts.body,
        fontSize: 12,
        color: theme.colors.muted,
        margin: 0,
      });
    } else {
      addSlideHeader(slide, entry.title, entry.subtitle || '', false);
      slide.addShape(pptx.ShapeType.roundRect, {
        x: theme.margins.contentLeft,
        y: 1.45,
        w: 11.0,
        h: 4.7,
        rectRadius: 0.12,
        line: { color: theme.colors.line, pt: 1.1 },
        fill: { color: theme.colors.panel },
      });
      addBullets(slide, entry.bullets || []);
      addFooter(slide, 'Workspace scaffold slide for future content expansion.');
    }

    const notes = Array.isArray(entry.notes) ? entry.notes.join('\n') : '';
    if (notes) {
      slide.addNotes(notes);
    }
    slide.addNotes(`Slide ${index + 1} generated from slide_data.js.`);
  });

  await pptx.writeFile({ fileName: outputPath });
}

function runQa() {
  const checks = [
    ['theme.js', fs.existsSync(path.join(baseDir, 'theme.js'))],
    ['slide_data.js', fs.existsSync(path.join(baseDir, 'slide_data.js'))],
    ['slide_manifest.md', fs.existsSync(path.join(baseDir, 'slide_manifest.md'))],
    ['speaker_notes.md', fs.existsSync(path.join(baseDir, 'speaker_notes.md'))],
    ['citations.md', fs.existsSync(path.join(baseDir, 'citations.md'))],
    ['pptx output', fs.existsSync(outputPath)],
  ];

  console.log('QA check');
  checks.forEach(([label, ok]) => {
    console.log(`${ok ? 'PASS' : 'FAIL'} ${label}`);
  });
}

const mode = process.argv[2] || 'render';

if (mode === 'qa') {
  runQa();
} else {
  renderDeck().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
