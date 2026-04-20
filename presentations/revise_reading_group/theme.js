const path = require('path');
const PptxGenJS = require('pptxgenjs');
const SHAPE = new PptxGenJS().ShapeType;

const COLORS = {
  paper: 'F5F1E8',
  paperShade: 'E8E0D2',
  ink: '1F2430',
  charcoal: '303641',
  slate: '66707B',
  navy: '21344D',
  clay: 'B8674A',
  moss: '77856A',
  line: 'D6CCBB',
  white: 'FFFFFF',
};

const FONTS = {
  title: 'Georgia',
  body: 'Aptos',
  accent: 'Trebuchet MS',
};

const PAGE = {
  width: 13.333,
  height: 7.5,
  headerY: 0.42,
  titleY: 0.78,
  contentTop: 1.7,
  footerY: 7.05,
};

const MARGIN = {
  left: 0.72,
  right: 0.72,
  top: 0.5,
  bottom: 0.42,
  gutter: 0.32,
};

const TYPE = {
  eyebrow: 10,
  title: 26,
  subtitle: 15,
  body: 15,
  bodyTight: 14,
  small: 10,
  caption: 10,
  cardTitle: 14,
  callout: 18,
};

function createPresentation() {
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'Codex';
  pptx.company = 'OpenAI';
  pptx.subject = 'REVISE reading group';
  pptx.title = 'REVISE: Towards Sparse Video Understanding and Reasoning';
  pptx.lang = 'en-US';
  pptx.theme = {
    headFontFace: FONTS.title,
    bodyFontFace: FONTS.body,
    lang: 'en-US',
  };
  return pptx;
}

function applyPaperBackground(slide) {
  slide.background = { color: COLORS.paper };

  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 0,
    w: PAGE.width,
    h: PAGE.height,
    fill: { color: COLORS.paper },
    line: { color: COLORS.paper },
  });

  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 0,
    w: 0.3,
    h: PAGE.height,
    fill: { color: COLORS.navy },
    line: { color: COLORS.navy },
  });

  slide.addShape(SHAPE.rect, {
    x: PAGE.width - 1.7,
    y: 0,
    w: 1.7,
    h: PAGE.height,
    fill: { color: COLORS.paperShade, transparency: 48 },
    line: { color: COLORS.paperShade, transparency: 100 },
  });

  slide.addShape(SHAPE.line, {
    x: MARGIN.left,
    y: 1.44,
    w: PAGE.width - MARGIN.left - MARGIN.right,
    h: 0,
    line: { color: COLORS.line, width: 1.25 },
  });
}

function addSlideTitle(slide, title, eyebrow) {
  if (eyebrow) {
    slide.addText(eyebrow.toUpperCase(), {
      x: MARGIN.left,
      y: PAGE.headerY,
      w: 3.2,
      h: 0.24,
      margin: 0,
      fontFace: FONTS.accent,
      fontSize: TYPE.eyebrow,
      bold: true,
      color: COLORS.clay,
      charSpacing: 1.2,
    });
  }

  slide.addText(title, {
    x: MARGIN.left,
    y: PAGE.titleY,
    w: PAGE.width - MARGIN.left - MARGIN.right - 0.2,
    h: 0.6,
    margin: 0,
    fontFace: FONTS.title,
    fontSize: TYPE.title,
    bold: true,
    color: COLORS.ink,
    breakLine: false,
  });
}

function addCitationFooter(slide, citations = []) {
  if (!citations.length) {
    return;
  }

  slide.addText(citations.join(' · '), {
    x: MARGIN.left,
    y: PAGE.footerY,
    w: PAGE.width - MARGIN.left - MARGIN.right,
    h: 0.18,
    margin: 0,
    fontFace: FONTS.body,
    fontSize: TYPE.small,
    color: COLORS.slate,
    italic: true,
    align: 'right',
  });
}

function addBulletList(slide, bullets, box, options = {}) {
  const runs = [];
  bullets.forEach((bullet, index) => {
    runs.push({
      text: bullet,
      options: {
        bullet: { indent: 16 },
        hanging: 2,
        breakLine: index < bullets.length - 1,
      },
    });
  });

  slide.addText(runs, {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    margin: 0,
    fontFace: FONTS.body,
    fontSize: options.fontSize || TYPE.body,
    color: options.color || COLORS.charcoal,
    valign: 'top',
    paraSpaceAfterPt: 8,
    fit: 'shrink',
  });
}

function addBodyCopy(slide, lines, box, options = {}) {
  slide.addText(lines.join('\n'), {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    margin: 0,
    fontFace: FONTS.body,
    fontSize: options.fontSize || TYPE.body,
    color: options.color || COLORS.charcoal,
    italic: Boolean(options.italic),
    fit: 'shrink',
    breakLine: false,
  });
}

function addCard(slide, card) {
  slide.addShape(SHAPE.roundRect, {
    x: card.x,
    y: card.y,
    w: card.w,
    h: card.h,
    rectRadius: 0.08,
    fill: { color: card.fill || COLORS.white },
    line: { color: card.line || COLORS.line, width: 1 },
  });

  if (card.kicker) {
    slide.addText(card.kicker.toUpperCase(), {
      x: card.x + 0.2,
      y: card.y + 0.15,
      w: card.w - 0.4,
      h: 0.18,
      margin: 0,
      fontFace: FONTS.accent,
      fontSize: TYPE.small,
      color: card.kickerColor || COLORS.clay,
      bold: true,
      charSpacing: 1,
    });
  }

  if (card.title) {
    slide.addText(card.title, {
      x: card.x + 0.2,
      y: card.y + 0.38,
      w: card.w - 0.4,
      h: 0.34,
      margin: 0,
      fontFace: FONTS.body,
      fontSize: TYPE.cardTitle,
      bold: true,
      color: COLORS.ink,
      fit: 'shrink',
    });
  }

  if (card.body && card.body.length) {
    addBodyCopy(
      slide,
      card.body,
      {
        x: card.x + 0.2,
        y: card.y + 0.8,
        w: card.w - 0.4,
        h: card.h - 0.95,
      },
      { fontSize: TYPE.bodyTight, color: COLORS.charcoal },
    );
  }
}

function humanizeKey(value) {
  return String(value)
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function resolveAssetPath(assetPath) {
  return path.resolve(__dirname, '..', '..', assetPath);
}

function addAssetPanel(slide, asset, box) {
  slide.addShape(SHAPE.roundRect, {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    rectRadius: 0.08,
    fill: { color: COLORS.white },
    line: { color: COLORS.line, width: 1 },
  });

  if (asset && asset.kind && asset.kind !== 'image') {
    throw new Error(`Unsupported asset kind "${asset.kind}" for asset panel`);
  }

  if (asset?.kind === 'image' && !asset.path) {
    throw new Error('Image asset is missing a path');
  }

  if (asset?.path) {
    slide.addImage({
      path: resolveAssetPath(asset.path),
      x: box.x + 0.14,
      y: box.y + 0.14,
      w: box.w - 0.28,
      h: box.h - 0.5,
      sizing: { type: 'contain', w: box.w - 0.28, h: box.h - 0.5 },
    });
  } else {
    slide.addShape(SHAPE.rect, {
      x: box.x + 0.14,
      y: box.y + 0.14,
      w: box.w - 0.28,
      h: box.h - 0.5,
      fill: { color: COLORS.paperShade, transparency: 20 },
      line: { color: COLORS.paperShade },
    });
    slide.addText('Visual or evidence panel', {
      x: box.x + 0.32,
      y: box.y + 0.95,
      w: box.w - 0.64,
      h: 0.3,
      margin: 0,
      fontFace: FONTS.title,
      fontSize: TYPE.callout,
      italic: true,
      color: COLORS.slate,
      align: 'center',
    });
  }

  if (asset?.caption) {
    slide.addText(asset.caption, {
      x: box.x + 0.18,
      y: box.y + box.h - 0.28,
      w: box.w - 0.36,
      h: 0.16,
      margin: 0,
      fontFace: FONTS.body,
      fontSize: TYPE.caption,
      color: COLORS.slate,
      italic: true,
      align: 'right',
    });
  }
}

function renderPreviewSlide(slide, slideData) {
  applyPaperBackground(slide);
  addSlideTitle(slide, slideData.title, slideData.section);

  const bullets = slideData.bullets || [];
  const cardCount = bullets.length;
  const columns = cardCount > 3 ? 2 : 1;
  const rows = cardCount > 0 ? Math.ceil(cardCount / columns) : 1;
  const availableW = PAGE.width - MARGIN.left - MARGIN.right - (columns - 1) * MARGIN.gutter;
  const cardW = availableW / columns;
  const availableH = PAGE.height - PAGE.contentTop - 0.86 - (rows - 1) * 0.18;
  const cardH = cardCount > 0 ? availableH / rows : 1.4;

  bullets.forEach((bullet, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    addCard(slide, {
      x: MARGIN.left + column * (cardW + MARGIN.gutter),
      y: PAGE.contentTop + row * (cardH + 0.18),
      w: cardW,
      h: cardH,
      kicker: `Preview 0${index + 1}`,
      title: bullet,
      body: [],
    });
  });

  addCitationFooter(slide, slideData.citations);
}

function renderTaxonomySlide(slide, slideData) {
  applyPaperBackground(slide);
  addSlideTitle(slide, slideData.title, slideData.section);

  const introH = 0.95;
  addBodyCopy(
    slide,
    slideData.bullets,
    {
      x: MARGIN.left,
      y: PAGE.contentTop - 0.04,
      w: PAGE.width - MARGIN.left - MARGIN.right,
      h: introH,
    },
    { fontSize: TYPE.bodyTight, color: COLORS.charcoal },
  );

  const startY = PAGE.contentTop + 1.05;
  const cols = 2;
  const rows = Math.ceil((slideData.categories || []).length / cols);
  const cardW = (PAGE.width - MARGIN.left - MARGIN.right - MARGIN.gutter) / cols;
  const cardH = rows > 0 ? (PAGE.height - startY - 0.75 - (rows - 1) * 0.18) / rows : 1.5;

  (slideData.categories || []).forEach((category, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    addCard(slide, {
      x: MARGIN.left + col * (cardW + MARGIN.gutter),
      y: startY + row * (cardH + 0.18),
      w: cardW,
      h: cardH,
      kicker: category.name,
      title: category.mechanism,
      body: [
        `Strength: ${category.strength}`,
        `Weakness: ${category.weakness}`,
        `Fit: ${category.fit}`,
      ],
    });
  });

  addCitationFooter(slide, slideData.citations);
}

function renderFigureSlide(slide, slideData) {
  applyPaperBackground(slide);
  addSlideTitle(slide, slideData.title, slideData.section);

  const leftW = 4.55;
  const leftX = MARGIN.left;
  const figureX = leftX + leftW + MARGIN.gutter;
  const figureW = PAGE.width - MARGIN.right - figureX;

  slide.addText('Why this example matters', {
    x: leftX,
    y: PAGE.contentTop - 0.04,
    w: leftW,
    h: 0.24,
    margin: 0,
    fontFace: FONTS.accent,
    fontSize: TYPE.eyebrow + 1,
    color: COLORS.clay,
    bold: true,
    charSpacing: 0.8,
  });

  addBulletList(slide, slideData.bullets || [], {
    x: leftX,
    y: PAGE.contentTop + 0.24,
    w: leftW,
    h: 4.35,
  });

  addAssetPanel(slide, slideData.asset, {
    x: figureX,
    y: PAGE.contentTop,
    w: figureW,
    h: 4.9,
  });

  addCitationFooter(slide, slideData.citations);
}

function renderComparisonSlide(slide, slideData) {
  if (!slideData.comparisons?.length && !slideData.columns?.length) {
    throw new Error(
      `Comparison slide "${slideData.id || slideData.title || 'unknown'}" is missing both comparisons and columns data`,
    );
  }

  applyPaperBackground(slide);
  addSlideTitle(slide, slideData.title, slideData.section);

  const summaryY = PAGE.contentTop - 0.04;
  const summaryH = 0.9;
  addBodyCopy(
    slide,
    slideData.bullets || [],
    {
      x: MARGIN.left,
      y: summaryY,
      w: PAGE.width - MARGIN.left - MARGIN.right,
      h: summaryH,
    },
    { fontSize: TYPE.bodyTight },
  );

  const contentY = summaryY + summaryH + 0.18;
  const contentH = PAGE.height - contentY - 0.82;

  if (slideData.comparisons?.length) {
    const comparisonKeys = Object.keys(slideData.comparisons[0] || {});
    const colW =
      (PAGE.width - MARGIN.left - MARGIN.right - (comparisonKeys.length - 1) * MARGIN.gutter) /
      comparisonKeys.length;

    comparisonKeys.forEach((key, index) => {
      const header = slideData.comparisonHeaders?.[key];
      addCard(slide, {
        x: MARGIN.left + index * (colW + MARGIN.gutter),
        y: contentY,
        w: colW,
        h: contentH,
        kicker: header?.kicker || humanizeKey(key),
        title: header?.title || humanizeKey(key),
        body: slideData.comparisons.map((item) => String(item[key] ?? '')),
      });
    });
  } else if (slideData.columns?.length) {
    const cardW = (PAGE.width - MARGIN.left - MARGIN.right - MARGIN.gutter) / 2;
    slideData.columns.forEach((column, index) => {
      addCard(slide, {
        x: MARGIN.left + index * (cardW + MARGIN.gutter),
        y: contentY,
        w: cardW,
        h: contentH,
        kicker: column.kicker || slideData.section,
        title: column.label || humanizeKey(`column ${index + 1}`),
        body: column.points,
      });
    });
  }

  addCitationFooter(slide, slideData.citations);
}

function renderTakeawaySlide(slide, slideData) {
  applyPaperBackground(slide);
  addSlideTitle(slide, slideData.title, slideData.section);

  const bodyW = 5.3;
  const calloutX = MARGIN.left + bodyW + MARGIN.gutter;
  const calloutW = PAGE.width - MARGIN.right - calloutX;

  addBulletList(slide, slideData.bullets || [], {
    x: MARGIN.left,
    y: PAGE.contentTop + 0.08,
    w: bodyW,
    h: 4.8,
  });

  addCard(slide, {
    x: calloutX,
    y: PAGE.contentTop,
    w: calloutW,
    h: 2.05,
    kicker: 'Editorial read',
    title: slideData.bullets?.[0] || 'Key takeaway',
    body: slideData.bullets?.slice(1, 3) || [],
    fill: 'FCFAF5',
  });

  if (slideData.asset) {
    addAssetPanel(slide, slideData.asset, {
      x: calloutX,
      y: PAGE.contentTop + 2.28,
      w: calloutW,
      h: 2.62,
    });
  } else {
    addCard(slide, {
      x: calloutX,
      y: PAGE.contentTop + 2.28,
      w: calloutW,
      h: 2.62,
      kicker: 'Speaker cue',
      title: 'Use this slide to frame the section transition',
      body: [
        'Contrast the current limitation with the mechanism REVISE introduces.',
        'Keep the right rail compact so the left-side bullets remain the primary reading path.',
      ],
      fill: 'FCFAF5',
    });
  }

  addCitationFooter(slide, slideData.citations);
}

function renderTitleHeroSlide(slide, slideData) {
  slide.background = { color: COLORS.navy };

  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 0,
    w: PAGE.width,
    h: PAGE.height,
    fill: { color: COLORS.navy },
    line: { color: COLORS.navy },
  });

  slide.addShape(SHAPE.rect, {
    x: PAGE.width - 4.2,
    y: 0,
    w: 4.2,
    h: PAGE.height,
    fill: { color: COLORS.paperShade, transparency: 82 },
    line: { color: COLORS.paperShade, transparency: 100 },
  });

  slide.addText(slideData.section.toUpperCase(), {
    x: 0.86,
    y: 0.7,
    w: 2.5,
    h: 0.2,
    margin: 0,
    fontFace: FONTS.accent,
    fontSize: TYPE.eyebrow + 1,
    color: 'E8C6B6',
    bold: true,
    charSpacing: 1.5,
  });

  slide.addText(slideData.title, {
    x: 0.86,
    y: 1.18,
    w: 6.15,
    h: 1.25,
    margin: 0,
    fontFace: FONTS.title,
    fontSize: 28,
    bold: true,
    color: COLORS.white,
    fit: 'shrink',
  });

  if (slideData.subtitle) {
    slide.addText(slideData.subtitle, {
      x: 0.86,
      y: 2.62,
      w: 5.5,
      h: 0.45,
      margin: 0,
      fontFace: FONTS.body,
      fontSize: TYPE.subtitle,
      color: 'EEE7DB',
      italic: true,
    });
  }

  addBulletList(slide, slideData.bullets || [], {
    x: 0.9,
    y: 3.45,
    w: 5.2,
    h: 2.5,
  }, {
    color: 'F6F1E7',
    fontSize: TYPE.bodyTight,
  });

  addAssetPanel(slide, slideData.asset, {
    x: 7.45,
    y: 1.05,
    w: 5.0,
    h: 5.35,
  });

  addCitationFooter(slide, slideData.citations);
}

module.exports = {
  COLORS,
  FONTS,
  PAGE,
  MARGIN,
  TYPE,
  createPresentation,
  applyPaperBackground,
  addSlideTitle,
  renderPreviewSlide,
  renderTaxonomySlide,
  renderFigureSlide,
  renderComparisonSlide,
  renderTakeawaySlide,
  renderTitleHeroSlide,
};
