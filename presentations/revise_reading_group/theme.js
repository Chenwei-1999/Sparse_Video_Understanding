const theme = {
  deckName: 'revise_reading_group',
  layout: 'LAYOUT_WIDE',
  fonts: {
    heading: 'Aptos Display',
    body: 'Aptos',
  },
  colors: {
    background: 'F7F4ED',
    title: '1C1B1F',
    text: '2D2B32',
    muted: '6C6875',
    accent: '8A5CF6',
    accentSoft: 'E4D9FF',
    panel: 'FFFDF9',
    line: 'D8D2C7',
  },
  margins: {
    slide: 0.55,
    contentTop: 1.0,
    contentLeft: 0.8,
    contentRight: 0.8,
  },
  titleSlide: {
    titleSize: 28,
    subtitleSize: 14,
  },
  contentSlide: {
    titleSize: 24,
    bodySize: 14,
  },
};

function applyTheme(pptx) {
  pptx.layout = theme.layout;
  pptx.author = 'Codex';
  pptx.company = 'OpenAI';
  pptx.subject = 'REVISE reading-group presentation scaffold';
  pptx.title = 'REVISE Reading Group';
  pptx.lang = 'en-US';
  return pptx;
}

module.exports = {
  theme,
  applyTheme,
};

