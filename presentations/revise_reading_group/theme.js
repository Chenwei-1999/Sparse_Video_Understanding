const PptxGenJS = require('pptxgenjs');

const COLORS = {
  background: 'F7F4ED',
  title: '1C1B1F',
  text: '2D2B32',
  muted: '6C6875',
  accent: '8A5CF6',
};

const FONTS = {
  heading: 'Aptos Display',
  body: 'Aptos',
};

function createPresentation() {
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'Codex';
  pptx.company = 'OpenAI';
  pptx.title = 'REVISE Reading Group';
  pptx.subject = 'REVISE reading-group presentation scaffold';
  pptx.lang = 'en-US';
  return pptx;
}

module.exports = {
  COLORS,
  FONTS,
  createPresentation,
};
