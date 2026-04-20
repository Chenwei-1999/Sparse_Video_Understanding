const slideData = {
  deck: {
    title: 'REVISE Reading Group',
    subtitle: 'Editable PPTX scaffold',
    presenter: 'TBD',
    date: '2026-04-20',
  },
  slides: [
    {
      id: 'title',
      layout: 'title',
      title: 'REVISE Reading Group',
      subtitle: 'Editable PPTX scaffold for slides, notes, and citations',
      notes: [
        'Open with the goals of the workspace.',
        'Mention that the first task is to establish a stable editing and rendering scaffold.',
      ],
    },
    {
      id: 'workspace',
      layout: 'content',
      title: 'Workspace scaffold',
      bullets: [
        'Slide content lives in slide_data.js.',
        'Presentation styling lives in theme.js.',
        'Slide tracking, notes, and citations are kept in separate markdown files.',
      ],
      notes: [
        'Use this slide to explain how the repository is organized.',
        'Emphasize that the generator is intentionally small so later tasks can extend it safely.',
      ],
    },
  ],
};

module.exports = slideData;

