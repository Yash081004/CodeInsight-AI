import React from 'react';
import { renderToString } from 'react-dom/server';
import SimilarityTab from '../src/components/SimilarityTab.jsx';
import CodeEditor from '../src/components/ui/CodeEditor.jsx';

try {
  const html = renderToString(React.createElement(SimilarityTab, { theme: 'dark' }));
  console.log('SimilarityTab OK', html.length);
} catch (e) {
  console.error('SimilarityTab FAIL', e);
  process.exit(1);
}

try {
  const html2 = renderToString(
    React.createElement(CodeEditor, { value: 'x=1', onChange: () => {}, label: 'Test', theme: 'light' }),
  );
  console.log('CodeEditor light OK', html2.length);
} catch (e) {
  console.error('CodeEditor light FAIL', e);
  process.exit(1);
}
