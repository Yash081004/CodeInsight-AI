/** Five-class taxonomy — shared copy for tooltips and overview */
export const ERROR_LABELS = [
  {
    id: 'correct',
    name: 'Correct',
    short: 'Correct',
    description: 'Code that matches the intended behavior for the given specification (not provable from syntax alone).',
  },
  {
    id: 'wrong_initialization',
    name: 'Wrong initialization',
    short: 'Wrong init',
    description: 'Accumulator or state variable starts with the wrong identity (e.g. 0 instead of 1 for products).',
  },
  {
    id: 'off_by_one_loop',
    name: 'Off-by-one loop',
    short: 'Off-by-one',
    description: 'Loop bounds omit or include an extra iteration relative to the intended range.',
  },
  {
    id: 'wrong_operator',
    name: 'Wrong operator',
    short: 'Wrong op',
    description: 'Arithmetic or comparison operator does not match intended min/max or arithmetic semantics.',
  },
  {
    id: 'wrong_condition',
    name: 'Wrong condition',
    short: 'Wrong cond',
    description: 'Boolean predicate or branch logic does not match the intended truth conditions.',
  },
];

export const PIPELINE_STEPS = [
  { id: 'parse', title: 'Parse', detail: 'Validate Python syntax (AST).' },
  { id: 'encode', title: 'Encode', detail: 'CodeBERT embeddings or TF-IDF vectors.' },
  { id: 'compare', title: 'Compare / classify', detail: 'Cosine similarity or softmax over five labels.' },
  { id: 'diagnose', title: 'Diagnose', detail: 'Scores, static hints, and stated limitations — not guaranteed fixes.' },
];
