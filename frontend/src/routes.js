/** Tab ids synced with URL hash for refresh and direct links */
export const PRIMARY_TABS = [
  { id: 'overview', label: 'Overview', hash: 'overview' },
  { id: 'similarity', label: 'Similarity', hash: 'similarity' },
  { id: 'classifier', label: 'Classifier', hash: 'classifier' },
  { id: 'evaluation', label: 'Evaluation', hash: 'evaluation' },
];

export const SECONDARY_TABS = [
  { id: 'history', label: 'History', hash: 'history' },
  { id: 'dataset', label: 'Datasets', hash: 'datasets' },
];

export const TABS = [...PRIMARY_TABS, ...SECONDARY_TABS];

const HASH_TO_TAB = Object.fromEntries(TABS.map((t) => [t.hash, t.id]));

export function tabFromHash() {
  const raw = (window.location.hash || '').replace(/^#/, '').toLowerCase();
  if (!raw) return 'overview';
  return HASH_TO_TAB[raw] || 'overview';
}

export function hashForTab(tabId) {
  const tab = TABS.find((t) => t.id === tabId);
  return tab ? `#${tab.hash}` : '#overview';
}

export const TAB_PAGE_TITLES = {
  overview: 'Overview',
  similarity: 'Similarity',
  classifier: 'Classifier',
  evaluation: 'Evaluation',
  dataset: 'Datasets',
  history: 'History',
};
