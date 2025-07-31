let lastGitHubSearch = '';
let lastDocsSearch = '';

async function clickGitHubWikiPage(page) {
  log(`clickGitHubWikiPage: page="${page}"`);
  lastGitHubSearch = page;
  const el = gradioApp().getElementById('github_md_btn');
  if (el) el.click();
}

function getGitHubWikiPage() {
  return lastGitHubSearch;
}

async function clickDocsPage(page) {
  log(`clickDocsPage: page="${page}"`);
  lastDocsSearch = page;
  const el = gradioApp().getElementById('docs_md_btn');
  if (el) el.click();
}

function getDocsPage() {
  return lastDocsSearch;
}
