import { log } from './logger';
import { gradioApp } from './script';

let lastGitHubSearch = '';
let lastDocsSearch = '';

export async function clickGitHubWikiPage(page: string): Promise<void> {
  log(`clickGitHubWikiPage: page="${page}"`);
  lastGitHubSearch = page;
  const el = gradioApp().getElementById('github_md_btn');
  if (el) el.click();
}

export function getGitHubWikiPage(): string {
  return lastGitHubSearch;
}

export async function clickDocsPage(page: string): Promise<void> {
  log(`clickDocsPage: page="${page}"`);
  lastDocsSearch = page;
  const el = gradioApp().getElementById('docs_md_btn');
  if (el) el.click();
}

export function getDocsPage(): string {
  return lastDocsSearch;
}

window.clickDocsPage = clickDocsPage;
window.getDocsPage = getDocsPage;
window.getGitHubWikiPage = getGitHubWikiPage;
