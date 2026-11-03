// Step-by-step trajectories: one tab per figure of the paper's appendix.
document.querySelectorAll('.tabs [role="tab"]').forEach((tab) => {
  tab.addEventListener('click', () => {
    const panel = document.getElementById(tab.getAttribute('aria-controls'));
    tab.parentElement.querySelectorAll('[role="tab"]').forEach((t) => t.setAttribute('aria-selected', 'false'));
    tab.setAttribute('aria-selected', 'true');
    panel.querySelector('a').href = tab.dataset.src;
    const img = panel.querySelector('img');
    img.removeAttribute('width');   // the figures differ in height; let the new one set its own size
    img.removeAttribute('height');
    img.src = tab.dataset.src;
    panel.querySelector('figcaption').textContent = tab.dataset.caption;
  });
});

// Copy the BibTeX entry.
const copyButton = document.querySelector('.bib .copy');
if (copyButton) {
  copyButton.addEventListener('click', async () => {
    const text = document.getElementById('bibtex-code').textContent;
    try {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = 'Copied';
    } catch (e) {
      copyButton.textContent = 'Select and copy';
    }
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  });
}
