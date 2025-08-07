var structuredData = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  "name": "FDC Estimation Results Archive",
  "alternateName": "FDC ERA",
  "url": "https://your-domain.com",
  "potentialAction": {
    "@type": "SearchAction",
    "target": {
      "@type": "EntryPoint",
      "urlTemplate": "https://your-domain.com/search.html?q={search_term_string}"
    },
    "query-input": "required name=search_term_string"
  }
};

const SDscript = document.createElement('script');
SDscript.setAttribute('type', 'application/ld+json');
SDscript.textContent = JSON.stringify(structuredData);
document.head.appendChild(SDscript);

function filterStations() {
  const input = document.getElementById('stationSearch').value.toLowerCase();
  const results = document.getElementById('searchResults');

  // Clear previous results
  results.innerHTML = '';

  if (input.length < 2) return;

  // Filter stations
  const matches = stations.filter(c =>
    c.id.toLowerCase().includes(input) ||
    c.name.toLowerCase().includes(input) ||
    c.source.toLowerCase().includes(input) ||
    `${c.source}-${c.id}`.toLowerCase().includes(input)
  );

  // Display results (max 10)
  const limitedMatches = matches.slice(0, 10);
  limitedMatches.forEach(c => {
    const div = document.createElement('div');
    div.className = 'search-result';
    div.innerHTML = `<a href='${c.folder}'>${c.source}-${c.id}: ${c.name}</a>`;
    results.appendChild(div);
  });

  // Show message if too many results
  if (matches.length > 10) {
    const div = document.createElement('div');
    div.className = 'search-more';
    div.textContent = `... and ${matches.length - 10} more matches`;
    results.appendChild(div);
  }
}
