#!/usr/bin/env node

const fs = require('fs');

/**
 * Custom stringifier that switches to minified format at a specific depth
 */
const mixedStringify = (data, maxDepth, indent = 2, currentDepth = 0) => {
  if (currentDepth >= maxDepth) {
    return JSON.stringify(data);
  }

  const spacing = ' '.repeat(indent * currentDepth);
  const nextSpacing = ' '.repeat(indent * (currentDepth + 1));

  if (Array.isArray(data)) {
    if (data.length === 0) return '[]';
    const items = data.map((item) => nextSpacing + mixedStringify(item, maxDepth, indent, currentDepth + 1));
    return `[\n${items.join(',\n')}\n${spacing}]`;
  }

  if (typeof data === 'object' && data !== null) {
    const keys = Object.keys(data);
    if (keys.length === 0) return '{}';
    const items = keys.map((key) => {
      const value = mixedStringify(data[key], maxDepth, indent, currentDepth + 1);
      return `${nextSpacing}"${key}": ${value}`;
    });
    return `{\n${items.join(',\n')}\n${spacing}}`;
  }

  return JSON.stringify(data);
};

// Capture CLI arguments
const [,, inputFile, outputFile, maxDepth] = process.argv;
console.log(`Input File: ${inputFile}, Output File: ${outputFile}, Max Depth: ${maxDepth}`);

if (!inputFile || !outputFile || !maxDepth) {
  console.log('Usage: node reformat.js <input.json> <output.json> <depth>');
  process.exit(1);
}

try {
  // Read input file
  const rawData = fs.readFileSync(inputFile, 'utf8');
  const jsonData = JSON.parse(rawData);

  // Reformat with mixed depth
  const result = mixedStringify(jsonData, parseInt(maxDepth, 10));

  // Write output file
  fs.writeFileSync(outputFile, result);
  console.log(`Success! File saved to ${outputFile} (levels expanded: ${maxDepth})`);
} catch (err) {
  console.error('Error processing JSON:', err.message);
}
