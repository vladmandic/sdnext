#!/usr/bin/env node
// script used to localize sdnext ui and hints to multiple languages using google gemini ai

const fs = require('node:fs');

const { GoogleGenerativeAI } = require('@google/generative-ai');

const api_key = process.env.GOOGLE_AI_API_KEY;
const model = 'gemini-2.5-flash';
const prompt = `Translate attached JSON from English to {language} using following rules: fields id, label and reload should be preserved from original, field localized should be a translated version of field label and field hint should be translated in-place.
if field is less than 3 characters, do not translate it and keep it as is.
Every JSON entry should have id, label, localized, reload and hint fields.
Output should be pure JSON without any additional text. To better match translation, context of the text is related to Stable Diffusion and topic of Generative AI.`;
const languages = {
  hr: 'Croatian',
  de: 'German',
  es: 'Spanish',
  fr: 'French',
  it: 'Italian',
  pt: 'Portuguese',
  zh: 'Chinese',
  ja: 'Japanese',
  ko: 'Korean',
  ru: 'Russian',
};
const chunkLines = 100;

async function localize() {
  if (!api_key || api_key.length < 10) {
    console.error('localize: set GOOGLE_AI_API_KEY env variable with your API key');
    process.exit();
  }
  const genAI = new GoogleGenerativeAI(api_key);
  const instance = genAI.getGenerativeModel({ model });
  const raw = fs.readFileSync('html/locale_en.json');
  const json = JSON.parse(raw);
  for (const locale of Object.keys(languages)) {
    const lang = languages[locale];
    const target = prompt.replace('{language}', lang).trim();
    const output = {};
    const fn = `html/locale_${locale}.json`;
    for (const section of Object.keys(json)) {
      const data = json[section];
      output[section] = [];
      for (let i = 0; i < data.length; i += chunkLines) {
        let markdown;
        try {
          const chunk = data.slice(i, i + chunkLines);
          const result = await instance.generateContent([target, JSON.stringify(chunk)]);
          markdown = result.response.text();
          const text = markdown.replaceAll('```', '').replace(/^.*\n/, '');
          const parsed = JSON.parse(text);
          output[section].push(...parsed);
          console.log(`localize: locale=${locale} lang=${lang} section=${section} chunk=${chunk.length} output=${output[section].length} fn=${fn}`);
        } catch (err) {
          console.error('localize:', err);
          console.error('localize input:', { target, section, i });
          console.error('localize output:', { markdown });
        }
      }
      const txt = JSON.stringify(output, null, 2);
      fs.writeFileSync(fn, txt);
    }
  }
}

localize();
