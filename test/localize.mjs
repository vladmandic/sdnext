#!/usr/bin/env node
// script used to localize sdnext ui and hints to multiple languages using google gemini ai

import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import * as fs from 'node:fs';
import * as process from 'node:process';

const apiKey = process.env.GOOGLE_API_KEY;
const model = 'gemini-3.1-flash-lite-preview';
const prompt = `## You are expert translator AI.
Translate attached JSON from English to {language}.
## Translation Rules:
- Fields \`id\`, \`label\` and \`reload\` should be preserved from original.
- Field \`localized\` should be a translated version of field \`label\`.
- If field \`localized\` is less than 3 characters, do not translate it and keep it as is.
- Field \`hint\` should be translated in-place.
- Use alphabet and writing system of the target language.
- Use terminology that is commonly used in the target language for software interfaces, especially in the context of Stable Diffusion and Generative AI.
- Do not leave non-translated words in the output, except for technical terms that do not have a widely accepted translation in the target language. In such cases, provide a transliteration or a brief explanation in parentheses.
- Ensure that all lines are present in the output, and that the output is valid JSON matching the provided schema.
`;

const languages = {
  tb: 'English: Techno-Babble',
  nb: 'English: For-N00bs',
  hr: 'Croatian',
  es: 'Spanish',
  it: 'Italian',
  xx: 'Esperanto',
  qq: 'Latin',
  fr: 'French',
  de: 'German',
  pt: 'Portuguese',
  ru: 'Russian',
  zh: 'Chinese',
  ja: 'Japanese',
  ko: 'Korean',
  hi: 'Hindi',
  ar: 'Arabic',
  bn: 'Bengali',
  ur: 'Urdu',
  id: 'Indonesian',
  vi: 'Vietnamese',
  tr: 'Turkish',
  sr: 'Serbian',
  po: 'Polish',
  he: 'Hebrew',
  tlh: 'Klingon',
};

const responseSchema = {
  type: Type.ARRAY,
  items: {
    type: Type.OBJECT,
    properties: {
      id: { type: Type.INTEGER, description: 'id of the item' },
      label: { type: Type.STRING, description: 'original label of the item' },
      localized: { type: Type.STRING, description: 'translated label of the item' },
      reload: { type: Type.STRING, description: 'n/a' },
      hint: { type: Type.STRING, description: 'long hint for the item' },
    },
    required: ['id', 'label', 'localized', 'reload', 'hint'],
  },
};

async function localize() {
  if (!apiKey || apiKey.length < 10) {
    console.error('localize: set GOOGLE_API_KEY env variable with your API key');
    process.exit();
  }

  const httpOptions = { timeout: 60000 };
  const ai = new GoogleGenAI({ apiKey, httpOptions });
  const thinkingConfig = {
    includeThoughts: false,
    thinkingLevel: ThinkingLevel.LOW,
  };

  const params = {
    model,
    contents: {
      parts: [
        { text: 'prompt' },
        { text: 'data' },
      ],
    },
    config: {
      responseMimeType: 'application/json',
      thinkingConfig,
      responseSchema,
    },
  };
  console.log('params:', params);

  const raw = fs.readFileSync('html/locale_en.json');
  console.log('raw:', { bytes: raw.length });
  const json = JSON.parse(raw);
  console.log('targets:', { lang: Object.keys(languages), count: Object.keys(languages).length });

  for (const index in Object.keys(languages)) {
    const locale = Object.keys(languages)[index];
    const lang = languages[locale];
    const langPrompt = prompt.replace('{language}', lang).trim();
    const output = {};
    const fn = `html/locale_${locale}.json`;
    if (fs.existsSync(fn)) {
      console.log('skip:', { index, locale, lang, fn });
      continue;
    }
    console.log('localize:', { index, locale, lang, fn });
    const t0 = performance.now();
    let allOk = true;
    for (const section of Object.keys(json)) {
      if (!allOk) continue;
      const keys = Object.keys(json[section]).length;
      console.log('  start:', { locale, section, keys });
      try {
        const t1 = performance.now();
        const sectionJSON = json[section];
        const sectionParams = { ...params };
        sectionParams.contents.parts[0].text = langPrompt;
        sectionParams.contents.parts[1].text = `## JSON Data: \n${JSON.stringify(sectionJSON)}`;
        const response = await ai.models.generateContent(sectionParams);
        const responseJSON = JSON.parse(response.text);
        const diff = Math.abs(keys - responseJSON.length);
        if (diff > 1) {
          console.error('  error:', { locale, section, input: keys, output: responseJSON.length });
          allOk = false;
          continue;
        }
        output[section] = JSON.parse(response.text);
        const t2 = performance.now();
        const kps = Math.round(1000000 * keys / (t2 - t1)) / 1000;
        console.log('  end:', { locale, section, time: Math.round(t2 - t1) / 1000, kps });
      } catch (err) {
        allOk = false;
        console.error('  error:', err);
        process.exit(1);
      }
      // break; // for testing, remove this to process all sections
    }
    if (allOk) {
      const txt = JSON.stringify(output, null, 2);
      fs.writeFileSync(fn, txt);
    } else {
      console.error('  error: something went wrong, output file not saved');
    }
    const t3 = performance.now();
    console.log('  time:', { locale, time: Math.round(t3 - t0) / 1000 });
  }
}

localize();
