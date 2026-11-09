// api.js
// Everything that talks to the back end is in this one file.
// The rest of the app just calls checkText(text) and doesn't care how the answer was made.
// That means when the real back end is ready, this is the only file that needs to change.

import axios from 'axios';

// These two settings come from the .env file (see .env.example).
// Until the real back end exists, USE_MOCK is true and we fake the answer.
const USE_MOCK = process.env.REACT_APP_USE_MOCK !== 'false';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// The answer the app expects from the back end is an object that looks like this:
//   { label: 'ai' or 'human', confidence: a number from 0 to 1 }
// If the back end sends something different, change it here and nowhere else.
export async function checkText(text) {
  if (USE_MOCK) {
    return mockCheck(text);
  }

  try {
    // axios.post sends the text to the back end as JSON and waits for the reply
    const response = await axios.post(`${API_URL}/detect`, { text: text });
    const { label, ai_probability: aiProbability } = response.data;
    return {
      label,
      confidence: label === 'ai' ? aiProbability : 1 - aiProbability,
    };
  } catch (error) {
    // error.response exists when the server replied with an error code (like 500).
    // If it doesn't exist, the server couldn't be reached at all.
    if (error.response) {
      throw new Error('The server had a problem checking your text. Please try again.');
    }
    throw new Error('Could not reach the server. Check your connection and try again.');
  }
}

// ---------- Fake back end, only used while building the front end ----------

function mockCheck(text) {
  return new Promise((resolve, reject) => {
    // setTimeout makes it wait 1 second, so the loading state is visible
    setTimeout(() => {
      // Typing "simulate error" in the text lets me show the error state on purpose
      if (text.toLowerCase().includes('simulate error')) {
        reject(new Error('The server had a problem checking your text. Please try again.'));
        return;
      }

      // Turn the text into a repeatable number between 0 and 1 (the same text gives the same answer).
      // This is NOT a real model, it just gives varied results for testing the charts.
      let hash = 0;
      for (let i = 0; i < text.length; i++) {
        hash = (hash * 31 + text.charCodeAt(i)) % 1000;
      }
      const probabilityAi = 0.05 + (hash / 1000) * 0.9; // between 0.05 and 0.95

      // The label is whichever side is more likely, and the confidence is how likely that side is
      resolve({
        label: probabilityAi >= 0.5 ? 'ai' : 'human',
        confidence: Math.max(probabilityAi, 1 - probabilityAi),
      });
    }, 1000);
  });
}
