// api.js
// Everything that talks to the back end is in this one file.
// The rest of the app just calls checkText(text) and doesn't care how the answer was made.

import axios from 'axios';

// These two settings come from the .env file (see .env.example).
// Until the real back end exists, USE_MOCK is true and we fake the answer.
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// The answer the app expects from the back end is an object that looks like this:
//   { label: 'ai' or 'human', confidence: a number from 0 to 1 }
// If the back end sends something different, change it here and nowhere else.
export async function checkText(text) {
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
