# AI Text Checker - Front End

React front end for the assignment. The user pastes in some text, the back end says whether it looks AI-generated or human-written, and the app shows the result with its confidence and two charts.

## Running it on a clean machine

You need [Node.js](https://nodejs.org/) 18 or newer (tested on Node 22).

```bash
cd frontend
npm install     # installs the exact pinned versions from package-lock.json
npm start       # opens http://localhost:3000
```

By default the app uses a **mock back end** (fake answers after a 1 second delay), so it works without the real back end.

### Connecting the real back end

1. Copy `.env.example` to `.env`
2. Set `REACT_APP_USE_MOCK=false` and `REACT_APP_API_URL` to the back end's address
3. Restart `npm start` (changes to `.env` are only read on start)

The front end expects `POST {REACT_APP_API_URL}/predict` with a body of `{ "text": "..." }`, replying with `{ "label": "ai" | "human", "confidence": 0.0 - 1.0 }`. If the real back end differs, only `src/api.js` needs to change.

## Trying the different states

| What to do | What you see |
| --- | --- |
| Click "Check text" with the box empty, or with fewer than 50 characters | Validation message under the input |
| Submit valid text | Loading state, then the result and both charts |
| Include the words `simulate error` in the text (mock only) | Error state |
| Visit `/anything-else` | 404 page |
| Visit `/crash-test` | Fallback page for unexpected failures (remove this route before submitting) |

## Project layout

```
src/
  index.js              starts the app
  App.js                header and routes
  api.js                the only file that talks to the back end (and the mock)
  colors.js             chart colours
  App.css               all styling
  pages/
    HomePage.js         holds the state, puts everything together
    NotFoundPage.js     404 page
    CrashTestPage.js    demo-only, throws an error on purpose
  components/
    TextForm.js         input form with validation messages
    ResultCard.js       result, loading and error states
    ConfidenceChart.js  chart 1: confidence split for the latest check
    HistoryChart.js     chart 2: every check this session, with filter and click to select
    ErrorBoundary.js    shows the fallback page if something crashes
```
