// CrashTestPage.js
// This page is used to show crash testing for the video
// It throws an error on purpose while it is rendering, which the ErrorBoundary then catches.
// Delete this file (and its route in App.js) before submitting.

function CrashTestPage() {
  throw new Error('This crash is on purpose, for testing the fallback page.');
}

export default CrashTestPage;
