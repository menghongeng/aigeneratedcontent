// ResultCard.js
// Shows the answer. The requirements say the result must be the most prominent thing on the screen,
// so the label and confidence are the biggest text on the page.
// It also covers the four states: idle (nothing yet), loading, success and error.
//
// The props are:
//   status       - 'idle', 'loading', 'success' or 'error'
//   result       - the latest check ({ label, confidence }), or null
//   errorMessage - the text to show when status is 'error'

function ResultCard({ status, result, errorMessage }) {
  // Turn the 0-1 confidence into a whole-number percentage, e.g. 0.874 becomes 87
  const percent = result ? Math.round(result.confidence * 100) : 0;
  const isAi = result && result.label === 'ai';

  return (
    // aria-live="polite" makes screen readers announce the new content when it changes
    <section className="card result-card" aria-live="polite" aria-labelledby="result-heading">
      <h2 id="result-heading" className="card-title">
        Result
      </h2>

      {/* Each line below only shows when status matches, && means "if the left side is true, show the right side" */}

      {status === 'idle' && <p className="muted">Your result will appear here once you check some text.</p>}

      {status === 'loading' && (
        <div className="loading" role="status">
          <span className="spinner" aria-hidden="true"></span>
          <span>Analysing your text...</span>
        </div>
      )}

      {status === 'error' && (
        <div className="error-box" role="alert">
          <strong>Something went wrong</strong>
          <p>{errorMessage}</p>
        </div>
      )}

      {status === 'success' && result && (
        <div>
          <p className="success-note">Check complete</p>
          {/* The colour changes with the result, but the words say it too so colour is never the only clue */}
          <p className={isAi ? 'verdict ai' : 'verdict human'}>{isAi ? 'Likely AI-generated' : 'Likely human-written'}</p>
          <p className="confidence">
            <span className="confidence-number">{percent}%</span> confident
          </p>
        </div>
      )}
    </section>
  );
}

export default ResultCard;
