// HomePage.js
// The main page. This is the "brain" of the app: it holds all the state (the data that changes)
// and passes it down to the smaller components as props.
//
//   HomePage
//     |- TextForm         (the input box and button)
//     |- ResultCard       (the big answer, plus loading / error)
//     |- ConfidenceChart  (visualisation 1, the latest result)
//     |- HistoryChart     (visualisation 2, every result this session)

import { useState } from 'react';
import { checkText } from '../api';
import TextForm from '../components/TextForm';
import ResultCard from '../components/ResultCard';
import ConfidenceChart from '../components/ConfidenceChart';
import HistoryChart from '../components/HistoryChart';

// Validation Limits
const MIN_LENGTH = 50;
const MAX_LENGTH = 5000;

// Checks the text and returns a message explaining the problem, or '' (empty) if the text is fine.
// This runs BEFORE anything is sent to the back end.
function validate(text) {
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return 'Please enter some text to check.';
  }
  if (trimmed.length < MIN_LENGTH) {
    return `Please enter at least ${MIN_LENGTH} characters. You have entered ${trimmed.length}.`;
  }
  if (trimmed.length > MAX_LENGTH) {
    return `Please keep the text under ${MAX_LENGTH} characters. You have entered ${trimmed.length}.`;
  }
  return '';
}

function HomePage() {
  // ----- State -----
  // useState gives a value and a function to change it. When changed, React redraws the page.
  const [text, setText] = useState(''); // what the user has typed
  const [validationError, setValidationError] = useState(''); // message shown under the input box
  const [status, setStatus] = useState('idle'); // 'idle', 'loading', 'success' or 'error'
  const [result, setResult] = useState(null); // the latest answer from the back end
  const [errorMessage, setErrorMessage] = useState(''); // message for when the back end fails
  const [history, setHistory] = useState([]); // every check made this session (for chart 2)
  const [selectedId, setSelectedId] = useState(null); // which check is picked in chart 2

  // When the user types, save the text and clear any old validation message
  function handleTextChange(newText) {
    setText(newText);
    setValidationError('');
  }

  // Runs when the form is submitted (button click, or pressing Enter)
  // It is "async" because it waits for the back end, and "await" pauses until the answer arrives.
  async function handleSubmit(event) {
    // Stops the browser from reloading the page, which is what a form does by default
    event.preventDefault();

    // Step 1: validate. If there's a problem, show it and stop. Nothing is sent to the back end.
    const message = validate(text);
    setValidationError(message);
    if (message !== '') {
      return;
    }

    // Step 2: show the loading state
    setStatus('loading');
    setErrorMessage('');

    // Step 3: ask the back end. try/catch lets us handle a failure instead of crashing.
    try {
      const answer = await checkText(text.trim());

      // Build a record of this check to save in the history
      const newCheck = {
        id: history.length + 1, // unique id for each check
        number: history.length + 1, // the number shown on the chart
        text: text.trim(),
        label: answer.label,
        confidence: answer.confidence,
        time: new Date().toLocaleTimeString(),
      };

      setResult(newCheck);
      // [...history, newCheck] makes a NEW array with the new check on the end.
      // React needs a new array (not a changed old one) to notice that something changed.
      setHistory([...history, newCheck]);
      setSelectedId(newCheck.id);
      setStatus('success');
    } catch (error) {
      setErrorMessage(error.message);
      setStatus('error');
    }
  }

  // ----- What gets drawn -----
  return (
    <div className="page">
      <TextForm
        text={text}
        onTextChange={handleTextChange}
        onSubmit={handleSubmit}
        errorMessage={validationError}
        isLoading={status === 'loading'}
        maxLength={MAX_LENGTH}
      />

      {/* The result goes straight under the form and is the biggest text, so it's what the eye lands on */}
      <ResultCard status={status} result={result} errorMessage={errorMessage} />

      <div className="chart-grid">
        <ConfidenceChart result={result} />
        <HistoryChart history={history} selectedId={selectedId} onSelect={setSelectedId} />
      </div>
    </div>
  );
}

export default HomePage;
