// TextForm.js
// The input form. It is a "controlled" component, which means it doesn't keep its own data.
// The parent (HomePage) owns the text and the error message, and passes them in as props.
// When the user types or submits, this component tells the parent by calling the functions it was given.

function TextForm({ text, onTextChange, onSubmit, errorMessage, isLoading, maxLength }) {
  return (
    // noValidate turns off the browser's own pop-up messages so that our own messages show instead
    <form className="card" onSubmit={onSubmit} noValidate>
      <h1>Is this text written by AI?</h1>
      <p className="muted">Paste a piece of writing below and the model will estimate whether it was written by a human or by AI.</p>

      {/* htmlFor matches the id of the textarea, so clicking the label focuses the box and screen readers read it out */}
      <label htmlFor="text-input" className="label">
        Text to check
      </label>

      <textarea
        id="text-input"
        rows="8"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        // aria-invalid tells screen readers the field has a problem
        aria-invalid={errorMessage ? 'true' : 'false'}
        // aria-describedby links the box to the hint and error text so they get read out with it
        aria-describedby={errorMessage ? 'text-hint text-error' : 'text-hint'}
        disabled={isLoading}
        className={errorMessage ? 'input-error' : ''}
      />

      <div className="field-footer">
        <span id="text-hint" className="muted">
          Between 50 and {maxLength} characters.
        </span>
        {/* Character counter, turns red if it goes over the limit */}
        <span className={text.length > maxLength ? 'counter over' : 'counter'}>
          {text.length} / {maxLength}
        </span>
      </div>

      {/* The validation message sits right under the field. It only appears when there is an error. */}
      {errorMessage && (
        <p id="text-error" className="field-error" role="alert">
          {errorMessage}
        </p>
      )}

      {/* The button is disabled while loading so the user can't send the same text twice */}
      <button type="submit" className="button" disabled={isLoading}>
        {isLoading ? 'Checking...' : 'Check text'}
      </button>
    </form>
  );
}

export default TextForm;
