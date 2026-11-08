// ErrorBoundary.js
// This is the "fallback page for unexpected failures" from the requirements.
// If any component inside it crashes while rendering, React would normally show a blank white screen.
// Instead, this component catches the crash and shows a friendly message.
//
// Note: this is the one class component in the app. React only lets class components act as
// error boundaries (there is no hook for it yet), so I can't write it as a function like the others.

import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    // hasError starts as false. It turns true if something inside crashes.
    this.state = { hasError: false };
  }

  // React calls this automatically when a child crashes. Whatever it returns updates the state.
  static getDerivedStateFromError() {
    return { hasError: true };
  }

  // This also runs on a crash. I just log it so a developer can see what went wrong in the console.
  componentDidCatch(error, info) {
    console.error('Something crashed:', error, info);
  }

  render() {
    if (this.state.hasError) {
      // Crashed: show the fallback page instead of the normal content.
      // I use a normal <a> (not <Link>) so the browser does a full reload, which clears the crash.
      return (
        <section className="card message-page" role="alert">
          <h1>Something went wrong</h1>
          <p>Sorry, the app ran into an unexpected problem. Going back to the home page and trying again should fix it.</p>
          <a className="button" href="/">
            Back to the home page
          </a>
        </section>
      );
    }

    // No crash: just show whatever was inside the ErrorBoundary as normal
    return this.props.children;
  }
}

export default ErrorBoundary;
