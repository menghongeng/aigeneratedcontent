// App.js
// This is the main component. It holds the parts that appear on every page
// (the skip link, the header) and decides which page to show based on the URL.

import { Routes, Route, Link } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import HomePage from './pages/HomePage';
import NotFoundPage from './pages/NotFoundPage';
import CrashTestPage from './pages/CrashTestPage';

function App() {
  return (
    <>
      {/* Skip link: invisible until a keyboard user presses Tab, then it lets them jump past the header */}
      <a className="skip-link" href="#main">
        Skip to main content
      </a>

      <header className="site-header">
        <div className="container">
          <Link to="/" className="site-title">
            AI Text Checker
          </Link>
        </div>
      </header>

      <main id="main" className="container">
        {/* ErrorBoundary catches unexpected crashes in anything inside it and shows a fallback page */}
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<HomePage />} />
            {/* Only for the demo video: visit /crash-test to see the fallback page. Delete before submitting. */}
            <Route path="/crash-test" element={<CrashTestPage />} />
            {/* The "*" path matches any address that isn't listed above, so it is our 404 page */}
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </>
  );
}

export default App;
