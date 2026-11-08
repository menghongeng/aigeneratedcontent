// NotFoundPage.js
// Shown for any address that doesn't exist, for example /banana.
// App.js sends every unknown URL here, so the user never sees a blank screen.

import { Link } from 'react-router-dom';

function NotFoundPage() {
  return (
    <section className="card message-page">
      <h1>404 - Page not found</h1>
      <p>We couldn't find the page you were looking for. The address might be wrong or the page may have moved.</p>
      {/* Link is React Router's version of <a>. It changes page without reloading the browser */}
      <Link className="button" to="/">
        Back to the home page
      </Link>
    </section>
  );
}

export default NotFoundPage;
