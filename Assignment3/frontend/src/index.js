// index.js
// This is the entry point of the app. It is the first file that runs.
// Its only job is to find the <div id="root"> in public/index.html and put our App inside it.

import React from 'react';
import ReactDOM from 'react-dom/client';
// BrowserRouter lets us have different "pages" (URLs) without reloading the browser
import { BrowserRouter } from 'react-router-dom';
import { Chart as ChartJS } from 'chart.js';
import App from './App';
import './App.css';

// Make the text inside the charts use the same font as the rest of the page (the canvas can't inherit it from CSS)
ChartJS.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
ChartJS.defaults.font.size = 13;
ChartJS.defaults.color = '#1f2937';

const root = ReactDOM.createRoot(document.getElementById('root'));

// I wrap App in BrowserRouter so that every component inside it can use routing
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
