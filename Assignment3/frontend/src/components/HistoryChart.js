// HistoryChart.js
// Visualisation 2. The question it answers: "How do ALL the texts I've checked this session compare?"
// It's a bar chart with one bar per check. The bar height is the confidence, and the colour is the result.
// This is different from chart 1, which only looks at the latest check.
//
// Interactivity (so it's more than just a tooltip):
//   1. Filter buttons: show all checks, only AI results, or only human results.
//   2. Click a bar (or press the matching button in the list) to see the details of that check.

import { useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { AI_COLOR, HUMAN_COLOR } from '../colors';

// Register only the parts a bar chart needs
ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// Small helper to shorten long text so it fits in a tooltip or list
function shorten(text, length) {
  return text.length > length ? text.slice(0, length) + '...' : text;
}

// Props:
//   history    - the list of every check made this session
//   selectedId - the id of the check the user picked (or null)
//   onSelect   - function to call when the user picks a check
function HistoryChart({ history, selectedId, onSelect }) {
  // The filter belongs to this chart only, so it is state here instead of in HomePage
  const [filter, setFilter] = useState('all');

  // Work out which checks to show. filter() keeps only the items where the test is true.
  const shown = history.filter((check) => filter === 'all' || check.label === filter);

  // The check the user picked, used for the details box. find() returns the first match, or undefined.
  const selected = history.find((check) => check.id === selectedId);

  // I use TWO datasets (one for AI, one for human) so Chart.js draws a legend with both colours.
  // For each check, only the matching dataset gets a value, the other gets null (no bar).
  function makeDataset(name, color, labelKey) {
    return {
      label: name,
      backgroundColor: color,
      data: shown.map((check) => (check.label === labelKey ? Math.round(check.confidence * 100) : null)),
      // The selected bar gets a thick dark outline so the user can see which one they picked
      borderColor: '#111827',
      borderWidth: shown.map((check) => (check.id === selectedId ? 3 : 0)),
    };
  }

  const data = {
    labels: shown.map((check) => `#${check.number}`),
    datasets: [makeDataset('AI-generated', AI_COLOR, 'ai'), makeDataset('Human-written', HUMAN_COLOR, 'human')],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    // Called when the user clicks the chart. "elements" is the list of bars under the mouse.
    onClick: (event, elements) => {
      if (elements.length > 0) {
        // elements[0].index is the position of the bar, which matches the position in "shown"
        onSelect(shown[elements[0].index].id);
      }
    },
    // Show a pointer (hand) cursor when hovering a bar so it's clear that it can be clicked
    onHover: (event, elements) => {
      event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
    },
    scales: {
      // stacked makes the two datasets share the same spot, so each bar is centred on its check number
      x: { stacked: true, title: { display: true, text: 'Check number (in the order you checked them)' } },
      y: { stacked: true, min: 0, max: 100, title: { display: true, text: 'Confidence (%)' } },
    },
    plugins: {
      legend: { position: 'top' },
      tooltip: {
        callbacks: {
          title: (items) => `Check ${items[0].label}`,
          label: (context) => `${context.dataset.label}: ${context.parsed.y}% confident`,
          // Show the start of the text that was checked
          afterLabel: (context) => `"${shorten(shown[context.dataIndex].text, 60)}"`,
        },
      },
    },
  };

  return (
    <section className="card" aria-labelledby="chart2-heading">
      <h2 id="chart2-heading" className="card-title">
        How do my checks compare this session?
      </h2>

      {history.length === 0 ? (
        <p className="muted">Each text you check will be added here as a bar so you can compare them.</p>
      ) : (
        <>
          {/* Filter buttons. aria-pressed tells screen readers which one is currently on. */}
          <div className="filters" role="group" aria-label="Filter checks by result">
            {[
              ['all', 'All'],
              ['ai', 'AI only'],
              ['human', 'Human only'],
            ].map(([value, text]) => (
              <button
                key={value}
                type="button"
                className={filter === value ? 'chip active' : 'chip'}
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
              >
                {text}
              </button>
            ))}
          </div>

          {shown.length === 0 ? (
            <p className="muted">No checks match this filter yet.</p>
          ) : (
            <div className="chart-box">
              <Bar
                data={data}
                options={options}
                role="img"
                aria-label={`Bar chart of confidence for ${shown.length} checked texts. A list of the same checks follows the chart.`}
              />
            </div>
          )}

          {/* This list has the same checks as the chart, but as buttons. It lets keyboard and screen reader
              users do everything the chart click does, since a canvas can't be reached with the Tab key. */}
          <ul className="check-list">
            {shown.map((check) => (
              <li key={check.id}>
                <button
                  type="button"
                  className={check.id === selectedId ? 'check-button selected' : 'check-button'}
                  aria-pressed={check.id === selectedId}
                  onClick={() => onSelect(check.id)}
                >
                  #{check.number} - {check.label === 'ai' ? 'AI-generated' : 'Human-written'}, {Math.round(check.confidence * 100)}%
                </button>
              </li>
            ))}
          </ul>

          {/* Details of the picked check. aria-live announces it when it changes. */}
          <div className="details" aria-live="polite">
            {selected ? (
              <>
                <h3>Check #{selected.number}</h3>
                <p>
                  <strong>{selected.label === 'ai' ? 'Likely AI-generated' : 'Likely human-written'}</strong> at{' '}
                  {Math.round(selected.confidence * 100)}% confidence
                </p>
                <p className="muted">
                  {selected.text.length} characters, checked at {selected.time}
                </p>
                <p className="excerpt">"{shorten(selected.text, 200)}"</p>
              </>
            ) : (
              <p className="muted">Click a bar, or pick a check from the list, to see its details.</p>
            )}
          </div>
        </>
      )}
    </section>
  );
}

export default HistoryChart;
