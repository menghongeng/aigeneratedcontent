// ConfidenceChart.js
// Visualisation 1. The question it answers: "How is the model's belief split for THIS text?"
// It's a doughnut chart showing the chance the text is AI vs the chance it is human.
// The two slices always add up to 100%, which is why a doughnut suits it.

import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { AI_COLOR, HUMAN_COLOR } from '../colors';

// Chart.js only includes the parts you register, which keeps the app small.
// A doughnut needs: ArcElement (the slices), Tooltip (hover box) and Legend (the key).
ChartJS.register(ArcElement, Tooltip, Legend);

function ConfidenceChart({ result }) {
  // The back end gives the confidence of the winning label only,
  // so for the other side the chance is whatever is left over (1 minus confidence).
  const chanceAi = result ? (result.label === 'ai' ? result.confidence : 1 - result.confidence) : 0;
  const chanceHuman = 1 - chanceAi;

  const data = {
    labels: ['AI-generated', 'Human-written'],
    datasets: [
      {
        data: [Math.round(chanceAi * 100), Math.round(chanceHuman * 100)],
        backgroundColor: [AI_COLOR, HUMAN_COLOR],
        borderWidth: 2,
        borderColor: '#ffffff',
      },
    ],
  };

  const options = {
    responsive: true,
    // false lets the chart fill the box whose height I set in the CSS (.chart-box)
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom' },
      tooltip: {
        callbacks: {
          // Customise the hover text so it reads "AI-generated: 87%"
          label: (context) => `${context.label}: ${context.parsed}%`,
        },
      },
    },
  };

  return (
    <section className="card" aria-labelledby="chart1-heading">
      <h2 id="chart1-heading" className="card-title">
        How sure is the model about this text?
      </h2>

      {result ? (
        <div className="chart-box">
          {/* role="img" and aria-label give screen readers a text version of the chart */}
          <Doughnut
            data={data}
            options={options}
            role="img"
            aria-label={`Doughnut chart: ${Math.round(chanceAi * 100)}% chance AI-generated, ${Math.round(chanceHuman * 100)}% chance human-written`}
          />
        </div>
      ) : (
        <p className="muted">Check some text to see how the model's confidence is split.</p>
      )}
    </section>
  );
}

export default ConfidenceChart;
