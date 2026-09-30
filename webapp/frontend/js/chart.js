// Chart.js reward curve: "current" (the viewed run, appended live from
// WebSocket status frames while it trains) in orange, "best" (the best run
// on disk, drawn once as a static reference) in purple.
const RewardChart = (() => {
  let chart;

  function init(canvas) {
    chart = new Chart(canvas, {
      type: "line",
      data: {
        datasets: [
          {
            label: "current",
            data: [],
            borderColor: "#ff7130",
            backgroundColor: "#ff7130",
            pointRadius: 0,
            tension: 0.15,
          },
          {
            label: "best",
            data: [],
            borderColor: "#872b97",
            backgroundColor: "#872b97",
            pointRadius: 0,
            tension: 0.15,
            borderDash: [4, 3],
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: true,
        parsing: false,
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "timesteps", color: "#9494ab" },
            ticks: { color: "#9494ab" },
            grid: { color: "#34344a" },
          },
          y: {
            title: { display: true, text: "mean eval reward", color: "#9494ab" },
            ticks: { color: "#9494ab" },
            grid: { color: "#34344a" },
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function setCurrentCurve(timesteps, meanReward) {
    chart.data.datasets[0].data = timesteps.map((t, i) => ({ x: t, y: meanReward[i] }));
    chart.update("none");
  }

  function appendCurrentPoint(t, reward) {
    if (reward === null || reward === undefined) return;
    chart.data.datasets[0].data.push({ x: t, y: reward });
    chart.update("none");
  }

  function setBestCurve(timesteps, meanReward) {
    chart.data.datasets[1].data = timesteps.map((t, i) => ({ x: t, y: meanReward[i] }));
    chart.update("none");
  }

  function clearCurrent() {
    chart.data.datasets[0].data = [];
    chart.update("none");
  }

  return { init, setCurrentCurve, appendCurrentPoint, setBestCurve, clearCurrent };
})();
