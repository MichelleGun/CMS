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
          {
            label: "passes / episode",
            data: [],
            borderColor: "#3fb68b",
            backgroundColor: "#3fb68b",
            pointRadius: 2,
            tension: 0.15,
            yAxisID: "y2",
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
          y2: {
            position: "right",
            min: 0,
            suggestedMax: 2,
            title: { display: true, text: "passes / episode", color: "#3fb68b" },
            ticks: { color: "#3fb68b" },
            grid: { drawOnChartArea: false },
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
    chart.data.datasets[2].data = [];
    chart.update("none");
  }

  // mean passes per evaluation episode of the viewed run (passes.json), on the right axis
  function setPasses(points) {
    chart.data.datasets[2].data = (points || []).filter((p) => p.mean !== null && p.mean !== undefined)
      .map((p) => ({ x: p.t, y: p.mean }));
    chart.update("none");
  }

  function appendPassesPoint(t, mean) {
    if (mean === null || mean === undefined) return;
    const d = chart.data.datasets[2].data;
    if (d.length && d[d.length - 1].x >= t) return;
    d.push({ x: t, y: mean });
    chart.update("none");
  }

  return { init, setCurrentCurve, appendCurrentPoint, setBestCurve, clearCurrent, setPasses,
           appendPassesPoint };
})();
