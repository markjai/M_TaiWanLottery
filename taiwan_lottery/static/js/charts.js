/* Chart.js helper functions for Taiwan Lottery */

function getGameColor(game) {
    const colors = {
        lotto649: '#e74c3c',
        super_lotto: '#3498db',
        daily_cash: '#2ecc71',
        bingo: '#f39c12',
    };
    return colors[game] || '#6c757d';
}

function getGameColorRGBA(game, alpha = 0.6) {
    const rgb = {
        lotto649: '231, 76, 60',
        super_lotto: '52, 152, 219',
        daily_cash: '46, 204, 113',
        bingo: '243, 156, 18',
    };
    return `rgba(${rgb[game] || '108, 117, 125'}, ${alpha})`;
}

function createBarChart(canvasId, labels, data, label, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: typeof color === 'string'
                    ? data.map(() => color + '99')
                    : color,
                borderColor: typeof color === 'string'
                    ? color
                    : color.map(c => c.replace(/[\d.]+\)$/, '1)')),
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 },
                },
                x: {
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 50,
                    },
                },
            },
        },
    });
}

function createLineChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            scales: {
                y: { beginAtZero: true },
            },
        },
    });
}

function createDoughnutChart(canvasId, labels, data, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' },
            },
        },
    });
}
