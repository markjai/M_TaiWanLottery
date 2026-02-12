/* Predictions page JavaScript */

const GAME_NAMES = {
    lotto649: '大樂透',
    super_lotto: '威力彩',
    daily_cash: '今彩539',
    bingo: '賓果賓果',
};

const MODEL_NAMES = {
    frequency: '統計頻率',
    lstm: 'LSTM',
    dqn: 'DQN',
    ensemble: '綜合模型',
};

function togglePickCount(selectId, wrapperId) {
    const game = document.getElementById(selectId).value;
    const wrapper = document.getElementById(wrapperId);
    wrapper.style.display = game === 'bingo' ? '' : 'none';
}

async function trainModel() {
    const game = document.getElementById('train-game').value;
    const model = document.getElementById('train-model').value;
    const btn = document.getElementById('train-btn');
    const resultDiv = document.getElementById('train-result');

    const body = { game_type: game, model_type: model };
    if (game === 'bingo') {
        body.pick_count = parseInt(document.getElementById('train-pick-count').value);
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 訓練中...';
    resultDiv.innerHTML = '<div class="alert alert-info">正在訓練模型，請稍候...</div>';

    try {
        const resp = await fetch('/api/v1/ml/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();

        if (resp.ok) {
            const pickInfo = body.pick_count ? ` (取${body.pick_count}個)` : '';
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h6>訓練完成!</h6>
                    <p class="mb-1">模型: ${GAME_NAMES[game]} - ${MODEL_NAMES[model]}${pickInfo}</p>
                    <p class="mb-1">版本: ${data.version}</p>
                    <pre class="mb-0 small">${JSON.stringify(data.metrics, null, 2)}</pre>
                </div>`;
            loadModels();
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger">訓練失敗: ${data.detail || JSON.stringify(data)}</div>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `<div class="alert alert-danger">錯誤: ${e.message}</div>`;
    }

    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> 開始訓練';
}

async function getPrediction() {
    const game = document.getElementById('predict-game').value;
    const model = document.getElementById('predict-model').value;
    const resultDiv = document.getElementById('prediction-result');

    let url = `/api/v1/ml/predict/${game}?model_type=${model}`;
    let pickCount = null;
    if (game === 'bingo') {
        pickCount = parseInt(document.getElementById('predict-pick-count').value);
        url += `&pick_count=${pickCount}`;
    }

    resultDiv.innerHTML = '<div class="text-center"><div class="spinner-border"></div></div>';

    try {
        const resp = await fetch(url);
        const data = await resp.json();

        if (resp.ok) {
            const ballClass = {
                lotto649: 'ball-red',
                super_lotto: 'ball-blue',
                daily_cash: 'ball-green',
                bingo: 'ball-yellow',
            }[game] || 'ball-normal';

            const pickInfo = pickCount ? ` (取${pickCount}個)` : '';
            let html = `
                <div class="prediction-card card p-3">
                    <h6>${GAME_NAMES[game]} - ${MODEL_NAMES[model]} 預測${pickInfo}</h6>
                    <div class="ball-container mb-2">
                        ${data.predicted_numbers.map(n =>
                            `<span class="ball ${ballClass}">${n}</span>`
                        ).join('')}
                    </div>`;

            if (data.confidence_details && data.confidence_details.length > 0) {
                html += `<div class="confidence-details mt-2">`;
                html += `<table class="table table-sm table-bordered mb-2" style="font-size:0.85em">
                    <thead><tr>
                        <th>號碼</th><th>Lift 倍率</th><th>排名</th><th>信心分數</th>
                    </tr></thead><tbody>`;
                for (const d of data.confidence_details) {
                    const liftColor = d.lift >= 1.5 ? 'text-success fw-bold'
                        : d.lift >= 1.1 ? 'text-success'
                        : d.lift >= 0.9 ? 'text-body'
                        : 'text-danger';
                    const barWidth = Math.min(Math.max(d.normalized, 0), 100);
                    const barColor = barWidth >= 70 ? '#28a745'
                        : barWidth >= 40 ? '#ffc107'
                        : '#dc3545';
                    html += `<tr>
                        <td><span class="ball ${ballClass} ball-sm">${d.number}</span></td>
                        <td class="${liftColor}">${d.lift.toFixed(2)}x</td>
                        <td>Top ${(100 - d.percentile).toFixed(0)}%</td>
                        <td>
                            <div class="d-flex align-items-center gap-1">
                                <div style="flex:1;background:#e9ecef;border-radius:3px;height:14px;min-width:60px">
                                    <div style="width:${barWidth}%;background:${barColor};height:100%;border-radius:3px"></div>
                                </div>
                                <span style="min-width:32px">${d.normalized.toFixed(0)}</span>
                            </div>
                        </td>
                    </tr>`;
                }
                html += `</tbody></table>`;
                if (data.expected_random_hit != null) {
                    html += `<div class="small text-muted">
                        隨機期望命中: ${data.expected_random_hit} 個
                        (max_num=${data.max_num}, pick=${data.pick_count})
                    </div>`;
                }
                html += `</div>`;
            } else if (data.confidence_scores) {
                html += `
                    <div class="confidence small text-muted">
                        信心分數: ${data.confidence_scores.map(s => (s * 100).toFixed(1) + '%').join(', ')}
                    </div>`;
            }

            html += `
                    <div class="small text-muted mt-1">
                        產生時間: ${new Date(data.created_at).toLocaleString('zh-TW')}
                    </div>
                </div>`;

            resultDiv.innerHTML = html;
        } else {
            resultDiv.innerHTML = `<div class="alert alert-warning">${data.detail || '無法產生預測，請先訓練模型'}</div>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `<div class="alert alert-danger">錯誤: ${e.message}</div>`;
    }
}

async function loadModels() {
    try {
        const resp = await fetch('/api/v1/ml/models');
        const data = await resp.json();

        if (data.length === 0) {
            document.getElementById('models-tbody').innerHTML =
                '<tr><td colspan="6" class="text-center text-muted">尚未訓練任何模型</td></tr>';
            return;
        }

        document.getElementById('models-tbody').innerHTML = data.map(m => {
            // Parse model_type display name (handle _pN suffix)
            let displayType = MODEL_NAMES[m.model_type] || m.model_type;
            const pMatch = m.model_type.match(/^(\w+)_p(\d+)$/);
            if (pMatch) {
                const baseName = MODEL_NAMES[pMatch[1]] || pMatch[1];
                displayType = `${baseName} (取${pMatch[2]}個)`;
            }
            return `
            <tr>
                <td>${m.id}</td>
                <td>${GAME_NAMES[m.game_type] || m.game_type}</td>
                <td>${displayType}</td>
                <td>${m.version}</td>
                <td>${m.is_active
                    ? '<span class="badge bg-success">啟用</span>'
                    : '<span class="badge bg-secondary">停用</span>'}</td>
                <td>${new Date(m.trained_at).toLocaleString('zh-TW')}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        document.getElementById('models-tbody').innerHTML =
            `<tr><td colspan="6" class="text-center text-danger">載入失敗</td></tr>`;
    }
}

async function evaluateModel() {
    const game = document.getElementById('eval-game').value;
    const resultDiv = document.getElementById('eval-result');

    resultDiv.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div></div>';

    try {
        const resp = await fetch('/api/v1/ml/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_type: game }),
        });
        const data = await resp.json();

        if (resp.ok) {
            resultDiv.innerHTML = `
                <div class="card p-3">
                    <h6>${GAME_NAMES[game]} 模型評估</h6>
                    <p class="mb-1">總預測次數: ${data.total_predictions}</p>
                    <p class="mb-1">平均命中數: ${data.average_hits}</p>
                    <p class="mb-0">命中分佈: ${JSON.stringify(data.hit_distribution)}</p>
                </div>`;
        } else {
            resultDiv.innerHTML = `<div class="alert alert-warning">${data.detail || '評估失敗'}</div>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `<div class="alert alert-danger">錯誤: ${e.message}</div>`;
    }
}

/* ── Backtest ───────────────────────────────────────────────── */

let btQuarterChart = null;
let btHitDistChart = null;

async function runBacktest(compareAll = false) {
    const game = document.getElementById('bt-game').value;
    const model = document.getElementById('bt-model').value;
    const testSize = parseInt(document.getElementById('bt-test-size').value);
    const btn = document.getElementById(compareAll ? 'bt-compare-btn' : 'bt-btn');
    const resultDiv = document.getElementById('bt-result');

    const body = {
        game_type: game,
        model_type: model,
        test_size: testSize,
        compare_all: compareAll,
    };
    if (game === 'bingo') {
        body.pick_count = parseInt(document.getElementById('bt-pick-count').value);
    }

    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 回測中...';
    resultDiv.innerHTML = `
        <div class="alert alert-info">
            <div class="d-flex align-items-center gap-2">
                <div class="spinner-border spinner-border-sm"></div>
                正在執行回測，可能需要數十秒至數分鐘...
            </div>
        </div>`;

    try {
        const resp = await fetch('/api/v1/ml/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();

        if (!resp.ok) {
            resultDiv.innerHTML = `<div class="alert alert-danger">${data.detail || JSON.stringify(data)}</div>`;
            return;
        }

        if (compareAll && data.comparison) {
            resultDiv.innerHTML = renderComparison(data, game);
        } else if (data.backtest) {
            resultDiv.innerHTML = renderBacktestResult(data.backtest, game);
        } else {
            resultDiv.innerHTML = `<div class="alert alert-warning">無回測結果</div>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `<div class="alert alert-danger">錯誤: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = origHtml;
    }
}

function renderBacktestResult(bt, game) {
    // Significance badge
    const sigBadge = bt.is_significant
        ? '<span class="badge bg-success">顯著優於隨機</span>'
        : '<span class="badge bg-secondary">未達顯著</span>';

    // p-value color
    const pColor = bt.p_value < 0.01 ? 'text-success fw-bold'
        : bt.p_value < 0.05 ? 'text-success'
        : bt.p_value < 0.1 ? 'text-warning'
        : 'text-danger';

    // Effect size interpretation
    const esLabel = Math.abs(bt.effect_size) >= 0.8 ? '大效果'
        : Math.abs(bt.effect_size) >= 0.5 ? '中效果'
        : Math.abs(bt.effect_size) >= 0.2 ? '小效果'
        : '極小效果';

    // Lift color
    const liftColor = bt.lift_vs_random >= 1.1 ? 'text-success fw-bold'
        : bt.lift_vs_random >= 1.0 ? 'text-success'
        : 'text-danger';

    // Quarterly trend arrow
    const qp = bt.quarterly_performance || [];
    let trendIcon = '';
    if (qp.length === 4) {
        const firstHalf = (qp[0] + qp[1]) / 2;
        const secondHalf = (qp[2] + qp[3]) / 2;
        trendIcon = secondHalf > firstHalf + 0.05
            ? '<i class="bi bi-arrow-up-circle-fill text-success"></i> 上升趨勢'
            : secondHalf < firstHalf - 0.05
            ? '<i class="bi bi-arrow-down-circle-fill text-danger"></i> 下降趨勢'
            : '<i class="bi bi-dash-circle text-muted"></i> 持平';
    }

    // Hit distribution bar
    const hitDist = bt.hit_distribution || {};
    const totalTests = bt.test_size;
    let hitDistBars = '';
    const distColors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745', '#0d6efd', '#6610f2'];
    for (const [hits, count] of Object.entries(hitDist)) {
        const pct = (count / totalTests * 100).toFixed(1);
        const color = distColors[Math.min(parseInt(hits), distColors.length - 1)];
        hitDistBars += `
            <div class="d-flex align-items-center gap-1 mb-1">
                <span style="min-width:60px">${hits} 中: ${count}次</span>
                <div style="flex:1;background:#e9ecef;border-radius:3px;height:18px">
                    <div style="width:${pct}%;background:${color};height:100%;border-radius:3px;min-width:${pct > 0 ? '2px' : '0'}"></div>
                </div>
                <span class="small text-muted" style="min-width:40px">${pct}%</span>
            </div>`;
    }

    // Quarterly chart
    const qpId = 'bt-quarter-chart-' + Date.now();
    const hitDistId = 'bt-hitdist-chart-' + Date.now();

    const html = `
        <div class="row g-3">
            <!-- 左欄：核心指標 -->
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <strong>${MODEL_NAMES[bt.model_type] || bt.model_type} 回測結果</strong>
                        ${sigBadge}
                    </div>
                    <div class="card-body">
                        <table class="table table-sm mb-3">
                            <tr><td>測試期數</td><td class="fw-bold">${bt.test_size} 期</td></tr>
                            <tr><td>訓練資料</td><td>${bt.train_size} 期</td></tr>
                            <tr>
                                <td>模型平均命中</td>
                                <td class="fw-bold fs-5">${bt.average_hits}</td>
                            </tr>
                            <tr>
                                <td>蒙特卡羅隨機基準</td>
                                <td>${bt.monte_carlo_avg} <small class="text-muted">(SD: ${bt.monte_carlo_std})</small></td>
                            </tr>
                            <tr>
                                <td>理論隨機期望值</td>
                                <td>${bt.expected_random}</td>
                            </tr>
                            <tr>
                                <td>Lift 倍率（vs 隨機）</td>
                                <td class="${liftColor}">${bt.lift_vs_random}x</td>
                            </tr>
                        </table>

                        <h6 class="mt-3">統計顯著性</h6>
                        <table class="table table-sm">
                            <tr>
                                <td>p-value</td>
                                <td class="${pColor}">${bt.p_value < 0.001 ? '< 0.001' : bt.p_value.toFixed(4)}</td>
                            </tr>
                            <tr>
                                <td>效果量 (Cohen's d)</td>
                                <td>${bt.effect_size.toFixed(4)} <small class="text-muted">(${esLabel})</small></td>
                            </tr>
                            <tr>
                                <td>95% 信賴區間</td>
                                <td>[${bt.confidence_interval_95[0]}, ${bt.confidence_interval_95[1]}]</td>
                            </tr>
                            <tr><td>命中標準差</td><td>${bt.std_hits}</td></tr>
                            <tr><td>最大/最小命中</td><td>${bt.max_hits} / ${bt.min_hits}</td></tr>
                            <tr><td>命中率（非零）</td><td>${(bt.hit_rate_nonzero * 100).toFixed(1)}%</td></tr>
                        </table>
                    </div>
                </div>
            </div>

            <!-- 右欄：圖表 -->
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">
                        <strong>命中分佈</strong>
                    </div>
                    <div class="card-body">
                        ${hitDistBars}
                    </div>
                </div>

                <div class="card mb-3">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <strong>各季度表現</strong>
                        <small>${trendIcon}</small>
                    </div>
                    <div class="card-body">
                        <canvas id="${qpId}" height="150"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header"><strong>連勝/連敗紀錄</strong></div>
                    <div class="card-body py-2">
                        <span class="badge bg-success me-2">最長連續命中: ${bt.best_hit_streak} 期</span>
                        <span class="badge bg-danger">最長連續未中: ${bt.worst_miss_streak} 期</span>
                    </div>
                </div>
            </div>
        </div>`;

    // Render charts after DOM update
    setTimeout(() => {
        const qpCanvas = document.getElementById(qpId);
        if (qpCanvas && qp.length > 0) {
            if (btQuarterChart) btQuarterChart.destroy();
            btQuarterChart = new Chart(qpCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['Q1 (早期)', 'Q2', 'Q3', 'Q4 (近期)'],
                    datasets: [{
                        label: '平均命中',
                        data: qp,
                        backgroundColor: ['#6c757d99', '#6c757d99', '#ffc10799', '#28a74599'],
                        borderColor: ['#6c757d', '#6c757d', '#ffc107', '#28a745'],
                        borderWidth: 1,
                    }, {
                        label: '隨機基準',
                        data: [bt.monte_carlo_avg, bt.monte_carlo_avg, bt.monte_carlo_avg, bt.monte_carlo_avg],
                        type: 'line',
                        borderColor: '#dc3545',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                    }],
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: true, position: 'bottom' } },
                    scales: { y: { beginAtZero: true, title: { display: true, text: '平均命中數' } } },
                },
            });
        }
    }, 100);

    return html;
}

function renderComparison(data, game) {
    const rows = data.comparison.map(c => {
        if (c.error) {
            return `<tr><td>${MODEL_NAMES[c.model] || c.model}</td><td colspan="7" class="text-danger">${c.error}</td></tr>`;
        }
        const sigBadge = c.is_significant
            ? '<span class="badge bg-success">Yes</span>'
            : '<span class="badge bg-secondary">No</span>';
        const liftColor = c.lift_vs_random >= 1.1 ? 'text-success fw-bold'
            : c.lift_vs_random >= 1.0 ? 'text-success' : 'text-danger';
        const pColor = c.p_value < 0.05 ? 'text-success' : 'text-muted';
        return `
            <tr>
                <td class="fw-bold">${MODEL_NAMES[c.model] || c.model}</td>
                <td class="fw-bold">${c.avg_hits}</td>
                <td>${c.std_hits}</td>
                <td>${c.max_hits}</td>
                <td class="${liftColor}">${c.lift_vs_random}x</td>
                <td>${(c.hit_rate_nonzero * 100).toFixed(1)}%</td>
                <td class="${pColor}">${c.p_value != null ? (c.p_value < 0.001 ? '< 0.001' : c.p_value.toFixed(4)) : '-'}</td>
                <td>${sigBadge}</td>
            </tr>`;
    }).join('');

    return `
        <div class="card">
            <div class="card-header"><strong>所有模型比較 (${data.comparison.length} 個模型, ${data.comparison[0]?.expected_random ? '隨機期望 ' + data.comparison[0].expected_random : ''})</strong></div>
            <div class="card-body table-responsive">
                <table class="table table-hover table-sm">
                    <thead>
                        <tr>
                            <th>模型</th>
                            <th>平均命中</th>
                            <th>標準差</th>
                            <th>最大命中</th>
                            <th>Lift 倍率</th>
                            <th>命中率</th>
                            <th>p-value</th>
                            <th>顯著?</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>`;
}

document.addEventListener('DOMContentLoaded', () => {
    loadModels();

    // Toggle pick_count selector visibility for bingo
    document.getElementById('train-game').addEventListener('change', () =>
        togglePickCount('train-game', 'train-pick-count-wrapper'));
    document.getElementById('predict-game').addEventListener('change', () =>
        togglePickCount('predict-game', 'predict-pick-count-wrapper'));

    const btGame = document.getElementById('bt-game');
    if (btGame) {
        btGame.addEventListener('change', () =>
            togglePickCount('bt-game', 'bt-pick-count-wrapper'));
    }
});
