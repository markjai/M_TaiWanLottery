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

document.addEventListener('DOMContentLoaded', () => {
    loadModels();

    // Toggle pick_count selector visibility for bingo
    document.getElementById('train-game').addEventListener('change', () =>
        togglePickCount('train-game', 'train-pick-count-wrapper'));
    document.getElementById('predict-game').addEventListener('change', () =>
        togglePickCount('predict-game', 'predict-pick-count-wrapper'));
});
