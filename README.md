# Taiwan Lottery Predictor

台灣彩券數據收集與 ML 預測系統。自動爬取台灣彩券歷史開獎資料，提供統計分析與機器學習預測功能。

## 支援彩種

- **大樂透** (Lotto 6/49)
- **威力彩** (Super Lotto)
- **今彩539** (Daily Cash)
- **賓果賓果** (Bingo Bingo)

## 功能

- 自動排程爬取歷史開獎號碼
- RESTful API 查詢開獎紀錄與統計資料
- ML 預測模型（Frequency、LSTM、DQN、Ensemble）
- 回測框架評估模型表現
- Web 前端儀表板（Jinja2 模板）

## 技術架構

- **後端**: FastAPI + Uvicorn
- **資料庫**: PostgreSQL (asyncpg)
- **ORM**: SQLAlchemy 2.0 (async)
- **Migration**: Alembic
- **ML**: PyTorch、scikit-learn
- **爬蟲**: TaiwanLotteryCrawler + BeautifulSoup

## 快速開始

### 前置需求

- Python 3.12+
- PostgreSQL（建議使用 Docker）

### 安裝

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate  # Windows

# 安裝套件
pip install -r requirements.txt
```

### 資料庫設定

```bash
# 啟動 PostgreSQL (Docker)
docker run -d --name taiwan_lottery_pg \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=taiwan_lottery \
  -p 5432:5432 \
  postgres:16-alpine

# 複製環境變數設定
cp .env.example .env

# 執行 migration
alembic upgrade head
```

### 啟動

```bash
python -m uvicorn taiwan_lottery.main:app --reload
```

開啟 http://localhost:8000 查看儀表板，API 文件在 http://localhost:8000/docs。

## API 端點

| 路徑 | 說明 |
|------|------|
| `GET /api/v1/lotto649/` | 大樂透開獎紀錄 |
| `GET /api/v1/super_lotto/` | 威力彩開獎紀錄 |
| `GET /api/v1/daily_cash/` | 今彩539開獎紀錄 |
| `GET /api/v1/bingo/` | 賓果賓果開獎紀錄 |
| `GET /api/v1/stats/{game}` | 統計分析 |
| `POST /api/v1/ml/train` | 訓練模型 |
| `POST /api/v1/ml/predict` | 取得預測 |
| `POST /api/v1/scraper/run` | 手動觸發爬蟲 |

## 專案結構

```
taiwan_lottery/
├── api/v1/endpoints/   # API 路由
├── db/
│   ├── models/         # SQLAlchemy 模型
│   └── crud/           # 資料庫操作
├── schemas/            # Pydantic schemas
├── scraper/
│   └── parsers/        # 各彩種解析器
├── services/           # 業務邏輯
├── ml/
│   ├── models/         # ML 模型 (Frequency, LSTM, DQN, Ensemble)
│   ├── features/       # 特徵工程
│   ├── training/       # 訓練流程
│   └── inference/      # 推論與模型管理
├── templates/          # Jinja2 前端模板
├── static/             # CSS / JS
├── config.py           # 設定
└── main.py             # 應用程式進入點
```

## License

MIT
