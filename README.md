# MI Toolkit — 輕量級機制可解釋性分析工具

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

**MI Toolkit** 是一套端到端的輕量級機制可解釋性（Mechanistic Interpretability）分析工具，整合 **ReIP**（Relevance-Integrated Patching）、**WeightLens** 靜態權重語意解析引擎、**CircuitLens** 上下文依賴解構模組，以及互動式 Dash 儀表板，目標是在單張 NVIDIA RTX 4090（24GB VRAM）上完成從模型載入到神經元迴路視覺化的完整分析流程。

---

## 核心特性

| 模組 | 功能 | 計算複雜度 |
|------|------|-----------|
| **ReIP** | LRP/梯度-激活差分原型（planned validation benchmark: PCC vs activation patching） | O(2F + B) |
| **WeightLens** | 無資料集、無外部 LLM 的靜態特徵語意提取 | O(V × d) |
| **CircuitLens** | 目標特徵對局部殘差流的敏感度分析 Prototype（Attention head 分解目前為 Placeholder） | O(N²) |
| **Dashboard** | 拓樸圖的視覺化 Demo UI（實際 ReIP 執行尚未串接） | — |
| **Hardware** | VRAM 自動分配、INT8 量化、Air-Gapped 離線模式 | — |

---

## 快速開始

### 安裝

```bash
# 克隆儲存庫
git clone https://github.com/rocky59487/MI.git
cd MI

# 建立虛擬環境（建議）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安裝核心依賴
pip install -e .

# 安裝所有可選依賴（量化、監控、開發工具）
pip install -e ".[all]"

# 下載 spaCy 英文模型（用於詞形還原）
python -m spacy download en_core_web_sm
```

### ReIP 迴路分析

```python
from transformer_lens import HookedTransformer
from src.reip import ReIPPipeline, ReIPConfig

# 載入模型（自動選擇最佳 VRAM 配置）
model = HookedTransformer.from_pretrained("gpt2")

# 配置 ReIP 管線
config = ReIPConfig(
    model_name="gpt2",
    pruning_threshold=0.02,
    pruning_top_k=500,
    verbose=True,
)
pipeline = ReIPPipeline(model, config)

# 執行 IOI 任務分析
result = pipeline.run(
    clean_prompt="When Mary and John went to the store, John gave a drink to",
    corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
    target_token=" Mary",
)

print(f"關聯性拓樸圖節點數: {len(result.topology_graph.nodes)}")
print(f"分析耗時: {result.runtime_seconds:.3f}s")
```

### WeightLens 特徵語意提取

```python
from src.weightlens import TranscoderLoader, VocabProjector, SemanticLemmatizer, FeatureCache

# 載入轉碼器權重
loader = TranscoderLoader(model_name="gpt2")
weights = loader.load_layer(layer_idx=6)

# 若無真實轉碼器，使用合成版本進行測試
if weights is None:
    weights = loader.create_synthetic_transcoder(layer_idx=6, d_model=768, n_features=3072)

# 詞彙空間投影
projector = VocabProjector(
    W_embed=model.embed.W_E,
    W_unembed=model.unembed.W_U.T,
    tokenizer=model.tokenizer,
    zscore_input=4.0,
)
semantics = projector.analyze_feature(weights.W_enc, weights.W_dec, feature_idx=42, layer_idx=6)

# 生成語意標籤
lemmatizer = SemanticLemmatizer()
label = lemmatizer.generate_label(
    semantics.input_tokens,
    semantics.output_tokens_promoted,
    semantics.output_tokens_suppressed,
)
print(f"特徵語意: {label}")

# 快取結果
cache = FeatureCache(model_name="gpt2")
semantics.raw_label = label
cache.store(semantics)
```

### 啟動互動式儀表板

```bash
# 啟動 Dash 儀表板（預設 port 8050）
python dashboards/app.py

# 或使用環境變數配置
MI_PORT=8080 MI_DEBUG=1 python dashboards/app.py
```

瀏覽器開啟 `http://localhost:8050` 即可看到互動式迴路視覺化介面。

---

## 儲存庫結構

```
MI/
├── src/
│   ├── reip/                    # ReIP 動態迴路定位
│   │   ├── __init__.py
│   │   ├── lrp_rules.py         # 五種 LRP 傳播規則
│   │   ├── backward_hooks.py    # TransformerLens 反向傳播鉤子
│   │   ├── pruning.py           # 稀疏拓撲圖修剪
│   │   └── pipeline.py          # 端到端執行管線
│   ├── weightlens/              # WeightLens 靜態語意解析
│   │   ├── __init__.py
│   │   ├── transcoder_loader.py # 轉碼器權重載入
│   │   ├── projection.py        # Z-分數詞彙投影
│   │   ├── lemmatizer.py        # 本地詞形還原
│   │   └── cache.py             # JSON 快取庫
│   ├── circuitlens/             # CircuitLens 上下文解構
│   │   ├── __init__.py
│   │   ├── jacobian.py          # 雅可比矩陣計算
│   │   ├── jaccard.py           # Jaccard 相似度矩陣
│   │   └── clustering.py        # DBSCAN 分群
│   └── hardware/                # 硬體效能最佳化
│       ├── __init__.py
│       ├── vram_manager.py      # VRAM 分配矩陣
│       └── monitor.py           # 非同步背景監控
├── dashboards/                  # 互動式儀表板
│   ├── __init__.py
│   ├── app.py                   # Dash 應用程式進入點
│   ├── layout.py                # 頁面佈局與資料轉換
│   ├── callbacks.py             # 互動回調函式
│   └── stylesheet.py            # Cytoscape 視覺樣式表
├── tests/                       # 測試驗證套件
│   ├── __init__.py
│   ├── conftest.py              # pytest fixtures
│   ├── test_reip_ioi.py         # IOI 任務保真度驗證
│   ├── test_fade.py             # FADE 評估框架
│   ├── test_weightlens.py       # WeightLens 單元測試
│   └── test_circuitlens.py      # CircuitLens 單元測試
├── configs/                     # 配置檔案
│   ├── hardware.yaml            # 硬體配置
│   ├── quantization.yaml        # 量化策略
│   └── zscore_thresholds.yaml   # Z-分數閾值
├── pyproject.toml               # 專案依賴與建構配置
├── README.md                    # 本文件
└── implementation_plan.md       # 詳盡實作計畫文件
```

---

## 技術棧

| 類別 | 套件 | 版本要求 |
|------|------|---------|
| 核心框架 | `torch` | ≥ 2.1.0 |
| 模型介面 | `transformer-lens` | ≥ 1.19.0 |
| 科學計算 | `numpy`, `scipy` | ≥ 1.24, ≥ 1.11 |
| 機器學習 | `scikit-learn` | ≥ 1.3.0 |
| NLP | `spacy` | ≥ 3.7.0 |
| 圖分析 | `networkx` | ≥ 3.2.0 |
| 視覺化 | `dash`, `dash-cytoscape` | ≥ 2.14, ≥ 0.3 |
| API 伺服器 | `fastapi`, `uvicorn` | ≥ 0.104, ≥ 0.24 |
| 量化（選用） | `bitsandbytes` | ≥ 0.41.0 |
| 監控（選用） | `pynvml`, `psutil` | ≥ 11.5, ≥ 5.9 |

---

## VRAM 分配矩陣

| 模型規模 | 精度 | 策略 | 估計 VRAM |
|---------|------|------|----------|
| 1.5B 參數 | FP16 | 全模型載入 VRAM | ~3.6 GB |
| 2–3B 參數 | FP16 | 全模型載入 VRAM | ~6 GB |
| 7–8B 參數 | INT8/FP8 | bitsandbytes 8-bit 量化 | ~9.6 GB |
| 9B+ 參數 | INT8/FP8 | 8-bit + KV-cache 分頁 | ~12 GB |

---

## 測試

```bash
# 執行所有單元測試
pytest tests/ -v

# 執行特定測試套件
pytest tests/test_reip_ioi.py -v
pytest tests/test_fade.py -v
pytest tests/test_weightlens.py -v
pytest tests/test_circuitlens.py -v

# 跳過需要 GPU 的測試
pytest tests/ -v -m "not gpu and not integration"

# 生成覆蓋率報告
pytest tests/ --cov=src --cov-report=html
```

---

## 驗證基準

| 指標 | 目標值 | 對比基準 |
|------|--------|---------|
| ReIP PCC（IOI 任務） | Planned benchmark (not yet claimed) | AtP baseline pending |
| FADE 純度（Purity） | > 0.80 | MaxAct + GPT-4 |
| FADE 清晰度（Clarity） | > 0.70 | MaxAct + GPT-4 |
| FADE 響應度（Responsiveness） | > 0.75 | MaxAct + GPT-4 |
| FADE 保真度（Fidelity） | > 0.80 | MaxAct + GPT-4 |

---

## 參考文獻

1. Circuit Insights: Towards Interpretability Beyond Activations (OpenReview)
2. FarnoushRJ/RelP (GitHub)
3. Circuit Insights (arXiv)
4. akshathmangudi/weightlens (GitHub)
5. TransformerLens (GitHub)
6. egolimblevskaia/WeightLens (GitHub)
7. FADE: Why Bad Descriptions Happen to Good Features (arXiv)

---

## 授權

MIT License — 詳見 [LICENSE](LICENSE) 文件。
