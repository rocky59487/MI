# 輕量級機制可解釋性分析工具：實作計畫文件

> **儲存庫**：[rocky59487/MI](https://github.com/rocky59487/MI)
> **文件版本**：v1.0
> **建立日期**：2026-05-05

---

## 1. 專案總覽與目標

本專案旨在為 GitHub 儲存庫 `rocky59487/MI` 構建一套「輕量級機制可解釋性分析工具（Lightweight Mechanistic Interpretability Toolkit）」。該工具的核心目標是無縫整合「相關性修補（Relevance Patching, RelP）」與「靜態權重解析框架（WeightLens/CircuitLens）」，以解決當代大型語言模型（LLMs）在自動化神經元迴路發現中所面臨的計算複雜度與認識論挑戰。

在機制可解釋性（Mechanistic Interpretability）研究領域中，傳統的「激勵修補（Activation Patching）」需要進行 $O(|E|)$ 次的窮舉前向傳播，對消費級硬體而言完全不可行。學界雖提出「歸因修補（Attribution Patching, AtP）」以降低計算成本，但此方法在深層網路中遭遇致命的「梯度病理（Gradient Pathology）」問題，導致皮爾森相關係數（PCC）僅達 0.006，幾乎完全失效。本專案透過引入層級相關性傳播（LRP）衍生出的傳播係數，徹底解決梯度崩潰問題，並結合轉碼器（Transcoder）架構進行靜態權重解析，實現無資料集（Dataset-Free）、無模型依賴（Explainer-Free）的特徵語意提取。

最終交付要求是確保整個端到端分析管線能夠在單張消費級 **NVIDIA RTX 4090（24GB VRAM）** 顯示卡上，以毫秒至數十秒級別的速度穩定運行，完成從動態因果節點定位到靜態人類語意標註的完整流程。本工具的設計哲學是將原本專屬於頂尖實驗室的迴路發現能力，下放至獨立研究者與學術單位的桌面環境。

---

## 2. 儲存庫架構規劃

為了確保專案的模組化、可維護性以及與開源社群最佳實踐接軌，儲存庫將採用以下高度結構化的目錄設計。頂層目錄劃分清晰隔離了不同生命週期的程式碼，並能確保持續整合與持續部署（CI/CD）管線的順暢運行。

```
rocky59487/MI/
├── src/
│   ├── reip/                        # 動態因果迴路定位引擎
│   │   ├── __init__.py
│   │   ├── lrp_rules.py             # LN-rule, Identity-rule, AH-rule, Half-rule, 0-rule
│   │   ├── backward_hooks.py        # TransformerLens add_hook(dir="bwd") 整合
│   │   ├── pipeline.py              # 端到端執行管線（run_with_cache → LRP → 修剪）
│   │   └── pruning.py               # 稀疏拓撲圖修剪演算法
│   ├── weightlens/                  # 靜態權重語意解析引擎
│   │   ├── __init__.py
│   │   ├── transcoder_loader.py     # 轉碼器權重載入器
│   │   ├── projection.py            # 詞彙空間投影與 Z-分數篩選
│   │   ├── lemmatizer.py            # spaCy 本地詞形還原
│   │   └── cache.py                 # JSON 快取庫讀寫機制
│   └── circuitlens/                 # 上下文依賴與多義性解構
│       ├── __init__.py
│       ├── jacobian.py              # 雅可比矩陣計算
│       ├── jaccard.py               # Jaccard 相似度矩陣產生器
│       └── clustering.py            # DBSCAN 無監督分群
├── dashboards/
│   ├── app.py                       # Dash 應用程式主入口
│   ├── layout.py                    # 頁面佈局定義
│   ├── callbacks.py                 # 懸浮/點擊互動回呼函式
│   └── stylesheet.py                # Cytoscape 視覺樣式表
├── tests/
│   ├── __init__.py
│   ├── test_reip_ioi.py             # IOI 任務保真度驗證（PCC > 0.95）
│   ├── test_weightlens_fade.py      # FADE 評估框架整合測試
│   ├── test_lrp_rules.py            # LRP 規則數學守恆性單元測試
│   └── benchmarks/
│       └── adversarial_baseline.py  # WeightLens vs MaxAct + GPT-4 對抗測試
├── configs/
│   ├── hardware.yaml                # RTX 4090 記憶體限制與 Tiling 策略
│   ├── quantization.yaml            # FP8/INT8 量化設定
│   └── zscore_thresholds.yaml       # 各模型 Z-分數閾值設定
├── pyproject.toml                   # 專案依賴與建構設定（uv 管理）
├── .gitignore
└── README.md
```

以下表格詳述各核心模組的功能定位與實作規範：

| 目錄路徑 | 模組功能定位 | 實作規範摘要 |
| :--- | :--- | :--- |
| `/src/reip/` | 動態因果迴路定位引擎 | 包含客製化的反向傳播鉤子、五種 LRP 傳播規則實作，以及與 TransformerLens 的整合介面。 |
| `/src/weightlens/` | 靜態權重語意解析引擎 | 包含轉碼器權重載入器、Z-分數投影計算矩陣乘法邏輯，以及本地決定性詞形還原演算法。 |
| `/src/circuitlens/` | 上下文依賴與多義性解構 | 實作 Jacobian 矩陣計算、Jaccard 相似度矩陣產生器與 DBSCAN 無監督分群演算法模組。 |
| `/dashboards/` | 互動式視覺化前端介面 | 基於 Dash 與 Plotly Cytoscape 的網路圖形渲染邏輯、樣式表與滑鼠懸浮回呼函式。 |
| `/tests/` | 基準測試與單元測試套件 | IOI 任務保真度驗證腳本、FADE 指標評估模組，確保每次 Commit 不破壞數學守恆性。 |
| `/configs/` | 模型量化與硬體配置文件 | 儲存針對 RTX 4090 的記憶體限制參數、FP8/INT8 量化設定檔與預設 Z-分數閾值設定。 |

---

## 3. 分階段開發里程碑與任務拆解

本專案的開發分為六個主要階段，總預計工時約 **10 週**。各階段的任務拆解、預計工時與依賴關係如下所述。

### 階段一：ReIP 動態神經元迴路定位系統（預計工時：2 週）

本階段的核心目標是實作客製化 LRP 規則，並將其整合至 TransformerLens 框架中，建立端到端的高效執行管線。**依賴關係：無（首要開發項目）。**

ReIP 模組的關鍵工作是在 `lrp_rules.py` 中，以 PyTorch `autograd.Function` 精確實作下表所列的五種傳播規則。每一規則皆對應特定的 Transformer 組件，並有明確的數學機制要求：

| Transformer 組件 | LRP 規則 | 核心機制 | 適用模型 |
| :--- | :--- | :--- | :--- |
| 層正規化（LayerNorm/RMSNorm） | LN-rule | 在反向傳播期間，將中心化運算與基於變異數的縮放因子強制視為常數，切斷梯度流經變異數計算的複雜路徑，防止「相關性崩潰」。 | GPT-2, Pythia, Qwen2, Gemma2-2B |
| 非線性激勵函數（GELU, SiLU） | Identity-rule | 將非線性函數的導數強制覆寫為 1，確保特徵相關性信號在跨越非線性邊界時嚴格維持守恆特性。 | GPT-2, Pythia, Qwen2, Gemma2-2B |
| 注意力機制（Attention） | AH-rule | 將龐大的注意力權重矩陣視為反向傳播的常數，確保相關性分數能精準分配至 Key、Query 與 Value 流。 | GPT-2 等早期架構 |
| 乘法閘機制（Multiplicative Gates） | Half-rule | 將進入乘法分支的相關性分數強制平均分配至分支兩端（各 50%），防止信號在反向傳遞中產生虛假倍增。 | Qwen2, Gemma2-2B |
| 線性投射層（Linear Layers） | 0-rule | 數學上等同於 Gradient × Input，設定為所有標準前饋神經網路與線性映射層的預設 LRP 模式。 | GPT-2, Pythia, Qwen2, Gemma2-2B |

在 `backward_hooks.py` 中，開發者需透過 TransformerLens 的 `add_hook(dir="bwd")` 攔截標準梯度，並注入上述 LRP 規則。使用者介面應支援透過設定 `model.cfg.use_lrp = True` 與 `model.cfg.LRP_rules = [...]` 來啟動不同的傳播規則。`pipeline.py` 負責組裝端到端執行管線：首先呼叫 `run_with_cache` 分別對乾淨輸入與損壞輸入執行前向傳播，接著啟動單次 LRP 修改的反向傳播，最後由 `pruning.py` 剔除低貢獻分數的節點，輸出稀疏的因果計算拓撲圖（以 NetworkX 結構或 JSON 字典儲存）。

### 階段二：WeightLens 靜態權重解析（預計工時：2 週）

本階段的目標是透過直接從靜態學習權重中解析特徵，提供無資料集、無模型依賴的語意解析解決方案。**依賴關係：階段一完成。**

WeightLens 的解析流程包含三個核心步驟。第一步，**候選標記提取**：在 `projection.py` 中，將轉碼器特徵的編碼器向量（$W_{enc}$）透過模型的嵌入矩陣（$W_{embed}$）直接點積投影至輸入詞彙空間，並以 Z-分數識別統計上異常突出的標記。第二步，**輸出效應分析**：將特徵的解碼器向量（$W_{dec}$）透過解嵌入矩陣（$W_{unembed}$）投影至輸出詞彙的邏輯值空間，找出被該特徵強烈影響機率的輸出標記。第三步，**本地詞形還原**：在 `lemmatizer.py` 中整合 spaCy，將提取出的高 Z-分數標記的各種屈折形式整合為單一基礎概念描述，徹底取代外部 LLM 後處理步驟。

Z-分數閾值的設定需依據不同模型動態調整，如下表所示：

| 模型系列 | Z-分數閾值 | 說明 |
| :--- | :---: | :--- |
| GPT-2 | 4.0 | 詞彙表較小，閾值相對較低。 |
| Gemma-2-2B / Llama-3.2-1B | 4.5 | 詞彙表龐大，需提升閾值確保僅保留最核心的驅動詞彙。 |
| 層間相連特徵（Connected Features） | 3.0 | 識別跨層特徵連接時使用較低閾值。 |

`cache.py` 負責實作 JSON 格式的本地預先計算快取庫。當系統接收到 ReIP 輸出的特徵拓撲圖時，WeightLens 模組首先查詢本地快取；若快取命中，直接讀取語意標籤；若快取未命中，才在 GPU 上啟動矩陣乘法與 Z-分數篩選，並將結果序列化寫入 JSON 檔案供未來重複使用。

### 階段三：CircuitLens 上下文依賴解構（預計工時：2 週）

本階段的目標是解決深層特徵的上下文依賴與多義性問題，提升語意解析的純度指標。**依賴關係：階段一、階段二完成。**

在採用旋轉位置編碼（RoPE）的模型（如 Llama 與 Gemma）的中段層次中，特徵會變得高度依賴上下文，單純依賴靜態權重分析會導致純度（Purity）指標下降。CircuitLens 透過兩個機制解決此問題。

首先，`jacobian.py` 實作雅可比矩陣計算，精確衡量特定輸入標記透過注意力頭網路傳遞給目標轉碼器特徵的確切梯度貢獻。開發者需編寫資料傳遞管道，將 ReIP 在第一階段計算出的高保真 LRP 歸因分數直接饋入雅可比矩陣運算核心，動態隔離並遮蔽（Masking）產生干擾的無關輸入，無需執行任何額外的激勵探測。

其次，`jaccard.py` 與 `clustering.py` 共同實作基於迴路計算拓撲的分群演算法。系統首先收集導致特定特徵激發的「注意力頭與標記配對」，計算這些配對集合之間的 Jaccard 相似度矩陣，量化不同輸入樣本觸發相同特徵時底層計算迴路的重疊程度。最後，DBSCAN 演算法對此矩陣進行無監督分群，將多義性特徵優雅地拆解為多個單義性（Monosemantic）子叢集，並為每個子叢集獨立生成精準的描述。

### 階段四：互動式儀表板（預計工時：1.5 週）

本階段的目標是建構高效能的互動式儀表板，視覺化渲染機制可解釋性分析結果。**依賴關係：階段一、二、三完成。**

`layout.py` 負責編寫資料轉換模組，將 ReIP 產生的稀疏拓撲字典與 WeightLens/CircuitLens 產生的 JSON 語意標籤，無縫轉換為 Cytoscape 可識別的 `elements` 列表。每個節點的資料結構必須包含 `data: {'id': 'layer_x_feature_y', 'label': 'Semantic Description'}`，其中 `label` 即為 WeightLens 解析出的人類可讀語意（例如「時間狀語提升」或「複數名詞檢測」）。

`stylesheet.py` 需撰寫複雜的 Cytoscape 樣式表，利用選擇器（Selectors）將 ReIP 計算出的因果歸因分數精確對映至邊緣的粗細（`width`）與顏色深淺（`line-color`）。佈局演算法需設定為層級導向的 `dagre` 或 `breadthfirst`，確保特徵節點能從輸入嵌入層，跨越各層 MLP 轉碼器，一路平滑過渡至最終的邏輯（Logits）輸出層，清晰呈現資訊流路徑。

`callbacks.py` 需實作強大的 Dash Callbacks，當使用者游標懸浮或點擊特定節點時，瞬間顯示一個富含資訊的懸浮提示框，動態渲染 WeightLens 解析出的詳細語意描述、統計信心水準（Z-分數大小）、核心驅動詞彙清單，以及 CircuitLens DBSCAN 演算法輸出的所屬分群結果。

### 階段五：硬體效能最佳化（預計工時：1.5 週）

本階段的目標是確保系統在單張 RTX 4090 上穩定運行，並符合企業級安全需求。**依賴關係：所有功能模組開發完成。**

RTX 4090 基於 Ada Lovelace 架構，配備 24GB GDDR6X 顯示記憶體與高達 **72MB 的巨型 L2 快取**（前代 RTX 3090 僅有 6MB）。開發者在設計 PyTorch 的矩陣乘法與張量切片時，必須精確配置批次大小（Batch Size）與記憶體區塊運算（Tiling）策略，目標是將注意力權重矩陣與轉碼器特徵向量盡可能長時間保留在 L2 快取內部，極大化資料的重複利用率，避免頻繁調用 GDDR6X 記憶體匯流排所引發的延遲瓶頸。

VRAM 分配需依循以下硬體承載力矩陣進行設計：

| 目標模型規模 | 推理精度 | VRAM 基礎需求 | 工具運算餘裕 | 可行性評估 |
| :--- | :--- | :---: | :--- | :--- |
| 1.5B 級別（如 DeepSeek-Distill） | FP16 | 約 4–6 GB | 剩餘約 18 GB，完整 LRP 梯度計算圖與轉碼器字典可常駐 VRAM。 | 極佳，毫秒級響應 |
| 2B–3B 級別（如 Gemma-2-2B） | FP16 | 約 8–10 GB | 剩餘約 14 GB，ReIP 前向快取與 WeightLens 靜態特徵可完全在 VRAM 內無縫對接。 | 極佳 |
| 7B–8B 級別（如 Llama-3-8B） | INT8/FP8 混合精度 | 約 10–14 GB | 剩餘約 10 GB，需利用第四代 Tensor 核心的 FP8 硬體支援載入模型。 | 高度可行，數秒至數十秒響應 |
| 9B 級別（如 Gemma-9B） | INT8/FP8 混合精度 | 約 16–18 GB | 剩餘約 6 GB，需實作記憶體分頁（Paging）演算法，將轉碼器改為逐層動態載入。 | 處於邊緣，效能降級 |

此外，開發者需在架構中規劃高安全性的「實體隔離（Air-Gapped）」執行模式，確保所有核心組件能在無網際網路連線的內部伺服器上獨立運作，以符合 GDPR 或 HIPAA 等資料隱私法規。系統亦需規劃非同步背景執行緒的 API 介面，並實作高效的記憶體清理機制（如 `reset_hooks_end=True` 參數或 `clear_contexts`），確保長時間的實時監控不會導致 VRAM 記憶體外洩。

### 階段六：測試驗證（預計工時：1 週）

本階段的目標是執行全面的基準測試，確保系統的保真度與語意品質完全超越傳統分析典範。**依賴關係：所有開發與最佳化工作完成。**

---

## 4. 技術棧與依賴套件清單

本專案的底層引擎依賴於特定的開源庫，以最大化硬體算力的利用率。建議使用 `uv` 進行快速相依性管理。

| 套件 | 版本要求 | 用途說明 |
| :--- | :--- | :--- |
| `transformer-lens` | 最新版（需支援 `add_hook(dir="bwd")`） | 核心框架，提供 HookedTransformer 介面，允許在計算圖任意節點安插探針。 |
| `torch` | ≥ 2.1（支援 FP8 量化） | 張量運算與自動微分基礎，需配置支援 FP8/INT8 混合精度量化的版本。 |
| `dash` | 最新穩定版 | 互動式儀表板框架。 |
| `dash-cytoscape` | 最新穩定版 | 高效能互動式有向無環圖（DAG）視覺化，支援流暢縮放與拖曳。 |
| `spacy` | ≥ 3.x | 本地詞形還原，徹底取代外部 LLM 後處理步驟。 |
| `scikit-learn` | 最新穩定版 | DBSCAN 無監督分群演算法。 |
| `networkx` | 最新穩定版 | 拓撲圖儲存與操作。 |
| `huggingface-hub` | 最新穩定版 | 下載 Hugging Face 上的稀疏轉碼器（Transcoders）權重字典。 |

---

## 5. 各模組的核心實作要點摘要

**ReIP 模組**的關鍵在於精確實作 LRP 傳播規則。LN-rule 必須以客製化的 `autograd.Function` 撰寫，在反向傳播期間將層正規化的縮放因子強制視為常數，以切斷梯度流經變異數計算的複雜路徑，防止相關性崩潰。Identity-rule 則需將非線性函數的導數強制覆寫為 1，確保特徵相關性信號在跨越非線性邊界時嚴格維持守恆特性。整個端到端管線在 $O(2F+B)$（兩次前向傳播加一次反向傳播）的極低計算成本內完成，最終輸出建議以 NetworkX 結構儲存，以利後續管線讀取。

**WeightLens 模組**的核心價值在於其「無資料集」的解析哲學。開發者需專注於提取轉碼器的「輸入不變」連接，這些連接代表了特徵跨層之間的固定幾何關聯。詞彙空間投影的公式為：特徵的編碼器向量 $W_{enc}$ 與嵌入矩陣 $W_{embed}$ 的點積，再透過 Z-分數機制篩選出統計上異常突出的標記。本地詞形還原是確保認識論純淨度的最後一道防線，必須完全在本地 CPU/GPU 上以決定性演算法完成。

**CircuitLens 模組**的設計旨在填補靜態分析在上下文依賴特徵上的缺陷。雅可比矩陣的計算必須與 ReIP 的 LRP 歸因分數深度整合，這種深度整合使得系統能夠動態隔離並遮蔽產生干擾的無關輸入，而無需執行任何額外耗時的激勵探測。DBSCAN 分群需基於 Jaccard 相似度矩陣進行，最終將多義性特徵優雅地拆解為多個單義性子叢集，並為每個子叢集獨立生成精準的描述。

**Dash 儀表板**的設計需確保在處理數千個節點與邊緣時的流暢度。視覺化的直觀性是關鍵：因果分數越高的關鍵傳播路徑，在視覺圖表上必須越粗且顏色越顯著，使研究人員能一眼看穿網路的決策主幹。Callbacks 的實作需賦予研究人員在宏觀拓撲架構與微觀標記特徵之間自由穿梭的強大能力。

---

## 6. 測試與驗證策略

本專案採用兩套互補的驗證框架，確保系統在科學嚴謹性與工程可靠性上均達到最高標準。

**迴路定位保真度驗證**採用間接賓語識別（Indirect Object Identification, IOI）任務作為核心基準。IOI 任務是評估機制可解釋性演算法的黃金標準，旨在測量演算法能否精準定位出模型識別語句中間接賓語所動用的注意力頭與 MLP 結構。測試腳本需記錄傳統 AtP 與 ReIP 在 MLP 輸出層的皮爾森相關係數（PCC），並與耗時數小時的黃金標準窮舉激勵修補進行比較。**嚴格通過標準**為：ReIP 系統必須在相同的 $O(2F+B)$ 極低運算時間內，達到 PCC > 0.95（文獻實測值為 0.9561），而傳統 AtP 在此任務中的 PCC 僅有 0.006。

**語意解析品質驗證**採用 FADE（Feature Alignment to Description Evaluation）評估框架進行自動化且標準化的測試。FADE 是一個可擴展且與模型無關的框架，專門用於評估特徵與其描述之間的對齊程度。測試腳本必須量化下表所列的四大關鍵指標：

| FADE 評估維度 | 定義 | 通過標準 |
| :--- | :--- | :--- |
| 純度（Purity） | 描述是否精確對應單一語意概念，不含混雜。 | 穩定持平或超越 MaxAct 基準。 |
| 清晰度（Clarity） | 描述是否簡潔明確，避免過度泛化。 | 必須穩定超越 MaxAct 基準（傳統方法常產生過度泛化描述）。 |
| 響應度（Responsiveness） | 描述是否能有效預測特徵的激發行為。 | 必須穩定超越 MaxAct 基準。 |
| 保真度（Faithfulness） | 描述是否忠實反映模型的實際計算行為。 | 穩定持平或超越 MaxAct 基準。 |

所有測試流程必須能在本地完全脫機運行，徹底避免任何外部 API 呼叫所產生的延遲與成本。

---

## 7. 建議的開發優先順序與風險評估

### 開發優先順序

建議的開發順序遵循「核心引擎優先、語意解析次之、複雜度解構第三、視覺化與最佳化並行、全面驗證收尾」的原則。具體而言，應優先完成 ReIP 模組（階段一），確保動態因果迴路定位的準確性，因為這是整個系統的數學基礎；接著開發 WeightLens 模組（階段二），實現靜態權重解析；再實作 CircuitLens 模組（階段三），處理深層特徵的上下文依賴；然後同步進行儀表板開發（階段四）與硬體效能最佳化（階段五）；最後執行測試驗證（階段六）。

### 風險評估矩陣

| 風險項目 | 嚴重程度 | 發生機率 | 緩解策略 |
| :--- | :---: | :---: | :--- |
| **VRAM 溢位**：處理 7B–9B 規模模型時極易觸及 24GB 上限。 | 高 | 中 | 嚴格執行 VRAM 分配矩陣，利用 FP8/INT8 量化技術，並在必要時實作記憶體分頁（Paging）演算法。 |
| **LRP 規則實作錯誤**：客製化反向傳播鉤子的編寫極具挑戰性，容易導致梯度計算錯誤，使 PCC 無法達標。 | 高 | 中 | 在開發初期即引入單元測試（`test_lrp_rules.py`），針對每個 LRP 規則進行獨立驗證，確保數學守恆性。 |
| **轉碼器權重可用性**：並非所有目標模型都有對應的 Hugging Face 轉碼器權重可供下載。 | 中 | 中 | 優先支援已有社群轉碼器的主流模型（Gemma-2-2B, Llama-3.2-1B），並在 `transcoder_loader.py` 中實作優雅的降級處理邏輯。 |
| **視覺化效能瓶頸**：Cytoscape 在渲染超大型網路圖（>5000 節點）時可能出現卡頓。 | 中 | 低 | 實作積極的修剪演算法，僅渲染高因果貢獻分數的節點與邊緣，並優化 Dash Callbacks 的響應速度。 |
| **spaCy 詞形還原覆蓋率不足**：對於專業術語或非英語詞彙，詞形還原結果可能不準確。 | 低 | 中 | 建立手動覆寫字典（Override Dictionary），允許使用者在 `configs/` 中自定義詞形還原規則。 |

---

## 參考文獻

[1] Circuit Insights: Towards Interpretability Beyond Activations, OpenReview. https://openreview.net/forum?id=2Jyb1yu3nN

[2] FarnoushRJ/RelP: Official implementation of "RelP: Faithful and Efficient Circuit Discovery in Language Models via Relevance Patching" (NeurIPS 2025 MechInterp Workshop Spotlight), GitHub. https://github.com/FarnoushRJ/RelP

[3] Circuit Insights: Towards Interpretability Beyond Activations, arXiv. https://arxiv.org/html/2510.14936v2

[4] akshathmangudi/weightlens: A unified API for extracting statistics and useful mathematical information for model weights, GitHub. https://github.com/akshathmangudi/weightlens

[5] TransformerLens, GitHub. https://github.com/TransformerLensOrg/TransformerLens

[6] egolimblevskaia/WeightLens, GitHub. https://github.com/egolimblevskaia/WeightLens

[7] FADE: Why Bad Descriptions Happen to Good Features, arXiv. https://arxiv.org/abs/2502.07771
