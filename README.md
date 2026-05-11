# 台灣美食 34 類影像分類 (Taiwan Food 34 — CNN)

以 ImageNet 預訓練 CNN 進行 34 類台灣美食影像分類,涵蓋兩階段微調 (head-only → full fine-tune)、bf16 混合精度訓練,以及測試集上的混淆矩陣與錯誤樣本分析。

主模型為 **ConvNeXt-Tiny**,baseline 對照為 **ResNet-50**。

---

## 1. 專案概要

| 項目 | 內容 |
|---|---|
| 任務 | 34 類台灣美食影像分類 |
| 方法 | Transfer Learning(ImageNet pretrained)+ 兩階段微調 |
| 框架 | PyTorch + timm |
| 訓練平台 | Google Colab Pro (A100) |
| 主模型 | ConvNeXt-Tiny (~29M params) |
| Baseline | ResNet-50 (~25M params) |
| 混合精度 | bfloat16 autocast |
| 評估 | Top-1 / Top-5 accuracy、macro-F1、per-class metrics、confusion matrix、誤判樣本可視化 |

---

## 2. 資料集

每張圖片屬於 34 種台灣美食之一(`bawan`、`beef_noodles`、`bubble_tea`、`xiaolongbao`、...)。

| Split | Images |
|---|---|
| Train | 5,568 |
| Val   | 686 |
| Test  | 719 |
| **Total** | **6,973** |

- 已 stratified 切分 80 / 10 / 10,每類在各 split 都有出現
- 訓練集每類平均約 164 張(範圍 129–286,輕度不平衡 ~2.2x)
- 圖片解析度差異大(寬 150–2992,長寬比 0.46–2.31),訓練時統一 resize 至 224×224
- 格式以 `.jpg` 為主,少量 `.png` / `.jpeg`

資料夾結構(zip 解壓後):
```
data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
├── test/<class_name>/*.jpg
├── class_mapping.csv      # label index ↔ class_name
├── dataset_split.csv      # 每張圖的 split 對應
└── dataset_summary.csv    # 各 split 各類張數
```

> ⚠️ 資料本身不放入 Git。請另外 zip 上傳到 Colab(見下方使用方式)。

---

## 3. 模型與訓練設計

### 影像處理
- **訓練 augmentation**: `RandomResizedCrop(224, scale=0.6–1.0)` → `HFlip` → `RandomRotation(±15°)` → `ColorJitter(0.2)` → `RandAugment(n=2, m=9)` → `Normalize(ImageNet mean/std)` → `RandomErasing(p=0.25)`
- **驗證 / 測試**: `Resize(256)` → `CenterCrop(224)` → `Normalize`
- **TTA**: 測試時做 horizontal flip,logits 平均

### Loss / Optimizer
- `CrossEntropyLoss` with **label smoothing 0.1**
- **AdamW**,weight decay = `0.05` (ConvNeXt) / `1e-4` (ResNet)
- **Cosine annealing LR + 3-epoch linear warmup**
- Gradient clipping(norm = 1.0)

### 兩階段微調 (Two-stage fine-tuning)

| 階段 | Epochs | Backbone | LR (head) | LR (backbone) |
|---|---|---|---|---|
| Stage 1 (head warm-up) | 3 | frozen | 1e-3 | – |
| Stage 2 (full fine-tune) | 25 | unfrozen | 1e-4 | 1e-5 |

- Batch size = **64**
- **bf16 autocast**(A100 原生支援)
- **Early stopping**: val top-1 patience = 8
- Norm layers 與 biases 排除 weight decay(標準做法)

### 評估與錯誤分析
每個模型訓練後自動產出:
- `test_metrics.json` — top-1 / top-5 / macro-F1 / weighted-F1
- `per_class_metrics.csv` — 34 類 precision / recall / F1 / support
- `confusion_matrix.png` — row-normalized 熱圖
- `top_confused_pairs.csv` — 最常被混淆的類別對 (top-10)
- `misclassified_samples.png` — F1 最低的 5 個類別,各列 8 張誤判圖
- `train_curves.png` — loss / accuracy 曲線
- `predictions.npz` — 原始 softmax 機率 + 標籤(供進一步分析)

---

## 4. 結果

| Model | Params | Test Top-1 | Test Top-5 | Macro-F1 | Weighted-F1 | Best Val Top-1 | Best Epoch |
|---|---|---|---|---|---|---|---|
| **ConvNeXt-Tiny** | 27.8M | **96.24** | **99.30** | **95.84** | **96.08** | 96.50 | 20 |
| ResNet-50         | 23.6M | 79.42 | 97.08 | 78.82 | 79.15 | 80.32 | 23 |

訓練在 Colab A100 上、每模型總共 28 epochs(3 stage-1 + 25 stage-2),ConvNeXt 觸發 early stopping。

### ConvNeXt-Tiny 最差 5 類 (by F1)
| 類別 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| spicy_duck_blood          | 0.824 | 0.824 | 0.824 | 17 |
| luwei                      | 0.882 | 0.833 | 0.857 | 18 |
| three-cup_chicken          | 0.889 | 0.941 | 0.914 | 17 |
| kung-pao_chicken           | 0.944 | 0.895 | 0.919 | 19 |
| grilled_taiwanese_sausage  | 1.000 | 0.857 | 0.923 | 21 |

### ConvNeXt-Tiny Top-5 混淆對
| Class A | Class B | A→B | B→A | Total |
|---|---|---|---|---|
| kung-pao_chicken | three-cup_chicken | 2 | 1 | 3 |
| luwei            | spicy_duck_blood  | 2 | 1 | 3 |
| grilled_taiwanese_sausage | taiwanese_sausage_in_rice_bun | 2 | 0 | 2 |
| spicy_duck_blood | stinky_tofu       | 1 | 1 | 2 |
| bawan            | tangyuan          | 1 | 0 | 1 |

混淆都集中在視覺上真的相似的料理(兩種雞 / 兩種香腸 / 滷味與鴨血等),屬於合理的 fine-grained 殘餘錯誤。

### 為什麼 ConvNeXt 比 ResNet 高 17 點?

兩條曲線對照(`outputs/{model}/train_curves.png`):
- ConvNeXt 的 frozen backbone + head-only 訓練 3 epochs 就到 **93% val**,顯示 ImageNet-1k 上學到的特徵直接對台灣美食非常具區分力
- ResNet-50 的 head-only 階段只到 **64% val**,full fine-tune 結束時 train acc 仍低於 val(被 RandAugment 拉太用力),且 cosine LR 已歸零,屬於「還能再爬但被排程提前剎車」的狀態
- 這證明主要差距來自 backbone 本身的預訓練特徵品質,不是訓練配方的問題

完整混淆矩陣與誤判樣本圖見 `outputs/{model}/` 底下。

---

## 5. 如何使用

### 方法 A — Colab(推薦)
1. 把 `notebooks/colab_run.ipynb` 開到 Colab
2. 編輯第 1 個程式碼 cell 中的 `GITHUB_URL` 為你的 repo
3. 把整個 `台灣美食34/` 資料夾壓成 `taiwan_food34.zip`(zip 內保留 `train/`、`val/`、`test/` 與 csv)
4. **Runtime → Run all**,中間會跳出檔案上傳對話框上傳 zip
5. 跑完會自動下載 `outputs.zip`(內含所有 metrics、混淆矩陣、誤判圖)

預估 A100 上總耗時 **~25–35 分鐘**。

### 方法 B — 本機 / 自架 GPU
```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 把資料放到 ./data,結構應為 data/train, data/val, data/test, data/class_mapping.csv

# 3. 訓練主模型
python -m scripts.run_train --config configs/convnext_tiny.yaml

# 4. 訓練 baseline
python -m scripts.run_train --config configs/resnet50.yaml

# 5. 整理比較表
python -m scripts.compare_models --runs outputs/convnext_tiny outputs/resnet50

# (可選)重跑分析、不重訓
python -m scripts.run_analysis --config configs/convnext_tiny.yaml
```

修改超參數請編輯對應的 `configs/*.yaml`,不需要改程式碼。

---

## 6. 專案結構

```
taiwan-food-cnn/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── convnext_tiny.yaml      # 主模型超參數
│   └── resnet50.yaml           # baseline 超參數
├── src/
│   ├── data.py                  # ImageFolder + transforms + dataloaders
│   ├── models.py                # timm 工廠 + freeze / param-group helpers
│   ├── engine.py                # 單 epoch 訓練 / 評估 (bf16 autocast)
│   ├── train.py                 # 兩階段微調 + early stopping + 曲線
│   ├── evaluate.py              # 測試集推論 + TTA + 指標
│   ├── analysis.py              # 混淆矩陣 / per-class / 誤判可視化
│   └── utils.py                 # seed, logger, AverageMeter, checkpoint
├── scripts/
│   ├── run_train.py             # 訓練 + 評估 + 分析(主進入點)
│   ├── run_analysis.py          # 只跑分析(checkpoint 必須已存在)
│   └── compare_models.py        # 多模型結果整合
└── notebooks/
    └── colab_run.ipynb          # Colab 一鍵跑完整流程
```

---

## 7. 輸出說明

每個訓練會在 `outputs/<model_name>/` 產生:

```
outputs/convnext_tiny/
├── best.pth                    # 最佳 val top-1 的權重(不入 Git)
├── train_log.csv               # epoch / loss / acc / lr
├── train_curves.png            # loss + accuracy 曲線
├── train_summary.json          # 最佳 epoch、val top-1、總 epoch 數
├── test_metrics.json           # 測試集 top-1 / top-5 / macro-F1 ...
├── test_metrics.csv            # 同上,CSV 版本
├── per_class_metrics.csv       # 34 類 precision / recall / F1
├── confusion_matrix.png        # row-normalized 熱圖
├── confusion_matrix_counts.png # counts 熱圖
├── top_confused_pairs.csv      # 最常被混淆的類別對 top-10
├── misclassified_samples.png   # F1 最低 5 類的誤判樣本
├── predictions.npz             # 測試集 softmax 機率 + 真實/預測 label
├── train.log / test.log
└── summary.json                # 整合 config + test metrics
```

---

## 8. License

MIT(資料集另行依其原始授權使用)
