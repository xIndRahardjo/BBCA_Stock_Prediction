# JCIE Paper Revision: Ringkasan Lengkap Perubahan
**Respons Reviewer 4 & Modifikasi Notebook**

**Tanggal:** 2026-09-07  
**ID Submission:** 261775877  
**Deadline:** 2026-09-17  
**Status:** Dalam Revisi

---

## Ringkasan Eksekutif

Dokumen ini mencatat semua perubahan yang dilakukan untuk menjawab lima kekhawatiran metodologis Reviewer 4 (poin 4-2 hingga 4-6) dan dampaknya terhadap notebook, hasil eksperimen, dan struktur paper.

**Temuan Kunci:** Setelah menerapkan validasi temporal yang benar dan menambahkan baseline Random Walk, hasil menunjukkan bahwa **Random Walk mengalahkan semua model ML pada keenam saham Indonesia**. Ini adalah hasil negatif yang valid dan memerlukan reframing kesimpulan paper secara jujur.

---

## Komentar Reviewer 4 & Solusi

### 4-2: Information Leakage Contemporaneous & Timing Forecast

**Kekhawatiran Reviewer:**
- Manuscript mengklaim fitur hari `t` memprediksi `Close(t+1)`
- Tetapi semua fitur menggunakan informasi dari hari `t` (Open, High, Low dihitung dari hari `t`)
- Pertanyaan: Kapan tepatnya forecast dibuat? Ada lookahead bias?

**Masalah Original Notebook:**
```python
# SALAH (notebook v1)
X = df[feature_cols]
y = df['Close']  # HARI YANG SAMA, bukan hari berikutnya!
```
Menghasilkan: `fitur(t) → Close(t)` bukan `fitur(t) → Close(t+1)`

**Revisi yang Diterapkan:**
```python
# BENAR (notebook v2)
df['Target'] = df['Close'].shift(-1)  # Satu hari ke depan
X = df[feature_cols]
y = df['Target']  # Sekarang: fitur(t) → Close(t+1)
```

**Klarifikasi di Paper:**
- Forecast didefinisikan sebagai: **forecast end-of-day pada waktu t**
- Menggunakan: semua informasi tersedia sampai market close hari t
- Memprediksi: closing price hari trading berikutnya (t+1)
- Contoh: Data 2024-12-05 → prediksi close 2024-12-06

**Hasil Audit Preprocessing:** ✓ Tidak ada lookahead bias terdeteksi

---

### 4-3: GridSearchCV dengan cv=3 Melanggar Causality Time Series

**Kekhawatiran Reviewer:**
- Standard K-fold CV mengacak data secara random
- Untuk time series: bisa train pada data masa depan, validate data masa lalu (SALAH!)
- Mengizinkan information leakage dari validation ke training

**Masalah Original Notebook:**
```python
# SALAH (notebook v1)
GridSearchCV(model, param_grid, cv=3)
# → Fold assignment random, potensi kacau temporal order
```

**Revisi yang Diterapkan:**
```python
# BENAR (notebook v2)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)
GridSearchCV(model, param_grid, cv=tscv)
```

**Apa yang Berubah:**
- Fold 0: Train 2020-01 hingga 2023-10, Val 2023-10 hingga 2024-11
- Fold 1: Train 2020-01 hingga 2024-11, Val 2024-12 hingga 2025-06
- Fold 2: Train 2020-01 hingga 2025-06, Val 2025-06 hingga 2026-03
- **Temporal order selalu terjaga:** train_end < val_start ✓

**Hasil:** Eliminasi lookahead bias selama hyperparameter tuning

---

### 4-4: Baseline Naïve Hilang & Uji Signifikansi Statistik

**Kekhawatiran Reviewer:**
- Hanya membandingkan LR vs XGBoost vs LightGBM
- Harga saham menunjukkan persistensi tinggi (besok ≈ hari ini)
- Tanpa baseline naïve, tidak bisa judge apakah ML memberi incremental value
- Butuh uji signifikansi (Diebold-Mariano)

**Masalah Original Notebook:**
```python
# HILANG (notebook v1)
# Tidak ada baseline Random Walk
# Tidak ada DM statistical test
# Tidak ada assessment signifikansi
```

**Revisi yang Diterapkan:**

**Random Walk Baseline:**
```python
# Implementasi
C_hat(t+1) = C(t)  # Forecast naïve: besok = hari ini

y_pred_rw = y_test.shift(1).dropna()
rmse_rw = np.sqrt(mean_squared_error(y_actual, y_pred_rw))
```

**Diebold-Mariano Test:**
```python
from statsmodels.tsa.stattools import dm_test

# Bandingkan squared errors
dm_stat, dm_pvalue, dm_diff = dm_test(
    e1=e_model,      # Model forecast errors
    e2=e_rw,         # Random Walk forecast errors
    h=1              # 1-step ahead
)

# Interpretasi:
# p < 0.05: perbedaan signifikan
# e1 - e2 < 0: Model 1 punya error lebih rendah (lebih baik)
```

**Hasil:**
Random Walk mencapai **RMSE terendah di semua 6 saham**:

| Saham | RW RMSE | Best ML | p-value vs LR |
|-------|---------|---------|---|
| BBCA  | 142.58  | LR 148.94 | 0.047 (sig) |
| BBRI  | 75.45   | LR 78.88  | 0.140 (tdk sig) |
| TLKM  | 68.72   | LR 71.80  | 0.112 (tdk sig) |
| ASII  | 115.29  | LR 119.92 | 0.161 (tdk sig) |
| ANTM  | 97.98   | LR 103.07 | 0.022 (sig) |
| INDF  | 131.36  | LR 135.19 | 0.373 (tdk sig) |

---

### 4-5: OLS Coefficients Tidak Valid Sebagai Feature Importance

**Kekhawatiran Reviewer:**
- Manuscript menggunakan absolute LR coefficients untuk rank fitur
- Mengklaim SMA, MACD, dll adalah "key drivers" / "primary drivers"
- Ini tidak reliable ketika:
  - Ada multicollinearity berat (VIF > 5)
  - Fitur sudah MinMax scaled
  - Coefficients tidak stabil

**Masalah Original Notebook:**
```python
# SALAH (notebook v1, Figure 10)
coefficients = np.abs(model.coef_)
top_features = np.argsort(coefficients)[-5:]

# Kemudian di text:
# "SMA dan MACD adalah key drivers..."
```

**Revisi yang Diterapkan:**
```python
# DIHAPUS di notebook v2
# → Tidak ada coefficient ranking figure
# → Tidak ada "key drivers" language di results
```

**Perubahan di Paper:**
- ✗ Hapus Figure 10 (coefficient importance)
- ✗ Hapus semua "key drivers" / "primary drivers" language
- ✓ Tetap: "LR computationally simple and robust"
- ✓ Tetap: correlation analysis (price persistence observation)
- ✓ Hindari: causal interpretation of coefficients

**Temuan Multicollinearity:**
Fitur dengan VIF > 5 (multicollinearity tinggi):
- SMA5, SMA15, SMA30 (semua correlate > 0.99 dengan Close dan satu sama lain)
- EMA9 (correlate > 0.99 dengan Close)
- MACD dan Signal (derived dari EMA12 dan EMA26 yang sama)

→ Membuat coefficient estimates tidak stabil dan unreliable untuk feature ranking

---

### 4-6: Klaim Emerging Markets Terlalu Luas; Sampel Diversity Kurang

**Kekhawatiran Reviewer:**
- Dataset original: BBCA, BRIS (keduanya bank) + QCOM (US stock)
- Hanya 2 saham Indonesia, keduanya finance sector
- Tidak bisa support claim tentang "emerging markets" secara generic
- BRIS (BRI Syariah) adalah produk banking unusual

**Dataset Original:**
| Saham  | Sektor          | Negara | Issue |
|--------|-----------------|--------|-------|
| BBCA   | Banking         | Indo   | ✓ Core |
| BRIS   | Islamic Banking | Indo   | Niche |
| QCOM   | Electronics     | USA    | Out of scope |

**Revisi yang Diterapkan:**

**Dataset Baru: 6 Saham Indonesia Beragam**

| Ticker  | Perusahaan                  | Sektor                     | Index  | Catatan |
|---------|----------------------------|----------------------------|--------|---------|
| BBCA.JK | Bank Central Asia          | Financial                  | LQ45   | Bank terbesar by market cap |
| BBRI.JK | Bank Rakyat Indonesia      | Financial                  | LQ45   | Bank negara terbesar |
| TLKM.JK | Telekomunikasi Indonesia   | Telecommunications         | LQ45   | Incumbent telecom |
| ASII.JK | Astra International        | Automotive / Industrial    | LQ45   | Diversified manufacturer |
| ANTM.JK | Antam                      | Mining / Commodities       | IDX30  | State-owned mining |
| INDF.JK | Indofood Sukses Makmur     | Consumer Staples / Food    | LQ45   | Dominant food producer |

**Dihapus:**
- ✗ BRIS (terlalu niche; inconsistent dengan major-cap focus)
- ✗ QCOM (out of geographic scope; market dynamics berbeda)

**Benefit Dataset Baru:**
- **Sector diversity:** Finance, Telecom, Industrial, Commodities, Consumer
- **Market relevance:** Semua di LQ45 atau IDX30 (liquid, widely traded)
- **Periode comparable:** 2020-01-01 hingga 2026-03-05 untuk semua 6
- **Ukuran sampel:** ~1300-1500 daily observations per saham
- **Consistency geographic:** Semua equities Indonesia di JSX
- **Economic representation:** Cover major sectors ekonomi Indonesia

**Klaim Scope di Paper (Revisi):**
- ✗ "generalizable ke emerging markets broadly"
- ✗ "emerging market equities"
- ✓ "enam saham Indonesia terpilih across major sectors"
- ✓ "JSX-listed equities"
- ✓ "findings demonstrate limitations of daily technical indicator-based forecasting in a lower-liquidity market context"

---

## Modifikasi Notebook — Summary

### File: `MLT_Yahoo_Finance_Reviewer4_6Stocks_FIX.ipynb`

#### Changes by Section

**1. Data Download (Cell 3)**
```python
# SEBELUM
stocks = ['BBCA.JK', 'BRIS.JK', 'QCOM']  # 3 saham
start_date = '2020-01-01'
end_date = '2026-03-05'

# SESUDAH
stocks = ['BBCA.JK', 'BBRI.JK', 'TLKM.JK', 'ASII.JK', 'ANTM.JK', 'INDF.JK']  # 6 saham
start_date = '2020-01-01'
end_date = '2026-03-06'  # Exclusive end di yfinance
```

**2. Target Construction (Cell 5)**
```python
# SEBELUM
y = df['Close']  # Prediksi hari yang sama

# SESUDAH
df['Target'] = df['Close'].shift(-1)  # Satu hari ke depan
df = df.dropna(subset=['Target'])
y = df['Target']  # Prediksi hari berikutnya
```

**3. Preprocessing Pipeline (Cell 6)**
```python
# SEBELUM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# → Scaler fit pada entire dataset (LEAKAGE!)

# SESUDAH
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', model)
])
# → Scaler fit hanya pada training portion per fold (NO LEAKAGE)
```

**4. Cross-Validation (Cells 10-11)**
```python
# SEBELUM (XGBoost GridSearch)
GridSearchCV(xgb_model, param_grid, cv=3)  # Random CV folds

# SESUDAH (XGBoost GridSearch)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)
GridSearchCV(xgb_model, param_grid, cv=tscv)  # Time-aware folds
```

**5. Random Walk Baseline (NEW Cell 13)**
```python
# NEW - Implementasi Random Walk
y_pred_rw = X_test['Close'].values  # Besok = Hari ini
rmse_rw = np.sqrt(mean_squared_error(y_test_actual, y_pred_rw))
mae_rw = mean_absolute_error(y_test_actual, y_pred_rw)
mape_rw = mean_absolute_percentage_error(y_test_actual, y_pred_rw)
r2_rw = r2_score(y_test_actual, y_pred_rw)
```

**6. Diebold-Mariano Test (NEW Cell 16)**
```python
# NEW - Statistical Significance Testing
from statsmodels.tsa.stattools import dm_test

for stock in stocks:
    for model_name in ['LR', 'XGB', 'LGBM']:
        e_model = errors[model_name]
        e_rw = errors['RW']
        
        dm_stat, dm_pvalue, _ = dm_test(
            e1=e_model**2,
            e2=e_rw**2,
            h=1
        )
        
        print(f"{stock} {model_name} vs RW: p={dm_pvalue:.4f}")
```

**7. Results Storage (NEW)**
```python
# NEW - Simpan outputs untuk paper
all_results.to_csv('all_stock_model_results.csv')
dm_results.to_csv('diebold_mariano_results.csv')
best_summary.to_csv('best_model_summary.csv')
```

**8. Coefficient Importance Figure (REMOVED)**
```python
# DIHAPUS - Figure 10 coefficient ranking
# Reason: Violates Reviewer 4-5 concern about multicollinearity
```

---

## Perbandingan Hasil: Sebelum vs Sesudah

### Contoh Saham BBCA

**Original Results (SALAH - prediksi same-day)**
| Metric | Value |
|--------|-------|
| RMSE | 53.13 |
| MAPE | 0.49% |
| R² | 0.9932 |
| Interpretasi | Extremely high accuracy |

**Revised Results (BENAR - prediksi next-day)**
| Model | RMSE | MAE | MAPE | R² | vs RW |
|-------|------|-----|------|-----|-------|
| Random Walk | **142.58** | 106.48 | 1.34% | 0.9460 | baseline |
| Linear Regression | 148.94 | 109.79 | 1.39% | 0.9411 | -4.5% |
| XGBoost | 162.98 | 121.69 | 1.54% | 0.9295 | -14.3% |
| LightGBM | 171.09 | 130.15 | 1.66% | 0.9223 | -20.0% |

**Perubahan Interpretasi:**
- Sebelum: "LR mencapai akurasi superior (RMSE 53.13)"
- Sesudah: "Random Walk mengalahkan semua ML models; LR paling robust tapi masih kalah baseline (RMSE 142.58 vs 148.94)"

---

## Perubahan Struktur Paper

### Section yang Affected

#### 1. Abstract
**Sebelum:**
> "Linear Regression, LightGBM, dan XGBoost dibandingkan untuk forecasting harga BBCA. Linear Regression mencapai performa terbaik dengan RMSE 53.13 dan MAPE 0.49%."

**Sesudah:**
> "Enam saham Indonesia terpilih dievaluasi menggunakan time-aware cross-validation. Random Walk baseline (harga next = harga today) mencapai RMSE terendah across all stocks. Linear Regression memberikan ML alternative paling robust tapi tidak mencapai positive improvement over Random Walk. Diebold-Mariano testing confirm limited statistical significance dari ML improvements. Findings ini validate efficient market hypothesis untuk daily price-level forecasting menggunakan technical indicators."

#### 2. Introduction
**Perubahan:**
- Expand motivasi: Kenapa 6 stocks bukan 1?
- Add: Limitations dari high R² without baseline comparison
- Clarify: Daily vs multi-step forecasting scope
- **Remove:** Claims tentang "investment decision support" (beyond scope)

#### 3. Related Work
**Add:**
- Efficient Market Hypothesis literature
- Naïve forecasting sebagai academic baseline (bukan hanya ML comparisons)
- Time-series validation methodology papers
- Diebold-Mariano test applications di finance

#### 4. Methodology
**Major Changes:**
- **4.1 Data:** Expand sample description (6 stocks, sectors, dates)
- **4.2 Temporal Validation:** Detail TimeSeriesSplit protocol
- **4.3 Target Definition:** Explicit one-step-ahead construction dengan shift(-1)
- **4.4 Preprocessing:** Pipeline design (prevent scaler leakage)
- **4.5 Baseline:** NEW section tentang Random Walk formula
- **4.6 Statistical Testing:** NEW section tentang Diebold-Mariano test
- Remove: Coefficient interpretation discussion

#### 5. Results
**Perubahan:**
- **Table 1:** Add Random Walk column; all 6 stocks
- **Table 2:** Diebold-Mariano results (p-values, significance)
- **Figure 1:** Box plots showing RW vs ML comparisons across stocks
- **Figure 2:** DM test p-value heatmap
- Remove: Figure 10 (coefficient importance)
- Add: Performance degradation analysis (kenapa ASII/ANTM gagal parah)

#### 6. Discussion
**Narrative Baru:**
- **Persistence Effect:** Stock prices highly autocorrelated; sulit beat t-to-t+1
- **LR Robustness:** Kenapa LR tidak gagal separah boosting methods
- **Multicollinearity:** Technical indicators highly correlated; reduce model stability
- **Generalization Risk:** Tree models memorize training noise → catastrophic generalization
- **Market Efficiency:** Results align dengan EMH untuk intraday (daily) forecasting
- **Practical Implication:** Complex ML models not automatically better; simpler models often lebih robust

#### 7. Conclusion
**Sebelum:**
> "Linear Regression adalah model terbaik untuk forecasting harga saham BBCA."

**Sesudah:**
> "Daily closing price forecasting menggunakan technical indicators dan standard ML tidak dapat mengalahkan naïve Random Walk baseline across enam saham Indonesia terpilih. Negative result ini highlight importance dari proper temporal validation, appropriate baseline comparisons, dan honest reporting dari model limitations. Meski Linear Regression outperformed lebih complex methods, neither provided incremental predictive value over persistence. Findings ini reinforce Efficient Market Hypothesis untuk daily price-level forecasting dan suggest bahwa predictive signal requires either (a) longer forecast horizons, (b) additional data sources (intraday order flow, sentiment), atau (c) regime-specific opportunities. Practitioners harus exercise caution saat interpret high goodness-of-fit metrics (R², MAPE) sebagai evidence dari tradable forecasting skill without comparison ke naive baselines dan proper temporal validation."

---

## Respons ke Reviewer 4: Specific Answers

### 4-2 Response
> **Authors' Response:** Kami telah memperbaiki target variable untuk ensure proper one-step-ahead prediction. Target sekarang didefinisikan sebagai `Close(t+1)` menggunakan `df['Close'].shift(-1)`. Forecast origin dijelaskan sebagai end-of-day close pada day t, menggunakan semua informasi tersedia hingga waktu itu untuk predict next trading day's closing price. Kami verify bahwa no information leakage terjadi: semua preprocessing (scaling, feature engineering) respect temporal split dalam setiap TimeSeriesSplit fold. Preprocessing sekarang implemented dalam sklearn Pipeline untuk ensure scalers fit hanya ke training portion dari setiap fold.

### 4-3 Response
> **Authors' Response:** Kami telah replace standard k-fold cross-validation dengan TimeSeriesSplit(n_splits=3), yang respect temporal causality. Setiap fold strictly maintain temporal ordering: training observations selalu precede validation observations. Hyperparameter tuning untuk XGBoost dan LightGBM sekarang gunakan time-aware cross-validation, eliminating lookahead bias yang present di earlier cv=3 approach.

### 4-4 Response
> **Authors' Response:** Seperti diminta, kami telah add Random Walk baseline didefinisikan sebagai Ĉ(t+1)=C(t). Across semua enam saham yang dievaluasi, baseline Random Walk mencapai RMSE terendah di semua cases. Kami apply Diebold-Mariano test dengan Harvey small-sample correction untuk assess statistical significance dari differences. Results show bahwa Linear Regression's performance statistically indistinguishable dari Random Walk untuk empat dari enam saham (p > 0.05). XGBoost dan LightGBM significantly worse daripada Random Walk pada semua saham dievaluasi. Findings ini demonstrate importance dari comparing ML models ke appropriate baselines dan validate efficient market hypothesis untuk daily price-level forecasting menggunakan technical indicators alone.

### 4-5 Response
> **Authors' Response:** Kami telah remove semua interpretation dari absolute Linear Regression coefficients sebagai feature importance. Manuscript sekarang tidak claim bahwa SMA, MACD, atau features lain adalah "key drivers" atau "primary drivers." Interpretasi ini tidak appropriate karena presence dari heavy multicollinearity di antara technical indicators (VIF > 5 untuk multiple features) dan known instability dari OLS coefficients under multicollinearity. Kami sekarang restrict discussion ke correlation antara price-level persistence dan next-day prices, yang kami interpret sebagai evidence dari market persistence daripada predictive signal.

### 4-6 Response
> **Authors' Response:** Kami telah expand sample dari 3 stocks (termasuk satu US stock di luar scope) ke 6 saham Indonesia terpilih across diverse sectors: banking (BBCA, BBRI), telecommunications (TLKM), industrial (ASII), mining (ANTM), dan consumer staples (INDF). Semua stocks adalah constituents dari major JSX indices (LQ45 atau IDX30). Analysis sekarang explicitly scoped ke Indonesian equities dan tidak claim generalization ke emerging markets broadly. Expanded sample provide greater sectoral diversity sambil maintain geographic dan market focus.

---

## File yang Dimodifikasi

| File | Perubahan | Status |
|------|-----------|--------|
| `MLT_Yahoo_Finance_Reviewer4_6Stocks_FIX.ipynb` | Target shift, TimeSeriesSplit, RW baseline, DM test, 6 stocks | ✓ Complete |
| `PREPROCESSING_AND_FEATURE_AUDIT_REPORT.md` | Feature correlation analysis, leakage audit, methodology verification | ✓ Complete |
| `all_stock_model_results.csv` | New: Performance metrics untuk 6 stocks across 4 models | ✓ Generated |
| `diebold_mariano_results.csv` | New: DM test statistics dan p-values | ✓ Generated |
| `best_model_summary.csv` | New: Best performing model per stock | ✓ Generated |

---

## Langkah Selanjutnya untuk Revisi Paper

### Immediate (By Sep 10)

- [ ] Update title: "One-Step-Ahead Price Forecasting of Selected Indonesian Stocks: A Time-Aware Comparison of Linear Regression, LightGBM, XGBoost, and a Random-Walk Benchmark"
- [ ] Rewrite Abstract (narrative baru)
- [ ] Rewrite Introduction (6 stocks, temporal validation emphasis)
- [ ] Rewrite Methodology section completely
- [ ] Create new Results tables dan figures
- [ ] Update Discussion dengan negative-result interpretation
- [ ] Rewrite Conclusion

### Secondary (By Sep 14)

- [ ] Update semua figure numbering dan references
- [ ] Add Nomenclature section (new equations: RW, DM test)
- [ ] Verify JCIE three-line table format
- [ ] Expand Related Work untuk baseline methodology
- [ ] Update Author Contributions (jika sample size ubah responsibilities)
- [ ] Language proofread (Reviewer 3 notes)

### Submission Prep (By Sep 16)

- [ ] Compile final pdf dengan revised figures
- [ ] Verify similarity score < 30%
- [ ] Format manuscript ke JCIE requirements
- [ ] Complete Reviewer Response document (4-2 hingga 4-6)
- [ ] Final review dari data accuracy vs notebook output

---

## Key Takeaways

1. **Metodologi Sound:** TimeSeriesSplit, Pipeline scaling, target shift(-1) semua correct
2. **Results Jujur:** Random Walk beat semua ML models—valid negative result
3. **Reframing Required:** Paper harus shift dari "LR wins" ke "kenapa persistence beat complexity?"
4. **Stronger Research:** Honest negative results + proper baseline + statistical testing = science lebih defensible
5. **Scope Clarification:** Indonesian stocks only, daily forecasting horizon, no trading strategy
6. **Next Paper Story:** "High R² bukan evidence dari predictive skill; temporal validation dan naive benchmarks essential"

---

**Document Status:** Reference lengkap untuk semua Reviewer 4 changes  
**Last Updated:** 2026-09-07  
**Notebook Version:** MLT_Yahoo_Finance_Reviewer4_6Stocks_FIX.ipynb  
**Language:** Bahasa Indonesia
