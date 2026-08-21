# Apr–May 2026 Weekly-Peak Stock Wave

## Definition

- History: `2025-08-01` → `2026-08-20`.
- Weekly point: last available adjusted close in each ISO week.
- Peak window: `2026-04-01` → `2026-05-31`.
- Universe: currently active U.S. common stocks (Massive type CS); first history must exist on or before 2025-08-31 and weekly coverage ≥ 80%.
- Pullback grouping: 1D k-means fit on all matched stocks; k=3..5 chosen by silhouette minus small complexity penalty; p95 winsorization for fit only.
- January reference: median weekly adjusted close during January 2026.

## Funnel

- `active_common_stocks`: 5,320
- `with_any_history`: 5,319
- `with_august_2025_history`: 4,679
- `coverage_pass`: 4,652
- `apr_may_global_peak`: 463
- **Matched class:** 463
- **Liquid matched class:** 197

## Data-driven pullback breaks

Selected k = **3**, silhouette = **0.587**.
Cluster centers: **9.0%, 28.4%, 52.7%**.
Boundaries: **18.7%, 40.6%**.

| Group | Drawdown range | Stocks | Liquid | Median drawdown | Median vs Jan-26 | Median Jan→peak rally retraced |
|---|---:|---:|---:|---:|---:|---:|
| G1 | ≤ 18.7% | 197 | 86 | 9.5% | 12.0% | 41.3% |
| G2 | > 18.7% – ≤ 40.6% | 169 | 80 | 28.4% | 4.0% | 90.6% |
| G3 | > 40.6% | 97 | 31 | 53.0% | -11.9% | 116.6% |

## Group members — liquid view

### G1

| Ticker | Name | Peak week | Peak→now | Now vs Jan | Jan→peak rally retraced | 20d $ volume |
|---|---|---|---:|---:|---:|---:|
| **NVDA** | Nvidia Corp | 2026-05-15 | -3.8% | +15.6% | 22.5% | $24805.3M |
| **GOOGL** | Alphabet Inc. Class A Common Stock | 2026-05-08 | -15.0% | +3.7% | 83.2% | $9505.4M |
| **AVGO** | Broadcom Inc. Common Stock | 2026-05-29 | -18.5% | +5.5% | 81.3% | $7768.9M |
| **GOOG** | Alphabet Inc. Class C Capital Stock | 2026-05-08 | -14.8% | +2.8% | 86.7% | $6361.4M |
| **LITE** | Lumentum Holdings Inc. Common Stock | 2026-05-15 | -9.4% | +150.2% | 14.8% | $4365.5M |
| **GEV** | GE Vernova Inc. | 2026-04-24 | -15.9% | +42.1% | 39.0% | $2329.2M |
| **COST** | Costco Wholesale Corp | 2026-05-15 | -11.0% | -0.7% | 106.2% | $1742.7M |
| **FERG** | Ferguson Enterprises Inc. | 2026-05-01 | -9.4% | -3.8% | 162.6% | $1120.2M |
| **PFE** | Pfizer Inc. | 2026-04-02 | -1.9% | +8.3% | 19.9% | $1063.4M |
| **NEE** | NextEra Energy, Inc. | 2026-05-01 | -12.3% | +1.7% | 89.5% | $971.5M |
| **PWR** | Quanta Services, Inc. | 2026-05-15 | -14.0% | +41.9% | 35.5% | $807.6M |
| **FIX** | Comfort Systems USA, Inc. | 2026-05-15 | -16.1% | +49.4% | 36.6% | $804.0M |
| **MO** | Altria Group, Inc. | 2026-05-01 | -10.2% | +8.4% | 59.5% | $700.0M |
| **EQIX** | Equinix, Inc. Common Stock REIT | 2026-04-24 | -2.4% | +35.3% | 8.5% | $660.1M |
| **SLB** | SLB Limited | 2026-05-22 | -6.5% | +14.6% | 35.4% | $579.0M |
| **TT** | Trane Technologies plc | 2026-05-01 | -7.3% | +15.8% | 36.5% | $542.4M |
| **DLR** | Digital Realty Trust, Inc. | 2026-04-17 | -4.6% | +22.1% | 21.0% | $504.8M |
| **BKR** | Baker Hughes Company | 2026-05-01 | -9.2% | +21.3% | 36.5% | $494.3M |
| **WMB** | Williams Companies Inc. | 2026-05-22 | -8.7% | +16.5% | 40.1% | $461.3M |
| **MSCI** | MSCI, Inc. | 2026-05-29 | -9.9% | -3.6% | 150.5% | $403.5M |
| **SWKS** | Skyworks Solutions Inc | 2026-05-22 | -17.6% | +15.2% | 61.7% | $373.4M |
| **HAL** | Halliburton Company | 2026-05-15 | -14.5% | +9.4% | 66.5% | $342.0M |
| **SRE** | Sempra | 2026-04-02 | -11.9% | -1.6% | 113.9% | $335.6M |
| **EME** | EMCOR Group, Inc. | 2026-05-08 | -14.6% | +13.4% | 59.0% | $333.0M |
| **PPL** | PPL Corporation | 2026-04-10 | -11.2% | -2.8% | 129.4% | $311.0M |
| **HUBB** | Hubbell Incorporated | 2026-04-24 | -15.1% | -3.2% | 123.3% | $306.7M |
| **ZM** | Zoom Communications, Inc. Class A Common Stock | 2026-05-08 | -2.7% | +24.8% | 12.1% | $297.8M |
| **SU** | Suncor Energy, Inc. | 2026-05-15 | -0.5% | +36.6% | 1.9% | $295.1M |
| **ENB** | Enbridge, Inc | 2026-05-22 | -12.2% | +6.0% | 71.1% | $283.4M |
| **CMS** | CMS Energy Corporation | 2026-04-10 | -12.0% | -1.2% | 110.0% | $283.1M |
| **UTHR** | United Therapeutics Corp | 2026-04-17 | -11.4% | +11.0% | 56.7% | $277.5M |
| **ETR** | Entergy Corporation | 2026-04-10 | -7.8% | +14.4% | 40.4% | $265.5M |
| **ED** | Consolidated Edison, Inc. | 2026-04-02 | -6.2% | +4.3% | 61.3% | $263.8M |
| **RIG** | Transocean LTD. | 2026-05-15 | -14.6% | +41.4% | 36.9% | $253.7M |
| **VRSN** | VeriSign Inc | 2026-05-22 | -10.7% | +11.2% | 54.2% | $241.8M |
| **FE** | FirstEnergy Corp. | 2026-04-10 | -8.3% | +1.3% | 87.3% | $218.4M |
| **TSN** | Tyson Foods, Inc. | 2026-05-08 | -14.6% | -2.8% | 120.1% | $193.0M |
| **ATO** | Atmos Energy Corporation | 2026-04-10 | -10.4% | +2.2% | 84.1% | $182.4M |
| **QSR** | Restaurant Brands International Inc. | 2026-04-24 | -1.7% | +16.4% | 10.7% | $179.7M |
| **BG** | Bunge Global SA | 2026-04-02 | -9.9% | +8.1% | 59.5% | $150.3M |

_+ 46 more liquid members in `stocks.csv`._

### G2

| Ticker | Name | Peak week | Peak→now | Now vs Jan | Jan→peak rally retraced | 20d $ volume |
|---|---|---|---:|---:|---:|---:|
| **IREN** | IREN Limited Ordinary Shares | 2026-05-29 | -33.0% | -20.7% | 213.7% | $1852.2M |
| **QCOM** | Qualcomm Inc | 2026-05-29 | -36.0% | +0.8% | 98.6% | $1813.5M |
| **VRT** | Vertiv Holdings Co Class A Common Stock | 2026-05-15 | -28.7% | +49.6% | 54.8% | $1692.2M |
| **AAOI** | Applied Optoelectronics, Inc. | 2026-05-15 | -32.2% | +248.5% | 40.0% | $1621.4M |
| **CLS** | Celestica, Inc. | 2026-05-01 | -27.9% | -0.1% | 100.3% | $1148.9M |
| **MPWR** | Monolithic Power Systems, Inc. | 2026-04-24 | -19.7% | +26.9% | 53.7% | $1131.5M |
| **NXPI** | NXP Semiconductors N.V. | 2026-05-29 | -30.6% | -4.1% | 110.7% | $1069.5M |
| **CIEN** | Ciena Corporation | 2026-05-22 | -32.8% | +61.2% | 56.2% | $881.3M |
| **F** | Ford Motor Company | 2026-05-29 | -19.8% | +2.9% | 89.8% | $718.4M |
| **FSLR** | First Solar, Inc. | 2026-05-29 | -30.2% | -11.6% | 143.5% | $607.9M |
| **APLD** | Applied Digital Corporation Common Stock | 2026-05-29 | -39.4% | -23.4% | 188.6% | $554.3M |
| **FN** | Fabrinet | 2026-05-15 | -38.4% | -7.2% | 114.2% | $498.8M |
| **FDX** | FedEx Corporation | 2026-05-29 | -20.8% | +5.8% | 82.8% | $484.0M |
| **MTZ** | MasTec, Inc. | 2026-05-01 | -34.9% | +13.1% | 82.3% | $470.9M |
| **HUT** | Hut 8 Corp. Common Stock | 2026-05-29 | -29.0% | +52.3% | 54.3% | $456.5M |
| **AKAM** | Akamai Technologies Inc | 2026-05-15 | -27.0% | +17.8% | 71.1% | $416.9M |
| **SITM** | SiTime Corporation Common Stock | 2026-05-08 | -28.1% | +64.9% | 49.9% | $319.5M |
| **CBOE** | Cboe Global Markets, Inc. | 2026-05-15 | -19.0% | +11.0% | 70.3% | $292.9M |
| **DOW** | Dow Inc. | 2026-04-02 | -20.5% | +19.4% | 61.4% | $280.5M |
| **ALB** | Albemarle Corporation | 2026-05-08 | -34.1% | -17.7% | 171.3% | $280.4M |
| **AA** | Alcoa Corporation | 2026-05-29 | -34.9% | -15.8% | 154.2% | $251.5M |
| **CAVA** | CAVA Group, Inc. | 2026-04-24 | -24.7% | +8.6% | 80.5% | $249.7M |
| **MXL** | MaxLinear, Inc. Common Stock | 2026-05-08 | -34.6% | +252.7% | 42.5% | $229.2M |
| **AEIS** | Advanced Energy Industries Inc | 2026-05-01 | -24.8% | +15.3% | 71.3% | $217.5M |
| **VIAV** | Viavi Solutions Inc. Common Stock | 2026-05-01 | -30.3% | +107.3% | 45.7% | $213.8M |
| **VICR** | Vicor Corp | 2026-05-29 | -38.0% | +38.5% | 68.8% | $190.8M |
| **AUR** | Aurora Innovation, Inc. Class A Common Stock | 2026-05-15 | -20.6% | +33.6% | 50.8% | $189.7M |
| **WSO** | Watsco, Inc. | 2026-04-24 | -28.6% | -17.6% | 213.6% | $180.6M |
| **DY** | Dycom Industries, Inc. | 2026-05-29 | -21.5% | +9.8% | 75.4% | $171.7M |
| **FSLY** | Fastly, Inc. Class A Common Stock | 2026-04-02 | -32.2% | +144.5% | 44.6% | $169.5M |
| **SANM** | Sanmina  Corp | 2026-05-29 | -26.6% | +19.7% | 68.7% | $163.2M |
| **NYT** | New York Times Co. | 2026-04-02 | -23.8% | -8.7% | 143.6% | $160.6M |
| **BWXT** | BWX Technologies, Inc. | 2026-04-17 | -33.6% | -23.9% | 261.4% | $153.9M |
| **POWL** | Powell Industries Inc | 2026-05-08 | -36.2% | +41.7% | 65.8% | $153.3M |
| **FORM** | FormFactor Inc. | 2026-04-24 | -25.4% | +64.2% | 46.5% | $152.1M |
| **GME** | GameStop Corp. Class A | 2026-05-01 | -32.0% | -15.0% | 160.2% | $135.3M |
| **LOGI** | Logitech International SA | 2026-05-29 | -21.8% | +0.2% | 99.1% | $121.9M |
| **CENX** | Century Aluminum Co | 2026-04-10 | -35.2% | -5.0% | 110.8% | $104.8M |
| **ALM** | Almonty Industries Inc. Common Shares | 2026-04-17 | -24.5% | +101.0% | 39.2% | $103.6M |
| **CRUS** | Cirrus Logic Inc | 2026-04-24 | -32.9% | -5.1% | 112.4% | $103.2M |

_+ 40 more liquid members in `stocks.csv`._

### G3

| Ticker | Name | Peak week | Peak→now | Now vs Jan | Jan→peak rally retraced | 20d $ volume |
|---|---|---|---:|---:|---:|---:|
| **RKLB** | Rocket Lab Corporation Common Stock | 2026-05-29 | -49.2% | -14.0% | 120.3% | $1409.1M |
| **AXTI** | AXT Inc | 2026-05-22 | -48.1% | +294.4% | 55.4% | $937.8M |
| **NXT** | Nextpower Inc. Class A Common Stock | 2026-05-29 | -43.8% | -8.8% | 114.2% | $271.8M |
| **RMBS** | Rambus Inc | 2026-04-24 | -41.8% | -14.7% | 131.4% | $241.2M |
| **RDW** | Redwire Corporation | 2026-05-29 | -52.1% | +0.5% | 99.5% | $205.0M |
| **NVTS** | Navitas Semiconductor Corporation Common Stock | 2026-05-22 | -55.7% | +28.8% | 84.9% | $185.0M |
| **ENPH** | Enphase Energy, Inc. | 2026-05-29 | -43.9% | +8.9% | 90.5% | $178.8M |
| **LUNR** | Intuitive Machines, Inc. Class A Common Stock | 2026-05-29 | -59.1% | -5.6% | 104.3% | $178.8M |
| **WOLF** | Wolfspeed, Inc. | 2026-05-22 | -62.3% | +41.7% | 84.9% | $137.3M |
| **SEDG** | SolarEdge Technologies, Inc. | 2026-05-29 | -59.5% | -6.0% | 104.6% | $136.8M |
| **PL** | Planet Labs PBC | 2026-05-29 | -56.6% | -11.2% | 110.7% | $131.8M |
| **AAON** | Aaon Inc | 2026-05-29 | -42.3% | -11.2% | 120.8% | $114.1M |
| **PRIM** | Primoris Services Corporation | 2026-05-01 | -56.1% | -46.5% | 313.2% | $104.4M |
| **POET** | POET Technologies Inc. Common Shares | 2026-05-15 | -48.2% | +15.5% | 87.4% | $104.2M |
| **LBRT** | Liberty Energy Inc. | 2026-05-08 | -41.9% | -4.7% | 107.3% | $103.1M |
| **LUMN** | Lumen Technologies, Inc. | 2026-05-29 | -45.3% | -28.9% | 196.1% | $88.2M |
| **CAPR** | Capricor Therapeutics Inc | 2026-04-24 | -80.6% | -71.3% | 248.8% | $75.7M |
| **AMPX** | Amprius Technologies, Inc. | 2026-05-01 | -51.1% | -7.3% | 108.1% | $68.9M |
| **CAR** | Avis Budget Group, Inc. | 2026-04-17 | -70.8% | +14.6% | 95.0% | $64.7M |
| **UI** | Ubiquiti Inc. Common Stock | 2026-04-17 | -47.1% | +4.1% | 95.8% | $59.4M |
| **CTRI** | Centuri Holdings, Inc. | 2026-05-01 | -44.5% | -21.1% | 150.0% | $53.7M |
| **BW** | Babcock & Wilcox Enterprises, Inc. | 2026-05-15 | -61.4% | +0.1% | 99.9% | $49.5M |
| **CC** | The Chemours Company | 2026-05-01 | -43.4% | +4.7% | 94.5% | $47.5M |
| **SHLS** | Shoals Technologies Group, Inc. Class A Common Stock | 2026-05-29 | -40.8% | -20.0% | 157.1% | $46.3M |
| **SATL** | Satellogic Inc. Class A Ordinary Shares | 2026-05-22 | -50.0% | +48.9% | 75.3% | $29.4M |
| **BKSY** | BlackSky Technology Inc. | 2026-05-29 | -44.9% | +4.7% | 94.7% | $28.0M |
| **AHCO** | AdaptHealth Corp. Common Stock | 2026-05-01 | -59.1% | -46.6% | 252.6% | $27.5M |
| **PLAB** | Photronics Inc | 2026-05-08 | -43.1% | -11.6% | 121.0% | $27.1M |
| **MRAM** | Everspin Technologies, Inc | 2026-05-15 | -54.8% | +30.6% | 83.8% | $22.7M |
| **LWLG** | Lightwave Logic, Inc. Common Stock | 2026-05-08 | -62.9% | +56.0% | 82.5% | $22.3M |
| **ADTN** | ADTRAN Holdings, Inc. Common Stock | 2026-05-01 | -57.4% | -14.8% | 114.7% | $21.8M |

## Notes

- `Peak→now` is the grouping variable. It measures how far the current weekly close sits below the Apr–May peak.
- `Now vs Jan` anchors the visual observation of stocks that have fallen back to January 2026 price levels.
- `Jan→peak rally retraced` = (peak − current) / (peak − Jan median). Around 100% means the whole January-to-peak rally has been given back; above 100% means current price is below the January median.
- The full matched class, including less-liquid names, is in `stocks.csv` and `analysis.json`.
- Current-active-CS filtering introduces survivorship bias by design: the result is intended as a current investable universe, not a historical delisting study.
