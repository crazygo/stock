# Apr–May Peak Class — Industry & Profitability Structure

## Coverage and source

- Wave class: **463** stocks (G1 197, G2 169, G3 97).
- Sector coverage: **452 / 463**; detailed-industry coverage: **452 / 463**.
- Stocks with ≥3 profitability features: **417 / 463**.
- Profit clustering fit count: **378**; selected k = **4**, silhouette = **0.539**.
- G1/G2/G3 and drawdowns remain Massive-derived. Sector/industry and profitability fields are a Yahoo Finance snapshot via yfinance.
- Financial Services and Real Estate are separated from operating-company profitability clustering because their statement economics are not directly comparable.

## Sector commonality

| Sector | N | G1 | G2 | G3 | Median drawdown | G1 lift | G2 lift | G3 lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Technology | 89 | 13 | 44 | 32 | 33.0% | 0.34x | 1.35x | 1.72x |
| Industrials | 76 | 24 | 26 | 26 | 26.5% | 0.74x | 0.94x | 1.63x |
| Healthcare | 66 | 23 | 29 | 14 | 22.9% | 0.82x | 1.20x | 1.01x |
| Energy | 56 | 44 | 9 | 3 | 11.2% | 1.85x | 0.44x | 0.26x |
| Financial Services | 32 | 22 | 7 | 3 | 9.7% | 1.62x | 0.60x | 0.45x |
| Basic Materials | 31 | 7 | 17 | 7 | 29.4% | 0.53x | 1.50x | 1.08x |
| Consumer Cyclical | 25 | 10 | 12 | 3 | 20.6% | 0.94x | 1.32x | 0.57x |
| Communication Services | 22 | 7 | 9 | 6 | 27.4% | 0.75x | 1.12x | 1.30x |
| Utilities | 22 | 16 | 4 | 2 | 11.2% | 1.71x | 0.50x | 0.43x |
| Consumer Defensive | 21 | 11 | 9 | 1 | 14.6% | 1.23x | 1.17x | 0.23x |
| Real Estate | 12 | 9 | 3 | 0 | 4.7% | 1.76x | 0.68x | 0.00x |
| Unknown | 11 | 11 | 0 | 0 | 1.5% | 2.35x | 0.00x | 0.00x |

### Sector signals

- **G1 over-represented:** Unknown (2.35x, N=11), Energy (1.85x, N=56), Real Estate (1.76x, N=12), Utilities (1.71x, N=22), Financial Services (1.62x, N=32), Consumer Defensive (1.23x, N=21)
- **G2 over-represented:** Basic Materials (1.50x, N=31), Technology (1.35x, N=89), Consumer Cyclical (1.32x, N=25), Healthcare (1.20x, N=66)
- **G3 over-represented:** Technology (1.72x, N=89), Industrials (1.63x, N=76), Communication Services (1.30x, N=22)

## Detailed-industry commonality

Only detailed industries with at least 5 matched stocks are shown.

| Industry | N | G1 | G2 | G3 | Median drawdown | G1 lift | G3 lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Biotechnology | 52 | 17 | 24 | 11 | 22.9% | 0.77x | 1.01x |
| Semiconductors | 24 | 4 | 11 | 9 | 35.8% | 0.39x | 1.79x |
| Oil & Gas Equipment & Services | 16 | 8 | 6 | 2 | 18.7% | 1.18x | 0.60x |
| Oil & Gas Midstream | 16 | 15 | 1 | 0 | 8.4% | 2.20x | 0.00x |
| Communication Equipment | 15 | 2 | 7 | 6 | 32.8% | 0.31x | 1.91x |
| Electrical Equipment & Parts | 13 | 3 | 4 | 6 | 36.2% | 0.54x | 2.20x |
| Oil & Gas E&P | 13 | 12 | 1 | 0 | 11.2% | 2.17x | 0.00x |
| Software - Application | 13 | 3 | 7 | 3 | 28.6% | 0.54x | 1.10x |
| Aerospace & Defense | 11 | 1 | 2 | 8 | 50.0% | 0.21x | 3.47x |
| Unknown | 11 | 11 | 0 | 0 | 1.5% | 2.35x | 0.00x |
| Engineering & Construction | 10 | 4 | 5 | 1 | 20.5% | 0.94x | 0.48x |
| Electronic Components | 9 | 1 | 7 | 1 | 29.6% | 0.26x | 0.53x |
| Internet Content & Information | 9 | 4 | 1 | 4 | 31.0% | 1.04x | 2.12x |
| Other Industrial Metals & Mining | 9 | 2 | 4 | 3 | 29.4% | 0.52x | 1.59x |
| Software - Infrastructure | 8 | 2 | 5 | 1 | 26.3% | 0.59x | 0.60x |
| Utilities - Regulated Gas | 8 | 7 | 0 | 1 | 9.8% | 2.06x | 0.60x |
| Asset Management | 7 | 6 | 0 | 1 | 5.4% | 2.01x | 0.68x |
| Oil & Gas Drilling | 7 | 6 | 1 | 0 | 12.9% | 2.01x | 0.00x |
| Specialty Business Services | 7 | 0 | 5 | 2 | 31.2% | 0.00x | 1.36x |
| Specialty Chemicals | 7 | 1 | 3 | 3 | 34.1% | 0.34x | 2.05x |
| Utilities - Regulated Electric | 7 | 7 | 0 | 0 | 8.3% | 2.35x | 0.00x |
| Marine Shipping | 6 | 6 | 0 | 0 | 7.2% | 2.35x | 0.00x |
| Semiconductor Equipment & Materials | 6 | 0 | 2 | 4 | 43.8% | 0.00x | 3.18x |
| Shell Companies | 6 | 5 | 0 | 1 | 2.8% | 1.96x | 0.80x |
| Solar | 6 | 0 | 1 | 5 | 43.9% | 0.00x | 3.98x |
| Chemicals | 5 | 1 | 3 | 1 | 31.7% | 0.47x | 0.95x |
| Farm Products | 5 | 3 | 2 | 0 | 14.6% | 1.41x | 0.00x |
| Information Technology Services | 5 | 1 | 3 | 1 | 39.4% | 0.47x | 0.95x |
| Medical Devices | 5 | 1 | 1 | 3 | 41.4% | 0.47x | 2.86x |
| Telecom Services | 5 | 1 | 3 | 1 | 36.4% | 0.47x | 0.95x |

## Profitability modules

Modules are learned cross-sectionally from revenue growth, operating margin, net margin, FCF margin and earnings growth. Labels describe cluster medians rather than hand-classifying individual stocks.

| Module | N | Rev growth | Op margin | Net margin | FCF margin | Earnings growth | Median drawdown | G1/G2/G3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2 盈利改善 | 311 | 10.1% | 9.5% | 4.1% | 4.6% | 23.3% | 22.6% | 127/123/61 |
| P-SPARSE 数据不足 | 41 | —% | —% | —% | —% | —% | 21.4% | 19/14/8 |
| P-FIN 金融口径单列 | 32 | —% | —% | —% | —% | —% | 9.7% | 22/7/3 |
| P4 持续亏损 | 26 | -17.9% | -3167.5% | 0.0% | -2107.0% | —% | 34.3% | 7/7/12 |
| P3 持续亏损 | 25 | 9.2% | -216.9% | -76.4% | -124.6% | —% | 39.4% | 3/10/12 |
| P1 高增长高盈利 | 16 | 38.6% | 22.6% | 17.9% | 10.7% | 1133.4% | 11.9% | 10/5/1 |
| P-REIT 房地产口径单列 | 12 | —% | —% | —% | —% | —% | 4.7% | 9/3/0 |

### Profit-module signals

- **G1 over-represented:** P-REIT 房地产口径单列 (1.76x, N=12), P-FIN 金融口径单列 (1.62x, N=32), P1 高增长高盈利 (1.47x, N=16)
- **G2 over-represented:** none
- **G3 over-represented:** P3 持续亏损 (2.29x, N=25), P4 持续亏损 (2.20x, N=26)

## Detailed industry × profitability module

| Industry | Profit module | N | G1 | G2 | G3 | Median drawdown |
|---|---|---:|---:|---:|---:|---:|
| Biotechnology | P-SPARSE 数据不足 | 25 | 7 | 12 | 6 | 34.0% |
| Semiconductors | P2 盈利改善 | 19 | 4 | 10 | 5 | 32.9% |
| Oil & Gas Equipment & Services | P2 盈利改善 | 16 | 8 | 6 | 2 | 18.7% |
| Oil & Gas Midstream | P2 盈利改善 | 14 | 13 | 1 | 0 | 8.5% |
| Communication Equipment | P2 盈利改善 | 13 | 2 | 6 | 5 | 32.2% |
| Electrical Equipment & Parts | P2 盈利改善 | 13 | 3 | 4 | 6 | 36.2% |
| Biotechnology | P4 持续亏损 | 11 | 6 | 2 | 3 | 15.8% |
| Oil & Gas E&P | P2 盈利改善 | 10 | 9 | 1 | 0 | 12.5% |
| Engineering & Construction | P2 盈利改善 | 10 | 4 | 5 | 1 | 20.5% |
| Software - Application | P2 盈利改善 | 10 | 3 | 5 | 2 | 30.1% |
| Unknown | P-SPARSE 数据不足 | 9 | 9 | 0 | 0 | 1.5% |
| Biotechnology | P3 持续亏损 | 9 | 2 | 5 | 2 | 23.5% |
| Electronic Components | P2 盈利改善 | 9 | 1 | 7 | 1 | 29.6% |
| Internet Content & Information | P2 盈利改善 | 8 | 3 | 1 | 4 | 38.5% |
| Asset Management | P-FIN 金融口径单列 | 7 | 6 | 0 | 1 | 5.4% |
| Utilities - Regulated Electric | P2 盈利改善 | 7 | 7 | 0 | 0 | 8.3% |
| Utilities - Regulated Gas | P2 盈利改善 | 7 | 6 | 0 | 1 | 9.6% |
| Oil & Gas Drilling | P2 盈利改善 | 7 | 6 | 1 | 0 | 12.9% |
| Biotechnology | P2 盈利改善 | 7 | 2 | 5 | 0 | 19.6% |
| Aerospace & Defense | P2 盈利改善 | 7 | 1 | 1 | 5 | 50.0% |
| Shell Companies | P-FIN 金融口径单列 | 6 | 5 | 0 | 1 | 2.8% |
| Specialty Business Services | P2 盈利改善 | 6 | 0 | 4 | 2 | 30.0% |
| Specialty Chemicals | P2 盈利改善 | 6 | 1 | 3 | 2 | 34.0% |
| Solar | P2 盈利改善 | 6 | 0 | 1 | 5 | 43.9% |
| Farm Products | P2 盈利改善 | 5 | 3 | 2 | 0 | 14.6% |
| Software - Infrastructure | P2 盈利改善 | 5 | 2 | 3 | 0 | 25.5% |
| Chemicals | P2 盈利改善 | 5 | 1 | 3 | 1 | 31.7% |
| Telecom Services | P2 盈利改善 | 5 | 1 | 3 | 1 | 36.4% |
| Real Estate Services | P-REIT 房地产口径单列 | 4 | 4 | 0 | 0 | 0.4% |
| REIT - Specialty | P-REIT 房地产口径单列 | 4 | 4 | 0 | 0 | 3.5% |
| Banks - Regional | P-FIN 金融口径单列 | 4 | 4 | 0 | 0 | 6.2% |
| Credit Services | P-FIN 金融口径单列 | 4 | 4 | 0 | 0 | 10.1% |
| Conglomerates | P2 盈利改善 | 4 | 1 | 3 | 0 | 24.7% |
| Building Products & Equipment | P2 盈利改善 | 4 | 1 | 1 | 2 | 30.7% |
| Other Industrial Metals & Mining | P-SPARSE 数据不足 | 4 | 1 | 2 | 1 | 30.8% |
| Capital Markets | P-FIN 金融口径单列 | 4 | 0 | 3 | 1 | 31.0% |
| Medical Devices | P2 盈利改善 | 4 | 1 | 1 | 2 | 32.5% |
| Packaged Foods | P2 盈利改善 | 4 | 0 | 3 | 1 | 33.0% |
| Information Technology Services | P2 盈利改善 | 4 | 1 | 2 | 1 | 34.3% |
| Semiconductor Equipment & Materials | P2 盈利改善 | 4 | 0 | 1 | 3 | 43.8% |

## Foundation for industry-neutral company analysis

Each stock now carries these control variables:

- `sector_adjusted_drawdown_residual_pp`: stock drawdown minus broad-sector median.
- `industry_adjusted_drawdown_residual_pp`: stock drawdown minus detailed-industry median when that industry has ≥5 members; otherwise sector median.
- `profit_module_adjusted_drawdown_residual_pp`: stock drawdown minus companies with a similar profitability structure.
- `peer_adjusted_drawdown_residual_pp`: stock drawdown minus detailed-industry × profitability-module peers when the cell has ≥5 members; otherwise the industry/sector benchmark.

A negative residual means the stock held up better than its comparison group; a positive residual means it fell more than its peers. These are price-behaviour residuals, not intrinsic-value or rebound-probability scores.
