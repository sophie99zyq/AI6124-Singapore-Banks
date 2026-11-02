# UOB and STI Data Audit

Audit date: 2026-10-23.

## Snapshot identity

| Series | Ticker | Rows | First date | Last date | SHA-256 |
|---|---|---:|---|---|---|
| UOB | U11.SI | 3,265 | 2013-10-01 | 2026-09-30 | `79626acecd223fe8d99fccdf66525c7c72ea6b547b93bcb0ac5d07c1ce4a4e6e` |
| Straits Times Index | ^STI | 3,258 | 2013-10-01 | 2026-09-30 | `9c4fe4be59efe4537def8a9223ba32c4a24302683b64b4d69be1e09e75e4712b` |

Yahoo Finance was accessed through yfinance 1.7.0 with `auto_adjust=False` and corporate actions retained. The research pipeline derives adjusted OHLC with the daily `Adj Close / Close` factor. The raw snapshots remain unchanged.

## Alignment policy

The model uses only dates with an observed UOB row, positive reported UOB volume, and an observed STI close. It does not fill missing STI prices or interpolate UOB observations. This policy produces 3,257 aligned rows. Exact exclusions are recorded by the preparation command in `data/processed/alignment.json`.

### Excluded UOB zero-volume rows

| Date | Local evidence | Decision |
|---|---|---|
| 2014-10-22 | OHLC equals the previous close; STI row is absent | Exclude as a non-tradable placeholder |
| 2015-08-07 | OHLC equals the previous close; STI row is absent | Exclude as a non-tradable placeholder |
| 2019-01-01 | Public holiday; OHLC equals the previous close | Exclude as a holiday placeholder |
| 2019-02-05 | Chinese New Year public holiday; OHLC equals the previous close | Exclude as a holiday placeholder |
| 2019-12-25 | Christmas public holiday; OHLC equals the previous close | Exclude as a holiday placeholder |
| 2020-03-18 | Normal weekday; OHLC exactly repeats 2020-03-17 while STI moved from 2,454.53 to 2,425.62 | Exclude as an anomalous placeholder; do not infer or interpolate a UOB price |

The March 2020 row deserves explicit disclosure because it occurs during the COVID-19 selloff. Contemporary sources confirm large UOB price movement and trading activity during March 2020, while the downloaded row reports no price range and no volume. A public monthly source reports March 2020 UOB prices spanning approximately SGD 19.45 to SGD 14.00, consistent with a volatile month rather than a valid unchanged zero-volume trading observation.

### UOB dates absent from downloaded STI data

| Date | External check | Decision |
|---|---|---|
| 2014-12-24 | External STI history reports a close of 3,345.91 | Exclude the joint sample because the raw Yahoo STI snapshot has no row |
| 2014-12-26 | External STI history reports a close of 3,353.68 | Exclude the joint sample because the raw Yahoo STI snapshot has no row |

These two dates demonstrate missing early Yahoo STI observations. The project does not copy values from a secondary website into the research snapshot. This preserves a single reproducible data source and avoids undocumented manual edits.

## Index-volume observation

Yahoo reports zero STI volume for most of 2016 and many dates in 2015 and 2017. The project uses STI close only, so index volume is neither a feature nor a validity filter. Its zero values do not remove otherwise observed index closes.

## Coverage after feature warm-up

The download begins in October 2013. This supplies enough history for the 252-day STI drawdown and a complete 30-day sequence before the first target in January 2015. Prepared samples cover target dates from 2015-01-02 through 2026-09-30.

## Remaining limitation

Yahoo Finance is a convenient vendor source rather than an independently certified SGX transaction record. Retrospective adjusted prices can change when the vendor updates corporate-action treatment. The final report must cite the snapshot hashes, adjustment convention, exclusion policy, and this limitation.

## External references used for the audit

- UOB investor-relations historical-price page: https://www.uob.com.sg/investor-relations/shares-and-dividends/historical-price.html
- UOB 2020 investor highlights and annual price range: https://www.uob.com.sg/AR2020/investor-highlights.html
- Public UOB monthly price history: https://www.digrin.com/stocks/detail/U11.SI/price
- Public STI December 2014 history: https://countryeconomy.com/stock-exchange/singapore?dr=2014-12
- yfinance history options: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
- yfinance price-repair limitations: https://ranaroussi.github.io/yfinance/advanced/price_repair.html
