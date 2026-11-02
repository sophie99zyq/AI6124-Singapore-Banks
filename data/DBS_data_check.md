# DBS 日线数据检查报告

检查日期：2026-10-11。原始 CSV 未修改。

源文件：`DBS_D05_SI_daily_2020_2025.csv`
SHA-256：`af7b26aab1e04c132f8bded1ee5c33b62beebb997557ee3c01d7234033cac7c8`

## 日期完整性

将周一至周五日期扣除人工核对的 MOM 公共假日、补休日及 2020/2023 选举日。此核对不是独立 SGX 行情逐行对账。

| 年份 | 平日数 | 假日数 | 预期记录 | 实际记录 |
|---|---:|---:|---:|---:|
| 2020 | 262 | 10 | 252 | 252 |
| 2021 | 261 | 8 | 253 | 253 |
| 2022 | 260 | 10 | 250 | 250 |
| 2023 | 260 | 11 | 249 | 249 |
| 2024 | 262 | 10 | 252 | 252 |
| 2025 | 261 | 9 | 252 | 252 |

预期 1508 行，实际 1508 行；无法解释的缺失日期 0，额外日期 0。

## 数值检查

- 八个字段无缺失或非有限值。
- 日期升序、无重复、无周末记录。
- 五个价格字段均为正，OHLC 范围一致，无负成交量。
- 存在一条待核实记录：2020-03-18，Volume=0，Open=High=Low=Close=16.363636，并且 Close 与前一日相同。
- 这可能是数据源的占位/异常记录，也可能有市场事件原因；目前没有独立行情证实，不能直接断言。不要插值、随意删除或声称全部行情已验证。

## 价格口径

- auto_adjust=False 不代表 Yahoo Close 是历史原始成交价：历史 OHLC 可已按拆股/送股调整；Adj Close 另含分红等调整。
- 2024-04-22 Stock Splits=1.1，对应 DBS 每十股送一股，除权日与 SGX 公告一致。
- Adj Close 的调整因子可能包含数据区间之后的公司行动；不能直接当作原始成交价格，也不能再无条件叠加分红。

## 人工核对的未出现平日

- 2020：2020-01-01, 2020-01-27, 2020-04-10, 2020-05-01, 2020-05-07, 2020-05-25, 2020-07-10, 2020-07-31, 2020-08-10, 2020-12-25
- 2021：2021-01-01, 2021-02-12, 2021-04-02, 2021-05-13, 2021-05-26, 2021-07-20, 2021-08-09, 2021-11-04
- 2022：2022-02-01, 2022-02-02, 2022-04-15, 2022-05-02, 2022-05-03, 2022-05-16, 2022-07-11, 2022-08-09, 2022-10-24, 2022-12-26
- 2023：2023-01-02, 2023-01-23, 2023-01-24, 2023-04-07, 2023-05-01, 2023-06-02, 2023-06-29, 2023-08-09, 2023-09-01, 2023-11-13, 2023-12-25
- 2024：2024-01-01, 2024-02-12, 2024-03-29, 2024-04-10, 2024-05-01, 2024-05-22, 2024-06-17, 2024-08-09, 2024-10-31, 2024-12-25
- 2025：2025-01-01, 2025-01-29, 2025-01-30, 2025-03-31, 2025-04-18, 2025-05-01, 2025-05-12, 2025-10-20, 2025-12-25

## 核对来源

- https://www.mom.gov.sg/newsroom/press-releases/2019/0408-public-holidays-for-2020
- https://www.mom.gov.sg/newsroom/press-releases/2020/0624-public-holiday-on-polling-day---10-july-2020
- https://www.mom.gov.sg/newsroom/press-releases/2020/0624-public-holidays-for-2021
- https://www.mom.gov.sg/newsroom/press-releases/2021/1021-revised-dates-for-hari-raya-puasa-and--hari-raya-haji-public-holidays-in-2022
- https://www.mom.gov.sg/newsroom/press-releases/2022/0929-revised-date-for-vesak-day-2023
- https://www.mom.gov.sg/newsroom/press-releases/2023/0822-public-holiday-on-polling-day---1-sep-2023
- https://www.mom.gov.sg/newsroom/press-releases/2023/0524-public-holidays-for-2024
- https://www.mom.gov.sg/newsroom/press-releases/2024/0805-public-holidays-for-2025
- https://links.sgx.com/1.0.0/corporate-announcements/MB7JJLVGBACMNWW6/
