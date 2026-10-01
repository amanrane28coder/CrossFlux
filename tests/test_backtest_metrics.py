from types import SimpleNamespace

import numpy as np

from backtest.engine import _annualised_sharpe


def _trades(pnls: list[float]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            timestamp_ms=i * 60_000,
            fill_ts_ms=i * 60_000,
            pnl_net=pnl,
        )
        for i, pnl in enumerate(pnls)
    ]


def test_sharpe_uses_time_buckets_not_trade_count() -> None:
    base = _trades([1.0, 2.0, 0.0, 3.0, -1.0, 2.0, 1.0, 0.0])
    duplicated = []
    for trade in base:
        duplicated.append(trade)
        duplicated.extend([
            SimpleNamespace(
                timestamp_ms=trade.timestamp_ms,
                fill_ts_ms=trade.fill_ts_ms,
                pnl_net=0.0,
            )
            for _ in range(99)
        ])

    assert _annualised_sharpe(base, 8 * 60) == _annualised_sharpe(duplicated, 8 * 60)


def test_hac_sharpe_accounts_for_persistent_minute_returns() -> None:
    # Returns remain at one level for 60-minute blocks, creating strong positive
    # serial correlation that an IID Sharpe would ignore.
    block_levels = [1.0, 2.0, 1.5, 2.5] * 5
    pnls = [level for level in block_levels for _ in range(60)]
    trades = _trades(pnls)

    hac = _annualised_sharpe(trades, len(pnls) * 60)
    minute_returns = np.asarray(pnls, dtype=float) / 100_000.0
    naive = minute_returns.mean() / minute_returns.std(ddof=1) * np.sqrt(365.25 * 24 * 60)

    assert np.isfinite(hac)
    assert 0.0 < hac < naive


def test_sharpe_returns_zero_when_all_fills_share_one_minute() -> None:
    trades = [
        SimpleNamespace(timestamp_ms=1_000, fill_ts_ms=1_000, pnl_net=1.0),
        SimpleNamespace(timestamp_ms=2_000, fill_ts_ms=2_000, pnl_net=2.0),
    ]

    assert _annualised_sharpe(trades, 60.0) == 0.0
