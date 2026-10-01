from types import SimpleNamespace

import pytest

from backtest.currency_sensitivity import currency_pnl_components
from src.fees import active as active_fees


def test_components_reconcile_to_existing_parity_accounting() -> None:
    fees = active_fees()
    buy_qty, sell_qty = 0.11, 0.10
    buy_price, sell_price = 60_010.0, 60_020.0
    matched, residual, legging = 0.10, 0.01, 0.20
    trade = SimpleNamespace(
        status="filled", buy_venue="binance", sell_venue="kraken",
        buy_price=buy_price, sell_price=sell_price,
        matched_qty=matched, buy_filled_qty=buy_qty,
        sell_filled_qty=sell_qty, residual_qty=residual,
        legging_cost=legging,
    )
    expected = (
        (sell_price - buy_price) * matched
        - buy_qty * buy_price * fees.taker_rate("binance")
        - sell_qty * sell_price * fees.taker_rate("kraken")
        - legging
    )

    components = currency_pnl_components(trade, fees)
    assert components.at_usdt_usd(1.0) == pytest.approx(expected)
    assert components.at_usdt_usd(1.001) - components.at_usdt_usd(1.0) == pytest.approx(
        components.usdt * 0.001
    )


def test_rejected_trade_has_no_cashflow() -> None:
    trade = SimpleNamespace(status="rejected")
    components = currency_pnl_components(trade)
    assert components.usd == 0.0
    assert components.usdt == 0.0


def test_nonpositive_scenario_rate_is_rejected() -> None:
    from backtest.currency_sensitivity import CurrencyPnlComponents

    with pytest.raises(ValueError, match="must be > 0"):
        CurrencyPnlComponents(1.0, 2.0).at_usdt_usd(0.0)
