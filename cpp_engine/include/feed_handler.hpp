#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <memory>

namespace crossflux {

struct NormalizedOrderBook {
    std::string symbol;
    double bid_price = 0.0;
    double bid_qty = 0.0;
    double ask_price = 0.0;
    double ask_qty = 0.0;
    uint64_t timestamp_ms = 0;
};

enum class AssetClass { CRYPTO, EQUITY };

class IMarketFeedHandler {
public:
    virtual ~IMarketFeedHandler() = default;
    virtual void connect(const std::function<void(const NormalizedOrderBook&)>& on_book) = 0;
    virtual void disconnect() = 0;
    virtual std::string name() const = 0;
    virtual AssetClass asset_class() const = 0;
    virtual std::vector<std::string> subscribed_symbols() const = 0;
};

std::unique_ptr<IMarketFeedHandler> create_binance_feed(
    const std::vector<std::string>& symbols = {"btcusdt"});

std::unique_ptr<IMarketFeedHandler> create_alpaca_feed(
    const std::vector<std::string>& symbols = {"AAPL"});

} // namespace crossflux
