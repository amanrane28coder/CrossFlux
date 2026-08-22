#include "feed_handler.hpp"
#include "models.hpp"

#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace crossflux {
namespace {

// ─── Shared utilities ───────────────────────────────────────────────────────

void load_system_ca_certs(boost::asio::ssl::context& ssl_ctx) {
    const std::vector<std::string> ca_paths = {
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/openssl/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
        "/usr/local/etc/openssl/cert.pem",
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/Library/Frameworks/Python.framework/Versions/Current/etc/openssl/cert.pem"
    };
    for (const auto& path : ca_paths) {
        std::ifstream f(path);
        if (f.good()) {
            boost::system::error_code ec;
            ssl_ctx.load_verify_file(path, ec);
            if (!ec) return;
        }
    }
    ssl_ctx.set_default_verify_paths();
}

using WSS = boost::beast::websocket::stream<boost::asio::ssl::stream<boost::asio::ip::tcp::socket>>;
using BookCB = std::function<void(const NormalizedOrderBook&)>;

uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ─── BinanceFeed ─────────────────────────────────────────────────────────────

class BinanceFeed : public IMarketFeedHandler {
public:
    explicit BinanceFeed(const std::vector<std::string>& symbols)
        : symbols_(symbols.empty() ? std::vector<std::string>{"btcusdt"} : symbols) {}

    void connect(const BookCB& on_book) override {
        on_book_ = on_book;
        running_.store(true);
        worker_ = std::thread(&BinanceFeed::run_loop, this);
    }

    void disconnect() override {
        running_.store(false);
        if (ws_) {
            boost::system::error_code ec;
            boost::beast::get_lowest_layer(*ws_).close(ec);
        }
        if (worker_.joinable()) worker_.join();
        if (ws_) {
            boost::system::error_code ec;
            if (ws_->is_open())
                ws_->close(boost::beast::websocket::close_code::normal, ec);
        }
    }

    std::string name() const override { return "binance"; }
    AssetClass asset_class() const override { return AssetClass::CRYPTO; }
    std::vector<std::string> subscribed_symbols() const override { return symbols_; }

private:
    void run_loop() {
        while (running_.load()) {
            try {
                connect_and_run();
                if (!running_.load()) break;
                reconnect();
            } catch (const std::exception& e) {
                std::cerr << "[BinanceFeed] " << e.what() << std::endl;
                reconnect();
            }
        }
    }

    void connect_and_run() {
        boost::asio::io_context ioc;
        boost::asio::ssl::context ssl_ctx(boost::asio::ssl::context::tls_client);
        load_system_ca_certs(ssl_ctx);

        boost::asio::ip::tcp::resolver resolver(ioc);
        auto results = resolver.resolve("stream.binance.com", "9443");

        auto ws = std::make_unique<WSS>(ioc, ssl_ctx);
        ws->set_option(boost::beast::websocket::stream_base::decorator(
            [](boost::beast::websocket::request_type& req) {
                req.set(boost::beast::http::field::user_agent,
                        "CrossFluxHFT/1.0");
            }));
        ws->set_option(boost::beast::websocket::stream_base::timeout::suggested(
            boost::beast::role_type::client));

        boost::beast::get_lowest_layer(*ws).connect(*results.begin());
        if(!SSL_set_tlsext_host_name(ws->next_layer().native_handle(), "stream.binance.com")) {
            boost::system::error_code ec{static_cast<int>(::ERR_get_error()),
                boost::asio::error::get_ssl_category()};
            std::cerr << "[BinanceFeed] SNI error: " << ec.message() << std::endl;
        }
        ws->next_layer().handshake(boost::asio::ssl::stream_base::client);

        std::string url = "/ws/" + symbols_[0] + "@depth5@100ms";
        boost::system::error_code ec;
        ws->handshake("stream.binance.com", url, ec);
        if (ec) return;

        reconnect_attempts_ = 0;
        ws_ = std::move(ws);

        boost::beast::flat_buffer buffer;
        while (running_.load() && ws_->is_open()) {
            ws_->read(buffer);
            std::string msg = boost::beast::buffers_to_string(buffer.data());
            buffer.consume(buffer.size());
            process_message(msg);
        }
        ws_.reset();
    }

    void process_message(const std::string& message) {
        try {
            auto json = nlohmann::json::parse(message);
            const auto& data = json.contains("data") ? json["data"] : json;

            std::string symbol = symbols_[0];
            if (json.contains("stream")) {
                std::string s = json["stream"];
                symbol = s.substr(0, s.find('@'));
            }

            const bool has_bids = data.contains("bids") && data.contains("asks");
            const bool has_diff = data.contains("b") && data.contains("a");
            if (!has_bids && !has_diff) return;

            const auto& bids_j = has_bids ? data["bids"] : data["b"];
            const auto& asks_j = has_bids ? data["asks"] : data["a"];
            uint64_t ts = data.value("E", now_ms());

            double best_bid = 0.0, best_bid_qty = 0.0;
            double best_ask = 0.0, best_ask_qty = 0.0;

            if (!bids_j.empty() && bids_j[0].is_array() && bids_j[0].size() >= 2) {
                best_bid = bids_j[0][0].is_string()
                    ? std::stod(bids_j[0][0].get<std::string>())
                    : bids_j[0][0].get<double>();
                best_bid_qty = bids_j[0][1].is_string()
                    ? std::stod(bids_j[0][1].get<std::string>())
                    : bids_j[0][1].get<double>();
            }
            if (!asks_j.empty() && asks_j[0].is_array() && asks_j[0].size() >= 2) {
                best_ask = asks_j[0][0].is_string()
                    ? std::stod(asks_j[0][0].get<std::string>())
                    : asks_j[0][0].get<double>();
                best_ask_qty = asks_j[0][1].is_string()
                    ? std::stod(asks_j[0][1].get<std::string>())
                    : asks_j[0][1].get<double>();
            }

            if (on_book_) {
                NormalizedOrderBook nb;
                nb.symbol = symbol;
                nb.bid_price = best_bid;
                nb.bid_qty = best_bid_qty;
                nb.ask_price = best_ask;
                nb.ask_qty = best_ask_qty;
                nb.timestamp_ms = ts;
                on_book_(nb);
            }

            static std::atomic<long long> tick_count{0};
            if (++tick_count % 10 == 0)
                std::cout << "[BinanceFeed] Tick #" << tick_count
                          << " | bid=" << best_bid << " ask=" << best_ask << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[BinanceFeed] parse error: " << e.what() << std::endl;
        }
    }

    void reconnect() {
        if (!running_.load()) return;
        int delay = 1000 * (1 << std::min(reconnect_attempts_++, 6));
        delay = std::max(100, delay + (std::rand() % (delay / 4)) - (delay / 8));
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    std::vector<std::string> symbols_;
    BookCB on_book_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::unique_ptr<WSS> ws_;
    int reconnect_attempts_ = 0;
};

// ─── AlpacaFeed ──────────────────────────────────────────────────────────────

class AlpacaFeed : public IMarketFeedHandler {
public:
    AlpacaFeed(const std::vector<std::string>& symbols,
               const std::string& api_key,
               const std::string& api_secret)
        : symbols_(symbols.empty() ? std::vector<std::string>{"AAPL"} : symbols)
        , api_key_(api_key)
        , api_secret_(api_secret) {}

    void connect(const BookCB& on_book) override {
        on_book_ = on_book;
        running_.store(true);
        worker_ = std::thread(&AlpacaFeed::run_loop, this);
    }

    void disconnect() override {
        running_.store(false);
        if (ws_) {
            boost::system::error_code ec;
            boost::beast::get_lowest_layer(*ws_).close(ec);
        }
        if (worker_.joinable()) worker_.join();
        if (ws_) {
            boost::system::error_code ec;
            if (ws_->is_open())
                ws_->close(boost::beast::websocket::close_code::normal, ec);
        }
    }

    std::string name() const override { return "alpaca"; }
    AssetClass asset_class() const override { return AssetClass::EQUITY; }
    std::vector<std::string> subscribed_symbols() const override { return symbols_; }

    static std::string alpaca_key() {
        const char* k = std::getenv("ALPACA_API_KEY");
        return k ? std::string(k) : "AK_DEMO";
    }

    static std::string alpaca_secret() {
        const char* s = std::getenv("ALPACA_API_SECRET");
        return s ? std::string(s) : "SK_DEMO";
    }

private:
    void run_loop() {
        while (running_.load()) {
            try {
                connect_and_run();
                if (!running_.load()) break;
                reconnect();
            } catch (const std::exception& e) {
                std::cerr << "[AlpacaFeed] " << e.what() << std::endl;
                reconnect();
            }
        }
    }

    void connect_and_run() {
        boost::asio::io_context ioc;
        boost::asio::ssl::context ssl_ctx(boost::asio::ssl::context::tls_client);
        load_system_ca_certs(ssl_ctx);

        boost::asio::ip::tcp::resolver resolver(ioc);
        auto results = resolver.resolve("stream.data.alpaca.markets", "443");

        auto ws = std::make_unique<WSS>(ioc, ssl_ctx);
        ws->set_option(boost::beast::websocket::stream_base::decorator(
            [](boost::beast::websocket::request_type& req) {
                req.set(boost::beast::http::field::user_agent,
                        "CrossFluxHFT/1.0");
            }));
        ws->set_option(boost::beast::websocket::stream_base::timeout::suggested(
            boost::beast::role_type::client));

        boost::beast::get_lowest_layer(*ws).connect(*results.begin());
        if(!SSL_set_tlsext_host_name(ws->next_layer().native_handle(),
                                     "stream.data.alpaca.markets")) {
            boost::system::error_code ec{static_cast<int>(::ERR_get_error()),
                boost::asio::error::get_ssl_category()};
            std::cerr << "[AlpacaFeed] SNI error: " << ec.message() << std::endl;
        }
        ws->next_layer().handshake(boost::asio::ssl::stream_base::client);

        boost::system::error_code ec;
        ws->handshake("stream.data.alpaca.markets", "/v2/iex", ec);
        if (ec) return;

        // Authenticate with Alpaca
        nlohmann::json auth;
        auth["action"] = "auth";
        auth["key"] = api_key_;
        auth["secret"] = api_secret_;
        ws->write(boost::asio::buffer(auth.dump()));

        // Subscribe to quote streams
        nlohmann::json sub;
        sub["action"] = "subscribe";
        sub["quotes"] = nlohmann::json::array();
        for (const auto& sym : symbols_) sub["quotes"].push_back(sym);
        ws->write(boost::asio::buffer(sub.dump()));

        reconnect_attempts_ = 0;
        ws_ = std::move(ws);

        boost::beast::flat_buffer buffer;
        while (running_.load() && ws_->is_open()) {
            ws_->read(buffer);
            std::string msg = boost::beast::buffers_to_string(buffer.data());
            buffer.consume(buffer.size());
            process_message(msg);
        }
        ws_.reset();
    }

    void process_message(const std::string& message) {
        try {
            auto json = nlohmann::json::parse(message);
            if (json.is_array()) {
                for (const auto& entry : json) {
                    if (!entry.is_object()) continue;
                    std::string msg_type = entry.value("T", "");
                    if (msg_type == "q" || msg_type == "Q") {
                        NormalizedOrderBook nb;
                        nb.symbol = entry.value("S", "UNKNOWN");
                        nb.bid_price = entry.value("bx", 0.0);
                        nb.bid_qty = entry.value("bs", 0.0);
                        nb.ask_price = entry.value("ax", 0.0);
                        nb.ask_qty = entry.value("as", 0.0);

                        std::string t_str = entry.value("t", "");
                        if (!t_str.empty()) {
                            // Parse RFC 3339 timestamp
                            // Example: 2024-01-01T00:00:00.000000Z
                            try {
                                std::tm tm = {};
                                std::istringstream ss(t_str);
                                ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
                                auto tp = std::chrono::system_clock::from_time_t(
                                    timegm(&tm));
                                auto dur = tp.time_since_epoch();
                                nb.timestamp_ms = std::chrono::duration_cast<
                                    std::chrono::milliseconds>(dur).count();
                            } catch (...) {
                                nb.timestamp_ms = now_ms();
                            }
                        } else {
                            nb.timestamp_ms = now_ms();
                        }

                        if (on_book_) on_book_(nb);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[AlpacaFeed] parse error: " << e.what() << std::endl;
        }
    }

    void reconnect() {
        if (!running_.load()) return;
        int delay = 1000 * (1 << std::min(reconnect_attempts_++, 6));
        delay = std::max(100, delay + (std::rand() % (delay / 4)) - (delay / 8));
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    std::vector<std::string> symbols_;
    std::string api_key_;
    std::string api_secret_;
    BookCB on_book_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::unique_ptr<WSS> ws_;
    int reconnect_attempts_ = 0;
};

} // anonymous namespace

// ─── Factory functions ───────────────────────────────────────────────────────

std::unique_ptr<IMarketFeedHandler> create_binance_feed(
    const std::vector<std::string>& symbols) {
    return std::make_unique<BinanceFeed>(symbols);
}

std::unique_ptr<IMarketFeedHandler> create_alpaca_feed(
    const std::vector<std::string>& symbols) {
    return std::make_unique<AlpacaFeed>(symbols,
        AlpacaFeed::alpaca_key(),
        AlpacaFeed::alpaca_secret());
}

} // namespace crossflux
