#pragma once

#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include "ingestion_engine.hpp"
#include "models.hpp"

namespace crossflux {

/**
 * @brief Base class for exchange WebSocket clients
 *
 * Handles connection, reconnection, and message parsing for exchange WebSocket streams.
 * Derived classes implement exchange-specific message handling.
 */
class WebSocketClientBase {
public:
    WebSocketClientBase(
        std::shared_ptr<crossflux::IngestionEngine> ingestion_engine,
        const std::string& exchange_name,
        const std::vector<std::string>& symbols)
        : ingestion_engine_(std::move(ingestion_engine)),
          exchange_name_(exchange_name),
          symbols_(symbols),
          running_(false),
          reconnect_attempts_(0),
          max_reconnect_attempts_(10),
          base_reconnect_delay_ms_(1000),
          max_reconnect_delay_ms_(30000)
    {}

    virtual ~WebSocketClientBase() noexcept {
        running_.store(false);
        if (ws_) {
            boost::system::error_code ec;
            boost::beast::get_lowest_layer(*ws_).close(ec);
        }
        if (worker_thread_.joinable()) worker_thread_.detach();
    }

    /** Start the WebSocket client */
    void start() {
        if (!running_.exchange(true)) {
            worker_thread_ = std::thread(&WebSocketClientBase::run_loop, this);
        }
    }

    /** Stop the WebSocket client */
    void stop() {
        running_.store(false);
        if (ws_) {
            boost::system::error_code ec;
            boost::beast::get_lowest_layer(*ws_).close(ec);
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (ws_) {
            boost::system::error_code ec;
            if (ws_->is_open()) ws_->close(boost::beast::websocket::close_code::normal, ec);
        }
    }

protected:
    /** Main run loop with reconnection logic */
    void run_loop() {
        while (running_.load()) {
            try {
                connect_and_run();
                if (!running_.load()) break;
                schedule_reconnect();
            } catch (const std::exception& e) {
                if (running_.load()) {
                    std::cerr << "[" << exchange_name_ << "] Error: " << e.what() << std::endl;
                }
                schedule_reconnect();
            }
        }
    }

    /** Load CA certificates from common system paths */
    void load_system_ca_certs() {
        // Common CA certificate paths on macOS/Linux
        const std::vector<std::string> ca_paths = {
            "/opt/homebrew/etc/openssl@3/cert.pem",
            "/opt/homebrew/etc/openssl/cert.pem",
            "/usr/local/etc/openssl@3/cert.pem",
            "/usr/local/etc/openssl/cert.pem",
            "/etc/ssl/cert.pem",
            "/etc/ssl/certs/ca-certificates.crt",
            "/Library/Frameworks/Python.framework/Versions/Current/etc/openssl/cert.pem"
        };
        bool loaded = false;
        for (const auto& path : ca_paths) {
            std::ifstream f(path);
            if (f.good()) {
                boost::system::error_code ec;
                ssl_ctx_.load_verify_file(path, ec);
                if (!ec) {
                    loaded = true;
                    break;
                }
            }
        }
        if (!loaded) {
            ssl_ctx_.set_default_verify_paths();
        }
    }

    /** Attempt to connect and run the WebSocket */
    virtual void connect_and_run() = 0;

    /** Schedule reconnection with exponential backoff */
    void schedule_reconnect() {
        if (!running_.load() || ++reconnect_attempts_ > max_reconnect_attempts_) {
            running_.store(false);
            return;
        }

        // Exponential backoff with jitter
        int delay_ms = base_reconnect_delay_ms_ * (1 << (reconnect_attempts_ - 1));
        delay_ms = std::min(delay_ms, max_reconnect_delay_ms_);

        // Add jitter (±25%)
        int jitter = (std::rand() % (std::max(1, delay_ms / 2))) - (delay_ms / 4);
        delay_ms = std::max(100, delay_ms + jitter);

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    /** Send a ping to keep connection alive */
    void send_ping() {
        if (ws_ && ws_->is_open()) {
            try {
                ws_->ping({});
            } catch (...) {
                // Ignore ping errors - connection will be detected as dead on next read
            }
        }
    }

    /** Convert exchange WebSocket message to MarketTick */
    virtual void process_message(const std::string& message) = 0;

    /** Parse order book data into OrderBookSnapshot */
    virtual crossflux::OrderBookSnapshot<10> parse_order_book(
        const nlohmann::json& data,
        uint64_t timestamp_ms,
        const std::string& exchange_id) = 0;

    // Member variables
    std::shared_ptr<crossflux::IngestionEngine> ingestion_engine_;
    std::string exchange_name_;
    std::vector<std::string> symbols_;

    std::atomic<bool> running_;
    std::thread worker_thread_;

    boost::asio::io_context io_context_;
    boost::asio::ssl::context ssl_ctx_{boost::asio::ssl::context::tls_client};
    std::unique_ptr<boost::beast::websocket::stream<boost::asio::ssl::stream<boost::asio::ip::tcp::socket>>> ws_;

    int reconnect_attempts_;
    const int max_reconnect_attempts_;
    const int base_reconnect_delay_ms_;
    const int max_reconnect_delay_ms_;
};

/**
 * @brief Binance WebSocket client for depth streams
 *
 * Connects to Binance's combined stream for multiple symbols:
 * wss://stream.binance.com:9443/stream?streams=btcusdt@depth5@100ms/ethusdt@depth5@100ms
 */
class BinanceWebSocketClient : public WebSocketClientBase {
public:
    BinanceWebSocketClient(
        std::shared_ptr<crossflux::IngestionEngine> ingestion_engine,
        const std::vector<std::string>& symbols)
        : WebSocketClientBase(std::move(ingestion_engine), "binance", symbols) {}

protected:
    void connect_and_run() override {
        load_system_ca_certs();

        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto const results = resolver.resolve("stream.binance.com", "9443");

        ws_ = std::make_unique<boost::beast::websocket::stream<boost::asio::ssl::stream<boost::asio::ip::tcp::socket>>>(io_context_, ssl_ctx_);

        // Set User-Agent and other headers for the WebSocket handshake
        ws_->set_option(boost::beast::websocket::stream_base::decorator(
            [](boost::beast::websocket::request_type& req) {
                req.set(boost::beast::http::field::user_agent,
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)");
            }));

        ws_->set_option(boost::beast::websocket::stream_base::timeout::suggested(
            boost::beast::role_type::client));

        boost::beast::get_lowest_layer(*ws_).connect(*results.begin());

        // Set SNI hostname for TLS
        if(!SSL_set_tlsext_host_name(ws_->next_layer().native_handle(), "stream.binance.com")) {
            boost::system::error_code ec{static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category()};
            std::cerr << "[BinanceWebSocket] SNI error: " << ec.message() << std::endl;
        }

        ws_->next_layer().handshake(boost::asio::ssl::stream_base::client);

        std::string url = "/ws/";
        bool first = true;
        for (const auto& symbol : symbols_) {
            if (!first) break;
            url += symbol + "@depth5@100ms";
            first = false;
        }
        if (symbols_.empty()) {
            url = "/ws/btcusdt@depth5@100ms";
        }

        boost::system::error_code ec;
        ws_->handshake("stream.binance.com", url, ec);
        if (ec) {
            // Check if this is an HTTP error (handshake failed due to HTTP response)
            if (ec.category() == boost::asio::error::get_system_category()) {
                // This might be a system error
                std::cerr << "[WebSocketClientBase] Handshake failed (system error): "
                          << ec.message() << " (" << ec.value() << ")" << std::endl;
            } else {
                // This is likely a WebSocket-specific error
                std::cerr << "[WebSocketClientBase] Handshake failed (WebSocket error): "
                          << ec.message() << " (" << ec.value() << ")" << std::endl;

                // Try to get more details if available
                if (ec.value() == 20) { // websocket::error::bad_handshake
                    std::cerr << "[WebSocketClientBase]   -> Bad handshake: Server did not upgrade to WebSocket"
                              << std::endl;
                }
            }
            return;
        }

        // Subscribe to streams
        nlohmann::json subscribe_msg;
        subscribe_msg["method"] = "SUBSCRIBE";
        nlohmann::json params;
        for (const auto& symbol : symbols_) {
            params.push_back(symbol + "@depth5@100ms");
        }
        subscribe_msg["params"] = params;
        subscribe_msg["id"] = 1;

        ws_->write(boost::asio::buffer(subscribe_msg.dump()));

        // Reset reconnect counter on successful connection
        reconnect_attempts_ = 0;

        // Buffer for incoming messages
        boost::beast::flat_buffer buffer;

        // Read messages until connection is closed
        while (running_.load() && ws_->is_open()) {
            ws_->read(buffer);
            std::string message = boost::beast::buffers_to_string(buffer.data());
            buffer.consume(buffer.size());

            process_message(message);
        }
    }

    void process_message(const std::string& message) override {
        try {
            auto json = nlohmann::json::parse(message);

            // Binance raw stream: {"bids":[...],"asks":[...],"lastUpdateId":123,...}
            // Binance combined/diff stream: {"stream":"...","data":{...}} or {"b":[...],"a":[...]}
            const auto& data = json.contains("data") ? json["data"] : json;
            std::string symbol = "btcusdt";
            if (json.contains("stream")) {
                std::string stream_name = json["stream"];
                symbol = stream_name.substr(0, stream_name.find('@'));
            }

            // Support both formats: "bids"/"asks" (partial depth) and "b"/"a" (diff depth)
            const bool has_bids = data.contains("bids") && data.contains("asks");
            const bool has_diff = data.contains("b") && data.contains("a");
            if (has_bids || has_diff) {
                const auto& bids_json = has_bids ? data["bids"] : data["b"];
                const auto& asks_json = has_bids ? data["asks"] : data["a"];
                uint64_t timestamp_ms = data.value("E",
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count());

                std::vector<crossflux::PriceLevel> bids;
                for (const auto& bid : bids_json) {
                    if (bid.is_array() && bid.size() >= 2) {
                        double price = bid[0].is_string()
                            ? std::stod(bid[0].get<std::string>())
                            : bid[0].get<double>();
                        double volume = bid[1].is_string()
                            ? std::stod(bid[1].get<std::string>())
                            : bid[1].get<double>();
                        if (price > 0 && volume >= 0) {
                            bids.emplace_back(price, volume);
                        }
                    }
                }

                std::vector<crossflux::PriceLevel> asks;
                for (const auto& ask : asks_json) {
                    if (ask.is_array() && ask.size() >= 2) {
                        double price = ask[0].is_string()
                            ? std::stod(ask[0].get<std::string>())
                            : ask[0].get<double>();
                        double volume = ask[1].is_string()
                            ? std::stod(ask[1].get<std::string>())
                            : ask[1].get<double>();
                        if (price > 0 && volume >= 0) {
                            asks.emplace_back(price, volume);
                        }
                    }
                }

                std::sort(bids.begin(), bids.end(),
                    [](const crossflux::PriceLevel& a, const crossflux::PriceLevel& b) {
                        return a.price > b.price;
                    });
                std::sort(asks.begin(), asks.end(),
                    [](const crossflux::PriceLevel& a, const crossflux::PriceLevel& b) {
                        return a.price < b.price;
                    });

                if (bids.size() > 10) bids.resize(10);
                if (asks.size() > 10) asks.resize(10);

                auto snap_a = create_snapshot_from_levels(bids, asks, timestamp_ms, "binance");
                auto snap_b = create_snapshot_from_levels(bids, asks, timestamp_ms, "binance");
                auto tick = crossflux::make_market_tick(timestamp_ms, snap_a, snap_b);
                static std::atomic<long long> binance_tick_count{0};
                if (++binance_tick_count % 10 == 0)
                    std::cout << "[Binance] Tick #" << binance_tick_count << " | bids=" << bids.size() << " asks=" << asks.size() << "\n";
                ingestion_engine_->push_tick(tick);
                static auto last_binance_price = std::chrono::steady_clock::now();
                auto now = std::chrono::steady_clock::now();
                if (now - last_binance_price > std::chrono::milliseconds(500)) {
                    last_binance_price = now;
                    std::ofstream pf("/tmp/live_prices.csv", std::ios::app);
                    if (pf.tellp() == 0) pf << "timestamp_ms,exchange,bid_price,bid_vol,ask_price,ask_vol\n";
                    pf << timestamp_ms << ",binance,"
                       << (bids.empty() ? 0 : bids[0].price) << ","
                       << (bids.empty() ? 0 : bids[0].volume) << ","
                       << (asks.empty() ? 0 : asks[0].price) << ","
                       << (asks.empty() ? 0 : asks[0].volume) << "\n";
                    std::ofstream df("/tmp/live_depth.csv", std::ios::app);
                    if (df.tellp() == 0) df << "timestamp_ms,exchange,side,level,price,volume\n";
                    for (size_t i = 0; i < bids.size() && i < 5; ++i)
                        df << timestamp_ms << ",binance,0," << i << "," << bids[i].price << "," << bids[i].volume << "\n";
                    for (size_t i = 0; i < asks.size() && i < 5; ++i)
                        df << timestamp_ms << ",binance,1," << i << "," << asks[i].price << "," << asks[i].volume << "\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[BinanceWebSocket] JSON parse error: " << e.what() << std::endl;
        }
    }

    crossflux::OrderBookSnapshot<10> create_snapshot_from_levels(
        const std::vector<crossflux::PriceLevel>& bids,
        const std::vector<crossflux::PriceLevel>& asks,
        uint64_t timestamp_ms,
        const std::string& exchange_id) {
        crossflux::OrderBookSnapshot<10> snap{};
        snap.timestamp_ms = timestamp_ms;
        std::memset(snap.exchange_id, 0, sizeof(snap.exchange_id));
        std::memcpy(snap.exchange_id, exchange_id.c_str(),
                   std::min(exchange_id.size(), sizeof(snap.exchange_id)-1));
        snap.exchange_id[sizeof(snap.exchange_id)-1] = '\0';

        // Fill bids (descending order)
        for (size_t i = 0; i < bids.size() && i < 10; ++i) {
            snap.bids[i] = bids[i];
        }
        snap.bid_depth = static_cast<uint8_t>(std::min(bids.size(), size_t(10)));

        // Fill asks (ascending order)
        for (size_t i = 0; i < asks.size() && i < 10; ++i) {
            snap.asks[i] = asks[i];
        }
        snap.ask_depth = static_cast<uint8_t>(std::min(asks.size(), size_t(10)));

        return snap;
    }

    crossflux::OrderBookSnapshot<10> parse_order_book(
        const nlohmann::json& data,
        uint64_t timestamp_ms,
        const std::string& exchange_id) override {
        return create_snapshot_from_levels({}, {}, timestamp_ms, exchange_id);
    }

};

/**
 * @brief Kraken WebSocket client for order book streams
 *
 * Connects to Kraken's WebSocket API for book streams
 */
class KrakenWebSocketClient : public WebSocketClientBase {
public:
    KrakenWebSocketClient(
        std::shared_ptr<crossflux::IngestionEngine> ingestion_engine,
        const std::vector<std::string>& symbols)
        : WebSocketClientBase(std::move(ingestion_engine), "kraken", symbols) {
        // Convert symbols to Kraken format (e.g., BTC/USD)
        for (const auto& symbol : symbols) {
            // Simple conversion - in practice you'd need proper symbol mapping
            if (symbol == "btcusd" || symbol == "btcusdt") {
                kraken_symbols_.push_back("XBT/USD");
            } else if (symbol == "ethusd" || symbol == "ethusdt") {
                kraken_symbols_.push_back("ETH/USD");
            } else {
                kraken_symbols_.push_back(symbol);  // Fallback
            }
        }
    }

protected:
    void connect_and_run() override {
        load_system_ca_certs();

        boost::asio::ip::tcp::resolver resolver(io_context_);
        auto const results = resolver.resolve("ws.kraken.com", "443");

        ws_ = std::make_unique<boost::beast::websocket::stream<boost::asio::ssl::stream<boost::asio::ip::tcp::socket>>>(io_context_, ssl_ctx_);

        ws_->set_option(boost::beast::websocket::stream_base::decorator(
            [](boost::beast::websocket::request_type& req) {
                req.set(boost::beast::http::field::user_agent,
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)");
            }));

        ws_->set_option(boost::beast::websocket::stream_base::timeout::suggested(
            boost::beast::role_type::client));

        boost::beast::get_lowest_layer(*ws_).connect(*results.begin());

        if(!SSL_set_tlsext_host_name(ws_->next_layer().native_handle(), "ws.kraken.com")) {
            boost::system::error_code ec{static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category()};
            std::cerr << "[KrakenWebSocket] SNI error: " << ec.message() << std::endl;
        }

        ws_->next_layer().handshake(boost::asio::ssl::stream_base::client);

        boost::system::error_code ec;
        ws_->handshake("ws.kraken.com", "/", ec);
        if (ec) {
            std::cerr << "[WebSocketClientBase] Handshake failed: "
                      << ec.message() << " (" << ec.value() << ")" << std::endl;
            return;
        }

        // Subscribe to book streams
        nlohmann::json subscribe_msg;
        subscribe_msg["event"] = "subscribe";
        subscribe_msg["pair"] = nlohmann::json::array();
        for (const auto& symbol : kraken_symbols_) {
            subscribe_msg["pair"].push_back(symbol);
        }
        subscribe_msg["subscription"] = {{"name", "book"}, {"depth", 10}};
        subscribe_msg["reqid"] = 1;

        ws_->write(boost::asio::buffer(subscribe_msg.dump()));

        // Reset reconnect counter on successful connection
        reconnect_attempts_ = 0;

        // Buffer for incoming messages
        boost::beast::flat_buffer buffer;

        // Read messages until connection is closed
        while (running_.load() && ws_->is_open()) {
            ws_->read(buffer);
            std::string message = boost::beast::buffers_to_string(buffer.data());
            buffer.consume(buffer.size());

            process_message(message);
        }
    }

    void process_message(const std::string& message) override {
        try {
            auto json = nlohmann::json::parse(message);

            // Kraken v2 message formats:
            //   [channelID, {book_data}, "channelName", "pair"]  (snapshot)
            //   [channelID, {},            {update_data}, "pair"] (incremental)
            //   [channelID, {book_data},   "channelName", "pair"] (book data)
            if (json.is_array() && json.size() >= 4) {
                std::string pair = json[3];

                // Determine which element contains the book data
                if (json[2].is_string()) {
                    // Format: [channelID, data, "channelName", "pair"]
                    auto channel_data = json[1];
                    process_book_update(pair, channel_data);
                } else if (json[2].is_object()) {
                    // Format: [channelID, {}, {update_data}, "pair"]
                    auto channel_data = json[2];
                    process_book_update(pair, channel_data);
                }
            }
            // Handle subscription status messages
            else if (json.contains("event")) {
                std::string event = json["event"];
                if (event == "subscriptionStatus") {
                    std::cout << "[KrakenWebSocket] Subscription status: "
                              << json.dump(2) << std::endl;
                } else if (event == "heartbeat") {
                    // Ignore heartbeat
                } else if (event == "systemStatus") {
                    std::cout << "[KrakenWebSocket] System status: "
                              << json.dump(2) << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[KrakenWebSocket] JSON parse error: " << e.what() << std::endl;
        }
    }

    void process_book_update(const std::string& pair, const nlohmann::json& data) {
        // Kraken book update format:
        // [
        //   CHANNEL_ID,
        //   {
        //     "as": [["ask_price", "ask_volume", "timestamp"], ...],
        //     "bs": [["bid_price", "bid_volume", "timestamp"], ...],
        //     "a": [["ask_price", "ask_volume", "timestamp"], ...],  // asks
        //     "b": [["bid_price", "bid_volume", "timestamp"], ...]   // bids
        //   },
        //   "book",
        //   "XBT/USD"
        // ]

        if (!data.is_object()) {
            return;
        }

        uint64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        // Process bids
        std::vector<crossflux::PriceLevel> bids;
        std::vector<crossflux::PriceLevel> asks;

        // Process bids (bids array)
        if (data.contains("b")) {
            for (const auto& bid_level : data["b"]) {
                if (bid_level.is_array() && bid_level.size() >= 2) {
                    double price = bid_level[0].is_string()
                        ? std::stod(bid_level[0].get<std::string>())
                        : bid_level[0].get<double>();
                    double volume = bid_level[1].is_string()
                        ? std::stod(bid_level[1].get<std::string>())
                        : bid_level[1].get<double>();
                    if (price > 0 && volume >= 0) {
                        bids.emplace_back(price, volume);
                    }
                }
            }
        }

        // Process asks (a array)
        if (data.contains("a")) {
            for (const auto& ask_level : data["a"]) {
                if (ask_level.is_array() && ask_level.size() >= 2) {
                    double price = ask_level[0].is_string()
                        ? std::stod(ask_level[0].get<std::string>())
                        : ask_level[0].get<double>();
                    double volume = ask_level[1].is_string()
                        ? std::stod(ask_level[1].get<std::string>())
                        : ask_level[1].get<double>();
                    if (price > 0 && volume >= 0) {
                        asks.emplace_back(price, volume);
                    }
                }
            }
        }

        // Sort bids descending (highest first)
        std::sort(bids.begin(), bids.end(),
                 [](const crossflux::PriceLevel& a, const crossflux::PriceLevel& b) {
                     return a.price > b.price;
                 });

        // Sort asks ascending (lowest first)
        std::sort(asks.begin(), asks.end(),
                 [](const crossflux::PriceLevel& a, const crossflux::PriceLevel& b) {
                     return a.price < b.price;
                 });

        // Take top 10 levels
        if (bids.size() > 10) bids.resize(10);
        if (asks.size() > 10) asks.resize(10);

        // Create snapshots for this exchange
        auto snap_a = create_snapshot_from_levels(bids, asks, timestamp_ms, "kraken");

        // For demonstration, create second identical snapshot
        auto snap_b = create_snapshot_from_levels(bids, asks, timestamp_ms, "kraken");

        // Create MarketTick and push to ingestion engine
        auto tick = crossflux::make_market_tick(timestamp_ms, snap_a, snap_b);
        static std::atomic<long long> kraken_tick_count{0};
        if (++kraken_tick_count % 10 == 0)
            std::cout << "[Kraken] Tick #" << kraken_tick_count << " | bids=" << bids.size() << " asks=" << asks.size() << "\n";
        ingestion_engine_->push_tick(tick);
        static auto last_kraken_price = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now - last_kraken_price > std::chrono::milliseconds(500)) {
            last_kraken_price = now;
            std::ofstream pf("/tmp/live_prices.csv", std::ios::app);
            if (pf.tellp() == 0) pf << "timestamp_ms,exchange,bid_price,bid_vol,ask_price,ask_vol\n";
            pf << timestamp_ms << ",kraken,"
               << (bids.empty() ? 0 : bids[0].price) << ","
               << (bids.empty() ? 0 : bids[0].volume) << ","
               << (asks.empty() ? 0 : asks[0].price) << ","
               << (asks.empty() ? 0 : asks[0].volume) << "\n";
            std::ofstream df("/tmp/live_depth.csv", std::ios::app);
            if (df.tellp() == 0) df << "timestamp_ms,exchange,side,level,price,volume\n";
            for (size_t i = 0; i < bids.size() && i < 5; ++i)
                df << timestamp_ms << ",kraken,0," << i << "," << bids[i].price << "," << bids[i].volume << "\n";
            for (size_t i = 0; i < asks.size() && i < 5; ++i)
                df << timestamp_ms << ",kraken,1," << i << "," << asks[i].price << "," << asks[i].volume << "\n";
        }
    }

    crossflux::OrderBookSnapshot<10> create_snapshot_from_levels(
        const std::vector<crossflux::PriceLevel>& bids,
        const std::vector<crossflux::PriceLevel>& asks,
        uint64_t timestamp_ms,
        const std::string& exchange_id) {
        crossflux::OrderBookSnapshot<10> snap{};
        snap.timestamp_ms = timestamp_ms;
        std::memset(snap.exchange_id, 0, sizeof(snap.exchange_id));
        std::memcpy(snap.exchange_id, exchange_id.c_str(),
                   std::min(exchange_id.size(), sizeof(snap.exchange_id)-1));
        snap.exchange_id[sizeof(snap.exchange_id)-1] = '\0';

        // Fill bids (descending order)
        for (size_t i = 0; i < bids.size() && i < 10; ++i) {
            snap.bids[i] = bids[i];
        }
        snap.bid_depth = static_cast<uint8_t>(std::min(bids.size(), size_t(10)));

        // Fill asks (ascending order)
        for (size_t i = 0; i < asks.size() && i < 10; ++i) {
            snap.asks[i] = asks[i];
        }
        snap.ask_depth = static_cast<uint8_t>(std::min(asks.size(), size_t(10)));

        return snap;
    }

    crossflux::OrderBookSnapshot<10> parse_order_book(
        const nlohmann::json& data,
        uint64_t timestamp_ms,
        const std::string& exchange_id) override {
        return create_snapshot_from_levels({}, {}, timestamp_ms, exchange_id);
    }

private:
    std::vector<std::string> kraken_symbols_;
};

} // namespace crossflux