#pragma once

#include <string>
#include <iostream>
#include <memory>
#include "models.hpp"
#include "risk.hpp"
#include "execution_manager.hpp"

namespace crossflux {

enum class OrderType {
    IOC,  // Immediate-Or-Cancel
    FOK   // Fill-Or-Kill
};

struct ExecutionResult {
    bool success;
    double fill_price;
    double fill_qty;
    std::string error_message;
};

/**
 * @brief Base interface for parallel order dispatching.
 */
class OrderDispatcher {
public:
    explicit OrderDispatcher(std::shared_ptr<CircuitBreaker> risk_mgr) 
        : risk_mgr_(std::move(risk_mgr)) {}

    virtual ~OrderDispatcher() = default;

    virtual bool execute(const DispatchSignal& ds) noexcept = 0;

    virtual ExecutionResult execute_buy(
        const std::string& exchange, 
        double price, 
        double qty, 
        OrderType type) noexcept = 0;

    virtual ExecutionResult execute_sell(
        const std::string& exchange, 
        double price, 
        double qty, 
        OrderType type) noexcept = 0;

protected:
    std::shared_ptr<CircuitBreaker> risk_mgr_;
};

/**
 * @brief Simulated execution with risk management and realistic fills.
 */
class SimulatedOrderDispatcher : public OrderDispatcher {
public:
    SimulatedOrderDispatcher(
        std::shared_ptr<CircuitBreaker> risk_mgr,
        std::string_view exchange_a,
        std::string_view exchange_b,
        double max_position = 5.0,
        double cooldown_ms = 10000.0,
        double min_profit_bps = 2.0,
        double slippage_model_bps = 0.5
    );

    bool execute(const DispatchSignal& ds) noexcept override;

    ExecutionResult execute_buy(
        const std::string& exchange, 
        double price, 
        double qty, 
        OrderType type) noexcept override;

    ExecutionResult execute_sell(
        const std::string& exchange, 
        double price, 
        double qty, 
        OrderType type) noexcept override;

    double current_position() const noexcept;
    double realized_pnl() const noexcept;
    int total_orders() const noexcept;
    int filled_orders() const noexcept;
    uint64_t last_trade_ms() const noexcept;
    double max_position_limit() const noexcept;
    double cooldown_ms() const noexcept;
    double min_profit_bps() const noexcept;
    bool is_cooldown_active(uint64_t now_ms) const noexcept;

private:
    SimulatedExecutor executor_;
};

} // namespace crossflux
