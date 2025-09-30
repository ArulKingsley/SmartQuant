# environment/trading_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd

class EnhancedTradingEnv(gym.Env):
    """
    Gym-like trading environment with risk controls:
      - transaction cost (pct)
      - slippage (pct)
      - stop-loss (pct) per trade (relative to entry)
      - max drawdown cap (pct)
      - position sizing (fractional or fixed shares)
    Observation: selected feature columns from a DataFrame (row t).
    Action (discrete variant): 0=Hold, 1=Buy (enter/raise long), 2=Sell (enter/raise short), 3=Close position
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list = None,
        price_col: str = "Close",
        initial_balance: float = 10000.0,
        max_position_size: float = 1.0,           # fraction of balance used for position (if using fraction sizing)
        fixed_lot_size: int = None,               # if provided, use fixed shares per trade
        transaction_cost_pct: float = 0.0005,     # 0.05% per trade
        slippage_pct: float = 0.0005,             # 0.05% slippage
        stop_loss_pct: float = 0.03,              # 3% stop-loss default
        max_drawdown_pct: float = 0.2,            # 20% max drawdown
        reward_scale: float = 1e-2,               # scale down reward magnitudes
        include_lstm_col: str = None,             # optional column name for LSTM forecast
        window_size: int = 1                      # if you want stacked obs, not used here usually
    ):
        super().__init__()

        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        self.df = df.reset_index(drop=True)
        self.price_col = price_col
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.max_position_size = max_position_size
        self.fixed_lot_size = fixed_lot_size
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.reward_scale = reward_scale
        self.include_lstm_col = include_lstm_col
        self.window_size = window_size

        # default features: if not provided, use all numeric columns except price_col
        if feature_columns is None:
            self.feature_columns = [c for c in df.columns if c != price_col]
        else:
            self.feature_columns = feature_columns

        # build observation space: vector length = len(feature_columns)
        obs_dim = len(self.feature_columns)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # discrete action space with Close action as well
        self.action_space = spaces.Discrete(4)  # 0:Hold,1:Buy,2:Sell,3:Close

        # runtime variables
        self.current_step = 0
        self.position = 0.0         # positive = long number of shares, negative = short
        self.entry_price = None
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.trade_count = 0
        self.last_trade_step = None

        # bookkeeping arrays for analysis
        self.history = []

    def _get_obs(self):
        # return features at current step as 1D np array
        row = self.df.iloc[self.current_step]
        obs = row[self.feature_columns].values.astype(np.float32)
        return obs

    def _get_price(self, step=None):
        if step is None:
            step = self.current_step
        return float(self.df.iloc[step][self.price_col])

    def _calc_max_affordable_shares(self, price):
        if self.fixed_lot_size is not None:
            return int(self.fixed_lot_size)  # fixed shares
        # fraction-of-balance sizing (use max_position_size fraction of current portfolio)
        available = self.portfolio_value * self.max_position_size
        return int(available // (price + 1e-9))  # integer shares

    def _apply_trade(self, action):
        """
        Executes the chosen action at current step -> moves position/cash.
        Returns executed_trade_pnl (immediate PnL impact, negative for cost).
        """
        price = self._get_price()
        executed_pnl = 0.0
        trade_info = None

        # choose size in shares
        shares = self._calc_max_affordable_shares(price)
        if shares <= 0:
            return 0.0, None  # can't buy any shares due to insufficient funds

        # slippage applied to executed price
        if action == 1:  # Buy: buy at price * (1 + slippage)
            exec_price = price * (1 + self.slippage_pct)
            cost = shares * exec_price
            fee = cost * self.transaction_cost_pct
            # if currently short, closing part or reversing must be handled
            if self.position < 0:
                # close short first partially or fully
                close_shares = min(abs(self.position), shares)
                # realize PnL for closed portion
                pnl_close = (self.entry_price - exec_price) * close_shares
                executed_pnl += pnl_close
                # update position
                self.position += close_shares
                shares -= close_shares
                if abs(self.position) < 1e-8:
                    self.entry_price = None
                # if shares remain, open long with remaining
            if shares > 0:
                # open/increase long
                self.position += shares
                # set/adjust entry price (weighted avg)
                if self.entry_price is None:
                    self.entry_price = exec_price
                else:
                    # weighted avg entry price
                    prev_pos = max(self.position - shares, 0)
                    if prev_pos <= 0:
                        self.entry_price = exec_price
                    else:
                        self.entry_price = (self.entry_price * prev_pos + exec_price * shares) / (prev_pos + shares)
                # deduct cash
                self.balance -= (shares * exec_price + fee)
                executed_pnl -= fee
            self.trade_count += 1
            trade_info = {"type": "buy", "exec_price": exec_price, "shares": shares}

        elif action == 2:  # Sell (enter short)
            exec_price = price * (1 - self.slippage_pct)
            proceeds = shares * exec_price
            fee = proceeds * self.transaction_cost_pct
            if self.position > 0:
                # close long first partially or fully
                close_shares = min(self.position, shares)
                pnl_close = (exec_price - self.entry_price) * close_shares
                executed_pnl += pnl_close
                self.position -= close_shares
                shares -= close_shares
                if abs(self.position) < 1e-8:
                    self.entry_price = None
            if shares > 0:
                # open/increase short
                self.position -= shares
                if self.entry_price is None:
                    self.entry_price = exec_price
                else:
                    prev_pos = min(self.position + shares, 0)  # negative or zero
                    # approximate weighted average for shorts (keep sign logic simple)
                    self.entry_price = (self.entry_price * abs(prev_pos) + exec_price * shares) / (abs(prev_pos) + shares)
                # receive proceeds to balance (we allow margin trading in simplified way)
                self.balance += (proceeds - fee)
                executed_pnl -= fee
            self.trade_count += 1
            trade_info = {"type": "sell", "exec_price": exec_price, "shares": shares}

        elif action == 3:  # Close position (flatten)
            if self.position == 0:
                return 0.0, None
            exec_price = price * (1 + self.slippage_pct) if self.position < 0 else price * (1 - self.slippage_pct)
            closed_shares = abs(self.position)
            if self.position > 0:
                pnl = (exec_price - self.entry_price) * closed_shares
                self.balance += (closed_shares * exec_price) - (closed_shares * exec_price * self.transaction_cost_pct)
                executed_pnl += pnl - (closed_shares * exec_price * self.transaction_cost_pct)
            else:
                # closing short: we buy back
                pnl = (self.entry_price - exec_price) * closed_shares
                self.balance -= (closed_shares * exec_price) + (closed_shares * exec_price * self.transaction_cost_pct)
                executed_pnl += pnl - (closed_shares * exec_price * self.transaction_cost_pct)
            # reset position
            self.position = 0
            self.entry_price = None
            self.trade_count += 1
            trade_info = {"type": "close", "exec_price": exec_price, "shares": closed_shares}
        else:
            # Hold -> no trade
            trade_info = None

        return executed_pnl, trade_info

    def _check_stoploss_and_drawdown(self):
        """
        Check stop-loss per position and global drawdown.
        If stop-loss hits, force close and apply penalty.
        If drawdown exceeds max, apply penalty and optionally end episode.
        """
        price = self._get_price()
        penalty = 0.0
        forced_action = None

        # per-trade stop-loss
        if self.entry_price is not None and self.position != 0:
            if self.position > 0:
                # long stop-loss
                if (price - self.entry_price) / (self.entry_price + 1e-9) <= -self.stop_loss_pct:
                    # force close
                    forced_action = 3
                    penalty += -0.5 * abs(self.position)  # custom penalty scale
            else:
                # short stop-loss
                if (self.entry_price - price) / (self.entry_price + 1e-9) <= -self.stop_loss_pct:
                    forced_action = 3
                    penalty += -0.5 * abs(self.position)

        # global drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        drawdown = 1.0 - self.portfolio_value / (self.peak_portfolio_value + 1e-9)
        if drawdown >= self.max_drawdown_pct:
            # emergency: flatten & penalize
            forced_action = 3
            penalty += -1.0 * (drawdown / (self.max_drawdown_pct + 1e-9))

        return forced_action, penalty

    def step(self, action):
        """
        action: int in {0,1,2,3}
        returns: obs, reward, done, info
        """
        done = False
        info = {}
        # Execute requested trade
        exec_pnl, trade_info = self._apply_trade(action)

        # Advance time
        prev_value = self.portfolio_value
        price = self._get_price()  # price at current step AFTER potential trade
        # Value of holdings (simplified: position * price) + cash balance
        holdings_value = self.position * price
        self.portfolio_value = self.balance + holdings_value

        # Compute immediate reward = change in portfolio value + exec_pnl (fees already included)
        reward = (self.portfolio_value - prev_value) * self.reward_scale

        # Check stop-loss and drawdown forced closures
        forced_action, penalty = self._check_stoploss_and_drawdown()
        if forced_action is not None:
            # force close position now
            _, forced_info = self._apply_trade(3)
            # recompute portfolio value after forced close
            price2 = self._get_price()
            self.portfolio_value = self.balance + self.position * price2
            # apply additional penalty
            reward += penalty

        # Overtrading penalty (discourage excessive trades)
        if self.trade_count > 0 and (self.trade_count / max(1, (self.current_step+1))) > 0.05:
            # if trade frequency > 5% of steps, penalize
            reward += -0.1

        # Step forward time
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        # Info for logging
        info['portfolio_value'] = self.portfolio_value
        info['balance'] = self.balance
        info['position'] = self.position
        info['entry_price'] = self.entry_price
        info['trade_info'] = trade_info

        # record history
        self.history.append({
            "step": self.current_step,
            "portfolio_value": float(self.portfolio_value),
            "position": float(self.position),
            "balance": float(self.balance),
            "reward": float(reward)
        })

        return obs, float(reward), done, info

    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.entry_price = None
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.trade_count = 0
        self.last_trade_step = None
        self.history = []
        return self._get_obs()

    def render(self, mode='human'):
        last = self.history[-1] if len(self.history) else {}
        print(f"Step: {self.current_step} PV: {self.portfolio_value:.2f} Pos: {self.position} Bal: {self.balance:.2f}")

