"""NYMEX WTI (CL) contract calendar and roll-adjusted front-month returns.

Why this matters: "contract 1" price series jump on the day the nearest
contract expires and the next one becomes the front. That jump is the
calendar spread, not a market move - a naive backtest on such a series books
fake PnL every month (huge in steep contango or backwardation).

CME rule for CL: trading terminates 3 business days before the 25th calendar
day of the month prior to the contract month; if the 25th is not a business
day, trading terminates 4 business days before the 25th.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NymexHolidayCalendar(AbstractHolidayCalendar):
    """Approximate CME Globex holiday calendar for energy futures."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


BDAY = CustomBusinessDay(calendar=NymexHolidayCalendar())


def cl_last_trade(year: int, month: int) -> pd.Timestamp:
    """Last trading day of the CL contract for delivery in (year, month)."""
    prior = pd.Period(year=year, month=month, freq="M") - 1
    d25 = pd.Timestamp(prior.year, prior.month, 25)
    n = 3 if BDAY.is_on_offset(d25) else 4
    return (d25 - n * BDAY).normalize()


def last_trade_schedule(start, end) -> pd.Series:
    """Last trade dates of all contracts whose life overlaps [start, end], indexed by delivery month."""
    first = pd.Period(pd.Timestamp(start), freq="M")
    last = pd.Period(pd.Timestamp(end), freq="M") + 3
    months = pd.period_range(first, last, freq="M")
    return pd.Series([cl_last_trade(p.year, p.month) for p in months], index=months, name="last_trade")


def front_contract(dates: pd.DatetimeIndex, schedule: pd.Series | None = None) -> pd.PeriodIndex:
    """Delivery month of contract 1 on each date (the nearest contract not yet expired)."""
    dates = pd.DatetimeIndex(dates).normalize()
    schedule = schedule if schedule is not None else last_trade_schedule(dates.min() - pd.Timedelta(days=40), dates.max())
    pos = np.searchsorted(schedule.to_numpy(dtype="datetime64[ns]"), dates.to_numpy(dtype="datetime64[ns]"), side="left")
    return pd.PeriodIndex(schedule.index[pos], freq="M")


def roll_adjusted_changes(cl1: pd.Series, cl2: pd.Series) -> pd.Series:
    """Daily $/bbl P&L of holding contract 1 and rolling into contract 2 at expiry.

    On the first observation after a roll, yesterday's contract 2 *is* today's
    contract 1, so the change is measured against yesterday's contract-2 price.
    If a data gap spans more than one roll the change is left undefined.
    """
    data = pd.DataFrame({"cl1": cl1, "cl2": cl2}).dropna(subset=["cl1"])
    front = front_contract(data.index)
    steps = pd.Series(front.asi8, index=data.index).diff()
    change = data["cl1"].diff()
    rolled = steps == 1
    change[rolled] = data["cl1"][rolled] - data["cl2"].shift(1)[rolled]
    change[steps > 1] = np.nan
    return change.reindex(cl1.index).rename("cl1_roll_adj_change")
