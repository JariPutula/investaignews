"""Pure financial calculations; values stay in USD until display formatting."""


def ratio(numerator, denominator, *, percentage=False):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator * (100 if percentage else 1)


def percentage_change(new, old):
    value = ratio(new, old)
    return (value - 1) * 100 if value is not None else None


def revenue_cagr(rows):
    if len(rows) < 2:
        return None
    years = [r["fiscal_year"] for r in rows]
    if years != list(range(years[0], years[-1] + 1)):
        return None
    first, last = rows[0]["revenue"], rows[-1]["revenue"]
    if first is None or last is None or first <= 0 or last < 0:
        return None
    return ((last / first) ** (1 / (years[-1] - years[0])) - 1) * 100


def calculate_metrics(row, previous=None):
    revenue = row.get("revenue")
    income = row.get("operating_income")
    cash_flow, capex = row.get("operating_cash_flow"), row.get("capex")
    cost = row.get("cost_of_revenue")
    gross_profit = revenue - cost if revenue is not None and cost is not None else None
    fcf = cash_flow - capex if cash_flow is not None and capex is not None else None
    return {
        "revenue_growth_pct": percentage_change(revenue, previous.get("revenue")) if previous else None,
        "operating_margin_pct": ratio(income, revenue, percentage=True),
        "gross_profit": gross_profit,
        "gross_margin_pct": ratio(gross_profit, revenue, percentage=True),
        "net_margin_pct": ratio(row.get("net_income"), revenue, percentage=True),
        "free_cash_flow": fcf,
        "fcf_margin_pct": ratio(fcf, revenue, percentage=True),
        "capex_to_revenue_pct": ratio(capex, revenue, percentage=True),
        "current_ratio": ratio(row.get("current_assets"), row.get("current_liabilities")),
    }
