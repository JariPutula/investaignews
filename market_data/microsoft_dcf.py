"""Five-year FCFF model, consistent equity bridge and conditional reverse DCF."""

from pydantic import BaseModel, ConfigDict, Field

from market_data.sec import SecDataError

MODEL_VERSION = 1
CASE_LABELS = {"weak": "Heikko", "base": "Perus", "strong": "Vahva"}


class PathAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    growth_pct: list[float] = Field(min_length=5, max_length=5)
    operating_margin_pct: list[float] = Field(min_length=5, max_length=5)
    net_margin_pct: list[float] = Field(min_length=5, max_length=5)
    cash_capex_pct: list[float] = Field(min_length=5, max_length=5)
    lease_capex_pct: list[float] = Field(min_length=5, max_length=5)
    da_pct: list[float] = Field(min_length=5, max_length=5)
    working_capital_pct_of_sales_change: float = Field(ge=-20, le=40)


class ValuationAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tax_pct: float = Field(ge=10, le=35)
    wacc_pct: float = Field(ge=6, le=18)
    terminal_growth_pct: float = Field(ge=0, le=4)
    terminal_roic_pct: float = Field(ge=5, le=60)


def validate_paths(paths, common):
    paths = PathAssumptions.model_validate(paths)
    common = ValuationAssumptions.model_validate(common)
    limits = {"growth_pct": (-40, 60), "operating_margin_pct": (0, 65), "net_margin_pct": (-20, 60), "cash_capex_pct": (0, 70),
              "lease_capex_pct": (0, 30), "da_pct": (0, 45)}
    for field, (low, high) in limits.items():
        if any(not low <= value <= high for value in getattr(paths, field)):
            raise SecDataError(f"Skenaarion {field} ei ole mallin sallitulla välillä.")
    if common.wacc_pct-common.terminal_growth_pct < 3 or common.terminal_roic_pct <= common.terminal_growth_pct:
        raise SecDataError("Jatkuvan kasvun, pääoman tuoton ja tuottovaatimuksen yhdistelmä ei ole kelvollinen.")
    return paths, common


def value_case(snapshot, paths, common, *, growth_override=None):
    paths, common = validate_paths(paths, common)
    revenue = snapshot["financials"]["ttm"]["revenue"]
    balance = snapshot["financials"]["balances"]
    required = ("cash_and_short_term_investments", "borrowings", "finance_lease_debt", "shares")
    if any(balance.get(key) is None for key in required) or not balance.get("borrowings_are_reported_total") or balance["shares"] <= 0 or revenue <= 0:
        raise SecDataError("Arvostus tarvitsee täydet velka-, käteis-, osakemäärä- ja liikevaihtotiedot.")
    discount = common.wacc_pct/100
    tax = common.tax_pct/100
    forecast = []
    for index in range(5):
        growth = paths.growth_pct[index] if growth_override is None else growth_override
        new_revenue = revenue*(1+growth/100)
        ebit = new_revenue*paths.operating_margin_pct[index]/100
        net_income = new_revenue*paths.net_margin_pct[index]/100
        nopat = ebit*(1-tax)
        da = new_revenue*paths.da_pct[index]/100
        cash_capex = new_revenue*paths.cash_capex_pct[index]/100
        lease_capex = new_revenue*paths.lease_capex_pct[index]/100
        wc = (new_revenue-revenue)*paths.working_capital_pct_of_sales_change/100
        fcff = nopat+da-cash_capex-lease_capex-wc
        forecast.append({"year": index+1, "growth_pct": growth, "revenue": new_revenue, "operating_income": ebit, "net_income": net_income,
                         "nopat": nopat, "da": da, "cash_capex": cash_capex, "lease_capex": lease_capex,
                         "working_capital_investment": wc, "fcff": fcff, "present_value": fcff/(1+discount)**(index+1)})
        revenue = new_revenue
    growth = common.terminal_growth_pct/100
    terminal_nopat = revenue*(1+growth)*paths.operating_margin_pct[-1]/100*(1-tax)
    reinvestment = terminal_nopat*common.terminal_growth_pct/common.terminal_roic_pct
    terminal_fcff = terminal_nopat-reinvestment
    terminal_pv = terminal_fcff/(discount-growth)/(1+discount)**5
    enterprise_value = sum(row["present_value"] for row in forecast)+terminal_pv
    net_debt = balance["borrowings"]+balance["finance_lease_debt"]-balance["cash_and_short_term_investments"]
    equity = enterprise_value-net_debt
    price = snapshot["market"]["summaries"]["MSFT"]["price"]
    return {"forecast": forecast, "enterprise_value": enterprise_value, "net_debt": net_debt, "equity_value": equity,
            "value_per_share": equity/balance["shares"], "difference_to_price_pct": (equity/balance["shares"]/price-1)*100,
            "terminal_value_present": terminal_pv, "terminal_share_pct": terminal_pv/enterprise_value*100 if enterprise_value > 0 else None,
            "terminal_nopat": terminal_nopat, "terminal_reinvestment": reinvestment, "terminal_fcff": terminal_fcff,
            "year5_to_terminal_fcff_change": terminal_fcff-forecast[-1]["fcff"],
            "first_year_market_cap_to_nopat": price*balance["shares"]/forecast[0]["nopat"] if forecast[0]["nopat"] > 0 else None}


def reverse_growth(snapshot, paths, common):
    """Solve one constant annual revenue growth rate; never claim unique market beliefs."""
    price = snapshot["market"]["summaries"]["MSFT"]["price"]
    grid = [(float(g), value_case(snapshot, paths, common, growth_override=float(g))["value_per_share"]) for g in range(-40, 61)]
    deltas = [b[1]-a[1] for a, b in zip(grid, grid[1:])]
    if not all(d >= 0 for d in deltas) or not any(d > 0 for d in deltas):
        return {"available": False, "reason": "Kasvun ja arvon suhde ei ole näillä investointioletuksilla yksikäsitteisesti nouseva. Yhtä kasvulukua ei ratkaista."}
    if not grid[0][1] <= price <= grid[-1][1]:
        return {"available": False, "reason": "Nykyhintaa vastaavaa kasvua ei löydy mallin −40–60 prosentin vuosikasvun alueelta."}
    low, high = -40., 60.
    for _ in range(70):
        middle = (low+high)/2
        if value_case(snapshot, paths, common, growth_override=middle)["value_per_share"] < price:
            low = middle
        else:
            high = middle
    value = (low+high)/2
    return {"available": True, "growth_pct": value,
            "matched_value_per_share": value_case(snapshot, paths, common, growth_override=value)["value_per_share"],
            "definition": "Sama vuotuinen liikevaihdon kasvu viiden vuoden ajan, muut tämän skenaarion oletukset ennallaan. Ehdollinen laskelma, ei mitattu markkinaodotus."}


def calculate_exercise(snapshot, cases, common):
    if list(cases) != list(CASE_LABELS):
        raise SecDataError("Harjoituksessa tarvitaan heikko, perus ja vahva skenaario tässä järjestyksessä.")
    values = {key: value_case(snapshot, paths, common) for key, paths in cases.items()}
    market_cap = snapshot["market"]["summaries"]["MSFT"]["price"]*snapshot["financials"]["balances"]["shares"]
    for value in values.values():
        income = value["forecast"][0]["net_income"]
        value["first_year_scenario_pe"] = market_cap/income if income > 0 else None
        value["difference_to_month_price_pct"] = (value["value_per_share"]/snapshot["market"]["summaries"]["MSFT"]["month"]["close"]-1)*100
    if not values["weak"]["value_per_share"] <= values["base"]["value_per_share"] <= values["strong"]["value_per_share"]:
        raise SecDataError("Skenaarioiden lasketut arvot eivät etene heikosta vahvaan. Oletukset tarvitsevat uuden tarkastelun.")
    sensitivity = []
    for wacc_delta in (-2, 0, 2):
        for growth_delta in (-1, 0, 1):
            changed = {**common, "wacc_pct": common["wacc_pct"]+wacc_delta,
                       "terminal_growth_pct": common["terminal_growth_pct"]+growth_delta}
            try:
                result = value_case(snapshot, cases["base"], changed)
                implied = reverse_growth(snapshot, cases["base"], changed)
            except (ValueError, SecDataError):
                continue
            sensitivity.append({"wacc_pct": changed["wacc_pct"], "terminal_growth_pct": changed["terminal_growth_pct"],
                                "value_per_share": result["value_per_share"], "implied_growth_pct": implied.get("growth_pct")})
    ttm = snapshot["financials"]["ttm"]
    return {"model_version": MODEL_VERSION, "values": values,
            "reverse": {key: reverse_growth(snapshot, paths, common) for key, paths in cases.items()},
            "sensitivity": sensitivity, "market_cap": market_cap,
            "ttm_price_to_earnings": market_cap/ttm["net_income"] if ttm["net_income"] > 0 else None,
            "ttm_fcf_yield_pct": ttm["free_cash_flow"]/market_cap*100,
            "definition": "FCFF = liikevoitto × (1 − veroaste) + poistot − käteisinvestoinnit − uudet rahoitusleasing-hyödykkeet − käyttöpääomaan sitoutuva lisäraha. Yritysarvosta vähennetään lainat ja rahoitusleasingvelat sekä lisätään käteinen ja lyhytaikaiset sijoitukset. Operating lease -kulut säilyvät liiketoiminnan kuluina, eikä niiden velkoja vähennetä toistamiseen."
            " Jatkuvan vaiheen rahavirta = verojen jälkeinen liikevoitto × (1 − jatkuva kasvu / uuden pääoman tuotto).",
            "limits": "Tuottovaatimus on WACC-mallioletus, ei henkilökohtainen tuottotakuu. Arvio käyttää raportointipäivän osakemäärää; tulevia takaisinostoja tai laimennusta ei ennusteta. Osakeperusteisten palkkioiden kulu säilyy liikevoitossa. Käyttöpääoman historia on valittujen lyhytaikaisten operatiivisten erien rahavirtamuutosten rajaus. Jatkuva vaihe alkaa viiden vuoden jälkeen ja voi poiketa voimakkaasti viimeisen ennustevuoden investointitasosta. Ei todennäköisyyksiä tai analyytikkojen konsensusennusteita."}
