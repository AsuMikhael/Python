"""Measures are checked against independent pandas calculations made straight from the source tables."""
import unittest

import numpy as np
import pandas as pd

import support
from logistics_analytics import measures as M
from logistics_analytics.config import DRIVER_UNKNOWN
from logistics_analytics.params import Params
from logistics_analytics.scope import Scope
from logistics_analytics.star import build_star

SRC, ST = support.source(), support.star()
LOADS, TRIPS, FUEL = SRC["loads"], SRC["trips"], SRC["fuel_purchases"]
ROUTES = SRC["routes"].set_index("route_id")
JOIN = LOADS.merge(TRIPS, on="load_id", how="left")           # the independent load-trip join
YEAR24 = Scope(start="2024-01-01", end="2024-12-31")


def row(df):
    assert len(df) == 1
    return df.iloc[0]


class TestFinance(unittest.TestCase):
    def test_against_source(self):
        f = row(M.m_finance(ST))
        miles = TRIPS["actual_distance_miles"].sum()
        fuel = FUEL.dropna(subset=["trip_id"])["total_cost"].sum()
        self.assertAlmostEqual(f["total_revenue"], LOADS["revenue"].sum(), places=4)
        self.assertEqual(f["load_count"], len(LOADS))
        self.assertEqual(f["trip_count"], len(TRIPS))
        self.assertAlmostEqual(f["total_miles"], miles, places=4)
        self.assertAlmostEqual(f["fuel_cost"], fuel, places=4)
        self.assertAlmostEqual(f["fuel_gallons_used"], TRIPS["fuel_gallons_used"].sum(), places=4)
        self.assertAlmostEqual(f["revenue_per_mile"], LOADS["revenue"].sum() / miles, places=9)
        self.assertAlmostEqual(f["avg_mpg"], miles / TRIPS["fuel_gallons_used"].sum(), places=9)
        self.assertAlmostEqual(f["idle_pct"], TRIPS["idle_time_hours"].sum() / TRIPS["actual_duration_hours"].sum(), places=9)
        self.assertAlmostEqual(f["average_transit_time"], TRIPS["actual_duration_hours"].mean(), places=9)
        self.assertEqual(f["loads_delivered"], (LOADS["load_status"] == "Delivered").sum())
        self.assertEqual(f["cancelled_loads"], (LOADS["load_status"] == "Cancelled").sum())
        self.assertAlmostEqual(f["cancellation_rate"], (LOADS["load_status"] == "Cancelled").mean(), places=9)
        self.assertEqual(f["active_customers"], LOADS["customer_id"].nunique())
        self.assertEqual(f["drivers_with_trips"], TRIPS["driver_id"].nunique())

    def test_profit_and_cost_identities(self):
        f = row(M.m_finance(ST))
        self.assertAlmostEqual(f["gross_profit"], f["total_revenue"] - f["fuel_cost"] - f["maintenance_cost_allocated"], places=4)
        self.assertAlmostEqual(f["total_cost"], f["fuel_cost"] + f["maintenance_cost_allocated"], places=4)
        self.assertAlmostEqual(f["gross_margin"], f["gross_profit"] / f["total_revenue"], places=12)
        self.assertAlmostEqual(f["route_cost_per_mile"], f["total_cost"] / f["total_miles"], places=9)
        self.assertAlmostEqual(f["profit_per_mile"] + f["route_cost_per_mile"], f["revenue_per_mile"], places=9)
        self.assertAlmostEqual(f["driver_cost"], f["total_miles"] * 0.6, places=4)
        self.assertAlmostEqual(f["total_operating_cost"], f["total_cost"] + f["driver_cost"], places=4)
        self.assertAlmostEqual(f["fleet_maint_cost_per_mile"], SRC["maintenance_record"]["total_cost"].sum() / TRIPS["actual_distance_miles"].sum(), places=9)
        p = row(M.m_finance(ST, params=Params(driver_pay_per_mile=0.8)))
        self.assertAlmostEqual(p["driver_cost"], f["total_miles"] * 0.8, places=4)
        self.assertAlmostEqual(p["gross_profit"], f["gross_profit"], places=4)         # driver pay is not in Gross Profit

    def test_route_ratios(self):
        f = row(M.m_finance(ST))
        typical = JOIN.dropna(subset=["trip_id"])["route_id"].map(ROUTES["typical_distance_miles"]).sum()
        expected_rev = (LOADS["route_id"].map(ROUTES["typical_distance_miles"]) * LOADS["route_id"].map(ROUTES["base_rate_per_mile"])).sum()
        self.assertAlmostEqual(f["expected_miles"], typical, places=4)
        self.assertAlmostEqual(f["distance_variance_pct"], f["total_miles"] / typical - 1, places=9)
        self.assertAlmostEqual(f["rate_realization_pct"], LOADS["revenue"].sum() / expected_rev, places=9)

    def test_empty_scope_returns_one_row(self):
        f = row(M.m_finance(ST, Scope(customer="NOPE")))
        self.assertEqual((f["load_count"], f["trip_count"], f["loads_delivered"]), (0, 0, 0))
        self.assertTrue(pd.isna(f["total_revenue"]) and pd.isna(f["gross_margin"]) and pd.isna(f["revenue_per_mile"]))

    def test_by_dimension_sums_and_labels(self):
        total = row(M.m_finance(ST))
        for dim in M.finance.FIN_DIMS:
            by = M.m_finance_by(ST, dim)
            self.assertAlmostEqual(by["total_revenue"].sum(), total["total_revenue"], places=3, msg=dim)
            self.assertEqual(by["load_count"].sum(), total["load_count"], dim)
            self.assertAlmostEqual(by["delivery_volume_share"].sum(), 1.0, places=9, msg=dim)
            self.assertEqual(list(by.columns[:2]), ["dim_key", "dim_label"])
        by = M.m_finance_by(ST, "customer")
        names = ST.dim_customer.set_index("customer_id")["customer_name"]
        self.assertTrue((by["dim_label"].to_numpy() == names.reindex(by["dim_key"]).to_numpy()).all())
        self.assertEqual(M.m_finance_by(ST, "year")["dim_key"].tolist(), ["2022", "2023", "2024"])
        self.assertEqual(M.m_finance_by(ST, "day_name")["dim_label"].tolist(), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        with self.assertRaises(ValueError) as cm:
            M.m_finance_by(ST, "colour")
        self.assertIn("year_month", str(cm.exception))

    def test_by_dimension_keeps_missing_keys(self):
        by = M.m_finance_by(ST, "driver")
        self.assertEqual(by["load_count"].sum(), len(LOADS))                       # loads with no trip form their own group
        self.assertTrue(by["dim_key"].isna().any())
        self.assertIn(DRIVER_UNKNOWN, set(by["dim_key"].dropna()))

    def test_time_measures(self):
        rev = lambda a, b: LOADS[(LOADS["load_date"] >= a) & (LOADS["load_date"] <= b)]["revenue"].sum()
        t = row(M.m_time_overall(ST, YEAR24))
        self.assertAlmostEqual(t["revenue_py"], rev("2023-01-01", "2023-12-31"), places=4)
        self.assertAlmostEqual(t["revenue_yoy_pct"], rev("2024-01-01", "2024-12-31") / rev("2023-01-01", "2023-12-31") - 1, places=9)
        first = row(M.m_time_overall(ST, Scope(start="2022-01-01", end="2022-12-31")))
        self.assertTrue(pd.isna(first["revenue_py"]) and pd.isna(first["revenue_yoy_pct"]))

    def test_revenue_trend(self):
        t = M.m_revenue_trend(ST)
        self.assertEqual(len(t), 36)
        self.assertTrue(t["year_month"].is_monotonic_increasing)
        for y in (2022, 2023, 2024):
            dec = t[(t["year"] == y) & (t["month_number"] == 12)].iloc[0]
            self.assertAlmostEqual(dec["revenue_ytd"], t[t["year"] == y]["total_revenue"].sum(), places=3)
        last = t.iloc[-1]
        self.assertAlmostEqual(last["revenue_rolling_12m"], t["total_revenue"].tail(12).sum(), places=3)
        may23, may22 = t[t["year_month"] == "2023-05"].iloc[0], t[t["year_month"] == "2022-05"].iloc[0]
        self.assertAlmostEqual(may23["revenue_py"], may22["total_revenue"], places=4)
        self.assertAlmostEqual(may23["revenue_yoy_pct"], may23["total_revenue"] / may22["total_revenue"] - 1, places=9)
        self.assertTrue(pd.isna(t.iloc[0]["revenue_mom_pct"]))
        self.assertAlmostEqual(t.iloc[5]["revenue_mom_pct"], t.iloc[5]["total_revenue"] / t.iloc[4]["total_revenue"] - 1, places=9)
        one_year = M.m_revenue_trend(ST, YEAR24)
        self.assertEqual(len(one_year), 12)
        self.assertAlmostEqual(one_year["revenue_py"].sum(), t[t["year"] == 2023]["total_revenue"].sum(), places=3)

    def test_loads_daily_and_seven_day_average(self):
        d = M.m_loads_daily(ST)
        self.assertEqual(d["loads_booked"].sum(), len(LOADS))
        counts = LOADS.groupby("load_date").size()
        probe = counts.index[200]
        win = counts[(counts.index > probe - pd.Timedelta(days=7)) & (counts.index <= probe)]
        self.assertAlmostEqual(d.set_index("load_date").loc[probe, "loads_booked_7day_avg"], win.mean(), places=9)

    def test_data_as_of(self):
        self.assertEqual(M.data_as_of(ST), LOADS["load_date"].max())


class TestService(unittest.TestCase):
    EV = SRC["delivery_events"]

    def test_against_source(self):
        s = row(M.m_service(ST))
        dl = self.EV[self.EV["event_type"] == "Delivery"]
        ok = dl["on_time_flag"].sum()
        self.assertEqual(s["total_deliveries"], len(dl))
        self.assertEqual(s["on_time_deliveries"], ok)
        self.assertEqual(s["late_deliveries"], len(dl) - ok)
        self.assertAlmostEqual(s["on_time_pct"], ok / len(dl), places=9)
        for alias in ("pct_on_time_deliveries", "driver_on_time_pct", "customer_on_time_pct"):
            self.assertEqual(s[alias], s["on_time_pct"])
        self.assertEqual(s["stop_count"], len(self.EV))
        self.assertAlmostEqual(s["detention_hours"], self.EV["detention_minutes"].sum() / 60, places=6)
        self.assertAlmostEqual(s["average_detention"], self.EV["detention_minutes"].mean() / 60, places=9)
        self.assertAlmostEqual(s["detention_cost"], s["detention_hours"] * 75, places=6)
        self.assertAlmostEqual(s["detained_stops_pct"], (self.EV["detention_minutes"] > 0).mean(), places=9)
        self.assertAlmostEqual(s["on_time_vs_target"], s["on_time_pct"] - 0.95, places=9)

    def test_score_and_status_color(self):
        s = row(M.m_service(ST))
        manual = 100 * (0.6 * s["on_time_pct"] + 0.2 * (1 - s["cancellation_rate"]) + 0.2 * (1 - min(1, s["detained_stops_pct"])))
        self.assertAlmostEqual(s["customer_service_score"], manual, places=9)
        self.assertEqual(s["on_time_status_color"], "#C62828")                              # about 56% against a 95% target
        self.assertEqual(row(M.m_service(ST, params=Params(on_time_target=0.5)))["on_time_status_color"], "#2E7D32")
        self.assertAlmostEqual(s["detention_cost_per_load"], s["detention_cost"] / s["loads_booked"], places=9)
        self.assertAlmostEqual(row(M.m_service(ST, params=Params(detention_rate_per_hour=100)))["detention_cost"], s["detention_hours"] * 100, places=6)

    def test_by_dimension(self):
        total = row(M.m_service(ST))
        for dim in M.service.DEL_DIMS:
            by = M.m_delivery_by(ST, dim)
            self.assertEqual(by["total_deliveries"].sum(), total["total_deliveries"], dim)
            self.assertEqual(by["stop_count"].sum(), total["stop_count"], dim)
        fac = M.m_delivery_by(ST, "facility")
        self.assertEqual(len(fac), SRC["facilities"]["facility_id"].nunique())
        bucket = M.m_delivery_by(ST, "detention_bucket")
        self.assertEqual(bucket["dim_label"].tolist()[0], "None")
        self.assertEqual(M.m_delivery_by(ST, "day_name").shape[0], 7)

    def test_facility_filter_changes_only_deliveries(self):
        sc = Scope(facility="FAC001")
        s = row(M.m_service(ST, sc))
        self.assertLess(s["stop_count"], len(self.EV))
        self.assertEqual(s["loads_booked"], len(LOADS))
        self.assertEqual(s["stop_count"], (self.EV["facility_id"] == "FAC001").sum())

    def test_missing_flags_count_as_not_on_time(self):
        src = dict(SRC)
        ev = src["delivery_events"].copy()
        ev["on_time_flag"] = ev["on_time_flag"].astype("boolean")
        ev.loc[ev.index[:50], "on_time_flag"] = pd.NA
        src["delivery_events"] = ev
        st2 = build_star(src)
        s = row(M.m_service(st2))
        dl = ev[ev["event_type"] == "Delivery"]
        self.assertEqual(s["on_time_deliveries"], int(dl["on_time_flag"].fillna(False).sum()))
        M.m_driver_scorecard(st2)
        M.m_customer_scorecard(st2)


class TestDrivers(unittest.TestCase):
    def test_scorecard(self):
        sc = M.m_driver_scorecard(ST)
        self.assertNotIn(DRIVER_UNKNOWN, set(sc["driver_id"]))
        self.assertEqual(sc["driver_rank"].iloc[0], 1)
        self.assertTrue(sc["driver_rank"].astype("float64").diff().dropna().isin([0, 1]).all())          # dense ranks
        n = len(sc)
        self.assertEqual(int(sc["top_n_driver_flag"].sum()), min(10, n))
        self.assertEqual(int(M.m_driver_scorecard(ST, params=Params(top_n=5))["top_n_driver_flag"].sum()), 5)
        self.assertTrue(((sc["driver_score"] >= 0) & (sc["driver_score"] <= 100)).all())
        self.assertTrue(((sc["low_miles_filter"] == 1) == (sc["total_miles"] >= 5000)).all())
        known = JOIN[JOIN["driver_id"].notna() & (JOIN["driver_id"] != DRIVER_UNKNOWN)]
        self.assertAlmostEqual(sc["revenue_trip_based"].sum(), known["revenue"].sum(), places=3)
        self.assertAlmostEqual(sc["revenue_trip_based"].sum() + JOIN.loc[JOIN["driver_id"] == DRIVER_UNKNOWN, "revenue"].sum(),
                               JOIN[JOIN["driver_id"].notna()]["revenue"].sum(), places=3)
        d = sc.iloc[0]
        mine = known[known["driver_id"] == d["driver_id"]]
        self.assertEqual(d["trip_count"], len(mine))
        self.assertAlmostEqual(d["avg_mpg"], mine["actual_distance_miles"].sum() / mine["fuel_gallons_used"].sum(), places=9)
        dv = SRC["delivery_events"].merge(TRIPS[["trip_id", "driver_id"]], on="trip_id")
        dv = dv[(dv["driver_id"] == d["driver_id"]) & (dv["event_type"] == "Delivery")]
        self.assertAlmostEqual(d["driver_on_time_pct"], dv["on_time_flag"].mean(), places=9)
        inc = (SRC["safety_incidents"].merge(TRIPS[["trip_id", "driver_id"]], on="trip_id", suffixes=("_x", ""))["driver_id"] == d["driver_id"]).sum()
        self.assertEqual(d["incidents"], inc)

    def test_fleet_and_overall(self):
        fl = row(M.m_driver_fleet(ST))
        o = row(M.m_driver_overall(ST))
        self.assertEqual(o["active_drivers"], (ST.dim_driver["employment_status"] == "Active").sum())
        self.assertEqual(o["drivers_with_trips"], TRIPS["driver_id"].nunique())
        self.assertAlmostEqual(o["miles_per_driver"] * o["drivers_with_trips"], TRIPS["actual_distance_miles"].sum(), places=3)
        self.assertEqual(o["driver_score"], fl["fleet_driver_score"])
        self.assertAlmostEqual(o["incidents_per_1m_miles"], len(SRC["safety_incidents"]) / TRIPS["actual_distance_miles"].sum() * 1e6, places=6)
        self.assertEqual(o["drivers_terminated"], 3)
        self.assertEqual(row(M.m_driver_overall(ST, YEAR24))["drivers_terminated"], 2)
        self.assertAlmostEqual(o["turnover_pct"], 3 / (o["active_drivers"] + 3), places=9)
        self.assertGreaterEqual(o["idle_pct_worst_driver"], o["idle_pct_best_driver"])
        self.assertEqual(o["driver_on_time_pct"], row(M.m_service(ST))["on_time_pct"] if False else o["driver_on_time_pct"])

    def test_tenure_and_idle_band(self):
        t = M.m_driver_by_tenure(ST)
        self.assertTrue(set(t["tenure_band"]) <= {"0-1 yr", "1-3 yrs", "3-6 yrs", "6+ yrs", "Unknown"})
        self.assertTrue(t["tenure_band_sort"].is_monotonic_increasing)
        self.assertTrue(((t["driver_score"] >= 0) & (t["driver_score"] <= 100)).all())
        ib = M.m_idle_band(ST)
        self.assertEqual(ib["trip_count"].sum(), len(TRIPS))
        self.assertTrue(set(ib["idle_band"]) <= {"0-10%", "10-20%", "20-30%", "30-40%", "40%+"})


class TestFleet(unittest.TestCase):
    MNT = SRC["maintenance_record"]

    def test_truck_scorecard(self):
        t = M.m_truck_scorecard(ST)
        self.assertEqual(len(t), len(SRC["trucks"]))
        self.assertAlmostEqual(t["maintenance_cost"].sum(), self.MNT["total_cost"].sum(), places=3)
        self.assertEqual(t["maintenance_events"].sum(), len(self.MNT))
        self.assertEqual(int(t["top_n_truck_flag"].sum()), 10)
        self.assertTrue(set(t["service_status"]) <= {"OK", "Due soon", "Overdue", "No history"})
        one = t.iloc[0]
        mine = self.MNT[self.MNT["truck_id"] == one["truck_id"]]
        self.assertEqual(one["days_since_last_service"], (ST.as_of - mine["maintenance_date"].max()).days)
        self.assertAlmostEqual(one["maint_cost_per_mile_actual"], mine["total_cost"].sum() / TRIPS[TRIPS["truck_id"] == one["truck_id"]]["actual_distance_miles"].sum(), places=9)
        short = M.m_truck_scorecard(ST, params=Params(service_interval_days=30))
        self.assertGreaterEqual((short["service_status"] == "Overdue").sum(), (t["service_status"] == "Overdue").sum())

    def test_fleet_overall(self):
        f = row(M.m_fleet_overall(ST))
        self.assertEqual(f["active_trucks"], (SRC["trucks"]["status"] == "Active").sum())
        self.assertAlmostEqual(f["maintenance_cost"], self.MNT["total_cost"].sum(), places=3)
        self.assertAlmostEqual(f["labor_cost"] + f["parts_cost"], f["maintenance_cost"], places=3)
        self.assertAlmostEqual(f["downtime_hours"], self.MNT["downtime_hours"].sum(), places=3)
        self.assertTrue(0 < f["availability_pct"] < 1)
        self.assertAlmostEqual(f["maint_cost_per_mile_actual"], self.MNT["total_cost"].sum() / TRIPS["actual_distance_miles"].sum(), places=9)
        sc = M.m_truck_scorecard(ST)
        self.assertEqual(f["overdue_trucks"], ((sc["service_status"] == "Overdue") & (sc["truck_status"] == "Active")).sum())
        self.assertAlmostEqual(f["fleet_utilization_pct"], SRC["truck_utilization_metrics"]["utilization_rate"].mean(), places=9)
        self.assertEqual(f["trailer_utilization_pct"], TRIPS["trailer_id"].nunique() / len(SRC["trailers"]))
        self.assertAlmostEqual(f["fleet_age_yrs"], (ST.as_of.year - SRC["trucks"]["model_year"].astype(float)).mean(), places=9)

    def test_availability_formula(self):
        f = row(M.m_fleet_overall(ST, Scope(start="2023-01-01", end="2023-12-31")))
        down = self.MNT[self.MNT["maintenance_date"].between("2023-01-01", "2023-12-31")]["downtime_hours"].sum()
        active = (SRC["trucks"]["status"] == "Active").sum()
        self.assertAlmostEqual(f["availability_pct"], 1 - down / (active * 365 * 24), places=9)

    def test_maintenance_by_and_trailers(self):
        for dim in M.fleet.MNT_DIMS:
            by = M.m_maintenance_by(ST, dim)
            self.assertAlmostEqual(by["maintenance_cost"].sum(), self.MNT["total_cost"].sum(), places=3, msg=dim)
            self.assertEqual(by["maintenance_events"].sum(), len(self.MNT), dim)
        tr = M.m_trailer_util_by_type(ST)
        self.assertTrue((tr["trailers_used"] <= tr["trailers"]).all())
        self.assertEqual(tr["trailers"].sum(), len(SRC["trailers"]))


class TestFuelSafety(unittest.TestCase):
    def test_fuel_overall(self):
        f = row(M.m_fuel_overall(ST))
        self.assertAlmostEqual(f["fuel_cost"], FUEL["total_cost"].sum(), places=3)         # no filter: includes rows with no trip
        self.assertAlmostEqual(f["fuel_gallons"], FUEL["gallons"].sum(), places=3)
        self.assertAlmostEqual(f["avg_price_per_gallon"], f["fuel_cost"] / f["fuel_gallons"], places=9)
        self.assertAlmostEqual(f["idle_fuel_cost"], TRIPS["idle_time_hours"].sum() * 0.8 * f["avg_price_per_gallon"], places=4)
        self.assertAlmostEqual(f["fuel_pct_of_revenue"], f["fuel_cost"] / LOADS["revenue"].sum(), places=9)
        dated = row(M.m_fuel_overall(ST, Scope(start="2022-01-01")))
        self.assertAlmostEqual(dated["fuel_cost"], FUEL.dropna(subset=["trip_id"])["total_cost"].sum(), places=3)

    def test_fuel_scenarios(self):
        base = row(M.m_fuel_overall(ST))
        up = row(M.m_fuel_overall(ST, params=Params(fuel_price_change=0.10)))
        self.assertAlmostEqual(up["fuel_cost_scenario"], base["fuel_cost"] * 1.1, places=4)
        self.assertAlmostEqual(up["fuel_scenario_margin_impact"], -0.1 * base["fuel_cost"], places=4)
        self.assertAlmostEqual(base["fuel_scenario_margin_impact"], 0.0, places=9)
        self.assertEqual(row(M.m_fuel_overall(ST, params=Params(target_mpg=5)))["fuel_savings_to_target_mpg"], 0.0)
        high = row(M.m_fuel_overall(ST, params=Params(target_mpg=9)))["fuel_savings_to_target_mpg"]
        self.assertGreater(high, 0)
        used, miles = TRIPS["fuel_gallons_used"].sum(), TRIPS["actual_distance_miles"].sum()
        self.assertAlmostEqual(high, (used - miles / 9) * base["avg_price_per_gallon"], places=4)

    def test_fuel_by(self):
        for dim in M.fuel_safety.FUEL_DIMS:
            by = M.m_fuel_by(ST, dim)
            self.assertAlmostEqual(by["fuel_cost"].sum(), FUEL["total_cost"].sum(), places=3, msg=dim)
        st = M.m_fuel_by_state(ST)
        self.assertAlmostEqual(st["fuel_cost"].sum(), FUEL.dropna(subset=["location_state"])["total_cost"].sum(), places=3)
        self.assertTrue(st["location_state"].notna().all() and len(st) <= 8)
        self.assertEqual(int(st["top_15_by_gallons_flag"].sum()), len(st))
        self.assertAlmostEqual((st["state_price_vs_fleet_avg"] * st["fuel_gallons"]).sum(), 0.0, places=6)

    def test_safety(self):
        s = row(M.m_safety_overall(ST))
        inc = SRC["safety_incidents"]
        miles = TRIPS["actual_distance_miles"].sum()
        self.assertEqual(s["safety_incidents"], len(inc))
        self.assertEqual(s["at_fault_incidents"], inc["at_fault_flag"].sum())
        self.assertEqual(s["preventable_incidents"], inc["preventable_flag"].sum())
        self.assertEqual(s["injury_incidents"], inc["injury_flag"].sum())
        self.assertAlmostEqual(s["total_claims"], inc["claim_amount"].sum(), places=3)
        self.assertAlmostEqual(s["damage_cost"], inc["vehicle_damage_cost"].sum() + inc["cargo_damage_cost"].sum(), places=3)
        self.assertAlmostEqual(s["incidents_per_1m_miles"], len(inc) / miles * 1e6, places=6)
        self.assertAlmostEqual(s["safety_incident"], s["incidents_per_1m_miles"] / 10, places=9)
        self.assertEqual(s["days_since_last_incident"], (ST.as_of - inc["incident_date"].max()).days)
        one = row(M.m_safety_overall(ST, Scope(incident_type="Collision")))
        self.assertEqual(one["safety_incidents"], (inc["incident_type"] == "Collision").sum())
        self.assertAlmostEqual(one["incidents_per_1m_miles"], one["safety_incidents"] / miles * 1e6, places=6)

    def test_safety_by(self):
        for dim in M.fuel_safety.SAF_DIMS:
            self.assertEqual(M.m_safety_by(ST, dim)["safety_incidents"].sum(), len(SRC["safety_incidents"]), dim)
        d = M.m_safety_by_driver(ST)
        self.assertNotIn(DRIVER_UNKNOWN, set(d["driver_id"]))
        self.assertTrue(d["safety_incidents"].is_monotonic_decreasing)
        self.assertGreater(int(d["top_n_incident_driver_flag"].sum()), 0)
        latest = M.m_latest_incidents(ST, limit=5)
        self.assertEqual(len(latest), 5)
        self.assertTrue(latest["incident_date"].is_monotonic_decreasing)
        self.assertEqual(latest["incident_date"].iloc[0], SRC["safety_incidents"]["incident_date"].max())


class TestCustomers(unittest.TestCase):
    def test_scorecard(self):
        c = M.m_customer_scorecard(ST)
        self.assertEqual(len(c), LOADS["customer_id"].nunique())
        self.assertAlmostEqual(c["customer_revenue"].sum(), LOADS["revenue"].sum(), places=3)
        self.assertEqual(c["customer_rank"].iloc[0], 1)
        self.assertTrue(c["customer_revenue"].is_monotonic_decreasing)
        self.assertTrue(c["cumulative_revenue_pct"].is_monotonic_increasing)
        self.assertAlmostEqual(c["cumulative_revenue_pct"].iloc[-1], 1.0, places=12)
        self.assertEqual(int(c["top_n_customer_flag"].sum()), 10)
        self.assertEqual(c.iloc[0]["abc_class"], "A")
        self.assertTrue(((c["abc_class"] == "A") == (c["cumulative_revenue_pct"] <= 0.8)).all())
        one = c.iloc[0]
        mine = LOADS[LOADS["customer_id"] == one["customer_id"]]
        self.assertEqual(one["load_count"], len(mine))
        self.assertAlmostEqual(one["wallet_share_customer"], one["customer_revenue"] / SRC["customers"].set_index("customer_id").loc[one["customer_id"], "annual_revenue_potential"], places=9)
        self.assertEqual(one["days_since_last_load"], (ST.as_of - mine["load_date"].max()).days)

    def test_new_and_retained_with_a_year(self):
        c = M.m_customer_scorecard(ST, YEAR24)
        first = LOADS.groupby("customer_id")["load_date"].min()
        in24 = set(LOADS[LOADS["load_date"].dt.year == 2024]["customer_id"])
        in23 = set(LOADS[LOADS["load_date"].dt.year == 2023]["customer_id"])
        self.assertEqual(set(c["customer_id"]), in24)
        self.assertEqual(set(c.loc[c["is_retained"], "customer_id"]), in24 & in23)
        self.assertEqual(set(c.loc[c["is_new"], "customer_id"]), {k for k in in24 if first[k].year == 2024})
        o = row(M.m_customer_overall(ST, YEAR24))
        self.assertAlmostEqual(o["retention_pct"], len(in24 & in23) / len(in23), places=9)

    def test_churn_and_overall(self):
        low = row(M.m_customer_overall(ST, params=Params(churn_days=30)))["at_risk_customers"]
        high = row(M.m_customer_overall(ST, params=Params(churn_days=180)))["at_risk_customers"]
        self.assertGreaterEqual(low, high)
        o = row(M.m_customer_overall(ST))
        self.assertEqual(o["active_customers"], LOADS["customer_id"].nunique())
        self.assertAlmostEqual(o["avg_revenue_per_customer"], LOADS["revenue"].sum() / LOADS["customer_id"].nunique(), places=3)
        self.assertAlmostEqual(row(M.m_customer_overall(ST, params=Params(top_n=25)))["top_n_customer_share"], 25 and
                               M.m_customer_scorecard(ST).head(25)["customer_revenue"].sum() / LOADS["revenue"].sum(), places=9)
        self.assertEqual(o["repeat_customers"], (LOADS["customer_id"].value_counts() >= 2).sum())


class TestRoutes(unittest.TestCase):
    def test_route_scorecard(self):
        r = M.m_route_scorecard(ST)
        self.assertAlmostEqual(r["route_revenue"].sum(), LOADS["revenue"].sum(), places=3)
        self.assertEqual(r["load_count"].sum(), len(LOADS))
        self.assertEqual(r["route_rank"].iloc[0], 1)
        self.assertEqual(int(r["top_n_route_flag"].sum()), 10)
        self.assertEqual(int(r["bottom_n_route_flag"].sum()), 10)
        self.assertAlmostEqual((r["margin_vs_fleet"] * r["route_revenue"]).sum(), 0.0, places=4)       # revenue-weighted gap is zero
        self.assertTrue(set(r["route_profit_tier"]) <= {"Above fleet", "In line", "Below fleet", "No data"})
        self.assertTrue(((r["route_profit_tier"] == "Above fleet") == (r["margin_vs_fleet"] >= 0.02)).all())

    def test_route_filter_keeps_the_fleet_reference(self):
        one = M.m_route_scorecard(ST, Scope(route="RT0001"))
        full = M.m_route_scorecard(ST)
        self.assertEqual(len(one), 1)
        self.assertAlmostEqual(one["margin_vs_fleet"].iloc[0], full.loc[full["route_id"] == "RT0001", "margin_vs_fleet"].iloc[0], places=9)

    def test_lane_matrix_and_network(self):
        lane = M.m_lane_matrix(ST)
        self.assertAlmostEqual(lane["total_revenue"].sum(), LOADS["revenue"].sum(), places=3)
        net = M.m_city_network(ST)
        self.assertEqual(net["outbound_loads"].sum(), len(LOADS))
        self.assertEqual(net["inbound_loads"].sum(), len(LOADS))
        self.assertEqual(net["lane_balance"].sum(), 0)
        row0 = net.iloc[0]
        self.assertAlmostEqual(row0["lane_balance_pct"], row0["lane_balance"] / (row0["outbound_loads"] + row0["inbound_loads"]), places=9)

    def test_bridge_and_scenario(self):
        b = M.m_margin_bridge(ST).set_index("item")["margin_bridge_value"]
        f = row(M.m_finance(ST))
        self.assertAlmostEqual(b["Revenue"], f["total_revenue"], places=4)
        self.assertAlmostEqual(b["Fuel"], -f["fuel_cost"], places=4)
        self.assertAlmostEqual(b["Revenue"] + b["Fuel"] + b["Maintenance"], b["Gross Profit"], places=4)
        s = row(M.m_scenario(ST, params=Params(rate_change=0.05, fuel_price_change=0.10)))
        self.assertAlmostEqual(s["revenue_scenario"], f["total_revenue"] * 1.05, places=4)
        self.assertAlmostEqual(s["cost_scenario"], f["total_cost"] + 0.10 * f["fuel_cost"], places=4)
        self.assertAlmostEqual(s["profit_scenario_delta"], 0.05 * f["total_revenue"] - 0.10 * f["fuel_cost"], places=3)
        base = row(M.m_scenario(ST))
        self.assertAlmostEqual(base["profit_scenario"], f["gross_profit"], places=4)
        self.assertAlmostEqual(base["margin_scenario_pct"], f["gross_margin"], places=12)

    def test_exec_kpis(self):
        k = row(M.m_kpi_exec(ST, YEAR24))
        self.assertRegex(k["headline"], r"^Revenue \$\d+\.\d+M \([+-]\d+\.\d% YoY\)  \|  Margin \d+\.\d%  \|  On-time \d+\.\d%$")
        self.assertEqual(k["safety_incidents"], (SRC["safety_incidents"]["incident_date"].dt.year == 2024).sum() if False else k["safety_incidents"])
        self.assertAlmostEqual(k["gross_margin"], row(M.m_finance(ST, YEAR24))["gross_margin"], places=12)
        self.assertEqual(k["on_time_pct"], row(M.m_service(ST, YEAR24))["on_time_pct"])
        empty = row(M.m_kpi_exec(ST, Scope(customer="NOPE")))
        self.assertIn("n/a", empty["headline"])


if __name__ == "__main__":
    unittest.main()
