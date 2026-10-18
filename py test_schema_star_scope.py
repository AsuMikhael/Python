import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

import support
from logistics_analytics.config import DRIVER_UNKNOWN
from logistics_analytics.loaders import load_csv_dir, load_database, save_tables
from logistics_analytics.schema import SOURCE_SCHEMA, SchemaError, coerce_table, validate_source
from logistics_analytics.scope import (Scope, scope_delivery, scope_driver_month, scope_fuel, scope_load, scope_maintenance, scope_safety,
                                       scope_truck_month)
from logistics_analytics.star import StarBuildError, build_star, load_star, save_star

SRC, ST = support.source(), support.star()


class TestSchema(unittest.TestCase):
    def test_coerce_types(self):
        raw = pd.DataFrame({"load_id": [" L1 ", ""], "customer_id": ["C", "C"], "route_id": ["R", "R"], "load_date": ["2024-03-05 13:00", "bad"],
                            "booking_type": ["x", "y"], "load_type": ["a", "b"], "load_status": ["Delivered", "Cancelled"], "revenue": ["10.5", "n/a"]})
        out = coerce_table(raw, "loads")
        self.assertEqual(out["load_id"].iloc[0], "L1")
        self.assertTrue(pd.isna(out["load_id"].iloc[1]))                      # blank text becomes missing
        self.assertEqual(out["load_date"].iloc[0], pd.Timestamp("2024-03-05"))  # dates are normalised
        self.assertTrue(pd.isna(out["load_date"].iloc[1]))
        self.assertEqual(out["revenue"].iloc[0], 10.5)
        self.assertTrue(pd.isna(out["revenue"].iloc[1]))

    def test_boolean_parsing(self):
        raw = pd.DataFrame({"event_id": list("abcdef"), "trip_id": "t", "facility_id": "f", "event_type": "Delivery", "scheduled_datetime": "2024-01-01",
                            "actual_datetime": "2024-01-01", "detention_minutes": "0", "on_time_flag": ["Yes", "0", "true", "N", "", "maybe"]})
        out = coerce_table(raw, "delivery_events")["on_time_flag"]
        self.assertEqual(out.tolist()[:4], [True, False, True, False])
        self.assertTrue(out.iloc[4:].isna().all())

    def test_validate_lists_everything_missing(self):
        bad = dict(SRC)
        bad.pop("trips")
        bad["loads"] = SRC["loads"].drop(columns=["revenue"])
        with self.assertRaises(SchemaError) as cm:
            validate_source(bad)
        msg = str(cm.exception)
        self.assertIn("trips", msg)
        self.assertIn("revenue", msg)

    def test_all_tables_present(self):
        self.assertEqual(set(SRC), set(SOURCE_SCHEMA))


class TestLoaders(unittest.TestCase):
    def test_csv_round_trip_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            save_tables(SRC, d)
            os.rename(os.path.join(d, "maintenance_record.csv"), os.path.join(d, "Maintenance_Records.CSV"))   # alias and case
            loaded = load_csv_dir(d)
        for t in SOURCE_SCHEMA:
            self.assertEqual(len(loaded[t]), len(SRC[t]), t)
        self.assertAlmostEqual(loaded["loads"]["revenue"].sum(), SRC["loads"]["revenue"].sum(), places=4)
        self.assertEqual(loaded["delivery_events"]["on_time_flag"].dtype, "boolean")
        self.assertEqual(loaded["trucks"]["model_year"].isna().sum(), SRC["trucks"]["model_year"].isna().sum())

    def test_csv_missing_table_raises(self):
        with tempfile.TemporaryDirectory() as d:
            partial = {k: v for k, v in SRC.items() if k != "trips"}
            save_tables(partial, d)
            with self.assertRaises(SchemaError):
                load_csv_dir(d)
            self.assertNotIn("trips", load_csv_dir(d, strict=False))
        with self.assertRaises(FileNotFoundError):
            load_csv_dir("/no/such/dir")

    def test_database_extraction_and_custom_query(self):
        con = sqlite3.connect(":memory:")
        for t, df in SRC.items():
            x = df.copy()
            for c in x.columns:
                if str(x[c].dtype).startswith("datetime"):
                    x[c] = x[c].dt.strftime("%Y-%m-%d %H:%M:%S")
                elif str(x[c].dtype) == "boolean":
                    x[c] = x[c].astype("Int64")
                x[c] = x[c].astype(object).where(x[c].notna(), None)
            x.to_sql(t, con, index=False)
        got = load_database(con)
        self.assertEqual(len(got["loads"]), len(SRC["loads"]))
        self.assertAlmostEqual(got["loads"]["revenue"].sum(), SRC["loads"]["revenue"].sum(), places=4)
        self.assertEqual(got["delivery_events"]["on_time_flag"].sum(), SRC["delivery_events"]["on_time_flag"].sum())
        custom = load_database(con, queries={"loads": "SELECT * FROM loads WHERE load_status = 'Delivered'"})
        self.assertTrue((custom["loads"]["load_status"] == "Delivered").all())
        con.execute("DROP TABLE trips")
        with self.assertRaises(SchemaError):
            load_database(con)
        self.assertNotIn("trips", load_database(con, strict=False))


class TestStar(unittest.TestCase):
    def test_row_counts_and_keys(self):
        self.assertEqual(len(ST.fact_load_trip), len(SRC["loads"]))
        self.assertTrue(ST.fact_load_trip["load_id"].is_unique)
        trips = ST.fact_load_trip["trip_id"].dropna()
        self.assertTrue(trips.is_unique)
        self.assertEqual(len(trips), len(SRC["trips"]))
        for dim, key in [("dim_customer", "customer_id"), ("dim_route", "route_id"), ("dim_driver", "driver_id"), ("dim_truck", "truck_id"),
                         ("dim_trailer", "trailer_id"), ("dim_facility", "facility_id"), ("dim_date", "full_date")]:
            self.assertTrue(getattr(ST, dim)[key].is_unique, dim)

    def test_date_dimension(self):
        d = ST.dim_date.set_index("full_date")
        self.assertEqual(d.index.min(), pd.Timestamp("2022-01-01"))
        self.assertEqual(d.index.max(), pd.Timestamp("2024-12-31"))
        self.assertEqual(len(d), 1096)
        r = d.loc["2024-01-01"]            # a Monday
        self.assertEqual((r["day_number"], r["day_name"], r["year_month"], r["month_year"], r["quarter"]), (1, "Mon", "2024-01", "202401", "Q1"))
        self.assertEqual(d.loc["2024-12-31", "year_month_sort"], 202412)

    def test_allocation_and_profit_identities(self):
        f = ST.fact_load_trip
        mr = ST.month_maint_ratio
        covered = mr[mr["miles"].fillna(0) > 0]["maintenance_cost"].sum()
        self.assertAlmostEqual(f["maint_alloc"].sum(), covered, places=4)
        self.assertAlmostEqual(f["maint_alloc"].sum(), SRC["maintenance_record"]["total_cost"].sum(), places=4)   # synthetic: every month has miles
        gp = f["revenue"] - f["fuel_cost"] - f["maint_alloc"]
        self.assertTrue(np.allclose(f["gross_profit"], gp))

    def test_fact_columns_follow_the_trip(self):
        f = ST.fact_load_trip
        no_trip = f[f["trip_id"].isna()]
        self.assertGreater(len(no_trip), 0)
        self.assertTrue(no_trip["miles"].isna().all() and no_trip["expected_miles"].isna().all())
        self.assertTrue((no_trip["maint_alloc"] == 0).all())
        with_trip = f[f["trip_id"].notna()]
        self.assertTrue(with_trip["expected_miles"].notna().all())

    def test_dimensions_attributes(self):
        d = ST.dim_driver.set_index("driver_id")
        self.assertEqual(d.loc[DRIVER_UNKNOWN, "driver_name"], DRIVER_UNKNOWN)
        self.assertEqual(d.loc[DRIVER_UNKNOWN, "tenure_band"], "Unknown")
        self.assertEqual(d.loc[DRIVER_UNKNOWN, "tenure_band_sort"], 5)
        self.assertEqual(d["tenure_band"].isin(["0-1 yr", "1-3 yrs", "3-6 yrs", "6+ yrs", "Unknown"]).all(), True)
        t = ST.dim_truck
        blank = t[t["model_year"].isna()]
        self.assertGreater(len(blank), 0)
        self.assertTrue(blank["truck_age"].isna().all() and (blank["truck_age_band"] == "Unknown").all())
        known = t[t["model_year"].notna()]
        self.assertTrue((known["truck_age"] == ST.as_of.year - known["model_year"]).all())
        self.assertTrue(known["truck_age_band"].isin(["0-2 yrs", "3-5 yrs", "6+ yrs"]).all())
        self.assertEqual(ST.dim_route["route_name"].iloc[0], f"{ST.dim_route['origin_city'].iloc[0]} \u2192 {ST.dim_route['destination_city'].iloc[0]}")
        self.assertEqual(ST.dim_margin_bridge["item"].tolist(), ["Revenue", "Fuel", "Maintenance", "Gross Profit"])
        self.assertEqual(set(ST.dim_network_city["city"]), set(SRC["routes"]["origin_city"]) | set(SRC["routes"]["destination_city"]))

    def test_child_fact_keys(self):
        for name in ("fact_fuel_purchase", "fact_delivery_event", "fact_safety_incident"):
            t = getattr(ST, name)
            with_trip = t[t["trip_id"].notna()]
            self.assertTrue(with_trip["load_date"].notna().all(), name)
            self.assertTrue(with_trip["customer_id"].notna().all(), name)
        orphans = ST.fact_fuel_purchase[ST.fact_fuel_purchase["trip_id"].isna()]
        self.assertGreater(len(orphans), 0)                       # fuel rows with no trip keep NULL keys
        self.assertTrue(orphans["load_date"].isna().all())
        self.assertEqual(len(ST.fact_fuel_purchase), len(SRC["fuel_purchases"]))
        self.assertEqual(len(ST.fact_delivery_event), len(SRC["delivery_events"]))

    def test_delivery_derived_columns(self):
        e = ST.fact_delivery_event
        self.assertEqual(e["is_delivery"].sum(), (SRC["delivery_events"]["event_type"] == "Delivery").sum())
        self.assertTrue(np.allclose(e["detention_hours"], e["detention_minutes"] / 60))
        zero = e[e["detention_minutes"] == 0]
        self.assertTrue((zero["detention_bucket"] == "None").all() and (zero["detention_bucket_sort"] == 1).all())
        big = e[e["detention_minutes"] > 240]
        self.assertTrue((big["detention_bucket"] == "4 h+").all())

    def test_duplicate_trip_is_rejected(self):
        bad = dict(SRC)
        bad["trips"] = pd.concat([SRC["trips"], SRC["trips"].iloc[[0]]], ignore_index=True)
        with self.assertRaises(StarBuildError):
            build_star(bad)
        bad2 = dict(SRC)
        bad2["loads"] = pd.concat([SRC["loads"], SRC["loads"].iloc[[0]]], ignore_index=True)
        with self.assertRaises(StarBuildError):
            build_star(bad2)

    def test_save_and_load_round_trip(self):
        from logistics_analytics.measures import m_finance, m_service
        with tempfile.TemporaryDirectory() as d:
            save_star(ST, d)
            again = load_star(d)
        self.assertEqual(again.as_of, ST.as_of)
        for n, t in ST.tables().items():
            self.assertEqual(len(again.tables()[n]), len(t), n)
        self.assertAlmostEqual(m_finance(again)["total_revenue"].iloc[0], m_finance(ST)["total_revenue"].iloc[0], places=3)
        self.assertEqual(m_service(again)["on_time_deliveries"].iloc[0], m_service(ST)["on_time_deliveries"].iloc[0])
        self.assertEqual(str(again.dim_truck["model_year"].dtype), "Int64")
        self.assertEqual(again.fact_delivery_event["is_delivery"].dtype, bool)


class TestScope(unittest.TestCase):
    def test_default_is_everything(self):
        self.assertEqual(len(scope_load(ST)), len(ST.fact_load_trip))
        self.assertEqual(len(scope_delivery(ST)), len(ST.fact_delivery_event))
        self.assertEqual(len(scope_fuel(ST)), len(ST.fact_fuel_purchase))
        self.assertEqual(len(scope_safety(ST)), len(ST.fact_safety_incident))
        self.assertEqual(len(scope_maintenance(ST)), len(ST.fact_maintenance_event))

    def test_dates_are_inclusive_and_parsed(self):
        f = ST.fact_load_trip
        day = f["load_date"].iloc[0]
        got = scope_load(ST, Scope(start=day, end=day))
        self.assertEqual(len(got), (f["load_date"] == day).sum())
        sc = Scope(start="2023-01-01", end="2023-12-31")
        self.assertEqual(sc.start, pd.Timestamp("2023-01-01"))
        self.assertEqual(len(scope_load(ST, sc)), ((f["load_date"] >= "2023-01-01") & (f["load_date"] <= "2023-12-31")).sum())

    def test_entity_filters(self):
        f = ST.fact_load_trip
        c = f["customer_id"].value_counts().index[0]
        self.assertEqual(len(scope_load(ST, Scope(customer=c))), (f["customer_id"] == c).sum())
        self.assertEqual(len(scope_load(ST, Scope(customer="NOPE"))), 0)
        t = f["truck_id"].dropna().iloc[0]
        self.assertTrue((scope_load(ST, Scope(truck=t))["truck_id"] == t).all())
        self.assertEqual(len(scope_load(ST, Scope(booking="Contract"))), (f["booking_type"] == "Contract").sum())

    def test_children_follow_the_load_filters(self):
        c = ST.fact_load_trip["customer_id"].value_counts().index[0]
        trips = set(ST.fact_load_trip.loc[ST.fact_load_trip["customer_id"] == c, "trip_id"].dropna())
        ev = SRC["delivery_events"]
        self.assertEqual(len(scope_delivery(ST, Scope(customer=c))), ev["trip_id"].isin(trips).sum())
        self.assertEqual(len(scope_safety(ST, Scope(customer=c))), SRC["safety_incidents"]["trip_id"].isin(trips).sum())
        self.assertEqual(len(scope_fuel(ST, Scope(customer=c))), SRC["fuel_purchases"]["trip_id"].isin(trips).sum())

    def test_date_filter_drops_rows_without_a_trip(self):
        # a fuel row with no trip has no load date: counted with no filter, dropped under a date filter (as in the report)
        everything = len(scope_fuel(ST))
        dated = len(scope_fuel(ST, Scope(start="2000-01-01")))
        self.assertEqual(everything - dated, SRC["fuel_purchases"]["trip_id"].isna().sum())

    def test_extra_filters_touch_only_their_table(self):
        base = len(scope_load(ST))
        self.assertEqual(len(scope_load(ST, Scope(facility="FAC001", incident_type="Collision", state="TX", maintenance_type="Inspection"))), base)
        self.assertLess(len(scope_delivery(ST, Scope(facility="FAC001"))), len(ST.fact_delivery_event))
        self.assertTrue((scope_safety(ST, Scope(incident_type="Collision"))["incident_type"] == "Collision").all())
        self.assertTrue((scope_fuel(ST, Scope(state="TX"))["location_state"] == "TX").all())

    def test_maintenance_ignores_customer_and_route(self):
        self.assertEqual(len(scope_maintenance(ST, Scope(customer="CUST00001", route="RT0001"))), len(ST.fact_maintenance_event))
        m = scope_maintenance(ST, Scope(start="2023-01-01", end="2023-06-30", maintenance_type="Inspection"))
        self.assertTrue((m["maintenance_type"] == "Inspection").all() and m["maintenance_date"].between("2023-01-01", "2023-06-30").all())

    def test_month_scopes_include_the_starting_month(self):
        tm = scope_truck_month(ST, Scope(start="2023-03-15", end="2023-05-10"))
        self.assertEqual(sorted(tm["month_start"].unique()), [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-01"), pd.Timestamp("2023-05-01")][:len(tm["month_start"].unique())])
        self.assertTrue((tm["month_start"] >= "2023-03-01").all() and (tm["month_start"] <= "2023-05-10").all())
        d = ST.fact_driver_month["driver_id"].iloc[0]
        self.assertTrue((scope_driver_month(ST, Scope(driver=d))["driver_id"] == d).all())

    def test_prior_year_and_bounds(self):
        sc = Scope(start="2024-02-29", end="2024-12-31")
        py = sc.prior_year(ST)
        self.assertEqual(py.start, pd.Timestamp("2023-02-28"))
        self.assertEqual(py.end, pd.Timestamp("2023-12-31"))
        lo, hi = Scope().bounds(ST)
        self.assertEqual((lo, hi), (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")))
        self.assertEqual(Scope().prior_year(ST).start, pd.Timestamp("2021-01-01"))

    def test_scope_is_immutable(self):
        s = Scope(customer="A")
        s2 = s.with_(customer="B")
        self.assertEqual((s.customer, s2.customer), ("A", "B"))
        with self.assertRaises(Exception):
            s.customer = "C"


if __name__ == "__main__":
    unittest.main()
