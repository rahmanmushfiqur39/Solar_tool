import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import numpy_financial as npf  # Only for IRR

# -------------------------
# Data directory
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# -------------------------
# Helpers
# -------------------------
def _get_series(df):
    if df is None or df.empty:
        return None, None
    col_time = df.columns[0]
    col_val = df.columns[1]
    try:
        t = pd.to_datetime(df[col_time])
    except Exception:
        t = df[col_time]
    v = pd.to_numeric(df[col_val], errors="coerce").fillna(0.0)
    return t, v

def _payback_year(cashflows):
    cum = np.cumsum(cashflows)
    for i, c in enumerate(cum, start=1):
        if c >= 0:
            return i
    return None

# -------------------------
# Load benchmark and solar profiles
# -------------------------
def load_profiles():
    profiles = {
        "Benchmark_Office": pd.read_csv(os.path.join(DATA_DIR, "Benchmark_Profile_Office.csv")),
        "Benchmark_Storage": pd.read_csv(os.path.join(DATA_DIR, "Benchmark_Profile_Storage.csv")),
        "Solar_South": pd.read_csv(os.path.join(DATA_DIR, "Solar_Profile_South.csv")),
        "Solar_Midlands": pd.read_csv(os.path.join(DATA_DIR, "Solar_Profile_Midlands.csv")),
        "Solar_Scotland": pd.read_csv(os.path.join(DATA_DIR, "Solar_Profile_Scotland.csv")),
    }
    return profiles

# -------------------------
# Energy + Financial calculations
# -------------------------
def prepare_year1_timeseries(
    system_size_kwp,
    solar_df,
    demand_df,
    used_regional_solar=False,
    used_benchmark_demand=False,
    site_type="Office",
    floor_space_m2=0.0,
    intensity_office=65.0,
    intensity_storage=27.0,
    export_allowed=True
):
    t_s, v_s = _get_series(solar_df)
    if used_regional_solar:
        solar_hh = (v_s.astype(float) * float(system_size_kwp)).values
    else:
        solar_hh = v_s.astype(float).values

    t_d, v_d = _get_series(demand_df)
    if used_benchmark_demand:
        intensity = intensity_office if site_type == "Office" else intensity_storage
        annual_demand = float(intensity) * float(floor_space_m2)
        shape = v_d.astype(float).values
        shape_sum = shape.sum()
        demand_hh = np.zeros_like(shape) if shape_sum <= 0 else shape / shape_sum * annual_demand
    else:
        demand_hh = v_d.astype(float).values

    n = min(len(solar_hh), len(demand_hh))
    solar_hh = solar_hh[:n]
    demand_hh = demand_hh[:n]
    if t_s is not None and len(t_s) >= n:
        index = t_s[:n]
    else:
        index = pd.RangeIndex(0, n, 1)

    return index, solar_hh, demand_hh

def aggregate_monthly_savings(
    index,
    solar_hh,
    demand_hh,
    model,
    import_tariff,
    export_tariff,
    ppa_rate,
    export_allowed=True
):
    df = pd.DataFrame({
        "ts": pd.to_datetime(index, errors="coerce"),
        "solar": solar_hh,
        "demand": demand_hh
    })
    if df["ts"].isna().all():
        df["month"] = ((df.index.values / (len(df) / 12))).astype(int) + 1
        df["month"] = df["month"].clip(1, 12)
    else:
        df["month"] = df["ts"].dt.month

    self_cons = np.minimum(df["solar"], df["demand"])
    export = np.maximum(df["solar"] - df["demand"], 0.0) if export_allowed else 0.0

    if model == "Owner Occupier":
        savings_gbp = self_cons * import_tariff + export * export_tariff
    else:
        savings_gbp = self_cons * max(import_tariff - ppa_rate, 0.0)

    monthly = savings_gbp.groupby(df["month"]).sum()
    monthly = monthly.reindex(range(1, 13), fill_value=0.0)
    return monthly

def calculate_project_financials(
    model,
    system_size_kwp,
    solar_df,
    demand_df,
    project_life=25,
    capex_per_kwp=800,
    opex_per_kwp=15,
    import_tariff=0.25,
    export_tariff=0.08,
    ppa_rate=0.18,
    replace_years=[15],
    inflation=0.0,
    export_allowed=True,
    used_regional_solar=False,
    used_benchmark_demand=False,
    site_type="Office",
    floor_space_m2=0.0,
    inverter_replacement_cost_per_kwp=50.0
):
    degradation = 0.005

    index, solar_hh, demand_hh = prepare_year1_timeseries(
        system_size_kwp,
        solar_df,
        demand_df,
        used_regional_solar=used_regional_solar,
        used_benchmark_demand=used_benchmark_demand,
        site_type=site_type,
        floor_space_m2=floor_space_m2,
        export_allowed=export_allowed
    )

    annual_yield_y1 = float(np.sum(solar_hh))
    self_consumption_y1 = float(np.sum(np.minimum(solar_hh, demand_hh)))
    export_y1 = float(np.sum(np.maximum(solar_hh - demand_hh, 0.0))) if export_allowed else 0.0
    cons_ratio = (self_consumption_y1 / annual_yield_y1) if annual_yield_y1 > 0 else 0.0

    years = list(range(1, project_life + 1))
    gen_y = [annual_yield_y1 * ((1 - degradation) ** (y - 1)) for y in years]
    selfc_y = [g * cons_ratio for g in gen_y]
    export_y = [(g - s) if export_allowed else 0.0 for g, s in zip(gen_y, selfc_y)]

    capex = float(system_size_kwp) * float(capex_per_kwp)
    annual_opex_y = [float(system_size_kwp) * float(opex_per_kwp) * ((1 + float(inflation)) ** (y - 1)) for y in years]

    owner_cf, landlord_cf, tenant_cf = [], [], []

    if model == "Owner Occupier":
        for i, y in enumerate(years):
            # inflow is savings + export revenue (these are the same "income/savings")
            inflow = selfc_y[i] * import_tariff + export_y[i] * export_tariff
            net = inflow - annual_opex_y[i]
            if y == 1:
                net -= capex
            if y in replace_years:
                net -= float(inverter_replacement_cost_per_kwp) * float(system_size_kwp)
            owner_cf.append(net)
    else:
        for i, y in enumerate(years):
            landlord_revenue = selfc_y[i] * ppa_rate + export_y[i] * export_tariff
            landlord_costs = annual_opex_y[i]
            net_landlord = landlord_revenue - landlord_costs
            if y == 1:
                net_landlord -= capex
            if y in replace_years:
                net_landlord -= float(inverter_replacement_cost_per_kwp) * float(system_size_kwp)
            landlord_cf.append(net_landlord)

            tenant_sav = selfc_y[i] * ma*
