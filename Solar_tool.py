import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import numpy_financial as npf

# -------------------------
# Data directory
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# -------------------------
# Load uploaded profiles
# -------------------------
def load_data():
    demand_upload = st.file_uploader("Upload half-hourly building demand profile (CSV)", type="csv", key="demand")
    solar_upload = st.file_uploader("Upload half-hourly solar generation profile (CSV)", type="csv", key="solar")
    
    demand_df = None
    solar_df = None
    
    if demand_upload is not None:
        demand_df = pd.read_csv(demand_upload, parse_dates=True, index_col=0)
    if solar_upload is not None:
        solar_df = pd.read_csv(solar_upload, parse_dates=True, index_col=0)
    
    original_kwp = None
    if solar_df is not None:
        original_kwp = st.number_input("Original kWp of the uploaded solar profile", min_value=0.1, value=100.0, step=10.0)
    
    return solar_df, original_kwp, demand_df

# -------------------------
# Financial simulation
# -------------------------
def simulate(solar_gen, original_kwp, demand, solar_kWp, capex_per_kW, opex_per_kW, ppa_rate,
             import_tariff, export_tariff, project_life, replace_years, inflation, export_allowed, model):
    
    results = []
    annual_cashflows = []
    
    capex = solar_kWp * capex_per_kW
    opex = solar_kWp * opex_per_kW
    replacement_cost = 0.2 * capex  # Inverter replacement assumption
    
    scale_factor = solar_kWp / original_kwp
    scaled_solar = solar_gen * scale_factor
    
    for year in range(1, project_life + 1):
        adj_factor = (1 + inflation) ** (year - 1)
        demand_total = demand.sum()
        
        overlap = np.minimum(scaled_solar, demand)
        export = scaled_solar - overlap if export_allowed else np.maximum(scaled_solar - demand, 0)
        grid_import = demand - overlap
        
        if model == "Owner Occupier":
            savings = overlap.sum() * import_tariff + export.sum() * export_tariff - opex
            if year == 1:
                savings -= capex
            annual_cashflows.append(savings)
        elif model == "Landlord Funded (PPA to Tenant)":
            cashflow = overlap.sum() * ppa_rate + export.sum() * export_tariff - opex
            if year == 1:
                cashflow -= capex
            annual_cashflows.append(cashflow)
        
        if year in replace_years:
            annual_cashflows[-1] -= replacement_cost * adj_factor
        
        results.append({
            'Year': year,
            'Demand (kWh)': demand_total,
            'Solar (kWh)': scaled_solar.sum(),
            'Self Consumption (kWh)': overlap.sum(),
            'Exported (kWh)': export.sum(),
            'Grid Import (kWh)': grid_import.sum(),
            'Annual Cashflow': annual_cashflows[-1]
        })
    
    df = pd.DataFrame(results)
    
    irr = npf.irr(annual_cashflows)
    npv = npf.npv(0.07, annual_cashflows)
    
    financials = {model: {"NPV": npv, "IRR": irr}}
    
    return df, financials, scaled_solar, capex

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.title("☀️ Solar Modelling Tool")
    
    solar_df, original_kwp, demand_df = load_data()
    
    if solar_df is not None and original_kwp is not None and demand_df is not None:
        st.sidebar.header("Inputs")
        solar_kWp = st.sidebar.number_input("Proposed System Size (kWp)", 1.0, 1000.0, 100.0)
        capex_per_kW = st.sidebar.number_input("CAPEX (£/kW)", 0.0, 5000.0, 800.0)
        opex_per_kW = st.sidebar.number_input("O&M (£/kW/year)", 0.0, 200.0, 20.0)
        ppa_rate = st.sidebar.number_input("PPA Rate (£/kWh)", 0.0, 1.0, 0.05)
        import_tariff = st.sidebar.number_input("Import Tariff (£/kWh)", 0.0, 1.0, 0.25)
        export_tariff = st.sidebar.number_input("Export Tariff (£/kWh)", 0.0, 1.0, 0.05)
        project_life = st.sidebar.number_input("Project Lifespan (years)", 1, 50, 25)
        replace_years = st.sidebar.multiselect("Replacement Years", list(range(1, 51)), [15])
        inflation = st.sidebar.number_input("Annual Inflation Rate (%)", 0.0, 10.0, 2.0) / 100
        export_allowed = st.sidebar.checkbox("Export Allowed?", True)
        
        model = st.radio("Financial Model", ["Owner Occupier", "Landlord Funded (PPA to Tenant)"])
        
        if st.button("Run Simulation"):
            df, financials, scaled_solar, capex = simulate(
                solar_df.iloc[:, 0], original_kwp, demand_df.iloc[:, 0], solar_kWp,
                capex_per_kW, opex_per_kW, ppa_rate, import_tariff, export_tariff,
                project_life, replace_years, inflation, export_allowed, model
            )
            
            st.header("Simulation Results")
            st.dataframe(df)
            
            st.write(f"**Initial CAPEX: £{capex:,.2f}**")
            st.write(f"**NPV: £{financials[model]['NPV']:,.2f}**")
            st.write(f"**IRR: {financials[model]['IRR']*100:.2f}%**")
            
            st.header("Visualisation")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['Year'], df['Annual Cashflow'], label="Annual Cashflow")
            ax.set_xlabel("Year")
            ax.set_ylabel("£")
            ax.set_title("Annual Cashflow over Project Life")
            ax.legend()
            st.pyplot(fig)
    
    else:
        st.info("Please upload both solar generation and demand CSV files and enter original kWp.")

if __name__ == "__main__":
    main()
