import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import numpy_financial as npf  # Fixed IRR/NPV

# -------------------------
# Data directory
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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
# Financial and energy calculations
# -------------------------
def calculate_financials(model, system_size_kw, solar_profile, demand_profile, project_life=25):
    capex_per_kw = 800
    opex_per_kw = 15
    import_tariff = 0.25
    export_tariff = 0.08
    ppa_rate = 0.18
    degradation = 0.005  # 0.5% per year

    capex = system_size_kw * capex_per_kw
    opex = system_size_kw * opex_per_kw

    years = list(range(1, project_life + 1))
    cashflows = []

    for y in years:
        solar_yield = solar_profile.iloc[:, 1] * (1 - degradation) ** (y - 1)
        demand = demand_profile.iloc[:, 1]

        self_consumption = np.minimum(solar_yield, demand).sum()
        export = np.maximum(solar_yield - demand, 0).sum()

        if model == "Owner Occupier":
            savings = self_consumption * import_tariff
            export_rev = export * export_tariff
            net = savings + export_rev - opex
            if y == 1:
                net -= capex
            cashflows.append(net)

        elif model == "Landlord Funded (PPA to Tenant)":
            landlord_cf = self_consumption * ppa_rate + export * export_tariff - opex
            if y == 1:
                landlord_cf -= capex
            cashflows.append(landlord_cf)

    # Convert to DataFrame
    if model == "Owner Occupier":
        df = pd.DataFrame({"Year": years, "Owner Cashflow": cashflows})
        irr_owner = npf.irr(cashflows)
        npv_owner = npf.npv(0.07, cashflows)
        return df, {"Owner Occupier": {"NPV": npv_owner, "IRR": irr_owner}}
    else:
        df = pd.DataFrame({"Year": years, "Landlord Cashflow": cashflows})
        irr_landlord = npf.irr(cashflows)
        npv_landlord = npf.npv(0.07, cashflows)
        return df, {"Landlord": {"NPV": npv_landlord, "IRR": irr_landlord}}

# -------------------------
# PDF export
# -------------------------
def export_pdf(summary, financials, df, chart_buf):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    # Logo
    logo_path = os.path.join(DATA_DIR, "savills_logo.png")
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=100, height=50, hAlign="RIGHT"))
    story.append(Spacer(1, 12))

    # Summary
    story.append(Paragraph("<b>Summary Inputs</b>", styles['Heading2']))
    for k, v in summary.items():
        story.append(Paragraph(f"{k}: {v}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Financials
    story.append(Paragraph("<b>Financial Metrics</b>", styles['Heading2']))
    table_data = [["Metric", *financials.keys()]]
    metrics = ["NPV (£)", "IRR (%)"]
    for m in metrics:
        row = [m]
        for actor in financials.keys():
            val = financials[actor][m.split()[0]]
            if val is None:
                row.append("-")
            elif "NPV" in m:
                row.append(f"£{val:,.0f}")
            else:
                row.append(f"{val*100:.1f}%")
        table_data.append(row)

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Chart
    story.append(Paragraph("<b>Cashflow</b>", styles['Heading2']))
    story.append(Image(chart_buf, width=400, height=200))
    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------
# Streamlit app
# -------------------------
def main():
    st.title("☀️ Solar Modelling Tool")

    profiles = load_profiles()

    # ---- Inputs ----
    st.subheader("Demand Profile")
    demand_option = st.radio("Do you have half-hourly demand profile?", ["Upload CSV", "Use Benchmark Profile"])
    if demand_option == "Upload CSV":
        demand_file = st.file_uploader("Upload demand CSV", type="csv", key="demand_upload")
        demand_profile = pd.read_csv(demand_file) if demand_file else None
    else:
        site_type = st.selectbox("Select Site Type", ["Office", "Storage"])
        benchmark_key = "Benchmark_Office" if site_type == "Office" else "Benchmark_Storage"
        demand_profile = profiles[benchmark_key]

    st.subheader("Solar Profile")
    solar_option = st.radio("Do you have half-hourly solar profile?", ["Upload CSV", "Use Regional Profile"])
    if solar_option == "Upload CSV":
        solar_file = st.file_uploader("Upload solar CSV", type="csv", key="solar_upload")
        solar_profile = pd.read_csv(solar_file) if solar_file else None
    else:
        region = st.selectbox("Select Region", ["South", "Midlands", "Scotland"])
        solar_profile = profiles[f"Solar_{region}"]

    system_size = st.number_input("System Size (kW)", min_value=10, max_value=5000, value=500, step=10)
    model = st.radio("Financial Model", ["Owner Occupier", "Landlord Funded (PPA to Tenant)"])

    # Run simulation button
    if st.button("Run / Update Simulation"):
        if demand_profile is not None and solar_profile is not None:
            df, financials = calculate_financials(model, system_size, solar_profile, demand_profile)

            # Summary
            st.subheader("Summary Inputs")
            st.write({
                "System Size (kW)": system_size,
                "Financial Model": model,
            })

            # Financials
            st.subheader("Financial Metrics")
            st.dataframe(financials)

            # Cashflow plot
            st.subheader("Yearly Cashflow")
            fig, ax = plt.subplots()
            if model == "Owner Occupier":
                ax.plot(df["Year"], df["Owner Cashflow"], label="Owner")
            else:
                ax.plot(df["Year"], df["Landlord Cashflow"], label="Landlord")
            ax.set_ylabel("£ per year")
            ax.set_xlabel("Year")
            ax.legend()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.pyplot(fig)

            # PDF export
            pdf_buf = export_pdf(
                {"System Size (kW)": system_size, "Financial Model": model},
                financials,
                df,
                buf
            )
            st.download_button("Download PDF Report", data=pdf_buf, file_name="report.pdf", mime="application/pdf")
        else:
            st.warning("Please provide both demand and solar profiles.")

if __name__ == "__main__":
    main()
