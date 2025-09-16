import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -------------------------
# Load Data
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_profiles():
    profiles = {
        "Office": pd.read_csv(os.path.join(DATA_DIR, "Benchmark_Profile_Office.csv")),
        "Storage": pd.read_csv(os.path.join(DATA_DIR, "Benchmark_Profile_Storage.csv")),
        "South": pd.read_csv(os.path.join(DATA_DIR, "Solar_Profile_South.csv")),
        "Midlands": pd.read_csv(os.path.join(DATA_DIR, "Solar_Profile_Midlands.csv")),
        "Scotland": pd.read_csv(os.path.join(DATA_DIR, "Solar_Profile_Scotland.csv")),
    }
    return profiles

# -------------------------
# Financial Calculations
# -------------------------
def calculate_financials(model, system_size_kw, solar_profile, demand_profile):
    years = list(range(1, 26))
    degradation = 0.005  # 0.5%/year
    capex_per_kw = 800
    opex_per_kw = 15
    tariff_import = 0.25
    tariff_export = 0.08
    ppa_price = 0.18

    capex = system_size_kw * capex_per_kw
    opex = system_size_kw * opex_per_kw

    cashflows = []
    npv_landlord = npv_tenant = npv_owner = 0
    irr_landlord = irr_tenant = irr_owner = None

    for y in years:
        solar_yield = solar_profile.sum() * (1 - degradation) ** (y - 1)
        demand = demand_profile.sum()
        self_consumption = min(solar_yield, demand)
        export = max(solar_yield - demand, 0)

        if model == "Owner Occupier":
            savings = self_consumption * tariff_import
            export_rev = export * tariff_export
            net = savings + export_rev - opex
            if y == 1:
                net -= capex
            cashflows.append(net)

        elif model == "Landlord/Tenant (Lease or Service Charge)":
            # Landlord pays capex + opex, tenant gets savings
            savings = self_consumption * tariff_import
            export_rev = export * tariff_export

            landlord_cf = export_rev - opex
            tenant_cf = savings
            if y == 1:
                landlord_cf -= capex

            cashflows.append((landlord_cf, tenant_cf))

        elif model == "PPA":
            # Landlord installs, tenant buys at ppa_price
            solar_used = min(solar_yield, demand)
            landlord_rev = solar_used * ppa_price - opex
            tenant_savings = (solar_used * (tariff_import - ppa_price))
            if y == 1:
                landlord_rev -= capex
            cashflows.append((landlord_rev, tenant_savings))

    # Convert to DataFrame
    if model == "Owner Occupier":
        df = pd.DataFrame({"Year": years, "Owner Cashflow": cashflows})
        irr_owner = np.irr(cashflows)
        npv_owner = np.npv(0.07, cashflows)
        return df, {"Owner Occupier": {"NPV": npv_owner, "IRR": irr_owner}}

    else:
        df = pd.DataFrame({
            "Year": years,
            "Landlord Cashflow": [c[0] for c in cashflows],
            "Tenant Cashflow": [c[1] for c in cashflows],
        })
        landlord_cf = [c[0] for c in cashflows]
        tenant_cf = [c[1] for c in cashflows]
        irr_landlord = np.irr(landlord_cf)
        irr_tenant = np.irr(tenant_cf) if model != "PPA" else None
        npv_landlord = np.npv(0.07, landlord_cf)
        npv_tenant = np.npv(0.07, tenant_cf) if model != "PPA" else None

        return df, {
            "Landlord": {"NPV": npv_landlord, "IRR": irr_landlord},
            "Tenant": {"NPV": npv_tenant, "IRR": irr_tenant},
        }

# -------------------------
# PDF Export
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
    table_data = [["", *financials.keys()]]
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
# Streamlit App
# -------------------------
def main():
    st.title("☀️ Solar Tool")

    profiles = load_profiles()

    # Inputs
    site_type = st.selectbox("Select Site Type", ["Office", "Storage"])
    region = st.selectbox("Select Region", ["South", "Midlands", "Scotland"])
    system_size = st.number_input("System Size (kW)", min_value=10, max_value=5000, value=500, step=10)
    model = st.radio("Select Business Model", ["Owner Occupier", "Landlord/Tenant (Lease or Service Charge)", "PPA"])

    if st.button("Run Analysis"):
        demand_profile = profiles[site_type]
        solar_profile = profiles[region]

        df, financials = calculate_financials(model, system_size, solar_profile, demand_profile)

        st.subheader("Summary")
        st.write({
            "Site Type": site_type,
            "Region": region,
            "System Size (kW)": system_size,
            "Model": model,
        })

        st.subheader("Financial Metrics")
        st.dataframe(financials)

        st.subheader("Cashflow")
        fig, ax = plt.subplots()
        if model == "Owner Occupier":
            ax.plot(df["Year"], df["Owner Cashflow"], label="Owner")
        else:
            ax.plot(df["Year"], df["Landlord Cashflow"], label="Landlord")
            ax.plot(df["Year"], df["Tenant Cashflow"], label="Tenant")
        ax.legend()
        ax.set_ylabel("£ per year")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.pyplot(fig)

        # PDF Export
        pdf_buf = export_pdf(
            {"Site Type": site_type, "Region": region, "System Size (kW)": system_size, "Model": model},
            financials,
            df,
            buf,
        )
        st.download_button("Download PDF Report", data=pdf_buf, file_name="report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
