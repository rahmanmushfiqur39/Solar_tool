import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import numpy_financial as npf  # for IRR

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
def calculate_financials(model,
                         system_size_kw,
                         solar_profile,
                         demand_profile,
                         project_life=25,
                         capex_per_kw=800,
                         opex_per_kw=15,
                         import_tariff=0.25,
                         export_tariff=0.08,
                         ppa_rate=0.18,
                         replace_years=None,
                         inflation=0.02,
                         export_allowed=True):
    if replace_years is None:
        replace_years = []

    degradation = 0.005  # 0.5% per year degradation
    capex = system_size_kw * capex_per_kw
    opex = system_size_kw * opex_per_kw

    years = list(range(1, project_life + 1))
    cashflows = []

    # Ensure second column used (datetime is first column)
    # solar_profile and demand_profile expected as DataFrame with values in column index 1
    for y in years:
        solar_yield_series = solar_profile.iloc[:, 1] * (1 - degradation) ** (y - 1)
        demand_series = demand_profile.iloc[:, 1]

        # per half-hour arrays
        self_consumption_total = np.minimum(solar_yield_series, demand_series).sum()
        export_total = np.maximum(solar_yield_series - demand_series, 0).sum() if export_allowed else 0.0

        if model == "Owner Occupier":
            savings = self_consumption_total * import_tariff
            export_rev = export_total * export_tariff
            net = savings + export_rev - opex
            if y == 1:
                net -= capex
            if y in replace_years:
                net -= 0.2 * capex
            cashflows.append(net)

        elif model == "Landlord Funded (PPA to Tenant)":
            # Landlord receives ppa for self_consumption, export rev, pays opex and capex
            landlord_cf = self_consumption_total * ppa_rate + export_total * export_tariff - opex
            if y == 1:
                landlord_cf -= capex
            if y in replace_years:
                landlord_cf -= 0.2 * capex
            cashflows.append(landlord_cf)

    # Build DataFrame and IRR-only results
    if model == "Owner Occupier":
        df = pd.DataFrame({"Year": years, "Owner Cashflow": cashflows})
        irr_owner = None
        try:
            irr_owner = npf.irr(cashflows)
        except Exception:
            irr_owner = None
        return df, {"Owner Occupier": {"IRR": irr_owner}}
    else:
        df = pd.DataFrame({"Year": years, "Landlord Cashflow": cashflows})
        irr_landlord = None
        try:
            irr_landlord = npf.irr(cashflows)
        except Exception:
            irr_landlord = None
        return df, {"Landlord": {"IRR": irr_landlord}}

# -------------------------
# PDF export (logo top-right, keep aspect ratio)
# -------------------------
def export_pdf(summary, financials, df, chart_buf):
    """
    summary: dict of input values
    financials: dict like {"Owner Occupier": {"IRR": 0.12}} or {"Landlord": {"IRR": 0.10}}
    df: cashflow dataframe
    chart_buf: BytesIO containing PNG chart (seeked to 0)
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    # Logo (top-right) with aspect ratio preserved
    logo_path = os.path.join(DATA_DIR, "savills_logo.png")
    if os.path.exists(logo_path):
        try:
            logo_reader = ImageReader(logo_path)
            lw, lh = logo_reader.getSize()
            target_w = 120.0
            target_h = (lh / lw) * target_w
            story.append(Image(logo_reader, width=target_w, height=target_h, hAlign="RIGHT"))
        except Exception:
            # fallback: add path (ReportLab can accept path)
            story.append(Image(logo_path, width=120, hAlign="RIGHT"))
    story.append(Spacer(1, 12))

    # Title under logo
    story.append(Paragraph("<b>Solar Modelling Tool - Summary Report</b>", styles['Title']))
    story.append(Spacer(1, 8))

    # Summary inputs
    story.append(Paragraph("<b>Summary Inputs</b>", styles['Heading2']))
    for k, v in summary.items():
        story.append(Paragraph(f"{k}: {v}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Financials table (IRR only)
    story.append(Paragraph("<b>Financial Metrics</b>", styles['Heading2']))
    header = ["Metric", *financials.keys()]
    table_data = [header]
    # IRR (%)
    row = ["IRR (%)"]
    for actor in financials.keys():
        irr_val = financials[actor].get("IRR")
        row.append("-" if irr_val is None else f"{irr_val * 100:.1f}%")
    table_data.append(row)

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Cashflow chart: chart_buf is a BytesIO (PNG). Use ImageReader to preserve size.
    if chart_buf is not None:
        try:
            chart_buf.seek(0)
            chart_reader = ImageReader(chart_buf)
            cw, ch = chart_reader.getSize()
            target_w = 420.0  # reasonable width in points
            target_h = (ch / cw) * target_w
            story.append(Paragraph("<b>Yearly Cashflow</b>", styles['Heading2']))
            story.append(Image(chart_reader, width=target_w, height=target_h))
        except Exception:
            # If image fails, skip chart
            story.append(Paragraph("Chart preview unavailable", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------
# Streamlit app
# -------------------------
def main():
    # Keep default layout (not wide)
    st.set_page_config(page_title="Solar Modelling Tool", layout="centered")

    # Logo + title in one row
    col1, col2 = st.columns([1, 5])
    with col1:
        logo_path = os.path.join(DATA_DIR, "savills_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)
    with col2:
        st.title("Solar Modelling Tool")

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

    # Main inputs
    system_size = st.number_input("System Size (kW)", min_value=10, max_value=5000, value=500, step=10)
    capex_per_kw = st.number_input("CAPEX (£/kW)", 0.0, 5000.0, 800.0)
    opex_per_kw = st.number_input("O&M (£/kW/year)", 0.0, 200.0, 15.0)
    ppa_rate = st.number_input("PPA Rate (£/kWh)", 0.0, 1.0, 0.18)
    import_tariff = st.number_input("Import Tariff (£/kWh)", 0.0, 1.0, 0.25)
    export_tariff = st.number_input("Export Tariff (£/kWh)", 0.0, 1.0, 0.08)
    project_life = st.number_input("Project Lifespan (years)", 1, 50, 25)
    inflation = st.number_input("Annual Inflation Rate (%)", 0.0, 10.0, 2.0) / 100
    replace_years = st.multiselect("Replacement Years", list(range(1, 51)), [15])
    export_allowed = st.checkbox("Export Allowed?", True)
    model = st.radio("Financial Model", ["Owner Occupier", "Landlord Funded (PPA to Tenant)"])

    # Run simulation
    if st.button("Run / Update Simulation"):
        if demand_profile is not None and solar_profile is not None:
            # run calculation
            df, financials = calculate_financials(
                model, system_size, solar_profile, demand_profile,
                project_life, capex_per_kw, opex_per_kw, import_tariff,
                export_tariff, ppa_rate, replace_years, inflation, export_allowed
            )

            # Summary
            st.subheader("Summary Inputs")
            st.write({
                "System Size (kW)": system_size,
                "Financial Model": model,
                "CAPEX (£/kW)": capex_per_kw,
                "OPEX (£/kW/year)": opex_per_kw,
                "PPA Rate (£/kWh)": ppa_rate,
                "Import Tariff (£/kWh)": import_tariff,
                "Export Tariff (£/kWh)": export_tariff,
                "Project Lifespan (years)": project_life,
                "Replacement Years": replace_years,
                "Inflation Rate": inflation,
                "Export Allowed": export_allowed
            })

            # Financials
            st.subheader("Financial Metrics (IRR only)")
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
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.pyplot(fig)

            # PDF export (chart bytes in buf)
            pdf_buf = export_pdf(
                {"System Size (kW)": system_size, "Financial Model": model,
                 "CAPEX (£/kW)": capex_per_kw, "OPEX (£/kW/year)": opex_per_kw},
                financials,
                df,
                buf
            )
            st.download_button("Download PDF Report", data=pdf_buf.getvalue(), file_name="report.pdf", mime="application/pdf")
        else:
            st.warning("Please provide both demand and solar profiles.")

if __name__ == "__main__":
    main()
