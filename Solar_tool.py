import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import numpy_financial as npf  # Only for IRR

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
def calculate_financials(model, system_size_kw, solar_profile, demand_profile,
                         project_life=25, capex_per_kw=800, opex_per_kw=15,
                         import_tariff=0.25, export_tariff=0.08, ppa_rate=0.18,
                         replace_years=[15], inflation=0.02, export_allowed=True):
    
    degradation = 0.005  # 0.5% per year
    capex = system_size_kw * capex_per_kw
    opex = system_size_kw * opex_per_kw

    years = list(range(1, project_life + 1))
    cashflows = []

    for y in years:
        solar_yield = solar_profile.iloc[:, 1] * (1 - degradation) ** (y - 1)
        demand = demand_profile.iloc[:, 1]

        self_consumption = np.minimum(solar_yield, demand).sum()
        export = np.maximum(solar_yield - demand, 0).sum() if export_allowed else 0

        if model == "Owner Occupier":
            savings = self_consumption * import_tariff
            export_rev = export * export_tariff
            net = savings + export_rev - opex
            if y == 1:
                net -= capex
            if y in replace_years:
                net -= 0.2 * capex  # replacement
            cashflows.append(net)

        elif model == "Landlord Funded (PPA to Tenant)":
            landlord_cf = self_consumption * ppa_rate + export * export_tariff - opex
            if y == 1:
                landlord_cf -= capex
            if y in replace_years:
                landlord_cf -= 0.2 * capex
            cashflows.append(landlord_cf)

    if model == "Owner Occupier":
        df = pd.DataFrame({"Year": years, "Owner Cashflow": cashflows})
        irr_owner = npf.irr(cashflows)
        return df, {"Owner Occupier": {"IRR": irr_owner}}
    else:
        df = pd.DataFrame({"Year": years, "Landlord Cashflow": cashflows})
        irr_landlord = npf.irr(cashflows)
        return df, {"Landlord": {"IRR": irr_landlord}}

# -------------------------
# PDF export
# -------------------------
def export_pdf(project_name, summary, financials, df, chart_buf):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    # Logo on top right
    logo_path = os.path.join(DATA_DIR, "savills_logo.png")
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=100, height=100, hAlign="RIGHT"))
    story.append(Spacer(1, 12))

    # Project name
    story.append(Paragraph(f"<b>{project_name}</b>", styles['Heading1']))
    story.append(Spacer(1, 6))

    # Summary Inputs table
    story.append(Paragraph("<b>Summary Inputs</b>", styles['Heading2']))
    summary_table_data = [["Parameter", "Value"]]
    for k, v in summary.items():
        summary_table_data.append([k, str(v)])  # convert values to string for PDF

    summary_table = Table(summary_table_data, hAlign="LEFT")
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Financials
    story.append(Paragraph("<b>Financial Metrics</b>", styles['Heading2']))
    table_data = [["Metric", *financials.keys()]]
    metrics = ["IRR (%)"]
    for m in metrics:
        row = [m]
        for actor in financials.keys():
            val = financials[actor][m.split()[0]]
            if val is None:
                row.append("-")
            else:
                row.append(f"{val*100:.1f}%")
        table_data.append(row)

    fin_table = Table(table_data)
    fin_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(fin_table)
    story.append(Spacer(1, 12))

    # Cashflow chart
    story.append(Paragraph("<b>Cashflow</b>", styles['Heading2']))
    story.append(Image(chart_buf, width=400, height=200))

    doc.build(story)
    buffer.seek(0)
    return buffer



# -------------------------
# Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="Solar Modelling Tool")  # default layout

    # create two columns for title and logo
    col1, col2 = st.columns([6, 1])

    with col1:
        st.title("☀️ Solar Modelling Tool")

    with col2:
        logo_path = os.path.join(DATA_DIR, "savills_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)  # keep aspect ratio

    profiles = load_profiles()

    # ---- Project name input ----
    project_name = st.text_input("Enter a name for the project", "Savills Solar Project")

    # ---- Inputs ----
    st.subheader("Demand Profile")
    demand_option = st.radio("Do you have half-hourly demand profile?", ["Yes - Upload CSV", "No - Use Benchmark Profile"])
    if demand_option == "Yes - Upload CSV":
        demand_file = st.file_uploader("Upload demand CSV", type="csv", key="demand_upload")
        demand_profile = pd.read_csv(demand_file) if demand_file else None
    else:
        site_type = st.selectbox("Select Site Type", ["Office", "Storage"])
        benchmark_key = "Benchmark_Office" if site_type == "Office" else "Benchmark_Storage"
        demand_profile = profiles[benchmark_key]

    st.subheader("Solar Profile")
    solar_option = st.radio("Do you have half-hourly solar profile?", ["Yes - Upload CSV", "No - Use Regional Profile"])
    if solar_option == "Yes - Upload CSV":
        solar_file = st.file_uploader("Upload solar CSV", type="csv", key="solar_upload")
        solar_profile = pd.read_csv(solar_file) if solar_file else None
    else:
        region = st.selectbox("Select Region", ["South", "Midlands", "Scotland"])
        solar_profile = profiles[f"Solar_{region}"]

    # Main inputs
    system_size = st.number_input("System Size (kWp)", min_value=10, max_value=5000, value=500, step=10)
    capex_per_kw = st.number_input("CAPEX (£/kWp)", 0.0, 5000.0, 800.0)
    opex_per_kw = st.number_input("O&M (£/kWp/year)", 0.0, 200.0, 15.0)
    ppa_rate = st.number_input("PPA Rate (£/kWh)", 0.0, 1.0, 0.18)
    import_tariff = st.number_input("Import Tariff (£/kWh)", 0.0, 1.0, 0.25)
    export_tariff = st.number_input("Export Tariff (£/kWh)", 0.0, 1.0, 0.08)
    project_life = st.number_input("Project Lifespan (years)", 1, 50, 25)
    inflation = st.number_input("Annual Inflation Rate (%)", 0.0, 10.0, 0.0) / 100
    replace_years = st.multiselect("Replacement Years", list(range(1, 51)), [15])
    export_allowed = st.checkbox("Export Allowed?", True)
    model = st.radio("Financial Model", ["Owner Occupier", "Landlord Funded (PPA to Tenant)"])

    # Run simulation
    if st.button("Run / Update Simulation"):
        if demand_profile is not None and solar_profile is not None:
            df, financials = calculate_financials(
                model, system_size, solar_profile, demand_profile,
                project_life, capex_per_kw, opex_per_kw, import_tariff,
                export_tariff, ppa_rate, replace_years, inflation, export_allowed
            )

            # Summary
            st.subheader("Summary Inputs")
            summary_dict = {
                "Project Name": project_name,
                "System Size (kWp)": system_size,
                "Financial Model": model,
                "CAPEX (£/kWp)": capex_per_kw,
                "OPEX (£/kWp/year)": opex_per_kw,
                "PPA Rate (£/kWh)": ppa_rate,
                "Import Tariff (£/kWh)": import_tariff,
                "Export Tariff (£/kWh)": export_tariff,
                "Project Lifespan (years)": project_life,
                "Replacement Years": ", ".join(map(str, replace_years)),
                "Inflation Rate": f"{inflation*100:.2f}%",
                "Export Allowed": export_allowed
            }

            summary_df = pd.DataFrame(list(summary_dict.items()), columns=["Parameter", "Value"])
            
            st.dataframe(summary_df, hide_index=True, use_container_width=True)



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
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.pyplot(fig)

            # PDF export
            pdf_buf = export_pdf(
                project_name,
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








