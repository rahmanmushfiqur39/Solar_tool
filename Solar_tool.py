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
# Helpers
# -------------------------
def _get_series(df):
    """
    Return (time_index, values_series) given a 2-column-like csv.
    First column is assumed to be datetime-like; second is numeric.
    Falls back gracefully if column names differ.
    """
    if df is None or df.empty:
        return None, None
    # Use first two columns by position
    col_time = df.columns[0]
    col_val = df.columns[1]
    # Parse datetime safely; if it fails, keep as is
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
    """
    Build year-1 half-hourly kWh series for solar and demand.
    - Solar regional profiles are normalised in kWh/kWp/interval, so multiply by system size (kWp).
    - Uploaded solar assumed already in kWh/interval.
    - Benchmark demand profiles are shape-only (sum to 1 or arbitrary scale). Scale to annual_demand = intensity * floorspace.
    - Uploaded demand assumed already in kWh/interval.
    Returns: index (datetime if available), solar_hh_kwh, demand_hh_kwh
    """
    # Solar
    t_s, v_s = _get_series(solar_df)
    if used_regional_solar:
        solar_hh = (v_s.astype(float) * float(system_size_kwp)).values
    else:
        solar_hh = v_s.astype(float).values  # already absolute kWh/interval

    # Demand
    t_d, v_d = _get_series(demand_df)
    if used_benchmark_demand:
        intensity = intensity_office if site_type == "Office" else intensity_storage
        annual_demand = float(intensity) * float(floor_space_m2)  # kWh/year
        shape = v_d.astype(float).values
        shape_sum = shape.sum()
        if shape_sum <= 0:
            demand_hh = np.zeros_like(shape)
        else:
            demand_hh = shape / shape_sum * annual_demand
    else:
        demand_hh = v_d.astype(float).values  # already absolute kWh/interval

    # Align lengths if needed
    n = min(len(solar_hh), len(demand_hh))
    solar_hh = solar_hh[:n]
    demand_hh = demand_hh[:n]
    if t_s is not None and len(t_s) >= n:
        index = t_s[:n]
    else:
        index = pd.RangeIndex(0, n, 1)

    # Apply export restriction
    if not export_allowed:
        # export will be zeroed later, but self-consumption uses min(solar,demand) anyway
        pass

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
    """Compute month-wise £ savings for Year 1 (owner: avoided import + export revenue; tenant: (import-ppa)*self-consumption)."""
    # Prepare frame
    df = pd.DataFrame({
        "ts": pd.to_datetime(index, errors="coerce"),
        "solar": solar_hh,
        "demand": demand_hh
    })
    # If timestamps aren't real datetimes, create month bins 1..12 as categorical-ish
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
        # Tenant-centric savings chart for PPA model
        savings_gbp = self_cons * max(import_tariff - ppa_rate, 0.0)

    monthly = savings_gbp.groupby(df["month"]).sum()
    # Ensure months 1..12 all present
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
    # Constants
    degradation = 0.005  # 0.5% per year

    # Build Year-1 half-hourly series
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

    # Year-1 technicals
    annual_yield_y1 = float(np.sum(solar_hh))
    self_consumption_y1 = float(np.sum(np.minimum(solar_hh, demand_hh)))
    export_y1 = float(np.sum(np.maximum(solar_hh - demand_hh, 0.0))) if export_allowed else 0.0
    cons_ratio = (self_consumption_y1 / annual_yield_y1) if annual_yield_y1 > 0 else 0.0

    # Yearly projections (from Y2 just degrade annual generation; keep Y1 split ratio)
    years = list(range(1, project_life + 1))
    gen_y = [annual_yield_y1 * ((1 - degradation) ** (y - 1)) for y in years]
    selfc_y = [g * cons_ratio for g in gen_y]
    export_y = [(g - s) if export_allowed else 0.0 for g, s in zip(gen_y, selfc_y)]

    capex = float(system_size_kwp) * float(capex_per_kwp)
    annual_opex_y = [float(system_size_kwp) * float(opex_per_kwp) * ((1 + float(inflation)) ** (y - 1)) for y in years]

    # Cashflows
    owner_cf, landlord_cf, tenant_cf = [], [], []

    if model == "Owner Occupier":
        for i, y in enumerate(years):
            savings = selfc_y[i] * import_tariff
            export_rev = export_y[i] * export_tariff
            net = savings + export_rev - annual_opex_y[i]
            if y == 1:
                net -= capex
            if y in replace_years:
                net -= float(inverter_replacement_cost_per_kwp) * float(system_size_kwp)
            owner_cf.append(net)

    else:  # PPA
        for i, y in enumerate(years):
            landlord_revenue = selfc_y[i] * ppa_rate + export_y[i] * export_tariff
            landlord_costs = annual_opex_y[i]
            net_landlord = landlord_revenue - landlord_costs
            if y == 1:
                net_landlord -= capex
            if y in replace_years:
                net_landlord -= float(inverter_replacement_cost_per_kwp) * float(system_size_kwp)
            landlord_cf.append(net_landlord)

            # Tenant savings = avoided import - PPA paid
            tenant_sav = selfc_y[i] * max(import_tariff - ppa_rate, 0.0)
            tenant_cf.append(tenant_sav)

    # IRR and payback
    def _safe_irr(cf):
        try:
            val = float(npf.irr(cf))
            if np.isfinite(val):
                return val
            return None
        except Exception:
            return None

    if model == "Owner Occupier":
        irr_owner = _safe_irr(owner_cf)
        pb_owner = _payback_year(owner_cf)
        # Aggregates
        owner_net_lifetime_savings = sum([selfc_y[i] * import_tariff + export_y[i] * export_tariff - annual_opex_y[i] for i in range(len(years))]) - capex - sum(
            float(inverter_replacement_cost_per_kwp) * float(system_size_kwp) for y in years if y in replace_years
        )
        owner_net_lifetime_income = sum([export_y[i] * export_tariff for i in range(len(years))])
        results = {
            "cashflow_yearly": pd.DataFrame({"Year": years, "Owner": owner_cf}),
            "irr": {"Owner Occupier": irr_owner},
            "payback": {"Owner Occupier": pb_owner},
            "lifetime": {
                "Owner Occupier": {
                    "Capex": capex,
                    "Annual Opex (Y1)": annual_opex_y[0],
                    "Net lifetime savings": owner_net_lifetime_savings,
                    "Net lifetime income": owner_net_lifetime_income,
                }
            },
        }
    else:
        irr_landlord = _safe_irr(landlord_cf)
        pb_landlord = _payback_year(landlord_cf)
        landlord_net_lifetime_income = sum(landlord_cf)
        tenant_net_lifetime_savings = sum(tenant_cf)
        results = {
            "cashflow_yearly": pd.DataFrame({"Year": years, "Landlord": landlord_cf, "Tenant Savings": tenant_cf}),
            "irr": {"Landlord": irr_landlord},
            "payback": {"Landlord": pb_landlord},
            "lifetime": {
                "Landlord": {
                    "Capex": capex,
                    "Annual Opex (Y1)": annual_opex_y[0],
                    "Net lifetime income": landlord_net_lifetime_income,
                },
                "Tenant": {
                    "Net lifetime savings": tenant_net_lifetime_savings
                }
            },
        }

    technical_y1 = {
        "Annual Yield (kWh)": annual_yield_y1,
        "Consumed on site (kWh)": self_consumption_y1,
        "Exported (kWh)": export_y1,
    }

    # Monthly savings (Year 1)
    monthly_savings_y1 = aggregate_monthly_savings(
        index, solar_hh, demand_hh, model, import_tariff, export_tariff, ppa_rate, export_allowed=export_allowed
    )

    return results, technical_y1, monthly_savings_y1

# -------------------------
# PDF export
# -------------------------
def export_pdf(project_name, summary, financials_view, technical_y1, monthly_chart_buf, cashflow_chart_buf):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    # Logo
    logo_path = os.path.join(DATA_DIR, "savills_logo.png")
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=100, height=100, hAlign="RIGHT"))
    story.append(Spacer(1, 12))

    # Title
    story.append(Paragraph(f"<b>{project_name}</b>", styles['Heading1']))
    story.append(Spacer(1, 6))

    # Summary Inputs
    story.append(Paragraph("<b>Summary Inputs</b>", styles['Heading2']))
    summary_table_data = [["Parameter", "Value"]]
    for k, v in summary.items():
        summary_table_data.append([k, str(v)])

    summary_table = Table(summary_table_data, hAlign="LEFT", colWidths=[220, 260])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Technical Output (Year 1)
    story.append(Paragraph("<b>Technical Output (Year 1)</b>", styles['Heading2']))
    tech_table_data = [["Metric", "Value (kWh)"]]
    tech_table_data.append(["Annual Yield", f"{technical_y1['Annual Yield (kWh)']:.0f}"])
    tech_table_data.append(["Consumed on site", f"{technical_y1['Consumed on site (kWh)']:.0f}"])
    tech_table_data.append(["Exported", f"{technical_y1['Exported (kWh)']:.0f}"])

    tech_table = Table(tech_table_data, hAlign="LEFT", colWidths=[220, 260])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 12))

    # Financial Metrics
    story.append(Paragraph("<b>Financial Metrics</b>", styles['Heading2']))
    fin_table = Table(financials_view, hAlign="LEFT")
    fin_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    story.append(fin_table)
    story.append(Spacer(1, 12))

    # Monthly Savings Bar
    story.append(Paragraph("<b>Monthly Savings (Year 1)</b>", styles['Heading2']))
    story.append(Image(monthly_chart_buf, width=450, height=250))
    story.append(Spacer(1, 12))

    # Cashflow chart
    story.append(Paragraph("<b>Cashflow</b>", styles['Heading2']))
    story.append(Image(cashflow_chart_buf, width=450, height=250))
    story.append(Spacer(1, 18))

    # Footer
    story.append(Paragraph("Mushfiqur Rahman, Energy Consultant, mushfiqur.rahman@savills.com", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------
# Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="Solar Modelling Tool", layout="wide")

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("☀️ Solar Modelling Tool")
    with col2:
        logo_path = os.path.join(DATA_DIR, "savills_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)

    profiles = load_profiles()

    # Project name input
    project_name = st.text_input("Enter a name for the project", "Savills Solar Project")

    # ---- Inputs ----
    st.subheader("Demand Profile")
    demand_option = st.radio("Do you have half-hourly demand profile?", ["Yes - Upload CSV", "No - Use Benchmark Profile"])
    floor_space_m2 = 0.0
    site_type = "Office"
    if demand_option == "Yes - Upload CSV":
        demand_file = st.file_uploader("Upload demand CSV", type="csv", key="demand_upload")
        demand_profile = pd.read_csv(demand_file) if demand_file else None
        used_benchmark_demand = False
    else:
        site_type = st.selectbox("Select Site Type", ["Office", "Storage"])
        floor_space_m2 = st.number_input("Floor space (m²)", min_value=0.0, value=1000.0, step=10.0)
        benchmark_key = "Benchmark_Office" if site_type == "Office" else "Benchmark_Storage"
        demand_profile = profiles[benchmark_key]
        used_benchmark_demand = True

    st.subheader("Solar Profile")
    solar_option = st.radio("Do you have half-hourly solar profile?", ["Yes - Upload CSV", "No - Use Regional Profile"])
    if solar_option == "Yes - Upload CSV":
        solar_file = st.file_uploader("Upload solar CSV", type="csv", key="solar_upload")
        solar_profile = pd.read_csv(solar_file) if solar_file else None
        used_regional_solar = False
    else:
        region = st.selectbox("Select Region", ["South", "Midlands", "Scotland"])
        solar_profile = profiles[f"Solar_{region}"]
        used_regional_solar = True

    # Main inputs
    st.subheader("System & Financial Inputs")
    system_size = st.number_input("System Size (kWp)", min_value=10.0, max_value=500000.0, value=500.0, step=10.0)
    capex_per_kw = st.number_input("CAPEX (£/kWp)", 0.0, 5000.0, 800.0)
    opex_per_kw = st.number_input("O&M (£/kWp/year)", 0.0, 200.0, 15.0)
    inverter_replacement_cost_per_kwp = st.number_input("Inverter Replacement Cost (£/kWp)", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
    ppa_rate = st.number_input("PPA Rate (£/kWh)", 0.0, 1.0, 0.18)
    import_tariff = st.number_input("Import Tariff (£/kWh)", 0.0, 1.0, 0.25)
    export_tariff = st.number_input("Export Tariff (£/kWh)", 0.0, 1.0, 0.08)
    project_life = st.number_input("Project Lifespan (years)", 1, 50, 25)
    inflation = st.number_input("Annual Inflation Rate (%) (applies to O&M only)", 0.0, 10.0, 0.0) / 100
    replace_years = st.multiselect("Inverter Replacement Years", list(range(1, 51)), [15])
    export_allowed = st.checkbox("Export Allowed?", True)
    model = st.radio("Financial Model", ["Owner Occupier", "Landlord Funded (PPA to Tenant)"])

    # Run simulation
    if st.button("Run / Update Simulation"):
        if demand_profile is not None and solar_profile is not None:
            results, technical_y1, monthly_savings_y1 = calculate_project_financials(
                model=model,
                system_size_kwp=system_size,
                solar_df=solar_profile,
                demand_df=demand_profile,
                project_life=project_life,
                capex_per_kwp=capex_per_kw,
                opex_per_kwp=opex_per_kw,
                import_tariff=import_tariff,
                export_tariff=export_tariff,
                ppa_rate=ppa_rate,
                replace_years=replace_years,
                inflation=inflation,
                export_allowed=export_allowed,
                used_regional_solar=used_regional_solar,
                used_benchmark_demand=used_benchmark_demand,
                site_type=site_type,
                floor_space_m2=floor_space_m2,
                inverter_replacement_cost_per_kwp=inverter_replacement_cost_per_kwp
            )

            # Summary
            st.subheader("Summary Inputs")
            summary_dict = {
                "Project Name": project_name,
                "System Size (kWp)": system_size,
                "Financial Model": model,
                "CAPEX (£/kWp)": capex_per_kw,
                "OPEX (£/kWp/year)": opex_per_kw,
                "Inverter Replacement (£/kWp)": inverter_replacement_cost_per_kwp,
                "PPA Rate (£/kWh)": ppa_rate,
                "Import Tariff (£/kWh)": import_tariff,
                "Export Tariff (£/kWh)": export_tariff,
                "Project Lifespan (years)": project_life,
                "Inverter Replacement Years": ", ".join(map(str, replace_years)),
                "Inflation Rate": f"{inflation*100:.2f}%",
                "Export Allowed": export_allowed,
                "Demand Source": "Benchmark" if used_benchmark_demand else "Uploaded",
                "Site Type": site_type if used_benchmark_demand else "-",
                "Floor space (m²)": floor_space_m2 if used_benchmark_demand else "-",
                "Solar Source": "Regional" if used_regional_solar else "Uploaded",
                "Degradation": "0.5%/yr",
            }
            summary_df = pd.DataFrame(list(summary_dict.items()), columns=["Parameter", "Value"])
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            # Technical Output (Year 1)
            st.subheader("Technical Output (Year 1)")
            tech_df = pd.DataFrame({
                "Metric": ["Annual Yield (kWh)", "Consumed on site (kWh)", "Exported (kWh)"],
                "Value": [f"{technical_y1['Annual Yield (kWh)']:.0f}", f"{technical_y1['Consumed on site (kWh)']:.0f}", f"{technical_y1['Exported (kWh)']:.0f}"]
            })
            st.dataframe(tech_df, hide_index=True, use_container_width=True)

            # Financial Metrics table build for UI + PDF view matrix
            st.subheader("Financial Metrics")
            fin_view = []
            if model == "Owner Occupier":
                headers = ["Metric", "Owner Occupier"]
                fin_view.append(headers)

                capex = results["lifetime"]["Owner Occupier"]["Capex"]
                opex_y1 = results["lifetime"]["Owner Occupier"]["Annual Opex (Y1)"]
                net_lifetime_savings = results["lifetime"]["Owner Occupier"]["Net lifetime savings"]
                net_lifetime_income = results["lifetime"]["Owner Occupier"]["Net lifetime income"]
                irr = results["irr"]["Owner Occupier"]
                pb = results["payback"]["Owner Occupier"]

                rows = [
                    ["Capex", f"£{capex:,.0f}"],
                    ["Annual Opex", f"£{opex_y1:,.0f} (Y1)"],
                    ["Net lifetime savings", f"£{net_lifetime_savings:,.0f}"],
                    ["Net lifetime income", f"£{net_lifetime_income:,.0f}"],
                    ["IRR", f"{irr*100:.1f}%" if irr is not None else "N/A"],
                    ["Payback years", f"{pb}" if pb is not None else "N/A"],
                ]
                fin_view.extend(rows)
                st.table(pd.DataFrame(rows, columns=headers).set_index("Metric"))
            else:
                headers = ["Metric", "Landlord", "Tenant"]
                fin_view.append(headers)

                capex = results["lifetime"]["Landlord"]["Capex"]
                opex_y1 = results["lifetime"]["Landlord"]["Annual Opex (Y1)"]
                landlord_net_income = results["lifetime"]["Landlord"]["Net lifetime income"]
                tenant_net_savings = results["lifetime"]["Tenant"]["Net lifetime savings"]
                irr_l = results["irr"]["Landlord"]
                pb_l = results["payback"]["Landlord"]

                rows = [
                    ["Capex", f"£{capex:,.0f}", "N/A"],
                    ["Annual Opex", f"£{opex_y1:,.0f} (Y1)", "N/A"],
                    ["Net lifetime savings", "N/A", f"£{tenant_net_savings:,.0f}"],
                    ["Net lifetime income", f"£{landlord_net_income:,.0f}", "N/A"],
                    ["IRR", f"{irr_l*100:.1f}%" if irr_l is not None else "N/A", "N/A"],
                    ["Payback years", f"{pb_l}" if pb_l is not None else "N/A", "N/A"],
                ]
                fin_view.extend(rows)
                st.table(pd.DataFrame(rows, columns=headers).set_index("Metric"))

            # Charts
            cash_df = results["cashflow_yearly"]
            st.subheader("Monthly Savings (Year 1)")
            fig_m, ax_m = plt.subplots()
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            ax_m.bar(months, monthly_savings_y1.values)
            ax_m.set_ylabel("£")
            ax_m.set_xlabel("Month")
            ax_m.set_title("Monthly Savings (Year 1)")
            m_buf = BytesIO()
            fig_m.tight_layout()
            fig_m.savefig(m_buf, format="png")
            m_buf.seek(0)
            st.pyplot(fig_m)

            st.subheader("Yearly Cashflow")
            fig_c, ax_c = plt.subplots()
            if model == "Owner Occupier":
                ax_c.plot(cash_df["Year"], cash_df["Owner"], marker="o")
                ax_c.set_ylabel("£ per year")
            else:
                ax_c.plot(cash_df["Year"], cash_df["Landlord"], marker="o", label="Landlord")
                ax_c.bar(cash_df["Year"], cash_df["Tenant Savings"], alpha=0.3, label="Tenant (savings)")
                ax_c.set_ylabel("£ per year")
                ax_c.legend()
            ax_c.set_xlabel("Year")
            ax_c.set_title("Cashflow")
            c_buf = BytesIO()
            fig_c.tight_layout()
            fig_c.savefig(c_buf, format="png")
            c_buf.seek(0)
            st.pyplot(fig_c)

            # PDF export
            pdf_buf = export_pdf(
                project_name=project_name,
                summary=summary_dict,
                financials_view=fin_view,
                technical_y1=technical_y1,
                monthly_chart_buf=m_buf,
                cashflow_chart_buf=c_buf
            )
            st.download_button("Download PDF Report", data=pdf_buf, file_name="report.pdf", mime="application/pdf")

            st.markdown("---")
            st.markdown("**Mushfiqur Rahman, Energy Consultant, mushfiqur.rahman@savills.com**")
        else:
            st.warning("Please provide both demand and solar profiles.")

if __name__ == "__main__":
    main()
