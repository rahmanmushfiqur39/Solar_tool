# Solar Modelling Tool - Streamlit Based UI
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --- Functions ---
def load_data():
    demand_upload = st.file_uploader("Upload half-hourly building demand profile (CSV)", type="csv", key="demand")
    solar_upload = st.file_uploader("Upload half-hourly solar generation profile (CSV)", type="csv", key="solar")

    demand_df = None
    solar_df = None

    if demand_upload is not None:
        demand_df = pd.read_csv(demand_upload, parse_dates=True, index_col=0)

    if solar_upload is not None:
        solar_df = pd.read_csv(solar_upload, parse_dates=True, index_col=0)
        # Show kWp input only after solar file is uploaded
        original_kwp = st.number_input("Original kWp of the uploaded solar profile",
                                       min_value=0.1, value=100.0, step=10.0)
        return solar_df, original_kwp, demand_df

    return solar_df, None, demand_df


def simulate(solar_gen, original_kwp, demand, solar_kWp, capex_per_kW, opex_per_kW, ppa_rate,
             import_tariff, export_tariff, project_life, replace_years, inflation, export_allowed):
    results = []
    annual_savings = []
    capex = solar_kWp * capex_per_kW
    opex = solar_kWp * opex_per_kW
    replacement_cost = 0.2 * capex  # Assumed inverter replacement at 20%

    # Scale solar generation based on original vs proposed kWp
    scale_factor = solar_kWp / original_kwp
    scaled_solar = solar_gen * scale_factor

    for year in range(1, project_life + 1):
        adj_factor = (1 + inflation) ** (year - 1)
        demand_total = demand.sum()

        # Use scaled solar generation
        overlap = np.minimum(scaled_solar, demand)
        export = scaled_solar - overlap if export_allowed else np.maximum(scaled_solar - demand, 0)
        grid = demand - overlap

        annual_cost_no_solar = demand_total * import_tariff * adj_factor
        annual_cost_with_solar = (
                                         grid.sum() * import_tariff - overlap.sum() * ppa_rate - export.sum() * export_tariff
                                 ) * adj_factor + opex * adj_factor
        savings = annual_cost_no_solar - annual_cost_with_solar

        if year in replace_years:
            annual_cost_with_solar += replacement_cost * adj_factor
        annual_savings.append(savings)

        results.append({
            'Year': year,
            'Demand (kWh)': demand_total,
            'Solar (kWh)': scaled_solar.sum(),
            'Self Consumption (kWh)': overlap.sum(),
            'Exported (kWh)': export.sum(),
            'Grid Import (kWh)': grid.sum(),
            'Annual Savings': savings
        })

    df = pd.DataFrame(results)
    total_savings = sum(annual_savings)
    return df, total_savings, capex, scaled_solar  # Return scaled solar for visualization


def optimise_self_consumption(target_ratio, solar_gen, original_kwp, demand):
    max_kwp = 1000
    step = 10
    for kwp in range(step, max_kwp + step, step):
        # Scale solar based on proposed vs original kWp
        scale_factor = kwp / original_kwp
        scaled_solar = solar_gen * scale_factor
        overlap = np.minimum(scaled_solar, demand)
        ratio = overlap.sum() / scaled_solar.sum()
        if ratio >= target_ratio:
            return kwp
    return max_kwp


def scenario_analysis(solar_gen, original_kwp, demand, capex_per_kW, opex_per_kW, ppa_rate,
                      import_tariff, export_tariff, project_life, replace_years, inflation, export_allowed):
    scenarios = []
    for kwp in range(0, 1001, 50):
        if kwp == 0:
            continue
        res, total_savings, capex, _ = simulate(
            solar_gen, original_kwp, demand, kwp, capex_per_kW, opex_per_kW,
            ppa_rate, import_tariff, export_tariff, project_life, replace_years,
            inflation, export_allowed
        )
        self_consumption = res['Self Consumption (kWh)'].sum() / res['Solar (kWh)'].sum()
        scenarios.append({
            'kWp': kwp,
            'Self Consumption': self_consumption,
            'Lifetime Savings': total_savings,
            'CAPEX': capex
        })
    return pd.DataFrame(scenarios)


# --- Streamlit UI ---
st.title("Solar Modelling Tool")
solar_df, kWp_design, demand_df = load_data()

# Check if all data is loaded
all_data_loaded = solar_df is not None and kWp_design is not None and demand_df is not None

if all_data_loaded:
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

    st.header("Simulation Output")
    sim_results, lifetime_savings, capex, scaled_solar = simulate(
        solar_df.iloc[:, 0],
        kWp_design,  # Use the kWp_design from load_data
        demand_df.iloc[:, 0],
        solar_kWp, capex_per_kW, opex_per_kW,
        ppa_rate, import_tariff, export_tariff,
        project_life, replace_years, inflation, export_allowed
    )
    st.dataframe(sim_results)

    st.write(f"**Lifetime Net Savings: £{lifetime_savings:,.2f}**")
    st.write(f"**Initial CAPEX: £{capex:,.2f}**")
    st.write(f"**Original System Size: {kWp_design} kWp**")
    st.write(f"**Proposed System Size: {solar_kWp} kWp**")
    st.write(f"**Scaling Factor: {solar_kWp / kWp_design:.2f}x**")

    st.header("Visualisation")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot daily profile for first week
    sample_days = 7 * 48  # 7 days * 48 half-hours
    if len(scaled_solar) > sample_days:
        ax.plot(scaled_solar[:sample_days].values, label=f"Solar ({solar_kWp} kWp)", alpha=0.7)
        ax.plot(demand_df.iloc[:sample_days, 0].values, label="Demand", alpha=0.7)
        ax.set_xlabel("Half-hour intervals")
        ax.set_ylabel("kW")
        ax.set_title("First Week: Solar vs Demand Profile")
    else:
        ax.plot(scaled_solar.values, label=f"Solar ({solar_kWp} kWp)", alpha=0.7)
        ax.plot(demand_df.iloc[:, 0].values, label="Demand", alpha=0.7)
        ax.set_xlabel("Half-hour intervals")
        ax.set_ylabel("kW")
        ax.set_title("Full Period: Solar vs Demand Profile")

    ax.legend()
    st.pyplot(fig)

    st.header("Optimise for Self Consumption")
    target = st.slider("Target Self Consumption (%)", 10, 100, 80)
    opt_kwp = optimise_self_consumption(
        target / 100.0,
        solar_df.iloc[:, 0],
        kWp_design,  # Use the kWp_design from load_data
        demand_df.iloc[:, 0]
    )
    st.write(f"Required kWp to achieve {target}% self consumption: **{opt_kwp} kWp**")
    st.write(f"Scaling from original: **{opt_kwp / kWp_design:.2f}x**")

    st.header("Scenario Analysis")
    scenario_df = scenario_analysis(
        solar_df.iloc[:, 0],
        kWp_design,  # Use the kWp_design from load_data
        demand_df.iloc[:, 0],
        capex_per_kW, opex_per_kW, ppa_rate,
        import_tariff, export_tariff, project_life,
        replace_years, inflation, export_allowed
    )
    st.dataframe(scenario_df)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(scenario_df['kWp'], scenario_df['Self Consumption'], 'o-', label='Self Consumption')
    ax2.set_xlabel("System Size (kWp)")
    ax2.set_ylabel("Self Consumption Ratio", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax3 = ax2.twinx()
    ax3.plot(scenario_df['kWp'], scenario_df['Lifetime Savings'], 's-', color='red', label='Savings (£)')
    ax3.plot(scenario_df['kWp'], scenario_df['CAPEX'], 'd-', color='green', label='CAPEX (£)')
    ax3.set_ylabel("Financial Metrics (£)", color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    fig2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    st.pyplot(fig2)

else:
    # Show appropriate messages based on what's missing
    if demand_df is not None and solar_df is None:
        st.warning("Please upload solar generation profile")
    elif solar_df is not None and kWp_design is None:
        st.warning("Please enter the original kWp for the solar profile")
    else:
        st.info("Please upload both solar generation and demand CSV files")