import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "manufacturing_planner.db"

st.set_page_config(page_title="Production & Capacity Planner", layout="wide")

# -----------------------------
# Database / persistence
# -----------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def save_scenario(name, state):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO scenarios(name, created_at, payload) VALUES (?, ?, ?)",
                (name, datetime.now().isoformat(timespec="seconds"), json.dumps(state, default=str)))
    con.commit()
    con.close()


def load_scenarios():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, name, created_at FROM scenarios ORDER BY id DESC").fetchall()
    con.close()
    return rows


def load_scenario(sid):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT payload FROM scenarios WHERE id=?", (sid,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else None


def df_to_records(df):
    return df.where(pd.notna(df), None).to_dict(orient="records")


def records_to_df(records, columns):
    df = pd.DataFrame(records)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns]


# -----------------------------
# Defaults derived
# -----------------------------
def default_state():
    products = pd.DataFrame([
        ["P01", "Alpha Pump", 5, 2.0, 1.00, 500, 100.0, 2.0, 80.0, 100],
        ["P02", "Beta Valve", 4, 2.0, 1.00, 400, 90.0, 2.0, 75.0, 80],
        ["P03", "Gamma Motor", 5, 3.0, 0.82, 300, 130.0, 3.0, 120.0, 100],
        ["P04", "Delta Gearbox", 3, 2.5, 1.00, 350, 110.0, 2.5, 70.0, 60],
        ["P05", "Epsilon Coupling", 2, 2.0, 1.00, 300, 95.0, 2.0, 60.0, 50],
    ], columns=["Product", "Description", "Product_Priority", "Defect_Rate_pct", "Raw_Material_Factor",
                "Max_Weekly_Production", "Production_Cost", "Holding_Cost", "Shortage_Penalty", "Initial_Inventory"])

    machines = pd.DataFrame([
        ["M1", 400, 1, 85, 20, 100, 75],
        ["M2", 500, 1, 90, 25, 100, 70],
        ["M3", 450, 1, 80, 30, 100, 90],
        ["M4", 300, 1, 85, 15, 80, 80],
    ], columns=["Machine", "Base_Hours_Per_Shift", "Shifts", "Efficiency_pct", "Downtime_Hours",
                "Max_Overtime_Hours", "Overtime_Cost_per_Hour"])

    routing = pd.DataFrame([
        ["P01", "M1", 2.0], ["P01", "M2", 1.0],
        ["P02", "M2", 1.5], ["P02", "M3", 1.0],
        ["P03", "M1", 1.0], ["P03", "M3", 2.5], ["P03", "M4", 1.0],
        ["P04", "M2", 2.0], ["P04", "M3", 1.5],
        ["P05", "M1", 1.5], ["P05", "M3", 1.0], ["P05", "M4", 2.0],
    ], columns=["Product", "Machine", "Hours_Per_Unit"])

    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    demand_rows = []
    base = {"P01": 500, "P02": 300, "P03": 200, "P04": 250, "P05": 180}
    for wi, w in enumerate(weeks):
        for p in products.Product:
            f = base[p] * (1 + 0.05 * wi)
            actual = f * (1 + (0.06 if (wi + len(p)) % 3 == 0 else -0.02))
            demand_rows.append([w, p, round(f), round(actual), 20.0])
    demand = pd.DataFrame(demand_rows, columns=["Week", "Product", "Forecast_Demand", "Actual_Demand", "Safety_Stock_pct"])
    return {"products": products, "machines": machines, "routing": routing, "demand": demand, "weeks": weeks}


# -----------------------------
# Calculation engine
# -----------------------------
def clean_numeric(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def prepare_data(state):
    products = clean_numeric(state["products"], ["Product_Priority", "Defect_Rate_pct", "Raw_Material_Factor",
                                                "Max_Weekly_Production", "Production_Cost", "Holding_Cost",
                                                "Shortage_Penalty", "Initial_Inventory"])
    machines = clean_numeric(state["machines"], ["Base_Hours_Per_Shift", "Shifts", "Efficiency_pct",
                                                 "Downtime_Hours", "Max_Overtime_Hours", "Overtime_Cost_per_Hour"])
    routing = clean_numeric(state["routing"], ["Hours_Per_Unit"])
    demand = clean_numeric(state["demand"], ["Forecast_Demand", "Actual_Demand", "Safety_Stock_pct"])
    return products, machines, routing, demand


def required_production_table(products, demand, weeks):
    rows = []
    init_inv = dict(zip(products.Product, products.Initial_Inventory))
    for p in products.Product:
        opening = init_inv[p]
        for w in weeks:
            d = demand[(demand.Product == p) & (demand.Week == w)]
            if d.empty:
                forecast = actual = ss_pct = 0.0
            else:
                forecast = float(d.iloc[0].Forecast_Demand)
                actual = float(d.iloc[0].Actual_Demand)
                ss_pct = float(d.iloc[0].Safety_Stock_pct)
            safety = forecast * ss_pct / 100.0
            req = max(0.0, forecast + safety - opening)
            rows.append([w, p, forecast, actual, opening, ss_pct, safety, req])
            opening = max(0.0, opening + req - actual)
    return pd.DataFrame(rows, columns=["Week", "Product", "Forecast_Demand", "Actual_Demand",
                                       "Opening_Inventory", "Safety_Stock_pct", "Safety_Stock", "Production_Required"])


def capacity_table(products, machines, routing, weeks, production=None, demand_scale=1.0, capacity_scale=1.0,
                   extra_overtime=0.0):
    route = routing.pivot_table(index="Product", columns="Machine", values="Hours_Per_Unit", aggfunc="sum", fill_value=0)
    route = route.reindex(index=products.Product, columns=machines.Machine, fill_value=0)
    rows = []
    for w in weeks:
        for _, m in machines.iterrows():
            base = float(m.Base_Hours_Per_Shift) * float(m.Shifts)
            avail = max(0.0, (base - float(m.Downtime_Hours)) * float(m.Efficiency_pct) / 100.0) * capacity_scale
            max_ot = float(m.Max_Overtime_Hours) + extra_overtime
            req = 0.0
            if production is not None:
                for p in products.Product:
                    q = float(production.get((w, p), 0.0))
                    req += q * float(route.loc[p, m.Machine])
            rows.append([w, m.Machine, base, float(m.Downtime_Hours), float(m.Efficiency_pct), avail, max_ot, req])
    cap = pd.DataFrame(rows, columns=["Week", "Machine", "Base_Hours", "Downtime_Hours", "Efficiency_pct",
                                      "Available_Hours", "Max_Overtime_Hours", "Required_Hours"])
    cap["Capacity_Utilisation_Pct"] = np.where(cap.Available_Hours > 0,
                                                 cap.Required_Hours / cap.Available_Hours * 100, np.inf)
    cap["Capacity_Gap_Hours"] = cap.Required_Hours - cap.Available_Hours
    cap["Status"] = np.where(cap.Capacity_Utilisation_Pct > 100, "OVER CAPACITY",
                              np.where(cap.Capacity_Utilisation_Pct >= 90, "HIGH UTILISATION", "CAPACITY AVAILABLE"))
    return cap


def optimize_plan(products, machines, routing, demand, weeks, allow_shortage=True,
                  demand_scale=1.0, capacity_scale=1.0, extra_overtime=0.0):
    P = list(products.Product)
    M = list(machines.Machine)
    W = list(weeks)
    nP, nM, nW = len(P), len(M), len(W)

    # variable blocks: production x[P,W], closing inventory inv[P,W], shortage short[P,W], overtime ot[M,W]
    def ix_x(p,w): return P.index(p)*nW + W.index(w)
    x_count = nP*nW
    def ix_inv(p,w): return x_count + P.index(p)*nW + W.index(w)
    inv_count = nP*nW
    def ix_short(p,w): return x_count + inv_count + P.index(p)*nW + W.index(w)
    short_count = nP*nW
    def ix_ot(m,w): return x_count + inv_count + short_count + M.index(m)*nW + W.index(w)
    total = x_count + inv_count + short_count + nM*nW

    c = np.zeros(total)
    bounds = [(0, None)] * total
    pmap = products.set_index("Product")
    mmap = machines.set_index("Machine")
    rmap = routing.pivot_table(index="Product", columns="Machine", values="Hours_Per_Unit", aggfunc="sum", fill_value=0)
    rmap = rmap.reindex(index=P, columns=M, fill_value=0)

    for p in P:
        pr = pmap.loc[p]
        for w in W:
            i = ix_x(p,w)
            c[i] = float(pr.Production_Cost)
            bounds[i] = (0, max(0.0, float(pr.Max_Weekly_Production) * float(pr.Raw_Material_Factor)))
            c[ix_inv(p,w)] = float(pr.Holding_Cost)
            # Priority increases shortage penalty for higher-priority products.
            c[ix_short(p,w)] = float(pr.Shortage_Penalty) * max(1.0, float(pr.Product_Priority))
    for m in M:
        mr = mmap.loc[m]
        for w in W:
            i = ix_ot(m,w)
            c[i] = float(mr.Overtime_Cost_per_Hour)
            bounds[i] = (0, max(0.0, float(mr.Max_Overtime_Hours) + extra_overtime))

    A_eq, b_eq = [], []
    # Inventory balance: opening + good production + shortage = actual demand + closing inventory.
    # For week 1, opening is the product's initial inventory. For later weeks, opening is the
    # previous week's closing inventory decision variable.
    for p in P:
        defect = float(pmap.loc[p, "Defect_Rate_pct"]) / 100.0
        for wi, w in enumerate(W):
            row = np.zeros(total)
            row[ix_x(p,w)] = 1.0 - defect
            row[ix_short(p,w)] = 1.0
            row[ix_inv(p,w)] = -1.0
            drow = demand[(demand.Product == p) & (demand.Week == w)]
            actual = float(drow.iloc[0].Actual_Demand) if not drow.empty else 0.0
            if wi == 0:
                b_eq.append(actual * demand_scale - float(pmap.loc[p, "Initial_Inventory"]))
            else:
                # Move previous closing inventory to the left-hand side.
                row[ix_inv(p, W[wi-1])] = -1.0
                b_eq.append(actual * demand_scale)
            A_eq.append(row)

    A_ub, b_ub = [], []
    # Machine capacity with overtime.
    for m in M:
        mr = mmap.loc[m]
        base_avail = max(0.0, (float(mr.Base_Hours_Per_Shift)*float(mr.Shifts) - float(mr.Downtime_Hours))
                         * float(mr.Efficiency_pct)/100.0) * capacity_scale
        for w in W:
            row = np.zeros(total)
            for p in P:
                row[ix_x(p,w)] = float(rmap.loc[p,m])
            row[ix_ot(m,w)] = -1.0
            A_ub.append(row)
            b_ub.append(base_avail)

    # Safety stock is treated as a target rather than a hard constraint when shortage is allowed.
    # When shortage is disabled, closing inventory must meet safety stock.
    if not allow_shortage:
        for p in P:
            for w in W:
                row = np.zeros(total)
                row[ix_inv(p,w)] = -1.0
                drow = demand[(demand.Product == p) & (demand.Week == w)]
                forecast = float(drow.iloc[0].Forecast_Demand) if not drow.empty else 0.0
                ss_pct = float(drow.iloc[0].Safety_Stock_pct) if not drow.empty else 0.0
                A_ub.append(row)
                b_ub.append(-forecast * ss_pct/100.0)
        # no shortage
        for p in P:
            for w in W:
                b = list(bounds[ix_short(p,w)])
                bounds[ix_short(p,w)] = (0, 0)

    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq) if A_eq else None, b_eq=np.array(b_eq) if b_eq else None,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"success": False, "message": res.message}

    prod = {(w,p): max(0.0, res.x[ix_x(p,w)]) for w in W for p in P}
    inv = {(w,p): max(0.0, res.x[ix_inv(p,w)]) for w in W for p in P}
    short = {(w,p): max(0.0, res.x[ix_short(p,w)]) for w in W for p in P}
    ot = {(w,m): max(0.0, res.x[ix_ot(m,w)]) for w in W for m in M}

    rows = []
    for w in W:
        for p in P:
            drow = demand[(demand.Product == p) & (demand.Week == w)]
            forecast = float(drow.iloc[0].Forecast_Demand) if not drow.empty else 0
            actual = float(drow.iloc[0].Actual_Demand) if not drow.empty else 0
            ss = forecast * (float(drow.iloc[0].Safety_Stock_pct) if not drow.empty else 0)/100
            defect = prod[(w,p)] * float(pmap.loc[p,"Defect_Rate_pct"])/100
            good = prod[(w,p)] - defect
            rows.append([w,p,forecast*demand_scale,actual*demand_scale,prod[(w,p)],defect,good,inv[(w,p)],ss,short[(w,p)]])
    plan = pd.DataFrame(rows, columns=["Week","Product","Forecast_Demand","Actual_Demand","Planned_Production",
                                       "Defective_Units","Good_Units","Closing_Inventory","Safety_Stock","Shortage"])
    return {"success": True, "message": res.message, "objective": float(res.fun), "plan": plan, "overtime": ot}


def teaching_calc(products, demand, weeks):
    req = required_production_table(products, demand, weeks)
    # Use basic requirement formula, independent of capacity.
    return req


def make_state(products, machines, routing, demand, weeks):
    return {"products": df_to_records(products), "machines": df_to_records(machines),
            "routing": df_to_records(routing), "demand": df_to_records(demand), "weeks": list(weeks)}


init_db()
if "state" not in st.session_state:
    st.session_state.state = default_state()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Production Planner")
st.sidebar.caption("Dynamic manufacturing planning model based on Production & Capacity Planning.")

with st.sidebar.expander("Load a saved scenario", expanded=False):
    saved = load_scenarios()
    if saved:
        labels = {f"{sid} — {name} ({created})": sid for sid, name, created in saved}
        chosen = st.selectbox("Saved scenarios", list(labels.keys()))
        if st.button("Load selected scenario"):
            payload = load_scenario(labels[chosen])
            if payload:
                st.session_state.state = payload
                st.success("Scenario loaded.")
                st.rerun()
    else:
        st.caption("No saved scenarios yet.")

state = st.session_state.state
products, machines, routing, demand = prepare_data(state)
weeks = state.get("weeks", sorted(demand.Week.astype(str).unique().tolist()))

st.title("Production & Capacity Planning Simulator")
st.markdown("**Demand → Inventory → Production Requirement → Machine Capacity → Feasible Plan → Management Decision**")
st.info("This is a manufacturing decision tool. Enter your own values, calculate the plan, optimise it, test scenarios, and explain why the recommendation changes.")

# -----------------------------
# Tabs
# -----------------------------
tab_setup, tab_demand, tab_routing, tab_calc, tab_opt, tab_whatif, tab_save = st.tabs([
    "1. Setup", "2. Demand & Inventory", "3. Routing & Capacity", "4. Calculations", "5. Optimise", "6. What-If", "7. Save / Export"
])

with tab_setup:
    st.subheader("Factory Setup")
    st.caption("You can change products, machines, costs, quality, material limits and production limits. Nothing is hard-coded into the calculations.")
    new_products = st.data_editor(products, num_rows="dynamic", use_container_width=True, key="products_editor")
    new_machines = st.data_editor(machines, num_rows="dynamic", use_container_width=True, key="machines_editor")
    if st.button("Apply product & machine setup", type="primary"):
        state["products"] = df_to_records(new_products)
        state["machines"] = df_to_records(new_machines)
        st.session_state.state = state
        st.success("Factory setup saved in the current session.")
        st.rerun()

    st.markdown("### Planning periods")
    n_weeks = st.number_input("Number of planning weeks", min_value=1, max_value=26, value=len(weeks), step=1)
    week_names = [f"Week {i+1}" for i in range(int(n_weeks))]
    if st.button("Set planning weeks"):
        # Preserve existing week data where possible.
        old = demand.copy()
        rows = []
        for w in week_names:
            for p in products.Product:
                oldrow = old[(old.Week == w) & (old.Product == p)]
                if oldrow.empty:
                    rows.append([w,p,0.0,0.0,20.0])
                else:
                    r=oldrow.iloc[0]
                    rows.append([w,p,r.Forecast_Demand,r.Actual_Demand,r.Safety_Stock_pct])
        state["weeks"] = week_names
        state["demand"] = rows
        st.session_state.state = state
        st.rerun()

    st.markdown("### Assumptions")
    st.write("• Safety stock is typically demonstrated as about **20% of Forecast Demand** (editable per week/product).")
    st.write("• Available hours = (Base capacity − Downtime) × Efficiency, with overtime added separately in optimisation.")
    st.write("• Defective units = Actual/Planned Production × Defect Rate; Good Units = Production − Defective Units.")

with tab_demand:
    st.subheader("Demand, Inventory & Safety Stock")
    st.caption("Forecast Demand is different from Actual Demand and uses inventory as the bridge between demand and production. Forecast Demand + Desired Closing Inventory − Opening Inventory for basic production requirement.")
    new_demand = st.data_editor(demand, num_rows="dynamic", use_container_width=True, key="demand_editor")
    if st.button("Apply demand & inventory inputs", type="primary"):
        state["demand"] = df_to_records(new_demand)
        st.session_state.state = state
        st.success("Demand inputs saved in the current session.")
        st.rerun()

    calc = teaching_calc(products, new_demand, weeks)
    st.markdown("### Basic production requirement")
    st.dataframe(calc, use_container_width=True, hide_index=True)
    st.caption("Formula: Production Required = Forecast Demand + Desired Closing Inventory − Opening Inventory; if negative, no additional production is required.")

with tab_routing:
    st.subheader("Product–Machine Routing")
    st.caption("Enter machine-hours required per unit. A product may use several machines.")
    new_routing = st.data_editor(routing, num_rows="dynamic", use_container_width=True, key="routing_editor")
    if st.button("Apply routing"):
        state["routing"] = df_to_records(new_routing)
        st.session_state.state = state
        st.success("Routing saved in the current session.")
        st.rerun()

    st.markdown("### Machine capacity")
    st.dataframe(machines, use_container_width=True, hide_index=True)
    st.caption("Available capacity is calculated from base hours, shifts, downtime and operating efficiency; overtime is handled as a separate decision variable.")
    route_pivot = new_routing.pivot_table(index="Product", columns="Machine", values="Hours_Per_Unit", aggfunc="sum", fill_value=0)
    st.markdown("### Routing matrix")
    st.dataframe(route_pivot, use_container_width=True)

with tab_calc:
    st.subheader("Trace the Planning Calculations")
    calc = teaching_calc(products, demand, weeks)
    st.markdown("### Step 1 — Demand variance")
    cv = calc.copy()
    cv["Demand_Variance"] = cv["Actual_Demand"] - cv["Forecast_Demand"]
    st.dataframe(cv[["Week","Product","Forecast_Demand","Actual_Demand","Demand_Variance"]], use_container_width=True, hide_index=True)

    st.markdown("### Step 2 — Required production")
    st.dataframe(calc, use_container_width=True, hide_index=True)

    # Baseline production = basic requirement, then show machine load.
    baseline = {(r.Week, r.Product): r.Production_Required for r in calc.itertuples()}
    cap = capacity_table(products, machines, routing, weeks, production=baseline)
    st.markdown("### Step 3 — Required machine hours and capacity")
    st.dataframe(cap, use_container_width=True, hide_index=True)

    if not cap.empty:
        bottlenecks = cap.sort_values("Capacity_Gap_Hours", ascending=False).groupby("Week").head(1)
        st.markdown("### Bottleneck by week")
        st.dataframe(bottlenecks[["Week","Machine","Required_Hours","Available_Hours","Capacity_Utilisation_Pct","Capacity_Gap_Hours","Status"]], use_container_width=True, hide_index=True)

    st.markdown("### Quality-adjusted production")
    qrows=[]
    for r in calc.itertuples():
        pr = products[products.Product==r.Product].iloc[0]
        defect = r.Production_Required * pr.Defect_Rate_pct/100
        good = r.Production_Required - defect
        qrows.append([r.Week,r.Product,r.Production_Required,pr.Defect_Rate_pct,defect,good])
    st.dataframe(pd.DataFrame(qrows, columns=["Week","Product","Planned_Production","Defect_Rate_pct","Defective_Units","Good_Units"]), use_container_width=True, hide_index=True)

with tab_opt:
    st.subheader("Optimised Production Plan")
    st.caption("The production quantity is the decision variable and we recommend Solver-style optimisation subject to machine capacity, production limits, inventory/safety stock and material limits.")
    c1,c2,c3 = st.columns(3)
    with c1:
        allow_shortage = st.checkbox("Allow shortage with penalty", value=True)
    with c2:
        demand_scale = st.slider("Demand multiplier", 0.50, 1.50, 1.00, 0.05)
    with c3:
        capacity_scale = st.slider("Capacity multiplier", 0.50, 1.20, 1.00, 0.05)
    extra_ot = st.number_input("Extra overtime available per machine (hours)", min_value=0.0, value=0.0, step=10.0)

    if st.button("🚀 Run optimisation", type="primary"):
        result = optimize_plan(products, machines, routing, demand, weeks, allow_shortage,
                               demand_scale, capacity_scale, extra_ot)
        st.session_state.opt_result = result

    result = st.session_state.get("opt_result")
    if result:
        if not result.get("success"):
            st.error("The model could not find a feasible plan: " + str(result.get("message")))
        else:
            plan = result["plan"]
            st.success(f"Optimisation completed. Objective value = {result['objective']:,.2f}")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total planned production", f"{plan.Planned_Production.sum():,.0f}")
            k2.metric("Total shortage", f"{plan.Shortage.sum():,.0f}")
            k3.metric("Total defective units", f"{plan.Defective_Units.sum():,.0f}")
            k4.metric("Total good units", f"{plan.Good_Units.sum():,.0f}")
            st.dataframe(plan, use_container_width=True, hide_index=True)

            ot_rows=[]
            for (w,m),v in result["overtime"].items(): ot_rows.append([w,m,v])
            ot_df=pd.DataFrame(ot_rows, columns=["Week","Machine","Overtime_Hours"])
            st.markdown("### Overtime decisions")
            st.dataframe(ot_df, use_container_width=True, hide_index=True)

            prod_dict={(r.Week,r.Product):r.Planned_Production for r in plan.itertuples()}
            cap=capacity_table(products,machines,routing,weeks,production=prod_dict,capacity_scale=capacity_scale,extra_overtime=extra_ot)
            st.markdown("### Machine capacity after optimisation")
            st.dataframe(cap,use_container_width=True,hide_index=True)

            overloaded=cap[cap.Capacity_Gap_Hours>1e-6]
            if overloaded.empty:
                st.success("The optimised plan is within base machine capacity plus the allowed overtime.")
            else:
                st.warning("Some machine/week combinations still have a positive capacity gap. Review product limits, material restrictions, demand, or overtime.")

            st.markdown("### Management recommendation")
            worst=cap.sort_values("Capacity_Gap_Hours",ascending=False).iloc[0]
            total_short=plan.Shortage.sum()
            if total_short > 0:
                rec=(f"{worst.Machine} is the tightest capacity resource in {worst.Week}. "
                     f"The optimised plan leaves {total_short:,.0f} units of shortage because the model is balancing demand, "
                     f"capacity, product limits, material factors and cost. Consider overtime, reallocation, rescheduling or accepting shortage.")
            else:
                rec=(f"The plan is feasible for the entered scenario. {worst.Machine} in {worst.Week} is the tightest resource. "
                     "Use the what-if controls to demonstrate how demand, capacity, overtime and material restrictions change the decision.")
            st.info(rec)

with tab_whatif:
    st.subheader("What-If Analysis")
    st.caption("We propose demand +20%, capacity −20%, overtime +100 hours, machine breakdown and raw-material constraint scenarios.")
    scenario = st.selectbox("Choose a scenario", [
        "Base case", "Demand +20%", "Machine capacity -20%", "Overtime +100 hours", "Raw material stress", "Combined stress"
    ])
    if scenario == "Base case":
        ds, cs, eo = 1.0, 1.0, 0.0
    elif scenario == "Demand +20%":
        ds, cs, eo = 1.20, 1.0, 0.0
    elif scenario == "Machine capacity -20%":
        ds, cs, eo = 1.0, 0.80, 0.0
    elif scenario == "Overtime +100 hours":
        ds, cs, eo = 1.0, 1.0, 100.0
    elif scenario == "Raw material stress":
        ds, cs, eo = 1.0, 1.0, 0.0
        products = products.copy()
        products["Raw_Material_Factor"] = products["Raw_Material_Factor"] * 0.82
    else:
        ds, cs, eo = 1.20, 0.80, 50.0
        products = products.copy()
        products["Raw_Material_Factor"] = products["Raw_Material_Factor"] * 0.90

    st.write(f"**Scenario settings:** demand × {ds:.2f}, capacity × {cs:.2f}, extra overtime = {eo:.0f} hours/machine.")
    if st.button("Run selected scenario", type="primary"):
        st.session_state.whatif = optimize_plan(products,machines,routing,demand,weeks,True,ds,cs,eo)

    wi=st.session_state.get("whatif")
    if wi and wi.get("success"):
        plan=wi["plan"]
        prod_dict={(r.Week,r.Product):r.Planned_Production for r in plan.itertuples()}
        cap=capacity_table(products,machines,routing,weeks,production=prod_dict,capacity_scale=cs,extra_overtime=eo)
        a,b,c,d=st.columns(4)
        a.metric("Objective",f"{wi['objective']:,.0f}")
        b.metric("Shortage",f"{plan.Shortage.sum():,.0f}")
        c.metric("Production",f"{plan.Planned_Production.sum():,.0f}")
        d.metric("Worst capacity gap",f"{max(0,cap.Capacity_Gap_Hours.max()):,.1f} h")
        st.dataframe(cap,use_container_width=True,hide_index=True)
        st.dataframe(plan,use_container_width=True,hide_index=True)
    elif wi:
        st.error(wi.get("message","Scenario could not be solved."))

with tab_save:
    st.subheader("Save, Load & Export")
    st.write("The app saves scenarios locally in **manufacturing_planner.db** when you click Save. This means your entered values do not disappear when you close and reopen the local app.")
    scenario_name=st.text_input("Scenario name", value=f"Scenario {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    current_state=make_state(products,machines,routing,demand,weeks)
    if st.button("💾 Save scenario to local database", type="primary"):
        save_scenario(scenario_name,current_state)
        st.success("Scenario saved. Use 'Load a saved scenario' in the sidebar to reopen it.")

    st.markdown("### Download current inputs")
    payload=json.dumps(current_state,indent=2,default=str)
    st.download_button("Download scenario JSON", payload, file_name="manufacturing_scenario.json", mime="application/json")

    st.markdown("### Download input tables")
    for label, df, fn in [
        ("Products CSV",products,"products.csv"),
        ("Machines CSV",machines,"machines.csv"),
        ("Routing CSV",routing,"routing.csv"),
        ("Demand CSV",demand,"demand.csv")]:
        st.download_button(label,df.to_csv(index=False),file_name=fn,mime="text/csv",key="dl_"+fn)

    st.markdown("### Explanation sequence")
    st.markdown("1. Enter **Forecast Demand** and **Actual Demand**.  2. Explain opening inventory and safety stock.  3. Calculate production requirement.  4. Show routing and machine-hours.  5. Calculate capacity utilisation and gap.  6. Identify the bottleneck.  7. Run optimisation.  8. Change one assumption in What-If and ask students: **What should management change, and why?**")

st.divider()
st.caption("Built directly around the uploaded Production & Capacity Planning concepts: demand, inventory, production requirement, routing, machine capacity, utilisation, bottleneck, quality, priority, cost, raw-material constraints, optimisation and what-if analysis. The final objective is a defensible production decision supported by calculations.")
