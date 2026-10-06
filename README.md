# Manufacturing Production & Capacity Planning — Streamlit App

This is a classroom-ready interactive production and capacity planning simulator based on the uploaded 24-slide teaching deck.

## What it does

- Dynamic product, machine, routing and weekly demand inputs
- Forecast vs actual demand
- Opening inventory and safety stock
- Production requirement calculation
- Product–machine routing and machine-hour calculation
- Available capacity from base hours, shifts, efficiency and downtime
- Capacity utilisation and capacity gap
- Bottleneck identification
- Defect rate and good-unit calculation
- Product priority, production cost, holding cost and shortage penalty
- Raw-material factor constraint
- Solver-style linear optimisation using SciPy
- Overtime decision variable
- What-if scenarios: demand +20%, capacity -20%, overtime +100 hours, raw-material stress and combined stress
- Local SQLite scenario saving/loading
- JSON/CSV export

## Run

1. Install Python 3.10+.
2. Open Terminal in this folder.
3. Create an environment:

   `python -m venv .venv`

4. Activate it on macOS/Linux:

   `source .venv/bin/activate`

5. Install packages:

   `pip install -r requirements.txt`

6. Start the app:

   `python -m streamlit run app.py`

The browser should open at the local Streamlit address shown in the terminal.

## Important classroom point

The app starts with editable teaching values so you can demonstrate the PPT immediately, but the calculations are not hard-coded to those values. Students can replace products, machines, routing, demand, inventory, capacity, costs and constraints.

The **Save scenario** button stores the current scenario in a local SQLite database. If you are using the app on a classroom laptop, the saved scenarios remain available when you reopen the app.

## Main formulas used

- Demand variance = Actual Demand − Forecast Demand
- Safety Stock = Forecast Demand × Safety Stock %
- Production Required = Forecast Demand + Desired Closing Inventory − Opening Inventory
- Defective Units = Production × Defect Rate
- Good Units = Production − Defective Units
- Required Machine Hours = Production Quantity × Machine Hours per Unit
- Available Hours = (Base Hours × Shifts − Downtime) × Efficiency
- Capacity Utilisation % = Required Hours ÷ Available Hours × 100
- Capacity Gap = Required Hours − Available Hours

For optimisation, production quantity is the main decision variable. The model balances production cost, holding cost, shortage penalty and overtime cost subject to production, machine, material and inventory constraints.
