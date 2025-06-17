import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime
import plotly.graph_objects as go

# ─── Tier multipliers ──────────────────────────────────────────────────────────
tier_multipliers = {
    'Luxury':   1.2,
    'Midrange': 1.0,
    'Economy':  0.8
}

# ─── Load model and encoders ────────────────────────────────────────────────────
with open('car_maintenance_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)
with open('le_make.pkl', 'rb') as f:
    le_make = pickle.load(f)
with open('le_model.pkl', 'rb') as f:
    le_model = pickle.load(f)

# ─── MSRP and Edmunds ratings ──────────────────────────────────────────────────
msrp_data = {
    ('Acura','MDX'):51000,('Acura','RDX'):41000,('Acura','TLX'):39000,('Acura','ILX'):31000,('Acura','NSX'):157000,
    ('BMW','X1'):39000,('BMW','X3'):46000,('BMW','X5'):61000,('BMW','X7'):74000,
    ('Chevrolet','Malibu'):26000,('Chevrolet','Tahoe'):54000,('Chevrolet','Silverado'):47000,
    ('Ford','F-150'):35000,('Ford','Mustang'):31000,('Ford','Escape'):27000,
    ('Honda','Civic'):23000,('Honda','Accord'):28000,('Honda','CR-V'):29000,
    ('Hyundai','Elantra'):20000,('Hyundai','Sonata'):25000,('Hyundai','Tucson'):27000,
    ('Kia','Optima'):24000,('Kia','Soul'):21000,('Kia','Sportage'):26000,
    ('Lexus','ES'):42000,('Lexus','RX'):50000,('Lexus','NX'):42000,
    ('Mazda','3'):22000,('Mazda','6'):25000,('Mazda','CX-5'):28000,
    ('Mercedes-Benz','C-Class'):44000,('Mercedes-Benz','E-Class'):56000,('Mercedes-Benz','GLC'):49000,
    ('Mini','Cooper'):25000,('Mini','Countryman'):32000,
    ('Nissan','Altima'):25000,('Nissan','Sentra'):21000,('Nissan','Rogue'):28000,
    ('Porsche','911'):105000,('Porsche','Cayenne'):79000,
    ('Subaru','Impreza'):21000,('Subaru','Outback'):28000,('Subaru','Forester'):27000,
    ('Toyota','Camry'):27000,('Toyota','Corolla'):21000,('Toyota','RAV4'):29000,
    ('Volvo','XC60'):48000,('Volvo','XC90'):56000
}
edmunds_ratings = {
    ('Acura','MDX'):{2020:4.3,2021:4.2,2022:4.1,2023:4.1,2024:4.0},
    ('Acura','RDX'):{2020:4.1,2021:4.2,2022:4.1,2023:4.0,2024:4.0},
    ('BMW','X1'):{2020:4.2,2021:4.3,2022:4.3,2023:4.2,2024:4.2},
    ('BMW','X3'):{2020:4.4,2021:4.5,2022:4.4,2023:4.3,2024:4.2},
    ('Chevrolet','Malibu'):{2020:3.9,2021:4.0,2022:4.0,2023:4.1,2024:4.1},
    ('Ford','F-150'):{2020:4.1,2021:4.2,2022:4.3,2023:4.2,2024:4.1},
    ('Ford','Escape'):{2020:3.8,2021:3.9,2022:4.0,2023:4.0,2024:4.0},
    ('Honda','Civic'):{2020:4.1,2021:4.2,2022:4.3,2023:4.4,2024:4.4},
    ('Honda','Accord'):{2020:4.2,2021:4.3,2022:4.3,2023:4.3,2024:4.4},
    ('Hyundai','Elantra'):{2020:4.0,2021:4.1,2022:4.2,2023:4.1,2024:4.1},
    ('Kia','Soul'):{2020:3.8,2021:3.9,2022:4.0,2023:4.0,2024:4.0},
    ('Lexus','RX'):{2020:4.3,2021:4.4,2022:4.4,2023:4.4,2024:4.3},
    ('Mazda','CX-5'):{2020:4.4,2021:4.5,2022:4.5,2023:4.5,2024:4.5},
    ('Mercedes-Benz','C-Class'):{2020:4.2,2021:4.3,2022:4.3,2023:4.2,2024:4.2},
    ('Mini','Cooper'):{2020:4.0,2021:4.1,2022:4.1,2023:4.0,2024:4.0},
    ('Nissan','Altima'):{2020:3.9,2021:4.0,2022:4.0,2023:4.1,2024:4.1},
    ('Porsche','911'):{2020:4.8,2021:4.9,2022:4.9,2023:4.8,2024:4.8},
    ('Subaru','Outback'):{2020:4.3,2021:4.4,2022:4.4,2023:4.4,2024:4.4},
    ('Toyota','Camry'):{2020:4.0,2021:4.1,2022:4.2,2023:4.2,2024:4.3},
    ('Toyota','Corolla'):{2020:3.9,2021:4.0,2022:4.0,2023:4.1,2024:4.1},
    ('Volvo','XC60'):{2020:4.5,2021:4.5,2022:4.5,2023:4.4,2024:4.4}
}

# ─── Lookup tables ─────────────────────────────────────────────────────────────
car_makes_and_models = {
    'Acura':['MDX','RDX','TLX','ILX','NSX'],
    'BMW':['X1','X3','X5','X7'],
    'Chevrolet':['Malibu','Tahoe','Silverado'],
    'Ford':['F-150','Mustang','Escape'],
    'Honda':['Civic','Accord','CR-V'],
    'Hyundai':['Elantra','Sonata','Tucson'],
    'Kia':['Optima','Soul','Sportage'],
    'Lexus':['ES','RX','NX'],
    'Mazda':['3','6','CX-5'],
    'Mercedes-Benz':['C-Class','E-Class','GLC'],
    'Mini':['Cooper','Countryman'],
    'Nissan':['Altima','Sentra','Rogue'],
    'Porsche':['911','Cayenne'],
    'Subaru':['Impreza','Outback','Forester'],
    'Toyota':['Camry','Corolla','RAV4'],
    'Volvo':['XC60','XC90']
}

average_mpg = {
    'Acura':{'MDX':22,'RDX':23,'TLX':24,'ILX':25,'NSX':21},
    'BMW':{'X1':28,'X3':25,'X5':22,'X7':21},
    'Chevrolet':{'Malibu':29,'Tahoe':20,'Silverado':23},
    'Ford':{'F-150':20,'Mustang':24,'Escape':27},
    'Honda':{'Civic':32,'Accord':30,'CR-V':29},
    'Hyundai':{'Elantra':33,'Sonata':31,'Tucson':26},
    'Kia':{'Optima':27,'Soul':28,'Sportage':26},
    'Lexus':{'ES':26,'RX':23,'NX':25},
    'Mazda':{'3':28,'6':26,'CX-5':25},
    'Mercedes-Benz':{'C-Class':25,'E-Class':24,'GLC':24},
    'Mini':{'Cooper':29,'Countryman':27},
    'Nissan':{'Altima':32,'Sentra':29,'Rogue':28},
    'Porsche':{'911':22,'Cayenne':22},
    'Subaru':{'Impreza':31,'Outback':29,'Forester':28},
    'Toyota':{'Camry':32,'Corolla':33,'RAV4':28},
    'Volvo':{'XC60':24,'XC90':21}
}

maintenance_schedule = {
    'Oil Change': 5000,
    'Oil Filter Replacement': 5000,
    'Tire Rotation': 7500,
    'Tire Balance': 15000,
    'Cabin Air Filter Replacement': 15000,
    'Engine Air Filter Replacement': 15000,
    'Brake Inspection': 10000,
    'Brake Pad Replacement': 40000,
    'Brake Fluid Replacement': 30000,
    'Coolant Flush': 60000,
    'Coolant Level/Condition Inspection': 20000,
    'Battery Check': 20000,
    'Battery Replacement': 50000,
    'Spark Plug Replacement': 40000,
    'Ignition Coil Inspection': 40000,
    'Valve Adjustment/Inspection': 60000,
    'Transmission Fluid Replacement': 60000,
    'Transmission Inspection': 30000,
    'Fuel Filter Replacement': 30000,
    'Power Steering Fluid Replacement': 30000,
    'Power Steering System Inspection': 30000,
    'Differential Fluid Replacement': 60000,
    'Transfer Case Fluid Replacement': 60000,
    'Drive Belt Inspection/Replacement': 60000,
    'Serpentine Belt Replacement': 90000,
    'Timing Belt Replacement': 100000,
    'Timing Chain Inspection': 100000,
    'Alternator Inspection': 90000,
    'Wheel Bearings Check': 90000,
    'Suspension Inspection': 30000,
    'Shock/Strut Replacement': 100000,
    'Exhaust System Inspection': 30000,
    'AC System Service': 60000,
    'Head Gasket Inspection': 120000,
    'Major Overhaul Recommended': 150000,
    'Radiator Hose Replacement': 90000,
    'PCV Valve Replacement': 60000,
    'Windshield Wiper Replacement': 12000,
    'Lights/Bulbs Inspection': 12000,
    'Chassis Lubrication': 12000,
    'Alignment Check': 20000,
    'Tire Replacement': 50000,
    'Engine Mount Inspection': 100000,
    'Emissions Inspection': 50000,
    'HVAC System Inspection': 30000,
    'Door/Hood Latch Lubrication': 30000,
    'Sunroof/Moonroof Lubrication': 30000,
    'Rust/Corrosion Inspection': 50000,
    'Underbody Inspection': 50000,
    'Fuel Injector Cleaning': 60000,
    'Throttle Body Cleaning': 60000,
    'Evaporative Emissions System Inspection': 100000,
    'Oxygen Sensor Replacement': 100000,
    'Catalytic Converter Inspection': 100000,
    'ABS System Inspection': 60000,
    'Parking Brake Adjustment': 30000,
    'Differential Inspection': 60000,
    'Transfer Case Inspection': 60000,
    'Steering Linkage Inspection': 60000
}
maintenance_costs = {
    'Oil Change': 50,
    'Oil Filter Replacement': 25,
    'Tire Rotation': 40,
    'Tire Balance': 50,
    'Cabin Air Filter Replacement': 50,
    'Engine Air Filter Replacement': 40,
    'Brake Inspection': 40,
    'Brake Pad Replacement': 150,
    'Brake Fluid Replacement': 100,
    'Coolant Flush': 120,
    'Coolant Level/Condition Inspection': 30,
    'Battery Check': 25,
    'Battery Replacement': 180,
    'Spark Plug Replacement': 125,
    'Ignition Coil Inspection': 40,
    'Valve Adjustment/Inspection': 200,
    'Transmission Fluid Replacement': 150,
    'Transmission Inspection': 60,
    'Fuel Filter Replacement': 80,
    'Power Steering Fluid Replacement': 75,
    'Power Steering System Inspection': 40,
    'Differential Fluid Replacement': 90,
    'Transfer Case Fluid Replacement': 90,
    'Drive Belt Inspection/Replacement': 80,
    'Serpentine Belt Replacement': 100,
    'Timing Belt Replacement': 750,
    'Timing Chain Inspection': 100,
    'Alternator Inspection': 60,
    'Wheel Bearings Check': 100,
    'Suspension Inspection': 60,
    'Shock/Strut Replacement': 600,
    'Exhaust System Inspection': 50,
    'AC System Service': 150,
    'Head Gasket Inspection': 100,
    'Major Overhaul Recommended': 3500,
    'Radiator Hose Replacement': 120,
    'PCV Valve Replacement': 60,
    'Windshield Wiper Replacement': 30,
    'Lights/Bulbs Inspection': 20,
    'Chassis Lubrication': 40,
    'Alignment Check': 80,
    'Tire Replacement': 700,  # for a set of 4
    'Engine Mount Inspection': 50,
    'Emissions Inspection': 40,
    'HVAC System Inspection': 60,
    'Door/Hood Latch Lubrication': 20,
    'Sunroof/Moonroof Lubrication': 30,
    'Rust/Corrosion Inspection': 40,
    'Underbody Inspection': 50,
    'Fuel Injector Cleaning': 120,
    'Throttle Body Cleaning': 80,
    'Evaporative Emissions System Inspection': 60,
    'Oxygen Sensor Replacement': 200,
    'Catalytic Converter Inspection': 60,
    'ABS System Inspection': 60,
    'Parking Brake Adjustment': 40,
    'Differential Inspection': 60,
    'Transfer Case Inspection': 60,
    'Steering Linkage Inspection': 60
}


state_cost_multipliers = {
    'Alabama':1.00,'Alaska':1.10,'Arizona':1.05,'Arkansas':0.95,
    'California':1.25,'Colorado':1.10,'Connecticut':1.20,'Delaware':1.05,
    'Florida':1.00,'Georgia':1.00,'Hawaii':1.30,'Idaho':0.95,
    'Illinois':1.10,'Indiana':0.95,'Iowa':0.90,'Kansas':0.95,
    'Kentucky':0.95,'Louisiana':1.00,'Maine':1.05,'Maryland':1.20,
    'Massachusetts':1.25,'Michigan':1.00,'Minnesota':1.00,'Mississippi':0.90,
    'Missouri':0.95,'Montana':0.95,'Nebraska':0.95,'Nevada':1.05,
    'New Hampshire':1.10,'New Jersey':1.25,'New Mexico':0.95,'New York':1.30,
    'North Carolina':1.00,'North Dakota':0.90,'Ohio':0.95,'Oklahoma':0.95,
    'Oregon':1.10,'Pennsylvania':1.10,'Rhode Island':1.15,'South Carolina':0.95,
    'South Dakota':0.90,'Tennessee':0.95,'Texas':1.00,'Utah':1.00,
    'Vermont':1.10,'Virginia':1.10,'Washington':1.15,'West Virginia':0.90,
    'Wisconsin':1.00,'Wyoming':0.90
}

def get_scheduled_activities(start_mileage,end_mileage):
    activities=[]
    for name,interval in maintenance_schedule.items():
        next_due=((start_mileage//interval)+1)*interval
        if start_mileage<next_due<=end_mileage:
            activities.append(name)
    return activities

def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0

def depreciation_fraction(t,V0,Vmin=2000.0,a=0.182):
    return (1 - Vmin/V0)*(1 - safe_exp(-a*t))

def residual_value(t,V0,Vmin=2000.0,a=0.182):
    return V0*(1 - depreciation_fraction(t,V0,Vmin,a))

def estimate_vehicle_value(msrp,model_year,current_year,Vmin=2000.0,a=0.182):
    elapsed=max(0,current_year-model_year)
    elapsed=min(elapsed,50)
    return round(residual_value(elapsed,msrp,Vmin,a),2)

def estimate_used_car_price(msrp,purchase_year,current_year,mileage):
    try:
        age=max(0,current_year-purchase_year)
        age=min(age,50)
        dep_rate=(1 - (1500/msrp))*(1 - safe_exp(-0.182*age))
        mile_pen=(mileage//10000)*0.03
        dep=1-((1-dep_rate)**age)
        val=msrp*(1-dep)*(1-mile_pen)
        return max(val,1500)
    except:
        return 1500

def get_car_tier(make):
    lux={'BMW','Mercedes-Benz','Audi','Lexus','Jaguar','Porsche','Volvo','Mini','McLaren'}
    eco={'Toyota','Honda','Ford','Hyundai','Kia','Chevrolet','Subaru','Mazda','Nissan'}
    if make in lux: return 'Luxury'
    if make in eco: return 'Economy'
    return 'Midrange'
def predict_5_years_cost(make,model,model_year,current_mileage,avg_mpy,
                        mpg,fuel_price,purchase_price,state,
                        years,loan_amount,irate,lt_years,
                        user_age, start_age, msrp):
    enc_make=le_make.transform([make])[0]
    enc_model=le_model.transform([model])[0]
    tier_mult=tier_multipliers[get_car_tier(make)]*state_cost_multipliers[state]
    r=irate/100
    loan_pay=(loan_amount*r/(1-(1+r)**-lt_years)) if r>0 else loan_amount/lt_years
    prev_val=purchase_price
    Y,M,F,L,D,T,V,Acts,Lines,Insurance=[],[],[],[],[],[],[],[],[],[]
    for i in range(1,years+1):
        yr=model_year+i
        fm=current_mileage+avg_mpy*i
        df=pd.DataFrame([{
            'Make_Encoded':enc_make,'Model_Encoded':enc_model,
            'Year':model_year,'Mileage':fm,'Avg_Miles_Per_Year':avg_mpy
        }])
        base=trained_model.predict(df)[0]
        acts=get_scheduled_activities(current_mileage+avg_mpy*(i-1),fm)
        act_cost=sum(maintenance_costs[a]*tier_mult for a in acts)
        maint=base+act_cost
        fuel=(avg_mpy/mpg)*fuel_price
        reg=150
        total=maint+fuel+reg+loan_pay
        cur_val=estimate_vehicle_value(purchase_price,model_year,yr)
        depr=round(prev_val-cur_val,2)
        prev_val=cur_val
        Y.append(f"Year {i}"); M.append(round(maint,2)); F.append(round(fuel,2))
        L.append(round(loan_pay,2)); D.append(depr); T.append(round(total,2))
        V.append(cur_val); Acts.append(', '.join(acts) or 'None')
        Lines.append(f"Year {i}: Maintenance ${round(maint):,}, Fuel ${round(fuel):,}, Loan ${round(loan_pay):,}, Depreciation ${depr:,}\\n")
        
        # Insurance calculation for this year
        this_year_age = user_age + i - 1
        years_driving = this_year_age - start_age
        premium = 1200  # base
        if this_year_age < 25:
            premium += 700
        if this_year_age >= 25 and years_driving >= 10:
            premium -= 200
        if msrp and msrp > 50000:
            premium += 400
        if avg_mpy > 15000:
            premium += 200
        elif avg_mpy < 7000:
            premium -= 100
        premium *= state_cost_multipliers.get(state, 1.0)
        premium = max(700, round(premium))
        Insurance.append(premium)
        
    df_out=pd.DataFrame({
        'Year':Y,
        'Maintenance Cost':M,
        'Fuel Cost':F,
        'Loan Payment':L,
        'Depreciation Cost':D,
        'Total Cost':T,
        'Car Value':V,
        'Activities':Acts,
        'Insurance Premium':Insurance
    })
    inter=next((y for y,m,v in zip(Y,M,V) if m>v),None)
    return ''.join(Lines),df_out,inter

# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Car Ownership Cost Forecast")

make = st.selectbox(
    "Make",
    sorted(car_makes_and_models.keys()),
    key="make"
)

model = st.selectbox(
    "Model",
    car_makes_and_models[make],
    key="model"
)

model_year = st.selectbox(
    "Model Year",
    list(range(2000, datetime.now().year+1))[::-1],
    key="model_year"
)

# MSRP & rating
msrp = msrp_data.get((make, model))
rating = edmunds_ratings.get((make, model), {}).get(model_year)

if msrp:
    st.markdown(f"**MSRP (New):** ${msrp:,.0f}")

if rating:
    st.markdown(f"**Edmunds Rating for {model_year}:** {rating}/5")
    st.caption("⭐ Ratings sourced from Edmunds.com (accessed 2025-05-10)")
else:
    st.markdown("**Edmunds Rating:** Not available")
    st.caption("⭐ From Edmunds.com")

mileage = st.number_input(
    "Current Mileage",
    0, 300000, 1000,
    key="mileage"
)

your_price = st.number_input(
    "Your Purchase Price ($)",
    0, 2000000, 20000,
    key="your_price"
)

default_mpg = average_mpg.get(make, {}).get(model, 25)
mpg = st.number_input(
    "Miles Per Gallon",
    10, 100, value=default_mpg,
    help=f"EPA avg: {default_mpg}",
    key="mpg"
)
st.caption(f"EPA Avg MPG for {make} {model}: {default_mpg}")

loan_amount = st.number_input(
    "Loan Amount ($)",
    0.0, 2000000.0, 10000.0,
    key="loan_amount"
)

irate = st.number_input(
    "Interest Rate (%)",
    0.0, 25.0, 5.0,
    key="irate"
)

lt_years = st.number_input(
    "Loan Term (years)",
    1, 8, 3,
    key="lt_years"
)

gross = st.number_input(
    "Gross Annual Income ($)",
    0.0, 1000000.0, 60000.0,
    key="gross"
)

avg_mpy = st.number_input(
    "Avg Miles/Year",
    0, 100000, 10000,
    key="avg_mpy"
)
st.caption("Average Annual Mileage is 10,000")



fuel_price = st.number_input(
    "Fuel $/gallon",
    0.0, 9.50, 3.50,
    key="fuel_price"
)

state = st.selectbox(
    "State",
    sorted(state_cost_multipliers.keys()),
    key="state"
)

years = st.slider(
    "Years to forecast",
    1, 10, 5,
    key="years"
)

# Insurance-related user inputs
user_age = st.number_input(
    "Your Age",
    min_value=16, max_value=100, value=30,
    key="user_age"
)

start_age = st.number_input(
    "At what age did you start driving?",
    min_value=14, max_value=user_age, value=16,
    key="start_age"
)


if st.button("Predict Cost"):
    # 1) run the 5-year cost prediction
    summary, chart_df, intersection = predict_5_years_cost(
        make, model, model_year, mileage, avg_mpy,
        mpg, fuel_price, your_price,
        state, years, loan_amount, irate, lt_years,
        user_age, start_age, msrp
    )
    if not summary:
        st.error("Prediction failed—check inputs.")
        st.stop()

    # 2) refresh mileage & scheduled activities
    chart_df['Mileage'] = [
        mileage + avg_mpy * i
        for i in range(1, years + 1)
    ]
    chart_df['Activities'] = [
        ', '.join(
            get_scheduled_activities(
                mileage + avg_mpy * (i - 1),
                mileage + avg_mpy * i
            )
        ) or 'None'
        for i in range(1, years + 1)
    ]

    # 3) show forecast table
    st.subheader("📊 Forecast Results")
    st.dataframe(chart_df)
    st.caption("Includes maintenance, fuel, reg., loan payments, and insurance.")

    # 4) compute totals & % of income
    total      = chart_df['Total Cost'].sum()
    avg_annual = total / years
    pct_inc    = (avg_annual / gross) * 100
    avg_insurance = chart_df['Insurance Premium'].mean()

    st.subheader("💵 Total Cost of Ownership")
    st.metric("Total Cost Over Period",   f"${total:,.0f}")
    st.metric("Average Annual Cost",      f"${avg_annual:,.0f}")
    st.metric("% of Gross Income",        f"{pct_inc:.1f}%")
    st.metric("Est. Annual Insurance",    f"${avg_insurance:,.0f}")

    st.caption(
        "Insurance estimate uses a $1,200 US average for a safe driver with a standard car. "
        "Adjusts for age, driving experience, vehicle price, annual mileage, and state. "
        "Discounts apply as you turn 25 and/or reach 10 years of driving. Actual premiums may vary."
    )