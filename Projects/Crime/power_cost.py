# Power cost estimator for GPU training jobs (on-prem)

# === USER SETTINGS ===
city_runtimes = {'Los Angeles' : 1.79, 'New Orleans' : 0.65, 'Philadelphia' : 0.25, 'Las Vegas' : 0.32}
gpu_power_watts = 400                       # GPU power draw in watts
cpu_other_power_watts = 300                 # Power draw for CPU, RAM, etc.
pue = 1.8                                   # Power Usage Effectiveness (1.0 = ideal, 2.0 = inefficient)
energy_rate = 0.06696                       # Electricity cost in $ per kWh
A100_cost = 12500                            # Nvidia A100 GPU cost
useful_life_years = 5                       # Useful life of the A100 GPU in years
runs_per_year = 365                         # Number of runs per year

# === CALCULATIONS ===
# Depreciation cost per run (straight-line)
annual_depreciation = A100_cost / useful_life_years
depreciation_per_run = annual_depreciation / runs_per_year  # Depreciation per run

# Total system power usage calculation
total_system_kw = (gpu_power_watts + cpu_other_power_watts) / 1000.0
effective_power_kw = total_system_kw * pue

city_costs = {}

for city, runtime in city_runtimes.items():
    # Calculate energy cost per run
    energy_used = effective_power_kw * runtime
    power_cost = energy_used * energy_rate

    # Add depreciation cost per run
    total_cost_per_run = power_cost + depreciation_per_run

    city_costs[city] = round(total_cost_per_run, 2)

# === OUTPUT ===
print("Estimated total cost per run (including depreciation):")
for city, cost in city_costs.items():
    print(f"{city}: ${cost}")
