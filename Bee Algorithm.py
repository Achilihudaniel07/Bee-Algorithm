# System dynamics for LFC (frequency response model)
def power_system_dynamics(x, t, u):
    frequency, integral_error = x
    control_signal = u
    damping = 0.8
    inertia = 8
    power_change = -0.6  # Step disturbance

    d_frequency = (control_signal - damping * frequency + power_change) / inertia
    d_integral_error = -frequency  # Integral of error (frequency deviation)

    return [d_frequency, d_integral_error]


# Fuzzy Logic Controller with undershoot and overshoot
def fuzzy_controller(frequency, d_frequency, mf_params):
    # Extract fuzzy membership function parameters
    undershoot_params, overshoot_params = mf_params[:3], mf_params[3:]

    # Define membership functions for undershoot and overshoot
    undershoot_mf = fuzz.trimf([frequency], [-undershoot_params[2], -undershoot_params[1], -undershoot_params[0]])
    overshoot_mf = fuzz.trimf([frequency], [overshoot_params[0], overshoot_params[1], overshoot_params[2]])

    # Define fuzzy membership functions for derivative of frequency
    dfreq_low = fuzz.trimf([d_frequency], [-1, -0.5, 0])
    dfreq_zero = fuzz.trimf([d_frequency], [-0.5, 0, 0.5])
    dfreq_high = fuzz.trimf([d_frequency], [0, 0.5, 1])

    # Define fuzzy rules (example rules for demonstration)
    if undershoot_mf > 0.5 or dfreq_low > 0.5:
        control_output = 1  # High positive control signal for undershoot
    elif overshoot_mf > 0.5 or dfreq_high > 0.5:
        control_output = -1  # High negative control signal for overshoot
    else:
        control_output = 0  # No change

    return control_output


# Objective function to minimize (ITAE)
def objective_function(mf_params):
    initial_conditions = [0.0, 0.0]
    t = np.linspace(0, 10, 100)
    frequency_deviation = []

    for i in range(len(t) - 1):
        u = fuzzy_controller(initial_conditions[0], initial_conditions[1], mf_params)
        response = odeint(power_system_dynamics, initial_conditions, [t[i], t[i+1]], args=(u,))
        initial_conditions = response[-1]
        frequency_deviation.append(initial_conditions[0])

    # Calculate ITAE
    itae = np.sum(t * np.abs(frequency_deviation))
    return itae

    


# Bees Algorithm parameters
num_bees = 30
num_elite_sites = 3
num_best_sites = 3
neighborhood_size = 0.1
num_iterations = 100
bounds = [(0.1, 1.0, 2.0), (0.1, 1.0, 2.0)]  # Bounds for undershoot and overshoot parameters


# Initialize bees' positions
bees_positions = np.random.uniform([b[0] for b in bounds] * 2, [b[1] for b in bounds] * 2, (num_bees, 6))
bees_fitness = np.array([objective_function(bee) for bee in bees_positions])



# Main loop of Bees Algorithm
for iteration in range(num_iterations):
    sorted_indices = np.argsort(bees_fitness)
    bees_positions = bees_positions[sorted_indices]
    bees_fitness = bees_fitness[sorted_indices]


    # Elite Site Exploitation
    for i in range(num_elite_sites):
        elite_position = bees_positions[i]
        for j in range(int(num_bees / num_elite_sites)):
            candidate = elite_position + np.random.uniform(-neighborhood_size, neighborhood_size, 6)
            candidate = np.clip(candidate, [b[0] for b in bounds] * 2, [b[1] for b in bounds] * 2)
            candidate_fitness = objective_function(candidate)
            if candidate_fitness < bees_fitness[i]:
                bees_positions[i] = candidate
                bees_fitness[i] = candidate_fitness


    # Best Site Exploration
    for i in range(num_elite_sites, num_best_sites):
        best_position = bees_positions[i]
        for j in range(int(num_bees / num_best_sites)):
            candidate = best_position + np.random.uniform(-neighborhood_size, neighborhood_size, 6)
            candidate = np.clip(candidate, [b[0] for b in bounds] * 2, [b[1] for b in bounds] * 2)
            candidate_fitness = objective_function(candidate)
            if candidate_fitness < bees_fitness[i]:
                bees_positions[i] = candidate
                bees_fitness[i] = candidate_fitness


    # Random Exploration
    for i in range(num_best_sites, num_bees):
        bees_positions[i] = np.random.uniform([b[0] for b in bounds] * 2, [b[1] for b in bounds] * 2)
        bees_fitness[i] = objective_function(bees_positions[i])

    # Reduce Neighborhood Size
    neighborhood_size *= 0.1

    # Track progress
    best_fitness = bees_fitness[0]
    print(f"Iteration {iteration+1}, Best Fitness (ITAE): {best_fitness}")


# Best solution
best_solution = bees_positions[0]
print("Optimal Fuzzy MF Parameters for Undershoot/Overshoot:", best_solution)
print("Minimum ITAE:", best_fitness)
