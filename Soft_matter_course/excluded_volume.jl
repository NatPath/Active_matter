function run_walk(initial_spot, steps)
    saved_states = [initial_spot]
    for i in 1:steps
        current = saved_states[end]
        possible_moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors = [current .+ mv for mv in possible_moves if (current .+ mv ∉ saved_states)]
        if isempty(neighbors)
            return (false, 0)  # Return false if the walk failed
        end
        new_spot = rand(neighbors)
        push!(saved_states, new_spot)
    end
    final_state = saved_states[end]
    return (true, final_state[1]^2 + final_state[2]^2)  # Return true if the walk succeeded
end

function run_many_walks(n_walks, steps)
    sum = 0
    successful_walks = 0
    for i in 1:n_walks
        success, distance_squared = run_walk([0, 0], steps)
        if success
            sum += distance_squared
            successful_walks += 1
        end
    end
    return sqrt(sum / successful_walks)
end

results_arr = []
max_N = 800
N_range = 300:100:max_N
for i in N_range
    push!(results_arr, run_many_walks(1000000, i))
end

using Statistics
using LinearAlgebra
using Plots

# Perform linear regression
x = log.(N_range)
y = log.(results_arr)
A = [ones(length(x)) x]
b = y
coefficients = A \ b

# Extract the slope
slope = coefficients[2]

# Plot the data and regression line
scatter!(x, y, label = "Data")
xlabel!("log(i)")
ylabel!("log(results_arr)")
plot!(x, A * coefficients, label = "Regression Line")

# Print the slope
println("Slope of the regression line: ", slope)

