###############################################################################
#                                                                             #
#   Run_all_instances.jl — Run all 4 instances sequentially                   #
#                                                                             #
#   Replication code for experiments in Section 9.2 of:                       #
#   "Exploration Optimization for Dynamic Assortment Personalization under    #
#    Linear Preferences"                                                      #
#                                                                             #
#   Usage:                                                                    #
#     julia Run_all_instances.jl                                              #
#                                                                             #
#   Terminal output for each instance is saved to:                            #
#     logs/instance_<n>_<timestamp>.log                                       #
#                                                                             #
###############################################################################

using Dates

const INSTANCES = [1, 2, 3, 4]

# ── Set up logs/ folder ───────────────────────────────────────────────────────
mkpath("logs")
session_timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

println("=" ^ 70)
println("  Running all $(length(INSTANCES)) instances")
println("  Logs will be saved to: logs/")
println("=" ^ 70)

total_start = time()

for instance in INSTANCES
    println("\n" * "=" ^ 70)
    println("  Starting Instance $instance")
    println("=" ^ 70 * "\n")

    start    = time()
    log_file = "logs/instance_$(instance)_$(session_timestamp).log"

    # Stream output to both terminal and log file simultaneously
    open(log_file, "w") do io
        proc = open(`julia Main.jl $instance`, "r")
        for line in eachline(proc)
            println(line)       # print to terminal
            println(io, line)   # write to log file
            flush(io)
        end
        wait(proc)
    end

    elapsed = round(time() - start; digits=1)
    println("  Instance $instance completed in $(elapsed)s")
    println("  Log saved to: $log_file")
end

total_elapsed = round(time() - total_start; digits=1)
println("\n" * "=" ^ 70)
println("  All instances completed in $(total_elapsed)s")
println("=" ^ 70)
