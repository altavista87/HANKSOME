
================================================================================
MALAYSIA ECONOMIC SHOCK TESTING FRAMEWORK
For EngramHANK GE Model (BNM/DOSM/WID Data Calibration)
================================================================================

FILE: src/MalaysiaShocks.jl
================================================================================
# Malaysia-Specific Shock Extensions for EngramHANK
# Maps real-world Malaysia economic events (2008-2025) to model shocks
# Data sources: BNM (Bank Negara Malaysia), DOSM, WID.world

module MalaysiaShocks

using LinearAlgebra
using FFTW
using ..EngramHANK

export MalaysiaShockSpec, ShockType, TransmissionChannel
export create_malaysia_params, apply_malaysia_shock
export solve_malaysia_scenario, run_shock_suite
export SHOCK_CATALOG

# =============================================================================
# SHOCK TYPE ENUMERATION
# =============================================================================

@enum ShockType begin
    EXTERNAL_DEMAND      # Global demand shocks (GFC, trade wars, China slowdown)
    TFP_SHOCK           # Productivity shocks (floods, supply disruptions)
    FISCAL_TAX          # GST, subsidy changes
    MONETARY_POLICY     # OPR changes, Fed spillovers
    TERMS_OF_TRADE      # Commodity price swings (oil, palm oil)
    RISK_PREMIUM        # Country risk, 1MDB, capital flows
    UNCERTAINTY         # Trade war uncertainty, policy uncertainty
    SECTORAL_DEMAND     # Tourism collapse, sector-specific
    TRANSFER_INCOME     # PRIHATIN transfers, fiscal support
    EXCHANGE_RATE       # Ringgit depreciation/appreciation
end

@enum TransmissionChannel begin
    EXPORT_CHANNEL      # Through export demand
    IMPORT_PRICE        # Through import costs
    CREDIT_SPREAD       # Through financial conditions
    CONSUMPTION         # Through household spending
    INVESTMENT          # Through firm investment
    LABOR_MARKET        # Through employment/wages
    FISCAL_BUDGET       # Through government finances
    EXCHANGE_RATE_CH    # Through currency movements
end

# =============================================================================
# SHOCK SPECIFICATION STRUCT
# =============================================================================

struct MalaysiaShockSpec
    id::Int
    name::String
    period::String
    shock_type::ShockType
    channels::Vector{TransmissionChannel}

    # Model mapping
    primary_shock::Symbol      # :r, :w, :TFP, :chi, :tau, :zeta, etc.
    magnitude::Float64         # Normalized magnitude
    persistence::Float64       # AR(1) persistence parameter

    # Calibration from data
    data_source::String
    actual_impact::Dict{String, Float64}  # Actual observed impacts
end

# =============================================================================
# MALAYSIA CALIBRATION PARAMETERS (BNM/DOSM/WID)
# =============================================================================

function create_malaysia_params(; year::Int=2024)
    """
    Create HANK parameters calibrated to Malaysia data.

    Sources:
    - BNM Annual Reports (interest rates, inflation)
    - DOSM Household Income Survey (income distribution)
    - WID.world (wealth inequality)
    """

    # Base Malaysia calibration
    base_params = Dict(
        # Preferences (estimated from consumption data)
        :β => 0.965,           # Discount factor (higher than US due to savings culture)
        :σ => 1.5,             # Risk aversion (moderate, Asian households)
        :ρ => 0.03,            # Death rate (younger population than US)

        # Income process (DOSM Household Income Survey)
        :α => 0.75,            # Income persistence (high in Malaysia)
        :σ_ε => 0.25,          # Income shock std

        # Steady-state prices (BNM data, 2010-2024 average)
        :r_ss => 0.025,        # Average OPR ~3%, real rate ~2.5%
        :w_ss => 1.0,          # Normalized

        # Asset grid (EPF data suggests significant liquid assets)
        :a_min => -0.5,        # Some borrowing (credit cards, personal loans)
        :a_max => 100.0,       # Upper tail of wealth distribution
        :na => 150,
        :ny => 7
    )

    # Adjust for specific periods
    if year <= 2010
        base_params[:r_ss] = 0.035  # Higher rates pre-GFC
    elseif year <= 2015
        base_params[:r_ss] = 0.030  # Post-GFC normalization
    elseif year <= 2020
        base_params[:r_ss] = 0.025  # Low rate era
    else
        base_params[:r_ss] = 0.028  # Post-COVID normalization
    end

    return HANKParameters{Float64}(; base_params...)
end

# =============================================================================
# SHOCK CATALOG - 20 MALAYSIA ECONOMIC EVENTS
# =============================================================================

const SHOCK_CATALOG = Dict{Int, MalaysiaShockSpec}(
    # 1. Global Financial Crisis (2008-2009)
    1 => MalaysiaShockSpec(
        1, "Global Financial Crisis", "2008-2009",
        EXTERNAL_DEMAND, [EXPORT_CHANNEL, CREDIT_SPREAD, INVESTMENT],
        :chi, -0.15, 0.7,  # Export demand shock, persistent
        "BNM Annual Report 2009, IMF Malaysia Article IV",
        Dict("export_drop" => -15.0, "gdp_impact" => -1.7, "employment" => -2.1)
    ),

    # 2. Eurozone Crisis Spillover (2010-2012)
    2 => MalaysiaShockSpec(
        2, "Eurozone Crisis Spillover", "2010-2012",
        EXTERNAL_DEMAND, [EXPORT_CHANNEL],
        :chi, -0.05, 0.6,
        "DOSM External Trade Statistics",
        Dict("electronics_export" => -8.0, "fdi_delay" => -15.0)
    ),

    # 3. Commodity Price Swings (2011-2014)
    3 => MalaysiaShockSpec(
        3, "Commodity Price Swings", "2011-2014",
        TERMS_OF_TRADE, [FISCAL_BUDGET, CONSUMPTION],
        :ToT, 0.10, 0.5,  # Terms of trade improvement then reversal
        "BNM Quarterly Bulletin, Palm Oil Board",
        Dict("palm_price_volatility" => 30.0, "gov_revenue_swing" => 5.0)
    ),

    # 4. Major Floods (2014-2015)
    4 => MalaysiaShockSpec(
        4, "Major Floods (East Coast)", "2014-2015",
        TFP_SHOCK, [EXPORT_CHANNEL, LABOR_MARKET],
        :TFP, -0.03, 0.3,  # Temporary TFP shock
        "NADMA Disaster Reports, DOSM",
        Dict("agri_output" => -4.0, "damages_RM_billion" => 2.5)
    ),

    # 5. GST Introduction (2015)
    5 => MalaysiaShockSpec(
        5, "GST Introduction", "2015",
        FISCAL_TAX, [CONSUMPTION, FISCAL_BUDGET],
        :tau_c, 0.06, 0.9,  # Consumption tax shock (6% GST)
        "RMCD, MOF Budget Reports",
        Dict("cpi_increase" => 2.1, "real_consumption" => -1.5, "revenue_growth" => 8.0)
    ),

    # 6. US-China Trade War (2018-2019)
    6 => MalaysiaShockSpec(
        6, "US-China Trade War", "2018-2019",
        EXTERNAL_DEMAND, [EXPORT_CHANNEL, UNCERTAINTY, INVESTMENT],
        :chi, -0.05, 0.65,
        "MIT Observatory of Economic Complexity, BNM",
        Dict("ee_exports" => -6.0, "fdi_uncertainty" => 20.0, "relocation_benefit" => 3.0)
    ),

    # 7. COVID Lockdown Q2 2020
    7 => MalaysiaShockSpec(
        7, "COVID Lockdown Q2 2020", "2020-Q2",
        TFP_SHOCK, [LABOR_MARKET, CONSUMPTION, SECTORAL_DEMAND],
        :TFP, -0.08, 0.2,  # Severe but short-lived
        "DOSM GDP Q2 2020, BNM Monetary Policy Statement",
        Dict("gdp_contraction" => -17.1, "employment_loss" => -2.4, "hours_worked" => -25.0)
    ),

    # 8. COVID Income Support - PRIHATIN (2020-2021)
    8 => MalaysiaShockSpec(
        8, "PRIHATIN Income Support", "2020-2021",
        TRANSFER_INCOME, [CONSUMPTION, FISCAL_BUDGET],
        :T_lump, 0.04, 0.8,  # Transfer ~4% of income
        "MOF Economic Stimulus Package Reports",
        Dict("b40_support" => 5.0, "mpc_estimate" => 0.6, "consumption_boost" => 3.0)
    ),

    # 9. Global Supply Chain Crunch (2021-2022)
    9 => MalaysiaShockSpec(
        9, "Global Supply Chain Crunch", "2021-2022",
        TFP_SHOCK, [IMPORT_PRICE, INVESTMENT],
        :zeta, 0.08, 0.7,  # Import cost shock
        "BNM Annual Report 2022, DOSM PPI",
        Dict("import_prices" => 12.0, "semiconductor_shortage_impact" => -5.0)
    ),

    # 10. Russia-Ukraine War (2022)
    10 => MalaysiaShockSpec(
        10, "Russia-Ukraine War", "2022",
        TERMS_OF_TRADE, [IMPORT_PRICE, FISCAL_BUDGET, CONSUMPTION],
        :p_energy, 0.25, 0.6,  # Energy price shock
        "BNM, PETRONAS Annual Report",
        Dict("energy_price" => 45.0, "cpi_food" => 5.2, "subsidy_cost" => 28.0)
    ),

    # 11. Fed Rapid Tightening (2022-2023)
    11 => MalaysiaShockSpec(
        11, "Fed Rapid Tightening", "2022-2023",
        MONETARY_POLICY, [CREDIT_SPREAD, EXCHANGE_RATE_CH, CAPITAL_FLOWS],
        :r_star, 0.035, 0.85,  # World interest rate shock
        "BNM Monetary Policy Statements, IMF",
        Dict("capital_outflows" => -12.0, "opr_defense" => 100.0, "spread_widening" => 50.0)
    ),

    # 12. Ringgit Depreciation (2022-2023)
    12 => MalaysiaShockSpec(
        12, "Ringgit Depreciation Episode", "2022-2023",
        EXCHANGE_RATE, [IMPORT_PRICE, CONSUMPTION],
        :E, -0.10, 0.75,  # 10% depreciation
        "BNM, Reuters FX Data",
        Dict("nominal_depreciation" => -12.0, "import_inflation" => 3.5, "real_income" => -2.0)
    ),

    # 13. Subsidy Rationalisation (2023)
    13 => MalaysiaShockSpec(
        13, "Subsidy Rationalisation", "2023",
        FISCAL_TAX, [CONSUMPTION, FISCAL_BUDGET],
        :tau_c, 0.015, 0.9,  # Effective consumption tax increase
        "MOF Budget 2023, BNM",
        Dict("cpi_energy" => 15.0, "b40_impact" => -3.0, "fiscal_savings" => 12.0)
    ),

    # 14. China Growth Slowdown (2023-2024)
    14 => MalaysiaShockSpec(
        14, "China Growth Slowdown", "2023-2024",
        EXTERNAL_DEMAND, [EXPORT_CHANNEL, COMMODITY_DEMAND],
        :chi, -0.04, 0.8,
        "DOSM Trade Statistics, World Bank China Economic Update",
        Dict("ee_exports_china" => -8.0, "palm_demand" => -5.0, "iron_ore" => -12.0)
    ),

    # 15. Tech-Cycle Rebound (2024)
    15 => MalaysiaShockSpec(
        15, "Tech-Cycle Rebound", "2024",
        EXTERNAL_DEMAND, [EXPORT_CHANNEL, INVESTMENT, LABOR_MARKET],
        :chi, 0.05, 0.7,  # Positive demand shock
        "MIDA, BNM Monthly Manufacturing",
        Dict("ee_export_growth" => 15.0, "fdi_manufacturing" => 25.0, "wage_growth" => 4.5)
    ),

    # 16. OPR Normalization (2024-2025)
    16 => MalaysiaShockSpec(
        16, "OPR Normalization", "2024-2025",
        MONETARY_POLICY, [CREDIT_SPREAD, CONSUMPTION, INVESTMENT],
        :r, 0.005, 0.9,  # 50bp rate increase
        "BNM Monetary Policy Committee Statements",
        Dict("opr_increase" => 50.0, "borrowing_cost" => 0.8, "consumption_dampening" => -0.5)
    ),

    # 17. Klang Valley Floods (2021-2022)
    17 => MalaysiaShockSpec(
        17, "Klang Valley Floods", "2021-2022",
        TFP_SHOCK, [LABOR_MARKET, SECTORAL_DEMAND],
        :TFP, -0.02, 0.25,
        "NADMA, DOSM",
        Dict("services_disruption" => -8.0, "informal_income_loss" => -15.0, "damages" => 6.0)
    ),

    # 18. 1MDB Confidence Overhang (2016-2018)
    18 => MalaysiaShockSpec(
        18, "1MDB Confidence Overhang", "2016-2018",
        RISK_PREMIUM, [CREDIT_SPREAD, INVESTMENT, CAPITAL_FLOWS],
        :zeta_risk, 0.008, 0.7,  # 80bp risk premium increase
        "BNM, IMF Article IV, Bloomberg",
        Dict("sovereign_spread" => 80.0, "investment_decline" => -5.0, "ringgit_pressure" => -8.0)
    ),

    # 19. Tourism Collapse (2020-2022)
    19 => MalaysiaShockSpec(
        19, "Tourism Collapse", "2020-2022",
        SECTORAL_DEMAND, [LABOR_MARKET, CONSUMPTION, SERVICES],
        :chi_services, -0.60, 0.4,  # 60% drop in tourism demand
        "Tourism Malaysia, DOSM Services Index",
        Dict("tourist_arrivals" => -83.0, "services_gdp" => -25.0, "urban_employment" => -8.0)
    ),

    # 20. Gig/Informal Income Volatility (Structural)
    20 => MalaysiaShockSpec(
        20, "Gig Economy Income Volatility", "Structural",
        UNCERTAINTY, [CONSUMPTION, LABOR_MARKET],
        :sigma_y, 0.25, 0.95,  # 25% increase in income volatility
        "DOSM Labour Force Survey, World Bank Malaysia Economic Monitor",
        Dict("informal_share" => 32.0, "precautionary_saving" => 5.0, "mpc_heterogeneity" => 0.4)
    )
)

# =============================================================================
# SHOCK APPLICATION FUNCTIONS
# =============================================================================

function apply_malaysia_shock(engram::JacobianEngram, spec::MalaysiaShockSpec, 
                               T::Int; verbose::Bool=false)
    """
    Apply a Malaysia-specific shock to the HANK model.

    Maps the shock specification to the appropriate model shock type
    and applies it using the Engram Jacobians.
    """
    verbose && println("Applying shock: $(spec.name)")
    verbose && println("  Type: $(spec.shock_type), Magnitude: $(spec.magnitude)")

    # Map shock to model implementation
    if spec.primary_shock in [:r, :r_star]
        # Interest rate shock (monetary policy or external)
        return solve_ge_toeplitz(engram, :r_shock, spec.magnitude, T)

    elseif spec.primary_shock in [:w, :chi, :chi_services]
        # Wage/demand shock (external demand, sectoral)
        # Map to wage shock as proxy for labor demand
        return solve_ge_toeplitz(engram, :w_shock, spec.magnitude, T)

    elseif spec.primary_shock == :TFP
        # TFP shock - use combination of r and w shocks
        # TFP affects both labor productivity and capital productivity
        result_r = solve_ge_toeplitz(engram, :r_shock, spec.magnitude * 0.5, T)
        result_w = solve_ge_toeplitz(engram, :w_shock, spec.magnitude * 0.5, T)
        return combine_results(result_r, result_w, 0.5, 0.5)

    elseif spec.primary_shock in [:tau_c, :T_lump]
        # Fiscal shocks - map to consumption demand shock
        # For now, approximate with wage shock (income effect)
        return solve_ge_toeplitz(engram, :w_shock, -spec.magnitude, T)

    else
        # Default: external demand shock via wage
        return solve_ge_toeplitz(engram, :w_shock, spec.magnitude, T)
    end
end

function combine_results(r1::GEOutput, r2::GEOutput, w1::Float64, w2::Float64)
    """Combine two GE results with weights."""
    GEOutput(
        w1 .* r1.r_path .+ w2 .* r2.r_path,
        w1 .* r1.w_path .+ w2 .* r2.w_path,
        w1 .* r1.Y_path .+ w2 .* r2.Y_path,
        w1 .* r1.C_path .+ w2 .* r2.C_path,
        w1 .* r1.A_path .+ w2 .* r2.A_path,
        w1 .* r1.irf_r .+ w2 .* r2.irf_r,
        w1 .* r1.irf_Y .+ w2 .* r2.irf_Y,
        w1 .* r1.irf_C .+ w2 .* r2.irf_C,
        max(r1.iterations, r2.iterations),
        max(r1.residual, r2.residual),
        r1.solve_time + r2.solve_time,
        r1.engrams_used,
        r1.engram_lookup_time + r2.engram_lookup_time
    )
end

# =============================================================================
# SCENARIO SOLVER
# =============================================================================

function solve_malaysia_scenario(shock_id::Int, T::Int=100;
                                  params::Union{HANKParameters, Nothing}=nothing,
                                  engram_store::Union{EngramStore, Nothing}=nothing,
                                  verbose::Bool=true)
    """
    Solve a specific Malaysia shock scenario.

    Example:
        result = solve_malaysia_scenario(1, 100)  # GFC shock
    """
    spec = SHOCK_CATALOG[shock_id]

    verbose && println("\n" * "="^60)
    verbose && println("MALAYSIA SHOCK SCENARIO #$(shock_id): $(spec.name)")
    verbose && println("Period: $(spec.period)")
    verbose && println("="^60)

    # Use default Malaysia params if not provided
    if params === nothing
        year = parse(Int, split(spec.period, "-")[1])
        params = create_malaysia_params(year=year)
    end

    # Create store if not provided
    if engram_store === nothing
        engram_store = EngramStore("malaysia_engrams.jld2")
    end

    # Retrieve or compute engram
    engram, cache_hit = retrieve_or_compute!(engram_store, params, T, verbose=verbose)

    # Apply shock
    result = apply_malaysia_shock(engram, spec, T, verbose=verbose)

    verbose && println("\nResults:")
    verbose && println("  Peak output response: $(round(maximum(abs.(result.irf_Y))*100, digits=2))%")
    verbose && println("  Peak consumption response: $(round(maximum(abs.(result.irf_C))*100, digits=2))%")
    verbose && println("  Convergence: $(result.iterations) iterations")

    return result, spec
end

# =============================================================================
# FULL SHOCK SUITE RUNNER
# =============================================================================

function run_shock_suite(T::Int=100; 
                         shock_ids::Vector{Int}=collect(1:20),
                         verbose::Bool=true)
    """
    Run the complete Malaysia shock testing suite.

    Returns a dictionary mapping shock_id to (result, spec) pairs.
    """
    verbose && println("\n" * "="^70)
    verbose && println("MALAYSIA 20-YEAR ECONOMIC SHOCK TESTING SUITE")
    verbose && println("Model: EngramHANK (BNM/DOSM/WID Calibration)")
    verbose && println("Time Horizon: $(T) quarters")
    verbose && println("="^70)

    results = Dict{Int, Tuple{GEOutput, MalaysiaShockSpec}}()

    # Shared engram store for efficiency
    engram_store = EngramStore("malaysia_shock_suite.jld2")

    # Track unique parameter sets
    param_cache = Dict{UInt64, HANKParameters}()

    for id in shock_ids
        spec = SHOCK_CATALOG[id]

        # Get or create params for this period
        year = try
            parse(Int, split(spec.period, "-")[1])
        catch
            2024
        end

        params = create_malaysia_params(year=year)
        param_hash = fnv1a_hash_params(params.β, params.σ, params.α, params.σ_ε, params.r_ss)

        if !haskey(param_cache, param_hash)
            param_cache[param_hash] = params
        end

        result, _ = solve_malaysia_scenario(id, T; 
                                            params=params, 
                                            engram_store=engram_store,
                                            verbose=verbose)

        results[id] = (result, spec)
    end

    verbose && println("\n" * "="^70)
    verbose && println("SHOCK SUITE COMPLETE")
    verbose && println("Total unique parameter sets: $(length(param_cache))")
    verbose && println("Total engrams stored: $(length(engram_store.engrams))")
    verbose && println("="^70)

    return results
end

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

function analyze_shock_results(results::Dict{Int, Tuple{GEOutput, MalaysiaShockSpec}})
    """
    Generate comparative analysis of all shock results.
    """
    println("\n" * "="^80)
    println("MALAYSIA SHOCK ANALYSIS SUMMARY")
    println("="^80)
    println()

    # Header
    @printf("%-4s %-25s %-12s %-12s %-12s %-12s\n",
            "ID", "Shock Name", "Peak Y (%)", "Peak C (%)", "Half-life", "Conv. Iter")
    println("-"^80)

    for id in sort(collect(keys(results)))
        result, spec = results[id]

        peak_Y = maximum(abs.(result.irf_Y)) * 100
        peak_C = maximum(abs.(result.irf_C)) * 100
        half_life = compute_half_life(result.irf_Y)

        @printf("%-4d %-25s %+-12.2f %+-12.2f %-12d %-12d\n",
                id, 
                length(spec.name) > 25 ? spec.name[1:22] * "..." : spec.name,
                peak_Y, peak_C, half_life, result.iterations)
    end

    println("="^80)

    # Category summary
    println("\nBy Shock Category:")
    category_impacts = Dict{ShockType, Vector{Float64}}()

    for (id, (result, spec)) in results
        peak_impact = maximum(abs.(result.irf_Y))
        if !haskey(category_impacts, spec.shock_type)
            category_impacts[spec.shock_type] = Float64[]
        end
        push!(category_impacts[spec.shock_type], peak_impact)
    end

    for (cat, impacts) in sort(collect(category_impacts))
        avg_impact = mean(impacts) * 100
        max_impact = maximum(impacts) * 100
        println("  $(cat): Avg=$(round(avg_impact, digits=2))%, Max=$(round(max_impact, digits=2))%")
    end
end

function export_results_to_csv(results::Dict{Int, Tuple{GEOutput, MalaysiaShockSpec}}, 
                                filename::String="malaysia_shock_results.csv")
    """Export shock results to CSV for further analysis."""

    open(filename, "w") do io
        println(io, "shock_id,name,period,type,peak_output,peak_consumption,peak_rate,half_life,iterations,residual")

        for id in sort(collect(keys(results)))
            result, spec = results[id]

            peak_Y = maximum(abs.(result.irf_Y))
            peak_C = maximum(abs.(result.irf_C))
            peak_r = maximum(abs.(result.irf_r))
            half_life = compute_half_life(result.irf_Y)

            println(io, "$id,$(spec.name),$(spec.period),$(spec.shock_type),$peak_Y,$peak_C,$peak_r,$half_life,$(result.iterations),$(result.residual)")
        end
    end

    println("Results exported to: $filename")
end

end # module MalaysiaShocks
