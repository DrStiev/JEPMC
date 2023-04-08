module continuous
    using Agents, Random
    using DrWatson: @dict
    using OrdinaryDiffEq

    include("ode.jl")

    @agent Person ContinuousAgent{2} begin
        mass::Float64
        days_infected::Int
        status::Symbol #:S, :E, :I, :R (:V)
    end

    # valori default esempio
    function model_init(;
        N = 1E3, initial_infected = 1,
		β = 6/14, γ = 1/14, σ = 1/4, ω = 1/240, 
		α = 9E-3, ϵ = 0.0, ξ = 0.0,
        interaction_radius = 0.012,
        dt = 1.0, speed = 0.002,
        space_dimension = (1.0, 1.0),
        spacing = 0.02, steps_per_day = 24,
        seed = 1234,
        )
        
        S = N - initial_infected
        E = 0
        I = initial_infected
        R = 0
        D = 0

        # TODO: trova un modo per usarlo
        prob = ode.get_ODE_problem(
            ode.SEIRD!,
            [S, E, I, R, D], (0.0, Inf), 
            [β, γ, σ ,ω, α, ϵ, ξ]
        )
        integrator = ode.get_ODE_integrator(prob)

        space = ContinuousSpace(space_dimension; spacing=spacing)
        model = ABM(Person, space;
        properties = @dict(
            N, initial_infected,
			β, γ, σ ,ω, α, ϵ, ξ,
			interaction_radius,
			dt, speed, space_dimension,
			spacing, steps_per_day,
            integrator,
        ), rng=Xoshiro(seed))

        # inserisco gli agenti nel modello
        for ind in 1:N
            pos = Tuple(rand(model.rng, 2) .* space_dimension)
            status = ind ≤ N - initial_infected ? :S : :I
            vel = sincos(2π * rand(model.rng)) .* speed
            add_agent!(pos, model, vel, 1.0, 0, status)
        end
        return model
    end

    function model_step!(model)
        for (a1, a2) in interacting_pairs(model, model.interaction_radius, :nearest)
            transmit!(a1, a2, model)
            elastic_collision!(a1, a2, :mass)
        end
    end

    function agent_step!(agent, model)
        move_agent!(agent, model, model.dt)
        update!(agent, model)
        recover_or_die!(agent, model)
    end

    # funzione per il controllo della trasmissione della malattia
    function transmit!(a1, a2, model)
        count(a.status == :I for a in (a1, a2)) ≠ 1 && return
        _, healthy = a1.status == :I ? (a1,a2) : (a2,a1)
        n = model.β * abs(randn(model.rng))
        n ≤ 0 && return
        if healthy.status == :S
            healthy.status = :E
            n -= 1
            n ≤ 0 && return
        end
    end

    # aggiornamento infetti
    function update!(agent, model) 
        # probabilità di vaccinarsi
        if agent.status == :S 
            rand(model.rng) ≤ model.ϵ && (agent.status = :R)
        end
        # periodo di esposizione
        if agent.status == :E
            # sistema immunitario forte
            if rand(model.rng) ≤ model.ξ
                agent.status = :S
                agent.days_infected = 0
                return
            end
            # fine latenza inizio infettività
            if agent.days_infected ≥ (1/model.σ) 
                agent.status = :I
				agent.days_infected = 1
                return
            end
            agent.days_infected += 1
        end
        # periodo infettivo
        agent.status == :I && (agent.days_infected += 1)
        # fine immunità
        if agent.status == :R 
          rand(model.rng) ≤ model.ω && (agent.status = :S)
        end            
    end

    function recover_or_die!(agent, model)
        # fine periodo infettivo
        if agent.days_infected ≥ (1/model.γ)
            # possibilità di morte
            if rand(model.rng) ≤ model.α
                kill_agent!(agent, model)
            else
                # guarigione
                agent.status = :R
                agent.days_infected = 0
            end
        end
    end
end