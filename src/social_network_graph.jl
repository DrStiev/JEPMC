module sn_graph
    using Agents, Random
    using SimpleWeightedGraphs: SimpleWeightedDiGraph
    using SparseArrays: findnz

    @agent Person ContinuousAgent{2} begin
        days_infected::Int
        status::Symbol #:S, :E, :I, :R (:V)
    end

    # FIXME: fix the parameter in order to be similar to the ode
    function init(;
        N = 1000, initial_infected = 1,
        attractors = 0.15, max_force = 1.7,
        max_connections = 10, β = 6/14, 
        γ = 1/14, σ = 1/4, ω = 1/240, 
		α = 9E-3, ϵ = 0.0, ξ = 0.0,
        speed = (0, 0), noise = 0.1,
        # TODO: find the right space dimension for the amount of agents in the model
        # TODO: find good amount of steps per day
        space_dimension = (1000, 1000), steps_per_day = 16,
        spacing = 4.0, interaction_radius = spacing / 1.6,
        seed = 1234,
        )

        model = ABM(
            Person,
            ContinuousSpace(space_dimension, spacing = spacing; periodic = false);
            properties = Dict(
                :attractors => attractors,
                :noise => noise, :N => N,
                :steps_per_day => steps_per_day,
                :max_connections => max_connections,
                :connections => SimpleWeightedDiGraph(N),
                :max_force => max_force,
                :interaction_radius => interaction_radius,
                :β => β, :γ => γ, :σ => σ, :ω => ω, 
                :α => α, :ϵ => ϵ, :ξ => ξ,
            ),
            rng = Xoshiro(seed)
        )

        for person in 1:N
            # gli individui iniziano posti randomicamente
            position = model.space.extent .* 0.5 .+ Tuple(rand(model.rng, 2)) .- 0.5
            status = person ≤ N - initial_infected ? :S : :I
            add_agent!(position, model, speed, 0, status)

            # aggiungo un numero randomico di connessioni
            # friend - foe alla rete sociale
            con = rand(1:max_connections)
            good_con = rand(1:con)
            for c in 1:con
                friend = rand(model.rng, filter(p -> p ≠ person, 1:N))
                # trovo good or bad connections
                ext = c ≤ good_con ? rand(model.rng) : -rand(model.rng)
                add_edge!(model.connections, person, friend, ext)
            end
        end
        return model
    end

    distance(pos) = sqrt(pos[1]^2 + pos[2]^2)
    scale(L, force) = (L / distance(force)) .* force

    function model_step!(model)
        for (a1, a2) in interacting_pairs(model, model.interaction_radius, :nearest)
            transmit!(a1, a2, model)
        end
    end

    function agent_step!(agent, model)
        #FIXME: capire come mai cade fuori dallo space extent
        try
            new_pos = get_new_pos(agent, model)
            move_agent!(agent, new_pos, model)
        catch
            # println("$(agent.pos) => $new_pos")
        end
        
        update!(agent, model)
        recover_or_die!(agent, model)
    end

    function get_new_pos(agent, model)
        # TODO: capire come usare più attrattori
        # TODO: attractors are places and cannot be contaged
        # place the attractors in a random spot 
        ats = (model.space.extent .* 0.5 .- agent.pos) .* model.attractors
        
        # add random noise
        noise = model.noise .* (Tuple(rand(model.rng, 2)) .- 0.5)

        # adhere to the social network
        network = model.connections.weights[agent.id, :]
        tidxs, tweights = findnz(network)
        network_force = (0.0, 0.0)
        for (widx, tidx) in enumerate(tidxs)
            # FIXME: sistemare asap 
            connections = tweights[widx]
            force = (agent.pos .- model[tidx].pos) .* connections
            if connections ≥ 0
                # the further i am, the more i want to stay close
                if distance(force) > model.max_force
                    force = scale(model.max_force, force)
                end
            else
                # the further i am, the better
                if distance(force) > model.max_force
                    force = (0.0, 0.0)
                else
                    L = model.max_force - distance(force)
                    force = scale(L, force)
                end
            end
            network_force = network_force .+ force
        end
        # add all forces together to assign new position
        return agent.pos .+ noise .+ ats .+ network_force
    end

    function transmit!(a1, a2, model)
        count(a.status == :I for a in (a1, a2)) ≠ 1 && return
        _, healthy = a1.status == :I ? (a1,a2) : (a2,a1)
        # n = model.β * abs(randn(model.rng))
        # n ≤ 0 && return
        # possibilità di contagio
        rand(model.rng) > model.β && return
        if healthy.status == :S
            healthy.status = :E
            # n -= 1
            # n ≤ 0 && return
        end
    end

    function update!(agent, model)
        # probabilità di vaccinarsi
        if agent.status == :S 
            rand(model.rng) ≤ model.ϵ && (agent.status = :R)
        end
        # periodo di esposizione
        if agent.status == :E
            # fine latenza inizio infettività
            if agent.days_infected ≥ (1/model.σ*model.steps_per_day)
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
        if agent.days_infected ≥ (1/model.γ*model.steps_per_day)
            # possibilità di morte
            if rand(model.rng) ≤ model.α
                remove_agent!(agent, model)
            else
                # guarigione
                agent.status = :R
                agent.days_infected = 0
            end
        end
    end

    function collect(model, astep, mstep; n = 1000)
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        dead(x) = model.N - length(x)

        to_collect = [(:status, f) for f in (susceptible, exposed, infected, recovered, dead)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
        return data
    end
end