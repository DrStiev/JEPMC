module sn_graph
    using Agents, Random, DataFrames
    using SimpleWeightedGraphs: SimpleWeightedDiGraph
    using SparseArrays: findnz

    @agent Person ContinuousAgent{2} begin
        days_infected::Int
        status::Symbol #:S, :E, :I, :R (:V)
    end

    # TODO: add initial exposed?
    function init(;
        N = 1000, initial_infected = 1,
        attractors = 0.15, max_force = 1.7, 
        attr_pos = (50, 50),
        max_connections = 10, R₀ = 1.6, 
        γ = 1/14, σ = 1/4, ω = 1/240, 
		δ = 9E-3, ϵ = 0.0, speed = (0, 0), noise = 0.1,
        # TODO: capire come creare uno spazio sufficientemente grande per contentere tutti gli agenti
        space_dimension = (1000, 1000), steps_per_day = 24,
        spacing = 4.0, interaction_radius = spacing / 1.6,
        seed = 1234,
        )

        @assert length(attractors) == length(attr_pos) == length(max_force) "Length of attractors and attractors position must be the same "

        model = ABM(
            Person,
            ContinuousSpace(space_dimension, spacing = spacing; periodic = false);
            properties = Dict(
                :attractors => attractors, 
                :attr_pos => attr_pos,
                :noise => noise, :N => N,
                :steps_per_day => steps_per_day,
                :max_connections => max_connections,
                :connections => SimpleWeightedDiGraph(N),
                :max_force => max_force,
                :interaction_radius => interaction_radius,
                :β => R₀*γ, :γ => γ, :σ => σ, :ω => ω, 
                :δ => δ, :ϵ => ϵ,
            ),
            rng = Xoshiro(seed)
        )

        for person in 1:N
            # position = model.space.extent .* 0.5 .+ Tuple(rand(model.rng, 2)) .- 0.5
            position = model.space.extent .* Tuple(rand(model.rng, 2))
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
        try
            new_pos = Tuple(get_new_pos(agent, model))
            # println("$(agent.pos) => $(new_pos)")
            move_agent!(agent, new_pos, model)
        catch e
            # println(e)
        end
        update!(agent, model)
        recover_or_die!(agent, model)
    end

    function get_nearest_attractor(agent, model)
        dist(pos1, pos2) = sqrt((pos2[1]-pos1[1])^2 + (pos2[2]-pos1[2])^2)
        mindist = dist(agent.pos, model.attr_pos[1])
        pos = model.attr_pos[1]
        attrforce = model.attractors[1]
        maxf = model.max_force[1]
        for i in 1:length(model.attr_pos)
            appodist = dist(agent.pos, model.attr_pos[i])
            if appodist ≤ mindist
                pos = model.attr_pos[i]
                mindist = appodist
                attrforce = model.attractors[i]
                maxf = model.max_force[i]
            end
        end
        return pos, attrforce, maxf
    end      

    function get_new_pos(agent, model)
        nearestattr, attrforce, maxf = get_nearest_attractor(agent, model)
        # place the attractors in a random spot
        # ats = Tuple([(model.attr_pos[i] .- agent.pos) .* model.attractors[i] for i in 1:length(model.attractors)])
        ats = (nearestattr .- agent.pos) .* attrforce
        # add random noise
        noise = model.noise .* (Tuple(rand(model.rng, 2)) .- rand(model.rng))

        # adhere to the social network
        network = model.connections.weights[agent.id, :]
        tidxs, tweights = findnz(network)
        network_force = (0.0, 0.0)
        for (widx, tidx) in enumerate(tidxs)
            connections = tweights[widx]
            force = (agent.pos .- model[tidx].pos) .* connections
            if connections ≥ 0
                # the further i am, the more i want to stay close
                if distance(force) > maxf #model.max_force
                    force = scale(maxf, force)#model.max_force, force)
                end
            else
                # the further i am, the better
                if distance(force) > maxf #model.max_force
                    force = (0.0, 0.0)
                else
                    L = maxf - distance(force)#model.max_force - distance(force)
                    force = scale(L, force)
                end
            end
            network_force = network_force .+ force
        end
        # add all forces together to assign new position
        # ats_sum = reduce((x,y) -> x .+ y, ats)
        # return agent.pos .+ noise .+ ats_sum .+ network_force
        return agent.pos .+ noise .+ ats .+ network_force
    end

    function transmit!(a1, a2, model)
        count(a.status == :I for a in (a1, a2)) ≠ 1 && return
        _, healthy = a1.status == :I ? (a1,a2) : (a2,a1)
        # possibilità di contagio
        rand(model.rng) > model.β && return
        healthy.status == :S && (healthy.status = :E)
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
            if rand(model.rng) ≤ model.δ
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
        data[!, :dead_status] = data[!, 6]
        select!(data, :susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead_status)
        for i in 1:5
            data[!, i] = data[!, i] / model.N
        end
        return data
    end
end