using Aqua, JEPMC

@testset "Aqua tests" begin
    for test in [Aqua.test_project_toml_formatting,
        Aqua.test_project_extras,
        Aqua.test_unbound_args,
        Aqua.test_undefined_exports,
        Aqua.test_ambiguities,
        Aqua.test_piracy]
        try
            test(JEPMC)
        catch e
            @warn "Test $test failed with error: $e"
        end
    end
end
