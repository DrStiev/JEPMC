using Aqua, JEPMC

# Aqua tests
Aqua.test_project_toml_formatting(JEPMC)
Aqua.test_project_extras(JEPMC)
Aqua.test_unbound_args(JEPMC)
Aqua.test_undefined_exports(JEPMC)
Aqua.test_ambiguities(JEPMC)
Aqua.test_piracy(JEPMC)
