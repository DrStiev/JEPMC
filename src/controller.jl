module controller
	include("uode.jl")
	include("graph.jl")

    # https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/SEIR_exposure/seir_exposure.jl
	# https://www.youtube.com/watch?v=5zaB1B4hOnQ

	# mando in run il modello per un tot numero di step, esempio 21.
	# prendo le informazioni del model.collect e le trasformo in matrice 
	# con Array(df) oppure Matrix(df). 
	# provo a predire la curva di infetti e di happiness senza alcun intervento
	# cerco di minimizzare la curva infetti e massimizzare la curca happiness.
	# in questo modo dovrei avere un array di parametri che mi definisce 
	# quali sono i migliori parametri da utilizzare. li applico e vedo che succedde.

end
