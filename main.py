import spectroscopy

source = 'original/'
dire = 'measurement'
path_to_graph = 'graphs'
a = 500
b = 650
line = 14

spectroscopy.copy_original(source=source, destination=dire)
spectroscopy.preprocessing(dire=dire, n_line=line)
table = spectroscopy.spectro_all(dire=dire, a=a, b=b, graph_to=path_to_graph)





