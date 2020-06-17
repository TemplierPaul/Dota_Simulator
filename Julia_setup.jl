using Pkg
ENV["PYTHON"]="C:\\ProgramData\\Anaconda3\\python.exe"
ENV["CONDA_JL_HOME"]="C:\\ProgramData\\Anaconda3"

Pkg.build("Conda")
Pkg.build("PyCall")

using PyCall
pyimport("torch")
