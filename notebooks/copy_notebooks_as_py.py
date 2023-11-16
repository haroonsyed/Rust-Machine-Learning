from pathlib import Path
import nbconvert
import nbformat

notebooks_path = str(Path().absolute())
output_dir = notebooks_path + "/profiling_py/"

print("Notebooks path:", notebooks_path)
print("Working dir:", output_dir)

notebooks = [
    file
    for file in Path(notebooks_path).iterdir()
    if file.is_file() and file.suffix == ".ipynb"
]

exporter = nbconvert.PythonExporter()

for notebook_file in notebooks:
    notebook_name = notebook_file.stem
    print("Processing notebook at:", notebook_file)

    # Read the notebook and export it to the desired format
    with open(notebook_file, "r", encoding="utf-8") as nb_file:
        notebook_content = nb_file.read()
        notebook_node = nbformat.reads(notebook_content, as_version=4)
        output, _ = exporter.from_notebook_node(notebook_node)

    # Define the output file path for the executable code
    output_file_path = output_dir + "/" + notebook_name + ".py"

    # Write the exported code to the output file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(output)
