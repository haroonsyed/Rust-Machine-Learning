from pathlib import Path
import nbconvert
import nbformat
import subprocess
import shutil

working_dir = str(Path().absolute())
notebooks_path = working_dir + "/../notebooks"

notebook_to_convert = input("What notebook would you like to convert? ").removesuffix(
    ".ipynb"
)

# Copy all files in notebooks_path to working_dir (overwrite)
for file in Path(notebooks_path).iterdir():
    if file.is_file():
        shutil.copy(file, working_dir)

# Copy the entire 'data' folder one level above working_dir
data_folder_source = Path(working_dir).parent / "data"
data_folder_destination = Path(working_dir) / "data"
shutil.copytree(data_folder_source, data_folder_destination)

notebooks = [
    file
    for file in Path(working_dir).iterdir()
    if file.is_file() and file.suffix == ".ipynb"
]

exporter = nbconvert.PythonExporter()

# Determine the path to syed_ml_lib module in the current directory
syed_ml_lib_path = str(Path().absolute() / "syed_ml_lib.py")

for notebook_file in notebooks:
    notebook_name = notebook_file.stem
    print("Processing notebook at:", notebook_file)

    # Read the notebook and export it to the desired format
    with open(notebook_file, "r", encoding="utf-8") as nb_file:
        notebook_content = nb_file.read()
        notebook_node = nbformat.reads(notebook_content, as_version=4)
        output, _ = exporter.from_notebook_node(notebook_node)

    # Define the output file path for the executable code
    output_file_path = str(Path().absolute()) + "/" + notebook_name + ".py"

    # Write the exported code to the output file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(output)

    if notebook_name == notebook_to_convert:
        pyinstaller_command = [
            "pyinstaller",
            "--onefile",
            "--clean",
            "--hidden-import",
            "syed_ml_lib",
            "--paths",
            working_dir,  # Use the determined path
            output_file_path,
        ]
        subprocess.run(pyinstaller_command)
