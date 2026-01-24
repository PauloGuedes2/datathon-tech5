import os


def export_project_to_txt(project_root, output_file="projeto2.txt"):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n\n===== {file_path} =====\n\n")

                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())


if __name__ == "__main__":
    project_directory = "."  # Diretório atual (ou substitua pelo caminho do projeto)
    export_project_to_txt(project_directory)
    print(f"Todo o código foi exportado para 'projeto2.txt'")
