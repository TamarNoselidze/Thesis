import argparse
import csv

model_order = ["VB/16", "VB/32", "VL/16", "Sw-B", "RN50", "RN152", "VGG"]


def to_detailed(csv_file):
    table_data = {}

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["Target class"]
            itr = int(row["Itr #"])
            if class_name not in table_data:
                table_data[class_name] = {}
            table_data[class_name][itr] = [row[m] for m in model_order]

    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table}[h!]")
    latex_lines.append("\\begin{tabular}{|c|c|" + "c" * len(model_order) + "|}")
    latex_lines.append("\\hline")
    latex_lines.append("TC & It. \# & " + " & ".join(model_order) + " \\\\ \\hline")

    for class_name in table_data:
        for i in range(1, 6):
            row = table_data[class_name].get(i, ["-"] * len(model_order))
            row = [f'{entry}\%' for entry in row]
            if i == 1:
                latex_lines.append(f"\\multirow{{5}}{{*}}{{\\begin{{sideways}}{class_name}\\end{{sideways}}}} & {i} & " + " & ".join(row) + " \\\\")
            else:
                latex_lines.append(f" & {i} & " + " & ".join(row) + " \\\\")
        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return latex_lines



def to_averaged(csv_file, caption="", label="tab:table"):
    
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data = rows[1:]

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("\\begin{tabular}{|c|" + "c" * (len(header) - 1) + "|}")
    latex.append("\\hline")
    latex.append(" & ".join(header) + " \\\\ \\hline")

    for row in data:
        escaped_row = [row[0]]
        escaped_row.extend([f'{entry}\%' for entry in row[1:]])
        latex.append(" & ".join(escaped_row) + " \\\\ \\hline")

    latex.append("\\end{tabular}")
    if caption:
        latex.append(f"\\caption{{{caption}}}")
    if label:
        latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table}")

    return latex




if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--csv_file')
    parser.add_argument('--output_file', help='Path to the output tex file where generators will be saved.', default='wandb_asr_table.tex')
    parser.add_argument('--averaged', default=False)

    args = parser.parse_args()

    
    if args.averaged:
        latex_lines = to_averaged(args.csv_file)
    else:
        print('here')
        latex_lines = to_detailed(args.csv_file)


    with open(args.output_file, "w") as f:
        f.write("\n".join(latex_lines))