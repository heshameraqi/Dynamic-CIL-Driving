import os


def generate_latex_table(table_string_id, data_to_visualize_town1, data_to_visualize_town2, directory,
                         data_to_visualize_town1_success=None, data_to_visualize_town2_success=None):
    latex_file = open(os.path.join(directory, table_string_id), "w")

    # ------------------------------------------------------------------------------------------
    # - Combined table (First + Second tables)
    # ------------------------------------------------------------------------------------------
    if table_string_id == "success_rates_and_avg_dist_table.tex":
        # ----------------- Header -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""\begin{table*}[h] % table* means the table takes the two columns page""",
            r"""	\centering""",
            r"""    % \arrayrulecolor{white}  % For lines color""",
            r"""	\caption{Driving success rate average percentage, average percentage of distance to goal travelled is between parentheses}""",
            r"""	\label{table:success_rates_distance_to_goal_results}""",
            r"""	\setlength{\tabcolsep}{0.1em} % for the horizontal padding""",
            r"""    \renewcommand{\arraystretch}{1.0}% for the vertical padding""",
            r"""	\begin{tabular}{|c|c||c|c|c|c|}""",
            r"""		\hline""",
            r"""		\multirow{3}{*}{Task}								& \multirow{3}{*}{Model}							            & \multicolumn{4}{c|}{\makecell{Percentages of average success rate and distance to goal}}	\\ \cline{3-6}""",
            r"""															&													            & \multicolumn{2}{c|}{Training town}	& \multicolumn{2}{c|}{New town} 					\\ \cline{3-6}""",
            r"""															&													            & Training weathers	    & New weathers	& Training weathers	& New weathers 	                \\ \hline \hline""",
        ])

        # Fix when success rate = 100%, but avg distance is not perfectly 100% rounding problem
        for i in range(4):
            for j in range(8):  # number of models to be tested x 2
                if data_to_visualize_town1_success[j+2][i] == 100:
                    data_to_visualize_town1[j][i] = 100
                if data_to_visualize_town2_success[j+2][i] == 100:
                    data_to_visualize_town2[j][i] = 100

        for (i, task) in zip([0, 1, 2, 3], ["Straight", r"""\makecell{Single\\Turn}""", "Navigation", r"""\makecell{Dynamic\\Navigation}"""]):
            latex_file.writelines(
                r"""\multirow{5}{*}{%s} & Camera only, \cite{dosovitskiy2017carla} results & %d (-) & %d (-) & %d (-) & %d (-) \\ """ % (  # \cline{2-6} at the end if needed, the number inside multirow{} indicates the number of models benchmarked
                    task, data_to_visualize_town1_success[0][i], data_to_visualize_town1_success[1][i],
                    data_to_visualize_town2_success[0][i], data_to_visualize_town2_success[1][i]) + "\n")

            latex_file.writelines(
                r"""& Camera only, \cite{dosovitskiy2017carla} pre-trained & %d (%.2f) & %d (%.2f) & %d (%.2f) & %d (%.2f) \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1_success[2][i], data_to_visualize_town1[0][i], data_to_visualize_town1_success[3][i], data_to_visualize_town1[1][i],
                    data_to_visualize_town2_success[2][i], data_to_visualize_town2[0][i], data_to_visualize_town2_success[3][i], data_to_visualize_town2[1][i]) + "\n")

            latex_file.writelines(
                r"""& Camera (trained on our data) & %d (%.2f) & %d (%.2f) & %d (%.2f) & %d (%.2f) \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1_success[4][i], data_to_visualize_town1[2][i], data_to_visualize_town1_success[5][i], data_to_visualize_town1[3][i],
                    data_to_visualize_town2_success[4][i], data_to_visualize_town2[2][i], data_to_visualize_town2_success[5][i], data_to_visualize_town2[3][i]) + "\n")

            latex_file.writelines(
                r"""& Camera + LiDAR & %d (%.2f) & %d (%.2f) & %d (%.2f) & %d (%.2f) \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1_success[6][i], data_to_visualize_town1[4][i], data_to_visualize_town1_success[7][i], data_to_visualize_town1[5][i],
                    data_to_visualize_town2_success[6][i], data_to_visualize_town2[4][i], data_to_visualize_town2_success[7][i], data_to_visualize_town2[5][i]) + "\n")
                    
            latex_file.writelines(
                r"""& \textbf{Camera + LiDAR (Our Route Planner)} & %d (%.2f) & %d (%.2f) & %d (%.2f) & %d (%.2f) \\ \hline \hline""" % (
                    data_to_visualize_town1_success[8][i], data_to_visualize_town1[6][i], data_to_visualize_town1_success[9][i], data_to_visualize_town1[7][i],
                    data_to_visualize_town2_success[8][i], data_to_visualize_town2[6][i], data_to_visualize_town2_success[9][i], data_to_visualize_town2[7][i]) + "\n")

        # ----------------- Footer -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""	\end{tabular}""",
            r"""\end{table*}"""
        ])

    # ------------------------------------------------------------------------------------------
    # - First table
    # ------------------------------------------------------------------------------------------
    elif table_string_id == "success_rates_table.tex":
        # ----------------- Header -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""\begin{table*}[h] % table* means the table takes the two columns page""",
            r"""	\centering""",
            r"""    % \arrayrulecolor{white}  % For lines color""",
            r"""	\caption{Autonomous Driving  success rate average percentage of our models benchmarked with state-of-the-art on different tasks and test conditions}""",
            r"""	\label{table:success_rates_in_different_tasks_results}""",
            r"""	\setlength{\tabcolsep}{0.2em} % for the horizontal padding""",
            r"""    \renewcommand{\arraystretch}{1.2}% for the vertical padding""",
            r"""	\begin{tabular}{|c|c||c|c|c|c|}""",
            r"""		\hline""",
            r"""		\multirow{3}{*}{Task}								& \multirow{3}{*}{Model}							            & \multicolumn{4}{c|}{\makecell{Average success rate percentage\\on different test conditions}}					\\ \cline{3-6}""",
            r"""															&													            & \multicolumn{2}{c|}{Training town}	& \multicolumn{2}{c|}{New town} 						\\ \cline{3-6}""",
            r"""															&													            & \makecell{Training\\weathers}	    & \makecell{New\\weathers}		& \makecell{Training\\weathers}	& \makecell{New\\weathers} 	\\ \hline \hline""",
            ])

        for (i, task) in zip([0, 1, 2, 3], ["Straight", r"""\makecell{Single\\Turn}""", "Navigation", r"""\makecell{Dynamic\\Navigation}"""]):
            latex_file.writelines(
                r"""\multirow{4}{*}{%s} & Camera only, \cite{dosovitskiy2017carla} results & %d & %d & %d & %d \\ """ % (  # \cline{2-6} at the end if needed
                    task, data_to_visualize_town1[0][i], data_to_visualize_town1[1][i],
                    data_to_visualize_town2[0][i], data_to_visualize_town2[1][i]) + "\n")

            latex_file.writelines(
                r"""& Camera only, \cite{dosovitskiy2017carla} pre-trained & %d & %d & %d & %d \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[2][i], data_to_visualize_town1[3][i],
                    data_to_visualize_town2[2][i], data_to_visualize_town2[3][i]) + "\n")

            latex_file.writelines(
                r"""& Camera only (trained on our data) & %d & %d & %d & %d \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[4][i], data_to_visualize_town1[5][i],
                    data_to_visualize_town2[4][i], data_to_visualize_town2[5][i]) + "\n")

            latex_file.writelines(
                r"""& Camera + LiDAR & %d & %d & %d & %d \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[6][i], data_to_visualize_town1[7][i],
                    data_to_visualize_town2[6][i], data_to_visualize_town2[7][i]) + "\n")
                    
            latex_file.writelines(
                r"""& \textbf{Camera + LiDAR (Our Route Planner)} &  %d & %d & %d & %d \\ \hline \hline""" % (
                    data_to_visualize_town1[8][i], data_to_visualize_town1[9][i],
                    data_to_visualize_town2[8][i], data_to_visualize_town2[9][i]) + "\n")

        # ----------------- Footer -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""	\end{tabular}""",
            r"""\end{table*}"""
            ])

    # ------------------------------------------------------------------------------------------
    # - Second table
    # ------------------------------------------------------------------------------------------
    elif table_string_id == "average_distance_percentage_table.tex":
        # ----------------- Header -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""\begin{table*}[h] % table* means the table takes the two columns page""",
            r"""	\centering""",
            r"""    % \arrayrulecolor{white}  % For lines color""",
            r"""	\caption{Average percentage of distance to goal travelled of our models benchmarked with state-of-the-art on different tasks and test conditions}""",
            r"""	\label{table:average_distance_percentage_in_different_tasks_results}""",
            r"""	\setlength{\tabcolsep}{0.2em} % for the horizontal padding""",
            r"""    \renewcommand{\arraystretch}{1.2}% for the vertical padding""",
            r"""	\begin{tabular}{|c|c||c|c|c|c|}""",
            r"""		\hline""",
            r"""		\multirow{3}{*}{Task}								& \multirow{3}{*}{Model}							            & \multicolumn{4}{c|}{\makecell{Average percentage of distance to goal\\travelled on different test conditions}}					\\ \cline{3-6}""",
            r"""															&													            & \multicolumn{2}{c|}{Training town}	& \multicolumn{2}{c|}{New town} 						\\ \cline{3-6}""",
            r"""															&													            & \makecell{Training\\weathers}	    & \makecell{New\\weathers}		& \makecell{Training\\weathers}	& \makecell{New\\weathers} 	\\ \hline \hline""",
        ])

        for (i, task) in zip([0, 1, 2, 3], ["Straight", r"""\makecell{Single\\Turn}""", "Navigation", r"""\makecell{Dynamic\\Navigation}"""]):
            latex_file.writelines(
                r"""\multirow{3}{*}{%s} & Camera only, \cite{dosovitskiy2017carla} pre-trained & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    task, data_to_visualize_town1[0][i], data_to_visualize_town1[1][i],
                    data_to_visualize_town2[0][i], data_to_visualize_town2[1][i]) + "\n")

            latex_file.writelines(
                r"""& Camera only (trained on our data) & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[2][i], data_to_visualize_town1[3][i],
                    data_to_visualize_town2[2][i], data_to_visualize_town2[3][i]) + "\n")

            latex_file.writelines(
                r"""& Camera + LiDAR & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[4][i], data_to_visualize_town1[5][i],
                    data_to_visualize_town2[4][i], data_to_visualize_town2[5][i]) + "\n")
                    
            latex_file.writelines(
                r"""& \textbf{Camera + LiDAR (Our Route Planner)} & %.2f & %.2f & %.2f & %.2f \\ \hline \hline""" % (
                    data_to_visualize_town1[6][i], data_to_visualize_town1[7][i],
                    data_to_visualize_town2[6][i], data_to_visualize_town2[7][i]) + "\n")

        # ----------------- Footer -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""	\end{tabular}""",
            r"""\end{table*}"""
        ])

    # ------------------------------------------------------------------------------------------
    # - Third table
    # ------------------------------------------------------------------------------------------
    elif table_string_id == "collisions_table.tex":
        # ----------------- Header -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""\begin{table*}[h] % table* means the table takes the two columns page""",
            r"""	\def\arraystretch{1.5}""",
            r"""	\centering""",
            r"""    % \arrayrulecolor{white}  % For lines color""",
            r"""	\caption{The average number of kilometers traveled before an infraction}""",
            r"""	\label{table:collisions_in_different_tasks_results}""",
            r"""	\setlength{\tabcolsep}{0.1em} % for the horizontal padding""",
            r"""    \renewcommand{\arraystretch}{1.0}% for the vertical padding""",
            r"""	\begin{tabular}{|c|c||c|c|c|c|}""",
            r"""	\hline""",
            r"""	\multirow{3}{*}{Infractions}         	& \multirow{3}{*}{Model}	& \multicolumn{4}{c|}{Average kilometers traveled before an infraction}					\\ \cline{3-6}""",
            r"""											&							& \multicolumn{2}{c|}{Same training town}	& \multicolumn{2}{c|}{New town} 						\\ \cline{3-6}""",
            r"""											&							& Training weather	& New weathers		& Training weather	& New weathers 	\\ \hline \hline"""
        ])

        for (i, infraction) in zip([0, 1, 2, 3, 4], [r"""\makecell{Collision\\to a\\Pedestrian}""", r"""\makecell{Collision\\to a\\Vehicle}""",
                                                     r"""\makecell{Collision\\to a\\Static Object}""", r"""\makecell{Going\\Outside\\of Road}""",
                                                     r"""\makecell{Invading the\\Opposite Lane}"""]):
            latex_file.writelines([r"""\multirow{4}{*}{%s} """ % (infraction)])  # the number inside multirow indicates the number of models compared 

            '''latex_file.writelines(
                r"""& Camera only, \cite{dosovitskiy2017carla} results & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[0][i], data_to_visualize_town1[1][i],
                    data_to_visualize_town2[0][i], data_to_visualize_town2[1][i]) + "\n")'''

            latex_file.writelines(
                r"""& Camera only, \cite{dosovitskiy2017carla} pre-trained & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[2][i], data_to_visualize_town1[3][i],
                    data_to_visualize_town2[2][i], data_to_visualize_town2[3][i]) + "\n")

            latex_file.writelines(
                r"""& Camera only (trained on our data) & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[4][i], data_to_visualize_town1[5][i],
                    data_to_visualize_town2[4][i], data_to_visualize_town2[5][i]) + "\n")

            latex_file.writelines(
                r"""& Camera + LiDAR & %.2f & %.2f & %.2f & %.2f \\ """ % (  # \cline{2-6} at the end if needed
                    data_to_visualize_town1[6][i], data_to_visualize_town1[7][i],
                    data_to_visualize_town2[6][i], data_to_visualize_town2[7][i]) + "\n")
                    
            latex_file.writelines(
                r"""& \textbf{Camera + LiDAR (Our Route Planner)} & %.2f & %.2f & %.2f & %.2f \\ \hline \hline""" % (
                    data_to_visualize_town1[8][i], data_to_visualize_town1[9][i],
                    data_to_visualize_town2[8][i], data_to_visualize_town2[9][i]) + "\n")

        # ----------------- Footer -----------------
        latex_file.writelines(line + "\n" for line in [
            r"""	\end{tabular}""",
            r"""\end{table*}"""
        ])

    latex_file.close()