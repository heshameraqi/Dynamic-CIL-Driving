import sys
import os
from plot_nn.pycore.blocks import *
from plot_nn.pycore.tikzeng import *
import subprocess

shift = 3

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 8 Convolutions of Right Camera Stream & Input image
    to_input(name="input img 3", pathfile='./imgs/Right.png', width=300, height=10, scale=3, offset="-5", to="(0,0,-60)", caption="Right Camera Image 200X88"),
    to_Conv(name="conv 16", width_label="32", height_label="42", depth_label="98", offset="(0,0,-60)", to="(0,0,0)",
            width=2, height=42, depth=98, caption="5X5-2 ReLU"),
    to_Conv(name="conv 17", width_label="32", height_label="40", depth_label="96", offset="(2,0,-60)", to="(0,0,0)",
            width=2, height=40, depth=96, caption="3X3-1 ReLU"),
    to_Conv(name="conv 18", width_label="64", height_label="19", depth_label="47", offset="(5,0,-60)", to="(0,0,0)",
            width=4, height=19, depth=47, caption="3X3-2 ReLU"),
    to_Conv(name="conv 19", width_label="64", height_label="17", depth_label="45", offset="(9,0,-60)", to="(0,0,0)",
            width=4, height=17, depth=45, caption="3X3-1 ReLU"),
    to_Conv(name="conv 20", width_label="128", height_label="8", depth_label="22", offset="(12,0,-60)", to="(0,0,0)",
            width=8, height=8, depth=22, caption="3X3-2 ReLU"),
    to_Conv(name="conv 21", width_label="128", height_label="6", depth_label="20", offset="(15,0,-60)", to="(0,0,0)",
            width=8, height=6, depth=20, caption="3X3-1 ReLU"),
    to_Conv(name="conv 22", width_label="256", height_label="4", depth_label="18", offset="(18,0,-60)", to="(0,0,0)",
            width=16, height=4, depth=18, caption="3X3-1 ReLU"),
    to_Conv(name="conv 23", width_label="256", height_label="2", depth_label="16", offset="(23,0,-60)", to="(0,0,0)",
            width=16, height=2, depth=16, caption="3X3-1 ReLU"),

    # 8 Convolutions of Centre Camera Stream & Input image
    to_input(name="input img 2", pathfile='./imgs/Centre.png', width=300, height=10, scale=3, offset="-5", to="(0,0,-30)", caption="Centre Camera Image 200X88"),
    to_Conv(name="conv 8", width_label="32", height_label="42", depth_label="98", offset="(0,0,-30)", to="(0,0,0)",
            width=2, height=42, depth=98, caption="5X5-2 ReLU"),
    to_Conv(name="conv 9", width_label="32", height_label="40", depth_label="96", offset="(2,0,-30)", to="(0,0,0)",
            width=2, height=40, depth=96, caption="3X3-1 ReLU"),
    to_Conv(name="conv 10", width_label="64", height_label="19", depth_label="47", offset="(5,0,-30)", to="(0,0,0)",
            width=4, height=19, depth=47, caption="3X3-2 ReLU"),
    to_Conv(name="conv 11", width_label="64", height_label="17", depth_label="45", offset="(9,0,-30)", to="(0,0,0)",
            width=4, height=17, depth=45, caption="3X3-1 ReLU"),
    to_Conv(name="conv 12", width_label="128", height_label="8", depth_label="22", offset="(12,0,-30)", to="(0,0,0)",
            width=8, height=8, depth=22, caption="3X3-2 ReLU"),
    to_Conv(name="conv 13", width_label="128", height_label="6", depth_label="20", offset="(15,0,-30)", to="(0,0,0)",
            width=8, height=6, depth=20, caption="3X3-1 ReLU"),
    to_Conv(name="conv 14", width_label="256", height_label="4", depth_label="18", offset="(18,0,-30)", to="(0,0,0)",
            width=16, height=4, depth=18, caption="3X3-1 ReLU"),
    to_Conv(name="conv 15", width_label="256", height_label="2", depth_label="16", offset="(23,0,-30)", to="(0,0,0)",
            width=16, height=2, depth=16, caption="3X3-1 ReLU"),

    # 8 Convolutions of Left Camera Stream & Input image
    to_input(name="input img 1", pathfile='./imgs/Left.png', width=300, height=10, scale=3, offset="-5", to="(0,0,0)", caption="Left Camera Image 200X88"),
    to_Conv(name="conv 0", width_label="32", height_label="42", depth_label="98", offset="(0,0,0)", to="(0,0,0)",
            width=2, height=42, depth=98, caption="5X5-2 ReLU"),
    to_Conv(name="conv 1", width_label="32", height_label="40", depth_label="96", offset="(2,0,0)", to="(0,0,0)",
            width=2, height=40, depth=96, caption="3X3-1 ReLU"),
    to_Conv(name="conv 2", width_label="64", height_label="19", depth_label="47", offset="(5,0,0)", to="(0,0,0)",
            width=4, height=19, depth=47, caption="3X3-2 ReLU"),
    to_Conv(name="conv 3", width_label="64", height_label="17", depth_label="45", offset="(9,0,0)", to="(0,0,0)",
            width=4, height=17, depth=45, caption="3X3-1 ReLU"),
    to_Conv(name="conv 4", width_label="128", height_label="8", depth_label="22", offset="(12,0,0)", to="(0,0,0)",
            width=8, height=8, depth=22, caption="3X3-2 ReLU"),
    to_Conv(name="conv 5", width_label="128", height_label="6", depth_label="20", offset="(15,0,0)", to="(0,0,0)",
            width=8, height=6, depth=20, caption="3X3-1 ReLU"),
    to_Conv(name="conv 6", width_label="256", height_label="4", depth_label="18", offset="(18,0,0)", to="(0,0,0)",
            width=16, height=4, depth=18, caption="3X3-1 ReLU"),
    to_Conv(name="conv 7", width_label="256", height_label="2", depth_label="16", offset="(23,0,0)", to="(0,0,0)",
            width=16, height=2, depth=16, caption="3X3-1 ReLU"),

    # 8 Convolutions of PGM Stream & Input image
    to_input(name="input img 4", pathfile='./imgs/PGM.jpg', width=300, height=10, scale=10, offset="-5", to="(0,0,30)", caption="LiDAR PGM 200X88"),
    to_Conv(name="conv 24", width_label="32", height_label="42", depth_label="98", offset="(0,0,30)", to="(0,0,0)",
            width=2, height=42, depth=98, caption="5X5-2 ReLU"),
    to_Conv(name="conv 25", width_label="32", height_label="40", depth_label="96", offset="(2,0,30)", to="(0,0,0)",
            width=2, height=40, depth=96, caption="3X3-1 ReLU"),
    to_Conv(name="conv 26", width_label="64", height_label="19", depth_label="47", offset="(5,0,30)", to="(0,0,0)",
            width=4, height=19, depth=47, caption="3X3-2 ReLU"),
    to_Conv(name="conv 27", width_label="64", height_label="17", depth_label="45", offset="(9,0,30)", to="(0,0,0)",
            width=4, height=17, depth=45, caption="3X3-1 ReLU"),
    to_Conv(name="conv 28", width_label="128", height_label="8", depth_label="22", offset="(12,0,30)", to="(0,0,0)",
            width=8, height=8, depth=22, caption="3X3-2 ReLU"),
    to_Conv(name="conv 29", width_label="128", height_label="6", depth_label="20", offset="(15,0,30)", to="(0,0,0)",
            width=8, height=6, depth=20, caption="3X3-1 ReLU"),
    to_Conv(name="conv 30", width_label="256", height_label="4", depth_label="18", offset="(18,0,30)", to="(0,0,0)",
            width=16, height=4, depth=18, caption="3X3-1 ReLU"),
    to_Conv(name="conv 31", width_label="256", height_label="2", depth_label="16", offset="(23,0,30)", to="(0,0,0)",
            width=16, height=2, depth=16, caption="3X3-1 ReLU"),

    # Images connections to first conv layers
    r"""\draw [connection]  (input img 1.center) -- node {\midarrow} (conv 0-west);""",
    r"""\draw [connection]  (input img 2.center) -- node {\midarrow} (conv 8-west);""",
    r"""\draw [connection]  (input img 3.center) -- node {\midarrow} (conv 16-west);""",
    r"""\draw [connection]  (input img 4.center) -- node {\midarrow} (conv 24-west);""",

    # First Concatenation node and connections to it
    r"""\node[shade, shading=ball, circle, ball color={rgb:magenta,5;black,7}, minimum size=1cm] (Concat 1) at (""" + str(23+2*1.6+shift) + """, 0, 0) {};""",
    r"""\node[color=white] at (""" + str(23+2*1.6+shift) + r""",0,0){\huge \bf +};""",

    r"""\draw [connection] (conv 7-east) -- node {\midarrow} (Concat 1);""",

    r"""\draw [connection] (conv 23-east) -- node {\midarrow} ([xshift="""+str(shift)+"""cm]conv 23-east);""",
    r"""\draw [connection] ([xshift="""+str(shift)+"""cm]conv 23-east) -- node {\midarrow} (Concat 1);""",

    r"""\draw [connection] (conv 15-east) -- node {\midarrow} ([xshift="""+str(shift)+"""cm]conv 15-east);""",
    r"""\draw [connection] ([xshift="""+str(shift)+"""cm]conv 15-east) -- node {\midarrow} (Concat 1);""",

    r"""\draw [connection] (conv 31-east) -- node {\midarrow} ([xshift="""+str(shift)+"""cm]conv 31-east);""",
    r"""\draw [connection] ([xshift="""+str(shift)+"""cm]conv 31-east) -- node {\midarrow} (Concat 1);""",

    # Fully connected layers
    to_FC(name="FC 1", width_label="512", offset="(38,0,0)", to="(0,0,0)",
            width=1, height=50, depth=2, caption="FC 1"),
    to_FC(name="FC 2", width_label="512", offset="(40,0,0)", to="(0,0,0)",
            width=1, height=50, depth=2, caption="FC 2"),
    r"""\draw [connection] (Concat 1) -- node {\midarrow} (FC 1-west);""",
    to_connection("FC 1", "FC 2"),

    # Speed measurement input
    r"""\node[draw, canvas is yz plane at x=-5, rotate=-90, anchor=center] (speed word) at (0,0,60) {\scalebox{3}{\fontsize{100}{120}\sffamily\bfseries Speed}};""",
    to_FC(name="FC 3", width_label="128", offset="(20,0,60)", to="(0,0,0)",
            width=1, height=20, depth=2, caption="FC 3"),
    to_FC(name="FC 4", width_label="128", offset="(25,0,60)", to="(0,0,0)",
            width=1, height=20, depth=2, caption="FC 4"),
    to_connection("FC 3", "FC 4"),
    r"""\draw [connection]  (speed word.center)    -- node {\midarrow} (FC 3-west);""",

    # Fully connected layer before control
    to_FC(name="FC 5", width_label="512", offset="(44,0,0)", to="(0,-5,0)",
            width=1, height=50, depth=2, caption="FC 5"),

    # Second Concatenation node
    r"""\node[shade, shading=ball, circle, ball color={rgb:magenta,5;black,7}, minimum size=1cm] (Concat 2) at (42, -5, 0) {};""",
    r"""\node[color=white] at (42,-5,0){\huge \bf +};""",
    r"""\draw [connection] (Concat 2) -- node {\midarrow} (FC 5-west);""",
    r"""\draw [connection] (FC 2-east) -- node {\midarrow} (Concat 2);""",
    r"""\draw [connection] (FC 4-east) -- node {\midarrow} ([xshift="""+str(shift+10)+"""cm]FC 4-east);""",
    r"""\draw [connection] ([xshift="""+str(shift+10)+"""cm]FC 4-east) -- node {\midarrow} (Concat 2);""",

    # Control node
    r"""\node[shade, shading=ball, circle, ball color={rgb:blue,2;green,1;black,0.3}, minimum size=2cm] (Control node) at (46, -5, 0) {};""",
    r"""\node[color=white] at (46,-5,0){\huge \bf C};""",
    r"""\draw [connection] (FC 5-east) -- node {\midarrow} (Control node);""",

    # Speed Branch
    to_FC(name="FC 18", width_label="256", offset="(49,0,0)", to="(0,14,0)",
            width=1, height=25, depth=2, caption="FC 18"),
    to_FC(name="FC 19", width_label="256", offset="(50,0,0)", to="(0,14,0)",
            width=1, height=25, depth=2, caption="FC 19"),
    to_FC(name="FC 20", width_label="1", offset="(51,0,0)", to="(0,14,0)",
            width=1, height=2, depth=2, caption="FC 20"),
    to_connection("FC 2", "FC 18"),
    to_connection("FC 18", "FC 19"),
    to_connection("FC 19", "FC 20"),
    r"""\node[shift={(52,0,0)}, anchor=west] at (0,14,0){\huge \bf Speed};""",

    # Branch 1
    to_FC(name="FC 6", width_label="256", offset="(49,0,0)", to="(0,7,0)",
            width=1, height=25, depth=2, caption="FC 6"),
    to_FC(name="FC 7", width_label="256", offset="(50,0,0)", to="(0,7,0)",
            width=1, height=25, depth=2, caption="FC 7"),
    to_FC(name="FC 8", width_label="3", offset="(51,0,0)", to="(0,7,0)",
            width=1, height=5, depth=2, caption="FC 8"),
    r"""\draw [connection] (Control node) -- node {\midarrow} (FC 6-west);""",
    to_connection("FC 6", "FC 7"),
    to_connection("FC 7", "FC 8"),
    r"""\node[shift={(52,0,0)}, anchor=west] at (0,7,0){\huge \bf Follow-lane Commands};""",

    # Branch 2
    to_FC(name="FC 9", width_label="256", offset="(49,0,0)", to="(0,0,0)",
            width=1, height=25, depth=2, caption="FC 9"),
    to_FC(name="FC 10", width_label="256", offset="(50,0,0)", to="(0,0,0)",
            width=1, height=25, depth=2, caption="FC 10"),
    to_FC(name="FC 11", width_label="3", offset="(51,0,0)", to="(0,0,0)",
            width=1, height=5, depth=2, caption="FC 11"),
    r"""\draw [connection] (Control node) -- node {\midarrow} (FC 9-west);""",
    to_connection("FC 9", "FC 10"),
    to_connection("FC 10", "FC 11"),
    r"""\node[shift={(52,0,0)}, anchor=west] at (0,0,0){\huge \bf Go Right Commands};""",

    # Branch 3
    to_FC(name="FC 12", width_label="256", offset="(49,0,0)", to="(0,-7,0)",
            width=1, height=25, depth=2, caption="FC 12"),
    to_FC(name="FC 13", width_label="256", offset="(50,0,0)", to="(0,-7,0)",
            width=1, height=25, depth=2, caption="FC 13"),
    to_FC(name="FC 14", width_label="3", offset="(51,0,0)", to="(0,-7,0)",
            width=1, height=5, depth=2, caption="FC 14"),
    r"""\draw [connection] (Control node) -- node {\midarrow} (FC 12-west);""",
    to_connection("FC 12", "FC 13"),
    to_connection("FC 13", "FC 14"),
    r"""\node[shift={(52,0,0)}, anchor=west] at (0,-7,0){\huge \bf Go Left Commands};""",

    # Branch 4
    to_FC(name="FC 15", width_label="256", offset="(49,0,0)", to="(0,-14,0)",
            width=1, height=25, depth=2, caption="FC 15"),
    to_FC(name="FC 16", width_label="256", offset="(50,0,0)", to="(0,-14,0)",
            width=1, height=25, depth=2, caption="FC 16"),
    to_FC(name="FC 17", width_label="3", offset="(51,0,0)", to="(0,-14,0)",
            width=1, height=5, depth=2, caption="FC 17"),
    r"""\draw [connection] (Control node) -- node {\midarrow} (FC 15-west);""",  # dashed or connection
    to_connection("FC 15", "FC 16"),
    to_connection("FC 16", "FC 17"),
    r"""\node[shift={(52,0,0)}, anchor=west] at (0,-14,0){\huge \bf Go Straight Commands};""",

    # end
    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'temp.tex')

    with open("temp.tex", "rt") as fin:
        with open("network.tex", "wt") as fout:
            for line in fin:
                fout.write(line.replace('../layers/', './plot_nn/layers/'))

    bashCommand = "pdflatex -output-directory " + os.getcwd() + " -interaction batchmode " + os.getcwd() + "/network.tex"
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(bashCommand, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    # p.kill() #TODO process should be closed?

