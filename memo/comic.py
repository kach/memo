from .core import *
import shutil, os

def frame_name(frame):
    return f'cluster_{frame.name}_{id(frame)}'

def node_name(frame, who, id_):
    return f'{who}_{id_}_{id(frame)}'

def comic_frame_edges(frame: Frame, io: StringIO) -> None:
    for c in frame.children.keys():
        comic_frame_edges(frame.children[c], io)

    if frame.parent is not None:
        for too, frm in frame.conditions.items():
            print(f'{node_name(frame.parent, *frm)} -> {node_name(frame, *too)}[style="dashed"];', file=io)

def comic_frame_nodes(frame: Frame, io: StringIO) -> None:
    for c in frame.children.values():
        comic_frame_nodes(c, io)

    print(f'subgraph {frame_name(frame)} {{', file=io)
    print(f'label="{frame.name}\'s frame";', file=io)
    print(f'labelloc="b";', file=io)
    print(f'''{frame_name(frame)}_dummy[style=invis];''', file=io)

    for c in frame.children.values():
        print(f'''{frame_name(frame)}_dummy -> {frame_name(c)}_dummy[ltail={frame_name(frame)}, lhead={frame_name(c)}, arrowhead="tee"];''', file=io)

    for (who, id), ch in frame.choices.items():
        color = "lightblue" if ch.known else "orange"
        label = f'{who}.{id}' if who != 'self' else f'{id}'
        print(f'''{node_name(frame, who, id)}[label="{label} : {ch.domain}", color={color}];''', file=io)

    print('}', file=io)

def comic(frame: Frame, fname: str) -> None:
    io = StringIO()
    print('digraph G {', file=io)
    print('rankdir=LR; compound=true;', file=io)
    print('node[shape="cds", style="filled"];', file=io)
    comic_frame_nodes(frame, io)
    comic_frame_edges(frame, io)
    print('}', file=io)

    with open(f'{fname}.dot', 'w') as f:
        io.seek(0)
        shutil.copyfileobj(io, f)
    if shutil.which('dot') is not None:
        os.system(f'dot {fname}.dot -Tpng -o {fname}.png')
        # os.remove(fname)
    else:
        print(f"memo couldn't find a graphviz installation, so it only produced the .dot file. If you don't have graphviz installed, you can paste the .dot file into an online editor, such as https://dreampuf.github.io/GraphvizOnline/")
