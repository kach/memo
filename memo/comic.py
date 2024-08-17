from .core import *
import shutil, os

def node_name(frame, who, id_):
    if who == 'self':
        return f'{frame.name}_{id_}_{id(frame)}'
    if who not in frame.children:
        ic(who, id_)
        return 'TODO_FIXME'
    return node_name(frame.children[who], 'self', id_)

def comic_frame_edges(frame: Frame, retval: Value, io: StringIO) -> None:
    for c in frame.children.keys():
        comic_frame_edges(frame.children[c], retval, io)

    print(f'// Edges from {frame.name}', file=io)
    for (who, id), ch in frame.choices.items():
        if who != 'self':
            continue
        for who_, id_ in ch.wpp_deps:
            if (who_, id_) != (who, id):
                print(f'{node_name(frame, who_, id_)} -> {node_name(frame, who, id)};', file=io)

    if frame.parent is not None:
        for too, frm in frame.conditions.items():
            print(f'{node_name(frame.parent, *frm)} -> {node_name(frame, *too)}[style="{"dotted" if too[0] == "self" else "dashed"}"];', file=io)

    if frame.parent is None:
        for dep in retval.deps:
            print(f'{node_name(frame, *dep)} -> return;', file=io)

def comic_frame_nodes(frame: Frame, retval: Value, io: StringIO) -> None:
    print(f'subgraph cluster_{frame.name} {{', file=io)
    print(f'label="{frame.name}";', file=io)
    # print(f'labelloc="b";', file=io)
    for c in frame.children.keys():
        comic_frame_nodes(frame.children[c], retval, io)

    for (who, id), ch in frame.choices.items():
        if who != 'self':
            continue
        print(f'''{node_name(frame, who, id)}[label="{id} : {ch.domain}", color={"pink" if frame.parent is not None and frame.parent.choices[frame.name, id].known else "lightblue"}];''', file=io)

    if frame.parent is None:
        print('return[color="orange", label="output"];', file=io)

    print('}', file=io)

def comic(frame: Frame, retval: Value, fname: str) -> None:
    io = StringIO()
    print('digraph G {', file=io)
    print('rankdir=LR;', file=io)
    print('node[shape="cds", style="filled", color="lightblue"];', file=io)
    comic_frame_nodes(frame, retval, io)
    comic_frame_edges(frame, retval, io)
    print('}', file=io)

    with open(fname, 'w') as f:
        io.seek(0)
        shutil.copyfileobj(io, f)
    if shutil.which('dot') is not None:
        os.system(f'dot {fname} -Tpng -o {fname}.png')
    else:
        print(f"memo couldn't find a graphviz installation, so it only produced the .dot file. If you don't have graphviz installed, you can paste the .dot file into an online editor, such as https://dreampuf.github.io/GraphvizOnline/")