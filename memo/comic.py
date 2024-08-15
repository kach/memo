from .core import *
import shutil, os

def node_name(frame, who, id_):
    if who == 'self':
        return f'{who}_{id_}_{id(frame)}'
    if who not in frame.children:
        return 'TODO_FIXME'
    return node_name(frame.children[who], 'self', id_)

def comic_frame(frame: Frame, io: StringIO) -> None:
    print(f'subgraph cluster_{frame.name} {{', file=io)
    print(f'label="{frame.name}";', file=io)
    # print(f'labelloc="b";', file=io)
    for c in frame.children.keys():
        comic_frame(frame.children[c], io)
    for (who, id), ch in frame.choices.items():
        if who != 'self':
            continue
        print(f'''{node_name(frame, who, id)}[label="{id} : {ch.domain}", color={"lightblue" if ch.known else "pink"}];''', file=io)

        for who_, id_ in ch.wpp_deps:
            if (who_, id_) != (who, id):
                print(f'{node_name(frame, who_, id_)} -> {node_name(frame, who, id)};', file=io)
    print('}', file=io)

def comic(frame: Frame, fname: str) -> None:
    io = StringIO()
    print('digraph G {', file=io)
    print('rankdir=LR;', file=io)
    print('node[shape="cds", style="filled", color="lightblue"];', file=io);
    comic_frame(frame, io)
    print('}', file=io)
    with open(fname, 'w') as f:
        io.seek(0)
        shutil.copyfileobj(io, f)
    if shutil.which('dot') is not None:
        os.system(f'dot {fname} -Tpng -o {fname}.png')
    else:
        print(f"memo couldn't find a graphviz installation, so it only produced the .dot file. If you don't have graphviz installed, you can paste the .dot file into an online editor, such as https://dreampuf.github.io/GraphvizOnline/")