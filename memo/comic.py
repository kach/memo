from .core import *

def node_name(frame, who, id_):
    if who == 'self':
        return f'{who}_{id_}_{id(frame)}'
    return node_name(frame.children[who], 'self', id_)

def comic_frame(frame: Frame, io: StringIO) -> None:
    print(f'subgraph cluster_{frame.name} {{', file=io)
    print(f'label="{frame.name}";', file=io)
    print(f'labelloc="b";', file=io)
    for c in frame.children.keys():
        comic_frame(frame.children[c], io)
    for (who, id), ch in frame.choices.items():
        if who != 'self':
            continue
        print(f'{node_name(frame, who, id)}[label="{id} : {ch.domain}"];', file=io)
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
    # ic(frame)
    print(io.getvalue())