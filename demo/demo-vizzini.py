# type: ignore
from memo import memo
from enum import IntEnum

# See: https://www.youtube.com/watch?v=rMz7JBRbmNo

class Cups(IntEnum):
    Near_Vizzini = 0
    Near_Westley = 1

@memo
def vizzini_pick[cup: Cups](level):
    vizzini: wants(survive = my_cup != westley.poison)

    vizzini: thinks[
        westley: wants(kill= vizzini.my_cup == poison),
        westley: chooses(poison in Cups, to_maximize=EU[kill]),
        westley: thinks[
            vizzini: chooses(
                my_cup in Cups,
                wpp=vizzini_pick[my_cup](level - 1)
                if level > 0 else my_cup == {Cups.Near_Vizzini}
            )
        ]
    ]
    vizzini: chooses(my_cup in Cups, to_maximize=EU[survive])
    return Pr[vizzini.my_cup == cup]

print("The battle of wits has begun...")
for i in range(5):
    print(f"At level {i}...")
    vizzini_pick(i, print_table=True)