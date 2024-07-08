from memo import memo
from icecream import ic

DOORS = [1, 2, 3]

@memo
def demo(reward):
    cast: [alice, monty]
    forall: revealed_door in DOORS

    alice: thinks[ monty: chooses(prize in DOORS, wpp=1) ]
    alice: chooses(initial_pick in DOORS, wpp=1)
    alice: thinks[
        monty: thinks[ alice: chooses(initial_pick in DOORS, wpp=1) ],
        monty: observes [alice.initial_pick] is self.initial_pick
    ]
    alice: thinks[
        monty: chooses(
            open in DOORS,
            wpp=0. if (open == alice.initial_pick or open == prize) else 1.)
    ]
    alice: observes [monty.open] is self.revealed_door
    alice: chooses(
        final_pick in DOORS,
        wpp=0. if final_pick == monty.open else
            exp(E[reward if final_pick == monty.prize else -reward]))
    return E[ alice.initial_pick == alice.final_pick ]


if __name__ == "__main__":
    ic(demo(5))
